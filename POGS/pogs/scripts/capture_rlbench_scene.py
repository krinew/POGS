"""
Capture multi-view images of an RLBench scene for POGS training.

This script replicates the POGS real-world capture pipeline in simulation:
  1. Resets an RLBench episode to a specific configuration
  2. Moves the ROBOT ARM (like UR5 in real POGS) to sweep the wrist camera
     around the workspace on a hemisphere
  3. Captures RGB from the wrist camera at each viewpoint
  4. Saves everything in nerfstudio-compatible JSON format

The NeRF coordinate frame convention requires flipping y and z axes
(nerf_frame_to_image_frame), matching POGS's save_data / save_pose logic.

Usage:
    python capture_rlbench_scene.py \\
        --task        open_drawer \\
        --episode     0 \\
        --data-root   /path/to/rlbench/raw/train \\
        --out-dir     data/pogs_scenes/open_drawer/shared \\
        --n-views     100
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import pickle
import time

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(_SCRIPT_DIR, "..")))


# ---------------------------------------------------------------------------
# Hemisphere viewpoint generation (same as POGS real-world get_hemi_translations)
# ---------------------------------------------------------------------------

def _get_hemi_positions(
    n_views: int,
    center: np.ndarray,
    radii: list[float] = [0.6, 0.8],
    phi_range: tuple = (30, 70),   # elevation range in degrees
    theta_range: tuple = (0, 360), # azimuth range in degrees
) -> list[np.ndarray]:
    """
    Generate positions on multiple concentric hemisphere shells around `center`,
    to capture "multiple times" from different depths.
    """
    total_positions = []
    
    n_views_per_radius = max(1, n_views // len(radii))
    phi_div = int(math.sqrt(n_views_per_radius))
    theta_div = max(1, n_views_per_radius // phi_div)
    
    sin = lambda x: np.sin(np.deg2rad(x))
    cos = lambda x: np.cos(np.deg2rad(x))
    
    for radius in radii:
        positions = []
        for i, phi in enumerate(np.linspace(phi_range[0], phi_range[1], phi_div)):
            row = []
            for j, theta in enumerate(np.linspace(theta_range[0], theta_range[1], theta_div, endpoint=False)):
                pos = np.array([
                    radius * sin(phi) * cos(theta),
                    radius * sin(phi) * sin(theta),
                    radius * cos(phi),
                ])
                row.append(pos + center)
            # Reverse alternate rows for a snake-like path (smoother robot motion)
            if i % 2 == 1:
                row.reverse()
            positions.extend(row)
        total_positions.extend(positions)
    
    return total_positions[:n_views] # enforce exact match to args.n_views


def _point_at(cam_pos: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    Compute a 4x4 camera-to-world matrix that looks from cam_pos toward target.
    Matches POGS's point_at() function from capture_utils.py.
    """
    direction = target - cam_pos
    z_axis = direction / np.linalg.norm(direction)
    
    x_axis_dir = -np.cross(np.array([0, 0, 1.0]), z_axis)
    if np.linalg.norm(x_axis_dir) < 1e-10:
        x_axis_dir = np.array([0, 1, 0.0])
    x_axis = x_axis_dir / np.linalg.norm(x_axis_dir)
    
    y_axis_dir = np.cross(z_axis, x_axis)
    y_axis = y_axis_dir / np.linalg.norm(y_axis_dir)
    
    R = np.column_stack([x_axis, y_axis, z_axis])
    
    c2w = np.eye(4)
    c2w[:3, :3] = R
    c2w[:3, 3]  = cam_pos
    return c2w


# NeRF coordinate convention: flip y and z (same as POGS scene_capture.py line 38-41)
NERF_FRAME_TO_IMAGE_FRAME = np.array([
    [1,  0,  0, 0],
    [0, -1,  0, 0],
    [0,  0, -1, 0],
    [0,  0,  0, 1],
], dtype=np.float64)


# ---------------------------------------------------------------------------
# Main capture logic
# ---------------------------------------------------------------------------

def capture_scene(args):
    import pickle
    from pyrep.objects.vision_sensor import VisionSensor

    from rlbench.environment import Environment
    from rlbench.action_modes.action_mode import MoveArmThenGripper
    from rlbench.action_modes.arm_action_modes import JointVelocity
    from rlbench.action_modes.gripper_action_modes import Discrete
    from rlbench.observation_config import ObservationConfig

    # ------------------------------------------------------------------
    # 1. Launch RLBench environment with the GUI (headless=False)
    # ------------------------------------------------------------------
    obs_config = ObservationConfig()
    obs_config.set_all(False)  # Disable all cameras by default
    obs_config.wrist_camera.set_all(True)  # Only enable the wrist camera

    env = Environment(
        action_mode=MoveArmThenGripper(
            arm_action_mode=JointVelocity(),
            gripper_action_mode=Discrete(),
        ),
        dataset_root=args.data_root,
        obs_config=obs_config,
        headless=False,
    )
    env.launch()

    # ------------------------------------------------------------------
    # 2. Reset to the target episode
    # ------------------------------------------------------------------
    from src.utils.rlbench_utils import task_file_to_task_class   # type: ignore
    task_cls  = task_file_to_task_class(args.task)
    task      = env.get_task(task_cls)

    var_num = pickle.load(open(
        os.path.join(args.data_root, args.task, "all_variations",
                     "episodes", f"episode{args.episode}", "variation_number.pkl"), "rb"
    ))
    task.set_variation(-1)
    demos = task.get_demos(1, random_selection=False,
                           from_episode_number=args.episode)
    task.set_variation(var_num)
    for obs in demos[0]._observations:
        obs.misc["variation_index"] = var_num
        
    # Skip motion planning validation (not needed for scene capture)
    task._task._feasible = lambda *args, **kwargs: (True, 0)
    
    _, obs = task.reset_to_demo(demos[0])

    print(f"[Capture] Scene reset to {args.task} episode {args.episode}")

    # ------------------------------------------------------------------
    # 3. Get the wrist camera (like POGS uses the ZED on the UR5 wrist)
    # ------------------------------------------------------------------
    pr = env._scene.pyrep
    wrist_cam: VisionSensor = env._scene._cam_wrist
    
    # Get the robot arm and gripper
    from pyrep.robots.arms.panda import Panda
    from pyrep.robots.end_effectors.panda_gripper import PandaGripper
    robot = Panda()
    gripper = PandaGripper()

    # CRITICAL: Hide the robot and gripper from the vision sensor entirely 
    # so we acquire a clean, unobstructed 3D NeRF map of the desk and drawer!
    for obj in robot.get_objects_in_tree(exclude_base=False):
        obj.set_renderable(False)
    for obj in gripper.get_objects_in_tree(exclude_base=False):
        obj.set_renderable(False)

    W, H = args.image_w, args.image_h
    wrist_cam.set_resolution([W, H])
    fov_rad = math.radians(wrist_cam.get_perspective_angle())
    fl = (W / 2) / math.tan(fov_rad / 2)

    intrinsics = dict(
        fl_x=fl, fl_y=fl,
        cx=W / 2, cy=H / 2,
        w=W, h=H,
        aabb_scale=2,
        scale=1.2,
    )

    # ------------------------------------------------------------------
    # 4. Generate hemisphere viewpoints around the scene center
    # ------------------------------------------------------------------
    from src.data.components.rlbench.constants import SCENE_BOUNDS  # type: ignore
    scene_center = np.array([
        (SCENE_BOUNDS[0] + SCENE_BOUNDS[3]) / 2,
        (SCENE_BOUNDS[1] + SCENE_BOUNDS[4]) / 2,
        0.75,  # Focus directly on the table surface instead of high in the air
    ])

    hemi_positions = _get_hemi_positions(
        n_views=args.n_views,
        center=scene_center,
        radii=[0.8, 1.1, 1.4],  # Massive orbits around the outside of the desk
        phi_range=(15, 80),     # Wide elevation (from near table-level up to top-down)
    )

    # ------------------------------------------------------------------
    # 5. Sweep the wrist camera around the scene on a hemisphere.
    #    In the real POGS pipeline the UR5 arm moves to pre-recorded joint
    #    positions.  In simulation we achieve the same result by directly
    #    repositioning the wrist camera object (after detaching it from the
    #    arm's kinematic chain) and forcing a render with handle_explicitly().
    # ------------------------------------------------------------------
    os.makedirs(os.path.join(args.out_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "depth"), exist_ok=True)

    # Detach camera from arm so it can be teleported freely
    wrist_cam.set_parent(None)
    wrist_cam.set_explicit_handling(1)

    frames = []
    for i, cam_pos in enumerate(hemi_positions):
        # Compute camera-to-world matrix pointing at the scene centre
        c2w = _point_at(cam_pos, scene_center)

        # Move the camera using individual position + quaternion calls
        # (most reliable PyRep API — avoids set_matrix / set_pose shape issues)
        quat_xyzw = Rotation.from_matrix(c2w[:3, :3]).as_quat()  # x y z w
        wrist_cam.set_position(list(cam_pos))
        wrist_cam.set_quaternion(list(quat_xyzw))

        # Force the vision sensor to re-render at the new pose
        # WITHOUT ticking physics (which would let RLBench snap it back)
        wrist_cam.handle_explicitly()

        # Capture RGB image
        rgb = wrist_cam.capture_rgb()           # (H, W, 3) float32 [0,1]
        rgb_uint8 = (rgb * 255).clip(0, 255).astype(np.uint8)
        img_fname = f"frame_{i:04d}.png"
        Image.fromarray(rgb_uint8).save(os.path.join(args.out_dir, "images", img_fname))

        # Capture Depth image (in meters)
        depth = wrist_cam.capture_depth(in_meters=True)
        depth_fname = f"frame_{i:04d}.npy"
        np.save(os.path.join(args.out_dir, "depth", depth_fname), depth.astype(np.float32))

        # Read back the actual camera matrix (ground truth pose from simulator)
        actual_c2w = np.array(wrist_cam.get_matrix()).reshape(4, 4)

        # Apply POGS's exact NeRF coordinate convention from capture_utils.py save_data():
        #   mat[:3, 1] *= -1  → negate the Y column
        #   mat[:3, 2] *= -1  → negate the Z column
        # This is NOT the same as post-multiplying by a diagonal flip matrix!
        nerf_c2w = actual_c2w.copy()
        nerf_c2w[:3, 1] *= -1
        nerf_c2w[:3, 2] *= -1

        frames.append({
            "file_path": f"images/{img_fname}",
            "depth_file_path": f"depth/{depth_fname}",
            "transform_matrix": nerf_c2w.tolist(),
        })

        if (i + 1) % 10 == 0:
            print(f"  Captured {i+1}/{args.n_views} frames")

    # ------------------------------------------------------------------
    # 6. Write transforms.json (nerfstudio format)
    # ------------------------------------------------------------------
    transforms = {**intrinsics, "frames": frames}
    out_json = os.path.join(args.out_dir, "transforms.json")
    with open(out_json, "w") as f:
        json.dump(transforms, f, indent=2)

    print(f"\n[Capture] Done! Saved {len(frames)} frames to {args.out_dir}")
    print(f"[Capture] transforms.json written to {out_json}")
    print(f"\nNext step — train POGS:")
    print(f"  ns-train pogs --data {args.out_dir}")
    env.shutdown()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Capture RLBench scene images for POGS training (wrist camera)"
    )
    parser.add_argument("--task",       required=True, help="RLBench task name")
    parser.add_argument("--episode",    type=int, default=0,
                        help="Episode number to reset to")
    parser.add_argument("--data-root",  default="data/rlbench/raw/train",
                        help="Root of raw RLBench demos")
    parser.add_argument("--out-dir",    required=True,
                        help="Output directory for images + transforms.json")
    parser.add_argument("--n-views",    type=int, default=100,
                        help="Number of camera viewpoints to capture")
    parser.add_argument("--radius",     type=float, default=0.80,
                        help="Camera orbital radius around scene center (m)")
    parser.add_argument("--elev-min",   type=float, default=30,
                        help="Min elevation angle (degrees)")
    parser.add_argument("--elev-max",   type=float, default=70,
                        help="Max elevation angle (degrees)")
    parser.add_argument("--image-w",    type=int, default=640)
    parser.add_argument("--image-h",    type=int, default=480)
    args = parser.parse_args()

    capture_scene(args)


if __name__ == "__main__":
    main()
