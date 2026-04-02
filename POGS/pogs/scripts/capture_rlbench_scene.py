"""
Capture multi-view images of an RLBench scene for POGS training.

This script replicates the POGS real-world capture pipeline in simulation:
  1. Resets an RLBench episode to a specific configuration
  2. Sweeps the wrist camera around the workspace center on three circular
      orbits with fixed pitch levels
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
import open3d as o3d
from PIL import Image
from scipy.spatial.transform import Rotation

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(_SCRIPT_DIR, "..")))


# ---------------------------------------------------------------------------
# Circular-ring viewpoint generation
# ---------------------------------------------------------------------------

def _get_three_circle_positions(
    n_views: int,
    center: np.ndarray,
    horizontal_radius: float,
    pitch_degs: tuple[float, float, float] = (0.0, -15.0, -30.0),
    theta_range: tuple[float, float] = (0.0, 360.0),
) -> list[np.ndarray]:
    """
    Generate exactly three circular orbits around `center`.

    The camera always faces `center` later in the loop via _point_at().
    Ring heights are selected so the viewing pitch toward center is:
      ring 1 -> 0 deg, ring 2 -> -15 deg, ring 3 -> -30 deg.
    """
    positions: list[np.ndarray] = []
    if n_views <= 0:
        return positions

    ring_count = len(pitch_degs)
    base = n_views // ring_count
    remainder = n_views % ring_count

    th_min = np.deg2rad(theta_range[0])
    th_max = np.deg2rad(theta_range[1])
    th_span = max(1e-8, th_max - th_min)

    for ring_idx, pitch_deg in enumerate(pitch_degs):
        n_ring = base + (1 if ring_idx < remainder else 0)
        if n_ring <= 0:
            continue

        # For look-at center and fixed horizontal radius rho:
        # pitch = atan2(center_z - cam_z, rho)  ->  cam_z = center_z - rho * tan(pitch)
        z_offset = -horizontal_radius * math.tan(math.radians(float(pitch_deg)))
        z_cam = center[2] + z_offset

        # Slight per-ring phase shift avoids identical azimuth stacks across rings.
        phase = ring_idx / ring_count
        for k in range(n_ring):
            theta = th_min + th_span * ((k + phase) / n_ring)
            theta = th_min + ((theta - th_min) % th_span)
            x_cam = center[0] + horizontal_radius * math.cos(theta)
            y_cam = center[1] + horizontal_radius * math.sin(theta)
            positions.append(np.array([x_cam, y_cam, z_cam], dtype=np.float64))

    return positions


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


def _roll_about_optical_axis(c2w: np.ndarray, roll_deg: float) -> np.ndarray:
    """Apply camera-local Z roll to mimic wrist-mounted camera flip behavior."""
    roll_rad = np.deg2rad(roll_deg)
    c = np.cos(roll_rad)
    s = np.sin(roll_rad)
    roll_local = np.array(
        [
            [c, -s, 0.0],
            [s,  c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    out = c2w.copy()
    out[:3, :3] = c2w[:3, :3] @ roll_local
    return out


def _depth_to_world_points(
    depth_m: np.ndarray,
    rgb_uint8: np.ndarray,
    c2w: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    *,
    stride: int = 4,
    min_depth: float = 0.05,
    max_depth: float = 4.0,
    max_points: int = 3000,
) -> tuple[np.ndarray, np.ndarray]:
    """Back-project one depth frame into world-space points for sparse GS init."""
    h, w = depth_m.shape
    ys = np.arange(0, h, stride, dtype=np.int32)
    xs = np.arange(0, w, stride, dtype=np.int32)
    grid_y, grid_x = np.meshgrid(ys, xs, indexing="ij")

    z = depth_m[grid_y, grid_x]
    valid = np.isfinite(z) & (z > min_depth) & (z < max_depth)
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.float32)

    u_pix = grid_x[valid]
    v_pix = grid_y[valid]
    z = z[valid].astype(np.float32)
    u = u_pix.astype(np.float32)
    v = v_pix.astype(np.float32)

    # c2w is now in OpenGL format: X right, Y up, -Z forward
    # Image frame: u right, v down.
    x = (u - cx) * z / fx
    y = -(v - cy) * z / fy
    z_opengl = -z  # depth is positive forward, but OpenGL local Z is backward
    points_cam = np.stack([x, y, z_opengl], axis=1)

    points_h = np.concatenate(
        [points_cam, np.ones((points_cam.shape[0], 1), dtype=np.float32)],
        axis=1,
    )
    points_world = (c2w @ points_h.T).T[:, :3].astype(np.float32)
    colors = rgb_uint8[v_pix, u_pix].astype(np.float32) / 255.0

    if points_world.shape[0] > max_points:
        keep = np.random.choice(points_world.shape[0], max_points, replace=False)
        points_world = points_world[keep]
        colors = colors[keep]

    return points_world, colors


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
    
    _ = task.reset_to_demo(demos[0])

    print(f"[Capture] Scene reset to {args.task} episode {args.episode}")

    # ------------------------------------------------------------------
    # 3. Get the wrist camera (like POGS uses the ZED on the UR5 wrist)
    # ------------------------------------------------------------------
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
        k1=0.0,
        k2=0.0,
        p1=0.0,
        p2=0.0,
        camera_model="OPENCV",
    )

    # ------------------------------------------------------------------
    # 4. Generate circular viewpoints around the scene center
    # ------------------------------------------------------------------
    from src.data.components.rlbench.constants import SCENE_BOUNDS, loc_bounds  # type: ignore

    if args.task in loc_bounds:
        bounds = np.asarray(loc_bounds[args.task], dtype=np.float64)
        scene_center = bounds.mean(axis=0)
    else:
        scene_center = np.array([
            (SCENE_BOUNDS[0] + SCENE_BOUNDS[3]) / 2,
            (SCENE_BOUNDS[1] + SCENE_BOUNDS[4]) / 2,
            (SCENE_BOUNDS[2] + SCENE_BOUNDS[5]) / 2,
        ])

    radius_base = max(0.15, float(args.radius))
    ring_pitches = tuple(args.circle_pitches_deg)

    orbit_positions = _get_three_circle_positions(
        n_views=args.n_views,
        center=scene_center,
        horizontal_radius=radius_base,
        pitch_degs=ring_pitches,
    )

    print(
        "[Capture] Circular trajectory: "
        f"radius={radius_base:.3f}m, pitches={list(ring_pitches)}, views={len(orbit_positions)}"
    )

    # ------------------------------------------------------------------
    # 5. Sweep the wrist camera around the scene on circular rings.
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
    seed_points_all = []
    seed_colors_all = []
    roll_correction_deg = args.camera_roll_deg if args.zed_flip_mode else 0.0
    for i, cam_pos in enumerate(orbit_positions):
        # Compute camera-to-world matrix pointing at the scene centre
        c2w = _point_at(cam_pos, scene_center)
        if roll_correction_deg != 0.0:
            c2w = _roll_about_optical_axis(c2w, roll_correction_deg)

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

        # RLBench/PyRep camera frame is X left, Y up, +Z forward. 
        # Nerfstudio expects OpenGL-style camera: X right, Y up, -Z backward.
        # So we flip both X and Z to maintain a valid SE(3) determinant=1 matrix
        nerf_c2w = actual_c2w.copy()
        nerf_c2w[:3, 0] *= -1
        nerf_c2w[:3, 2] *= -1

        # Nerfstudio perspective rays point along -Z in camera space.
        # Sanity check that the exported pose is facing the scene center.
        to_target = scene_center - nerf_c2w[:3, 3]
        to_target_norm = np.linalg.norm(to_target)
        if to_target_norm > 1e-8:
            cam_forward_world = -nerf_c2w[:3, 2]
            look_dot = float(np.dot(cam_forward_world, to_target / to_target_norm))
            if look_dot <= 0.0:
                raise RuntimeError(
                    f"Invalid transform at view {i}: camera faces away from target center "
                    f"(dot={look_dot:.4f})."
                )

        seed_points, seed_colors = _depth_to_world_points(
            depth_m=depth,
            rgb_uint8=rgb_uint8,
            c2w=nerf_c2w,
            fx=fl,
            fy=fl,
            cx=W / 2.0,
            cy=H / 2.0,
            stride=4,
            min_depth=0.05,
            max_depth=4.0,
            max_points=3000,
        )
        if seed_points.shape[0] > 0:
            seed_points_all.append(seed_points)
            seed_colors_all.append(seed_colors)

        frames.append({
            "file_path": f"images/{img_fname}",
            "depth_file_path": f"depth/{depth_fname}",
            "transform_matrix": nerf_c2w.tolist(),
        })

        if (i + 1) % 10 == 0:
            print(f"  Captured {i+1}/{len(orbit_positions)} frames")

    # ------------------------------------------------------------------
    # 6. Write transforms.json (nerfstudio format)
    # ------------------------------------------------------------------
    if not seed_points_all:
        raise RuntimeError("No valid depth points extracted. Cannot create sparse_pc.ply.")

    sparse_points = np.concatenate(seed_points_all, axis=0).astype(np.float32)
    sparse_colors = np.concatenate(seed_colors_all, axis=0).astype(np.float32)

    if sparse_points.shape[0] > 1_000_000:
        keep = np.random.choice(sparse_points.shape[0], 1_000_000, replace=False)
        sparse_points = sparse_points[keep]
        sparse_colors = sparse_colors[keep]

    sparse_pcd = o3d.geometry.PointCloud()
    sparse_pcd.points = o3d.utility.Vector3dVector(sparse_points.astype(np.float64))
    sparse_pcd.colors = o3d.utility.Vector3dVector(np.clip(sparse_colors, 0.0, 1.0).astype(np.float64))

    sparse_pcd = sparse_pcd.voxel_down_sample(voxel_size=0.005)
    if len(sparse_pcd.points) > 0:
        sparse_pcd, _ = sparse_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    max_sparse_points = 200_000
    if len(sparse_pcd.points) > max_sparse_points:
        pts = np.asarray(sparse_pcd.points)
        cols = np.asarray(sparse_pcd.colors)
        keep = np.random.choice(pts.shape[0], max_sparse_points, replace=False)
        sparse_pcd.points = o3d.utility.Vector3dVector(pts[keep])
        sparse_pcd.colors = o3d.utility.Vector3dVector(cols[keep])

    sparse_ply_path = os.path.join(args.out_dir, "sparse_pc.ply")
    o3d.io.write_point_cloud(sparse_ply_path, sparse_pcd)

    transforms = {**intrinsics, "frames": frames, "ply_file_path": "sparse_pc.ply"}
    out_json = os.path.join(args.out_dir, "transforms.json")
    with open(out_json, "w") as f:
        json.dump(transforms, f, indent=2)

    print(f"\n[Capture] Done! Saved {len(frames)} frames to {args.out_dir}")
    print(f"[Capture] sparse_pc.ply written to {sparse_ply_path} with {len(sparse_pcd.points)} points")
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
                        help="Horizontal radius of each circular orbit around scene center (m)")
    parser.add_argument(
        "--circle-pitches-deg",
        type=float,
        nargs=3,
        default=[0.0, -15.0, -30.0],
        help="Three pitch angles (deg) for the circular rings, e.g. 0 -15 -30.",
    )
    parser.add_argument("--elev-min",   type=float, default=30,
                        help="Legacy arg (unused in circular trajectory mode).")
    parser.add_argument("--elev-max",   type=float, default=70,
                        help="Legacy arg (unused in circular trajectory mode).")
    parser.add_argument(
        "--zed-flip-mode",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply ZED-like 180-degree roll correction for wrist-mounted camera orientation.",
    )
    parser.add_argument(
        "--camera-roll-deg",
        type=float,
        default=180.0,
        help="Camera roll correction in degrees (used when --zed-flip-mode is enabled).",
    )
    parser.add_argument("--image-w",    type=int, default=1280)
    parser.add_argument("--image-h",    type=int, default=720)
    args = parser.parse_args()

    capture_scene(args)


if __name__ == "__main__":
    main()
