"""Generate timestep-wise dynamic POGS datasets from raw RLBench demos.

This script replays recorded RLBench episodes from the raw dataset split,
runs POGS tracking at every timestep, and saves a per-episode `.pkl` file with:

- full-scene tracked point clouds per timestep
- gripper state trajectory
- ACT-style action trajectory (next-state pose/open/collision)

Output format per episode:
{
    "tracked_coords": (T, N, 3),
    "tracked_colors": (T, N, 3),
    "joint_positions": (T, 7),
    "gripper_open": (T, 1),
    "action": (T, 9),
    "variation_id": int,
    "task_goal": (512,),
}
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
from tqdm import tqdm

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(_SCRIPT_DIR, "../../")))
sys.path.insert(0, os.path.normpath(os.path.join(_SCRIPT_DIR, "../../../PointCloudMatters")))

from pogs.tracking.optim import Optimizer


def task_file_to_task_class(task_file: str):
    import importlib

    name = task_file.replace(".py", "")
    class_name = "".join([w[0].upper() + w[1:] for w in name.split("_")])
    mod = importlib.import_module(f"rlbench.tasks.{name}")
    mod = importlib.reload(mod)
    return getattr(mod, class_name)


def sample_to_fixed_size(
    coords: np.ndarray,
    colors: np.ndarray,
    num_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample points to a fixed size so episodes can be stacked safely."""
    if coords.shape[0] == 0:
        raise RuntimeError("Tracked point cloud is empty.")

    if coords.shape[0] >= num_points:
        idx = np.random.choice(coords.shape[0], num_points, replace=False)
    else:
        idx = np.random.choice(coords.shape[0], num_points, replace=True)

    return coords[idx].astype(np.float32), colors[idx].astype(np.float32)


def extract_full_scene_pointcloud(
    optimizer: Optimizer,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Export full-scene tracked Gaussians and load them as point clouds."""
    optimizer.state_to_ply(obj_id=None)
    global_ply_path = optimizer.config_path.parent.joinpath("global.ply")

    pcd = o3d.io.read_point_cloud(str(global_ply_path))
    coords = np.asarray(pcd.points, dtype=np.float32)
    colors = np.asarray(pcd.colors, dtype=np.float32)

    return sample_to_fixed_size(coords, colors, max_points)


def build_action_from_obs(obs) -> np.ndarray:
    """Build one ACT action vector from an RLBench observation.

    Action layout here is 9D (xyz + quat + gripper + collision), matching
    the same preprocessing path as PointCloudMatters RLBench ACT datasets.
    """
    return np.concatenate(
        [
            np.asarray(obs.gripper_pose, dtype=np.float32),
            np.asarray([obs.gripper_open], dtype=np.float32),
            np.asarray([obs.ignore_collisions], dtype=np.float32),
        ],
        axis=0,
    ).astype(np.float32)


def get_episode_ids(episodes_root: Path) -> list[int]:
    episode_ids = []
    for d in sorted(episodes_root.glob("episode*")):
        if d.is_dir():
            try:
                episode_ids.append(int(d.name.replace("episode", "")))
            except ValueError:
                continue
    return episode_ids


def obs_to_tensors(obs, camera: str, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    rgb = getattr(obs, f"{camera}_rgb").astype(np.float32)
    depth = getattr(obs, f"{camera}_depth").astype(np.float32)
    return (
        torch.from_numpy(rgb).to(device),
        torch.from_numpy(depth).to(device),
    )


def save_episode(
    out_path: Path,
    tracked_coords: list[np.ndarray],
    tracked_colors: list[np.ndarray],
    joint_positions: list[np.ndarray],
    gripper_open: list[np.ndarray],
    actions: list[np.ndarray],
    variation_id: int,
) -> None:
    payload = {
        "tracked_coords": np.stack(tracked_coords, axis=0).astype(np.float32),
        "tracked_colors": np.stack(tracked_colors, axis=0).astype(np.float32),
        "joint_positions": np.stack(joint_positions, axis=0).astype(np.float32),
        "gripper_open": np.stack(gripper_open, axis=0).astype(np.float32),
        "action": np.stack(actions, axis=0).astype(np.float32),
        "variation_id": int(variation_id),
        # Keep this key for compatibility with existing ACT dataloaders.
        "task_goal": np.zeros(512, dtype=np.float32),
    }
    with out_path.open("wb") as f:
        pickle.dump(payload, f)


def process_episode(
    optimizer: Optimizer,
    demo,
    out_path: Path,
    camera: str,
    max_points: int,
    first_niters: int,
    niters: int,
    device: str,
    variation_id: int,
) -> None:
    observations = list(demo._observations)
    if len(observations) < 2:
        raise RuntimeError("Episode must contain at least 2 timesteps.")

    optimizer.reset_optimizer()

    first_rgb, first_depth = obs_to_tensors(observations[0], camera, device)
    optimizer.set_frame(first_rgb, optimizer.cam2world_ns_ds, first_depth)
    optimizer.init_obj_pose()

    tracked_coords = []
    tracked_colors = []
    joint_positions = []
    gripper_open = []

    for t, obs in enumerate(observations):
        rgb_t, depth_t = obs_to_tensors(obs, camera, device)
        optimizer.set_observation(rgb_t, optimizer.cam2world_ns_ds, depth_t)
        optimizer.step_opt(niter=first_niters if t == 0 else niters)

        coords_t, colors_t = extract_full_scene_pointcloud(optimizer, max_points)
        tracked_coords.append(coords_t)
        tracked_colors.append(colors_t)

        joint_positions.append(np.asarray(obs.gripper_pose, dtype=np.float32))
        gripper_open.append(np.asarray([obs.gripper_open], dtype=np.float32))

    actions = []
    for t in range(len(observations)):
        next_obs = observations[t + 1] if (t + 1) < len(observations) else observations[t]
        actions.append(build_action_from_obs(next_obs))

    save_episode(
        out_path=out_path,
        tracked_coords=tracked_coords,
        tracked_colors=tracked_colors,
        joint_positions=joint_positions,
        gripper_open=gripper_open,
        actions=actions,
        variation_id=variation_id,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate dynamic POGS ACT episodes")
    parser.add_argument("--task", required=True, help="RLBench task, e.g. open_drawer")
    parser.add_argument(
        "--raw-root",
        required=True,
        help="Raw RLBench split root containing task folders (train/val)",
    )
    parser.add_argument("--pogs-config", required=True, help="Path to trained POGS config.yml")
    parser.add_argument("--out-dir", required=True, help="Directory to write episode*.pkl")
    parser.add_argument("--camera", default="front", help="RLBench camera stream to track")
    parser.add_argument("--max-points", type=int, default=8192)
    parser.add_argument("--first-niters", type=int, default=15)
    parser.add_argument("--niters", type=int, default=5)
    parser.add_argument("--start-episode", type=int, default=0)
    parser.add_argument("--max-episodes", type=int, default=-1, help="-1 means all")
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    from rlbench.action_modes.action_mode import MoveArmThenGripper
    from rlbench.action_modes.arm_action_modes import JointVelocity
    from rlbench.action_modes.gripper_action_modes import Discrete
    from rlbench.environment import Environment
    from rlbench.observation_config import ObservationConfig

    raw_root = Path(args.raw_root)
    episodes_root = raw_root / args.task / "all_variations" / "episodes"
    if not episodes_root.exists():
        raise FileNotFoundError(f"Episodes root not found: {episodes_root}")

    episode_ids = [eid for eid in get_episode_ids(episodes_root) if eid >= args.start_episode]
    if args.max_episodes > 0:
        episode_ids = episode_ids[: args.max_episodes]
    if not episode_ids:
        raise RuntimeError("No episodes selected. Check --raw-root, --task, and episode range args.")

    obs_config = ObservationConfig()
    obs_config.set_all(False)
    getattr(obs_config, f"{args.camera}_camera").set_all(True)
    obs_config.gripper_pose = True
    obs_config.gripper_open = True

    env = Environment(
        action_mode=MoveArmThenGripper(
            arm_action_mode=JointVelocity(),
            gripper_action_mode=Discrete(),
        ),
        dataset_root=str(raw_root),
        obs_config=obs_config,
        headless=args.headless,
    )
    env.launch()

    task_cls = task_file_to_task_class(args.task)
    task = env.get_task(task_cls)

    first_ep = episode_ids[0]
    variation_path = episodes_root / f"episode{first_ep}" / "variation_number.pkl"
    with variation_path.open("rb") as f:
        variation_id = pickle.load(f)

    task.set_variation(-1)
    first_demo = task.get_demos(
        1,
        random_selection=False,
        live_demos=False,
        from_episode_number=first_ep,
    )[0]
    task.set_variation(variation_id)
    _, first_obs = task.reset_to_demo(first_demo)

    K = np.asarray(first_obs.misc[f"{args.camera}_camera_intrinsics"], dtype=np.float32)
    extrinsics = np.asarray(first_obs.misc[f"{args.camera}_camera_extrinsics"], dtype=np.float32)
    if extrinsics.shape == (4, 4):
        init_cam_pose = torch.from_numpy(extrinsics[:3, :]).float().unsqueeze(0)
    elif extrinsics.shape == (3, 4):
        init_cam_pose = torch.from_numpy(extrinsics).float().unsqueeze(0)
    else:
        raise RuntimeError(f"Unexpected extrinsics shape: {extrinsics.shape}")

    height, width = getattr(first_obs, f"{args.camera}_rgb").shape[:2]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    optimizer = Optimizer(
        Path(args.pogs_config),
        K,
        width,
        height,
        init_cam_pose,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {len(episode_ids)} episodes for task={args.task}")
    for episode_id in tqdm(episode_ids):
        variation_path = episodes_root / f"episode{episode_id}" / "variation_number.pkl"
        with variation_path.open("rb") as f:
            variation_id = pickle.load(f)

        task.set_variation(-1)
        demo = task.get_demos(
            1,
            random_selection=False,
            live_demos=False,
            from_episode_number=episode_id,
        )[0]
        task.set_variation(variation_id)
        task.reset_to_demo(demo)

        out_path = out_dir / f"episode{episode_id}.pkl"
        process_episode(
            optimizer=optimizer,
            demo=demo,
            out_path=out_path,
            camera=args.camera,
            max_points=args.max_points,
            first_niters=args.first_niters,
            niters=args.niters,
            device=device,
            variation_id=int(variation_id),
        )

    env.shutdown()
    print(f"Done. Saved episodes to: {out_dir}")


if __name__ == "__main__":
    main()
