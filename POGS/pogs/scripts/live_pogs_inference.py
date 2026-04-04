"""Separate end-to-end simulation loop: POGS tracking + PointNet++ + ACT.

This script is intentionally separate from PointCloudMatters test entrypoints.
It evaluates a trained ACT checkpoint by:
1. resetting RLBench episodes from raw demos,
2. running dynamic POGS tracking on each step,
3. encoding tracked full-scene point clouds with PointNet++,
4. feeding embeddings to ACT and stepping the simulator.
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
from pyrep.errors import ConfigurationPathError, IKError
from rlbench.backend.exceptions import InvalidActionError

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(_SCRIPT_DIR, "../../")))
sys.path.insert(0, os.path.normpath(os.path.join(_SCRIPT_DIR, "../../../PointCloudMatters")))

from pogs.tracking.optim import Optimizer
from src.data.components.rlbench.constants import loc_bounds
from src.data.components.transformpcd import (
    CollectPCD,
    ComposePCD,
    GridSamplePCD,
    NormalizeColorPCD,
    ToTensorPCD,
)
from src.models.components.pcd_encoder.pointnet2_encoder import PointNet2Encoder
from src.models.rlbench_act_bc_module import RLBenchACTBCModule
from src.utils.rotation_conversions import matrix_to_rotation_6d, quaternion_to_matrix


def task_file_to_task_class(task_file: str):
    import importlib

    name = task_file.replace(".py", "")
    class_name = "".join([w[0].upper() + w[1:] for w in name.split("_")])
    mod = importlib.import_module(f"rlbench.tasks.{name}")
    mod = importlib.reload(mod)
    return getattr(mod, class_name)


def sample_to_fixed_size(coords: np.ndarray, colors: np.ndarray, n_points: int) -> tuple[np.ndarray, np.ndarray]:
    if coords.shape[0] == 0:
        raise RuntimeError("Tracked point cloud is empty.")
    if coords.shape[0] >= n_points:
        idx = np.random.choice(coords.shape[0], n_points, replace=False)
    else:
        idx = np.random.choice(coords.shape[0], n_points, replace=True)
    return coords[idx].astype(np.float32), colors[idx].astype(np.float32)


def extract_full_scene_pointcloud(optimizer: Optimizer, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    optimizer.state_to_ply(obj_id=None)
    pcd = o3d.io.read_point_cloud(str(optimizer.config_path.parent.joinpath("global.ply")))
    coords = np.asarray(pcd.points, dtype=np.float32)
    colors = np.asarray(pcd.colors, dtype=np.float32)
    return sample_to_fixed_size(coords, colors, max_points)


def build_pcd_transform(grid_size: float) -> ComposePCD:
    return ComposePCD(
        [
            GridSamplePCD(
                grid_size=grid_size,
                hash_type="fnv",
                mode="train",
                return_grid_coord=False,
                return_displacement=False,
                keys=("coord", "color"),
            ),
            NormalizeColorPCD(),
            ToTensorPCD(),
            CollectPCD(keys=["coord"], feat_keys=["color", "coord"]),
        ]
    )


def encode_obs_embed(
    coords: np.ndarray,
    colors: np.ndarray,
    transform: ComposePCD,
    encoder: PointNet2Encoder,
    device: str,
) -> torch.Tensor:
    colors_255 = colors * 255.0 if colors.max() <= 1.0 else colors
    pcd = transform({"coord": coords, "color": colors_255})
    pcd = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in pcd.items()}
    with torch.no_grad():
        feats = encoder(pcd)
    return feats[0].detach().float().unsqueeze(0)


def obs_to_qpos(obs, task_name: str, collision: bool, device: str) -> torch.Tensor:
    if collision:
        qpos = np.concatenate(
            [obs.gripper_pose, [obs.gripper_open], [obs.ignore_collisions]],
            axis=0,
        )
    else:
        qpos = np.concatenate([obs.gripper_pose, [obs.gripper_open]], axis=0)

    qpos_t = torch.from_numpy(qpos).float().to(device)
    pos_min = torch.FloatTensor(loc_bounds[task_name][0]).to(device)
    pos_max = torch.FloatTensor(loc_bounds[task_name][1]).to(device)

    qpos_t[:3] = (qpos_t[:3] - pos_min) / (pos_max - pos_min) * 2.0 - 1.0
    qpos_t[3:7] = torch.nn.functional.normalize(qpos_t[3:7], dim=-1)
    qpos_rot = matrix_to_rotation_6d(quaternion_to_matrix(qpos_t[3:7]))

    if collision:
        qpos_t = torch.cat([qpos_t[:3], qpos_rot, qpos_t[7:9]], dim=0)
    else:
        qpos_t = torch.cat([qpos_t[:3], qpos_rot, qpos_t[7:8]], dim=0)

    return qpos_t.unsqueeze(0)


def maybe_init_optimizer_for_episode(
    optimizer: Optimizer,
    obs,
    camera: str,
    first_niters: int,
    device: str,
) -> None:
    del first_niters  # retained for CLI compatibility with training-generation settings
    optimizer.reset_optimizer()
    rgb = torch.from_numpy(getattr(obs, f"{camera}_rgb").astype(np.float32)).to(device)
    depth = torch.from_numpy(getattr(obs, f"{camera}_depth").astype(np.float32)).to(device)
    optimizer.set_frame(rgb, optimizer.cam2world_ns_ds, depth)
    optimizer.init_obj_pose()


def main() -> None:
    parser = argparse.ArgumentParser(description="Dynamic POGS + PointNet++ + ACT simulation loop")
    parser.add_argument("--pogs-config", required=True, help="Path to trained POGS config.yml")
    parser.add_argument("--act-ckpt", required=True, help="Path to trained ACT checkpoint")
    parser.add_argument("--pointnet2-ckpt", required=True, help="Path to pretrained PointNet++ checkpoint")
    parser.add_argument("--task", default="open_drawer")
    parser.add_argument("--data-root", required=True, help="Raw RLBench split root, e.g. data/rlbench/raw/val")
    parser.add_argument("--episodes", type=int, default=25)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--camera", default="front")
    parser.add_argument("--max-points", type=int, default=8192)
    parser.add_argument("--first-niters", type=int, default=15)
    parser.add_argument("--niters", type=int, default=5)
    parser.add_argument("--grid-size", type=float, default=0.005)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    from rlbench.action_modes.action_mode import MoveArmThenGripper
    from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning
    from rlbench.action_modes.gripper_action_modes import Discrete
    from rlbench.environment import Environment
    from rlbench.observation_config import ObservationConfig

    device = "cuda" if torch.cuda.is_available() else "cpu"

    obs_config = ObservationConfig()
    obs_config.set_all(False)
    getattr(obs_config, f"{args.camera}_camera").set_all(True)
    obs_config.gripper_pose = True
    obs_config.gripper_open = True

    env = Environment(
        action_mode=MoveArmThenGripper(
            arm_action_mode=EndEffectorPoseViaPlanning(),
            gripper_action_mode=Discrete(),
        ),
        dataset_root=args.data_root,
        obs_config=obs_config,
        headless=args.headless,
    )
    env.launch()

    task_cls = task_file_to_task_class(args.task)
    task = env.get_task(task_cls)

    import hydra
    from omegaconf import OmegaConf
    try:
        OmegaConf.register_new_resolver("eval", eval)
    except ValueError:
        pass
    ckpt_path = Path(args.act_ckpt)
    run_dir = ckpt_path.parent.parent
    cfg = OmegaConf.load(run_dir / ".hydra" / "config.yaml")

    act_model = hydra.utils.instantiate(cfg.model)
    act_state = torch.load(ckpt_path, map_location=device)["state_dict"]
    act_model.load_state_dict(act_state)
    act_model.eval()
    act_model = act_model.to(device)

    pointnet2 = PointNet2Encoder(in_channels=6, pretrained_path=args.pointnet2_ckpt, freeze=True)
    pointnet2 = pointnet2.to(device)
    pointnet2.eval()
    transform = build_pcd_transform(args.grid_size)

    episodes_root = Path(args.data_root) / args.task / "all_variations" / "episodes"
    episode_ids = []
    for d in sorted(episodes_root.glob("episode*")):
        if d.is_dir():
            try:
                episode_ids.append(int(d.name.replace("episode", "")))
            except ValueError:
                continue
    episode_ids = episode_ids[: args.episodes]
    if not episode_ids:
        raise RuntimeError(f"No episodes found under {episodes_root}")

    first_ep = episode_ids[0]
    var_path = episodes_root / f"episode{first_ep}" / "variation_number.pkl"
    with var_path.open("rb") as f:
        first_var = pickle.load(f)

    task.set_variation(-1)
    first_demo = task.get_demos(
        1,
        random_selection=False,
        live_demos=False,
        from_episode_number=first_ep,
    )[0]
    task.set_variation(first_var)
    first_demo._observations[0].misc["variation_index"] = first_var
    _, first_obs = task.reset_to_demo(first_demo)

    K = np.asarray(first_obs.misc[f"{args.camera}_camera_intrinsics"], dtype=np.float32)
    extrinsics = np.asarray(first_obs.misc[f"{args.camera}_camera_extrinsics"], dtype=np.float32)
    init_cam_pose = torch.from_numpy(extrinsics[:3, :] if extrinsics.shape == (4, 4) else extrinsics).float().unsqueeze(0)

    h, w = getattr(first_obs, f"{args.camera}_rgb").shape[:2]
    optimizer = Optimizer(Path(args.pogs_config), K, w, h, init_cam_pose)

    pos_min = np.array(loc_bounds[args.task][0])
    pos_max = np.array(loc_bounds[args.task][1])

    success = 0
    for ep_id in episode_ids:
        var_path = episodes_root / f"episode{ep_id}" / "variation_number.pkl"
        with var_path.open("rb") as f:
            var_num = pickle.load(f)

        task.set_variation(-1)
        demo = task.get_demos(
            1,
            random_selection=False,
            live_demos=False,
            from_episode_number=ep_id,
        )[0]
        task.set_variation(var_num)
        demo._observations[0].misc["variation_index"] = var_num
        _, obs = task.reset_to_demo(demo)

        maybe_init_optimizer_for_episode(optimizer, obs, args.camera, args.first_niters, device)

        ep_success = False
        for step_id in range(args.max_steps):
            rgb = torch.from_numpy(getattr(obs, f"{args.camera}_rgb").astype(np.float32)).to(device)
            depth = torch.from_numpy(getattr(obs, f"{args.camera}_depth").astype(np.float32)).to(device)

            optimizer.set_observation(rgb, optimizer.cam2world_ns_ds, depth)
            optimizer.step_opt(niter=args.first_niters if step_id == 0 else args.niters)

            coords, colors = extract_full_scene_pointcloud(optimizer, args.max_points)
            obs_embed = encode_obs_embed(coords, colors, transform, pointnet2, device)
            qpos = obs_to_qpos(obs, args.task, collision=True, device=device)

            data_dict = {
                "qpos": qpos,
                "obs_embeds": obs_embed,
                "actions": None,
                "is_pad": None,
            }

            with torch.no_grad():
                pred_action_chunk = act_model(data_dict)["a_hat"][0].cpu().numpy()

            pred_action_chunk[..., :3] = (pred_action_chunk[..., :3] + 1.0) / 2.0 * (pos_max - pos_min) + pos_min
            pred_action_chunk[..., -1] = (pred_action_chunk[..., -1] > 0.5).astype(float)
            pred_action_chunk[..., -2] = (pred_action_chunk[..., -2] > 0.5).astype(float)

            action = pred_action_chunk[0]
            try:
                obs, reward, terminate = task.step(action)
            except (IKError, ConfigurationPathError, InvalidActionError):
                break

            if reward == 1:
                ep_success = True
                break
            if terminate:
                break

        success += int(ep_success)
        print(f"Episode {ep_id}: success={int(ep_success)}")

    env.shutdown()
    success_rate = success / float(len(episode_ids))
    print(f"Success: {success}/{len(episode_ids)} ({success_rate:.3f})")


if __name__ == "__main__":
    main()
