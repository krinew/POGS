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
import torch
from pyrep.errors import ConfigurationPathError, IKError
from rlbench.backend.exceptions import InvalidActionError

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(_SCRIPT_DIR, "../../")))
sys.path.insert(0, os.path.normpath(os.path.join(_SCRIPT_DIR, "../../../PointCloudMatters")))

from pogs.tracking.optim import Optimizer
from pogs.gs_to_pointcloud import extract_pointcloud_from_gaussians
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


def extract_full_scene_pointcloud(optimizer: Optimizer, max_points: int) -> tuple[np.ndarray, np.ndarray, int]:
    coords, colors = extract_pointcloud_from_gaussians(
        optimizer.pipeline.model.gauss_params,
        opacity_threshold=0.0,
        max_points=None,
    )
    raw_points = int(coords.shape[0])
    coords_sampled, colors_sampled = sample_to_fixed_size(coords, colors, max_points)
    return coords_sampled, colors_sampled, raw_points


def _cosine_1d(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
    return float(np.dot(a, b) / denom)


def _short_vec(vec: np.ndarray, precision: int = 4) -> str:
    return np.array2string(np.asarray(vec), precision=precision, suppress_small=True)


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
    ignore_coll = getattr(obs, "ignore_collisions", 0.0)
    if collision:
        qpos = np.concatenate(
            [obs.gripper_pose, [obs.gripper_open], [ignore_coll]],
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
    
    import torchvision.transforms.functional as TVF
    cam = optimizer.cam2world_ns_ds
    tgt_h = int(cam.height.item() if isinstance(cam.height, torch.Tensor) else cam.height)
    tgt_w = int(cam.width.item() if isinstance(cam.width, torch.Tensor) else cam.width)
    if rgb.shape[0] != tgt_h or rgb.shape[1] != tgt_w:
        rgb = TVF.resize(rgb.permute(2, 0, 1), [tgt_h, tgt_w], antialias=True).permute(1, 2, 0)
        dim = depth.dim()
        if dim == 2: depth = depth.unsqueeze(-1)
        depth = TVF.resize(depth.permute(2, 0, 1), [tgt_h, tgt_w], antialias=True).permute(1, 2, 0)
        if dim == 2: depth = depth.squeeze(-1)
        
    optimizer.set_frame(rgb, optimizer.cam2world_ns_ds, depth)
    optimizer.init_obj_pose()


def main() -> None:
    parser = argparse.ArgumentParser(description="Dynamic POGS + PointNet++ + ACT simulation loop")
    parser.add_argument("--pogs-config", required=True, help="Path to trained POGS config.yml")
    parser.add_argument("--act-ckpt", required=True, help="Path to trained ACT checkpoint")
    parser.add_argument("--pointnet2-ckpt", required=True, help="Path to pretrained PointNet++ checkpoint")
    parser.add_argument("--task", default="open_drawer")
    parser.add_argument("--data-root", required=True, help="Raw RLBench split root, e.g. data/rlbench/raw/val")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to run (compatibility alias for --max-episodes)")
    parser.add_argument("--start-episode", type=int, default=1)
    parser.add_argument("--max-episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--camera", default="front")
    parser.add_argument("--max-points", type=int, default=8192)
    parser.add_argument("--first-niters", type=int, default=15)
    parser.add_argument("--niters", type=int, default=5)
    parser.add_argument("--grid-size", type=float, default=0.005)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--verify-pipeline",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print step-level checks for POGS embedding and ACT actions.",
    )
    parser.add_argument(
        "--verify-every",
        type=int,
        default=1,
        help="Print verification every N control steps.",
    )
    args = parser.parse_args()

    from rlbench.action_modes.action_mode import MoveArmThenGripper
    from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning
    from rlbench.action_modes.gripper_action_modes import Discrete
    from rlbench.environment import Environment
    from rlbench.observation_config import ObservationConfig

    device = "cuda" if torch.cuda.is_available() else "cpu"

    obs_config = ObservationConfig()
    obs_config.set_all(False)
    camera_obs_config = getattr(obs_config, f"{args.camera}_camera")
    camera_obs_config.set_all(True)
    camera_obs_config.depth_in_meters = True
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

    # Monkey patch to avoid waypoints validation triggering QTimer crashes in headless=False
    if hasattr(task._scene.task, 'validate'):
        print("[INFO] Patching task.validate() to avoid PyRep IK crash during reset_to_demo.")
        task._scene.task.validate = lambda: None

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
    episode_ids = [eid for eid in episode_ids if eid >= args.start_episode]
    max_count = args.max_episodes if args.max_episodes > 0 else args.episodes
    episode_ids = episode_ids[:max_count]
    if not episode_ids:
        raise RuntimeError(f"No episodes found under {episodes_root} with start_episode={args.start_episode}")

    first_ep = episode_ids[0]
    var_path = episodes_root / f"episode{first_ep}" / "variation_number.pkl"
    with var_path.open("rb") as f:
        first_var = pickle.load(f)

    print("[INFO] Setting PyRep task variation and resetting to baseline...")
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

    print("[INFO] Constructing POGS Online Optimizer (this will take a very long time ~30s)...")
    K = np.asarray(first_obs.misc[f"{args.camera}_camera_intrinsics"], dtype=np.float32)
    K = K.copy()
    K[0, 0] = abs(float(K[0, 0]))
    K[1, 1] = abs(float(K[1, 1]))
    extrinsics = np.asarray(first_obs.misc[f"{args.camera}_camera_extrinsics"], dtype=np.float32)
    init_cam_pose = torch.from_numpy(extrinsics[:3, :] if extrinsics.shape == (4, 4) else extrinsics).float().unsqueeze(0)

    h, w = getattr(first_obs, f"{args.camera}_rgb").shape[:2]
    optimizer = Optimizer(Path(args.pogs_config), K, w, h, init_cam_pose)
    print("[INFO] POGS Optimizer initialized and ready!")

    pos_min = np.array(loc_bounds[args.task][0])
    pos_max = np.array(loc_bounds[args.task][1])

    success = 0
    verify_every = max(1, int(args.verify_every))
    for ep_id in episode_ids:
        var_path = episodes_root / f"episode{ep_id}" / "variation_number.pkl"
        with var_path.open("rb") as f:
            var_num = pickle.load(f)

        print(f"\n[INFO] Starting Episode {ep_id}")
        print(f"[INFO] Fetching demos to reset PyRep physically...")
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

        print(f"[INFO] Running first step alignment in POGS...")
        maybe_init_optimizer_for_episode(optimizer, obs, args.camera, args.first_niters, device)
        prev_embed_np: np.ndarray | None = None

        ep_success = False
        for step_id in range(args.max_steps):
            print(f"[DEBUG] Ep {ep_id} | Step {step_id}/{args.max_steps} | Optimizing...", end="\r", flush=True)
            rgb = torch.from_numpy(getattr(obs, f"{args.camera}_rgb").astype(np.float32)).to(device)
            depth = torch.from_numpy(getattr(obs, f"{args.camera}_depth").astype(np.float32)).to(device)

            import torchvision.transforms.functional as TVF
            cam = optimizer.cam2world_ns_ds
            tgt_h = int(cam.height.item() if isinstance(cam.height, torch.Tensor) else cam.height)
            tgt_w = int(cam.width.item() if isinstance(cam.width, torch.Tensor) else cam.width)
            
            if rgb.shape[0] != tgt_h or rgb.shape[1] != tgt_w:
                rgb = TVF.resize(rgb.permute(2, 0, 1), [tgt_h, tgt_w], antialias=True).permute(1, 2, 0)
                dim = depth.dim()
                if dim == 2: depth = depth.unsqueeze(-1)
                depth = TVF.resize(depth.permute(2, 0, 1), [tgt_h, tgt_w], antialias=True).permute(1, 2, 0)
                if dim == 2: depth = depth.squeeze(-1)

            optimizer.set_observation(rgb, optimizer.cam2world_ns_ds, depth)
            optimizer.step_opt(niter=args.first_niters if step_id == 0 else args.niters, use_depth=True)

            coords, colors, raw_points = extract_full_scene_pointcloud(optimizer, args.max_points)
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

            embed_np = obs_embed.squeeze(0).detach().cpu().numpy()
            if args.verify_pipeline and (step_id % verify_every == 0):
                embed_norm = float(np.linalg.norm(embed_np))
                embed_std = float(embed_np.std())
                embed_min = float(embed_np.min())
                embed_max = float(embed_np.max())
                embed_cos_prev = _cosine_1d(prev_embed_np, embed_np) if prev_embed_np is not None else float("nan")
                act0 = pred_action_chunk[0]
                act0_quat_norm = float(np.linalg.norm(act0[3:7]))
                print(
                    "\n"
                    f"[VERIFY] Ep {ep_id} Step {step_id} | "
                    f"pcd_raw={raw_points} pcd_used={coords.shape[0]} | "
                    f"embed_dim={embed_np.shape[0]} norm={embed_norm:.4f} std={embed_std:.4f} "
                    f"min={embed_min:.4f} max={embed_max:.4f} cos_prev={embed_cos_prev:.4f} | "
                    f"act0_pos={_short_vec(act0[:3], precision=3)} "
                    f"act0_quat_norm={act0_quat_norm:.4f} "
                    f"act0_gripper={int(act0[-2])} act0_collision={int(act0[-1])}"
                )

            max_retry = min(pred_action_chunk.shape[0], 20 if step_id == 0 else 3)
            step_ok = False
            reward, terminate = 0.0, False
            last_err: Exception | None = None
            chosen_retry = -1
            chosen_variant = "none"
            chosen_action: np.ndarray | None = None
            ee_pos_before = np.asarray(obs.gripper_pose[:3], dtype=np.float32)

            for retry_id in range(max_retry):
                cand = pred_action_chunk[retry_id].copy()

                # Keep Cartesian targets near current EE pose to reduce planning failures.
                cur_pos = np.asarray(obs.gripper_pose[:3], dtype=np.float32)
                delta = cand[:3] - cur_pos
                dist = float(np.linalg.norm(delta))
                max_pos_delta = 0.05
                if dist > max_pos_delta:
                    cand[:3] = cur_pos + (delta / (dist + 1e-8)) * max_pos_delta

                quat = cand[3:7].astype(np.float32)
                quat_norm = float(np.linalg.norm(quat))
                if quat_norm < 1e-6:
                    quat = np.asarray(obs.gripper_pose[3:7], dtype=np.float32)
                    quat_norm = float(np.linalg.norm(quat))
                quat = quat / (quat_norm + 1e-8)

                gripper_open = float(cand[-2])
                action_primary = np.concatenate([cand[:3], quat, [gripper_open]], axis=0).astype(np.float32)

                # Fallback ordering in case quaternion convention differs.
                action_alt_quat = action_primary.copy()
                action_alt_quat[3:7] = np.array([quat[1], quat[2], quat[3], quat[0]], dtype=np.float32)

                for variant_name, action in (("primary", action_primary), ("alt_quat", action_alt_quat)):
                    try:
                        obs, reward, terminate = task.step(action)
                        step_ok = True
                        chosen_retry = retry_id
                        chosen_variant = variant_name
                        chosen_action = action.copy()
                        break
                    except (IKError, ConfigurationPathError, InvalidActionError, RuntimeError) as e:
                        last_err = e

                if step_ok:
                    break

            if not step_ok:
                print(f"\n[ERROR] Task step failed after {max_retry} retries: {type(last_err).__name__}: {last_err}")
                break

            if args.verify_pipeline and (step_id % verify_every == 0) and chosen_action is not None:
                commanded_delta = float(np.linalg.norm(chosen_action[:3] - ee_pos_before))
                print(
                    f"[VERIFY] ActionExec Ep {ep_id} Step {step_id} | "
                    f"retry={chosen_retry} variant={chosen_variant} "
                    f"delta_xyz={commanded_delta:.4f} reward={float(reward):.3f} terminate={bool(terminate)}"
                )

            prev_embed_np = embed_np

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
