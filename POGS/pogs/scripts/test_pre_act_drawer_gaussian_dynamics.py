"""Pre-ACT diagnostic: replay RLBench demo actions and verify Gaussian dynamics.

This script checks whether tracked Gaussian clusters move during the pull phase
of an RLBench episode (for tasks like open_drawer), without using ACT.

Workflow:
1. Reset simulator to a recorded episode demo.
2. Replay ground-truth end-effector actions from that demo.
3. Run POGS tracking at each frame.
4. Measure per-cluster centroid motion and score pull-phase dynamics.
"""

from __future__ import annotations

import argparse
import json
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

from pogs.tracking.optim import Optimizer


def task_file_to_task_class(task_file: str):
    import importlib

    name = task_file.replace(".py", "")
    class_name = "".join([w[0].upper() + w[1:] for w in name.split("_")])
    mod = importlib.import_module(f"rlbench.tasks.{name}")
    mod = importlib.reload(mod)
    return getattr(mod, class_name)


def _obs_to_tensors(obs, camera: str, device: str, target_hw: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
    import torchvision.transforms.functional as tvf

    rgb = torch.from_numpy(getattr(obs, f"{camera}_rgb").astype(np.float32)).to(device)
    depth = torch.from_numpy(getattr(obs, f"{camera}_depth").astype(np.float32)).to(device)

    tgt_h, tgt_w = target_hw
    if rgb.shape[0] != tgt_h or rgb.shape[1] != tgt_w:
        rgb = tvf.resize(rgb.permute(2, 0, 1), [tgt_h, tgt_w], antialias=True).permute(1, 2, 0)
        depth_dim = depth.dim()
        if depth_dim == 2:
            depth = depth.unsqueeze(-1)
        depth = tvf.resize(depth.permute(2, 0, 1), [tgt_h, tgt_w], antialias=True).permute(1, 2, 0)
        if depth_dim == 2:
            depth = depth.squeeze(-1)

    return rgb, depth


def _init_optimizer_for_episode(
    optimizer: Optimizer,
    first_obs,
    camera: str,
    device: str,
) -> tuple[int, int]:
    optimizer.reset_optimizer()
    cam = optimizer.cam2world_ns_ds
    tgt_h = int(cam.height.item() if isinstance(cam.height, torch.Tensor) else cam.height)
    tgt_w = int(cam.width.item() if isinstance(cam.width, torch.Tensor) else cam.width)
    rgb, depth = _obs_to_tensors(first_obs, camera, device, (tgt_h, tgt_w))
    optimizer.set_frame(rgb, optimizer.cam2world_ns_ds, depth)
    optimizer.init_obj_pose()
    return tgt_h, tgt_w


def _extract_scene_coords_and_clusters(
    optimizer: Optimizer,
    max_points: int,
    opacity_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    with torch.no_grad():
        optimizer.optimizer.apply_to_model(optimizer.optimizer.part_deltas, optimizer.group_labels)

        prev_state = optimizer.pipeline.state_stack[-1]
        means = prev_state["means"].detach().cpu().float().clone()
        tracked_means = optimizer.pipeline.model.gauss_params["means"].detach().cpu().float()
        keep_inds = optimizer.keep_inds.cpu() if optimizer.keep_inds is not None else optimizer.pipeline.model.keep_inds.cpu()

        if tracked_means.shape[0] == keep_inds.shape[0]:
            means[keep_inds] = tracked_means

        if hasattr(optimizer.pipeline.model, "cluster_labels") and optimizer.pipeline.model.cluster_labels is not None:
            cluster_labels = optimizer.pipeline.model.cluster_labels.detach().cpu().reshape(-1)
            cluster_labels = torch.round(cluster_labels).to(torch.int64)
            if cluster_labels.shape[0] != means.shape[0]:
                cluster_labels = torch.zeros((means.shape[0],), dtype=torch.int64)
        else:
            cluster_labels = torch.zeros((means.shape[0],), dtype=torch.int64)

        opacities = prev_state["opacities"].detach().cpu().float().squeeze(-1)
        keep_mask = torch.sigmoid(opacities) > float(opacity_threshold)
        means = means[keep_mask]
        cluster_labels = cluster_labels[keep_mask]

        if means.shape[0] == 0:
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.int32)

        coords = means.numpy().astype(np.float32)
        clusters = cluster_labels.numpy().astype(np.int32)

        if max_points > 0 and coords.shape[0] > max_points:
            idx = np.random.choice(coords.shape[0], max_points, replace=False)
            coords = coords[idx]
            clusters = clusters[idx]

        return coords, clusters


def _cluster_centroids(coords: np.ndarray, clusters: np.ndarray) -> dict[int, dict[str, np.ndarray | int]]:
    out: dict[int, dict[str, np.ndarray | int]] = {}
    if coords.shape[0] == 0:
        return out

    uniq_ids, counts = np.unique(clusters, return_counts=True)
    for cid, cnt in zip(uniq_ids, counts):
        mask = clusters == cid
        centroid = coords[mask].mean(axis=0)
        out[int(cid)] = {
            "count": int(cnt),
            "centroid": centroid.astype(np.float32),
        }
    return out


def _build_replay_action(next_obs) -> np.ndarray:
    # RLBench EndEffectorPoseViaPlanning expects [x, y, z, qx, qy, qz, qw, gripper].
    return np.concatenate(
        [
            np.asarray(next_obs.gripper_pose, dtype=np.float32),
            np.asarray([next_obs.gripper_open], dtype=np.float32),
        ],
        axis=0,
    ).astype(np.float32)


def _alt_quat_variant(action: np.ndarray) -> np.ndarray:
    alt = action.copy()
    quat = alt[3:7].copy()
    alt[3:7] = np.asarray([quat[1], quat[2], quat[3], quat[0]], dtype=np.float32)
    return alt


def _infer_pull_mask(
    ee_positions: list[np.ndarray],
    gripper_open: list[float],
    close_thresh: float,
    min_step_delta: float,
    fallback_window: int,
) -> tuple[np.ndarray, dict[str, float | int | str]]:
    n_obs = len(gripper_open)
    if n_obs < 2:
        return np.zeros((0,), dtype=bool), {"mode": "insufficient_observations"}

    ee = np.asarray(ee_positions, dtype=np.float32)
    go = np.asarray(gripper_open, dtype=np.float32)
    delta = np.linalg.norm(ee[1:] - ee[:-1], axis=1)

    # Transition i corresponds to obs[i] -> obs[i+1], so use go[1:] for "current" grip state.
    pull_mask = (go[1:] <= close_thresh) & (delta >= min_step_delta)
    mode = "closed_and_moving"

    if not pull_mask.any():
        closed_idx = np.where(go <= close_thresh)[0]
        if closed_idx.size > 0:
            start_obs = max(1, int(closed_idx[0]))
            mode = "fallback_from_first_closed"
        else:
            start_obs = max(1, n_obs // 3)
            mode = "fallback_middle_window"
        end_obs = min(n_obs - 1, start_obs + max(1, int(fallback_window)))
        pull_mask = np.zeros((n_obs - 1,), dtype=bool)
        pull_mask[start_obs - 1 : end_obs] = True

    info: dict[str, float | int | str] = {
        "mode": mode,
        "transition_count": int(n_obs - 1),
        "pull_transitions": int(pull_mask.sum()),
        "delta_mean": float(delta.mean()) if delta.size else 0.0,
        "delta_max": float(delta.max()) if delta.size else 0.0,
    }
    return pull_mask, info


def _compute_cluster_metrics(
    snapshots: list[dict[int, dict[str, np.ndarray | int]]],
    pull_mask: np.ndarray,
) -> list[dict[str, float | int]]:
    n_obs = len(snapshots)
    cluster_ids = sorted({cid for snap in snapshots for cid in snap.keys()})

    metrics: list[dict[str, float | int]] = []
    for cid in cluster_ids:
        centroids = np.full((n_obs, 3), np.nan, dtype=np.float32)
        counts = np.zeros((n_obs,), dtype=np.int32)

        for t, snap in enumerate(snapshots):
            if cid in snap:
                centroids[t] = np.asarray(snap[cid]["centroid"], dtype=np.float32)
                counts[t] = int(snap[cid]["count"])

        disp = np.linalg.norm(centroids[1:] - centroids[:-1], axis=1)
        valid = np.isfinite(disp)
        if not valid.any():
            continue

        disp = np.where(valid, disp, np.nan)
        total_disp = float(np.nansum(disp))
        pull_valid = valid & pull_mask
        pull_disp = float(np.nansum(disp[pull_valid])) if pull_valid.any() else 0.0
        nonpull_disp = max(total_disp - pull_disp, 0.0)
        ratio = float(pull_disp / (nonpull_disp + 1e-8))

        observed_counts = counts[counts > 0]
        metrics.append(
            {
                "cluster_id": int(cid),
                "support_steps": int(valid.sum()),
                "observed_frames": int((counts > 0).sum()),
                "mean_points": float(observed_counts.mean()) if observed_counts.size else 0.0,
                "max_step_disp": float(np.nanmax(disp)),
                "total_disp": total_disp,
                "pull_disp": pull_disp,
                "nonpull_disp": nonpull_disp,
                "pull_to_nonpull_ratio": ratio,
            }
        )

    metrics.sort(key=lambda m: float(m["pull_disp"]), reverse=True)
    return metrics


def _select_drawer_cluster(
    metrics: list[dict[str, float | int]],
    drawer_cluster_id: int | None,
    min_support_steps: int,
    min_cluster_points: int,
) -> dict[str, float | int] | None:
    if not metrics:
        return None

    if drawer_cluster_id is not None:
        for m in metrics:
            if int(m["cluster_id"]) == int(drawer_cluster_id):
                return m
        return None

    eligible = [
        m
        for m in metrics
        if int(m["support_steps"]) >= int(min_support_steps)
        and float(m["mean_points"]) >= float(min_cluster_points)
    ]
    if eligible:
        return max(eligible, key=lambda m: float(m["pull_disp"]))

    return metrics[0]


def _build_group_diagnostics(optimizer: Optimizer) -> dict[str, object]:
    rigid_group_count = int(len(optimizer.group_masks))
    group_sizes = [int(mask.sum().item()) for mask in optimizer.group_masks]

    cluster_info: list[dict[str, int]] = []
    cluster_unique_count = 0
    cluster_labels = getattr(optimizer.pipeline.model, "cluster_labels", None)
    if cluster_labels is not None:
        labels = cluster_labels.detach().reshape(-1)
        if optimizer.keep_inds is not None:
            keep_inds = optimizer.keep_inds.detach().reshape(-1).to(torch.long)
            keep_inds = keep_inds[(keep_inds >= 0) & (keep_inds < labels.shape[0])]
            if keep_inds.numel() > 0:
                labels = labels[keep_inds]
        labels = torch.round(labels).to(torch.int64)
        uniq, counts = torch.unique(labels, return_counts=True)
        cluster_unique_count = int(uniq.numel())
        pairs = sorted(
            zip(uniq.detach().cpu().tolist(), counts.detach().cpu().tolist()),
            key=lambda x: x[1],
            reverse=True,
        )
        cluster_info = [{"cluster_id": int(cid), "count": int(cnt)} for cid, cnt in pairs[:12]]

    keep_count = int(optimizer.keep_inds.numel()) if optimizer.keep_inds is not None else int(sum(group_sizes))
    suspicious_single_group = rigid_group_count == 1 and cluster_unique_count > 1

    return {
        "rigid_group_count": rigid_group_count,
        "rigid_group_sizes": group_sizes,
        "tracked_keep_count": keep_count,
        "cluster_unique_count": cluster_unique_count,
        "top_clusters": cluster_info,
        "single_rigid_group_with_multiple_clusters": suspicious_single_group,
    }


def _tracking_loss_proxy(optimizer: Optimizer, use_depth: bool = False) -> float:
    # This is the same DINO-driven objective used inside the optimizer step.
    frame = optimizer.optimizer.frame if optimizer.optimizer.config.use_roi else optimizer.optimizer.frame.frame
    with torch.no_grad():
        loss, _ = optimizer.optimizer.get_optim_loss(
            frame=frame,
            part_deltas=optimizer.optimizer.part_deltas.detach().clone(),
            use_dino=True,
            use_depth=bool(use_depth),
            use_rgb=False,
            use_atap=False,
            use_hand_mask=False,
            use_mask=False,
            use_roi=optimizer.optimizer.config.use_roi,
        )
    if loss is None:
        return float("nan")
    return float(loss.detach().item())


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay RLBench demo actions and verify drawer Gaussian dynamics")
    parser.add_argument("--task", default="open_drawer")
    parser.add_argument("--raw-root", required=True, help="RLBench raw split root, e.g. .../data/rlbench/raw/train")
    parser.add_argument("--pogs-config", required=True, help="Path to trained POGS config.yml")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--camera", default="front")
    parser.add_argument("--max-steps", type=int, default=-1, help="-1 = use full demo")
    parser.add_argument("--max-points", type=int, default=8192)
    parser.add_argument("--first-niters", type=int, default=15)
    parser.add_argument("--niters", type=int, default=5)
    parser.add_argument("--opacity-threshold", type=float, default=0.05)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--drawer-cluster-id", type=int, default=None, help="Use a known drawer cluster id")
    parser.add_argument("--min-cluster-points", type=int, default=80)
    parser.add_argument("--min-support-steps", type=int, default=8)
    parser.add_argument("--close-thresh", type=float, default=0.5)
    parser.add_argument("--pull-min-step-delta", type=float, default=0.002)
    parser.add_argument("--fallback-pull-window", type=int, default=25)
    parser.add_argument("--pull-disp-thresh", type=float, default=0.02)
    parser.add_argument("--total-disp-thresh", type=float, default=0.03)
    parser.add_argument("--pull-ratio-thresh", type=float, default=0.10)
    parser.add_argument(
        "--enforce-pull-ratio",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If enabled, fail when pull_to_nonpull_ratio is below threshold.",
    )
    parser.add_argument("--verify-every", type=int, default=10)
    parser.add_argument("--json-out", type=str, default="")
    args = parser.parse_args()

    from rlbench.action_modes.action_mode import MoveArmThenGripper
    from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning
    from rlbench.action_modes.gripper_action_modes import Discrete
    from rlbench.environment import Environment
    from rlbench.observation_config import ObservationConfig

    raw_root = Path(args.raw_root)
    episodes_root = raw_root / args.task / "all_variations" / "episodes"
    ep_path = episodes_root / f"episode{args.episode}"
    if not ep_path.exists():
        raise FileNotFoundError(f"Episode directory not found: {ep_path}")

    var_path = ep_path / "variation_number.pkl"
    if not var_path.exists():
        raise FileNotFoundError(f"Variation file not found: {var_path}")

    with var_path.open("rb") as f:
        variation_id = pickle.load(f)

    obs_config = ObservationConfig()
    obs_config.set_all(False)
    getattr(obs_config, f"{args.camera}_camera").set_all(True)
    obs_config.gripper_pose = True
    obs_config.gripper_open = True

    env = Environment(
        action_mode=MoveArmThenGripper(
            arm_action_mode=EndEffectorPoseViaPlanning(collision_checking=False),
            gripper_action_mode=Discrete(),
        ),
        dataset_root=str(raw_root),
        obs_config=obs_config,
        headless=args.headless,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("[INFO] Launching RLBench environment...")
    env.launch()
    try:
        task_cls = task_file_to_task_class(args.task)
        task = env.get_task(task_cls)

        if hasattr(task._scene.task, "validate"):
            print("[INFO] Patching task.validate() to avoid reset-to-demo IK checks.")
            task._scene.task.validate = lambda: None

        task.set_variation(-1)
        demo = task.get_demos(
            1,
            random_selection=False,
            live_demos=False,
            from_episode_number=args.episode,
        )[0]
        task.set_variation(variation_id)
        demo._observations[0].misc["variation_index"] = variation_id
        _, obs = task.reset_to_demo(demo)

        demo_obs = list(demo._observations)
        if len(demo_obs) < 2:
            raise RuntimeError("Demo must have at least 2 observations.")

        max_frames = len(demo_obs)
        if args.max_steps > 0:
            max_frames = min(max_frames, int(args.max_steps) + 1)

        K = np.asarray(obs.misc[f"{args.camera}_camera_intrinsics"], dtype=np.float32)
        extr = np.asarray(obs.misc[f"{args.camera}_camera_extrinsics"], dtype=np.float32)
        if extr.shape == (4, 4):
            init_cam_pose = torch.from_numpy(extr[:3, :]).float().unsqueeze(0)
        elif extr.shape == (3, 4):
            init_cam_pose = torch.from_numpy(extr).float().unsqueeze(0)
        else:
            raise RuntimeError(f"Unexpected extrinsics shape: {extr.shape}")

        h, w = getattr(obs, f"{args.camera}_rgb").shape[:2]
        optimizer = Optimizer(Path(args.pogs_config), K, w, h, init_cam_pose)
        target_h, target_w = _init_optimizer_for_episode(optimizer, obs, args.camera, device)

        group_diag = _build_group_diagnostics(optimizer)
        print("[INFO] Rigid-group diagnostics:")
        print(
            "  "
            f"rigid_groups={group_diag['rigid_group_count']} "
            f"keep_points={group_diag['tracked_keep_count']} "
            f"cluster_labels_in_keep={group_diag['cluster_unique_count']}"
        )
        print(f"  rigid_group_sizes={group_diag['rigid_group_sizes']}")
        if group_diag["single_rigid_group_with_multiple_clusters"]:
            print(
                "[WARN] Single rigid transform is controlling multiple semantic clusters. "
                "If drawer + cabinet are in one crop group, motion can be diluted or visually suppressed."
            )

        ee_positions: list[np.ndarray] = []
        gripper_open: list[float] = []
        reward_trace: list[float] = []
        terminate_trace: list[bool] = []
        snapshots: list[dict[int, dict[str, np.ndarray | int]]] = []
        tracking_checks: list[dict[str, float | int | bool]] = []

        verify_every = max(1, int(args.verify_every))
        print(f"[INFO] Replaying episode={args.episode} for up to {max_frames} frames...")

        for frame_idx in range(max_frames):
            rgb, depth = _obs_to_tensors(obs, args.camera, device, (target_h, target_w))
            optimizer.set_observation(rgb, optimizer.cam2world_ns_ds, depth)

            do_2d_check = (frame_idx % verify_every == 0)
            loss_before = float("nan")
            if do_2d_check:
                loss_before = _tracking_loss_proxy(optimizer, use_depth=False)

            part_before = optimizer.optimizer.part_deltas.detach().clone()
            optimizer.step_opt(niter=args.first_niters if frame_idx == 0 else args.niters)
            part_after = optimizer.optimizer.part_deltas.detach().clone()
            part_delta_step = float(torch.linalg.norm(part_after - part_before).item())

            loss_after = float("nan")
            loss_drop = float("nan")
            improved = False
            if do_2d_check:
                loss_after = _tracking_loss_proxy(optimizer, use_depth=False)
                if np.isfinite(loss_before) and np.isfinite(loss_after):
                    loss_drop = float(loss_before - loss_after)
                    improved = bool(loss_drop > 0.0)
                tracking_checks.append(
                    {
                        "frame": int(frame_idx),
                        "dino_loss_before": float(loss_before),
                        "dino_loss_after": float(loss_after),
                        "dino_loss_drop": float(loss_drop) if np.isfinite(loss_drop) else float("nan"),
                        "part_delta_step_l2": part_delta_step,
                        "improved": bool(improved),
                    }
                )

            coords, clusters = _extract_scene_coords_and_clusters(
                optimizer,
                max_points=args.max_points,
                opacity_threshold=args.opacity_threshold,
            )
            snap = _cluster_centroids(coords, clusters)
            snapshots.append(snap)

            ee = np.asarray(obs.gripper_pose[:3], dtype=np.float32)
            go = float(obs.gripper_open)
            ee_positions.append(ee)
            gripper_open.append(go)

            if frame_idx % verify_every == 0:
                uniq_count = len(snap)
                print(
                    f"[VERIFY] frame={frame_idx:03d} points={coords.shape[0]:05d} "
                    f"clusters={uniq_count:03d} gripper_open={go:.3f} ee={np.round(ee, 4)}"
                )
                if do_2d_check:
                    trend = "improved" if improved else "not_improved"
                    print(
                        "[VERIFY_2D] "
                        f"frame={frame_idx:03d} "
                        f"dino_loss_before={loss_before:.6f} "
                        f"dino_loss_after={loss_after:.6f} "
                        f"drop={loss_drop:.6f} "
                        f"part_delta_l2={part_delta_step:.6f} "
                        f"status={trend}"
                    )

            if frame_idx == max_frames - 1:
                break

            next_obs = demo_obs[frame_idx + 1]
            primary_action = _build_replay_action(next_obs)
            alt_action = _alt_quat_variant(primary_action)

            stepped = False
            last_err: Exception | None = None
            chosen_variant = "primary"
            for variant_name, action in (("primary", primary_action), ("alt_quat", alt_action)):
                try:
                    obs, reward, terminate = task.step(action)
                    reward_trace.append(float(reward))
                    terminate_trace.append(bool(terminate))
                    chosen_variant = variant_name
                    stepped = True
                    break
                except (IKError, ConfigurationPathError, InvalidActionError, RuntimeError) as exc:
                    last_err = exc

            if not stepped:
                raise RuntimeError(
                    "Replay action failed at frame "
                    f"{frame_idx} ({type(last_err).__name__}: {last_err})"
                )

            if frame_idx % verify_every == 0:
                print(
                    f"[VERIFY] replay_step={frame_idx:03d} variant={chosen_variant} "
                    f"reward={reward_trace[-1]:.3f} terminate={terminate_trace[-1]}"
                )

            if terminate_trace[-1]:
                print(f"[INFO] Episode terminated by RLBench at replay step {frame_idx}.")
                break

        if len(snapshots) < 2:
            raise RuntimeError("Need at least 2 tracked frames to evaluate Gaussian dynamics.")

        pull_mask, pull_info = _infer_pull_mask(
            ee_positions=ee_positions,
            gripper_open=gripper_open,
            close_thresh=args.close_thresh,
            min_step_delta=args.pull_min_step_delta,
            fallback_window=args.fallback_pull_window,
        )

        metrics = _compute_cluster_metrics(snapshots, pull_mask)
        candidate = _select_drawer_cluster(
            metrics=metrics,
            drawer_cluster_id=args.drawer_cluster_id,
            min_support_steps=args.min_support_steps,
            min_cluster_points=args.min_cluster_points,
        )

        print("\n[SUMMARY] Top clusters by pull_disp:")
        for row in metrics[:10]:
            print(
                "  "
                f"cid={int(row['cluster_id']):>4d} "
                f"pull={float(row['pull_disp']):.4f} "
                f"total={float(row['total_disp']):.4f} "
                f"ratio={float(row['pull_to_nonpull_ratio']):.3f} "
                f"support={int(row['support_steps']):>3d} "
                f"mean_pts={float(row['mean_points']):.1f}"
            )

        valid_tracking = [
            c for c in tracking_checks if np.isfinite(float(c["dino_loss_before"])) and np.isfinite(float(c["dino_loss_after"]))
        ]
        improved_count = sum(1 for c in valid_tracking if bool(c["improved"]))
        if valid_tracking:
            drops = np.asarray([float(c["dino_loss_drop"]) for c in valid_tracking], dtype=np.float32)
            step_l2 = np.asarray([float(c["part_delta_step_l2"]) for c in valid_tracking], dtype=np.float32)
            print("\n[SUMMARY_2D]")
            print(
                "  "
                f"checks={len(valid_tracking)} improved={improved_count} "
                f"improve_rate={improved_count / max(1, len(valid_tracking)):.3f}"
            )
            print(
                "  "
                f"dino_loss_drop_mean={float(drops.mean()):.6f} "
                f"dino_loss_drop_min={float(drops.min()):.6f} "
                f"part_delta_step_l2_mean={float(step_l2.mean()):.6f}"
            )
        else:
            print("\n[SUMMARY_2D]")
            print("  checks=0 (no finite DINO-loss diagnostics collected)")

        reasons: list[str] = []
        passed = True
        if candidate is None:
            passed = False
            reasons.append("No candidate cluster found.")
        else:
            if float(candidate["pull_disp"]) < float(args.pull_disp_thresh):
                passed = False
                reasons.append(
                    f"pull_disp {float(candidate['pull_disp']):.4f} < threshold {float(args.pull_disp_thresh):.4f}"
                )
            if float(candidate["total_disp"]) < float(args.total_disp_thresh):
                passed = False
                reasons.append(
                    f"total_disp {float(candidate['total_disp']):.4f} < threshold {float(args.total_disp_thresh):.4f}"
                )
            if args.enforce_pull_ratio and float(candidate["pull_to_nonpull_ratio"]) < float(args.pull_ratio_thresh):
                passed = False
                reasons.append(
                    "pull_to_nonpull_ratio "
                    f"{float(candidate['pull_to_nonpull_ratio']):.3f} < threshold {float(args.pull_ratio_thresh):.3f}"
                )

        reward_success = any(r >= 1.0 for r in reward_trace)
        print("\n[RESULT]")
        print(f"  pull_mask_mode={pull_info.get('mode')} pull_transitions={pull_info.get('pull_transitions')}")
        print(f"  replay_reward_success={int(reward_success)} max_reward={max(reward_trace) if reward_trace else 0.0:.3f}")
        if candidate is not None:
            print(
                "  chosen_cluster="
                f"{int(candidate['cluster_id'])} "
                f"pull_disp={float(candidate['pull_disp']):.4f} "
                f"total_disp={float(candidate['total_disp']):.4f} "
                f"ratio={float(candidate['pull_to_nonpull_ratio']):.3f}"
            )
        print(f"  gaussian_dynamics_pass={int(passed)}")
        if reasons:
            print("  fail_reasons:")
            for reason in reasons:
                print(f"    - {reason}")

        json_path = Path(args.json_out) if args.json_out else Path("outputs/dynamics_checks") / (
            f"{args.task}_ep_{args.episode}_pre_act_dynamics.json"
        )
        json_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "task": args.task,
            "episode": int(args.episode),
            "camera": args.camera,
            "pogs_config": str(args.pogs_config),
            "frames_tracked": int(len(snapshots)),
            "replay_steps": int(len(reward_trace)),
            "reward_success": bool(reward_success),
            "pull_info": pull_info,
            "thresholds": {
                "pull_disp_thresh": float(args.pull_disp_thresh),
                "total_disp_thresh": float(args.total_disp_thresh),
                "pull_ratio_thresh": float(args.pull_ratio_thresh),
                "enforce_pull_ratio": bool(args.enforce_pull_ratio),
            },
            "group_diagnostics": group_diag,
            "tracking_2d_signal": {
                "checks": int(len(valid_tracking)),
                "improved_checks": int(improved_count),
                "improve_rate": float(improved_count / max(1, len(valid_tracking))) if valid_tracking else 0.0,
            },
            "chosen_cluster": candidate,
            "gaussian_dynamics_pass": bool(passed),
            "fail_reasons": reasons,
            "top_clusters": metrics[:20],
            "tracking_checks": tracking_checks,
        }
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"[INFO] Saved diagnostic report: {json_path}")

        if passed:
            raise SystemExit(0)
        raise SystemExit(2)
    finally:
        env.shutdown()


if __name__ == "__main__":
    main()
