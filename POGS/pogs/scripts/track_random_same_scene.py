"""Run random-action tracking rollouts on the same RLBench scene used for POGS training.

This script is designed to test drift caused by tracking dynamics while removing
scene-mismatch as a confounder:
1. Reset to a specific RLBench episode (fixed scene state)
2. Initialize POGS tracking from that same scene
3. Execute random robot actions for a fixed number of steps
4. Log optimization metrics and save rollout frames for inspection
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pyrep.errors import ConfigurationPathError, IKError
from rlbench.backend.exceptions import InvalidActionError
from scipy.spatial.transform import Rotation

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


def _set_seed(seed: int) -> None:
    np.random.seed(int(seed))
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _infer_episode_from_pogs_config(config_path: Path) -> int | None:
    text = config_path.read_text(encoding="utf-8", errors="ignore")
    matches = [int(x) for x in re.findall(r"ep_(\d+)", text)]
    if not matches:
        return None
    counts: dict[int, int] = {}
    for m in matches:
        counts[m] = counts.get(m, 0) + 1
    return max(counts.items(), key=lambda kv: kv[1])[0]


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


def _init_optimizer_for_episode(optimizer: Optimizer, first_obs, camera: str, device: str) -> tuple[int, int]:
    optimizer.reset_optimizer()
    cam = optimizer.cam2world_ns_ds
    tgt_h = int(cam.height.item() if isinstance(cam.height, torch.Tensor) else cam.height)
    tgt_w = int(cam.width.item() if isinstance(cam.width, torch.Tensor) else cam.width)
    rgb, depth = _obs_to_tensors(first_obs, camera, device, (tgt_h, tgt_w))
    optimizer.set_frame(rgb, optimizer.cam2world_ns_ds, depth)
    optimizer.init_obj_pose()
    return tgt_h, tgt_w


def _tracking_loss_proxy(optimizer: Optimizer, use_depth: bool, use_rgb: bool) -> float:
    get_2d_loss = getattr(optimizer.optimizer, "get_2d_loss", None)
    if not callable(get_2d_loss):
        return float("nan")

    camera = getattr(optimizer.optimizer, "camera", None)
    if camera is None and hasattr(optimizer.optimizer, "frame"):
        frame_obj = optimizer.optimizer.frame
        if hasattr(frame_obj, "camera"):
            camera = frame_obj.camera
        elif hasattr(frame_obj, "frame") and hasattr(frame_obj.frame, "camera"):
            camera = frame_obj.frame.camera
    if camera is None:
        return float("nan")

    with torch.no_grad():
        loss = get_2d_loss(
            optimizer.optimizer.part_deltas,
            optimizer.group_labels,
            camera,
            use_l1=bool(use_rgb),
            use_depth=bool(use_depth),
            use_mask=False,
            use_roi=optimizer.optimizer.config.use_roi,
        )
    if loss is None:
        return float("nan")
    return float(loss.detach().item())


def _load_workspace_bounds(task: str) -> tuple[np.ndarray, np.ndarray] | None:
    try:
        from src.data.components.rlbench.constants import loc_bounds  # type: ignore
    except Exception:
        return None

    if task not in loc_bounds:
        return None
    bounds = np.asarray(loc_bounds[task], dtype=np.float32)
    if bounds.shape != (2, 3):
        return None

    margin = np.asarray([0.01, 0.01, 0.01], dtype=np.float32)
    low = bounds[0] + margin
    high = bounds[1] - margin
    return low, high


def _sample_random_action(
    obs,
    rng: np.random.Generator,
    pos_step: float,
    rot_step_deg: float,
    gripper_flip_prob: float,
    workspace_bounds: tuple[np.ndarray, np.ndarray] | None,
) -> np.ndarray:
    pose = np.asarray(obs.gripper_pose, dtype=np.float32)
    pos = pose[:3].copy()
    quat = pose[3:7].copy()
    quat_norm = np.linalg.norm(quat)
    if quat_norm < 1e-8:
        quat = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    else:
        quat = quat / quat_norm

    pos_delta = rng.uniform(-pos_step, pos_step, size=(3,)).astype(np.float32)
    pos = pos + pos_delta
    if workspace_bounds is not None:
        low, high = workspace_bounds
        pos = np.clip(pos, low, high)

    axis = rng.normal(size=(3,)).astype(np.float32)
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-8:
        axis = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    else:
        axis = axis / axis_norm
    angle = np.deg2rad(float(rng.uniform(-rot_step_deg, rot_step_deg)))
    delta_rot = Rotation.from_rotvec(axis * angle)
    cur_rot = Rotation.from_quat(quat)
    new_quat = (delta_rot * cur_rot).as_quat().astype(np.float32)
    new_quat = new_quat / max(np.linalg.norm(new_quat), 1e-8)

    if rng.random() < gripper_flip_prob:
        gripper_cmd = 0.0 if float(obs.gripper_open) > 0.5 else 1.0
    else:
        gripper_cmd = 1.0 if float(obs.gripper_open) > 0.5 else 0.0

    return np.concatenate(
        [pos.astype(np.float32), new_quat.astype(np.float32), np.asarray([gripper_cmd], dtype=np.float32)],
        axis=0,
    )


def _to_uint8_rgb(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == np.uint8:
        return arr
    arr = np.asarray(arr)
    if arr.size > 0 and np.nanmax(arr) <= 1.0 + 1e-6:
        arr = arr * 255.0
    return np.clip(arr, 0, 255).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Random-action rollout on the same scene as a trained POGS model, with tracking diagnostics."
    )
    parser.add_argument("--task", default="stack_blocks")
    parser.add_argument("--raw-root", required=True, help="RLBench raw split root, e.g. .../data/rlbench/raw/train")
    parser.add_argument("--pogs-config", required=True, help="Path to trained POGS config.yml")
    parser.add_argument("--episode", type=int, required=True, help="Episode id used for both scene reset and rollout")
    parser.add_argument("--camera", default="front")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--first-niters", type=int, default=15)
    parser.add_argument("--niters", type=int, default=5)
    parser.add_argument(
        "--track-use-depth",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable depth loss during tracking optimization.",
    )
    parser.add_argument(
        "--track-use-rgb",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable RGB loss during tracking optimization.",
    )
    parser.add_argument("--pos-step", type=float, default=0.015, help="Max random cartesian delta (m) per step")
    parser.add_argument("--rot-step-deg", type=float, default=8.0, help="Max random EE rotation delta (deg) per step")
    parser.add_argument("--gripper-flip-prob", type=float, default=0.15, help="Probability of toggling gripper open/close")
    parser.add_argument("--action-retries", type=int, default=20)
    parser.add_argument("--verify-every", type=int, default=10)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--hide-robot-from-camera",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Hide Panda arm and gripper meshes from rendered observations to match POGS capture conditions.",
    )
    parser.add_argument(
        "--save-frames",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save rollout RGB/depth frames under out-dir for later inspection.",
    )
    parser.add_argument("--out-dir", type=str, default="outputs/random_scene_rollouts")
    parser.add_argument(
        "--strict-scene-episode",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail if pogs-config appears to be from a different episode.",
    )
    args = parser.parse_args()

    _set_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    raw_root = Path(args.raw_root)
    episodes_root = raw_root / args.task / "all_variations" / "episodes"
    episode_dir = episodes_root / f"episode{args.episode}"
    if not episode_dir.exists():
        raise FileNotFoundError(f"Episode directory not found: {episode_dir}")

    config_path = Path(args.pogs_config)
    if not config_path.exists():
        raise FileNotFoundError(f"POGS config not found: {config_path}")

    inferred_config_episode = _infer_episode_from_pogs_config(config_path)
    if inferred_config_episode is not None and inferred_config_episode != int(args.episode):
        msg = (
            "POGS config appears tied to a different episode: "
            f"config_episode={inferred_config_episode}, requested_episode={args.episode}"
        )
        if args.strict_scene_episode:
            raise RuntimeError(msg)
        print(f"[WARN] {msg}")

    var_path = episode_dir / "variation_number.pkl"
    if not var_path.exists():
        raise FileNotFoundError(f"Variation file not found: {var_path}")
    with var_path.open("rb") as f:
        variation_id = pickle.load(f)

    run_tag = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.out_dir) / args.task / f"ep_{args.episode}_seed_{args.seed}_{run_tag}"
    rgb_dir = run_dir / f"{args.camera}_rgb"
    tracked_rgb_dir = run_dir / "tracked_rgb"
    depth_dir = run_dir / f"{args.camera}_depth"
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.save_frames:
        rgb_dir.mkdir(parents=True, exist_ok=True)
        tracked_rgb_dir.mkdir(parents=True, exist_ok=True)
        depth_dir.mkdir(parents=True, exist_ok=True)

    from rlbench.action_modes.action_mode import MoveArmThenGripper
    from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning
    from rlbench.action_modes.gripper_action_modes import Discrete
    from rlbench.environment import Environment
    from rlbench.observation_config import ObservationConfig

    obs_config = ObservationConfig()
    obs_config.set_all(False)
    camera_obs_config = getattr(obs_config, f"{args.camera}_camera")
    camera_obs_config.set_all(True)
    camera_obs_config.depth_in_meters = True
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

    print("[INFO] Launching RLBench environment...")
    env.launch()

    if args.hide_robot_from_camera:
        try:
            from pyrep.robots.arms.panda import Panda
            from pyrep.robots.end_effectors.panda_gripper import PandaGripper

            robot = Panda()
            gripper = PandaGripper()
            for obj in robot.get_objects_in_tree(exclude_base=False):
                obj.set_renderable(False)
            for obj in gripper.get_objects_in_tree(exclude_base=False):
                obj.set_renderable(False)
            print("[INFO] Robot renderables hidden for rollout observations.")
        except Exception as exc:
            print(f"[WARN] Failed to hide robot renderables: {exc}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    rollout_rows: list[dict[str, object]] = []
    termination_reason = "max_steps"
    workspace_bounds = _load_workspace_bounds(args.task)
    if workspace_bounds is not None:
        print(f"[INFO] Using workspace clamp for task={args.task}")
    else:
        print(f"[WARN] No workspace bounds found for task={args.task}; actions are unconstrained.")

    try:
        task_cls = task_file_to_task_class(args.task)
        task = env.get_task(task_cls)

        if hasattr(task._scene.task, "validate"):
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

        K = np.asarray(obs.misc[f"{args.camera}_camera_intrinsics"], dtype=np.float32)
        K = K.copy()
        K[0, 0] = abs(float(K[0, 0]))
        K[1, 1] = abs(float(K[1, 1]))
        extr = np.asarray(obs.misc[f"{args.camera}_camera_extrinsics"], dtype=np.float32)
        if extr.shape == (4, 4):
            init_cam_pose = torch.from_numpy(extr[:3, :]).float().unsqueeze(0)
        elif extr.shape == (3, 4):
            init_cam_pose = torch.from_numpy(extr).float().unsqueeze(0)
        else:
            raise RuntimeError(f"Unexpected extrinsics shape: {extr.shape}")

        h, w = getattr(obs, f"{args.camera}_rgb").shape[:2]
        optimizer = Optimizer(config_path, K, w, h, init_cam_pose)
        target_h, target_w = _init_optimizer_for_episode(optimizer, obs, args.camera, device)

        verify_every = max(1, int(args.verify_every))
        print(
            "[INFO] Starting random rollout: "
            f"task={args.task} episode={args.episode} seed={args.seed} steps={args.steps}"
        )

        for step_idx in range(int(args.steps)):
            rgb_tensor, depth_tensor = _obs_to_tensors(obs, args.camera, device, (target_h, target_w))
            optimizer.set_observation(rgb_tensor, optimizer.cam2world_ns_ds, depth_tensor)

            loss_before = _tracking_loss_proxy(
                optimizer,
                use_depth=bool(args.track_use_depth),
                use_rgb=bool(args.track_use_rgb),
            )

            step_render_dict = optimizer.step_opt(
                niter=args.first_niters if step_idx == 0 else args.niters,
                use_depth=bool(args.track_use_depth),
                use_rgb=bool(args.track_use_rgb),
            )

            step_metrics: dict[str, float] = {}
            if isinstance(step_render_dict, dict):
                maybe_metrics = step_render_dict.get("metrics", {})
                if isinstance(maybe_metrics, dict):
                    for key, value in maybe_metrics.items():
                        try:
                            step_metrics[str(key)] = float(value)
                        except (TypeError, ValueError):
                            continue

            loss_after = _tracking_loss_proxy(
                optimizer,
                use_depth=bool(args.track_use_depth),
                use_rgb=bool(args.track_use_rgb),
            )

            rgb_frame = getattr(obs, f"{args.camera}_rgb")
            depth_frame = getattr(obs, f"{args.camera}_depth")
            if args.save_frames:
                Image.fromarray(_to_uint8_rgb(np.asarray(rgb_frame))).save(rgb_dir / f"{step_idx:04d}.png")
                np.save(depth_dir / f"{step_idx:04d}.npy", np.asarray(depth_frame, dtype=np.float32))
                if isinstance(step_render_dict, dict) and "rgb" in step_render_dict:
                    render_rgb = step_render_dict["rgb"]
                    if isinstance(render_rgb, torch.Tensor):
                        render_rgb = render_rgb.detach().cpu().numpy()
                    render_rgb = np.asarray(render_rgb)
                    if render_rgb.ndim == 4 and render_rgb.shape[0] == 1:
                        render_rgb = render_rgb[0]
                    Image.fromarray(_to_uint8_rgb(render_rgb)).save(tracked_rgb_dir / f"{step_idx:04d}.png")

            row: dict[str, object] = {
                "step": int(step_idx),
                "dino_loss_before": float(loss_before),
                "dino_loss_after": float(loss_after),
                "dino_loss_drop": float(loss_before - loss_after)
                if np.isfinite(loss_before) and np.isfinite(loss_after)
                else float("nan"),
                "opt_total_loss_last": float(step_metrics.get("total_loss_last", float("nan"))),
                "opt_dino_loss_last": float(step_metrics.get("dino_loss_last", float("nan"))),
                "opt_depth_loss_last": float(step_metrics.get("depth_loss_last", float("nan"))),
                "reward": float("nan"),
                "terminate": False,
                "action_retry": -1,
            }

            if step_idx == int(args.steps) - 1:
                rollout_rows.append(row)
                break

            stepped = False
            last_err: Exception | None = None
            chosen_action: np.ndarray | None = None
            chosen_retry = -1
            reward = float("nan")
            terminate = False

            for retry_idx in range(max(1, int(args.action_retries))):
                action = _sample_random_action(
                    obs=obs,
                    rng=rng,
                    pos_step=float(args.pos_step),
                    rot_step_deg=float(args.rot_step_deg),
                    gripper_flip_prob=float(args.gripper_flip_prob),
                    workspace_bounds=workspace_bounds,
                )
                try:
                    obs, reward, terminate = task.step(action)
                    chosen_action = action
                    chosen_retry = int(retry_idx)
                    stepped = True
                    break
                except (IKError, ConfigurationPathError, InvalidActionError, RuntimeError) as exc:
                    last_err = exc

            if not stepped:
                row["step_error"] = f"{type(last_err).__name__}: {last_err}"
                row["terminate"] = True
                termination_reason = "action_failure"
                rollout_rows.append(row)
                print(f"[WARN] Step {step_idx}: all random action retries failed. Stopping rollout.")
                break

            row["reward"] = float(reward)
            row["terminate"] = bool(terminate)
            row["action_retry"] = int(chosen_retry)
            if chosen_action is not None:
                row["action"] = [float(x) for x in chosen_action.tolist()]
            rollout_rows.append(row)

            if step_idx % verify_every == 0:
                print(
                    "[VERIFY] "
                    f"step={step_idx:03d} "
                    f"dino_before={float(loss_before):.6f} "
                    f"dino_after={float(loss_after):.6f} "
                    f"reward={float(reward):.3f} "
                    f"terminate={bool(terminate)}"
                )

            if terminate:
                termination_reason = "rlbench_terminate"
                print(f"[INFO] Rollout terminated by RLBench at step={step_idx}.")
                break

        finite_rows = [
            r
            for r in rollout_rows
            if np.isfinite(float(r.get("dino_loss_before", float("nan"))))
            and np.isfinite(float(r.get("dino_loss_after", float("nan"))))
        ]
        improved = [
            float(r["dino_loss_after"]) < float(r["dino_loss_before"])
            for r in finite_rows
        ]
        improve_rate = float(np.mean(improved)) if improved else 0.0

        summary = {
            "task": args.task,
            "episode": int(args.episode),
            "seed": int(args.seed),
            "camera": args.camera,
            "raw_root": str(raw_root),
            "episodes_root": str(episodes_root),
            "variation_id": int(variation_id),
            "pogs_config": str(config_path),
            "config_episode_inferred": inferred_config_episode,
            "steps_requested": int(args.steps),
            "steps_recorded": int(len(rollout_rows)),
            "termination_reason": termination_reason,
            "track_use_depth": bool(args.track_use_depth),
            "track_use_rgb": bool(args.track_use_rgb),
            "first_niters": int(args.first_niters),
            "niters": int(args.niters),
            "tracking_2d_signal": {
                "checks": int(len(finite_rows)),
                "improve_rate": improve_rate,
            },
            "frames_dir": str(rgb_dir) if args.save_frames else "",
            "tracked_frames_dir": str(tracked_rgb_dir) if args.save_frames else "",
            "depth_dir": str(depth_dir) if args.save_frames else "",
            "rows": rollout_rows,
        }

        out_json = run_dir / "random_tracking_rollout.json"
        with out_json.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print(f"[INFO] Saved rollout report: {out_json}")
        if args.save_frames:
            print(f"[INFO] Saved RGB frames: {rgb_dir}")
            print(f"[INFO] Saved tracked renders: {tracked_rgb_dir}")
            print(f"[INFO] Saved depth frames: {depth_dir}")
    finally:
        env.shutdown()


if __name__ == "__main__":
    main()
