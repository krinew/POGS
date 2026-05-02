"""Render frozen POGS scene from an RLBench camera for calibration debugging.

This script intentionally does not run tracking, segmentation, or optimization.
It answers one narrow question:

    "Does the trained POGS scene project correctly into the RLBench front camera?"

It renders several camera-convention candidates so we can distinguish a tracking
failure from an extrinsics/sign-convention failure.
"""

from __future__ import annotations

import argparse
import importlib
import json
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str((_SCRIPT_DIR / "../..").resolve()))

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.utils.eval_utils import eval_setup

from pogs.pogs_pipeline import POGSPipeline
from pogs.tracking.optim import _patch_nerfstudio_pillow_compat


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _task_file_to_task_class(task_file: str):
    name = task_file.replace(".py", "")
    class_name = "".join([w[0].upper() + w[1:] for w in name.split("_")])
    mod = importlib.import_module(f"rlbench.tasks.{name}")
    mod = importlib.reload(mod)
    return getattr(mod, class_name)


def _to_u8_rgb(arr: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()
    arr = np.asarray(arr)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[-1] == 4:
        arr = arr[..., :3]
    arr = arr.astype(np.float32)
    if arr.size > 0 and float(np.nanmax(arr)) <= 1.0 + 1e-6:
        arr = arr * 255.0
    return np.clip(arr, 0, 255).astype(np.uint8)


def _save_depth_vis(path: Path, depth: np.ndarray) -> None:
    depth = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0)
    if not bool(valid.any()):
        Image.fromarray(np.zeros(depth.shape, dtype=np.uint8)).save(path)
        return
    lo, hi = np.nanpercentile(depth[valid], [2, 98])
    if hi <= lo:
        hi = lo + 1e-6
    vis = np.clip((depth - lo) / (hi - lo), 0, 1)
    vis[~valid] = 0
    Image.fromarray((vis * 255).astype(np.uint8)).save(path)


def _make_camera(
    extr_cv: np.ndarray,
    K: np.ndarray,
    width: int,
    height: int,
    dataset_scale: float,
    flip: np.ndarray,
) -> Cameras:
    if extr_cv.shape == (3, 4):
        extr4 = np.eye(4, dtype=np.float32)
        extr4[:3, :] = extr_cv
    elif extr_cv.shape == (4, 4):
        extr4 = extr_cv.astype(np.float32)
    else:
        raise ValueError(f"Unexpected extrinsics shape: {extr_cv.shape}")

    c2w = extr4 @ flip.astype(np.float32)
    c2w = c2w[None, :3, :]
    c2w[:, :3, 3] *= float(dataset_scale)
    return Cameras(
        camera_to_worlds=torch.from_numpy(c2w).float(),
        fx=float(K[0, 0]),
        fy=float(K[1, 1]),
        cx=float(K[0, 2]),
        cy=float(K[1, 2]),
        width=int(width),
        height=int(height),
    )


def _render_rgb(model, camera: Cameras, tracking: bool) -> np.ndarray:
    with torch.no_grad():
        model.eval()
        outputs = model.get_outputs(
            camera.to("cuda"),
            tracking=bool(tracking),
            rgb_only=True,
            BLOCK_WIDTH=16,
        )
    return _to_u8_rgb(outputs["rgb"])


def _contact_sheet(panels: list[tuple[str, Path]], out_path: Path, thumb_size: tuple[int, int] = (260, 260)) -> None:
    pad = 10
    label_h = 30
    w = len(panels) * (thumb_size[0] + pad) + pad
    h = thumb_size[1] + label_h + 2 * pad
    canvas = Image.new("RGB", (w, h), (20, 20, 20))
    draw = ImageDraw.Draw(canvas)
    for idx, (label, path) in enumerate(panels):
        x = pad + idx * (thumb_size[0] + pad)
        draw.text((x, pad), label, fill=(240, 240, 240))
        if path.exists():
            image = Image.open(path).convert("RGB").resize(thumb_size, Image.NEAREST)
        else:
            image = Image.new("RGB", thumb_size, (80, 0, 0))
        canvas.paste(image, (x, pad + label_h))
    canvas.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Check POGS frozen scene projection from an RLBench camera.")
    parser.add_argument("--task", default="stack_blocks")
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--pogs-config", required=True)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--camera", default="front")
    parser.add_argument("--out-dir", default="outputs/camera_projection_checks")
    parser.add_argument("--render-size", type=int, default=500)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    _set_seed(args.seed)
    _patch_nerfstudio_pillow_compat()

    raw_root = Path(args.raw_root)
    config_path = Path(args.pogs_config)
    run_tag = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) / args.task / f"ep_{args.episode}_seed_{args.seed}_{run_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    from rlbench.action_modes.action_mode import MoveArmThenGripper
    from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning
    from rlbench.action_modes.gripper_action_modes import Discrete
    from rlbench.environment import Environment
    from rlbench.observation_config import ObservationConfig

    obs_config = ObservationConfig()
    obs_config.set_all(False)
    cam_cfg = getattr(obs_config, f"{args.camera}_camera")
    cam_cfg.set_all(True)
    cam_cfg.depth_in_meters = True
    obs_config.gripper_pose = True
    obs_config.gripper_open = True

    env = Environment(
        action_mode=MoveArmThenGripper(
            arm_action_mode=EndEffectorPoseViaPlanning(collision_checking=False),
            gripper_action_mode=Discrete(),
        ),
        dataset_root=str(raw_root),
        obs_config=obs_config,
        headless=bool(args.headless),
    )

    env.launch()
    try:
        episode_dir = raw_root / args.task / "all_variations" / "episodes" / f"episode{args.episode}"
        with (episode_dir / "variation_number.pkl").open("rb") as f:
            variation_id = pickle.load(f)

        task = env.get_task(_task_file_to_task_class(args.task))
        if hasattr(task._scene.task, "validate"):
            task._scene.task.validate = lambda: None
        task.set_variation(-1)
        demo = task.get_demos(1, random_selection=False, live_demos=False, from_episode_number=args.episode)[0]
        task.set_variation(variation_id)
        demo._observations[0].misc["variation_index"] = variation_id
        _, obs = task.reset_to_demo(demo)

        rgb = _to_u8_rgb(getattr(obs, f"{args.camera}_rgb"))
        depth = np.asarray(getattr(obs, f"{args.camera}_depth"), dtype=np.float32)
        K_raw = np.asarray(obs.misc[f"{args.camera}_camera_intrinsics"], dtype=np.float32)
        K = K_raw.copy()
        K[0, 0] = abs(float(K[0, 0]))
        K[1, 1] = abs(float(K[1, 1]))
        extr = np.asarray(obs.misc[f"{args.camera}_camera_extrinsics"], dtype=np.float32)

        Image.fromarray(rgb).save(out_dir / f"{args.camera}_rgb_raw.png")
        Image.fromarray(rgb).resize((args.render_size, args.render_size), Image.NEAREST).save(
            out_dir / f"{args.camera}_rgb_render_size.png"
        )
        _save_depth_vis(out_dir / f"{args.camera}_depth_vis.png", depth)

        train_config, pipeline, _, _ = eval_setup(config_path)
        assert isinstance(pipeline, POGSPipeline)
        pipeline.model.eval()
        dataset_scale = pipeline.datamanager.train_dataset._dataparser_outputs.dataparser_scale

        flips = {
            "optimizer_current_xz": np.diag([-1.0, 1.0, -1.0, 1.0]).astype(np.float32),
            "yz_rx_pi": np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32),
            "xy_roll_pi": np.diag([-1.0, -1.0, 1.0, 1.0]).astype(np.float32),
            "none": np.eye(4, dtype=np.float32),
        }

        panels: list[tuple[str, Path]] = [
            ("front rgb", out_dir / f"{args.camera}_rgb_render_size.png"),
        ]
        render_info = {}
        h, w = rgb.shape[:2]
        for name, flip in flips.items():
            cam_raw = _make_camera(extr, K, w, h, dataset_scale, flip)
            raw_render = _render_rgb(pipeline.model, cam_raw, tracking=False)
            raw_path = out_dir / f"render_{name}_rawres.png"
            Image.fromarray(raw_render).save(raw_path)

            cam_render = _make_camera(extr, K, w, h, dataset_scale, flip)
            cam_render.rescale_output_resolution(float(args.render_size) / float(min(w, h)))
            render = _render_rgb(pipeline.model, cam_render, tracking=False)
            render_path = out_dir / f"render_{name}_{args.render_size}.png"
            Image.fromarray(render).save(render_path)
            panels.append((name, render_path))
            render_info[name] = {
                "raw_render": str(raw_path),
                "render_size": str(render_path),
                "flip": flip.tolist(),
            }

        _contact_sheet(panels, out_dir / "projection_contact_sheet.png")

        summary = {
            "task": args.task,
            "episode": int(args.episode),
            "seed": int(args.seed),
            "camera": args.camera,
            "raw_root": str(raw_root),
            "pogs_config": str(config_path),
            "variation_id": int(variation_id),
            "dataset_scale": float(dataset_scale),
            "raw_image_shape": list(rgb.shape),
            "K_raw": K_raw.tolist(),
            "K_abs": K.tolist(),
            "extrinsics": extr.tolist(),
            "render_info": render_info,
            "contact_sheet": str(out_dir / "projection_contact_sheet.png"),
        }
        with (out_dir / "projection_summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print(f"[INFO] Saved projection check: {out_dir}")
        print(f"[INFO] Contact sheet: {out_dir / 'projection_contact_sheet.png'}")
    finally:
        env.shutdown()


if __name__ == "__main__":
    main()
