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
import warp as wp
from PIL import Image
from pyrep.errors import ConfigurationPathError, IKError
from rlbench.backend.exceptions import InvalidActionError
from scipy.spatial.transform import Rotation

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(_SCRIPT_DIR, "../../")))

from pogs.tracking.optim import Optimizer
from pogs.tracking.segmentation import (
    GroundingDinoBoxer,
    GroundingDinoConfig,
    Sam2Segmenter,
    Sam2Config,
)
from pogs.tracking.utils2 import overlay


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


def _init_optimizer_for_episode(
    optimizer: Optimizer,
    first_obs,
    camera: str,
    device: str,
    skip_init_pose_opt: bool = False,
) -> tuple[int, int]:
    optimizer.reset_optimizer()
    cam = optimizer.cam2world_ns_ds
    tgt_h = int(cam.height.item() if isinstance(cam.height, torch.Tensor) else cam.height)
    tgt_w = int(cam.width.item() if isinstance(cam.width, torch.Tensor) else cam.width)
    rgb, depth = _obs_to_tensors(first_obs, camera, device, (tgt_h, tgt_w))
    optimizer.set_frame(rgb, optimizer.cam2world_ns_ds, depth)
    if skip_init_pose_opt:
        optimizer.initialized = True
        optimizer.optimizer.is_initialized = True
        print("[INFO] Skipping initial pose optimization; using raw clustered Gaussian priors.")
    else:
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


def _pose_step_metrics(before: torch.Tensor, after: torch.Tensor) -> dict[str, float]:
    """Summarize per-group pose changes made by one optimizer step."""
    before_cpu = before.detach().cpu()
    after_cpu = after.detach().cpu()
    metrics: dict[str, float] = {}
    n = min(int(before_cpu.shape[0]), int(after_cpu.shape[0]))
    for group_idx in range(n):
        trans_before = before_cpu[group_idx, :3]
        trans_after = after_cpu[group_idx, :3]
        trans_delta = torch.linalg.norm(trans_after - trans_before).item()
        trans_norm = torch.linalg.norm(trans_after).item()

        q_before = before_cpu[group_idx, 3:]
        q_after = after_cpu[group_idx, 3:]
        q_before = q_before / torch.clamp(torch.linalg.norm(q_before), min=1e-8)
        q_after = q_after / torch.clamp(torch.linalg.norm(q_after), min=1e-8)
        dot = torch.clamp(torch.abs(torch.dot(q_before, q_after)), 0.0, 1.0)
        rot_delta_rad = 2.0 * torch.acos(dot).item()

        prefix = f"group_{group_idx:02d}_"
        metrics[prefix + "step_trans_delta_m"] = float(trans_delta)
        metrics[prefix + "trans_norm_m"] = float(trans_norm)
        metrics[prefix + "step_rot_delta_rad"] = float(rot_delta_rad)
    return metrics


def _restore_group_poses(
    optimizer: Optimizer,
    source_part_deltas: torch.Tensor,
    group_indices: list[int],
) -> None:
    """Restore selected groups and clear Adam momentum for those rows."""
    if not group_indices:
        return
    with torch.no_grad():
        current = optimizer.optimizer.part_deltas
        source = source_part_deltas.to(current.device)
        for group_idx in group_indices:
            if 0 <= group_idx < current.shape[0] and group_idx < source.shape[0]:
                current.data[group_idx].copy_(source[group_idx])

        state = optimizer.optimizer.part_optimizer.state.get(current, {})
        for state_name in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            state_value = state.get(state_name)
            if isinstance(state_value, torch.Tensor) and state_value.shape[:1] == current.shape[:1]:
                for group_idx in group_indices:
                    if 0 <= group_idx < state_value.shape[0]:
                        state_value[group_idx].zero_()


def _unreliable_groups_from_assignments(assignment_records: list[dict[str, object]]) -> list[int]:
    groups: list[int] = []
    for rec in assignment_records:
        if bool(rec.get("ignored", False)):
            try:
                groups.append(int(rec["group"]))
            except Exception:
                continue
    return groups


def _update_last_reliable_part_deltas(
    last_reliable_part_deltas: torch.Tensor,
    current_part_deltas: torch.Tensor,
    assignment_records: list[dict[str, object]],
    min_target_render_ratio: float,
    min_iou: float,
) -> list[int]:
    """Update reliable pose memory from groups with strong current supervision."""
    updated: list[int] = []
    with torch.no_grad():
        current = current_part_deltas.detach().to(last_reliable_part_deltas.device)
        for rec in assignment_records:
            try:
                group_idx = int(rec["group"])
                ratio = float(rec.get("target_render_ratio", 0.0))
                iou = float(rec.get("iou", 0.0))
            except Exception:
                continue
            if bool(rec.get("ignored", False)):
                continue
            if ratio < float(min_target_render_ratio) or iou < float(min_iou):
                continue
            if 0 <= group_idx < last_reliable_part_deltas.shape[0] and group_idx < current.shape[0]:
                last_reliable_part_deltas[group_idx].copy_(current[group_idx])
                updated.append(group_idx)
    return updated


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


def _write_image_sequence_video(image_dir: Path, out_path: Path, fps: float) -> str:
    image_paths = sorted(image_dir.glob("*.png"))
    if not image_paths:
        return ""

    # Import lazily so normal rollout startup only depends on MoviePy when
    # video export is explicitly requested.
    import moviepy as mpy

    frames = [np.asarray(Image.open(path).convert("RGB")) for path in image_paths]
    clip = mpy.ImageSequenceClip(frames, fps=float(fps))
    clip.write_videofile(str(out_path), logger=None)
    return str(out_path)


def _save_rgb_resized(path: Path, rgb: np.ndarray, shape_hw: tuple[int, int]) -> None:
    rgb_u8 = _to_uint8_rgb(np.asarray(rgb))
    resized = Image.fromarray(rgb_u8).resize((int(shape_hw[1]), int(shape_hw[0])), resample=Image.NEAREST)
    resized.save(path)


def _render_current_tracking_rgb(optimizer: Optimizer) -> np.ndarray | None:
    """Render current tracked Gaussians before/after an optimization step."""
    with torch.no_grad(), optimizer.optimizer.render_lock:
        optimizer.optimizer.pogs_model.eval()
        optimizer.optimizer.apply_to_model(
            optimizer.optimizer.part_deltas,
            optimizer.group_labels,
        )
        outputs = optimizer.optimizer.pogs_model.get_outputs(
            optimizer.cam2world_ns_ds.to("cuda"),
            tracking=True,
            rgb_only=True,
        )
        rgb = outputs.get("rgb")
        if rgb is None:
            return None
        return _to_uint8_rgb(rgb.detach().cpu().numpy() if isinstance(rgb, torch.Tensor) else np.asarray(rgb))


def _resize_bool_mask(mask: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    tgt_h, tgt_w = int(shape_hw[0]), int(shape_hw[1])
    if mask.shape == (tgt_h, tgt_w):
        return mask
    resized = Image.fromarray(mask.astype(np.uint8) * 255).resize((tgt_w, tgt_h), resample=Image.NEAREST)
    return np.asarray(resized) > 0


def _collect_robot_handles() -> set[int]:
    from pyrep.robots.arms.panda import Panda
    from pyrep.robots.end_effectors.panda_gripper import PandaGripper

    robot = Panda()
    gripper = PandaGripper()
    handles: set[int] = set()
    for root in (robot, gripper):
        try:
            handles.add(int(root.get_handle()))
        except Exception:
            pass
        for obj in root.get_objects_in_tree(exclude_base=False):
            try:
                handles.add(int(obj.get_handle()))
            except Exception:
                continue
    return handles


def _set_robot_renderable(renderable: bool) -> None:
    from pyrep.robots.arms.panda import Panda
    from pyrep.robots.end_effectors.panda_gripper import PandaGripper

    robot = Panda()
    gripper = PandaGripper()
    for root in (robot, gripper):
        for obj in root.get_objects_in_tree(exclude_base=False):
            obj.set_renderable(bool(renderable))


def _robot_exclusion_mask_from_obs(obs, camera: str, robot_handles: set[int], shape_hw: tuple[int, int]) -> np.ndarray | None:
    if not robot_handles:
        return None
    sim_mask = getattr(obs, f"{camera}_mask", None)
    if sim_mask is None:
        return None

    sim_mask = np.asarray(sim_mask)
    if sim_mask.ndim == 3:
        # Defensive fallback if a camera config ever emits RGB-coded handles.
        sim_mask = (
            sim_mask[:, :, 0].astype(np.int64)
            + sim_mask[:, :, 1].astype(np.int64) * 256
            + sim_mask[:, :, 2].astype(np.int64) * 256 * 256
        )
    robot_mask = np.isin(sim_mask.astype(np.int64), np.asarray(sorted(robot_handles), dtype=np.int64))
    return _resize_bool_mask(robot_mask, shape_hw)


def _filter_instances_with_exclusion(
    instance_masks: list[np.ndarray],
    boxes: list[list[float]],
    exclusion_mask: np.ndarray | None,
    max_overlap: float,
    min_area_px: int,
) -> tuple[list[np.ndarray], list[list[float]], dict[str, object]]:
    if exclusion_mask is None:
        return instance_masks, boxes, {
            "enabled": False,
            "dropped_instances": 0,
            "kept_instances": len(instance_masks),
        }

    kept_masks: list[np.ndarray] = []
    kept_boxes: list[list[float]] = []
    dropped = 0
    overlap_fracs: list[float] = []
    exclusion = np.asarray(exclusion_mask, dtype=bool)

    for idx, mask in enumerate(instance_masks):
        inst = np.asarray(mask, dtype=bool)
        if inst.shape != exclusion.shape:
            exclusion_for_inst = _resize_bool_mask(exclusion, inst.shape)
        else:
            exclusion_for_inst = exclusion

        inst_area = int(inst.sum())
        if inst_area <= 0:
            dropped += 1
            continue

        overlap = int(np.logical_and(inst, exclusion_for_inst).sum())
        overlap_frac = float(overlap / max(inst_area, 1))
        overlap_fracs.append(overlap_frac)
        inst_without_robot = np.logical_and(inst, np.logical_not(exclusion_for_inst))

        if overlap_frac > float(max_overlap) or int(inst_without_robot.sum()) < int(min_area_px):
            dropped += 1
            continue

        kept_masks.append(inst_without_robot)
        if idx < len(boxes):
            kept_boxes.append(boxes[idx])

    return kept_masks, kept_boxes, {
        "enabled": True,
        "dropped_instances": int(dropped),
        "kept_instances": int(len(kept_masks)),
        "robot_overlap_fracs": overlap_fracs,
    }


def _box_iou(box_a: list[float], box_b: list[float]) -> float:
    ax0, ay0, ax1, ay1 = box_a
    bx0, by0, bx1, by1 = box_b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    denom = area_a + area_b - inter
    return 0.0 if denom <= 0 else inter / denom


def _dedupe_boxes(boxes: list[list[float]], iou_threshold: float) -> list[list[float]]:
    kept: list[list[float]] = []
    for box in boxes:
        if all(_box_iou(box, prev) < iou_threshold for prev in kept):
            kept.append(box)
    return kept


def _parse_prompts(prompt: str, extra_prompts: list[str]) -> list[str]:
    prompts: list[str] = []
    for item in [prompt, *extra_prompts]:
        prompts.extend([p.strip() for p in item.split("|") if p.strip()])
    return prompts


def _mask_centroid(mask: np.ndarray) -> tuple[float, float] | None:
    ys, xs = np.where(mask.astype(bool))
    if len(xs) == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def _mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return 0.0 if union == 0 else float(inter / union)


def _depth_to_numpy(depth_tensor: torch.Tensor | np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    depth = depth_tensor.detach().cpu().numpy() if isinstance(depth_tensor, torch.Tensor) else np.asarray(depth_tensor)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    elif depth.ndim == 3 and depth.shape[0] == 1:
        depth = depth[0]
    depth = depth.astype(np.float32)
    if depth.shape != shape_hw:
        resized = Image.fromarray(depth).resize((shape_hw[1], shape_hw[0]), resample=Image.BILINEAR)
        depth = np.asarray(resized, dtype=np.float32)
    return depth


def _save_depth_debug(path: Path, depth: np.ndarray, valid_mask: np.ndarray | None = None) -> None:
    depth = np.asarray(depth, dtype=np.float32)
    if valid_mask is None:
        valid = np.isfinite(depth) & (depth > 0)
    else:
        valid = np.asarray(valid_mask, dtype=bool) & np.isfinite(depth) & (depth > 0)
    if not bool(valid.any()):
        Image.fromarray(np.zeros(depth.shape, dtype=np.uint8)).save(path)
        return
    lo, hi = np.nanpercentile(depth[valid], [2, 98])
    if hi <= lo:
        hi = lo + 1e-6
    vis = np.clip((depth - lo) / (hi - lo), 0.0, 1.0)
    vis[~valid] = 0.0
    Image.fromarray((vis * 255.0).astype(np.uint8)).save(path)


def _render_group_observables(
    optimizer: Optimizer,
    observed_depth_m: np.ndarray,
) -> tuple[list[np.ndarray], list[np.ndarray], list[dict[str, float]]]:
    """Render current group masks/depths for assignment and debug.

    Depth returned by the POGS renderer is in dataset-scaled units; convert it to
    metric depth to compare directly against RLBench / real depth images.
    """
    cam = optimizer.cam2world_ns_ds.to("cuda")
    num_groups = int(optimizer.num_groups)
    group_masks: list[np.ndarray] = []
    group_depths_m: list[np.ndarray] = []
    stats: list[dict[str, float]] = []
    dataset_scale = float(getattr(optimizer.optimizer, "dataset_scale", 1.0))

    with torch.no_grad(), optimizer.optimizer.render_lock:
        optimizer.optimizer.pogs_model.eval()
        optimizer.optimizer.apply_to_model(
            optimizer.optimizer.part_deltas,
            optimizer.group_labels,
        )
        for group_idx in range(num_groups):
            outputs = optimizer.optimizer.pogs_model.get_outputs(
                cam,
                tracking=True,
                obj_id=group_idx,
                BLOCK_WIDTH=8,
                rgb_only=False,
            )
            accum = outputs["accumulation"].squeeze(-1).detach().cpu().numpy()
            depth = outputs["depth"].squeeze(-1).detach().cpu().numpy().astype(np.float32) / max(dataset_scale, 1e-8)
            mask = accum > 0.9
            if mask.shape != observed_depth_m.shape:
                mask = _resize_bool_mask(mask, observed_depth_m.shape)
                depth = np.asarray(
                    Image.fromarray(depth).resize(
                        (observed_depth_m.shape[1], observed_depth_m.shape[0]),
                        resample=Image.BILINEAR,
                    ),
                    dtype=np.float32,
                )
            group_masks.append(mask.astype(bool))
            group_depths_m.append(depth)

            depth_valid = mask & np.isfinite(depth) & (depth > 0)
            obs_valid = mask & np.isfinite(observed_depth_m) & (observed_depth_m > 0)
            if bool(depth_valid.any()) and bool(obs_valid.any()):
                depth_err = np.abs(depth[mask] - observed_depth_m[mask])
                depth_err = depth_err[np.isfinite(depth_err)]
                median_depth_err = float(np.median(depth_err)) if depth_err.size else float("nan")
            else:
                median_depth_err = float("nan")
            stats.append(
                {
                    "group": float(group_idx),
                    "render_area": float(mask.sum()),
                    "render_depth_median": float(np.median(depth[depth_valid])) if bool(depth_valid.any()) else float("nan"),
                    "obs_depth_median": float(np.median(observed_depth_m[obs_valid])) if bool(obs_valid.any()) else float("nan"),
                    "median_abs_depth_err": median_depth_err,
                }
            )

    return group_masks, group_depths_m, stats


def _visible_gate_from_depth(
    observed_depth_m: np.ndarray,
    rendered_depth_m: np.ndarray,
    rendered_mask: np.ndarray,
    occlusion_margin_m: float,
) -> np.ndarray:
    observed = np.asarray(observed_depth_m, dtype=np.float32)
    rendered = np.asarray(rendered_depth_m, dtype=np.float32)
    mask = np.asarray(rendered_mask, dtype=bool)
    valid = mask & np.isfinite(observed) & np.isfinite(rendered) & (observed > 0) & (rendered > 0)
    visible = valid & (observed >= (rendered - float(occlusion_margin_m)))
    return visible.astype(bool)


def _assign_instance_masks_to_groups(
    optimizer: Optimizer,
    instance_masks: list[np.ndarray],
    previous_target_masks: list[np.ndarray] | None,
    min_iou: float,
    group_masks: list[np.ndarray] | None = None,
    group_depths_m: list[np.ndarray] | None = None,
    observed_depth_m: np.ndarray | None = None,
    depth_sigma_m: float = 0.05,
    iou_weight: float = 1.0,
    depth_weight: float = 0.4,
    temporal_weight: float = 0.25,
    occlusion_margin_m: float = 0.03,
    min_depth_score: float = 0.15,
    min_depth_valid_px: int = 25,
    min_target_area_px: int = 25,
    min_target_render_ratio: float = 0.15,
    use_depth_score: bool = True,
    use_depth_gate: bool = True,
) -> tuple[list[np.ndarray], list[int | None], list[dict[str, object]]]:
    num_groups = optimizer.num_groups
    cam_h = optimizer.cam2world_ns_ds.height
    cam_w = optimizer.cam2world_ns_ds.width
    h = int(cam_h.item() if isinstance(cam_h, torch.Tensor) else cam_h)
    w = int(cam_w.item() if isinstance(cam_w, torch.Tensor) else cam_w)

    if group_masks is None:
        render_camera = optimizer.cam2world_ns_ds.to("cuda")
        group_masks = []
        for group_idx in range(num_groups):
            rendered = optimizer.optimizer.render_mask(render_camera, group_idx)
            group_masks.append(rendered.squeeze().detach().cpu().numpy().astype(bool))
    group_masks = [_resize_bool_mask(mask, (h, w)) for mask in group_masks]

    if group_depths_m is not None:
        group_depths_m = [_depth_to_numpy(depth, (h, w)) for depth in group_depths_m]
    if observed_depth_m is not None:
        observed_depth_m = _depth_to_numpy(observed_depth_m, (h, w))

    def _pair_depth_terms(group_idx: int, inst_mask: np.ndarray) -> tuple[float, float, int]:
        if (
            not use_depth_score
            or group_depths_m is None
            or observed_depth_m is None
            or group_idx >= len(group_depths_m)
        ):
            return 0.0, float("nan"), 0
        overlap = group_masks[group_idx] & np.asarray(inst_mask, dtype=bool)
        valid = (
            overlap
            & np.isfinite(observed_depth_m)
            & np.isfinite(group_depths_m[group_idx])
            & (observed_depth_m > 0)
            & (group_depths_m[group_idx] > 0)
        )
        if not bool(valid.any()):
            return 0.0, float("nan"), 0
        err = np.abs(group_depths_m[group_idx][valid] - observed_depth_m[valid])
        median_err = float(np.median(err))
        depth_score = float(np.exp(-median_err / max(float(depth_sigma_m), 1e-6)))
        return depth_score, median_err, int(valid.sum())

    ignore_masks = [np.full(mask.shape, -1.0, dtype=np.float32) for mask in group_masks]

    if len(instance_masks) == 0:
        records = [
            {
                "group": int(idx),
                "instance": None,
                "iou": 0.0,
                "score": 0.0,
                "depth_score": 0.0,
                "median_depth_err_m": float("nan"),
                "depth_valid_px": 0,
                "prev_mask_iou": 0.0,
                "render_area": int(group_masks[idx].sum()),
                "visible_area": 0,
                "target_area": 0,
                "ignored": True,
                "ignore_reason": "no_instances",
            }
            for idx in range(num_groups)
        ]
        return ignore_masks, [None] * num_groups, records

    score_details: dict[tuple[int, int], dict[str, float]] = {}
    scores: list[tuple[float, int, int]] = []
    for group_idx, group_mask in enumerate(group_masks):
        for inst_idx, inst_mask in enumerate(instance_masks):
            iou = _mask_iou(group_mask, inst_mask)
            depth_score, median_depth_err, depth_valid_px = _pair_depth_terms(group_idx, inst_mask)
            prev_mask_iou = 0.0
            if (
                previous_target_masks is not None
                and group_idx < len(previous_target_masks)
                and previous_target_masks[group_idx] is not None
            ):
                prev_mask = np.asarray(previous_target_masks[group_idx])
                prev_valid = prev_mask >= 0.0
                if bool(prev_valid.any()):
                    prev_mask_iou = _mask_iou(prev_mask > 0.5, inst_mask)
            score = float(iou_weight) * iou + float(depth_weight) * depth_score + float(temporal_weight) * prev_mask_iou
            score_details[(group_idx, inst_idx)] = {
                "score": float(score),
                "iou": float(iou),
                "depth_score": float(depth_score),
                "median_depth_err_m": float(median_depth_err),
                "depth_valid_px": float(depth_valid_px),
                "prev_mask_iou": float(prev_mask_iou),
            }
            scores.append((score, group_idx, inst_idx))

    assignments: list[int | None] = [None] * num_groups
    used_instances: set[int] = set()

    for score, group_idx, inst_idx in sorted(scores, reverse=True):
        detail = score_details.get((group_idx, inst_idx), {})
        iou_ok = float(detail.get("iou", 0.0)) >= min_iou
        temporal_ok = float(detail.get("prev_mask_iou", 0.0)) >= min_iou
        depth_ok = (
            (not use_depth_score)
            or (
                float(detail.get("depth_score", 0.0)) >= float(min_depth_score)
                and int(detail.get("depth_valid_px", 0)) >= int(min_depth_valid_px)
            )
        )
        if not (iou_ok or temporal_ok) or not depth_ok:
            continue
        if assignments[group_idx] is not None or inst_idx in used_instances:
            continue
        assignments[group_idx] = inst_idx
        used_instances.add(inst_idx)

    assigned_masks: list[np.ndarray] = []
    assignment_records: list[dict[str, object]] = []
    for group_idx, inst_idx in enumerate(assignments):
        ignored = False
        ignore_reason = ""
        if inst_idx is None:
            best_iou = 0.0
            detail = {
                "score": 0.0,
                "depth_score": 0.0,
                "median_depth_err_m": float("nan"),
                "depth_valid_px": 0.0,
                "prev_mask_iou": 0.0,
            }
            visible_area = 0
            target_area = 0
            target = ignore_masks[group_idx]
            ignored = True
            ignore_reason = "unmatched"
        else:
            target = instance_masks[inst_idx].astype(bool)
            best_iou = _mask_iou(group_masks[group_idx], instance_masks[inst_idx])
            detail = score_details.get((group_idx, inst_idx), {})
            if (
                use_depth_gate
                and group_depths_m is not None
                and observed_depth_m is not None
                and group_idx < len(group_depths_m)
            ):
                visible_gate = _visible_gate_from_depth(
                    observed_depth_m=observed_depth_m,
                    rendered_depth_m=group_depths_m[group_idx],
                    rendered_mask=group_masks[group_idx],
                    occlusion_margin_m=occlusion_margin_m,
                )
                target = target & visible_gate
                visible_area = int(visible_gate.sum())
            else:
                visible_gate = group_masks[group_idx]
                visible_area = int(visible_gate.sum())
            target_area = int(target.sum())
            render_area = int(group_masks[group_idx].sum())
            target_render_ratio = (
                float(target_area) / float(max(render_area, 1))
            )
            if target_area == 0:
                target = ignore_masks[group_idx]
                ignored = True
                ignore_reason = "depth_gated_empty"
            elif (
                target_area < int(min_target_area_px)
                or target_render_ratio < float(min_target_render_ratio)
            ):
                target = ignore_masks[group_idx]
                ignored = True
                ignore_reason = "target_too_small"
            else:
                target = target.astype(np.float32)
        render_area = int(group_masks[group_idx].sum())
        target_render_ratio = (
            float(target_area) / float(max(render_area, 1))
            if render_area > 0
            else 0.0
        )
        assigned_masks.append(target)
        assignment_records.append(
            {
                "group": int(group_idx),
                "instance": None if inst_idx is None else int(inst_idx),
                "iou": float(best_iou),
                "score": float(detail.get("score", 0.0)),
                "depth_score": float(detail.get("depth_score", 0.0)),
                "median_depth_err_m": float(detail.get("median_depth_err_m", float("nan"))),
                "depth_valid_px": int(detail.get("depth_valid_px", 0)),
                "prev_mask_iou": float(detail.get("prev_mask_iou", 0.0)),
                "render_area": int(group_masks[group_idx].sum()),
                "visible_area": int(visible_area),
                "target_area": int(target_area),
                "target_render_ratio": float(target_render_ratio),
                "ignored": bool(ignored),
                "ignore_reason": ignore_reason,
            }
        )

    return assigned_masks, assignments, assignment_records


class GdinoSam2Masker:
    """Online text-driven mask predictor: GroundingDINO boxes + SAM2 masks."""

    def __init__(
        self,
        prompt: str,
        extra_prompts: list[str],
        gdino_model_id: str,
        gdino_box_threshold: float,
        gdino_text_threshold: float,
        gdino_max_boxes: int,
        gdino_min_box_area: float | None,
        gdino_max_box_area: float | None,
        gdino_dedupe_iou: float,
        sam2_model_id: str,
        sam2_model_cfg: str | None,
        sam2_checkpoint: str | None,
    ) -> None:
        self.prompts = _parse_prompts(prompt, extra_prompts)
        self.dedupe_iou = float(gdino_dedupe_iou)
        self.boxer = GroundingDinoBoxer(
            GroundingDinoConfig(
                model_id=gdino_model_id,
                box_threshold=gdino_box_threshold,
                text_threshold=gdino_text_threshold,
                max_boxes=gdino_max_boxes,
                min_box_area=gdino_min_box_area,
                max_box_area=gdino_max_box_area,
            )
        )
        self.segmenter = Sam2Segmenter(
            Sam2Config(
                model_id=sam2_model_id,
                model_cfg=sam2_model_cfg,
                checkpoint=sam2_checkpoint,
            )
        )

    def predict(self, image: np.ndarray) -> tuple[np.ndarray, list[list[float]], list[np.ndarray]]:
        boxes: list[list[float]] = []
        for prompt in self.prompts:
            boxes.extend(self.boxer.predict_boxes(image, prompt))
        boxes = _dedupe_boxes(boxes, self.dedupe_iou)
        if len(boxes) == 0:
            mask = np.zeros(image.shape[:2], dtype=bool)
            instance_masks = []
        else:
            instance_masks = self.segmenter.predict_instance_masks(image, boxes)
            mask = np.logical_or.reduce(instance_masks) if instance_masks else np.zeros(image.shape[:2], dtype=bool)
        return mask.astype(bool), boxes, instance_masks


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
        "--skip-init-pose-opt",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Keep raw clustered Gaussian poses instead of running init_obj_pose(). "
            "Default: true for gdino-sam2 mask/depth tracking, false otherwise."
        ),
    )
    parser.add_argument(
        "--seg-mode",
        choices=["none", "gdino-sam2"],
        default="gdino-sam2",
        help="Silhouette source for tracking. Use gdino-sam2 for text-driven masks.",
    )
    parser.add_argument("--seg-prompt", type=str, default="stacked blocks. blocks. cubes.")
    parser.add_argument(
        "--seg-extra-prompt",
        "--extra-prompt",
        action="append",
        default=[],
        help="Additional segmentation prompt. You can also separate prompts with '|'.",
    )
    parser.add_argument("--gdino-model-id", type=str, default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument("--gdino-box-threshold", type=float, default=0.2)
    parser.add_argument("--gdino-text-threshold", type=float, default=0.2)
    parser.add_argument("--gdino-max-boxes", type=int, default=10)
    parser.add_argument("--gdino-min-box-area", type=float, default=None)
    parser.add_argument("--gdino-max-box-area", type=float, default=0.25)
    parser.add_argument("--gdino-dedupe-iou", type=float, default=0.92)
    parser.add_argument("--seg-assignment-min-iou", type=float, default=0.01)
    parser.add_argument("--sam2-model-id", type=str, default="facebook/sam2-hiera-large")
    parser.add_argument("--sam2-model-cfg", type=str, default=None)
    parser.add_argument("--sam2-checkpoint", type=str, default=None)
    parser.add_argument(
        "--seg-exclude-robot-mask",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="DEBUG ONLY: use RLBench object-ID masks to remove Panda/gripper pixels. Off by default because it is not deployable.",
    )
    parser.add_argument(
        "--seg-robot-overlap-drop-threshold",
        type=float,
        default=0.35,
        help="Drop a detected instance if this fraction of its pixels overlap the robot object-ID mask.",
    )
    parser.add_argument(
        "--seg-min-instance-area-px",
        type=int,
        default=16,
        help="Drop segmentation instances smaller than this after robot-mask subtraction.",
    )
    parser.add_argument(
        "--seg-depth-aware-assignment",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Assign SAM2 instances to Gaussian groups using rendered-mask overlap plus depth consistency.",
    )
    parser.add_argument(
        "--seg-depth-gate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Remove likely occluded pixels from per-group mask targets using observed-vs-rendered depth.",
    )
    parser.add_argument(
        "--seg-depth-sigma-m",
        type=float,
        default=0.05,
        help="Depth error scale in meters for assignment scoring.",
    )
    parser.add_argument(
        "--seg-occlusion-margin-m",
        type=float,
        default=0.03,
        help="Observed depth must be at least rendered_depth - this margin to supervise a group pixel.",
    )
    parser.add_argument(
        "--seg-min-depth-score",
        type=float,
        default=0.15,
        help="Minimum depth-consistency score required before an instance can supervise a group.",
    )
    parser.add_argument(
        "--seg-min-depth-valid-px",
        type=int,
        default=25,
        help="Minimum overlap pixels with valid depth required before an instance can supervise a group.",
    )
    parser.add_argument(
        "--seg-min-target-area-px",
        type=int,
        default=25,
        help="Ignore a group for the frame if the final depth-gated target has fewer pixels than this.",
    )
    parser.add_argument(
        "--seg-min-target-render-ratio",
        type=float,
        default=0.6,
        help="Ignore a group for the frame if final target area / rendered group area is below this.",
    )
    parser.add_argument(
        "--seg-rollback-unreliable-pose",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Restore ignored/unreliable groups to their last reliable pose before optimization.",
    )
    parser.add_argument(
        "--seg-reliable-target-render-ratio",
        type=float,
        default=0.75,
        help="Minimum target/render ratio required to update a group's last reliable pose memory.",
    )
    parser.add_argument(
        "--seg-reliable-iou",
        type=float,
        default=0.7,
        help="Minimum assignment IoU required to update a group's last reliable pose memory.",
    )
    parser.add_argument("--seg-iou-weight", type=float, default=1.0)
    parser.add_argument("--seg-depth-weight", type=float, default=0.4)
    parser.add_argument("--seg-temporal-weight", type=float, default=0.25)
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
    parser.add_argument(
        "--save-seg-debug",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save per-step binary masks, overlays, and detected boxes.",
    )
    parser.add_argument(
        "--save-videos",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write MP4 videos from saved RGB, tracked-render, and mask-overlay frame sequences.",
    )
    parser.add_argument("--video-fps", type=float, default=20.0, help="FPS for exported rollout videos.")
    parser.add_argument("--out-dir", type=str, default="outputs/random_scene_rollouts")
    parser.add_argument(
        "--strict-scene-episode",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail if pogs-config appears to be from a different episode.",
    )
    args = parser.parse_args()
    if args.skip_init_pose_opt is None:
        args.skip_init_pose_opt = args.seg_mode == "gdino-sam2"

    _set_seed(args.seed)
    wp.init()
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
    rgb_resized_dir = run_dir / f"{args.camera}_rgb_tracking_res"
    preopt_tracked_rgb_dir = run_dir / "preopt_tracked_rgb"
    tracked_rgb_dir = run_dir / "tracked_rgb"
    depth_dir = run_dir / f"{args.camera}_depth"
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.save_frames:
        rgb_dir.mkdir(parents=True, exist_ok=True)
        rgb_resized_dir.mkdir(parents=True, exist_ok=True)
        preopt_tracked_rgb_dir.mkdir(parents=True, exist_ok=True)
        tracked_rgb_dir.mkdir(parents=True, exist_ok=True)
        depth_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = run_dir / "mask"
    mask_overlay_dir = run_dir / "mask_overlay"
    instance_mask_dir = run_dir / "instance_masks"
    assigned_mask_dir = run_dir / "assigned_group_masks"
    robot_mask_dir = run_dir / "robot_exclusion_mask"
    render_group_mask_dir = run_dir / "rendered_group_masks"
    visible_gate_dir = run_dir / "depth_visible_gates"
    gated_mask_dir = run_dir / "depth_gated_group_masks"
    render_depth_dir = run_dir / "rendered_group_depth"
    observed_depth_vis_dir = run_dir / "observed_depth_vis"
    if args.save_seg_debug:
        mask_dir.mkdir(parents=True, exist_ok=True)
        mask_overlay_dir.mkdir(parents=True, exist_ok=True)
        instance_mask_dir.mkdir(parents=True, exist_ok=True)
        assigned_mask_dir.mkdir(parents=True, exist_ok=True)
        robot_mask_dir.mkdir(parents=True, exist_ok=True)
        render_group_mask_dir.mkdir(parents=True, exist_ok=True)
        visible_gate_dir.mkdir(parents=True, exist_ok=True)
        gated_mask_dir.mkdir(parents=True, exist_ok=True)
        render_depth_dir.mkdir(parents=True, exist_ok=True)
        observed_depth_vis_dir.mkdir(parents=True, exist_ok=True)

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

    robot_handles: set[int] = set()
    if args.seg_exclude_robot_mask or args.hide_robot_from_camera:
        try:
            robot_handles = _collect_robot_handles()
            print(f"[INFO] Collected {len(robot_handles)} Panda/gripper handles for robot masking.")
        except Exception as exc:
            print(f"[WARN] Failed to collect robot handles for masking: {exc}")
    if args.seg_exclude_robot_mask:
        print("[WARN] --seg-exclude-robot-mask uses simulator object IDs and is not deployable; prefer depth gating.")

    if args.hide_robot_from_camera:
        try:
            _set_robot_renderable(False)
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

    masker = None
    if args.seg_mode == "gdino-sam2":
        print("[INFO] Using segmentation mode: gdino-sam2")
        masker = GdinoSam2Masker(
            prompt=args.seg_prompt,
            extra_prompts=args.seg_extra_prompt,
            gdino_model_id=args.gdino_model_id,
            gdino_box_threshold=float(args.gdino_box_threshold),
            gdino_text_threshold=float(args.gdino_text_threshold),
            gdino_max_boxes=int(args.gdino_max_boxes),
            gdino_min_box_area=args.gdino_min_box_area,
            gdino_max_box_area=args.gdino_max_box_area,
            gdino_dedupe_iou=float(args.gdino_dedupe_iou),
            sam2_model_id=args.sam2_model_id,
            sam2_model_cfg=args.sam2_model_cfg,
            sam2_checkpoint=args.sam2_checkpoint,
        )

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
        if args.seg_mode == "gdino-sam2":
            # Keep initialization stable (no mask required there), then switch to mask-only tracking.
            optimizer.optimizer.config.use_mask_loss = False
        target_h, target_w = _init_optimizer_for_episode(
            optimizer,
            obs,
            args.camera,
            device,
            skip_init_pose_opt=bool(args.skip_init_pose_opt),
        )
        if args.seg_mode == "gdino-sam2":
            optimizer.optimizer.config.use_dino_loss = False
            optimizer.optimizer.config.use_mask_loss = True
            print("[INFO] Tracking configured: use_dino_loss=False, use_mask_loss=True")

        verify_every = max(1, int(args.verify_every))
        print(
            "[INFO] Starting random rollout: "
            f"task={args.task} episode={args.episode} seed={args.seed} steps={args.steps}"
        )

        previous_target_masks: list[np.ndarray] | None = None
        last_reliable_part_deltas = optimizer.optimizer.part_deltas.detach().clone()

        for step_idx in range(int(args.steps)):
            rgb_tensor, depth_tensor = _obs_to_tensors(obs, args.camera, device, (target_h, target_w))
            obj_mask = None
            group_obj_masks = None
            seg_boxes = []
            instance_masks: list[np.ndarray] = []
            assignment_records: list[dict[str, object]] = []
            robot_exclusion_mask = None
            group_render_masks: list[np.ndarray] = []
            group_render_depths_m: list[np.ndarray] = []
            group_render_stats: list[dict[str, float]] = []
            observed_depth_m = _depth_to_numpy(depth_tensor, (target_h, target_w))
            seg_filter_record: dict[str, object] = {
                "enabled": False,
                "dropped_instances": 0,
                "kept_instances": 0,
            }
            if masker is not None:
                rgb_np = _to_uint8_rgb(rgb_tensor.detach().cpu().numpy())
                obj_mask, seg_boxes, instance_masks = masker.predict(rgb_np)
                group_render_masks, group_render_depths_m, group_render_stats = _render_group_observables(
                    optimizer=optimizer,
                    observed_depth_m=observed_depth_m,
                )
                if args.seg_exclude_robot_mask:
                    robot_exclusion_mask = _robot_exclusion_mask_from_obs(
                        obs=obs,
                        camera=args.camera,
                        robot_handles=robot_handles,
                        shape_hw=rgb_np.shape[:2],
                    )
                    instance_masks, seg_boxes, seg_filter_record = _filter_instances_with_exclusion(
                        instance_masks=instance_masks,
                        boxes=seg_boxes,
                        exclusion_mask=robot_exclusion_mask,
                        max_overlap=float(args.seg_robot_overlap_drop_threshold),
                        min_area_px=int(args.seg_min_instance_area_px),
                    )
                    obj_mask = (
                        np.logical_or.reduce(instance_masks)
                        if instance_masks
                        else np.zeros(rgb_np.shape[:2], dtype=bool)
                    )
                group_obj_masks, current_assignments, assignment_records = _assign_instance_masks_to_groups(
                    optimizer=optimizer,
                    instance_masks=instance_masks,
                    previous_target_masks=previous_target_masks,
                    min_iou=float(args.seg_assignment_min_iou),
                    group_masks=group_render_masks,
                    group_depths_m=group_render_depths_m,
                    observed_depth_m=observed_depth_m,
                    depth_sigma_m=float(args.seg_depth_sigma_m),
                    iou_weight=float(args.seg_iou_weight),
                    depth_weight=float(args.seg_depth_weight),
                    temporal_weight=float(args.seg_temporal_weight),
                    occlusion_margin_m=float(args.seg_occlusion_margin_m),
                    min_depth_score=float(args.seg_min_depth_score),
                    min_depth_valid_px=int(args.seg_min_depth_valid_px),
                    min_target_area_px=int(args.seg_min_target_area_px),
                    min_target_render_ratio=float(args.seg_min_target_render_ratio),
                    use_depth_score=bool(args.seg_depth_aware_assignment),
                    use_depth_gate=bool(args.seg_depth_gate),
                )
                previous_target_masks = [np.asarray(mask).copy() for mask in group_obj_masks]
            pose_rollback_groups = (
                _unreliable_groups_from_assignments(assignment_records)
                if bool(args.seg_rollback_unreliable_pose)
                else []
            )
            if pose_rollback_groups:
                _restore_group_poses(
                    optimizer,
                    last_reliable_part_deltas,
                    pose_rollback_groups,
                )
            optimizer.set_observation(
                rgb_tensor,
                optimizer.cam2world_ns_ds,
                depth_tensor,
                obj_mask=group_obj_masks if group_obj_masks is not None else obj_mask,
            )

            loss_before = _tracking_loss_proxy(
                optimizer,
                use_depth=bool(args.track_use_depth),
                use_rgb=bool(args.track_use_rgb),
            )

            preopt_render_rgb = _render_current_tracking_rgb(optimizer) if args.save_frames else None
            pose_deltas_before_step = optimizer.optimizer.part_deltas.detach().clone()

            step_render_dict = optimizer.step_opt(
                niter=args.first_niters if step_idx == 0 else args.niters,
                use_depth=bool(args.track_use_depth),
                use_rgb=bool(args.track_use_rgb),
            )
            pose_step_metrics = _pose_step_metrics(
                pose_deltas_before_step,
                optimizer.optimizer.part_deltas.detach(),
            )
            if pose_rollback_groups:
                _restore_group_poses(
                    optimizer,
                    last_reliable_part_deltas,
                    pose_rollback_groups,
                )
            reliable_pose_updates = _update_last_reliable_part_deltas(
                last_reliable_part_deltas=last_reliable_part_deltas,
                current_part_deltas=optimizer.optimizer.part_deltas.detach(),
                assignment_records=assignment_records,
                min_target_render_ratio=float(args.seg_reliable_target_render_ratio),
                min_iou=float(args.seg_reliable_iou),
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
                _save_rgb_resized(rgb_resized_dir / f"{step_idx:04d}.png", np.asarray(rgb_frame), (target_h, target_w))
                if preopt_render_rgb is not None:
                    Image.fromarray(preopt_render_rgb).save(preopt_tracked_rgb_dir / f"{step_idx:04d}.png")
                np.save(depth_dir / f"{step_idx:04d}.npy", np.asarray(depth_frame, dtype=np.float32))
                if isinstance(step_render_dict, dict) and "rgb" in step_render_dict:
                    render_rgb = step_render_dict["rgb"]
                    if isinstance(render_rgb, torch.Tensor):
                        render_rgb = render_rgb.detach().cpu().numpy()
                    render_rgb = np.asarray(render_rgb)
                    if render_rgb.ndim == 4 and render_rgb.shape[0] == 1:
                        render_rgb = render_rgb[0]
                    Image.fromarray(_to_uint8_rgb(render_rgb)).save(tracked_rgb_dir / f"{step_idx:04d}.png")
            if args.save_seg_debug and obj_mask is not None:
                mask_u8 = (obj_mask.astype(np.uint8) * 255)
                Image.fromarray(mask_u8).save(mask_dir / f"{step_idx:04d}.png")
                rgb_u8 = _to_uint8_rgb(rgb_tensor.detach().cpu().numpy())
                ov = overlay(rgb_u8, obj_mask.astype(np.uint8), color=(255, 0, 0), alpha=0.45)
                Image.fromarray(ov.astype(np.uint8)).save(mask_overlay_dir / f"{step_idx:04d}.png")
                step_instance_dir = instance_mask_dir / f"{step_idx:04d}"
                step_instance_dir.mkdir(parents=True, exist_ok=True)
                for inst_idx, inst_mask in enumerate(instance_masks):
                    Image.fromarray((inst_mask.astype(np.uint8) * 255)).save(step_instance_dir / f"{inst_idx:02d}.png")
                step_assigned_dir = assigned_mask_dir / f"{step_idx:04d}"
                step_assigned_dir.mkdir(parents=True, exist_ok=True)
                if group_obj_masks is not None:
                    for group_idx, group_mask in enumerate(group_obj_masks):
                        Image.fromarray(((np.asarray(group_mask) > 0.5).astype(np.uint8) * 255)).save(step_assigned_dir / f"{group_idx:02d}.png")
                if robot_exclusion_mask is not None:
                    Image.fromarray((robot_exclusion_mask.astype(np.uint8) * 255)).save(
                        robot_mask_dir / f"{step_idx:04d}.png"
                    )
                _save_depth_debug(observed_depth_vis_dir / f"{step_idx:04d}.png", observed_depth_m)
                step_render_mask_dir = render_group_mask_dir / f"{step_idx:04d}"
                step_render_mask_dir.mkdir(parents=True, exist_ok=True)
                step_visible_gate_dir = visible_gate_dir / f"{step_idx:04d}"
                step_visible_gate_dir.mkdir(parents=True, exist_ok=True)
                step_gated_mask_dir = gated_mask_dir / f"{step_idx:04d}"
                step_gated_mask_dir.mkdir(parents=True, exist_ok=True)
                step_render_depth_dir = render_depth_dir / f"{step_idx:04d}"
                step_render_depth_dir.mkdir(parents=True, exist_ok=True)
                for group_idx, render_mask in enumerate(group_render_masks):
                    Image.fromarray((render_mask.astype(np.uint8) * 255)).save(step_render_mask_dir / f"{group_idx:02d}.png")
                    if group_idx < len(group_render_depths_m):
                        _save_depth_debug(
                            step_render_depth_dir / f"{group_idx:02d}.png",
                            group_render_depths_m[group_idx],
                            valid_mask=render_mask,
                        )
                        visible_gate = _visible_gate_from_depth(
                            observed_depth_m=observed_depth_m,
                            rendered_depth_m=group_render_depths_m[group_idx],
                            rendered_mask=render_mask,
                            occlusion_margin_m=float(args.seg_occlusion_margin_m),
                        )
                        Image.fromarray((visible_gate.astype(np.uint8) * 255)).save(
                            step_visible_gate_dir / f"{group_idx:02d}.png"
                        )
                        if group_obj_masks is not None and group_idx < len(group_obj_masks):
                            Image.fromarray(((np.asarray(group_obj_masks[group_idx]) > 0.5).astype(np.uint8) * 255)).save(
                                step_gated_mask_dir / f"{group_idx:02d}.png"
                            )

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
                "opt_mask_bce_loss_last": float(step_metrics.get("mask_bce_loss_last", float("nan"))),
                "reward": float("nan"),
                "terminate": False,
                "action_retry": -1,
                "seg_num_boxes": int(len(seg_boxes)),
                "seg_boxes": seg_boxes,
                "seg_num_instances": int(len(instance_masks)),
                "seg_ignored_groups": int(sum(1 for rec in assignment_records if rec.get("ignored"))),
                "seg_filter": seg_filter_record,
                "seg_assignments": assignment_records,
                "seg_render_group_stats": group_render_stats,
                "per_group_loss_metrics": {
                    key: value for key, value in step_metrics.items() if key.startswith("group_")
                },
                "per_group_pose_step_metrics": pose_step_metrics,
                "pose_rollback_groups": [int(g) for g in pose_rollback_groups],
                "reliable_pose_update_groups": [int(g) for g in reliable_pose_updates],
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
                mean_iou = float(np.mean([float(r.get("iou", 0.0)) for r in assignment_records])) if assignment_records else 0.0
                mean_depth_err = [
                    float(r.get("median_depth_err_m", float("nan")))
                    for r in assignment_records
                    if np.isfinite(float(r.get("median_depth_err_m", float("nan"))))
                ]
                mean_depth_err_val = float(np.mean(mean_depth_err)) if mean_depth_err else float("nan")
                total_visible = int(sum(int(r.get("visible_area", 0)) for r in assignment_records))
                print(
                    "[VERIFY] "
                    f"step={step_idx:03d} "
                    f"total={float(row['opt_total_loss_last']):.6f} "
                    f"mask_bce={float(row['opt_mask_bce_loss_last']):.6f} "
                    f"instances={int(row['seg_num_instances'])} "
                    f"ignored={int(row['seg_ignored_groups'])} "
                    f"assign_iou={mean_iou:.3f} "
                    f"depth_err_m={mean_depth_err_val:.4f} "
                    f"visible_px={total_visible} "
                    f"robot_dropped={int(seg_filter_record.get('dropped_instances', 0))} "
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

        video_paths: dict[str, str] = {}
        if args.save_videos:
            if not args.save_frames:
                print("[WARN] --save-videos requested but --no-save-frames was set; skipping video export.")
            else:
                video_fps = float(args.video_fps)
                rgb_video = _write_image_sequence_video(rgb_dir, run_dir / f"{args.camera}_rgb.mp4", video_fps)
                if rgb_video:
                    video_paths[f"{args.camera}_rgb"] = rgb_video

                rgb_resized_video = _write_image_sequence_video(
                    rgb_resized_dir,
                    run_dir / f"{args.camera}_rgb_tracking_res.mp4",
                    video_fps,
                )
                if rgb_resized_video:
                    video_paths[f"{args.camera}_rgb_tracking_res"] = rgb_resized_video

                preopt_video = _write_image_sequence_video(
                    preopt_tracked_rgb_dir,
                    run_dir / "preopt_tracked_rgb.mp4",
                    video_fps,
                )
                if preopt_video:
                    video_paths["preopt_tracked_rgb"] = preopt_video

                tracked_video = _write_image_sequence_video(tracked_rgb_dir, run_dir / "tracked_rgb.mp4", video_fps)
                if tracked_video:
                    video_paths["tracked_rgb"] = tracked_video

                if args.save_seg_debug:
                    overlay_video = _write_image_sequence_video(
                        mask_overlay_dir,
                        run_dir / "mask_overlay.mp4",
                        video_fps,
                    )
                    if overlay_video:
                        video_paths["mask_overlay"] = overlay_video

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
            "hide_robot_from_camera": bool(args.hide_robot_from_camera),
            "seg_mode": args.seg_mode,
            "seg_prompt": args.seg_prompt,
            "seg_extra_prompt": list(args.seg_extra_prompt),
            "seg_assignment_min_iou": float(args.seg_assignment_min_iou),
            "seg_exclude_robot_mask": bool(args.seg_exclude_robot_mask),
            "seg_robot_overlap_drop_threshold": float(args.seg_robot_overlap_drop_threshold),
            "seg_min_instance_area_px": int(args.seg_min_instance_area_px),
            "seg_depth_aware_assignment": bool(args.seg_depth_aware_assignment),
            "seg_depth_gate": bool(args.seg_depth_gate),
            "seg_depth_sigma_m": float(args.seg_depth_sigma_m),
            "seg_occlusion_margin_m": float(args.seg_occlusion_margin_m),
            "seg_min_depth_score": float(args.seg_min_depth_score),
            "seg_min_depth_valid_px": int(args.seg_min_depth_valid_px),
            "seg_min_target_area_px": int(args.seg_min_target_area_px),
            "seg_min_target_render_ratio": float(args.seg_min_target_render_ratio),
            "seg_rollback_unreliable_pose": bool(args.seg_rollback_unreliable_pose),
            "seg_reliable_target_render_ratio": float(args.seg_reliable_target_render_ratio),
            "seg_reliable_iou": float(args.seg_reliable_iou),
            "seg_iou_weight": float(args.seg_iou_weight),
            "seg_depth_weight": float(args.seg_depth_weight),
            "seg_temporal_weight": float(args.seg_temporal_weight),
            "first_niters": int(args.first_niters),
            "niters": int(args.niters),
            "skip_init_pose_opt": bool(args.skip_init_pose_opt),
            "tracking_2d_signal": {
                "checks": int(len(finite_rows)),
                "improve_rate": improve_rate,
            },
            "frames_dir": str(rgb_dir) if args.save_frames else "",
            "frames_tracking_res_dir": str(rgb_resized_dir) if args.save_frames else "",
            "preopt_tracked_frames_dir": str(preopt_tracked_rgb_dir) if args.save_frames else "",
            "tracked_frames_dir": str(tracked_rgb_dir) if args.save_frames else "",
            "depth_dir": str(depth_dir) if args.save_frames else "",
            "mask_dir": str(mask_dir) if args.save_seg_debug else "",
            "mask_overlay_dir": str(mask_overlay_dir) if args.save_seg_debug else "",
            "instance_mask_dir": str(instance_mask_dir) if args.save_seg_debug else "",
            "assigned_mask_dir": str(assigned_mask_dir) if args.save_seg_debug else "",
            "robot_mask_dir": str(robot_mask_dir) if args.save_seg_debug else "",
            "render_group_mask_dir": str(render_group_mask_dir) if args.save_seg_debug else "",
            "visible_gate_dir": str(visible_gate_dir) if args.save_seg_debug else "",
            "gated_mask_dir": str(gated_mask_dir) if args.save_seg_debug else "",
            "render_depth_dir": str(render_depth_dir) if args.save_seg_debug else "",
            "observed_depth_vis_dir": str(observed_depth_vis_dir) if args.save_seg_debug else "",
            "video_paths": video_paths,
            "rows": rollout_rows,
        }

        out_json = run_dir / "random_tracking_rollout.json"
        with out_json.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print(f"[INFO] Saved rollout report: {out_json}")
        if args.save_frames:
            print(f"[INFO] Saved RGB frames: {rgb_dir}")
            print(f"[INFO] Saved tracking-res RGB frames: {rgb_resized_dir}")
            print(f"[INFO] Saved pre-optimization tracked renders: {preopt_tracked_rgb_dir}")
            print(f"[INFO] Saved tracked renders: {tracked_rgb_dir}")
            print(f"[INFO] Saved depth frames: {depth_dir}")
        for name, path in video_paths.items():
            print(f"[INFO] Saved {name} video: {path}")
    finally:
        env.shutdown()


if __name__ == "__main__":
    main()
