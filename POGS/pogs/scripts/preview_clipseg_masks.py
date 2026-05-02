"""Preview text-prompt masks for saved rollout frames."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

from pogs.tracking.segmentation import (
    ClipSegConfig,
    ClipSegSegmenter,
    GroundingDinoConfig,
    GroundingDinoBoxer,
    Sam2Config,
    Sam2Segmenter,
)
from pogs.tracking.utils2 import overlay


INSTANCE_COLORS = [
    (255, 0, 0),
    (0, 180, 255),
    (0, 220, 80),
    (255, 220, 0),
    (220, 0, 255),
    (255, 128, 0),
    (128, 255, 255),
    (255, 128, 200),
    (180, 180, 180),
    (80, 120, 255),
]


def _list_images(path: Path) -> list[Path]:
    if not path.exists():
        raise FileNotFoundError(f"Image directory not found: {path}")
    exts = {".png", ".jpg", ".jpeg"}
    images = [p for p in path.iterdir() if p.suffix.lower() in exts]
    return sorted(images)


def _parse_views(view_args: Iterable[str], rollout_json: Path | None) -> dict[str, Path]:
    views: dict[str, Path] = {}
    if rollout_json is not None:
        if not rollout_json.exists():
            raise FileNotFoundError(f"Rollout JSON not found: {rollout_json}")
        payload = json.loads(rollout_json.read_text(encoding="utf-8"))
        frames_dir = payload.get("frames_dir")
        tracked_dir = payload.get("tracked_frames_dir")
        if frames_dir:
            views["front"] = Path(frames_dir)
        if tracked_dir:
            views["splat"] = Path(tracked_dir)

    for item in view_args:
        if "=" not in item:
            raise ValueError(f"Invalid --view entry '{item}', expected name=path")
        name, path = item.split("=", 1)
        views[name.strip()] = Path(path.strip())

    if not views:
        raise ValueError("Provide --rollout-json or at least one --view name=path")
    return views


def _parse_indices(
    total: int,
    duration_sec: float,
    fps: float,
    source_fps: float,
    stride: int | None,
    start_index: int,
    indices_arg: str | None,
) -> list[int]:
    if indices_arg:
        indices = [int(x) for x in indices_arg.split(",") if x.strip()]
        return [i for i in indices if 0 <= i < total]

    num_frames = int(round(duration_sec * fps))
    if num_frames <= 0:
        raise ValueError("duration_sec * fps must be >= 1")

    if stride is None:
        stride = max(1, int(round(source_fps / fps)))

    indices = [start_index + i * stride for i in range(num_frames)]
    return [i for i in indices if 0 <= i < total]


def _save_mask(path: Path, mask: np.ndarray) -> None:
    mask_u8 = (mask.astype(np.uint8) * 255)
    Image.fromarray(mask_u8).save(path)


def _save_overlay(path: Path, image: np.ndarray, mask: np.ndarray, alpha: float) -> None:
    overlay_img = overlay(image, mask.astype(np.uint8), (255, 0, 0), alpha)
    Image.fromarray(overlay_img.astype(np.uint8)).save(path)


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
    prompts = []
    for item in [prompt, *extra_prompts]:
        prompts.extend([p.strip() for p in item.split("|") if p.strip()])
    return prompts


def _save_instance_overlay(path: Path, image: np.ndarray, masks: list[np.ndarray], alpha: float) -> None:
    out = image.astype(np.float32).copy()
    for idx, mask in enumerate(masks):
        color = np.asarray(INSTANCE_COLORS[idx % len(INSTANCE_COLORS)], dtype=np.float32)
        mask_bool = mask.astype(bool)
        out[mask_bool] = (1.0 - alpha) * out[mask_bool] + alpha * color
    Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)).save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview text-driven masks for saved frames.")
    parser.add_argument("--mode", choices=["clipseg", "gdino-sam2"], default="clipseg")
    parser.add_argument("--prompt", required=True, help="Text prompt, e.g. 'stacked blocks'.")
    parser.add_argument(
        "--extra-prompt",
        action="append",
        default=[],
        help="Additional text prompt. You can also separate prompts with '|'.",
    )
    parser.add_argument("--rollout-json", type=str, default=None, help="Rollout JSON with frames_dir paths.")
    parser.add_argument(
        "--view",
        action="append",
        default=[],
        help="Add view as name=path. Example: --view front=/path --view splat=/path",
    )
    parser.add_argument("--out-dir", type=str, default="outputs/clipseg_mask_previews")
    parser.add_argument("--duration-sec", type=float, default=5.0)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--source-fps", type=float, default=20.0)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--indices", type=str, default=None, help="Comma-separated frame indices.")
    parser.add_argument("--model-id", type=str, default=None, help="CLIPSeg model id.")
    parser.add_argument("--threshold", type=float, default=0.5, help="CLIPSeg mask threshold.")
    parser.add_argument("--gdino-model-id", type=str, default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument("--gdino-box-threshold", type=float, default=0.35)
    parser.add_argument("--gdino-text-threshold", type=float, default=0.25)
    parser.add_argument("--gdino-max-boxes", type=int, default=5)
    parser.add_argument("--gdino-min-box-area", type=float, default=None)
    parser.add_argument("--gdino-max-box-area", type=float, default=None)
    parser.add_argument("--gdino-dedupe-iou", type=float, default=0.92)
    parser.add_argument("--sam2-model-id", type=str, default="facebook/sam2-hiera-large")
    parser.add_argument("--sam2-model-cfg", type=str, default=None)
    parser.add_argument("--sam2-checkpoint", type=str, default=None)
    parser.add_argument("--overlay-alpha", type=float, default=0.45)
    args = parser.parse_args()

    rollout_json = Path(args.rollout_json) if args.rollout_json else None
    views = _parse_views(args.view, rollout_json)

    clipseg = None
    gdino = None
    sam2 = None

    if args.mode == "clipseg":
        config = ClipSegConfig(
            model_id=args.model_id or ClipSegConfig().model_id,
            threshold=float(args.threshold),
        )
        clipseg = ClipSegSegmenter(config)
    else:
        gdino = GroundingDinoBoxer(
            GroundingDinoConfig(
                model_id=args.gdino_model_id,
                box_threshold=float(args.gdino_box_threshold),
                text_threshold=float(args.gdino_text_threshold),
                max_boxes=int(args.gdino_max_boxes),
                min_box_area=args.gdino_min_box_area,
                max_box_area=args.gdino_max_box_area,
            )
        )
        sam2 = Sam2Segmenter(
            Sam2Config(
                model_id=args.sam2_model_id,
                model_cfg=args.sam2_model_cfg,
                checkpoint=args.sam2_checkpoint,
            )
        )

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    summary: dict[str, object] = {
        "meta": {
            "mode": args.mode,
            "prompt": args.prompt,
            "extra_prompts": args.extra_prompt,
        },
        "views": {},
    }
    prompts = _parse_prompts(args.prompt, args.extra_prompt)

    for view_name, view_dir in views.items():
        frames = _list_images(view_dir)
        indices = _parse_indices(
            total=len(frames),
            duration_sec=float(args.duration_sec),
            fps=float(args.fps),
            source_fps=float(args.source_fps),
            stride=args.stride,
            start_index=int(args.start_index),
            indices_arg=args.indices,
        )
        if not indices:
            raise RuntimeError(f"No frames selected for view '{view_name}'.")

        view_out = out_root / view_name
        masks_dir = view_out / "masks"
        overlays_dir = view_out / "overlays"
        instance_masks_dir = view_out / "instance_masks"
        instance_overlays_dir = view_out / "instance_overlays"
        masks_dir.mkdir(parents=True, exist_ok=True)
        overlays_dir.mkdir(parents=True, exist_ok=True)
        if args.mode == "gdino-sam2":
            instance_masks_dir.mkdir(parents=True, exist_ok=True)
            instance_overlays_dir.mkdir(parents=True, exist_ok=True)

        view_records = []
        for idx in indices:
            frame_path = frames[idx]
            image = np.asarray(Image.open(frame_path).convert("RGB"))
            boxes = None
            if args.mode == "clipseg":
                assert clipseg is not None
                mask = clipseg.predict_text_mask(image, args.prompt, threshold=args.threshold)
            else:
                assert gdino is not None and sam2 is not None
                boxes = []
                for prompt in prompts:
                    boxes.extend(
                        gdino.predict_boxes(
                            image,
                            prompt,
                            box_threshold=args.gdino_box_threshold,
                            text_threshold=args.gdino_text_threshold,
                            max_boxes=args.gdino_max_boxes,
                            min_box_area=args.gdino_min_box_area,
                            max_box_area=args.gdino_max_box_area,
                        )
                    )
                boxes = _dedupe_boxes(boxes, float(args.gdino_dedupe_iou))
                if not boxes:
                    mask = np.zeros(image.shape[:2], dtype=bool)
                    instance_masks = []
                else:
                    instance_masks = sam2.predict_instance_masks(image, boxes)
                    mask = np.logical_or.reduce(instance_masks) if instance_masks else np.zeros(image.shape[:2], dtype=bool)

            _save_mask(masks_dir / f"{idx:04d}.png", mask)
            _save_overlay(overlays_dir / f"{idx:04d}.png", image, mask, float(args.overlay_alpha))
            record = {"index": int(idx), "file": str(frame_path)}
            if boxes is not None:
                record["boxes"] = boxes
                record["num_instances"] = len(instance_masks)
                _save_instance_overlay(
                    instance_overlays_dir / f"{idx:04d}.png",
                    image,
                    instance_masks,
                    float(args.overlay_alpha),
                )
                frame_instance_dir = instance_masks_dir / f"{idx:04d}"
                frame_instance_dir.mkdir(parents=True, exist_ok=True)
                for inst_idx, inst_mask in enumerate(instance_masks):
                    _save_mask(frame_instance_dir / f"{inst_idx:02d}.png", inst_mask)
            view_records.append(record)

        summary["views"][view_name] = {
            "source_dir": str(view_dir),
            "indices": indices,
            "frames": view_records,
        }

    (out_root / "preview_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[INFO] Saved previews under: {out_root}")


if __name__ == "__main__":
    main()
