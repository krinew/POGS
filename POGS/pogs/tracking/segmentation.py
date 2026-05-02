from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


@dataclass
class Sam2Config:
    model_id: Optional[str] = "facebook/sam2-hiera-large"
    model_cfg: Optional[str] = None
    checkpoint: Optional[str] = None
    device: str = "cuda"


@dataclass
class ClipSegConfig:
    model_id: str = "CIDAS/clipseg-rd64-refined"
    device: str = "cuda"
    threshold: float = 0.5


@dataclass
class GroundingDinoConfig:
    model_id: str = "IDEA-Research/grounding-dino-tiny"
    device: str = "cuda"
    box_threshold: float = 0.35
    text_threshold: float = 0.25
    max_boxes: int = 5
    min_box_area: Optional[float] = None
    max_box_area: Optional[float] = None


def _prepare_image(image: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu()
        if image.ndim == 3 and image.shape[0] in (1, 3):
            image = image.permute(1, 2, 0)
        image = image.numpy()

    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected HxWx3 RGB image, got shape {image.shape}.")

    image = image.astype(np.float32)
    if image.max() <= 1.0 + 1e-6:
        image = image * 255.0
    image = np.clip(image, 0.0, 255.0).astype(np.uint8)
    return image


class Sam2Segmenter:
    """Prompt-driven SAM2 segmentation for single images."""

    def __init__(self, config: Optional[Sam2Config] = None) -> None:
        self.config = config or Sam2Config()
        self._predictor = None

    def _load_predictor(self) -> None:
        if self._predictor is not None:
            return

        try:
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            if self.config.model_cfg and self.config.checkpoint:
                from sam2.build_sam import build_sam2

                model = build_sam2(self.config.model_cfg, self.config.checkpoint)
                self._predictor = SAM2ImagePredictor(model)
            elif self.config.model_id:
                self._predictor = SAM2ImagePredictor.from_pretrained(self.config.model_id)
            else:
                raise ValueError("Provide model_id or model_cfg+checkpoint for SAM2.")
        except Exception as exc:  # pragma: no cover - depends on optional dependency
            raise RuntimeError(
                "SAM2 is not available. Install with: pip install -e \"POGS/POGS[sam2]\""
            ) from exc

    def _prepare_image(self, image: np.ndarray | torch.Tensor) -> np.ndarray:
        return _prepare_image(image)

    def _autocast_ctx(self):
        if self.config.device.startswith("cuda") and torch.cuda.is_available():
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return nullcontext()

    def predict_box_mask(
        self,
        image: np.ndarray | torch.Tensor,
        box_xyxy: Sequence[float],
        multimask_output: bool = False,
    ) -> np.ndarray:
        """Return a binary mask (HxW) from a single box prompt."""
        self._load_predictor()
        assert self._predictor is not None

        image = self._prepare_image(image)
        box = np.array(box_xyxy, dtype=np.float32)
        if box.shape == (4,):
            box = box[None, :]

        with torch.inference_mode(), self._autocast_ctx():
            self._predictor.set_image(image)
            masks, scores, _ = self._predictor.predict(
                box=box,
                multimask_output=multimask_output,
            )

        if masks.ndim == 2:
            mask = masks
        else:
            if scores is not None and len(scores) == masks.shape[0]:
                mask = masks[int(np.argmax(scores))]
            else:
                mask = masks[0]

        return mask.astype(bool)

    def predict_union_mask(
        self,
        image: np.ndarray | torch.Tensor,
        boxes_xyxy: Iterable[Sequence[float]],
    ) -> np.ndarray:
        """Union multiple box masks into a single binary mask."""
        union = None
        for box in boxes_xyxy:
            mask = self.predict_box_mask(image, box)
            union = mask if union is None else (union | mask)
        if union is None:
            raise ValueError("boxes_xyxy must contain at least one box.")
        return union

    def predict_instance_masks(
        self,
        image: np.ndarray | torch.Tensor,
        boxes_xyxy: Iterable[Sequence[float]],
    ) -> list[np.ndarray]:
        """Return one SAM2 mask per box prompt."""
        return [self.predict_box_mask(image, box) for box in boxes_xyxy]


class ClipSegSegmenter:
    """Text-prompt CLIPSeg segmentation for single images."""

    def __init__(self, config: Optional[ClipSegConfig] = None) -> None:
        self.config = config or ClipSegConfig()
        self._processor = None
        self._model = None

    def _load_model(self) -> None:
        if self._model is not None and self._processor is not None:
            return

        try:
            from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

            self._processor = CLIPSegProcessor.from_pretrained(self.config.model_id)
            self._model = CLIPSegForImageSegmentation.from_pretrained(self.config.model_id)
            self._model.to(self.config.device)
            self._model.eval()
        except Exception as exc:  # pragma: no cover - depends on optional dependency
            raise RuntimeError(
                "CLIPSeg is not available. Install transformers or check the model id."
            ) from exc

    def predict_text_mask(
        self,
        image: np.ndarray | torch.Tensor,
        prompt: str,
        threshold: Optional[float] = None,
    ) -> np.ndarray:
        """Return a binary mask (HxW) from a text prompt."""
        self._load_model()
        assert self._processor is not None
        assert self._model is not None

        image_uint8 = _prepare_image(image)
        h, w = image_uint8.shape[:2]
        image_pil = Image.fromarray(image_uint8)

        inputs = self._processor(text=[prompt], images=[image_pil], return_tensors="pt")
        inputs = {k: v.to(self.config.device) for k, v in inputs.items()}
        with torch.inference_mode():
            outputs = self._model(**inputs)

        logits = outputs.logits
        if logits.ndim == 2:
            logits = logits.unsqueeze(0)
        mask = torch.sigmoid(logits)
        if mask.shape[-2:] != (h, w):
            mask = F.interpolate(
                mask.unsqueeze(1),
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)

        mask_np = mask[0].detach().cpu().numpy()
        thresh = self.config.threshold if threshold is None else float(threshold)
        return mask_np > thresh


class GroundingDinoBoxer:
    """Text-prompt GroundingDINO boxes for single images."""

    def __init__(self, config: Optional[GroundingDinoConfig] = None) -> None:
        self.config = config or GroundingDinoConfig()
        self._processor = None
        self._model = None

    def _load_model(self) -> None:
        if self._processor is not None and self._model is not None:
            return

        try:
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

            self._processor = AutoProcessor.from_pretrained(self.config.model_id)
            self._model = AutoModelForZeroShotObjectDetection.from_pretrained(self.config.model_id)
            self._model.to(self.config.device)
            self._model.eval()
        except Exception as exc:  # pragma: no cover - depends on optional dependency
            raise RuntimeError(
                "GroundingDINO is not available. Install transformers or check the model id."
            ) from exc

    def predict_boxes(
        self,
        image: np.ndarray | torch.Tensor,
        prompt: str,
        box_threshold: Optional[float] = None,
        text_threshold: Optional[float] = None,
        max_boxes: Optional[int] = None,
        min_box_area: Optional[float] = None,
        max_box_area: Optional[float] = None,
    ) -> list[list[float]]:
        """Return a list of boxes in XYXY pixel coords from a text prompt."""
        self._load_model()
        assert self._processor is not None
        assert self._model is not None

        image_uint8 = _prepare_image(image)
        h, w = image_uint8.shape[:2]
        image_pil = Image.fromarray(image_uint8)

        text = prompt.strip()
        if not text:
            raise ValueError("Prompt must be non-empty.")
        if not text.endswith("."):
            text = text + "."

        inputs = self._processor(images=[image_pil], text=text, return_tensors="pt")
        inputs = {k: v.to(self.config.device) for k, v in inputs.items()}
        with torch.inference_mode():
            outputs = self._model(**inputs)

        thresh = self.config.box_threshold if box_threshold is None else float(box_threshold)
        text_thresh = self.config.text_threshold if text_threshold is None else float(text_threshold)

        if hasattr(self._processor, "post_process_grounded_object_detection"):
            postprocess = self._processor.post_process_grounded_object_detection
            try:
                results = postprocess(
                    outputs,
                    inputs.get("input_ids"),
                    threshold=thresh,
                    text_threshold=text_thresh,
                    target_sizes=[(h, w)],
                )
            except TypeError as exc:
                if "threshold" not in str(exc):
                    raise
                results = postprocess(
                    outputs,
                    inputs.get("input_ids"),
                    box_threshold=thresh,
                    text_threshold=text_thresh,
                    target_sizes=[(h, w)],
                )
        elif hasattr(self._processor, "post_process_object_detection"):
            results = self._processor.post_process_object_detection(
                outputs,
                threshold=thresh,
                target_sizes=[(h, w)],
            )
        else:
            raise RuntimeError("Processor does not support post-processing for GroundingDINO.")

        if not results:
            return []

        boxes = results[0].get("boxes", [])
        scores = results[0].get("scores", None)
        if isinstance(boxes, torch.Tensor):
            boxes = boxes.detach().cpu().numpy()
        if isinstance(scores, torch.Tensor):
            scores = scores.detach().cpu().numpy()

        if boxes is None or len(boxes) == 0:
            return []

        order = None
        if scores is not None:
            order = np.argsort(scores)[::-1]

        max_keep = self.config.max_boxes if max_boxes is None else int(max_boxes)
        if max_keep > 0 and order is not None:
            order = order[:max_keep]

        if order is not None:
            boxes = boxes[order]

        min_area = self.config.min_box_area if min_box_area is None else float(min_box_area)
        max_area = self.config.max_box_area if max_box_area is None else float(max_box_area)

        out: list[list[float]] = []
        for box in boxes:
            x0, y0, x1, y1 = [float(x) for x in box.tolist()]
            x0 = max(0.0, min(float(w - 1), x0))
            x1 = max(0.0, min(float(w - 1), x1))
            y0 = max(0.0, min(float(h - 1), y0))
            y1 = max(0.0, min(float(h - 1), y1))
            if x1 <= x0 or y1 <= y0:
                continue
            area_frac = ((x1 - x0) * (y1 - y0)) / max(float(w * h), 1.0)
            if min_area is not None and area_frac < min_area:
                continue
            if max_area is not None and area_frac > max_area:
                continue
            out.append([x0, y0, x1, y1])

        return out
