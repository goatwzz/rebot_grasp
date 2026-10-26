"""Local SAM box-prompt segmentation; no automatic weight downloads."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .target_mask_utils import validate_bbox, validate_mask

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class SAMSettings:
    checkpoint: Path
    device: str = "cuda:0"
    min_mask_area_px: int = 64

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> SAMSettings:
        raw = cfg.get("sam") or {}
        path = Path(str(raw.get("checkpoint", "models/sam_b.pt"))).expanduser()
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if not path.is_file():
            raise ValueError(f"SAM 权重不存在：{path}；请配置 sam.checkpoint，不会自动下载")
        area = int(raw.get("min_mask_area_px", 64))
        if area < 1:
            raise ValueError("sam.min_mask_area_px 必须大于 0")
        return cls(path, str(raw.get("device", "cuda:0")), area)


@dataclass
class SegmentedTarget:
    mask: np.ndarray
    bbox_xyxy: tuple[int, int, int, int]
    area_px: int


class SAMSegmenter:
    def __init__(self, settings: SAMSettings) -> None:
        from ultralytics import SAM

        self.settings = settings
        self.model = SAM(str(settings.checkpoint))

    def segment(self, color_bgr: np.ndarray, bbox_xyxy) -> SegmentedTarget:
        bbox = validate_bbox(bbox_xyxy, color_bgr.shape[:2])
        results = self.model.predict(
            source=color_bgr, bboxes=[list(bbox)], device=self.settings.device,
            verbose=False, save=False, stream=False,
        )
        masks = getattr(results[0], "masks", None) if results else None
        data = getattr(masks, "data", None)
        # A single box must produce exactly one mask; never silently pick a
        # different instance from a batch or resize a mask to hide misalignment.
        if data is None or len(data) != 1:
            raise ValueError("SAM 未返回唯一目标掩码，本轮不生成抓取候选")
        raw = data[0].detach().cpu().numpy()
        mask = validate_mask(raw, color_bgr.shape[:2])
        # SAM may extend beyond the prompt box. Restrict to the VLM selection.
        x1, y1, x2, y2 = bbox
        clipped = np.zeros_like(mask)
        clipped[y1:y2 + 1, x1:x2 + 1] = mask[y1:y2 + 1, x1:x2 + 1]
        mask = validate_mask(clipped, color_bgr.shape[:2])
        if int(mask.sum()) < self.settings.min_mask_area_px:
            raise ValueError("SAM 目标掩码面积过小，本轮不生成抓取候选")
        return SegmentedTarget(mask, bbox, int(mask.sum()))


def draw_target_mask(image: np.ndarray, target: SegmentedTarget) -> np.ndarray:
    display = image.copy()
    display[target.mask] = (
        display[target.mask].astype(np.float32) * 0.55 + np.array([255, 200, 0]) * 0.45
    ).astype(np.uint8)
    contours, _ = cv2.findContours(target.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(display, contours, -1, (255, 255, 0), 2)
    return display
