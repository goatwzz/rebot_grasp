"""Validated image-space target constraints, independent of CUDA and robot SDKs."""

from __future__ import annotations

import numpy as np


def validate_bbox(bbox, image_shape: tuple[int, int]) -> tuple[int, int, int, int]:
    h, w = image_shape
    values = np.asarray(bbox)
    if values.shape != (4,) or values.dtype.kind not in "iuf" or not np.isfinite(values).all():
        raise ValueError("目标框必须是四个有限像素坐标 [x1,y1,x2,y2]")
    x1, y1, x2, y2 = values
    if not (0 <= x1 < x2 < w and 0 <= y1 < y2 < h):
        raise ValueError(f"目标框必须位于原图 {w}x{h} 范围内，且左右、上下有序")
    if not np.equal(values, np.floor(values)).all():
        raise ValueError("目标框必须先换算为原图整数像素坐标")
    return tuple(int(v) for v in values)


def validate_mask(
    mask: np.ndarray,
    image_shape: tuple[int, int],
    *,
    allow_empty: bool = False,
) -> np.ndarray:
    value = np.asarray(mask)
    if value.shape != tuple(image_shape) or value.dtype.kind not in "buif":
        raise ValueError("目标掩码必须与原图高宽一致，不能自动缩放或广播")
    if not np.isfinite(value).all() or not np.isin(value, [0, 1, 255]).all():
        raise ValueError("目标掩码必须是二值掩码（0/1 或 0/255）")
    value = value != 0
    if not allow_empty and not value.any():
        raise ValueError("目标掩码为空，本轮不生成抓取候选")
    return value


def validate_rgbd(color: np.ndarray, depth_mm: np.ndarray, K: np.ndarray) -> None:
    if color is None or color.ndim != 3 or color.shape[2] != 3 or color.dtype != np.uint8:
        raise ValueError("需要 uint8 BGR 原始图像")
    if depth_mm is None or depth_mm.shape != color.shape[:2]:
        raise ValueError("需要同一快照、对齐到彩色图像的深度图（毫米），不能使用另一帧深度")
    if depth_mm.dtype.kind not in "uif" or not np.isfinite(depth_mm).all() or (depth_mm < 0).any():
        raise ValueError("深度图必须包含有限、非负的毫米数值")
    camera = np.asarray(K, dtype=np.float64)
    if camera.shape != (3, 3) or not np.isfinite(camera).all() or camera[0, 0] <= 0 or camera[1, 1] <= 0:
        raise ValueError("需要有效的 3x3 彩色相机内参 K")


def target_grasp_keep_mask(
    translations: np.ndarray, mask: np.ndarray, K: np.ndarray, depth_mm: np.ndarray,
    *, depth_tolerance_m: float = 0.04, return_counts: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, int]]:
    """Require each camera-frame grasp center to match target pixels AND depth."""
    if not np.isfinite(depth_tolerance_m) or depth_tolerance_m <= 0:
        raise ValueError("目标深度容差必须是正数（米）")
    target = validate_mask(mask, depth_mm.shape)
    points = np.asarray(translations, dtype=np.float64).reshape(-1, 3)
    keep = np.zeros(len(points), dtype=bool)
    counts = {"mask": 0, "valid_depth": 0, "mask_depth": 0}
    valid = np.isfinite(points).all(axis=1) & (points[:, 2] > 1e-6)
    indices = np.flatnonzero(valid)
    if not len(indices):
        return (keep, counts) if return_counts else keep
    xyz = points[indices]
    u = K[0, 0] * xyz[:, 0] / xyz[:, 2] + K[0, 2]
    v = K[1, 1] * xyz[:, 1] / xyz[:, 2] + K[1, 2]
    h, w = target.shape
    inside = np.isfinite(u) & np.isfinite(v) & (u >= 0) & (u <= w - 1) & (v >= 0) & (v <= h - 1)
    indices = indices[inside]
    cols, rows = np.rint(u[inside]).astype(int), np.rint(v[inside]).astype(int)
    observed_z = depth_mm[rows, cols].astype(np.float64) / 1000.0
    on_mask = target[rows, cols]
    has_depth = on_mask & (observed_z > 0)
    keep[indices] = has_depth & (np.abs(points[indices, 2] - observed_z) <= depth_tolerance_m)
    counts["mask"] = int(np.count_nonzero(on_mask))
    counts["valid_depth"] = int(np.count_nonzero(has_depth))
    counts["mask_depth"] = int(np.count_nonzero(keep))
    return (keep, counts) if return_counts else keep
