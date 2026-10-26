"""Image-space quality ranking for elongated grasp targets."""

from __future__ import annotations

import cv2
import numpy as np

from .target_mask_utils import validate_mask


def rank_elongated_grasp_centers(
    translations: np.ndarray,
    scores: np.ndarray,
    target_mask: np.ndarray,
    camera_matrix: np.ndarray,
    *,
    min_clearance_m: float,
    end_fraction: float,
    approach_tilt_deg: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return safe indices in preference order, longitudinal positions, and edge clearance."""
    points = np.asarray(translations, dtype=np.float64).reshape(-1, 3)
    network_scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if len(points) != len(network_scores):
        raise ValueError("抓取候选坐标与分数数量不一致")
    tilt_angles = (np.zeros(len(points), dtype=np.float64) if approach_tilt_deg is None
                   else np.asarray(approach_tilt_deg, dtype=np.float64).reshape(-1))
    if len(tilt_angles) != len(points):
        raise ValueError("抓取候选倾角与坐标数量不一致")
    if not np.isfinite(min_clearance_m) or min_clearance_m < 0:
        raise ValueError("目标掩码边缘安全距离必须非负")
    if not np.isfinite(end_fraction) or not 0 <= end_fraction < 0.5:
        raise ValueError("目标端部排除比例必须在 [0, 0.5) 内")
    camera = np.asarray(camera_matrix, dtype=np.float64)
    if camera.shape != (3, 3) or not np.isfinite(camera).all() or camera[0, 0] <= 0 or camera[1, 1] <= 0:
        raise ValueError("需要有效的相机内参计算抓取位置")
    if np.asarray(target_mask).ndim != 2:
        raise ValueError("香蕉目标掩码必须是二维图像")
    mask = validate_mask(target_mask, np.asarray(target_mask).shape)
    rows, cols = np.nonzero(mask)
    pixels = np.column_stack((cols, rows)).astype(np.float64)
    if len(pixels) < 3:
        raise ValueError("香蕉掩码面积不足，无法可靠区分中段和头尾")
    center = pixels.mean(axis=0)
    eigenvalues, eigenvectors = np.linalg.eigh(np.cov((pixels - center).T))
    if eigenvalues[-1] < 2 * max(eigenvalues[0], 1e-9):
        raise ValueError("香蕉掩码不呈细长形，无法可靠区分中段和头尾")
    longitudinal_axis = eigenvectors[:, -1]
    lower, upper = np.quantile((pixels - center) @ longitudinal_axis, (0.02, 0.98))
    if upper - lower < 1:
        raise ValueError("香蕉掩码长度不足，无法可靠区分中段和头尾")

    fractions = np.full(len(points), np.nan, dtype=np.float64)
    clearances_m = np.zeros(len(points), dtype=np.float64)
    inside_target = np.zeros(len(points), dtype=bool)
    valid = np.isfinite(points).all(axis=1) & np.isfinite(network_scores) & (points[:, 2] > 0)
    indices = np.flatnonzero(valid)
    if len(indices):
        xyz = points[indices]
        projected_u = camera[0, 0] * xyz[:, 0] / xyz[:, 2] + camera[0, 2]
        projected_v = camera[1, 1] * xyz[:, 1] / xyz[:, 2] + camera[1, 2]
        columns = np.rint(projected_u).astype(int)
        image_rows = np.rint(projected_v).astype(int)
        in_image = ((columns >= 0) & (columns < mask.shape[1])
                    & (image_rows >= 0) & (image_rows < mask.shape[0]))
        indices = indices[in_image]
        columns, image_rows = columns[in_image], image_rows[in_image]
        inside_target[indices] = mask[image_rows, columns]
        projected = np.column_stack((projected_u[in_image], projected_v[in_image]))
        fractions[indices] = ((projected - center) @ longitudinal_axis - lower) / (upper - lower)
        distance_pixels = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
        clearances_m[indices] = (
            distance_pixels[image_rows, columns] * points[indices, 2]
            / np.sqrt(camera[0, 0] * camera[1, 1])
        )

    safe = (inside_target & np.isfinite(tilt_angles) & np.isfinite(fractions) & (fractions >= end_fraction)
            & (fractions <= 1 - end_fraction) & (clearances_m >= min_clearance_m))
    safe_indices = np.flatnonzero(safe)
    centrality = np.clip(1 - 2 * np.abs(fractions[safe_indices] - 0.5), 0, 1)
    interiority = np.clip(clearances_m[safe_indices] / 0.02, 0, 1)
    quality = (network_scores[safe_indices] + 0.25 * centrality + 0.10 * interiority
               - 0.20 * np.clip(tilt_angles[safe_indices] / 40.0, 0, 1))
    ranked = safe_indices[np.argsort(-quality, kind="stable")]
    if len(ranked):
        comparable_score = network_scores[ranked] >= 0.55 * network_scores[ranked].max()
        middle = (fractions[ranked] >= 0.30) & (fractions[ranked] <= 0.70)
        preferred = comparable_score & middle
        ranked = np.concatenate((ranked[preferred], ranked[~preferred]))
    return ranked, fractions, clearances_m
