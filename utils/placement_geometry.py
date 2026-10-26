"""Depth-aware container geometry and object placement preview helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .target_mask_utils import validate_mask


DISPLAY_FLIP_X = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float64)


@dataclass(frozen=True)
class PlacementPreviewSettings:
    enabled: bool = True
    min_points: int = 96
    interior_depth_quantile: float = 0.50
    min_depth_span_m: float = 0.018
    plane_distance_threshold_m: float = 0.008
    floor_normal_tolerance_deg: float = 25.0
    wall_clearance_m: float = 0.004
    object_bottom_clearance_m: float = 0.005
    container_xy_trim_quantile: float = 0.005
    rim_inset_m: float = 0.006
    object_xy_trim_quantile: float = 0.005
    yaw_step_deg: float = 1.0
    morph_kernel_px: int = 7
    max_visual_points: int = 12000

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "PlacementPreviewSettings":
        parent = cfg.get("vlm_grasp") or {}
        if not isinstance(parent, dict):
            raise ValueError("vlm_grasp 必须是 YAML 映射")
        raw = parent.get("placement_preview") or {}
        if not isinstance(raw, dict):
            raise ValueError("vlm_grasp.placement_preview 必须是 YAML 映射")
        enabled = raw.get("enabled", True)
        if not isinstance(enabled, bool):
            raise ValueError("vlm_grasp.placement_preview.enabled 必须是 true 或 false")
        value = cls(
            enabled=enabled,
            min_points=int(raw.get("min_points", 96)),
            interior_depth_quantile=float(raw.get("interior_depth_quantile", 0.50)),
            min_depth_span_m=float(raw.get("min_depth_span_m", 0.018)),
            plane_distance_threshold_m=float(raw.get("plane_distance_threshold_m", 0.008)),
            floor_normal_tolerance_deg=float(raw.get("floor_normal_tolerance_deg", 25.0)),
            wall_clearance_m=float(raw.get("wall_clearance_m", 0.004)),
            object_bottom_clearance_m=float(raw.get("object_bottom_clearance_m", 0.005)),
            container_xy_trim_quantile=float(raw.get("container_xy_trim_quantile", 0.005)),
            rim_inset_m=float(raw.get("rim_inset_m", 0.006)),
            object_xy_trim_quantile=float(raw.get("object_xy_trim_quantile", 0.005)),
            yaw_step_deg=float(raw.get("yaw_step_deg", 1.0)),
            morph_kernel_px=int(raw.get("morph_kernel_px", 7)),
            max_visual_points=int(raw.get("max_visual_points", 12000)),
        )
        numeric = np.asarray([
            value.min_points, value.interior_depth_quantile, value.min_depth_span_m,
            value.plane_distance_threshold_m, value.floor_normal_tolerance_deg,
            value.wall_clearance_m,
            value.object_bottom_clearance_m, value.container_xy_trim_quantile,
            value.rim_inset_m,
            value.object_xy_trim_quantile, value.yaw_step_deg, value.morph_kernel_px,
            value.max_visual_points,
        ], dtype=np.float64)
        if not np.isfinite(numeric).all():
            raise ValueError("放置预览参数必须是有限数值")
        if value.min_points < 16 or value.max_visual_points < 16:
            raise ValueError("放置几何最少点数和最大可视化点数必须不小于 16")
        if not 0.1 <= value.interior_depth_quantile <= 0.9:
            raise ValueError("interior_depth_quantile 必须在 [0.1,0.9] 内")
        if value.min_depth_span_m < 0 or value.plane_distance_threshold_m <= 0:
            raise ValueError("深度跨度必须非负，平面距离阈值必须为正数")
        if not 0 < value.floor_normal_tolerance_deg <= 60:
            raise ValueError("floor_normal_tolerance_deg 必须在 (0,60] 度内")
        if value.wall_clearance_m < 0 or value.object_bottom_clearance_m < 0:
            raise ValueError("箱壁和物体底部间距不能为负数")
        if not 0 <= value.container_xy_trim_quantile < 0.1:
            raise ValueError("container_xy_trim_quantile 必须在 [0,0.1) 内")
        if value.rim_inset_m < 0:
            raise ValueError("rim_inset_m 不能为负数")
        if not 0 <= value.object_xy_trim_quantile < 0.1:
            raise ValueError("object_xy_trim_quantile 必须在 [0,0.1) 内")
        if not 0 < value.yaw_step_deg <= 45:
            raise ValueError("yaw_step_deg 必须在 (0,45] 度内")
        if value.morph_kernel_px < 1:
            raise ValueError("morph_kernel_px 必须大于 0")
        return value


@dataclass
class ContainerGeometry:
    target_name: str
    relation: str
    container_mask: np.ndarray
    interior_mask: np.ndarray
    container_points_camera: np.ndarray
    interior_points_camera: np.ndarray
    center_camera_m: np.ndarray
    long_axis_camera: np.ndarray
    short_axis_camera: np.ndarray
    normal_camera: np.ndarray
    corners_camera_m: np.ndarray
    inner_length_m: float
    inner_width_m: float
    visible_floor_center_camera_m: np.ndarray
    visible_floor_corners_camera_m: np.ndarray
    visible_floor_length_m: float
    visible_floor_width_m: float
    opening_projection_quantile: float
    support_normal_camera: np.ndarray | None
    floor_support_angle_deg: float | None
    plane_rmse_m: float
    extraction_method: str
    depth_span_m: float
    interior_truncated_by_image: bool
    opening_source: str = "projected_mask"
    rim_point_count: int = 0
    projected_opening_center_camera_m: np.ndarray | None = None
    projected_opening_length_m: float | None = None
    projected_opening_width_m: float | None = None


@dataclass
class PlacementPlan:
    fits: bool
    rotation_about_normal_deg: float
    object_length_m: float
    object_width_m: float
    object_footprint_quantile: float
    clearance_long_m: float
    clearance_short_m: float
    required_clearance_m: float
    object_center_on_plane_camera_m: np.ndarray
    offset_long_m: float
    offset_short_m: float
    gripper_translation_camera_m: np.ndarray
    gripper_rotation_camera: np.ndarray
    planned_object_axis_camera: np.ndarray
    simulated_object_points_camera: np.ndarray
    simulated_outline_camera: np.ndarray
    status: str
    fit_region: str = "opening"


@dataclass
class PlacementAnalysis:
    container: ContainerGeometry
    plan: PlacementPlan | None
    candidate_angles_deg: tuple[float, ...] = ()
    _object_points_camera: np.ndarray | None = None
    _gripper_pose_camera: Any | None = None
    _settings: PlacementPreviewSettings | None = None


def _largest_component(mask: np.ndarray) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if count <= 1:
        return np.zeros_like(mask, dtype=bool)
    label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == label


def _odd_kernel(value: int) -> int:
    value = max(1, int(value))
    return value if value % 2 else value + 1


def _interior_mask(
    container_mask: np.ndarray,
    depth_mm: np.ndarray,
    relation: str,
    settings: PlacementPreviewSettings,
) -> tuple[np.ndarray, str, float]:
    valid = container_mask & (depth_mm > 0)
    depths = depth_mm[valid].astype(np.float64) / 1000.0
    if len(depths) < settings.min_points:
        raise ValueError(f"放置目标有效深度点不足：{len(depths)} < {settings.min_points}")
    q10, q90 = np.quantile(depths, [0.10, 0.90])
    span = float(q90 - q10)
    kernel_size = _odd_kernel(settings.morph_kernel_px)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    # An open container usually exposes a bottom/interior surface farther from
    # the camera than its rim and front wall. Use that depth layer only when
    # the separation is meaningful; shallow plates fall back to an eroded mask.
    if relation == "in" and span >= settings.min_depth_span_m:
        threshold = float(np.quantile(depths, settings.interior_depth_quantile))
        candidate = valid & (depth_mm.astype(np.float64) / 1000.0 >= threshold)
        candidate = cv2.morphologyEx(candidate.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)
        candidate &= valid
        candidate = _largest_component(candidate)
        if int(candidate.sum()) >= settings.min_points:
            return candidate, "depth_quantile", span

    eroded = cv2.erode(container_mask.astype(np.uint8), kernel, iterations=1).astype(bool) & (depth_mm > 0)
    eroded = _largest_component(eroded)
    if int(eroded.sum()) < settings.min_points:
        eroded = _largest_component(valid)
    if int(eroded.sum()) < settings.min_points:
        raise ValueError("放置目标内部轮廓过小，无法估计三维几何")
    return eroded, "eroded_mask", span


def _mask_points(mask: np.ndarray, depth_mm: np.ndarray, K: np.ndarray) -> np.ndarray:
    rows, cols = np.nonzero(mask & (depth_mm > 0))
    if not len(rows):
        return np.empty((0, 3), dtype=np.float64)
    z = depth_mm[rows, cols].astype(np.float64) / 1000.0
    pixels = np.column_stack([cols, rows, np.ones(len(rows), dtype=np.float64)])
    rays = pixels @ np.linalg.inv(np.asarray(K, dtype=np.float64)).T
    return rays * z[:, None]


def _grow_seed_plane_mask(
    container_mask: np.ndarray,
    seed_mask: np.ndarray,
    depth_mm: np.ndarray,
    K: np.ndarray,
    plane_center: np.ndarray,
    plane_normal: np.ndarray,
    settings: PlacementPreviewSettings,
) -> np.ndarray:
    """Recover the connected bottom plane across the full container mask.

    Raw camera depth changes continuously across an oblique box bottom, so a
    depth quantile is suitable only as a seed. Plane-distance growth recovers
    the rest of that surface, while connected-component selection removes
    isolated coplanar pixels that would stretch a minimum-area rectangle.
    """
    valid = container_mask & (depth_mm > 0)
    rows, cols = np.nonzero(valid)
    z = depth_mm[rows, cols].astype(np.float64) / 1000.0
    pixels = np.column_stack([cols, rows, np.ones(len(rows), dtype=np.float64)])
    points = (pixels @ np.linalg.inv(np.asarray(K, dtype=np.float64)).T) * z[:, None]
    distances = np.abs((points - plane_center) @ plane_normal)
    candidate = np.zeros_like(container_mask, dtype=np.uint8)
    on_plane = distances <= settings.plane_distance_threshold_m
    candidate[rows[on_plane], cols[on_plane]] = 1
    kernel_size = _odd_kernel(settings.morph_kernel_px)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, kernel)
    candidate = candidate.astype(bool) & valid

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        candidate.astype(np.uint8), connectivity=8,
    )
    if count <= 1:
        raise ValueError("盒底平面扩张后没有连续区域")
    overlaps = np.asarray([
        int(np.count_nonzero((labels == label) & seed_mask))
        for label in range(1, count)
    ])
    if int(overlaps.max(initial=0)) < settings.min_points:
        raise ValueError("盒底平面扩张结果与初始内部区域不一致")
    selected = 1 + int(np.argmax(overlaps))
    grown = labels == selected
    if int(grown.sum()) < settings.min_points:
        raise ValueError("盒底连续平面点不足")
    return grown


def _downsample(points: np.ndarray, limit: int) -> np.ndarray:
    if len(points) <= limit:
        return points.copy()
    indices = np.linspace(0, len(points) - 1, limit, dtype=np.int64)
    return points[indices].copy()


def _fit_plane_rectangle(
    points: np.ndarray,
    distance_threshold_m: float,
    min_points: int,
    *,
    preferred_normal: np.ndarray | None = None,
    max_normal_deviation_deg: float = 25.0,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    float, float, float, np.ndarray,
]:
    if len(points) < min_points:
        raise ValueError("盒子内部点云不足，无法拟合底面")
    work = _downsample(points, 3000)
    preferred = None
    min_alignment = -1.0
    if preferred_normal is not None:
        preferred = np.asarray(preferred_normal, dtype=np.float64).reshape(3)
        preferred_norm = float(np.linalg.norm(preferred))
        if preferred_norm < 1e-9 or not np.isfinite(preferred).all():
            raise ValueError("参考桌面法线无效")
        preferred /= preferred_norm
        min_alignment = float(np.cos(np.deg2rad(max_normal_deviation_deg)))
    rng = np.random.default_rng(0)
    best_mask = None
    best_score = (-1, -np.inf)
    iterations = min(240, max(80, len(work) // 8))
    for _ in range(iterations):
        sample = work[rng.choice(len(work), 3, replace=False)]
        normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        norm = float(np.linalg.norm(normal))
        if norm < 1e-9:
            continue
        normal /= norm
        if preferred is not None and abs(float(np.dot(normal, preferred))) < min_alignment:
            continue
        distances = np.abs((work - sample[0]) @ normal)
        inliers = distances <= distance_threshold_m
        count = int(inliers.sum())
        residual = float(distances[inliers].mean()) if count else np.inf
        score = (count, -residual)
        if score > best_score:
            best_score, best_mask = score, inliers
    if best_mask is None or int(best_mask.sum()) < min_points:
        if preferred is not None:
            raise ValueError("盒子内部没有与桌面平行的足够平面点")
        raise ValueError("盒子内部没有足够的共面深度点")

    inliers = work[best_mask]
    center = inliers.mean(axis=0)
    _, _, vh = np.linalg.svd(inliers - center, full_matrices=False)
    normal = vh[-1]
    all_distances = np.abs((points - center) @ normal)
    plane_points = points[all_distances <= distance_threshold_m]
    if len(plane_points) < min_points:
        raise ValueError("盒子内部平面拟合后剩余点不足")
    center = plane_points.mean(axis=0)
    _, _, vh = np.linalg.svd(plane_points - center, full_matrices=False)
    normal = vh[-1]
    normal /= np.linalg.norm(normal)
    if preferred is not None and abs(float(np.dot(normal, preferred))) < min_alignment:
        raise ValueError("盒底拟合法线偏离桌面法线过大")
    if np.dot(normal, -center) < 0:
        normal = -normal

    reference = np.array([1.0, 0.0, 0.0])
    basis_u = reference - normal * np.dot(reference, normal)
    if np.linalg.norm(basis_u) < 1e-6:
        reference = np.array([0.0, 1.0, 0.0])
        basis_u = reference - normal * np.dot(reference, normal)
    basis_u /= np.linalg.norm(basis_u)
    basis_v = np.cross(normal, basis_u)
    projected = np.column_stack([(plane_points - center) @ basis_u, (plane_points - center) @ basis_v])
    rect = cv2.minAreaRect(projected.astype(np.float32))
    box = cv2.boxPoints(rect).astype(np.float64)
    edges = np.roll(box, -1, axis=0) - box
    lengths = np.linalg.norm(edges, axis=1)
    edge_index = int(np.argmax(lengths))
    length = float(lengths[edge_index])
    width = float(lengths[(edge_index + 1) % 4])
    if width > length:
        length, width = width, length
        edge_index = (edge_index + 1) % 4
    if width <= 1e-4:
        raise ValueError("盒子内部平面退化为直线，无法计算长短轴")
    direction_2d = edges[edge_index] / np.linalg.norm(edges[edge_index])
    long_axis = basis_u * direction_2d[0] + basis_v * direction_2d[1]
    long_axis /= np.linalg.norm(long_axis)
    short_axis = np.cross(normal, long_axis)
    short_axis /= np.linalg.norm(short_axis)
    rect_center = center + basis_u * float(rect[0][0]) + basis_v * float(rect[0][1])
    corners = np.asarray([
        rect_center - long_axis * length / 2 - short_axis * width / 2,
        rect_center + long_axis * length / 2 - short_axis * width / 2,
        rect_center + long_axis * length / 2 + short_axis * width / 2,
        rect_center - long_axis * length / 2 + short_axis * width / 2,
    ])
    rmse = float(np.sqrt(np.mean(((plane_points - center) @ normal) ** 2)))
    return plane_points, rect_center, long_axis, short_axis, normal, length, width, rmse, corners


def _estimate_support_normal(
    pick_mask: np.ndarray,
    placement_mask: np.ndarray,
    depth_mm: np.ndarray,
    K: np.ndarray,
    settings: PlacementPreviewSettings,
    preferred_normal: np.ndarray | None = None,
) -> np.ndarray | None:
    """Estimate the tabletop normal outside both segmented targets.

    An open box can expose more side-wall pixels than bottom pixels after it
    is rotated in the image.  The surrounding support surface provides the
    orientation cue needed to distinguish those planes without relying on a
    fixed camera pose.
    """
    exclusion = (pick_mask | placement_mask).astype(np.uint8)
    kernel_size = _odd_kernel(max(9, settings.morph_kernel_px * 3))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    exclusion = cv2.dilate(exclusion, kernel, iterations=1).astype(bool)
    support_mask = (depth_mm > 0) & ~exclusion
    support_points = _mask_points(support_mask, depth_mm, K)
    if len(support_points) < settings.min_points:
        return None
    try:
        return _fit_plane_rectangle(
            support_points,
            settings.plane_distance_threshold_m,
            settings.min_points,
            preferred_normal=preferred_normal,
            max_normal_deviation_deg=settings.floor_normal_tolerance_deg,
        )[4]
    except ValueError:
        return None


def estimate_table_normal(
    depth_mm: np.ndarray,
    K: np.ndarray,
    *,
    exclusion_masks: tuple[np.ndarray, ...] | list[np.ndarray] = (),
    settings: PlacementPreviewSettings,
    expected_normal_camera: np.ndarray | None = None,
) -> np.ndarray | None:
    """Estimate the tabletop normal while excluding known foreground masks.

    The returned unit vector is expressed in the camera optical frame and is
    oriented toward the camera.  ``None`` means that the frame does not
    contain enough reliable support-plane depth; robot code must treat that as
    a failed safety check instead of silently accepting every grasp.
    """
    depth = np.asarray(depth_mm)
    if depth.ndim != 2:
        raise ValueError("深度图必须是二维数组")
    camera_matrix = np.asarray(K, dtype=np.float64)
    if camera_matrix.shape != (3, 3) or not np.isfinite(camera_matrix).all():
        raise ValueError("相机内参必须是有限的 3x3 矩阵")
    excluded = np.zeros(depth.shape, dtype=bool)
    for mask in exclusion_masks:
        excluded |= validate_mask(mask, depth.shape)
    empty = np.zeros(depth.shape, dtype=bool)
    expected = None
    if expected_normal_camera is not None:
        expected = np.asarray(expected_normal_camera, dtype=np.float64).reshape(3)
        length = float(np.linalg.norm(expected))
        if length < 1e-9 or not np.isfinite(expected).all():
            raise ValueError("相机坐标系中的预期桌面法线无效")
        expected /= length
    return _estimate_support_normal(
        excluded, empty, depth, camera_matrix, settings, preferred_normal=expected,
    )


def mask_points_camera(mask: np.ndarray, depth_mm: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Return valid masked RGB-D points in the camera optical frame."""
    valid_mask = validate_mask(mask, np.asarray(depth_mm).shape)
    return _mask_points(valid_mask, np.asarray(depth_mm), np.asarray(K, dtype=np.float64))


def _quantile_interval(values: np.ndarray, trim_quantile: float) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("平面投影包含无效数值")
    if trim_quantile == 0:
        return float(values.min()), float(values.max())
    low, high = np.quantile(values, [trim_quantile, 1.0 - trim_quantile])
    return float(low), float(high)


def _projected_opening_rectangle(
    container_points: np.ndarray,
    plane_center: np.ndarray,
    floor_long_axis: np.ndarray,
    floor_short_axis: np.ndarray,
    normal: np.ndarray,
    trim_quantile: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, np.ndarray]:
    """Estimate the full XY opening while retaining the fitted floor plane.

    The front wall can hide a large part of the bottom, so the continuous
    visible floor is only a reliable source for the plane and its axes.  The
    complete container SAM point cloud is projected orthogonally onto that
    plane to recover the opening.  Symmetric quantile trimming prevents a few
    depth outliers from stretching the rectangle.
    """
    relative = np.asarray(container_points, dtype=np.float64) - plane_center
    x = relative @ floor_long_axis
    y = relative @ floor_short_axis
    x0, x1 = _quantile_interval(x, trim_quantile)
    y0, y1 = _quantile_interval(y, trim_quantile)
    size_x, size_y = x1 - x0, y1 - y0
    if min(size_x, size_y) <= 1e-4:
        raise ValueError("容器完整投影退化，无法估计平面开口")
    center = (
        plane_center
        + floor_long_axis * ((x0 + x1) / 2.0)
        + floor_short_axis * ((y0 + y1) / 2.0)
    )
    if size_x >= size_y:
        long_axis = floor_long_axis.copy()
        short_axis = floor_short_axis.copy()
        length, width = size_x, size_y
    else:
        long_axis = floor_short_axis.copy()
        short_axis = np.cross(normal, long_axis)
        short_axis /= np.linalg.norm(short_axis)
        length, width = size_y, size_x
    corners = np.asarray([
        center - long_axis * length / 2 - short_axis * width / 2,
        center + long_axis * length / 2 - short_axis * width / 2,
        center + long_axis * length / 2 + short_axis * width / 2,
        center - long_axis * length / 2 + short_axis * width / 2,
    ])
    return center, long_axis, short_axis, float(length), float(width), corners


def _rim_opening_rectangle(
    container_points: np.ndarray,
    floor_center: np.ndarray,
    floor_long_axis: np.ndarray,
    floor_short_axis: np.ndarray,
    floor_normal: np.ndarray,
    floor_length: float,
    floor_width: float,
    settings: PlacementPreviewSettings,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, np.ndarray, int] | None:
    relative = np.asarray(container_points, dtype=np.float64) - floor_center
    heights = relative @ floor_normal
    upper_height = float(np.quantile(heights, 0.90))
    if upper_height < settings.min_depth_span_m:
        return None
    rim_threshold = max(settings.min_depth_span_m, upper_height * 0.5)
    rim = relative[heights >= rim_threshold]
    if len(rim) < settings.min_points * 4:
        return None
    coordinates = np.column_stack([rim @ floor_long_axis, rim @ floor_short_axis])
    trim = max(0.01, settings.container_xy_trim_quantile)
    low, high = np.quantile(coordinates, [trim, 1.0 - trim], axis=0)
    spans = high - low
    if np.any(spans <= 2 * settings.rim_inset_m + 2 * settings.wall_clearance_m):
        return None
    floor_half = np.array([floor_length, floor_width], dtype=np.float64) / 2.0
    tolerance = max(0.008, 2 * settings.plane_distance_threshold_m)
    if np.any(low > -floor_half + tolerance) or np.any(high < floor_half - tolerance):
        return None
    if np.any(spans > 2.5 * np.array([floor_length, floor_width])):
        return None
    offset_limit = np.maximum(0.05, floor_half)
    if np.any(low < -floor_half - offset_limit) or np.any(high > floor_half + offset_limit):
        return None
    for axis in (0, 1):
        other = 1 - axis
        band = max(0.005, 0.04 * float(spans[axis]))
        for boundary in (low[axis], high[axis]):
            side = coordinates[np.abs(coordinates[:, axis] - boundary) <= band, other]
            if len(side) < settings.min_points:
                return None
            if np.quantile(side, 0.95) - np.quantile(side, 0.05) < 0.4 * spans[other]:
                return None
    low += settings.rim_inset_m
    high -= settings.rim_inset_m
    center = floor_center + floor_long_axis * float((low[0] + high[0]) / 2)
    center += floor_short_axis * float((low[1] + high[1]) / 2)
    length, width = (high - low).tolist()
    corners = np.asarray([
        center - floor_long_axis * length / 2 - floor_short_axis * width / 2,
        center + floor_long_axis * length / 2 - floor_short_axis * width / 2,
        center + floor_long_axis * length / 2 + floor_short_axis * width / 2,
        center - floor_long_axis * length / 2 + floor_short_axis * width / 2,
    ])
    return center, floor_long_axis.copy(), floor_short_axis.copy(), length, width, corners, len(rim)


def _axis_angle_matrix(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    C = 1.0 - c
    return np.array([
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ], dtype=np.float64)


def _principal_axis(points: np.ndarray, normal: np.ndarray) -> np.ndarray:
    _, _, vh = np.linalg.svd(points - points.mean(axis=0), full_matrices=False)
    for candidate in vh:
        projected = candidate - normal * np.dot(candidate, normal)
        if np.linalg.norm(projected) > 1e-6:
            return projected / np.linalg.norm(projected)
    raise ValueError("抓取目标点云无法计算平面内长轴")


def _convex_outline(points: np.ndarray, center: np.ndarray, u: np.ndarray, v: np.ndarray, n: np.ndarray) -> np.ndarray:
    relative = points - center
    xy = np.column_stack([relative @ u, relative @ v]).astype(np.float32)
    hull = cv2.convexHull(xy).reshape(-1, 2).astype(np.float64)
    height = float(np.median(relative @ n))
    return center + hull[:, 0, None] * u + hull[:, 1, None] * v + height * n


def _plan_placement(
    object_points: np.ndarray,
    best_grasp: Any,
    container: ContainerGeometry,
    settings: PlacementPreviewSettings,
    *,
    preferred_angle_deg: float | None = None,
    return_candidates: bool = False,
) -> PlacementPlan | tuple[PlacementPlan, tuple[float, ...]]:
    if len(object_points) < settings.min_points:
        raise ValueError(f"抓取目标有效深度点不足：{len(object_points)} < {settings.min_points}")
    translation = np.asarray(best_grasp.translation, dtype=np.float64).reshape(3)
    rotation = np.asarray(best_grasp.rotation_matrix, dtype=np.float64).reshape(3, 3)
    if not np.isfinite(translation).all() or not np.isfinite(rotation).all():
        raise ValueError("GraspNet 最佳抓取位姿包含无效数值")
    # Build the rigid transform around the object's own point-cloud centroid.
    # A grasp center is allowed to be off-center (and often is for a curved
    # object), so it must not be used as the visual/placement center.
    object_center = object_points.mean(axis=0)
    relative = object_points - object_center
    n = container.normal_camera
    use_fused_rim = container.relation == "in" and getattr(container, "opening_source", None) == "rim_floor_fused"
    use_visible_floor = (
        container.relation == "in" and not use_fused_rim
        and hasattr(container, "visible_floor_corners_camera_m")
    )
    if use_visible_floor:
        floor_corners = container.visible_floor_corners_camera_m
        u = floor_corners[1] - floor_corners[0]
        v = floor_corners[2] - floor_corners[1]
        u /= np.linalg.norm(u)
        v /= np.linalg.norm(v)
        fit_center = container.visible_floor_center_camera_m
        fit_length = container.visible_floor_length_m
        fit_width = container.visible_floor_width_m
    else:
        u, v = container.long_axis_camera, container.short_axis_camera
        fit_center = container.center_camera_m
        fit_length = container.inner_length_m
        fit_width = container.inner_width_m
    candidates = []
    angles = np.arange(-90.0, 90.0 + settings.yaw_step_deg * 0.25, settings.yaw_step_deg)
    for angle_deg in angles:
        delta = _axis_angle_matrix(n, np.deg2rad(angle_deg))
        rotated = relative @ delta.T
        x, y = rotated @ u, rotated @ v
        x0, x1 = _quantile_interval(x, settings.object_xy_trim_quantile)
        y0, y1 = _quantile_interval(y, settings.object_xy_trim_quantile)
        object_length = x1 - x0
        object_width = y1 - y0
        clearance_long = (fit_length - object_length) / 2.0
        clearance_short = (fit_width - object_width) / 2.0
        score = (min(clearance_long, clearance_short), clearance_long + clearance_short, -abs(float(angle_deg)))
        candidates.append((score, angle_deg, delta, rotated, object_length, object_width,
                           clearance_long, clearance_short, x0, x1, y0, y1))
    ordered = sorted(candidates, key=lambda item: item[0], reverse=True)
    feasible_angles = tuple(
        float(item[1])
        for item in ordered
        if (container.relation in ("in", "on")
            and item[6] >= settings.wall_clearance_m - 1e-9
            and item[7] >= settings.wall_clearance_m - 1e-9)
    )
    selected = (
        ordered[0]
        if preferred_angle_deg is None
        else min(ordered, key=lambda item: abs(float(item[1]) - float(preferred_angle_deg)))
    )
    (_, angle_deg, delta, rotated, object_length, object_width,
     clearance_long, clearance_short, x0, x1, y0, y1) = selected
    x, y, h = rotated @ u, rotated @ v, rotated @ n
    robust_xy = (
        (x >= x0) & (x <= x1)
        & (y >= y0) & (y <= y1)
    )
    if int(robust_xy.sum()) < 3:
        robust_xy = np.ones(len(rotated), dtype=bool)

    # Center the robust footprint, which maximizes its minimum distance to
    # opposing walls.  Centering the point-mass centroid can push a curved or
    # asymmetric object such as a banana unnecessarily close to one wall.
    offset_x = -float((x0 + x1) / 2.0)
    offset_y = -float((y0 + y1) / 2.0)
    # Height is used only to make the point-cloud preview sit on the fitted
    # plane.  It is deliberately excluded from the fit decision: placement
    # planning in this stage consists only of X, Y and yaw.
    robust_bottom = float(np.quantile(h, settings.object_xy_trim_quantile))
    object_center_on_plane = (
        fit_center
        + u * offset_x
        + v * offset_y
    )
    shift = object_center_on_plane + n * float(settings.object_bottom_clearance_m - robust_bottom)
    simulated = rotated + shift
    placed_x0, placed_x1 = x0 + offset_x, x1 + offset_x
    placed_y0, placed_y1 = y0 + offset_y, y1 + offset_y
    clearance_long = min(
        float(placed_x0 + fit_length / 2.0),
        float(fit_length / 2.0 - placed_x1),
    )
    clearance_short = min(
        float(placed_y0 + fit_width / 2.0),
        float(fit_width / 2.0 - placed_y1),
    )
    # A mask touching the image boundary means part of the container is not
    # observed; it does not by itself make the visible interior unsafe.  The
    # fitted rectangle then represents a conservative observed region, and
    # the object is centered inside that region.  Keep the truncation flag in
    # the result, but accept the plan when the complete robust object footprint
    # still satisfies the requested clearance inside the observed geometry.
    fits = bool(
        container.relation in ("in", "on")
        and clearance_long >= settings.wall_clearance_m - 1e-9
        and clearance_short >= settings.wall_clearance_m - 1e-9
    )
    planned_axis = delta @ _principal_axis(object_points, n)
    planned_axis -= n * np.dot(planned_axis, n)
    planned_axis /= np.linalg.norm(planned_axis)
    simulated_footprint = simulated[robust_xy]
    outline = _convex_outline(simulated_footprint, fit_center, u, v, n)
    if container.relation not in ("in", "on"):
        status = f"无可用放置位姿：关系 {container.relation!r} 不是放入或放在表面"
    elif fits and use_fused_rim:
        status = "平面 X/Y/Yaw 轮廓在盒缘与可见盒底融合区域内满足安全间距（未验证夹爪与箱壁碰撞）"
    elif fits and use_visible_floor and container.interior_truncated_by_image:
        status = "平面 X/Y/Yaw 轮廓在已观测容器区域内满足安全间距（容器部分超出画面）"
    elif fits and use_visible_floor:
        status = "平面 X/Y/Yaw 轮廓在可见盒底区域内满足安全间距"
    elif fits:
        status = "平面 X/Y/Yaw 轮廓满足容器内壁安全间距"
    elif use_fused_rim:
        status = "无可用放置位姿：物体轮廓超过盒缘与盒底融合后的安全区域"
    elif use_visible_floor:
        status = "无可用放置位姿：物体轮廓超过可见盒底安全区域；请调整相机或盒子以观察完整内部"
    elif container.interior_truncated_by_image:
        status = "无可用放置位姿：物体轮廓超过已观测容器区域的安全空间"
    else:
        status = "无可用放置位姿：平面轮廓超过容器开口安全空间"
    plan = PlacementPlan(
        fits=fits,
        rotation_about_normal_deg=float(angle_deg),
        object_length_m=object_length,
        object_width_m=object_width,
        object_footprint_quantile=settings.object_xy_trim_quantile,
        clearance_long_m=clearance_long,
        clearance_short_m=clearance_short,
        required_clearance_m=settings.wall_clearance_m,
        object_center_on_plane_camera_m=object_center_on_plane,
        offset_long_m=offset_x,
        offset_short_m=offset_y,
        gripper_translation_camera_m=(
            delta @ (translation - object_center) + shift
        ),
        gripper_rotation_camera=delta @ rotation,
        planned_object_axis_camera=planned_axis,
        simulated_object_points_camera=_downsample(simulated_footprint, settings.max_visual_points),
        simulated_outline_camera=outline,
        status=status,
        fit_region=("rim_floor_fused" if use_fused_rim else "visible_floor" if use_visible_floor else "opening"),
    )
    return (plan, feasible_angles) if return_candidates else plan


def select_placement_plan(
    analysis: PlacementAnalysis,
    angle_deg: float,
) -> PlacementPlan:
    """Materialize one geometrically valid yaw candidate for robot IK.

    The preview plan maximizes wall clearance, but that single wrist yaw may
    be unreachable even when another equally valid in-container yaw is
    reachable.  The robot path uses this function to test the feasible yaw
    set and then keeps the selected plan for preview and saved metadata.
    """
    if (analysis._object_points_camera is None or analysis._gripper_pose_camera is None
            or analysis._settings is None):
        raise ValueError("放置分析缺少候选重建数据")
    plan, _ = _plan_placement(
        analysis._object_points_camera,
        analysis._gripper_pose_camera,
        analysis.container,
        analysis._settings,
        preferred_angle_deg=float(angle_deg),
        return_candidates=True,
    )
    analysis.plan = plan
    return plan


def analyze_placement_geometry(
    pick_mask: np.ndarray,
    placement_mask: np.ndarray,
    depth_mm: np.ndarray,
    K: np.ndarray,
    *,
    best_grasp: Any | None,
    target_name: str,
    relation: str,
    settings: PlacementPreviewSettings,
    object_points_camera: np.ndarray | None = None,
    support_exclusion_mask: np.ndarray | None = None,
) -> PlacementAnalysis:
    """Estimate container interior and a preview-only object placement pose."""
    image_shape = depth_mm.shape
    # After the robot has lifted the object, its current image mask is
    # intentionally empty.  In that path ``object_points_camera`` contains
    # the saved object cloud transported with the measured TCP pose, so the
    # empty mask is only an exclusion-mask placeholder and is valid.
    pick_mask = validate_mask(
        pick_mask,
        image_shape,
        allow_empty=object_points_camera is not None,
    )
    placement_mask = validate_mask(placement_mask, image_shape)
    support_pick_mask = (
        pick_mask
        if support_exclusion_mask is None
        else validate_mask(support_exclusion_mask, image_shape)
    )
    support_normal = _estimate_support_normal(
        support_pick_mask, placement_mask, depth_mm, K, settings,
    )
    container_points = _mask_points(placement_mask, depth_mm, K)
    if support_normal is not None and relation in ("in", "on"):
        valid_depths = depth_mm[placement_mask & (depth_mm > 0)].astype(np.float64) / 1000.0
        if len(valid_depths) < settings.min_points:
            raise ValueError(
                f"放置目标有效深度点不足：{len(valid_depths)} < {settings.min_points}"
            )
        q10, q90 = np.quantile(valid_depths, [0.10, 0.90])
        depth_span = float(q90 - q10)
        # Find any sufficiently large plane parallel to the surrounding table
        # in the full container cloud.  Its initial rectangle can include
        # disconnected coplanar patches; connected plane growth below keeps
        # only the physical box-bottom component.
        (_, rough_center, _, _, rough_normal,
         _, _, _, _) = _fit_plane_rectangle(
            container_points,
            settings.plane_distance_threshold_m,
            settings.min_points,
            preferred_normal=support_normal,
            max_normal_deviation_deg=settings.floor_normal_tolerance_deg,
        )
        interior_mask = _grow_seed_plane_mask(
            placement_mask,
            placement_mask,
            depth_mm,
            K,
            rough_center,
            rough_normal,
            settings,
        )
        interior_points = _mask_points(interior_mask, depth_mm, K)
        (plane_points, floor_center, floor_long_axis, floor_short_axis, normal,
         floor_length, floor_width, rmse, floor_corners) = _fit_plane_rectangle(
            interior_points,
            settings.plane_distance_threshold_m,
            settings.min_points,
            preferred_normal=support_normal,
            max_normal_deviation_deg=settings.floor_normal_tolerance_deg,
        )
        method = "support_normal+plane_grow"
    else:
        interior_mask, method, depth_span = _interior_mask(
            placement_mask, depth_mm, relation, settings,
        )
        interior_points = _mask_points(interior_mask, depth_mm, K)
        (plane_points, floor_center, floor_long_axis, floor_short_axis, normal,
         floor_length, floor_width, rmse, floor_corners) = _fit_plane_rectangle(
            interior_points,
            settings.plane_distance_threshold_m,
            settings.min_points,
        )
        if method == "depth_quantile":
            interior_mask = _grow_seed_plane_mask(
                placement_mask,
                interior_mask,
                depth_mm,
                K,
                floor_center,
                normal,
                settings,
            )
            interior_points = _mask_points(interior_mask, depth_mm, K)
            (plane_points, floor_center, floor_long_axis, floor_short_axis, normal,
             floor_length, floor_width, rmse, floor_corners) = _fit_plane_rectangle(
                interior_points,
                settings.plane_distance_threshold_m,
                settings.min_points,
            )
            method = "depth_quantile+plane_grow"
    floor_support_angle = None
    if support_normal is not None:
        alignment = float(np.clip(abs(np.dot(normal, support_normal)), 0.0, 1.0))
        floor_support_angle = float(np.rad2deg(np.arccos(alignment)))
    (opening_center, opening_long_axis, opening_short_axis,
     opening_length, opening_width, opening_corners) = _projected_opening_rectangle(
        container_points,
        floor_center,
        floor_long_axis,
        floor_short_axis,
        normal,
        settings.container_xy_trim_quantile,
    )
    projected_center = opening_center.copy()
    projected_length, projected_width = opening_length, opening_width
    opening_source = "projected_mask"
    rim_point_count = 0
    if relation == "in":
        rim_opening = _rim_opening_rectangle(
            container_points, floor_center, floor_long_axis, floor_short_axis,
            normal, floor_length, floor_width, settings,
        )
        if rim_opening is not None:
            (opening_center, opening_long_axis, opening_short_axis,
             opening_length, opening_width, opening_corners, rim_point_count) = rim_opening
            opening_source = "rim_floor_fused"
    container = ContainerGeometry(
        target_name=target_name,
        relation=relation,
        container_mask=placement_mask,
        interior_mask=interior_mask,
        container_points_camera=_downsample(container_points, settings.max_visual_points),
        interior_points_camera=_downsample(plane_points, settings.max_visual_points),
        center_camera_m=opening_center,
        long_axis_camera=opening_long_axis,
        short_axis_camera=opening_short_axis,
        normal_camera=normal,
        corners_camera_m=opening_corners,
        inner_length_m=opening_length,
        inner_width_m=opening_width,
        visible_floor_center_camera_m=floor_center,
        visible_floor_corners_camera_m=floor_corners,
        visible_floor_length_m=floor_length,
        visible_floor_width_m=floor_width,
        opening_projection_quantile=settings.container_xy_trim_quantile,
        support_normal_camera=support_normal,
        floor_support_angle_deg=floor_support_angle,
        plane_rmse_m=rmse,
        extraction_method=method,
        depth_span_m=depth_span,
        interior_truncated_by_image=bool(
            placement_mask[0].any() or placement_mask[-1].any()
            or placement_mask[:, 0].any() or placement_mask[:, -1].any()
        ),
        opening_source=opening_source,
        rim_point_count=rim_point_count,
        projected_opening_center_camera_m=projected_center,
        projected_opening_length_m=projected_length,
        projected_opening_width_m=projected_width,
    )
    plan = None
    candidate_angles: tuple[float, ...] = ()
    planning_points = None
    if best_grasp is not None:
        if object_points_camera is None:
            object_points = _mask_points(pick_mask, depth_mm, K)
        else:
            object_points = np.asarray(object_points_camera, dtype=np.float64).reshape(-1, 3)
            if not len(object_points) or not np.isfinite(object_points).all():
                raise ValueError("抓取物点云为空或包含无效数值")
        plan, candidate_angles = _plan_placement(
            object_points, best_grasp, container, settings, return_candidates=True,
        )
        planning_points = object_points.copy()
    return PlacementAnalysis(
        container,
        plan,
        candidate_angles,
        planning_points,
        best_grasp,
        settings if best_grasp is not None else None,
    )


def analyze_placement_geometry_from_points(
    object_points_camera: np.ndarray,
    placement_mask: np.ndarray,
    depth_mm: np.ndarray,
    K: np.ndarray,
    *,
    gripper_pose_camera: Any,
    target_name: str,
    relation: str,
    settings: PlacementPreviewSettings,
    support_exclusion_mask: np.ndarray | None = None,
) -> PlacementAnalysis:
    """Plan placement after lift using the already grasped object's point cloud.

    A second camera image may no longer contain the picked object, or may show
    another similar object on the table.  Passing the carried point cloud and
    current gripper pose keeps the placement plan tied to the object that was
    actually grasped while using the new frame only for container geometry.
    """
    empty_pick_mask = np.zeros(np.asarray(depth_mm).shape, dtype=bool)
    return analyze_placement_geometry(
        empty_pick_mask,
        placement_mask,
        depth_mm,
        K,
        best_grasp=gripper_pose_camera,
        target_name=target_name,
        relation=relation,
        settings=settings,
        object_points_camera=object_points_camera,
        support_exclusion_mask=support_exclusion_mask,
    )


def placement_geometry_record(analysis: PlacementAnalysis | None, error: str | None = None) -> dict[str, Any]:
    if analysis is None:
        return {"available": False, "error": error or "placement geometry unavailable"}
    container = analysis.container
    record: dict[str, Any] = {
        "available": True,
        "error": error,
        "coordinate_frame": "camera_optical",
        "length_unit": "m",
        "container": {
            "target_name": container.target_name,
            "relation": container.relation,
            "center_m": container.center_camera_m.tolist(),
            "long_axis": container.long_axis_camera.tolist(),
            "short_axis": container.short_axis_camera.tolist(),
            "normal": container.normal_camera.tolist(),
            "opening_corners_m": container.corners_camera_m.tolist(),
            "opening_length_m": container.inner_length_m,
            "opening_width_m": container.inner_width_m,
            # Keep the former keys for consumers written before the opening
            # and visible-floor measurements were separated.
            "inner_corners_m": container.corners_camera_m.tolist(),
            "inner_length_m": container.inner_length_m,
            "inner_width_m": container.inner_width_m,
            "opening_projection_trim_quantile": container.opening_projection_quantile,
            "opening_source": container.opening_source,
            "rim_point_count": container.rim_point_count,
            "projected_mask_center_m": (
                None if container.projected_opening_center_camera_m is None
                else container.projected_opening_center_camera_m.tolist()
            ),
            "projected_mask_length_m": container.projected_opening_length_m,
            "projected_mask_width_m": container.projected_opening_width_m,
            "visible_floor_center_m": container.visible_floor_center_camera_m.tolist(),
            "visible_floor_corners_m": container.visible_floor_corners_camera_m.tolist(),
            "visible_floor_length_m": container.visible_floor_length_m,
            "visible_floor_width_m": container.visible_floor_width_m,
            "support_normal": (
                None if container.support_normal_camera is None
                else container.support_normal_camera.tolist()
            ),
            "floor_support_angle_deg": container.floor_support_angle_deg,
            "plane_rmse_m": container.plane_rmse_m,
            "sam_area_px": int(container.container_mask.sum()),
            "interior_area_px": int(container.interior_mask.sum()),
            "extraction_method": container.extraction_method,
            "depth_span_m": container.depth_span_m,
            "interior_truncated_by_image": container.interior_truncated_by_image,
        },
        "plan": None,
        "validation_scope": {
            "xy_translation_yaw_and_wall_clearance": True,
            "object_footprint_and_wall_clearance": True,
            "partial_container_uses_observed_geometry": True,
            "container_depth_or_wall_height": False,
            "object_height_or_thickness": False,
            "gripper_wall_collision": False,
            "robot_reachability": False,
            "robot_motion_executed": False,
        },
    }
    if analysis.plan is not None:
        plan = analysis.plan
        record["plan"] = {
            "fits": plan.fits,
            "container_visibility": (
                "partial_at_image_boundary"
                if container.interior_truncated_by_image else "complete_in_image"
            ),
            "uses_observed_container_geometry": bool(container.interior_truncated_by_image),
            "rotation_about_container_normal_deg": plan.rotation_about_normal_deg,
            "object_length_m": plan.object_length_m,
            "object_width_m": plan.object_width_m,
            "object_footprint_trim_quantile": plan.object_footprint_quantile,
            "clearance_long_m": plan.clearance_long_m,
            "clearance_short_m": plan.clearance_short_m,
            "required_wall_clearance_m": plan.required_clearance_m,
            "planned_dofs": ["x", "y", "yaw"],
            "object_center_on_opening_plane_m": plan.object_center_on_plane_camera_m.tolist(),
            "offset_along_opening_long_axis_m": plan.offset_long_m,
            "offset_along_opening_short_axis_m": plan.offset_short_m,
            "gripper_translation_is_preview_only": True,
            "gripper_translation_m": plan.gripper_translation_camera_m.tolist(),
            "gripper_rotation_matrix": plan.gripper_rotation_camera.tolist(),
            "planned_object_axis": plan.planned_object_axis_camera.tolist(),
            "status": plan.status,
            "fit_region": plan.fit_region,
        }
    return record


def _project(points: np.ndarray, K: np.ndarray) -> np.ndarray | None:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if not len(points) or not np.isfinite(points).all() or (points[:, 2] <= 1e-6).any():
        return None
    pixels = points @ np.asarray(K, dtype=np.float64).T
    return np.rint(pixels[:, :2] / pixels[:, 2:3]).astype(np.int32)


def draw_placement_preview(
    image: np.ndarray,
    placement_mask: np.ndarray,
    analysis: PlacementAnalysis | None,
    K: np.ndarray,
) -> np.ndarray:
    """Draw only the container's SAM contour.

    The caller draws the VLM detection rectangle. Detailed geometry belongs
    in the dedicated Open3D preview and JSON output; drawing it over the RGB
    image made the container itself hard to inspect.
    """
    display = image.copy()
    mask = np.asarray(placement_mask, dtype=bool)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(display, contours, -1, (0, 140, 255), 2)
    return display


class Open3DPlacementWindow:
    """Container frame and simulated object placement in camera coordinates."""

    def __init__(self, title: str = "VLM Placement Geometry Preview", *, visible: bool = True) -> None:
        import open3d as o3d

        self._o3d = o3d
        self._vis = o3d.visualization.Visualizer()
        if not self._vis.create_window(title, width=1280, height=720, visible=visible):
            self._vis.destroy_window()
            raise RuntimeError("Open3D placement visualizer window could not be created")
        self._geometries = []

    def _point_cloud(self, points: np.ndarray, color: tuple[float, float, float]):
        cloud = self._o3d.geometry.PointCloud()
        cloud.points = self._o3d.utility.Vector3dVector(points)
        cloud.paint_uniform_color(color)
        return cloud

    def _line_set(self, points: np.ndarray, closed: bool, color: tuple[float, float, float]):
        count = len(points)
        lines = [[i, i + 1] for i in range(count - 1)]
        if closed and count > 2:
            lines.append([count - 1, 0])
        item = self._o3d.geometry.LineSet()
        item.points = self._o3d.utility.Vector3dVector(points)
        item.lines = self._o3d.utility.Vector2iVector(lines)
        item.colors = self._o3d.utility.Vector3dVector(np.tile(color, (len(lines), 1)))
        return item

    def update(self, analysis: PlacementAnalysis) -> None:
        for geometry in self._geometries:
            self._vis.remove_geometry(geometry, reset_bounding_box=False)
        container, plan = analysis.container, analysis.plan
        use_floor = container.relation == "in" and container.opening_source != "rim_floor_fused"
        planning_corners = (
            container.visible_floor_corners_camera_m if use_floor else container.corners_camera_m
        )
        planning_center = (
            container.visible_floor_center_camera_m if use_floor else container.center_camera_m
        )
        planning_long_axis = (
            (planning_corners[1] - planning_corners[0]) / container.visible_floor_length_m
            if use_floor else container.long_axis_camera
        )
        planning_short_axis = (
            (planning_corners[2] - planning_corners[1]) / container.visible_floor_width_m
            if use_floor else container.short_axis_camera
        )
        geometries = [
            self._point_cloud(container.container_points_camera, (0.85, 0.45, 0.05)),
            self._point_cloud(container.interior_points_camera, (1.0, 0.85, 0.0)),
            self._line_set(planning_corners, True, (1.0, 1.0, 0.0)),
        ]
        if container.relation == "in" and not use_floor:
            geometries.append(self._line_set(
                container.visible_floor_corners_camera_m, True, (0.0, 0.8, 1.0),
            ))
        frame = self._o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=max(0.04, min(0.12, (
                container.visible_floor_width_m if use_floor else container.inner_width_m
            ) * 0.7)),
        )
        frame_pose = np.eye(4)
        frame_pose[:3, :3] = np.column_stack([
            planning_long_axis, planning_short_axis, container.normal_camera,
        ])
        frame_pose[:3, 3] = planning_center
        frame.transform(frame_pose)
        geometries.append(frame)
        if plan is not None and plan.fits:
            color = (0.1, 0.9, 0.2)
            geometries.extend([
                self._point_cloud(plan.simulated_object_points_camera, color),
                self._line_set(plan.simulated_outline_camera, True, color),
            ])
        for index, geometry in enumerate(geometries):
            geometry.transform(DISPLAY_FLIP_X)
            self._vis.add_geometry(
                geometry,
                reset_bounding_box=index == 0,
            )
        # Recompute this view for every snapshot because the container can
        # move between instructions. View along the container normal; the
        # former arbitrary angle made a centered object look like it sat on a
        # side wall.
        display_rotation = DISPLAY_FLIP_X[:3, :3]
        view = self._vis.get_view_control()
        view.set_lookat((display_rotation @ planning_center).tolist())
        view.set_front((display_rotation @ container.normal_camera).tolist())
        view.set_up((display_rotation @ planning_long_axis).tolist())
        view.set_zoom(0.72)
        self._geometries = geometries
        self.poll()

    def poll(self) -> bool:
        alive = self._vis.poll_events()
        self._vis.update_renderer()
        return alive

    def clear(self) -> None:
        for geometry in self._geometries:
            self._vis.remove_geometry(geometry, reset_bounding_box=False)
        self._geometries = []
        self.poll()

    def capture(self, path: str | Path) -> None:
        self._vis.capture_screen_image(str(path), do_render=True)

    def frame(self) -> np.ndarray:
        """Return the current Open3D view as an OpenCV BGR image."""
        self.poll()
        rgb = np.asarray(self._vis.capture_screen_float_buffer(do_render=True))
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.size == 0:
            raise RuntimeError("Open3D placement renderer returned an empty frame")
        return cv2.cvtColor(
            np.clip(np.rint(rgb * 255.0), 0, 255).astype(np.uint8),
            cv2.COLOR_RGB2BGR,
        )

    def rotate(self, dx: float, dy: float) -> None:
        self._vis.get_view_control().rotate(float(dx), float(dy))

    def pan(self, dx: float, dy: float) -> None:
        self._vis.get_view_control().translate(float(dx), float(dy))

    def zoom(self, wheel_steps: float) -> None:
        self._vis.get_view_control().scale(0.9 ** float(wheel_steps))

    def close(self) -> None:
        self._vis.destroy_window()
