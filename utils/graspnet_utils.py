"""GraspNet helpers for RGB-D frames, YOLO, and explicit target boxes/masks."""

from __future__ import annotations

import os
import sys
from pathlib import Path
import time
from dataclasses import dataclass
from typing import Any, Optional

import cv2
import numpy as np
import open3d as o3d
import torch

from utils.target_mask_utils import target_grasp_keep_mask, validate_bbox, validate_mask, validate_rgbd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts/truetype")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GRASPNET_ROOT = PROJECT_ROOT / "sdk" / "graspnet-baseline"
DEFAULT_NUM_VIEW = 300
DEFAULT_VOXEL_SIZE = 0.01
DEFAULT_WARMUP_FRAMES = 20
DISPLAY_FLIP_X = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float64)


def prepare_graspnet_imports(graspnet_root: Path = GRASPNET_ROOT) -> None:
    for subdir in ("models", "dataset", "utils", "pointnet2", "graspnetAPI"):
        path = str(graspnet_root / subdir)
        if path not in sys.path:
            sys.path.insert(0, path)
    root = str(graspnet_root)
    if root not in sys.path:
        sys.path.insert(0, root)


prepare_graspnet_imports()

try:
    from .yolo_utils import YoloDetection
except ImportError:  
    from yolo_utils import YoloDetection

from collision_detector import ModelFreeCollisionDetector  # noqa: E402
from data_utils import CameraInfo, create_point_cloud_from_depth_image  # noqa: E402
from graspnet import GraspNet, pred_decode  # noqa: E402
from graspnetAPI import Grasp, GraspGroup  # noqa: E402

try:
    from .transforms import canonicalize_parallel_gripper_tcp_rotation, graspnet_rotation_to_rebot_tcp_rotation, transform_grasp_pose_to_base_with_retreat
except ImportError:
    from transforms import canonicalize_parallel_gripper_tcp_rotation, graspnet_rotation_to_rebot_tcp_rotation, transform_grasp_pose_to_base_with_retreat


@dataclass
class GraspNetFrameResult:
    grasps: GraspGroup
    pre_bbox_grasps: GraspGroup
    bbox_grasps: GraspGroup
    best: Optional[Grasp]
    status: str
    target_status: str
    detections: list[YoloDetection]
    selected_target: Optional[YoloDetection]
    o3d_cloud: o3d.geometry.PointCloud
    raw_cloud: np.ndarray
    filter_counts: Optional[dict[str, int]] = None
    raw_grasps: Optional[GraspGroup] = None
    pre_width_grasps: Optional[GraspGroup] = None


class Open3DGraspWindow:
    def __init__(self, title: str, top_k: Optional[int], *, apply_nms: bool = True,
                 visible: bool = True, focus_radius_m: Optional[float] = None) -> None:
        self._top_k = top_k
        self._apply_nms = apply_nms
        self._focus_radius_m = focus_radius_m
        if focus_radius_m is not None and focus_radius_m <= 0:
            raise ValueError("focus_radius_m must be positive")
        self._vis = o3d.visualization.Visualizer()
        if not self._vis.create_window(title, width=1280, height=720, visible=visible):
            self._vis.destroy_window()
            raise RuntimeError("Open3D visualizer window could not be created")
        self._geometries = []
        self._initialized = False

    def _focus_cloud(self, cloud: o3d.geometry.PointCloud, grasps: GraspGroup,
                     context_points: Optional[np.ndarray] = None):
        """Keep the grasp and container neighborhood without distant scene points."""
        if self._focus_radius_m is None or len(grasps) == 0 or not cloud.has_points():
            return o3d.geometry.PointCloud(cloud), None
        translations = np.asarray(grasps.translations, dtype=np.float64)
        translations = translations[np.isfinite(translations).all(axis=1)]
        if len(translations) == 0:
            return o3d.geometry.PointCloud(cloud), None
        center = np.median(translations, axis=0)
        spread = np.quantile(np.abs(translations - center), 0.95, axis=0)
        # The configured radius is the minimum half-size.  A dispersed set of
        # candidates may expand it, but extreme candidates cannot pull the
        # camera back to the scale of the complete RGB-D scene.
        half_extent = np.clip(
            np.maximum(self._focus_radius_m, spread + 0.06),
            self._focus_radius_m,
            self._focus_radius_m * 2.0,
        )
        lower = center - half_extent
        upper = center + half_extent
        if context_points is not None:
            context = np.asarray(context_points, dtype=np.float64).reshape(-1, 3)
            context = context[np.isfinite(context).all(axis=1)]
            if len(context) > 0:
                # Container geometry can contain a few edge-depth outliers.
                # Robust bounds retain the box while preventing those points
                # from shrinking the useful scene again.
                quantiles = (0.01, 0.99) if len(context) >= 100 else (0.0, 1.0)
                context_lower = np.quantile(context, quantiles[0], axis=0) - 0.03
                context_upper = np.quantile(context, quantiles[1], axis=0) + 0.03
                lower = np.minimum(lower, context_lower)
                upper = np.maximum(upper, context_upper)
        focus_center = (lower + upper) * 0.5
        points = np.asarray(cloud.points)
        keep = np.isfinite(points).all(axis=1) & np.all((points >= lower) & (points <= upper), axis=1)
        indices = np.flatnonzero(keep)
        if len(indices) < 16:
            return o3d.geometry.PointCloud(cloud), focus_center
        return cloud.select_by_index(indices.tolist()), focus_center

    def update(self, cloud: o3d.geometry.PointCloud, grasps: GraspGroup,
               *, context_points: Optional[np.ndarray] = None) -> None:
        for geom in self._geometries:
            self._vis.remove_geometry(geom, reset_bounding_box=False)
        self._geometries = []

        focus_center = None
        if len(grasps) > 0:
            grasps_vis = GraspGroup(grasps.grasp_group_array.copy())
            if self._apply_nms:
                try:
                    grasps_vis = grasps_vis.nms()
                except Exception as exc:
                    print(f"Grasp NMS skipped: {exc}")
            grasps_vis.sort_by_score()
            if self._top_k is not None:
                grasps_vis = grasps_vis[: self._top_k]
            cloud_vis, focus_center = self._focus_cloud(cloud, grasps_vis, context_points)
            grasps_vis.transform(DISPLAY_FLIP_X)
        else:
            grasps_vis = None
            cloud_vis = o3d.geometry.PointCloud(cloud)
        cloud_vis.transform(DISPLAY_FLIP_X)
        geometries = [cloud_vis]
        if grasps_vis is not None:
            geometries.extend(grasps_vis.to_open3d_geometry_list())

        for index, geom in enumerate(geometries):
            # Every inference may contain a different object and depth range.
            # Refit once per new result; mouse interaction after this update is
            # preserved until the next inference.
            self._vis.add_geometry(geom, reset_bounding_box=index == 0)
        if focus_center is not None:
            display_center = DISPLAY_FLIP_X[:3, :3] @ focus_center
            view = self._vis.get_view_control()
            view.set_lookat(display_center.tolist())
            # Open3D uses smaller zoom values for a closer view.  This fills
            # the dashboard tile while retaining enough local table context.
            view.set_zoom(0.60)
        self._geometries = geometries
        self._initialized = True
        self.poll()

    def poll(self) -> bool:
        alive = self._vis.poll_events()
        self._vis.update_renderer()
        return alive

    def clear(self) -> None:
        for geom in self._geometries:
            self._vis.remove_geometry(geom, reset_bounding_box=False)
        self._geometries = []
        self.poll()

    def capture(self, path: str | Path) -> None:
        self._vis.capture_screen_image(str(path), do_render=True)

    def frame(self) -> np.ndarray:
        """Return the current Open3D view as an OpenCV BGR image."""
        self.poll()
        rgb = np.asarray(self._vis.capture_screen_float_buffer(do_render=True))
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.size == 0:
            raise RuntimeError("Open3D grasp renderer returned an empty frame")
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


def copy_grasp_group(grasps: GraspGroup) -> GraspGroup:
    return GraspGroup(grasps.grasp_group_array.copy())


def visualization_grasps(result: GraspNetFrameResult, mode: str) -> GraspGroup:
    """Return the grasp set used for Open3D diagnostics."""
    if mode == "raw":
        return result.raw_grasps if result.raw_grasps is not None else GraspGroup()
    if mode == "best":
        if result.best is None:
            return GraspGroup()
        return GraspGroup(result.best.grasp_array.reshape(1, -1).copy())
    if mode == "pre-bbox":
        return result.pre_bbox_grasps
    if mode == "bbox":
        return result.bbox_grasps
    if mode == "width":
        return result.grasps
    return result.grasps


def resolve_checkpoint_path(
    checkpoint: str,
    *,
    project_root: Path = PROJECT_ROOT,
    graspnet_root: Path = GRASPNET_ROOT,
) -> Path:
    checkpoint_path = Path(str(checkpoint)).expanduser()
    if checkpoint_path.is_absolute():
        return checkpoint_path
    if len(checkpoint_path.parts) > 1:
        return project_root / checkpoint_path
    return graspnet_root / "checkpoints" / checkpoint_path


def build_net(checkpoint_path: str | Path, num_view: int = DEFAULT_NUM_VIEW) -> GraspNet:
    checkpoint_path = resolve_checkpoint_path(str(checkpoint_path))
    if not torch.cuda.is_available():
        raise RuntimeError("GraspNet pointnet2 operators require CUDA, but torch.cuda is unavailable.")

    net = GraspNet(
        input_feature_dim=0,
        num_view=num_view,
        num_angle=12,
        num_depth=4,
        cylinder_radius=0.05,
        hmin=-0.02,
        hmax_list=[0.01, 0.02, 0.03, 0.04],
        is_training=False,
    )
    device = torch.device("cuda:0")
    net.to(device)

    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    net.load_state_dict(checkpoint["model_state_dict"])
    net.eval()
    print(f"Loaded checkpoint {checkpoint_path} (epoch: {checkpoint['epoch']})")
    return net


def _expanded_bbox_pixel_bounds(
    bbox_xyxy: tuple[int, int, int, int],
    expand_ratio: float,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    """Pixel bounds of a bbox expanded about its center, clipped to the image."""
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
    expand_ratio = max(1.0, float(expand_ratio))
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    half_w = 0.5 * max(1.0, x2 - x1) * expand_ratio
    half_h = 0.5 * max(1.0, y2 - y1) * expand_ratio
    x1 = int(np.clip(int(round(cx - half_w)), 0, max(0, width - 1)))
    y1 = int(np.clip(int(round(cy - half_h)), 0, max(0, height - 1)))
    x2 = int(np.clip(int(round(cx + half_w)), 0, max(0, width - 1)))
    y2 = int(np.clip(int(round(cy + half_h)), 0, max(0, height - 1)))
    return x1, y1, x2, y2


def build_end_points(
    color_bgr: np.ndarray,
    depth_mm: np.ndarray,
    K: np.ndarray,
    num_point: int,
    min_depth_m: float,
    max_depth_m: float,
    bbox_xyxy: Optional[tuple[int, int, int, int]] = None,
    expand_ratio: float = 1.0,
    target_mask: Optional[np.ndarray] = None,
    strict_target: bool = False,
    min_target_points: int = 16,
    target_context_expand_ratio: Optional[float] = None,
) -> tuple[dict, o3d.geometry.PointCloud, np.ndarray]:
    if target_mask is not None or strict_target:
        validate_rgbd(color_bgr, depth_mm, K)
    if color_bgr.shape[:2] != depth_mm.shape[:2]:
        depth_mm = cv2.resize(depth_mm, (color_bgr.shape[1], color_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)

    min_mm = int(max(0.0, min_depth_m) * 1000.0)
    max_mm = int(max_depth_m * 1000.0)
    depth = depth_mm.astype(np.uint16, copy=False)
    mask = (depth > min_mm) & (depth < max_mm)
    if int(mask.sum()) == 0:
        raise RuntimeError("No valid depth pixels in the configured depth range.")

    # Concentrate network input points on the target mask or bbox when available;
    # collision detection / visualization still use the full masked scene cloud.
    sample_mask = mask
    if target_mask is not None:
        sample_mask = mask & validate_mask(target_mask, depth.shape)
        if int(sample_mask.sum()) < min_target_points:
            raise ValueError("目标掩码内有效深度点不足，本轮停止，不回退到整个场景")
        if target_context_expand_ratio is not None:
            if not np.isfinite(target_context_expand_ratio) or target_context_expand_ratio < 1.0:
                raise ValueError("目标点云上下文扩展比例必须为不小于 1 的有限数值")
            # GraspNet needs nearby table/object geometry to predict approach
            # directions. Keep the target validity gate above, sample its local
            # context, and enforce exact mask membership on decoded grasps.
            context_bbox = bbox_xyxy
            if context_bbox is None:
                rows, cols = np.nonzero(target_mask)
                context_bbox = (cols.min(), rows.min(), cols.max(), rows.max())
            x1, y1, x2, y2 = _expanded_bbox_pixel_bounds(
                context_bbox, target_context_expand_ratio, depth.shape[1], depth.shape[0],
            )
            context_mask = np.zeros_like(mask)
            context_mask[y1:y2 + 1, x1:x2 + 1] = True
            sample_mask = mask & context_mask
    elif bbox_xyxy is not None:
        x1, y1, x2, y2 = _expanded_bbox_pixel_bounds(bbox_xyxy, expand_ratio, depth.shape[1], depth.shape[0])
        bbox_mask = np.zeros_like(mask)
        bbox_mask[y1 : y2 + 1, x1 : x2 + 1] = True
        cropped = mask & bbox_mask
        required_points = min_target_points if strict_target else 16
        if int(cropped.sum()) >= required_points:
            sample_mask = cropped
        elif strict_target:
            raise ValueError("目标框内有效深度点不足，本轮停止，不回退到整个场景")

    color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    h, w = depth.shape
    camera = CameraInfo(w, h, float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2]), 1000.0)
    cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)
    cloud_masked = cloud[sample_mask]
    color_masked = color_rgb[sample_mask]

    if len(cloud_masked) >= num_point:
        idxs = np.random.choice(len(cloud_masked), num_point, replace=False)
    else:
        idxs = np.concatenate(
            [
                np.arange(len(cloud_masked)),
                np.random.choice(len(cloud_masked), num_point - len(cloud_masked), replace=True),
            ],
            axis=0,
        )

    end_points = {
        "point_clouds": torch.from_numpy(cloud_masked[idxs].astype(np.float32)[np.newaxis]).cuda(non_blocking=True),
        "cloud_colors": color_masked[idxs],
    }

    o3d_cloud = o3d.geometry.PointCloud()
    o3d_cloud.points = o3d.utility.Vector3dVector(cloud[mask].astype(np.float32))
    o3d_cloud.colors = o3d.utility.Vector3dVector(color_rgb[mask].astype(np.float32))
    return end_points, o3d_cloud, cloud[mask]


def infer_grasps(
    net: GraspNet,
    end_points: dict,
    raw_cloud: np.ndarray,
    collision_thresh: float,
    voxel_size: float = DEFAULT_VOXEL_SIZE,
) -> tuple[GraspGroup, dict[str, int], GraspGroup]:
    with torch.no_grad():
        end_points = net(end_points)
        grasp_preds = pred_decode(end_points)

    gg = GraspGroup(grasp_preds[0].detach().cpu().numpy())
    raw_grasps = copy_grasp_group(gg)
    decoded_count = len(raw_grasps)
    collision_removed = 0
    if len(gg) > 0 and collision_thresh > 0:
        detector = ModelFreeCollisionDetector(raw_cloud, voxel_size=voxel_size)
        collision_mask = detector.detect(gg, approach_dist=0.05, collision_thresh=collision_thresh)
        collision_removed = int(np.count_nonzero(collision_mask))
        gg = gg[~collision_mask]

    return gg, {
        "decoded": decoded_count,
        "pre_collision": decoded_count,
        "collision_removed": collision_removed,
        "final": len(gg),
    }, raw_grasps


def filter_grasps_by_bbox(
    grasps: GraspGroup,
    bbox_xyxy: tuple[int, int, int, int],
    K: np.ndarray,
    *,
    margin_px: int = 0,
    expand_ratio: float = 1.0,
    image_shape: Optional[tuple[int, int]] = None,
) -> GraspGroup:
    if len(grasps) == 0:
        return grasps

    translations = np.asarray(grasps.translations, dtype=np.float64)
    z = translations[:, 2]
    valid_z = z > 1e-6
    u = np.full(len(grasps), np.nan, dtype=np.float64)
    v = np.full(len(grasps), np.nan, dtype=np.float64)
    u[valid_z] = float(K[0, 0]) * translations[valid_z, 0] / z[valid_z] + float(K[0, 2])
    v[valid_z] = float(K[1, 1]) * translations[valid_z, 1] / z[valid_z] + float(K[1, 2])

    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
    expand_ratio = max(1.0, float(expand_ratio))
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    half_w = 0.5 * max(1.0, x2 - x1) * expand_ratio
    half_h = 0.5 * max(1.0, y2 - y1) * expand_ratio
    x1 = int(round(cx - half_w)) - int(margin_px)
    y1 = int(round(cy - half_h)) - int(margin_px)
    x2 = int(round(cx + half_w)) + int(margin_px)
    y2 = int(round(cy + half_h)) + int(margin_px)
    if image_shape is not None:
        h, w = image_shape
        x1 = int(np.clip(x1, 0, max(0, w - 1)))
        x2 = int(np.clip(x2, 0, max(0, w - 1)))
        y1 = int(np.clip(y1, 0, max(0, h - 1)))
        y2 = int(np.clip(y2, 0, max(0, h - 1)))

    keep = valid_z & (u >= x1) & (u <= x2) & (v >= y1) & (v <= y2)
    return grasps[keep]


def filter_grasps_by_width(grasps: GraspGroup, max_width_m: Optional[float]) -> GraspGroup:
    if max_width_m is None or len(grasps) == 0:
        return grasps
    return grasps[np.asarray(grasps.widths, dtype=np.float64) <= float(max_width_m)]


def approach_tilt_angles_deg(grasps: GraspGroup, table_normal_camera: np.ndarray) -> np.ndarray:
    """Angle from the into-table approach to the camera-facing tabletop normal."""
    normal = np.asarray(table_normal_camera, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(normal))
    if norm < 1e-9 or not np.isfinite(normal).all():
        raise ValueError("桌面法线为空或包含无效数值")
    if len(grasps) == 0:
        return np.empty(0, dtype=np.float64)
    normal /= norm
    rotations = np.asarray(grasps.rotation_matrices, dtype=np.float64).reshape(-1, 3, 3)
    approaches = rotations[:, :, 0]
    lengths = np.linalg.norm(approaches, axis=1)
    valid = np.isfinite(approaches).all(axis=1) & (lengths > 1e-9)
    angles = np.full(len(approaches), np.inf, dtype=np.float64)
    dots = -((approaches[valid] / lengths[valid, None]) @ normal)
    angles[valid] = np.rad2deg(np.arccos(np.clip(dots, -1.0, 1.0)))
    return angles


def filter_grasps_by_approach_tilt(
    grasps: GraspGroup,
    table_normal_camera: np.ndarray,
    max_tilt_deg: float,
) -> tuple[GraspGroup, np.ndarray]:
    """Reject side and reverse approaches before robot-frame and IK selection."""
    limit = float(max_tilt_deg)
    if not np.isfinite(limit) or not 0.0 <= limit < 90.0:
        raise ValueError("max_approach_tilt_deg 必须在 [0,90) 度内")
    angles = approach_tilt_angles_deg(grasps, table_normal_camera)
    return grasps[np.isfinite(angles) & (angles <= limit + 1e-9)], angles


def ready_rotation_angles_deg(
    grasps: GraspGroup,
    T_camera_to_base: np.ndarray,
    ready_rotation_base: np.ndarray,
) -> np.ndarray:
    """Measure each executed TCP orientation against ready in SO(3)."""
    camera_rotation = np.asarray(T_camera_to_base, dtype=np.float64)[:3, :3]
    ready_rotation = np.asarray(ready_rotation_base, dtype=np.float64).reshape(3, 3)
    if not np.isfinite(camera_rotation).all() or not np.isfinite(ready_rotation).all():
        raise ValueError("相机或 ready 姿态包含无效数值")
    angles = np.full(len(grasps), np.inf, dtype=np.float64)
    for index, rotation in enumerate(np.asarray(grasps.rotation_matrices, dtype=np.float64).reshape(-1, 3, 3)):
        if not np.isfinite(rotation).all():
            continue
        tcp_rotation = canonicalize_parallel_gripper_tcp_rotation(
            camera_rotation @ graspnet_rotation_to_rebot_tcp_rotation(rotation)
        )
        relative = ready_rotation.T @ tcp_rotation
        angles[index] = np.rad2deg(np.arccos(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)))
    return angles


def filter_grasps_by_ready_rotation(
    grasps: GraspGroup,
    T_camera_to_base: np.ndarray,
    ready_rotation_base: np.ndarray,
    max_rotation_deg: float,
) -> tuple[GraspGroup, np.ndarray]:
    limit = float(max_rotation_deg)
    if not np.isfinite(limit) or not 0.0 < limit <= 180.0:
        raise ValueError("max_ready_rotation_deg 必须在 (0,180] 度内")
    angles = ready_rotation_angles_deg(grasps, T_camera_to_base, ready_rotation_base)
    return grasps[np.isfinite(angles) & (angles <= limit + 1e-9)], angles


def filter_grasps_by_mask(
    grasps: GraspGroup, target_mask: np.ndarray, K: np.ndarray, depth_mm: np.ndarray,
    *, depth_tolerance_m: float = 0.04, counts: Optional[dict[str, int]] = None,
) -> GraspGroup:
    if counts is None:
        keep = target_grasp_keep_mask(
            grasps.translations, target_mask, K, depth_mm, depth_tolerance_m=depth_tolerance_m,
        )
    else:
        keep, stage_counts = target_grasp_keep_mask(
            grasps.translations, target_mask, K, depth_mm,
            depth_tolerance_m=depth_tolerance_m, return_counts=True,
        )
        counts.update(stage_counts)
    return grasps[keep]


def select_best_grasp(grasps: GraspGroup) -> Optional[Grasp]:
    if len(grasps) == 0:
        return None
    ranked = GraspGroup(grasps.grasp_group_array.copy())
    try:
        ranked = ranked.nms()
    except Exception as exc:
        print(f"[WARN] GraspNet NMS skipped: {exc}")
    ranked.sort_by_score()
    return ranked[0] if len(ranked) > 0 else None


def select_target(detections: list[YoloDetection], target_class: Optional[str]) -> Optional[YoloDetection]:
    if not detections:
        return None
    candidates = detections
    if target_class:
        target_norm = target_class.casefold()
        exact = [target for target in detections if target.class_name.casefold() == target_norm]
        contains = [target for target in detections if target_norm in target.class_name.casefold()]
        candidates = exact or contains
    if not candidates:
        return None
    return max(candidates, key=lambda target: target.conf)


def selected_target_text(selected: Optional[YoloDetection], target_class: Optional[str]) -> str:
    if selected is None:
        return f"target={target_class or 'best'} not found"
    return f"target={selected.class_name} {selected.conf:.2f}"


def target_status_text(selected: Optional[YoloDetection], detections: list[YoloDetection], target_class: Optional[str]) -> str:
    if selected is not None:
        return f"target={selected.class_name} {selected.conf:.2f} detections={len(detections)}"
    if target_class:
        return f"target={target_class} not found detections={len(detections)}"
    return f"target not found detections={len(detections)}"


def draw_detections_overlay(
    frame: np.ndarray,
    detections: list[YoloDetection],
    selected: Optional[YoloDetection],
    target_class: Optional[str],
) -> np.ndarray:
    display = frame.copy()
    selected_key = None
    if selected is not None:
        selected_key = (selected.result_index, selected.detection_index)
    for target in detections:
        is_selected = selected_key == (target.result_index, target.detection_index)
        color = (0, 255, 80) if is_selected else (0, 185, 255)
        thickness = 3 if is_selected else 2
        x1, y1, x2, y2 = target.bbox_xyxy
        cv2.rectangle(display, (x1, y1), (x2, y2), color, thickness)
        label = f"{target.class_name} {target.conf:.2f}"
        if target_class and is_selected:
            label = f"TARGET {label}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        bg_y1 = max(0, y1 - label_size[1] - 8)
        cv2.rectangle(display, (x1, bg_y1), (x1 + label_size[0] + 8, y1), (0, 0, 0), -1)
        cv2.putText(display, label, (x1 + 4, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    return display


def draw_best_grasp_projection(display: np.ndarray, grasp: Optional[Grasp], K: np.ndarray) -> None:
    if grasp is None:
        return
    x, y, z = [float(v) for v in grasp.translation]
    if z <= 1e-6:
        return
    u = int(round(float(K[0, 0]) * x / z + float(K[0, 2])))
    v = int(round(float(K[1, 1]) * y / z + float(K[1, 2])))
    if 0 <= u < display.shape[1] and 0 <= v < display.shape[0]:
        cv2.drawMarker(display, (u, v), (0, 0, 255), cv2.MARKER_CROSS, 22, 2, cv2.LINE_AA)
        label = f"best score={grasp.score:.2f} width={grasp.width * 100:.1f}cm"
        cv2.putText(display, label, (u + 10, max(24, v - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)


def grasp_to_base_poses(
    grasp: Grasp,
    T_cam2base: np.ndarray,
    pregrasp_offset_m: float,
    retreat_offset_m: float,
    insertion_depth_m: float = 0.0,
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    return transform_grasp_pose_to_base_with_retreat(
        np.asarray(grasp.translation, dtype=np.float64),
        graspnet_rotation_to_rebot_tcp_rotation(grasp.rotation_matrix),
        T_cam2base,
        pregrasp_offset_m,
        retreat_offset_m,
        insertion_depth_m,
    )


def draw_status(
    frame: np.ndarray,
    status: str,
    target_status: str = "",
    frozen: bool = False,
    title: str = "GraspNet Full-Scene Demo",
) -> np.ndarray:
    display = frame.copy()
    lines = [
        title,
        "G/SPACE: infer   R: resume   Q/ESC: quit",
    ]
    if target_status:
        lines.append(target_status)
    lines.append(status)
    y = 28
    for line in lines:
        cv2.putText(display, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(display, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
        y += 26
    if frozen:
        cv2.putText(display, "[FROZEN]", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 215, 255), 2)
    return display


def infer_frame(
    net: GraspNet,
    color_bgr: np.ndarray,
    depth_mm: np.ndarray,
    K: np.ndarray,
    *,
    num_point: int,
    min_depth: float,
    max_depth: float,
    collision_thresh: float,
    voxel_size: float = DEFAULT_VOXEL_SIZE,
    yolo_model: Optional[Any] = None,
    yolo_opts: Optional[dict[str, Any]] = None,
    target_class: Optional[str] = None,
    target_margin_px: int = 0,
    target_expand_ratio: float = 1.0,
    max_grasp_width_m: Optional[float] = None,
    yolo_detections: Optional[list[YoloDetection]] = None,
    target_bbox_xyxy: Optional[tuple[int, int, int, int]] = None,
    target_mask: Optional[np.ndarray] = None,
    target_name: Optional[str] = None,
    target_depth_tolerance_m: float = 0.04,
    min_target_points: int = 16,
    target_context_expand_ratio: Optional[float] = None,
) -> GraspNetFrameResult:
    try:
        from .yolo_utils import detect_objects
    except ImportError:
        from yolo_utils import detect_objects

    detections: list[YoloDetection] = []
    selected_target: Optional[YoloDetection] = None
    target_label = "full scene"
    explicit_target = target_bbox_xyxy is not None or target_mask is not None
    if explicit_target:
        if min_target_points < 16 or num_point < 16:
            raise ValueError("外部目标的有效深度点和 GraspNet 采样点至少为 16")
        if not np.isfinite(target_depth_tolerance_m) or target_depth_tolerance_m <= 0:
            raise ValueError("目标深度容差必须是正数（米）")
        if yolo_model is not None or yolo_detections is not None:
            raise ValueError("外部目标框/掩码与 YOLO 目标选择不能同时使用")
        validate_rgbd(color_bgr, depth_mm, K)
        if target_bbox_xyxy is not None:
            target_bbox_xyxy = validate_bbox(target_bbox_xyxy, color_bgr.shape[:2])
        if target_mask is not None:
            target_mask = validate_mask(target_mask, color_bgr.shape[:2]).copy()
            if target_bbox_xyxy is not None:
                x1, y1, x2, y2 = target_bbox_xyxy
                bounded = np.zeros_like(target_mask)
                bounded[y1:y2 + 1, x1:x2 + 1] = target_mask[y1:y2 + 1, x1:x2 + 1]
                target_mask = validate_mask(bounded, color_bgr.shape[:2])
        target_label = target_name or "external target"
    if yolo_model is not None:
        if yolo_detections is None:
            _, detections = detect_objects(yolo_model, color_bgr, yolo_opts or {})
        else:
            detections = yolo_detections
        selected_target = select_target(detections, target_class)
        if selected_target is None:
            target_status = target_status_text(selected_target, detections, target_class)
            empty = GraspGroup()
            empty_cloud = o3d.geometry.PointCloud()
            return GraspNetFrameResult(
                grasps=empty,
                pre_bbox_grasps=empty,
                bbox_grasps=empty,
                best=None,
                status=f"inference skipped: {target_status}",
                target_status=target_status,
                detections=detections,
                selected_target=None,
                o3d_cloud=empty_cloud,
                raw_cloud=np.empty((0, 3), dtype=np.float32),
            )
        target_label = f"{selected_target.class_name} {selected_target.conf:.2f}"

    tic = time.time()
    crop_bbox = target_bbox_xyxy if explicit_target else (selected_target.bbox_xyxy if selected_target is not None else None)
    target_options = {}
    if explicit_target:
        target_options = dict(target_mask=target_mask, strict_target=True, min_target_points=min_target_points)
        if target_context_expand_ratio is not None:
            target_options["target_context_expand_ratio"] = target_context_expand_ratio
    end_points, o3d_cloud, raw_cloud = build_end_points(
        color_bgr,
        depth_mm,
        K,
        num_point,
        min_depth,
        max_depth,
        bbox_xyxy=crop_bbox,
        expand_ratio=target_expand_ratio,
        **target_options,
    )
    grasps, counts, raw_grasps = infer_grasps(net, end_points, raw_cloud, collision_thresh, voxel_size)
    pre_bbox_grasps = copy_grasp_group(grasps)
    if selected_target is not None:
        grasps = filter_grasps_by_bbox(
            grasps,
            selected_target.bbox_xyxy,
            K,
            margin_px=target_margin_px,
            expand_ratio=target_expand_ratio,
            image_shape=color_bgr.shape[:2],
        )
    bbox_grasps = copy_grasp_group(grasps)
    mask_counts = {}
    if explicit_target:
        # External target membership is exact; YOLO's expansion/margin are not
        # allowed to admit another object outside the VLM-selected target.
        if target_bbox_xyxy is not None:
            grasps = filter_grasps_by_bbox(grasps, target_bbox_xyxy, K, image_shape=color_bgr.shape[:2])
        bbox_grasps = copy_grasp_group(grasps)
        if target_mask is not None:
            grasps = filter_grasps_by_mask(
                grasps, target_mask, K, depth_mm, depth_tolerance_m=target_depth_tolerance_m,
                counts=mask_counts,
            )
    mask_count = len(grasps)
    pre_width_grasps = copy_grasp_group(grasps)
    grasps = filter_grasps_by_width(grasps, max_grasp_width_m)
    width_count = len(grasps)
    best = select_best_grasp(grasps)
    elapsed = time.time() - tic

    if explicit_target:
        status = (
            f"target grasps={len(grasps)} bbox={len(bbox_grasps)} scene={len(pre_bbox_grasps)} "
            f"mask={mask_counts.get('mask', 0)} valid_depth={mask_counts.get('valid_depth', 0)} "
            f"mask_depth={mask_count} decoded={counts['decoded']} collide={counts['collision_removed']} inference={elapsed:.2f}s"
        )
        target_status = f"external mask target: {target_label}" if target_mask is not None else f"external bbox target: {target_label}"
    elif yolo_model is None:
        status = f"grasps={len(grasps)} decoded={counts['decoded']} inference={elapsed:.2f}s"
        target_status = "YOLO disabled: full-scene GraspNet"
    else:
        status = (
            f"{target_label} grasps={len(grasps)}/{len(bbox_grasps)}/{len(pre_bbox_grasps)} decoded={counts['decoded']} "
            f"collide={counts['collision_removed']}/{counts['pre_collision']} inference={elapsed:.2f}s"
        )
        target_status = target_status_text(selected_target, detections, target_class)

    filter_counts = {
        "decoded": counts["decoded"], "collision_free": len(pre_bbox_grasps),
        "bbox": len(bbox_grasps), **mask_counts, "mask_depth": mask_count,
        "width": width_count, "final": len(grasps),
    }
    return GraspNetFrameResult(
        grasps=grasps,
        pre_bbox_grasps=pre_bbox_grasps,
        bbox_grasps=bbox_grasps,
        best=best,
        status=status,
        target_status=target_status,
        detections=detections,
        selected_target=selected_target,
        o3d_cloud=o3d_cloud,
        raw_cloud=raw_cloud,
        filter_counts=filter_counts,
        raw_grasps=raw_grasps,
        pre_width_grasps=pre_width_grasps,
    )
