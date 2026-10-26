"""Same-snapshot VLM, dual SAM masks, GraspNet, and placement geometry preview."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import numpy as np

from . import placement_geometry as default_placement_backend
from .placement_geometry import PlacementAnalysis, PlacementPreviewSettings
from .sam_utils import SAMSegmenter, SAMSettings, SegmentedTarget
from .target_mask_utils import validate_rgbd
from .vlm_utils import TargetGrounding, VLMClient


@dataclass
class VLMGraspResult:
    grounding: TargetGrounding
    segmentation: SegmentedTarget | None
    grasp_result: Any | None
    elapsed_s: float
    camera_matrix: np.ndarray
    placement_segmentation: SegmentedTarget | None = None
    placement_analysis: PlacementAnalysis | None = None
    placement_error: str | None = None
    rejected_container_segmentation: SegmentedTarget | None = None
    rejected_container_error: str | None = None
    placement_only: bool = False


class VLMGraspPipeline:
    def __init__(self, cfg, client: VLMClient, *, segmenter=None, net=None, grasp_backend=None,
                 placement_backend=None) -> None:
        if grasp_backend is None:
            from . import graspnet_utils as grasp_backend
        self.backend = grasp_backend
        self.client = client
        raw = cfg.get("graspnet") or {}
        target = cfg.get("vlm_grasp") or {}
        self.options = dict(
            num_point=int(raw.get("num_point", 20000)),
            min_depth=float(raw.get("min_depth", 0.05)),
            max_depth=float(raw.get("max_depth", 1.0)),
            collision_thresh=float(raw.get("collision_thresh", 0.01)),
            target_depth_tolerance_m=float(target.get("depth_tolerance_m", 0.04)),
            min_target_points=int(target.get("min_target_points", 32)),
            max_grasp_width_m=float(target.get("max_grasp_width_m", 0.09)),
            target_context_expand_ratio=float(target.get("context_expand_ratio", raw.get("target_expand_ratio", 1.35))),
        )
        values = np.array(list(self.options.values()), dtype=float)
        if not np.isfinite(values).all() or self.options["min_target_points"] < 16 or self.options["num_point"] < 16:
            raise ValueError("GraspNet 采样点至少 16，目标有效点至少 16，参数必须为有限数值")
        if not 0 <= self.options["min_depth"] < self.options["max_depth"] <= 65.535:
            raise ValueError("GraspNet 深度范围必须满足 0<=min_depth<max_depth<=65.535 米")
        if (self.options["target_depth_tolerance_m"] <= 0 or self.options["max_grasp_width_m"] <= 0
                or self.options["collision_thresh"] <= 0):
            raise ValueError("目标深度容差、夹爪宽度和碰撞检测阈值必须为正数")
        if self.options["target_context_expand_ratio"] < 1:
            raise ValueError("vlm_grasp.context_expand_ratio 必须不小于 1")
        self.segmenter = segmenter if segmenter is not None else SAMSegmenter(SAMSettings.from_config(cfg))
        self.net = net if net is not None else grasp_backend.build_net(raw.get("checkpoint", "checkpoint-rs.tar"))
        self.placement_backend = placement_backend or default_placement_backend
        self.placement_settings = PlacementPreviewSettings.from_config(cfg)

    def process(self, instruction, color, depth_mm, K) -> VLMGraspResult:
        validate_rgbd(color, depth_mm, K)
        # Own all three arrays so no camera buffer can mutate them between stages.
        color, depth_mm, K = color.copy(), depth_mm.copy(), np.asarray(K, dtype=np.float64).copy()
        tic = time.monotonic()
        print("[阶段 1/4] VLM 正在选择抓取和放置目标……", flush=True)
        grounding = self.client.ground(instruction, color)
        if grounding.image_size != (color.shape[1], color.shape[0]):
            raise ValueError("VLM 目标框的原图尺寸与本轮 RGB-D 快照不一致")
        placement = grounding.placement
        rejected_container_segmentation = None
        rejected_container_error = None
        rejected = placement.rejected_container if placement is not None else None
        if rejected is not None:
            try:
                print(f"[阶段 2/4] SAM 正在分割不可用容器（仅供预览）：{rejected.target_name}，框={rejected.bbox_xyxy}",
                      flush=True)
                rejected_container_segmentation = self.segmenter.segment(color, rejected.bbox_xyxy)
            except ValueError as exc:
                rejected_container_error = f"不可用容器分割失败：{exc}"
                print(f"[SAM] {rejected_container_error}；抓取预览继续。", flush=True)
        if not grounding.found:
            return VLMGraspResult(
                grounding, None, None, time.monotonic() - tic, K,
                rejected_container_segmentation=rejected_container_segmentation,
                rejected_container_error=rejected_container_error,
            )
        print(f"[阶段 2/4] SAM 正在分割抓取目标：{grounding.target_name}，框={grounding.bbox_xyxy}", flush=True)
        segmentation = self.segmenter.segment(color, grounding.bbox_xyxy)
        placement_segmentation = None
        placement_error = None
        if (self.placement_settings.enabled and placement is not None and placement.required
                and placement.found and placement.bbox_xyxy is not None):
            try:
                print(f"[阶段 2/4] SAM 正在分割放置目标：{placement.target_name}，框={placement.bbox_xyxy}",
                      flush=True)
                placement_segmentation = self.segmenter.segment(color, placement.bbox_xyxy)
            except ValueError as exc:
                placement_error = f"放置目标分割失败：{exc}"
                print(f"[放置几何] {placement_error}；抓取预览继续。", flush=True)
        print(f"[阶段 3/4] GraspNet 正在生成目标候选，掩码面积={segmentation.area_px} 像素……", flush=True)
        grasp_result = self.backend.infer_frame(
            self.net, color, depth_mm, K,
            target_bbox_xyxy=grounding.bbox_xyxy, target_mask=segmentation.mask,
            target_name=grounding.target_name,
            **self.options,
        )
        placement_analysis = None
        if placement_segmentation is not None:
            try:
                print("[阶段 4/4] 正在估计容器内部几何并规划预览放置方向……", flush=True)
                placement_analysis = self.placement_backend.analyze_placement_geometry(
                    segmentation.mask,
                    placement_segmentation.mask,
                    depth_mm,
                    K,
                    best_grasp=grasp_result.best,
                    target_name=placement.target_name,
                    relation=placement.relation or "in",
                    settings=self.placement_settings,
                )
            except ValueError as exc:
                placement_error = f"放置几何分析失败：{exc}"
                print(f"[放置几何] {placement_error}；抓取预览继续。", flush=True)
        return VLMGraspResult(
            grounding, segmentation, grasp_result, time.monotonic() - tic, K,
            placement_segmentation, placement_analysis, placement_error,
            rejected_container_segmentation, rejected_container_error,
        )

    def recheck_placement(
        self,
        instruction,
        color,
        depth_mm,
        K,
        *,
        object_points_camera,
        gripper_pose_camera,
    ) -> VLMGraspResult:
        """Re-observe and plan only the destination after the object is lifted."""
        validate_rgbd(color, depth_mm, K)
        color = color.copy()
        depth_mm = depth_mm.copy()
        K = np.asarray(K, dtype=np.float64).copy()
        object_points = np.asarray(object_points_camera, dtype=np.float64).reshape(-1, 3).copy()
        if not len(object_points) or not np.isfinite(object_points).all():
            raise ValueError("抬高后的抓取物点云为空或包含无效数值")

        tic = time.monotonic()
        print("[抬高复查 1/3] VLM 正在重新判断放置目标……", flush=True)
        grounding = self.client.ground(instruction, color, placement_only=True)
        if grounding.image_size != (color.shape[1], color.shape[0]):
            raise ValueError("VLM 放置复查框的原图尺寸与本轮 RGB-D 快照不一致")
        placement = grounding.placement
        rejected_segmentation = None
        rejected_error = None
        rejected = placement.rejected_container if placement is not None else None
        if rejected is not None:
            try:
                print(f"[抬高复查 2/3] SAM 正在分割不可用容器：{rejected.target_name}", flush=True)
                rejected_segmentation = self.segmenter.segment(color, rejected.bbox_xyxy)
            except ValueError as exc:
                rejected_error = f"不可用容器分割失败：{exc}"
                print(f"[SAM] {rejected_error}", flush=True)

        placement_segmentation = None
        placement_analysis = None
        placement_error = None
        if not self.placement_settings.enabled:
            placement_error = "放置几何已在配置中关闭"
        elif (placement is not None and placement.required and placement.found
              and placement.bbox_xyxy is not None):
            try:
                print(f"[抬高复查 2/3] SAM 正在分割放置目标：{placement.target_name}", flush=True)
                placement_segmentation = self.segmenter.segment(color, placement.bbox_xyxy)
                exclusion = None
                if rejected_segmentation is not None:
                    exclusion = rejected_segmentation.mask
                print("[抬高复查 3/3] 正在用新 RGB-D 计算容器几何和放置位姿……", flush=True)
                placement_analysis = self.placement_backend.analyze_placement_geometry_from_points(
                    object_points,
                    placement_segmentation.mask,
                    depth_mm,
                    K,
                    gripper_pose_camera=gripper_pose_camera,
                    target_name=placement.target_name,
                    relation=placement.relation or "in",
                    settings=self.placement_settings,
                    support_exclusion_mask=exclusion,
                )
            except ValueError as exc:
                placement_error = f"抬高后放置几何分析失败：{exc}"
                print(f"[放置复查] {placement_error}", flush=True)
        else:
            placement_error = "抬高后没有确认到可用放置目标"

        return VLMGraspResult(
            grounding=grounding,
            segmentation=None,
            grasp_result=None,
            elapsed_s=time.monotonic() - tic,
            camera_matrix=K,
            placement_segmentation=placement_segmentation,
            placement_analysis=placement_analysis,
            placement_error=placement_error,
            rejected_container_segmentation=rejected_segmentation,
            rejected_container_error=rejected_error,
            placement_only=True,
        )


def grasp_record(grasp) -> dict[str, Any] | None:
    if grasp is None:
        return None
    return {
        "coordinate_frame": "camera_optical", "length_unit": "m",
        "translation_m": np.asarray(grasp.translation).tolist(),
        "rotation_matrix": np.asarray(grasp.rotation_matrix).tolist(),
        "score": float(grasp.score), "width_m": float(grasp.width),
        "depth_m": float(grasp.depth),
    }
