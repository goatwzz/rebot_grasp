"""Safety, geometry, and motion helpers for the VLM robot workflow."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np

from .place_utils import move_ready, wait_motion
from .transforms import canonicalize_parallel_gripper_tcp_rotation, mat4_to_pose6d, pose6d_to_mat4


@dataclass(frozen=True)
class VLMRobotSettings:
    max_approach_tilt_deg: float = 40.0
    max_ready_rotation_deg: float = 90.0
    banana_min_mask_clearance_m: float = 0.006
    banana_end_fraction: float = 0.15
    place_release_clearance_m: float = 0.08
    place_approach_offset_m: float = 0.08
    place_ik_yaw_max_candidates: int = 181
    pregrasp_duration_s: float = 2.0
    grasp_duration_s: float = 1.5
    retreat_duration_s: float = 1.5
    place_duration_s: float = 2.0
    gripper_open_timeout_s: float = 3.0
    release_min_open_motion_rad: float = 0.5

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "VLMRobotSettings":
        raw = cfg.get("vlm_robot") or {}
        if not isinstance(raw, dict):
            raise ValueError("vlm_robot 必须是 YAML 映射")
        value = cls(
            max_approach_tilt_deg=float(raw.get("max_approach_tilt_deg", 40.0)),
            max_ready_rotation_deg=float(raw.get("max_ready_rotation_deg", 90.0)),
            banana_min_mask_clearance_m=float(raw.get("banana_min_mask_clearance_m", 0.006)),
            banana_end_fraction=float(raw.get("banana_end_fraction", 0.15)),
            place_release_clearance_m=float(raw.get("place_release_clearance_m", 0.08)),
            place_approach_offset_m=float(raw.get("place_approach_offset_m", 0.08)),
            place_ik_yaw_max_candidates=int(raw.get("place_ik_yaw_max_candidates", 181)),
            pregrasp_duration_s=float(raw.get("pregrasp_duration_s", 2.0)),
            grasp_duration_s=float(raw.get("grasp_duration_s", 1.5)),
            retreat_duration_s=float(raw.get("retreat_duration_s", 1.5)),
            place_duration_s=float(raw.get("place_duration_s", 2.0)),
            gripper_open_timeout_s=float(raw.get("gripper_open_timeout_s", 3.0)),
            release_min_open_motion_rad=float(raw.get("release_min_open_motion_rad", 0.5)),
        )
        numeric = np.asarray(list(value.__dict__.values()), dtype=np.float64)
        if not np.isfinite(numeric).all():
            raise ValueError("vlm_robot 参数必须是有限数值")
        if not 0.0 <= value.max_approach_tilt_deg < 90.0:
            raise ValueError("vlm_robot.max_approach_tilt_deg 必须在 [0,90) 度内")
        if not 0.0 < value.max_ready_rotation_deg <= 180.0:
            raise ValueError("vlm_robot.max_ready_rotation_deg 必须在 (0,180] 度内")
        if value.banana_min_mask_clearance_m < 0:
            raise ValueError("vlm_robot.banana_min_mask_clearance_m 不能为负")
        if not 0 <= value.banana_end_fraction < 0.5:
            raise ValueError("vlm_robot.banana_end_fraction 必须在 [0,0.5) 内")
        if value.place_release_clearance_m < 0 or value.place_approach_offset_m <= 0:
            raise ValueError("放置释放高度必须非负，放置接近距离必须为正数")
        if value.place_ik_yaw_max_candidates < 1:
            raise ValueError("放置 IK 搜索候选数量必须至少为 1")
        if min(
            value.pregrasp_duration_s,
            value.grasp_duration_s,
            value.retreat_duration_s,
            value.place_duration_s,
            value.gripper_open_timeout_s,
        ) <= 0:
            raise ValueError("机械臂动作时间必须为正数")
        if value.release_min_open_motion_rad <= 0:
            raise ValueError("放置开爪最小运动量必须为正数")
        return value


@dataclass(frozen=True)
class PickMotionResult:
    grasped: bool
    at_ready: bool
    failure_stage: str | None = None


@dataclass(frozen=True)
class PlaceMotionResult:
    released: bool
    at_ready: bool


def transform_points(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    transform = np.asarray(T, dtype=np.float64)
    cloud = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("点云变换必须是有限的 4x4 矩阵")
    if not len(cloud) or not np.isfinite(cloud).all():
        raise ValueError("抓取物点云为空或包含无效数值")
    homogeneous = np.column_stack([cloud, np.ones(len(cloud), dtype=np.float64)])
    return (homogeneous @ transform.T)[:, :3]


def transport_carried_object_points(
    object_points_camera_before: np.ndarray,
    T_camera_before_to_base: np.ndarray,
    grasp_pose_base: tuple[float, ...],
    T_tcp_lifted_to_base: np.ndarray,
    T_camera_lifted_to_base: np.ndarray,
) -> np.ndarray:
    """Move pre-grasp object points with the measured lifted TCP pose."""
    points_base = transform_points(T_camera_before_to_base, object_points_camera_before)
    T_tcp_grasp_to_base = pose6d_to_mat4(*grasp_pose_base)
    points_tcp = transform_points(np.linalg.inv(T_tcp_grasp_to_base), points_base)
    lifted_base = transform_points(T_tcp_lifted_to_base, points_tcp)
    return transform_points(np.linalg.inv(T_camera_lifted_to_base), lifted_base)


def current_gripper_pose_camera(
    T_tcp_to_base: np.ndarray,
    T_camera_to_base: np.ndarray,
) -> SimpleNamespace:
    T = np.linalg.inv(np.asarray(T_camera_to_base, dtype=np.float64)) @ np.asarray(T_tcp_to_base, dtype=np.float64)
    return SimpleNamespace(translation=T[:3, 3].copy(), rotation_matrix=T[:3, :3].copy())


def placement_poses_to_base(
    placement_analysis: Any,
    T_camera_to_base: np.ndarray,
    settings: VLMRobotSettings,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Create release and pre-place TCP poses from a post-lift placement plan."""
    plan = placement_analysis.plan
    if plan is None or not plan.fits:
        raise ValueError("放置几何没有可执行的 XY/Yaw 位姿")
    normal = np.asarray(placement_analysis.container.normal_camera, dtype=np.float64).reshape(3)
    normal_norm = float(np.linalg.norm(normal))
    if normal_norm < 1e-9 or not np.isfinite(normal).all():
        raise ValueError("容器开口法线无效")
    normal /= normal_norm

    T_release_camera = np.eye(4, dtype=np.float64)
    T_release_camera[:3, :3] = np.asarray(plan.gripper_rotation_camera, dtype=np.float64).reshape(3, 3)
    T_release_camera[:3, 3] = (
        np.asarray(plan.gripper_translation_camera_m, dtype=np.float64).reshape(3)
        + normal * settings.place_release_clearance_m
    )
    T_release_base = np.asarray(T_camera_to_base, dtype=np.float64) @ T_release_camera
    T_release_base[:3, :3] = canonicalize_parallel_gripper_tcp_rotation(T_release_base[:3, :3])

    T_pre_base = T_release_base.copy()
    normal_base = np.asarray(T_camera_to_base, dtype=np.float64)[:3, :3] @ normal
    normal_base /= max(float(np.linalg.norm(normal_base)), 1e-9)
    T_pre_base[:3, 3] += normal_base * settings.place_approach_offset_m
    return mat4_to_pose6d(T_release_base), mat4_to_pose6d(T_pre_base)


def placement_is_executable(result: Any) -> tuple[bool, str]:
    placement = result.grounding.placement
    if placement is None or not placement.required or not placement.found:
        return False, "VLM 未确认可从上方进入的放置目标"
    if result.placement_error:
        return False, result.placement_error
    if result.placement_segmentation is None or result.placement_analysis is None:
        return False, "放置目标缺少 SAM 掩码或三维几何"
    if result.placement_analysis.plan is None or not result.placement_analysis.plan.fits:
        return False, "抓取物的 XY/Yaw 轮廓无法安全放入当前容器"
    return True, "放置目标和位姿已确认"


def execute_pick_and_lift(
    controller: Any,
    grasp_driver: Any,
    grasp6d: tuple[float, ...],
    pre6d: tuple[float, ...],
    retreat6d: tuple[float, ...],
    ready_cfg: dict[str, Any],
    settings: VLMRobotSettings,
    *,
    dry_run: bool,
) -> PickMotionResult:
    print(f"[抓取] pregrasp xyz={pre6d[:3]} rpy={pre6d[3:]}")
    print(f"[抓取] grasp    xyz={grasp6d[:3]} rpy={grasp6d[3:]}")
    print(f"[抓取] retreat  xyz={retreat6d[:3]} rpy={retreat6d[3:]}")
    if dry_run:
        print("[抓取] dry-run：已完成位姿与 IK 检查，不发送运动或夹爪命令。")
        return PickMotionResult(False, False)
    try:
        grasp_driver.open_gripper_for_pick(timeout=settings.gripper_open_timeout_s)
    except (RuntimeError, ValueError) as exc:
        print(f"[抓取] 夹爪安全张开失败：{exc}；机械臂尚未开始抓取运动。")
        return PickMotionResult(False, False, "gripper_open")
    for label, pose, duration in (
        ("pregrasp", pre6d, settings.pregrasp_duration_s),
        ("grasp", grasp6d, settings.grasp_duration_s),
    ):
        print(f"[抓取] 移动到 {label}")
        if not controller.move_to_traj(*pose, duration=duration) or not wait_motion(controller, duration):
            print(f"[抓取] {label} 运动失败，停止本轮。")
            return PickMotionResult(False, False, label)
    print("[抓取] 闭合夹爪")
    grasped = bool(grasp_driver.grasp())
    print("[抓取] 已检测到物体" if grasped else "[抓取] 未检测到物体，本轮不复查或放置")
    if (not controller.move_to_traj(*retreat6d, duration=settings.retreat_duration_s)
            or not wait_motion(controller, settings.retreat_duration_s)):
        print("[抓取] 抬高退让失败；停止后续动作。")
        return PickMotionResult(grasped, False, "retreat")
    print("[抓取] 抬高并返回 ready 位")
    at_ready = bool(move_ready(controller, ready_cfg))
    return PickMotionResult(grasped, at_ready, None if at_ready else "return_ready")


def execute_place(
    controller: Any,
    grasp_driver: Any,
    release6d: tuple[float, ...],
    preplace6d: tuple[float, ...],
    ready_cfg: dict[str, Any],
    settings: VLMRobotSettings,
    *,
    dry_run: bool,
) -> PlaceMotionResult:
    print(f"[放置] pre-place xyz={preplace6d[:3]} rpy={preplace6d[3:]}")
    print(f"[放置] release   xyz={release6d[:3]} rpy={release6d[3:]}")
    if dry_run:
        print("[放置] dry-run：不发送运动或松爪命令。")
        return PlaceMotionResult(False, False)
    for label, pose in (("pre-place", preplace6d), ("release", release6d)):
        print(f"[放置] 移动到 {label}")
        if not controller.move_to_traj(*pose, duration=settings.place_duration_s):
            print(f"[放置] {label} 运动失败，保持夹持。")
            return PlaceMotionResult(False, False)
        if not wait_motion(controller, settings.place_duration_s):
            print(f"[放置] {label} 等待超时，保持夹持。")
            return PlaceMotionResult(False, False)
    print("[放置] 松开夹爪")
    if not grasp_driver.open_gripper_for_release(
        timeout=settings.gripper_open_timeout_s,
        min_motion_rad=settings.release_min_open_motion_rad,
    ):
        print("[放置] 夹爪未确认张开；禁止返回 ready 或启动下一轮，请人工检查物体是否仍被夹持。")
        return PlaceMotionResult(False, False)
    if (not controller.move_to_traj(*preplace6d, duration=settings.place_duration_s)
            or not wait_motion(controller, settings.place_duration_s)):
        print("[放置] 释放后退让失败；停止后续动作。")
        return PlaceMotionResult(True, False)
    print("[放置] 返回 ready 位")
    return PlaceMotionResult(True, bool(move_ready(controller, ready_cfg)))
