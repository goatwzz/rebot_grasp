"""VLM + SAM + GraspNet pick, post-lift container recheck, and robot placement.

The preview-only ``scripts/vlm_grasp_demo.py`` remains unchanged.  This entry
point moves the real robot only after a frozen RGB-D result has passed table-
normal, approach-angle, base-height, and IK checks.

Keys:
  G/Space: execute the verified task, or recheck the container while holding.
  R: discard a pre-pick result and resume live preview.
  Q/Esc: stop without automatically opening a holding gripper.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import queue
from pathlib import Path
import sys
import threading
from typing import Any

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from drivers.camera import make_camera  # noqa: E402
from drivers.robot.grasp_driver import GraspDriver, selected_arm_config  # noqa: E402
from reBotArm_control_py.actuator import RebotArm  # noqa: E402
from reBotArm_control_py.controllers import RebotArmEndPose  # noqa: E402
from scripts.grasp import IkChecker, _print_grasp, _select_executable_grasp  # noqa: E402
from scripts.vlm_grasp_demo import GraspPreviewProcessor, check_models, resolve_dashboard  # noqa: E402
from scripts.vlm_target_demo import (  # noqa: E402
    normalize_terminal_command,
    start_terminal_input,
)
from utils import graspnet_utils, placement_geometry  # noqa: E402
from utils.grasp_candidate_quality import rank_elongated_grasp_centers  # noqa: E402
from utils.transforms import pose6d_to_mat4  # noqa: E402
from utils.camera_utils import (  # noqa: E402
    compose_cam_to_base_transform,
    configure_camera,
    load_config,
    load_hand_eye,
)
from utils.terminal_utils import TerminalStateGuard  # noqa: E402
from utils.place_utils import move_ready  # noqa: E402
from utils.vlm_grasp_pipeline import VLMGraspPipeline  # noqa: E402
from utils.vlm_sequence_utils import SequenceProgress  # noqa: E402
from utils.vlm_robot_utils import (  # noqa: E402
    VLMRobotSettings,
    current_gripper_pose_camera,
    execute_pick_and_lift,
    execute_place,
    placement_is_executable,
    placement_poses_to_base,
    transport_carried_object_points,
)
from utils.vlm_utils import VLMClient, VLMError, VLMSettings  # noqa: E402
from utils.wrist_video import RecordingCamera  # noqa: E402


@dataclass
class PendingPick:
    command: str
    result: Any
    color: np.ndarray
    depth_mm: np.ndarray
    K: np.ndarray
    T_camera_to_base: np.ndarray
    object_points_camera: np.ndarray
    grasp: Any
    grasp6d: tuple[float, ...]
    pregrasp6d: tuple[float, ...]
    retreat6d: tuple[float, ...]
    approach_tilt_deg: float
    initial_place: PendingPlace | None = None
    recheck_after_lift: bool = True


@dataclass
class HeldObject:
    command: str
    object_points_camera_before: np.ndarray
    T_camera_before_to_base: np.ndarray
    grasp6d: tuple[float, ...]


@dataclass
class PendingPlace:
    result: Any
    release6d: tuple[float, ...]
    preplace6d: tuple[float, ...]


def parse_args(*, default_output_dir: Path | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VLM＋SAM＋GraspNet 实机抓取、抬高复查容器并放置")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config/default.yaml")
    parser.add_argument("--instruction", help="启动后立即分析的中文指令；省略时在终端输入")
    parser.add_argument("--dry-run", action="store_true", help="连接机械臂并做 FK/IK，但不发送运动或夹爪命令")
    parser.add_argument("--check-config", action="store_true", help="只检查视觉模型与参数，不连接相机或机械臂")
    parser.add_argument("--no-open3d", action="store_true")
    parser.add_argument("--no-mask-window", action="store_true")
    parser.add_argument("--camera-type", choices=("realsense_d435i", "realsense_d405", "orbbec_gemini2"))
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--fps", type=int)
    parser.add_argument("--warmup", type=int, default=15)
    parser.add_argument("--max-steps", type=int, default=10, help="连续任务最多成功放置次数")
    parser.add_argument("--output-dir", type=Path, default=default_output_dir or PROJECT_ROOT / "outputs/vlm_robot")
    args = parser.parse_args()
    if not args.check_config and not args.instruction and not sys.stdin.isatty():
        parser.error("当前终端不能交互输入，请提供 --instruction")
    return args


def _combined_exclusion_masks(result: Any) -> list[np.ndarray]:
    masks = []
    for name in ("segmentation", "placement_segmentation", "rejected_container_segmentation"):
        segmented = getattr(result, name, None)
        if segmented is not None:
            masks.append(segmented.mask)
    return masks


def _return_ready_after_sequence(
    controller: Any,
    ready_cfg: dict[str, Any],
    *,
    holding_object: bool,
    verification_pending: bool,
    dry_run: bool,
) -> bool:
    if holding_object or verification_pending:
        print("[连续任务] 夹持或放置复核状态未清除；禁止自动归位，请人工检查。")
        return False
    if dry_run:
        print("[连续任务] dry-run 不发送归位命令；结束程序。")
        return True
    print("[连续任务] 任务完成，返回配置的 ready 位。")
    try:
        if not move_ready(controller, ready_cfg):
            print("[连续任务] 归位未确认成功；停止控制，请人工检查机械臂状态。")
            return False
    except Exception as exc:
        print(f"[连续任务] 归位异常：{exc}；停止控制，请人工检查机械臂状态。")
        return False
    print("[连续任务] 已返回 ready 位；自动退出程序。")
    return True


def _finish_prepare_pick(
    command: str,
    result: Any,
    color: np.ndarray,
    depth_mm: np.ndarray,
    K: np.ndarray,
    T_camera_to_base: np.ndarray,
    ik_checker: IkChecker,
    robot_settings: VLMRobotSettings,
    grasp_cfg: dict[str, Any],
    table_normal: np.ndarray | None,
    ready_rotation_base: np.ndarray,
) -> PendingPick | None:
    # Keep a hard failure when no table plane is observable.
    if table_normal is None:
        print("[方向筛选] 无法可靠估计桌面法线；拒绝本轮实机抓取。")
        return None
    original = result.grasp_result.grasps
    target_name = str(result.grounding.target_name or "").lower()
    banana_target = "banana" in target_name or "香蕉" in target_name
    banana_geometry_before = None
    if banana_target:
        try:
            geometry_order, fractions, _ = rank_elongated_grasp_centers(
                original.translations, original.scores, result.segmentation.mask, K,
                min_clearance_m=robot_settings.banana_min_mask_clearance_m,
                end_fraction=robot_settings.banana_end_fraction,
            )
        except ValueError as exc:
            result.grasp_result.best = None
            print(f"[香蕉候选] {exc}；本轮不抓取。")
            return None
        banana_geometry_before = len(geometry_order)
        stage_counts = []
        for key, label, group in (
            ("decoded", "解码", getattr(result.grasp_result, "raw_grasps", None)),
            ("collision", "碰撞后", getattr(result.grasp_result, "pre_bbox_grasps", None)),
            ("bbox", "目标框后", getattr(result.grasp_result, "bbox_grasps", None)),
            ("mask_depth", "掩码/深度后", getattr(result.grasp_result, "pre_width_grasps", None)),
        ):
            if group is None:
                continue
            stage_order, stage_fractions, _ = rank_elongated_grasp_centers(
                group.translations, group.scores, result.segmentation.mask, K,
                min_clearance_m=robot_settings.banana_min_mask_clearance_m,
                end_fraction=robot_settings.banana_end_fraction,
            )
            middle_count = int(np.count_nonzero(
                (stage_fractions[stage_order] >= 0.30) & (stage_fractions[stage_order] <= 0.70)
            ))
            stage_counts.append(f"{label}={middle_count}/{len(group)}")
            if result.grasp_result.filter_counts is not None:
                result.grasp_result.filter_counts[f"banana_middle_{key}"] = middle_count
        middle_count = int(np.count_nonzero(
            (fractions[geometry_order] >= 0.30) & (fractions[geometry_order] <= 0.70)
        ))
        stage_counts.append(f"宽度后={middle_count}/{len(original)}")
        if result.grasp_result.filter_counts is not None:
            result.grasp_result.filter_counts["banana_middle_width"] = middle_count
        print("[香蕉中段审计] 掩码中段且边缘余量合格/总候选：" + " → ".join(stage_counts))
    filtered, angles = graspnet_utils.filter_grasps_by_approach_tilt(
        original, table_normal, robot_settings.max_approach_tilt_deg,
    )
    finite = angles[np.isfinite(angles)]
    angle_range = "无有效角度" if not len(finite) else f"{finite.min():.1f}°~{finite.max():.1f}°"
    print(
        f"[方向筛选] 输入={len(original)} → 上方接近={len(filtered)}；"
        f"反向={int(np.count_nonzero(finite > 90.0))}，"
        f"阈值≤{robot_settings.max_approach_tilt_deg:.1f}°，候选范围={angle_range}"
    )
    if len(filtered) == 0:
        result.grasp_result.grasps = filtered
        result.grasp_result.best = None
        if result.grasp_result.filter_counts is not None:
            result.grasp_result.filter_counts["approach"] = 0
            if banana_target:
                result.grasp_result.filter_counts["banana_middle_approach"] = 0
        if banana_target:
            print(f"[香蕉审计] 宽度后几何可用={banana_geometry_before}，倾角后=0。")
        print("[方向筛选] 所有候选都过于侧向或反向；拒绝本轮实机抓取。")
        return None
    approach_count = len(filtered)
    banana_geometry_after_approach = None
    banana_middle_after_approach = None
    if banana_target:
        geometry_order, approach_fractions, _ = rank_elongated_grasp_centers(
            filtered.translations, filtered.scores, result.segmentation.mask, K,
            min_clearance_m=robot_settings.banana_min_mask_clearance_m,
            end_fraction=robot_settings.banana_end_fraction,
        )
        banana_geometry_after_approach = len(geometry_order)
        banana_middle_after_approach = int(np.count_nonzero(
            (approach_fractions[geometry_order] >= 0.30)
            & (approach_fractions[geometry_order] <= 0.70)
        ))
    filtered, rotation_angles = graspnet_utils.filter_grasps_by_ready_rotation(
        filtered, T_camera_to_base, ready_rotation_base, robot_settings.max_ready_rotation_deg,
    )
    print(
        f"[姿态筛选] ready 相对旋转≤{robot_settings.max_ready_rotation_deg:.1f}°："
        f"{len(rotation_angles)} → {len(filtered)}；过大={len(rotation_angles) - len(filtered)}"
    )
    if len(filtered) == 0:
        result.grasp_result.grasps = filtered
        result.grasp_result.best = None
        if result.grasp_result.filter_counts is not None:
            result.grasp_result.filter_counts["approach"] = approach_count
            result.grasp_result.filter_counts["ready_rotation"] = 0
            if banana_target:
                result.grasp_result.filter_counts["banana_middle_approach"] = banana_middle_after_approach
                result.grasp_result.filter_counts["banana_middle_ready"] = 0
        if banana_target:
            print(f"[香蕉审计] 宽度后几何可用={banana_geometry_before}，"
                  f"倾角后={banana_geometry_after_approach}，ready 后=0。")
        print("[姿态筛选] 无安全抓取姿态；机械臂不运动。")
        return None
    if result.grasp_result.filter_counts is not None:
        result.grasp_result.filter_counts["approach"] = approach_count
        result.grasp_result.filter_counts["ready_rotation"] = len(filtered)
    result.grasp_result.grasps = filtered

    if banana_target:
        final_tilt_angles = graspnet_utils.approach_tilt_angles_deg(filtered, table_normal)
        order, fractions, clearances_m = rank_elongated_grasp_centers(
            filtered.translations,
            filtered.scores,
            result.segmentation.mask,
            K,
            min_clearance_m=robot_settings.banana_min_mask_clearance_m,
            end_fraction=robot_settings.banana_end_fraction,
            approach_tilt_deg=final_tilt_angles,
        )
        end_count = int(np.count_nonzero(
            np.isfinite(fractions) & ((fractions < robot_settings.banana_end_fraction)
                                        | (fractions > 1 - robot_settings.banana_end_fraction))
        ))
        edge_count = int(np.count_nonzero(
            np.isfinite(fractions) & (clearances_m < robot_settings.banana_min_mask_clearance_m)
        ))
        filtered = filtered[order]
        result.grasp_result.grasps = filtered
        middle_after_ready = int(np.count_nonzero(
            (fractions[order] >= 0.30) & (fractions[order] <= 0.70)
        ))
        if result.grasp_result.filter_counts is not None:
            result.grasp_result.filter_counts["banana_geometry_before_approach"] = banana_geometry_before
            result.grasp_result.filter_counts["banana_geometry_after_approach"] = banana_geometry_after_approach
            result.grasp_result.filter_counts["banana_quality"] = len(filtered)
            result.grasp_result.filter_counts["banana_middle_approach"] = banana_middle_after_approach
            result.grasp_result.filter_counts["banana_middle_ready"] = middle_after_ready
        print(
            f"[香蕉候选] 宽度后几何可用={banana_geometry_before} → "
            f"倾角后={banana_geometry_after_approach} → ready 后={len(filtered)}；"
            f"中段={banana_middle_after_approach}→{middle_after_ready}；"
            f"头尾={end_count}，贴边={edge_count}（可重复计数）；"
            "优先兼顾网络分数、中段、掩码内部余量和接近倾角。"
        )
        if len(filtered) == 0:
            result.grasp_result.best = None
            print("[香蕉候选] 没有满足上方接近与中段接触要求的位姿；本轮不抓取。")
            return None

    pregrasp_offset = float(grasp_cfg.get("pregrasp_offset_m", 0.08))
    retreat_offset = float(grasp_cfg.get("retreat_offset_m", pregrasp_offset))
    insertion_depth = float(grasp_cfg.get("insertion_depth_m", 0.0))
    min_base_z = float(grasp_cfg.get("min_base_z_m", 0.03))
    selected = _select_executable_grasp(
        ik_checker,
        filtered,
        T_camera_to_base,
        pregrasp_offset,
        retreat_offset,
        insertion_depth,
        min_base_z,
        preordered=banana_target,
    )
    if selected is None:
        result.grasp_result.best = None
        return None
    best, grasp6d, pre6d, retreat6d = selected
    result.grasp_result.best = best
    if banana_target:
        _, selected_fraction, selected_clearance = rank_elongated_grasp_centers(
            np.asarray(best.translation, dtype=np.float64)[None, :],
            np.asarray([best.score], dtype=np.float64),
            result.segmentation.mask,
            K,
            min_clearance_m=robot_settings.banana_min_mask_clearance_m,
            end_fraction=robot_settings.banana_end_fraction,
        )
        print(f"[香蕉候选] 最终位置处于长轴 {selected_fraction[0]:.0%}，"
              f"距掩码侧边约 {selected_clearance[0] * 1000:.1f} mm。")
    approach = np.asarray(best.rotation_matrix, dtype=np.float64)[:, 0]
    normal = np.asarray(table_normal, dtype=np.float64)
    chosen_tilt = float(np.rad2deg(np.arccos(np.clip(
        -float(approach @ normal) / (np.linalg.norm(approach) * np.linalg.norm(normal)), -1.0, 1.0,
    ))))
    chosen_rotation = pose6d_to_mat4(*grasp6d)[:3, :3]
    ready_delta = np.asarray(ready_rotation_base, dtype=np.float64).T @ chosen_rotation
    chosen_ready_angle = float(np.rad2deg(np.arccos(np.clip((np.trace(ready_delta) - 1.0) / 2.0, -1.0, 1.0))))
    points = placement_geometry.mask_points_camera(result.segmentation.mask, depth_mm, K)
    print(f"[实机检查] 已选接近角={chosen_tilt:.1f}°，相对 ready 旋转={chosen_ready_angle:.1f}°；抓取/预抓取 IK 与基座高度检查通过。")
    _print_grasp(best)
    return PendingPick(
        command, result, color.copy(), depth_mm.copy(), K.copy(), T_camera_to_base.copy(),
        points, best, grasp6d, pre6d, retreat6d, chosen_tilt,
    )


def prepare_pick(
    command: str,
    result: Any,
    color: np.ndarray,
    depth_mm: np.ndarray,
    K: np.ndarray,
    T_camera_to_base: np.ndarray,
    ik_checker: IkChecker,
    robot_settings: VLMRobotSettings,
    grasp_cfg: dict[str, Any],
    placement_settings: Any,
    ready_rotation_base: np.ndarray,
) -> PendingPick | None:
    if result.segmentation is None or result.grasp_result is None or len(result.grasp_result.grasps) == 0:
        print("[实机检查] 没有通过目标框/SAM/深度/宽度筛选的抓取候选。")
        return None
    normal = placement_geometry.estimate_table_normal(
        depth_mm,
        K,
        exclusion_masks=_combined_exclusion_masks(result),
        settings=placement_settings,
        expected_normal_camera=(
            np.asarray(T_camera_to_base, dtype=np.float64)[:3, :3].T
            @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
        ),
    )
    pending = _finish_prepare_pick(
        command, result, color, depth_mm, K, T_camera_to_base, ik_checker,
        robot_settings, grasp_cfg, normal, ready_rotation_base,
    )
    if pending is None:
        return None

    placement = result.grounding.placement
    if placement is not None and placement.selection_mode == "disabled":
        pending.recheck_after_lift = False
        print("[第一次容器检查] 指令明确要求只抓取；抬高后保持夹持，不执行容器复查或放置。")
        return pending
    if (placement is not None and placement.required and placement.found
            and result.placement_segmentation is not None):
        try:
            # The pipeline's first preview used its pre-IK best grasp. Rebuild
            # the placement plan around the exact direction-safe, IK-reachable
            # grasp that will actually be executed.
            result.placement_analysis = placement_geometry.analyze_placement_geometry(
                result.segmentation.mask,
                result.placement_segmentation.mask,
                depth_mm,
                K,
                best_grasp=pending.grasp,
                target_name=placement.target_name,
                relation=placement.relation or "in",
                settings=placement_settings,
            )
            result.placement_error = None
        except ValueError as exc:
            result.placement_analysis = None
            result.placement_error = f"第一次快照放置几何分析失败：{exc}"

    selected_place, place_reason = _select_executable_placement(
        result,
        T_camera_to_base,
        ik_checker,
        robot_settings,
        float(grasp_cfg.get("min_base_z_m", 0.03)),
    )
    if selected_place is not None:
        release6d, preplace6d = selected_place
        pending.initial_place = PendingPlace(result, release6d, preplace6d)
        print(f"[第一次容器检查] {place_reason}；抓取后将直接执行放置，不再拍第二张图。")
    else:
        print(f"[第一次容器检查] {place_reason}；允许抓取，抬高后重新拍照检查容器。")
    return pending


def _capture_fresh(camera: Any, count: int = 4) -> tuple[np.ndarray, np.ndarray]:
    latest = (None, None)
    consecutive_valid = 0
    for _ in range(max(1, count)):
        frame = camera.get_frame()
        if frame[0] is not None and frame[1] is not None:
            latest = frame
            consecutive_valid += 1
        else:
            latest = (None, None)
            consecutive_valid = 0
    if consecutive_valid < min(2, max(1, count)) or latest[0] is None or latest[1] is None:
        raise RuntimeError("未能获取连续的新 RGB-D 图像；禁止沿用旧帧")
    return latest[0].copy(), latest[1].copy()


def _save_and_report(processor: GraspPreviewProcessor, output_dir: Path, color, depth, K, result) -> np.ndarray:
    display = processor.draw(color, result)
    processor.show_result(color, result)
    try:
        directory = processor.save(output_dir, color, result, depth_mm=depth, K=K)
        processor.report(result, directory)
    except OSError as exc:
        print(f"[保存失败] {exc}")
    return display


def _pose_checks(
    ik_checker: IkChecker,
    release6d: tuple[float, ...],
    preplace6d: tuple[float, ...],
    min_base_z: float,
) -> tuple[bool, str]:
    if release6d[2] < min_base_z or preplace6d[2] < min_base_z:
        return False, f"放置位姿低于基座 Z 下限 {min_base_z:.3f} m"
    sequence_ok, errors = ik_checker.check_sequence((preplace6d, release6d))
    if not sequence_ok:
        stage = "pre-place" if len(errors) == 1 else "release"
        return False, f"放置 {stage} IK 不可达（误差={errors[-1]:.4f}）"
    return True, "放置 pre-place/release IK 检查通过"


def _select_executable_placement(
    result: Any,
    T_camera_to_base: np.ndarray,
    ik_checker: IkChecker,
    robot_settings: VLMRobotSettings,
    min_base_z: float,
) -> tuple[tuple[tuple[float, ...], tuple[float, ...]] | None, str]:
    """Choose a geometrically valid placement yaw that also passes sequential IK."""
    place_valid, place_reason = placement_is_executable(result)
    if not place_valid:
        return None, place_reason

    analysis = result.placement_analysis
    original_plan = analysis.plan
    angles = list(getattr(analysis, "candidate_angles_deg", ()) or ())
    if not angles:
        angles = [float(original_plan.rotation_about_normal_deg)]
    limit = min(len(angles), robot_settings.place_ik_yaw_max_candidates)
    last_reason = "没有生成放置 IK 候选"
    checked = 0
    for rank, angle_deg in enumerate(angles[:limit], 1):
        try:
            current_angle = float(analysis.plan.rotation_about_normal_deg)
            if abs(current_angle - float(angle_deg)) > 1e-9:
                placement_geometry.select_placement_plan(analysis, angle_deg)
            if analysis.plan is None or not analysis.plan.fits:
                continue
            release6d, preplace6d = placement_poses_to_base(
                analysis, T_camera_to_base, robot_settings,
            )
            checked += 1
            pose_ok, pose_reason = _pose_checks(
                ik_checker, release6d, preplace6d, min_base_z,
            )
            last_reason = pose_reason
            if pose_ok:
                print(
                    f"[放置 IK] 选择可达候选 {rank}/{limit}："
                    f"平面旋转={float(angle_deg):+.1f}°；{pose_reason}"
                )
                return (release6d, preplace6d), pose_reason
        except ValueError as exc:
            last_reason = str(exc)

    analysis.plan = original_plan
    return None, f"{last_reason}；已检查 {checked}/{limit} 个 XY/Yaw 可放入候选"


def _recheck_after_lift(
    held: HeldObject,
    camera: Any,
    K: np.ndarray,
    grasp_driver: GraspDriver,
    T_hand_eye: np.ndarray,
    cfg: dict[str, Any],
    pipeline: VLMGraspPipeline,
    processor: GraspPreviewProcessor,
    output_dir: Path,
    ik_checker: IkChecker,
    robot_settings: VLMRobotSettings,
    min_base_z: float,
) -> tuple[PendingPlace | None, np.ndarray, str]:
    color, depth = _capture_fresh(camera)
    lifted_tcp = grasp_driver.get_tcp_pose()
    T_camera_to_base = compose_cam_to_base_transform(lifted_tcp, T_hand_eye, cfg)
    carried_points = transport_carried_object_points(
        held.object_points_camera_before,
        held.T_camera_before_to_base,
        held.grasp6d,
        lifted_tcp,
        T_camera_to_base,
    )
    gripper_camera = current_gripper_pose_camera(lifted_tcp, T_camera_to_base)
    recheck_instruction = (
        held.command
        + "\n当前抓取物已经被机械臂夹持并抬高。请只根据这张新图重新严格判断放置目标是否可从上方进入。"
    )
    result = pipeline.recheck_placement(
        recheck_instruction,
        color,
        depth,
        K,
        object_points_camera=carried_points,
        gripper_pose_camera=gripper_camera,
    )
    selected_place, reason = _select_executable_placement(
        result, T_camera_to_base, ik_checker, robot_settings, min_base_z,
    )
    # Save only after IK-aware yaw selection so the RGB/Open3D preview and
    # placement_geometry.json describe the pose that will actually execute.
    display = _save_and_report(processor, output_dir, color, depth, K, result)
    if selected_place is None:
        print(f"[抬高复查] {reason}；保持夹持并停在 ready 位。按 G/Space 可重新拍照复查。")
        return None, display, "HOLDING OBJECT | NO VALID CONTAINER | G=RECHECK"
    release6d, preplace6d = selected_place
    print(f"[抬高复查] {reason}；立即继续执行放置。")
    return PendingPlace(result, release6d, preplace6d), display, "PLACE VERIFIED | EXECUTING"


def main(*, pipeline_factory=VLMGraspPipeline, processor_factory=GraspPreviewProcessor,
         prepare_pick_fn=prepare_pick, default_output_dir: Path | None = None,
         window_title="VLM Grasp Robot - REAL CONTROL",
         workflow_description: str | None = None, sequence_mode: bool = False) -> int:
    args = parse_args(default_output_dir=default_output_dir)
    if sequence_mode and args.max_steps < 1:
        raise ValueError("--max-steps 必须至少为 1")
    terminal_guard = TerminalStateGuard()
    camera = None
    processor = None
    rebotarm = None
    controller = None
    grasp_driver = None
    holding_object = False
    placement_pending_verification = False
    state = "idle"
    try:
        cfg = configure_camera(load_config(args.config), args)
        vlm_settings = VLMSettings.from_config(cfg)
        robot_settings = VLMRobotSettings.from_config(cfg)
        record_video = (cfg.get("vlm_robot") or {}).get("record_video", False)
        if type(record_video) is not bool:
            raise ValueError("vlm_robot.record_video 必须为 true 或 false")
        check_models(cfg)
        if args.check_config:
            print(
                f"[配置就绪] VLM={vlm_settings.model}，接近角≤{robot_settings.max_approach_tilt_deg:.1f}°，"
                f"释放高度={robot_settings.place_release_clearance_m:.3f} m，"
                f"接近距离={robot_settings.place_approach_offset_m:.3f} m，"
                f"腕部录像={'开启' if record_video else '关闭'}。"
            )
            return 0

        camera = make_camera(cfg)
        camera.open()
        camera.warm_up(args.warmup)
        K = camera.K.astype(np.float64)
        cam_type = str(cfg["camera"]["type"]).lower()
        T_hand_eye, hand_eye_mode = load_hand_eye(PROJECT_ROOT, cam_type)
        if T_hand_eye is None or hand_eye_mode != "eye_in_hand":
            raise ValueError("实机脚本要求有效的 eye_in_hand 手眼标定文件")

        pipeline = pipeline_factory(cfg, VLMClient(vlm_settings))
        dashboard, tile_width, tile_height, focus_radius = resolve_dashboard(cfg)
        processor = processor_factory(
            pipeline,
            show_mask=not args.no_mask_window,
            show_open3d=not args.no_open3d,
            top_k=int((cfg.get("graspnet") or {}).get("top_k", 50)),
            open3d_grasps="best",
            dashboard=dashboard,
            dashboard_tile_width=tile_width,
            dashboard_tile_height=tile_height,
            grasp_cloud_focus_radius_m=focus_radius,
        )

        robot_cfg = cfg.get("robot") or {}
        ready_cfg = robot_cfg.get("ready_pose") or {
            "x": 0.25, "y": 0.0, "z": 0.35, "roll": 0.0, "pitch": 1.2, "yaw": 0.0, "duration": 3.0,
        }
        grasp_cfg = (cfg.get("grasp_pipeline") or {}).get("grasp") or {}
        min_base_z = float(grasp_cfg.get("min_base_z_m", 0.03))
        selected_arm = selected_arm_config(robot_cfg.get("repo_root"))
        rebotarm = RebotArm()
        controller = RebotArmEndPose(rebotarm, arm_control_mode=selected_arm.controller_mode)
        grasp_driver = GraspDriver(
            rebotarm, controller, gripper_config=robot_cfg.get("gripper"), repo_root=robot_cfg.get("repo_root"),
        )
        ik_checker = IkChecker(rebotarm)
        if record_video:
            camera = RecordingCamera(
                camera, args.output_dir, float(getattr(camera, "_fps", cfg["camera"]["fps"])),
            )
        if args.dry_run:
            rebotarm.connect()
            rebotarm.get_state(request_feedback=True)
            print("[Robot] dry-run：只读取机械臂状态做 FK/IK，不启动控制线程，不发送运动或夹爪命令。")
        else:
            grasp_driver.start()
            print("[Robot] 首先移动到 ready 初始位姿。")
            if not move_ready(controller, ready_cfg):
                raise RuntimeError("机械臂未能到达 ready 初始位姿，停止启动实机流程")
            camera.warm_up(5)

        commands: queue.Queue = queue.Queue()
        completions: queue.Queue = queue.Queue()
        if args.instruction:
            commands.put(args.instruction)
        input_pending = False
        terminal_enabled = sys.stdin.isatty()
        busy = False
        generation = 0
        last_instruction = ""
        state = "idle"
        pending_pick: PendingPick | None = None
        held: HeldObject | None = None
        frozen = None
        # cv2.putText only supports Hershey ASCII glyphs.  Keep all text drawn
        # into the RGB window in English; Chinese guidance remains in terminal.
        status = "LIVE | TYPE INSTRUCTION IN TERMINAL"
        sequence_active = False
        progress = SequenceProgress()
        next_snapshot_pending = False
        placement_pending_verification = False
        auto_execute_next = False
        window = window_title + (" [DRY RUN]" if args.dry_run else "")
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        if processor.display_size() is not None:
            cv2.resizeWindow(window, *processor.display_size())
        processor.bind_window(window)
        print("\n[实机流程] 第一张 RGB-D 同时检查抓取与容器；G 执行已经通过检查的连续任务。")
        if workflow_description is None:
            print("[实机流程] 第一张图容器有效：抓取→抬高→直接放置；无效：抓取抬高后才拍第二张图复查。")
        else:
            print(workflow_description)
        print("[图像窗口] G/Space=执行当前安全动作或重查  R=丢弃抓取前结果  Q/Esc=退出")
        if processor.display_size() is not None:
            print("[总览交互] 双击任意格放大；1~4=放大对应格，0=返回四宫格；3D 格左拖旋转、右拖平移、滚轮缩放。")
        print("[终端] 任务通过检查后输入 g 并回车，也会执行当前安全动作；q=退出。")

        def submit(command: str, color: np.ndarray, depth: np.ndarray) -> None:
            nonlocal generation, busy, last_instruction, frozen, status
            generation += 1
            request_id = generation
            snapshot = color.copy()
            snapshot_depth = depth.copy()
            snapshot_K = K.copy()
            snapshot_tcp = grasp_driver.get_tcp_pose()
            T_camera = compose_cam_to_base_transform(snapshot_tcp, T_hand_eye, cfg)
            busy = True
            last_instruction = command
            frozen = snapshot.copy()
            status = "SNAPSHOT | VLM/SAM/GRASPNET PROCESSING"
            processor.reset_preview(color.shape[:2], "PROCESSING")
            print(f"\n[VLM] 正在分析 ready 位快照，指令：{command}")

            def worker() -> None:
                try:
                    result = pipeline.process(command, snapshot, snapshot_depth, snapshot_K)
                    error = None
                except (VLMError, ValueError, RuntimeError) as exc:
                    result, error = None, str(exc)
                except Exception:
                    result, error = None, "视觉管线异常，请检查终端日志和模型配置"
                completions.put((request_id, command, snapshot, snapshot_depth, snapshot_K, T_camera, result, error))

            threading.Thread(target=worker, daemon=True, name="vlm-robot-vision").start()

        def run_verified_place(plan: PendingPlace) -> None:
            nonlocal holding_object, held, state, frozen, status, next_snapshot_pending
            nonlocal placement_pending_verification
            status = "PLACE VERIFIED | EXECUTING"
            place_motion = execute_place(
                controller,
                grasp_driver,
                plan.release6d,
                plan.preplace6d,
                ready_cfg,
                robot_settings,
                dry_run=args.dry_run,
            )
            if place_motion.released:
                holding_object = False
                held = None
                if place_motion.at_ready:
                    state = "idle"
                    frozen = None
                    status = "PLACE MOTION COMPLETE | VERIFYING RELEASE"
                    if sequence_active:
                        placement_pending_verification = True
                        next_snapshot_pending = True
                        print("[连续任务] 放置动作结束；等待新 RGB-D 确认夹爪未持有物体。")
                else:
                    state = "fault"
                    status = "OBJECT RELEASED | READY MOTION FAILED | MANUAL CHECK"
            else:
                state = "fault_holding"
                status = "HOLDING OBJECT | PLACE MOTION FAILED | MANUAL CHECK"

        while True:
            execute_requested = False
            retry_sequence = False
            color, depth = camera.get_frame()
            processor.poll_preview()
            if color is None or depth is None:
                continue
            if next_snapshot_pending and state == "idle" and not busy:
                next_snapshot_pending = False
                try:
                    fresh_color, fresh_depth = _capture_fresh(camera)
                    auto_execute_next = progress.placed > 0 or placement_pending_verification
                    submit(last_instruction, fresh_color, fresh_depth)
                except RuntimeError as exc:
                    state = "sequence_paused"
                    status = "FRESH FRAME FAILED | G=RETRY"
                    auto_execute_next = False
                    print(f"[连续任务] {exc}；已暂停。按 g 重新拍照检查，q 退出。")
            try:
                completed = completions.get_nowait()
            except queue.Empty:
                completed = None
            if completed is not None:
                request_id, command, snap_color, snap_depth, snap_K, T_camera, result, error = completed
                busy = False
                if request_id == generation:
                    if error:
                        print(f"[视觉失败] {error}")
                        status = ("VISION ERROR | G=RETRY | Q=EXIT" if placement_pending_verification
                                  else "VISION ERROR | G=RETRY | R=DISCARD")
                        processor.reset_preview(color.shape[:2], status)
                        state = "sequence_paused" if sequence_active else "idle"
                        auto_execute_next = False
                        if sequence_active:
                            if placement_pending_verification:
                                print("[连续任务] 放置仍待复核；输入 g 重拍，q 退出，不能放弃安全检查。")
                            else:
                                print("[连续任务] 保留原指令及进度；输入 g 重拍，r 放弃本轮，q 退出。")
                    else:
                        if sequence_active and placement_pending_verification:
                            placement_pending_verification = False
                            progress.placed += 1
                            print(f"[连续任务] 新图未发现夹爪持有目标；已确认 {progress.placed} 次放置动作。")
                        decision = (progress.observe(pipeline.inventory.remaining, args.max_steps)
                                    if sequence_active else None)
                        if decision in ("confirm_empty", "complete"):
                            frozen = _save_and_report(
                                processor, args.output_dir, snap_color, snap_depth, snap_K, result,
                            )
                            if decision == "complete":
                                state = "sequence_complete"
                                status = "TASK COMPLETE | NO OBJECTS OUTSIDE"
                                print(f"[连续任务] 连续两张新图均未发现盒外目标；总放置={progress.placed}。")
                                returned_ready = _return_ready_after_sequence(
                                    controller, ready_cfg,
                                    holding_object=holding_object,
                                    verification_pending=placement_pending_verification,
                                    dry_run=args.dry_run,
                                )
                                return 0 if returned_ready else 2
                            else:
                                next_snapshot_pending = True
                                auto_execute_next = False
                                state = "idle"
                                status = "EMPTY INVENTORY | VERIFYING NEW FRAME"
                                print("[连续任务] 首次清点为空；再次拍照确认后才结束。")
                            continue
                        if sequence_active:
                            remaining = pipeline.inventory.remaining
                            if decision == "limit":
                                state = "sequence_paused"
                                status = "STEP LIMIT REACHED | OBJECTS REMAIN"
                                auto_execute_next = False
                                frozen = _save_and_report(
                                    processor, args.output_dir, snap_color, snap_depth, snap_K, result,
                                )
                                print(f"[连续任务] 已达步数上限，盒外仍有 {remaining} 个目标；停止。")
                                continue
                            if result.placement_segmentation is None:
                                state = "sequence_paused"
                                status = "NO CONTAINER MASK | MANUAL CHECK"
                                auto_execute_next = False
                                frozen = _save_and_report(
                                    processor, args.output_dir, snap_color, snap_depth, snap_K, result,
                                )
                                print("[连续任务] 未获得容器掩码；禁止连续任务抓取。")
                                continue
                        pending_pick = prepare_pick_fn(
                            command, result, snap_color, snap_depth, snap_K, T_camera,
                            ik_checker, robot_settings, grasp_cfg, pipeline.placement_settings,
                            pose6d_to_mat4(
                                0.0, 0.0, 0.0,
                                float(ready_cfg.get("roll", 0.0)),
                                float(ready_cfg.get("pitch", 1.2)),
                                float(ready_cfg.get("yaw", 0.0)),
                            )[:3, :3],
                        )
                        frozen = _save_and_report(
                            processor, args.output_dir, snap_color, snap_depth, snap_K, result,
                        )
                        if pending_pick is None:
                            state = "sequence_paused" if sequence_active else "idle"
                            status = "NO SAFE EXECUTABLE GRASP"
                            auto_execute_next = False
                        else:
                            state = "pick_ready"
                            if auto_execute_next:
                                execute_requested = True
                                auto_execute_next = False
                            if pending_pick.initial_place is not None:
                                status = (
                                    f"PICK+PLACE READY | approach={pending_pick.approach_tilt_deg:.1f} deg "
                                    "| G/SPACE=EXECUTE"
                                )
                            elif not pending_pick.recheck_after_lift:
                                status = (
                                    f"GRASP-ONLY READY | approach={pending_pick.approach_tilt_deg:.1f} deg "
                                    "| G/SPACE=PICK"
                                )
                            else:
                                status = (
                                    f"PICK READY | approach={pending_pick.approach_tilt_deg:.1f} deg "
                                    "| CONTAINER RECHECK AFTER LIFT | G/SPACE=PICK"
                                )

            try:
                command = commands.get_nowait()
            except queue.Empty:
                command = ""
            else:
                input_pending = False
                if command is None:
                    terminal_enabled = False
                    command = ""
            if command:
                command = normalize_terminal_command(command)
            if command == "/quit":
                break
            if command == "/live":
                if placement_pending_verification:
                    print("[Robot] 放置后夹爪状态尚未通过新图复核；请按 g 重拍，不能清除安全状态。")
                elif holding_object:
                    if state == "fault_holding":
                        print("[Robot] 当前夹持物体且机械臂运动失败，请先人工检查。")
                    else:
                        print("[Robot] 当前仍夹持物体；按 G/Space 重新检查容器，不能开始新抓取。")
                elif state in ("fault", "fault_gripper"):
                    print("[Robot] 故障未排除：先检查机械臂或夹爪，不能恢复新任务。")
                else:
                    generation += 1
                    sequence_active = False
                    progress = SequenceProgress()
                    next_snapshot_pending = False
                    placement_pending_verification = False
                    auto_execute_next = False
                    pending_pick = None
                    state = "idle"
                    frozen = None
                    status = "LIVE | TYPE INSTRUCTION IN TERMINAL"
                command = ""
            elif command == "/retry":
                if state in ("pick_ready", "holding") and not busy:
                    execute_requested = True
                    command = ""
                elif state == "grasp_missed" and not busy:
                    retry_sequence = True
                    command = ""
                elif sequence_mode and state == "sequence_paused" and not busy:
                    if progress.placed >= args.max_steps:
                        print("[连续任务] 已达步数上限；请输入新指令重新开始，不能用 g 继续。")
                    else:
                        retry_sequence = True
                    command = ""
                else:
                    if sequence_mode and state == "sequence_complete":
                        print("[连续任务] 已完成；若有新目标，请明确输入新指令。")
                        command = ""
                    else:
                        command = last_instruction
            if command and not command.startswith("/"):
                if placement_pending_verification:
                    print("[Robot] 放置后夹爪状态尚未通过新图复核；请按 g 重拍或 q 退出，不能开始新抓取。")
                elif holding_object:
                    print("[Robot] 当前仍夹持物体；请先完成放置或人工接管。")
                elif state in ("fault", "fault_gripper"):
                    print("[Robot] 故障未排除：先检查机械臂或夹爪，当前不接受新任务。")
                elif busy:
                    print("[VLM] 上一轮仍在处理，请稍后重试。")
                else:
                    if sequence_mode:
                        sequence_active = True
                        progress = SequenceProgress()
                        next_snapshot_pending = False
                        placement_pending_verification = False
                        auto_execute_next = False
                    pending_pick = None
                    state = "idle"
                    submit(command, color, depth)

            display = frozen.copy() if frozen is not None else color.copy()
            # The dashboard reserves the top 30 px for the panel title.
            status_y = 60 if processor.display_size() is not None else 28
            cv2.putText(display, status, (10, status_y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (255, 220, 80), 2)
            cv2.imshow(window, processor.compose_display(display))
            key = cv2.waitKey(1) & 0xFF
            processor.handle_preview_key(key)
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1 or key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("r"), ord("R")):
                if placement_pending_verification:
                    print("[Robot] 放置后夹爪状态尚未通过新图复核；R 不会清除安全状态，请按 G 重拍。")
                elif state == "grasp_missed":
                    print("[抓取] 已放弃本轮未夹取成功的目标；不执行放置或自动进入下一轮。")
                    generation += 1
                    sequence_active = False
                    progress = SequenceProgress()
                    next_snapshot_pending = False
                    auto_execute_next = False
                    pending_pick = None
                    state = "idle"
                    frozen = None
                    status = "LIVE | TYPE INSTRUCTION IN TERMINAL"
                elif holding_object:
                    if state == "fault_holding":
                        print("[Robot] 当前夹持物体且机械臂运动失败，R 不会清除故障状态。")
                    else:
                        frozen = None
                        status = "HOLDING OBJECT | G/SPACE=RECHECK"
                elif state in ("fault", "fault_gripper"):
                    print("[Robot] 机械臂或夹爪故障未排除，R 不会清除故障状态。")
                else:
                    generation += 1
                    if sequence_mode:
                        sequence_active = False
                        progress = SequenceProgress()
                        next_snapshot_pending = False
                        placement_pending_verification = False
                        auto_execute_next = False
                    pending_pick = None
                    state = "idle"
                    frozen = None
                    status = "LIVE | TYPE INSTRUCTION IN TERMINAL"
            elif (key in (ord("g"), ord("G"), ord(" ")) or execute_requested) and not busy:
                if state == "pick_ready" and pending_pick is not None:
                    initial_place = pending_pick.initial_place
                    recheck_after_lift = pending_pick.recheck_after_lift
                    pick_motion = execute_pick_and_lift(
                        controller, grasp_driver,
                        pending_pick.grasp6d, pending_pick.pregrasp6d, pending_pick.retreat6d,
                        ready_cfg, robot_settings, dry_run=args.dry_run,
                    )
                    if args.dry_run:
                        if initial_place is not None:
                            status = "DRY RUN PICK+PLACE PLAN PASSED | NO MOTION"
                        elif recheck_after_lift:
                            status = "DRY RUN PICK PLAN PASSED | CONTAINER WOULD RECHECK AFTER REAL LIFT"
                        else:
                            status = "DRY RUN GRASP-ONLY PLAN PASSED | NO MOTION"
                    elif pick_motion.failure_stage == "gripper_open":
                        state = "fault_gripper"
                        pending_pick = None
                        frozen = None
                        auto_execute_next = False
                        status = "GRIPPER OPEN FAILED | ARM DID NOT MOVE | MANUAL CHECK"
                        print("[Robot] 夹爪开度不足或反馈未达标；抓取运动未开始，停止连续任务。"
                              "请检查夹爪机械状态，不能直接重试。")
                    elif not pick_motion.grasped:
                        state = "grasp_missed" if pick_motion.at_ready else "fault"
                        pending_pick = None
                        frozen = None
                        status = ("GRASP MISSED | G=RECAPTURE | R=ABANDON" if pick_motion.at_ready else
                                  "PICK FAILED | ARM POSITION UNCONFIRMED | MANUAL CHECK")
                        if pick_motion.at_ready:
                            print("[抓取] 未确认夹住物体，机械臂已返回 ready；未执行放置。"
                                  "输入 g 重拍 RGB-D、重新计算并检查抓取位姿；输入 r 放弃本轮，q 退出。"
                                  "请先现场确认夹爪无物；重试不会复用上次位姿，且需再次确认执行。")
                        else:
                            print("[Robot] 抓取未成功且机械臂未确认回到 ready；禁止开始下一轮。")
                    else:
                        holding_object = True
                        held = HeldObject(
                            pending_pick.command,
                            pending_pick.object_points_camera,
                            pending_pick.T_camera_to_base,
                            pending_pick.grasp6d,
                        )
                        state = "holding"
                        pending_pick = None
                        if not pick_motion.at_ready:
                            state = "fault_holding"
                            status = "HOLDING OBJECT | READY MOTION FAILED | MANUAL CHECK"
                            print("[Robot] 已夹住物体，但返回 ready 失败；保持夹持，不进行视觉复查或放置。")
                        else:
                            if initial_place is not None:
                                print("[容器流程] 第一张 RGB-D 已确认容器和放置位姿；跳过第二次拍照，继续放置。")
                                run_verified_place(initial_place)
                            elif not recheck_after_lift:
                                print("[容器流程] 已按指令完成只抓取并抬高；保持夹持，不执行放置。")
                                status = "GRASP-ONLY COMPLETE | HOLDING OBJECT"
                            else:
                                try:
                                    rechecked_place, frozen, status = _recheck_after_lift(
                                        held, camera, K, grasp_driver, T_hand_eye, cfg, pipeline, processor,
                                        args.output_dir, ik_checker, robot_settings, min_base_z,
                                    )
                                    if rechecked_place is not None:
                                        run_verified_place(rechecked_place)
                                except (VLMError, ValueError, RuntimeError) as exc:
                                    print(f"[抬高复查失败] {exc}；保持夹持。")
                                    status = "HOLDING OBJECT | RECHECK ERROR | G=RETRY"
                elif state == "holding" and held is not None:
                    try:
                        rechecked_place, frozen, status = _recheck_after_lift(
                            held, camera, K, grasp_driver, T_hand_eye, cfg, pipeline, processor,
                            args.output_dir, ik_checker, robot_settings, min_base_z,
                        )
                        if rechecked_place is not None:
                            run_verified_place(rechecked_place)
                    except (VLMError, ValueError, RuntimeError) as exc:
                        print(f"[抬高复查失败] {exc}；保持夹持。")
                        status = "HOLDING OBJECT | RECHECK ERROR | G=RETRY"
                elif state == "grasp_missed":
                    retry_sequence = True
                elif last_instruction:
                    if state in ("fault", "fault_gripper"):
                        print("[Robot] 故障未排除：先检查机械臂或夹爪，当前不接受新任务。")
                    elif state == "sequence_paused":
                        if progress.placed >= args.max_steps:
                            print("[连续任务] 已达步数上限；请输入新指令重新开始。")
                        else:
                            retry_sequence = True
                    elif state == "sequence_complete":
                        print("[连续任务] 已完成；若有新目标，请明确输入新指令。")
                    elif sequence_mode and not sequence_active:
                        print("[连续任务] 请明确输入新指令开始新任务。")
                    else:
                        submit(last_instruction, color, depth)
                else:
                    print("[VLM] 请先在终端输入中文指令。")

            if retry_sequence and not busy:
                try:
                    fresh_color, fresh_depth = _capture_fresh(camera)
                    auto_execute_next = False
                    pending_pick = None
                    submit(last_instruction, fresh_color, fresh_depth)
                    state = "idle"
                    print("[重试] 保留原指令和进度；重新拍照并执行视觉与安全检查。"
                          "若检查通过，请再次按 g 执行。")
                except RuntimeError as exc:
                    state = "grasp_missed" if state == "grasp_missed" else "sequence_paused"
                    status = "FRESH FRAME FAILED | G=RETRY"
                    print(f"[重试] 重新拍照失败：{exc}；按 g 重试、r 放弃本轮或 q 退出。")

            if terminal_enabled and not busy and not input_pending and state in (
                    "idle", "pick_ready", "holding", "grasp_missed", "sequence_paused", "sequence_complete",
                    "fault", "fault_gripper"):
                input_pending = True
                terminal_guard.restore()
                if state == "pick_ready":
                    prompt = "\n当前任务已通过安全检查。输入 g 执行，r 放弃本轮，q 退出并回车："
                elif state == "holding":
                    prompt = "\n当前仍夹持物体。调整容器后输入 g 重新检查，q 退出并回车："
                elif state == "grasp_missed":
                    prompt = "\n本轮未确认夹取成功，已返回 ready；确认夹爪无物后，g=重拍重新规划，r=放弃，q=退出："
                elif state in ("sequence_paused", "sequence_complete"):
                    prompt = ("\n放置后仍待复核；g=重拍，q=退出，不能开始新抓取："
                              if placement_pending_verification else
                              "\n连续任务已暂停；g=保留原指令重拍，r=放弃本轮，q=退出；也可输入新指令："
                              if state == "sequence_paused" and progress.placed < args.max_steps else
                              "\n连续任务已完成或达到步数上限；请输入新指令重新开始，r=预览，q=退出：")
                elif state in ("fault", "fault_gripper"):
                    prompt = "\n机械臂或夹爪故障已停止任务；请人工检查，输入 q 退出并回车："
                else:
                    prompt = "\n请输入下一条中文指令（r=预览，g=重试，q=退出）并回车："
                start_terminal_input(commands, prompt=prompt)
    finally:
        if holding_object:
            print("\n[Exit] 当前夹爪仍持有物体；程序不会自动松爪，避免物体掉落。")
        else:
            print("\n[Exit] 检查夹爪零位并返回 Home。")
        try:
            if controller is not None and getattr(controller, "_running", False):
                if (grasp_driver is not None and not holding_object
                        and not placement_pending_verification
                        and state not in ("fault", "fault_gripper", "fault_holding")):
                    try:
                        grasp_driver.home_gripper_if_needed()
                    except Exception as exc:
                        print(f"[Exit] 夹爪归零失败：{exc}；仍尝试机械臂 Home。")
                controller.end()
            elif rebotarm is not None:
                rebotarm.disconnect()
        except Exception as exc:
            print(f"[Exit] Robot: {exc}")
        if camera is not None:
            try:
                camera.close()
            except Exception:
                pass
        if processor is not None:
            processor.close_preview()
        cv2.destroyAllWindows()
        terminal_guard.restore()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.")
        raise SystemExit(130)
    except (VLMError, ValueError, OSError, RuntimeError, KeyError, ImportError, cv2.error) as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        raise SystemExit(2)
