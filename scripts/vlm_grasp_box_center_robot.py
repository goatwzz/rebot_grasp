"""Box-center placement baseline: VLM box, center depth, fixed TCP pose.

Unlike vlm_grasp_robot.py, this experiment never estimates the container
opening, orientation, or object fit. The fixed release pose follows set.py.
Run --dry-run first; IK checks do not prove the box opening is collision-free.
"""

from __future__ import annotations

import json
import importlib
from pathlib import Path
import sys

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.vlm_grasp_demo import GraspPreviewProcessor  # noqa: E402
from utils import placement_geometry  # noqa: E402
from utils.place_utils import PLACE_RPY, PLACE_Z_M  # noqa: E402
from utils.vlm_grasp_pipeline import VLMGraspPipeline  # noqa: E402
from utils.vlm_utils import VLMError  # noqa: E402


class BoxCenterPipeline(VLMGraspPipeline):
    def __init__(self, cfg, client, **kwargs):
        baseline_cfg = dict(cfg)
        baseline_cfg["vlm_grasp"] = dict(cfg.get("vlm_grasp") or {})
        baseline_cfg["vlm_grasp"]["placement_preview"] = {
            **((cfg.get("vlm_grasp") or {}).get("placement_preview") or {}),
            "enabled": False,
        }
        super().__init__(baseline_cfg, client, **kwargs)


def box_center_camera(bbox_xyxy, depth_mm, K, *, radius_px=5):
    x1, y1, x2, y2 = (int(value) for value in bbox_xyxy)
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    if not (0 <= center_x < depth_mm.shape[1] and 0 <= center_y < depth_mm.shape[0]):
        raise ValueError("VLM 容器框中心超出深度图")
    left, right = max(x1, center_x - radius_px), min(x2 + 1, center_x + radius_px + 1)
    top, bottom = max(y1, center_y - radius_px), min(y2 + 1, center_y + radius_px + 1)
    patch = np.asarray(depth_mm[top:bottom, left:right])
    valid = patch[(patch >= 50) & (patch <= 1000)]
    if not len(valid):
        raise ValueError("VLM 容器框中心附近没有 0.05–1.0 m 的有效深度")
    depth_m = float(np.median(valid)) / 1000.0
    point = np.linalg.solve(np.asarray(K, dtype=np.float64), [center_x, center_y, 1.0]) * depth_m
    if not np.isfinite(point).all():
        raise ValueError("VLM 容器框中心反投影无效")
    return (center_x, center_y), point, int(len(valid))


def fixed_place_poses(box_center_camera_m, T_camera_to_base, approach_offset_m):
    transform = np.asarray(T_camera_to_base, dtype=np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("相机到基座的变换无效")
    point_base = transform @ np.append(np.asarray(box_center_camera_m, dtype=np.float64), 1.0)
    if not np.isfinite(point_base).all() or abs(point_base[3] - 1.0) > 1e-6:
        raise ValueError("容器中心基座坐标无效")
    release = (float(point_base[0]), float(point_base[1]), PLACE_Z_M, *PLACE_RPY)
    preplace = (release[0], release[1], PLACE_Z_M + approach_offset_m, *PLACE_RPY)
    return release, preplace


def prepare_box_center_pick(command, result, color, depth_mm, K, T_camera_to_base,
                            ik_checker, robot_settings, grasp_cfg, placement_settings):
    robot = importlib.import_module("scripts.vlm_grasp_robot")
    placement = result.grounding.placement
    if placement is None or not placement.required or not placement.found or placement.relation != "in":
        print("[框中心对照] VLM 未确认可从上方放入的容器；不执行抓取或放置。")
        return None
    if result.segmentation is None or result.grasp_result is None or len(result.grasp_result.grasps) == 0:
        print("[框中心对照] 没有通过筛选的抓取候选。")
        return None
    exclusion_masks = robot._combined_exclusion_masks(result)
    box_exclusion = np.zeros(depth_mm.shape, dtype=bool)
    x1, y1, x2, y2 = placement.bbox_xyxy
    box_exclusion[y1:y2 + 1, x1:x2 + 1] = True
    exclusion_masks.append(box_exclusion)
    table_normal = placement_geometry.estimate_table_normal(
        depth_mm,
        K,
        exclusion_masks=exclusion_masks,
        settings=placement_settings,
        expected_normal_camera=(
            np.asarray(T_camera_to_base, dtype=np.float64)[:3, :3].T
            @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
        ),
    )
    pending = robot._finish_prepare_pick(
        command, result, color, depth_mm, K, T_camera_to_base, ik_checker,
        robot_settings, grasp_cfg, table_normal,
    )
    if pending is None:
        return None
    try:
        pixel, center_camera, depth_samples = box_center_camera(placement.bbox_xyxy, depth_mm, K)
        release, preplace = fixed_place_poses(
            center_camera, T_camera_to_base, robot_settings.place_approach_offset_m,
        )
        pose_ok, reason = robot._pose_checks(
            ik_checker, release, preplace, float(grasp_cfg.get("min_base_z_m", 0.03)),
        )
    except ValueError as exc:
        print(f"[框中心对照] {exc}；不执行抓取或放置。")
        return None
    result.box_center_plan = {
        "method": "vlm_bbox_center_fixed_tcp",
        "bbox_xyxy": list(placement.bbox_xyxy),
        "pixel_xy": list(pixel),
        "depth_samples": depth_samples,
        "center_camera_m": center_camera.tolist(),
        "release_base_pose6d": list(release),
        "preplace_base_pose6d": list(preplace),
        "fixed_release_z_m": PLACE_Z_M,
        "fixed_release_rpy_rad": list(PLACE_RPY),
        "ik_valid": pose_ok,
        "ik_status": reason,
        "container_pose_estimated": False,
        "fit_or_collision_checked": False,
    }
    if not pose_ok:
        print(f"[框中心对照] {reason}；不执行抓取或放置。")
        return None
    pending.initial_place = robot.PendingPlace(result, release, preplace)
    pending.recheck_after_lift = False
    print(f"[框中心对照] 框中心像素={pixel}，深度样本={depth_samples}，"
          f"基座放置 XY=({release[0]:.3f}, {release[1]:.3f}) m，"
          f"固定 Z={PLACE_Z_M:.3f} m、RPY={PLACE_RPY}；{reason}。")
    return pending


class BoxCenterPreviewProcessor(GraspPreviewProcessor):
    placement_panel_title = "FIXED-POSE BOX CENTER"

    def draw(self, image, result):
        display = super().draw(image, result)
        placement = result.grounding.placement
        if placement is not None and placement.required and placement.found:
            x1, y1, x2, y2 = placement.bbox_xyxy
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            cv2.drawMarker(display, center, (255, 255, 0), cv2.MARKER_CROSS, 18, 2)
            cv2.putText(display, "BOX CENTER", (max(0, center[0] - 55), max(18, center[1] - 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
        return display

    def status(self, result):
        plan = getattr(result, "box_center_plan", None)
        if plan is not None:
            return "BOX-CENTER BASELINE | READY" if plan["ik_valid"] else "BOX-CENTER BASELINE | IK FAILED"
        return "BOX-CENTER BASELINE | NO VERIFIED PLACE"

    def show_result(self, image, result):
        super().show_result(image, result)
        panel = np.full(image.shape, 18, dtype=np.uint8)
        plan = getattr(result, "box_center_plan", None)
        lines = ["BOX-CENTER BASELINE", "No container pose or fit estimation"]
        if plan is not None:
            release = plan["release_base_pose6d"]
            lines.extend((
                f"VLM center: {plan['pixel_xy']}",
                f"Base XY: {release[0]:+.3f}, {release[1]:+.3f} m",
                f"Fixed Z: {release[2]:.3f} m",
                f"Fixed RPY: {tuple(release[3:])}",
                "IK: PASS" if plan["ik_valid"] else "IK: FAIL",
            ))
        else:
            lines.append("NO VERIFIED TARGET")
        for index, line in enumerate(lines):
            cv2.putText(panel, line, (16, 58 + 32 * index), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (220, 220, 220), 1, cv2.LINE_AA)
        self._placement_frame = panel

    def save(self, output_dir, image, result, *, depth_mm=None, K=None):
        directory = super().save(output_dir, image, result, depth_mm=depth_mm, K=K)
        plan = getattr(result, "box_center_plan", None)
        (directory / "box_center_baseline.json").write_text(
            json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
        )
        return directory

    def report(self, result, directory):
        super().report(result, directory)
        plan = getattr(result, "box_center_plan", None)
        if plan is not None:
            print(f"[框中心对照] 固定放置位姿={plan['release_base_pose6d']}；"
                  f"IK={'通过' if plan['ik_valid'] else '失败'}；未检查容器开口和碰撞。")

    def window_title(self):
        return "VLM Box-Center Baseline"


def main():
    robot = importlib.import_module("scripts.vlm_grasp_robot")
    return robot.main(
        pipeline_factory=BoxCenterPipeline,
        processor_factory=BoxCenterPreviewProcessor,
        prepare_pick_fn=prepare_box_center_pick,
        default_output_dir=PROJECT_ROOT / "outputs/vlm_box_center",
        window_title="VLM Box-Center Baseline - REAL CONTROL",
        workflow_description=(
            "[框中心对照] VLM 框中心深度定位 → 固定基座 Z/RPY 放置；"
            "不计算盒子姿态、开口、物体 fit 或盒壁碰撞。"
        ),
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.")
        raise SystemExit(130)
    except (VLMError, ValueError, OSError, RuntimeError, KeyError, ImportError, cv2.error) as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        raise SystemExit(2)
