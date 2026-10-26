"""Stage 2: VLM + dual SAM masks + grasp/place geometry previews; no robot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

OPEN3D_CONFIG_MODES = ("best", "width")

import cv2
import numpy as np

from scripts.vlm_target_demo import TargetPreviewProcessor, report_result, run_camera, run_image, save_result
from utils.camera_utils import load_config
from utils.placement_geometry import PlacementPreviewSettings
from utils.sam_utils import SAMSettings
from utils.vlm_grasp_pipeline import VLMGraspPipeline, grasp_record
from utils.vlm_utils import VLMClient, VLMError, VLMSettings, draw_grounding


class GraspPreviewProcessor(TargetPreviewProcessor):
    MASK_WINDOW = "SAM Masks (white=pick + container)"
    placement_panel_title = "PLACEMENT GEOMETRY"
    show_status_overlay = False

    def __init__(self, pipeline: VLMGraspPipeline, *, show_mask=True, show_open3d=True,
                 top_k=50, open3d_grasps="width", dashboard=True,
                 dashboard_tile_width=640, dashboard_tile_height=360,
                 grasp_cloud_focus_radius_m=0.12) -> None:
        self.pipeline = pipeline
        self.show_mask = show_mask
        self.show_open3d = show_open3d
        self.top_k = top_k
        self.open3d_grasps = open3d_grasps
        self.dashboard = bool(dashboard)
        self.dashboard_tile_width = int(dashboard_tile_width)
        self.dashboard_tile_height = int(dashboard_tile_height)
        self.grasp_cloud_focus_radius_m = float(grasp_cloud_focus_radius_m)
        if self.dashboard_tile_width < 160 or self.dashboard_tile_height < 120:
            raise ValueError("dashboard tile_width/tile_height 太小，至少需要 160×120")
        if self.grasp_cloud_focus_radius_m <= 0:
            raise ValueError("grasp_cloud_focus_radius_m 必须大于 0")
        self._mask_created = False
        self._viewer = None
        self._placement_viewer = None
        self._shown_result = None
        self._shown_placement_result = None
        self._mask_frame = None
        self._grasp_frame = None
        self._placement_frame = None
        self._focused_panel = None
        self._drag_panel = None
        self._drag_button = None
        self._last_mouse = None

    @staticmethod
    def combined_mask(image_shape, pick_segmentation=None, placement_segmentation=None,
                      rejected_container_segmentation=None):
        combined = np.zeros(image_shape, dtype=np.uint8)
        if pick_segmentation is not None:
            combined[np.asarray(pick_segmentation.mask, dtype=bool)] = 255
        if placement_segmentation is not None:
            combined[np.asarray(placement_segmentation.mask, dtype=bool)] = 255
        if rejected_container_segmentation is not None:
            combined[np.asarray(rejected_container_segmentation.mask, dtype=bool)] = 255
        return combined

    @staticmethod
    def _container_context_points(result, grasp_result):
        """Find container points in the full scene cloud, with VLM boxes as fallback."""
        analysis = getattr(result, "placement_analysis", None)
        if analysis is not None:
            points = np.asarray(analysis.container.container_points_camera, dtype=np.float64)
            if points.ndim == 2 and points.shape[1] == 3 and len(points):
                return points
        segmentations = [
            getattr(result, "placement_segmentation", None),
            getattr(result, "rejected_container_segmentation", None),
        ]
        segmentations = [item for item in segmentations if item is not None]
        placement = getattr(getattr(result, "grounding", None), "placement", None)
        boxes = []
        if placement is not None:
            if placement.required and placement.found and placement.bbox_xyxy is not None:
                boxes.append(placement.bbox_xyxy)
            rejected = placement.rejected_container
            if rejected is not None:
                boxes.append(rejected.bbox_xyxy)
        if not segmentations and not boxes:
            return None
        points = np.asarray(grasp_result.o3d_cloud.points, dtype=np.float64)
        K = np.asarray(result.camera_matrix, dtype=np.float64)
        valid = np.isfinite(points).all(axis=1) & (points[:, 2] > 0)
        selected = np.zeros(len(points), dtype=bool)
        indices = np.flatnonzero(valid)
        if len(indices) == 0:
            return None
        projected = points[indices]
        u = np.rint(K[0, 0] * projected[:, 0] / projected[:, 2] + K[0, 2]).astype(np.int64)
        v = np.rint(K[1, 1] * projected[:, 1] / projected[:, 2] + K[1, 2]).astype(np.int64)
        for segmented in segmentations:
            mask = np.asarray(segmented.mask, dtype=bool)
            inside = (u >= 0) & (u < mask.shape[1]) & (v >= 0) & (v < mask.shape[0])
            local = np.zeros(len(indices), dtype=bool)
            local[inside] = mask[v[inside], u[inside]]
            selected[indices[local]] = True
        for x1, y1, x2, y2 in boxes:
            selected[indices[(u >= x1) & (u <= x2) & (v >= y1) & (v <= y2)]] = True
        return points[selected] if selected.any() else None

    def _display_mask(self, mask):
        if not self.show_mask:
            self._mask_frame = None
            return
        self._mask_frame = cv2.cvtColor(np.asarray(mask, dtype=np.uint8), cv2.COLOR_GRAY2BGR)
        if self.dashboard:
            return
        if not self._mask_created:
            cv2.namedWindow(self.MASK_WINDOW, cv2.WINDOW_NORMAL)
            height, width = mask.shape
            display_width = min(800, width)
            cv2.resizeWindow(
                self.MASK_WINDOW,
                display_width,
                max(1, round(height * display_width / width)),
            )
            self._mask_created = True
        cv2.imshow(self.MASK_WINDOW, mask)

    def reset_preview(self, image_shape, status):
        self._shown_result = None
        self._shown_placement_result = None
        self._grasp_frame = None
        self._placement_frame = None
        if self.show_mask:
            blank = np.zeros(image_shape, dtype=np.uint8)
            cv2.putText(blank, status, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 1)
            self._display_mask(blank)
        if self._viewer is not None:
            self._viewer.clear()
        if self._placement_viewer is not None:
            self._placement_viewer.clear()

    def show_result(self, image, result):
        self._grasp_frame = None
        self._placement_frame = None
        placement_segmentation = getattr(result, "placement_segmentation", None)
        rejected_segmentation = getattr(result, "rejected_container_segmentation", None)
        if result.segmentation is None and placement_segmentation is None and rejected_segmentation is None:
            self.reset_preview(image.shape[:2], "NO TARGET / NO MASK")
            return
        placement_analysis = getattr(result, "placement_analysis", None)
        self._display_mask(self.combined_mask(
            image.shape[:2], result.segmentation, placement_segmentation, rejected_segmentation,
        ))
        self._shown_placement_result = None
        if self.show_open3d and placement_analysis is not None:
            try:
                if self._placement_viewer is None:
                    self._placement_viewer = self.pipeline.placement_backend.Open3DPlacementWindow(
                        visible=not self.dashboard,
                    )
                self._placement_viewer.update(placement_analysis)
                if self.dashboard:
                    self._placement_frame = self._placement_viewer.frame()
                self._shown_placement_result = result
            except RuntimeError as exc:
                print(f"[放置3D] 无法显示容器点云窗口：{exc}；二维放置预览继续。", flush=True)
                self._placement_frame = None
                if self._placement_viewer is not None:
                    self._placement_viewer.close()
                    self._placement_viewer = None
        elif self._placement_viewer is not None:
            self._placement_viewer.clear()
            self._placement_frame = None
        g = result.grasp_result
        self._shown_result = None
        if g is None:
            if self._viewer is not None:
                self._viewer.clear()
            self._grasp_frame = None
            return
        if self.show_open3d:
            try:
                if self._viewer is None:
                    if self.open3d_grasps == "best":
                        title, limit, apply_nms = "VLM GraspNet - Selected Grasp", 1, False
                    elif self.open3d_grasps in ("width", "final"):
                        title = ("VLM GraspNet - All Raw Decoded Grasps"
                                 if self.open3d_grasps == "width" else
                                 "VLM GraspNet - All Final Grasps")
                        limit, apply_nms = None, False
                    else:
                        title, limit, apply_nms = "VLM GraspNet - Diagnostic Candidates", self.top_k, True
                    self._viewer = self.pipeline.backend.Open3DGraspWindow(
                        title, limit, apply_nms=apply_nms, visible=not self.dashboard,
                        focus_radius_m=self.grasp_cloud_focus_radius_m,
                    )
                display_mode = "raw" if self.open3d_grasps == "width" else self.open3d_grasps
                candidates = self.pipeline.backend.visualization_grasps(g, display_mode)
                container_points = self._container_context_points(result, g)
                self._viewer.update(g.o3d_cloud, candidates, context_points=container_points)
                if self.dashboard:
                    self._grasp_frame = self._viewer.frame()
                self._shown_result = result
                if self.open3d_grasps == "best":
                    print("[3D] 仅显示本轮最终选中的 1 个抓取位姿。" if len(candidates) else
                          "[3D] 本轮没有合格的最终抓取位姿，点云中不显示夹爪。", flush=True)
                elif self.open3d_grasps == "width":
                    print(f"[3D] 显示 GraspNet 解码后、碰撞和目标筛选前的全部原始候选：{len(candidates)} 个；仅供诊断。",
                          flush=True)
                elif self.open3d_grasps == "final":
                    print(f"[3D] 显示本轮全部最终候选：{len(candidates)} 个。", flush=True)
                else:
                    print(f"[3D] {self.open3d_grasps} 候选={len(candidates)}，去重后最多显示 {self.top_k} 个。", flush=True)
            except RuntimeError as exc:
                print(f"[3D] 无法显示点云窗口：{exc}；二维预览继续。", flush=True)
                self._grasp_frame = None
                if self._viewer is not None:
                    self._viewer.close()
                    self._viewer = None
                self.show_open3d = False

    @staticmethod
    def _fit_image(image, width, height):
        """Letterbox an image without stretching it."""
        frame = np.asarray(image)
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        elif frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("dashboard panel must be a gray or BGR image")
        scale = min(width / frame.shape[1], height / frame.shape[0])
        resized = cv2.resize(
            frame,
            (max(1, round(frame.shape[1] * scale)), max(1, round(frame.shape[0] * scale))),
            interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,
        )
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        y = (height - resized.shape[0]) // 2
        x = (width - resized.shape[1]) // 2
        canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
        return canvas

    def _dashboard_tile(self, image, title, empty_text, *, width=None, height=None):
        width = self.dashboard_tile_width if width is None else int(width)
        height = self.dashboard_tile_height if height is None else int(height)
        title_height = min(30, max(22, height // 10))
        canvas = np.full((height, width, 3), 18, dtype=np.uint8)
        if image is None:
            size, _ = cv2.getTextSize(empty_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.putText(
                canvas, empty_text,
                (max(8, (width - size[0]) // 2), max(title_height + 24, (height + size[1]) // 2)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 150, 150), 1, cv2.LINE_AA,
            )
        else:
            # Keep the full 16:9 tile for the image.  The compact title bar is
            # overlaid so four 16:9 sources remain as large as possible.
            canvas = self._fit_image(image, width, height)
        cv2.rectangle(canvas, (0, 0), (width - 1, title_height - 1), (38, 38, 38), -1)
        cv2.putText(canvas, title, (10, title_height - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.52, (245, 245, 245), 1, cv2.LINE_AA)
        cv2.rectangle(canvas, (0, 0), (width - 1, height - 1), (80, 80, 80), 1)
        return canvas

    def compose_display(self, main_image):
        """Build one equal-sized 2x2 preview instead of four visible windows."""
        if not self.dashboard:
            return main_image
        mask_empty = "MASK DISABLED" if not self.show_mask else "WAITING FOR MASK"
        cloud_empty = "3D PREVIEW DISABLED" if not self.show_open3d else "NO GRASP CLOUD"
        place_empty = "3D PREVIEW DISABLED" if not self.show_open3d else "NO PLACEMENT GEOMETRY"
        panels = (
            (main_image, "RGB / VLM DETECTIONS", "WAITING FOR RGB"),
            (self._mask_frame, "PICK + PLACE MASKS", mask_empty),
            (self._grasp_frame, "GRASPNET GRASP CLOUD", cloud_empty),
            (self._placement_frame, self.placement_panel_title, place_empty),
        )
        if self._focused_panel is not None:
            image, title, empty = panels[self._focused_panel]
            return self._dashboard_tile(
                image, title + " | DOUBLE CLICK OR 0: ALL VIEWS", empty,
                width=self.dashboard_tile_width * 2,
                height=self.dashboard_tile_height * 2,
            )
        tiles = tuple(self._dashboard_tile(*panel) for panel in panels)
        return cv2.vconcat((cv2.hconcat(tiles[:2]), cv2.hconcat(tiles[2:])))

    def display_size(self):
        if not self.dashboard:
            return None
        return self.dashboard_tile_width * 2, self.dashboard_tile_height * 2

    def bind_window(self, window):
        if self.dashboard:
            cv2.setMouseCallback(window, self.handle_mouse)

    def handle_preview_key(self, key):
        if not self.dashboard:
            return False
        if ord("1") <= key <= ord("4"):
            self._focused_panel = key - ord("1")
            return True
        if key == ord("0"):
            self._focused_panel = None
            return True
        return False

    def _panel_at(self, x, y):
        if self._focused_panel is not None:
            return self._focused_panel
        column = 1 if x >= self.dashboard_tile_width else 0
        row = 1 if y >= self.dashboard_tile_height else 0
        return row * 2 + column

    @staticmethod
    def _wheel_steps(flags):
        # OpenCV packs a signed wheel delta in the high 16 bits.  Some Linux
        # backends instead pass the signed value directly.
        raw = (int(flags) >> 16) & 0xFFFF
        if raw >= 0x8000:
            raw -= 0x10000
        if raw == 0 and flags:
            raw = 1 if flags > 0 else -1
        return raw / 120.0 if abs(raw) >= 120 else float(np.sign(raw))

    def _viewer_for_panel(self, panel):
        if panel == 2:
            return self._viewer, "_grasp_frame"
        if panel == 3:
            return self._placement_viewer, "_placement_frame"
        return None, None

    def _refresh_3d_panel(self, panel):
        viewer, frame_name = self._viewer_for_panel(panel)
        if viewer is None:
            return
        try:
            setattr(self, frame_name, viewer.frame())
        except RuntimeError as exc:
            print(f"[3D交互] 刷新视图失败：{exc}", flush=True)

    def handle_mouse(self, event, x, y, flags, _param=None):
        """Route OpenCV dashboard mouse gestures to the hidden Open3D view."""
        if not self.dashboard:
            return
        panel = self._panel_at(x, y)
        if event == cv2.EVENT_LBUTTONDBLCLK:
            self._focused_panel = None if self._focused_panel is not None else panel
            self._drag_panel = self._drag_button = self._last_mouse = None
            return
        if event in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
            viewer, _ = self._viewer_for_panel(panel)
            if viewer is not None:
                self._drag_panel = panel
                self._drag_button = "rotate" if event == cv2.EVENT_LBUTTONDOWN else "pan"
                self._last_mouse = (x, y)
            return
        if event in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP):
            self._drag_panel = self._drag_button = self._last_mouse = None
            return
        if event == cv2.EVENT_MOUSEWHEEL:
            viewer, _ = self._viewer_for_panel(panel)
            steps = self._wheel_steps(flags)
            if viewer is not None and steps:
                viewer.zoom(steps)
                self._refresh_3d_panel(panel)
            return
        if event != cv2.EVENT_MOUSEMOVE or self._drag_panel is None or self._last_mouse is None:
            return
        viewer, _ = self._viewer_for_panel(self._drag_panel)
        if viewer is None:
            return
        dx, dy = x - self._last_mouse[0], y - self._last_mouse[1]
        self._last_mouse = (x, y)
        if not dx and not dy:
            return
        if self._drag_button == "rotate":
            viewer.rotate(dx, dy)
        else:
            viewer.pan(dx, dy)
        self._refresh_3d_panel(self._drag_panel)

    def poll_preview(self):
        if self._mask_created and cv2.getWindowProperty(self.MASK_WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            self._mask_created = False
        if self._viewer is not None and not self._viewer.poll():
            self._viewer.close()
            self._viewer = None
            self._shown_result = None
            self._grasp_frame = None
        if self._placement_viewer is not None and not self._placement_viewer.poll():
            self._placement_viewer.close()
            self._placement_viewer = None
            self._shown_placement_result = None
            self._placement_frame = None

    def close_preview(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        if self._placement_viewer is not None:
            self._placement_viewer.close()
            self._placement_viewer = None
        self._shown_result = None
        self._shown_placement_result = None
        self._mask_frame = None
        self._grasp_frame = None
        self._placement_frame = None
        if self._mask_created:
            try:
                cv2.destroyWindow(self.MASK_WINDOW)
            except cv2.error:
                pass  # The user may already have closed the mask window.
            self._mask_created = False

    def process(self, command, color, depth_mm, K):
        return self.pipeline.process(command, color, depth_mm, K)

    def draw(self, image, result):
        display = image.copy()
        if result.placement_segmentation is not None:
            display = self.pipeline.placement_backend.draw_placement_preview(
                display,
                result.placement_segmentation.mask,
                result.placement_analysis,
                result.camera_matrix,
            )
        analysis = result.placement_analysis
        plan = analysis.plan if analysis is not None else None
        blocked = bool(result.placement_error) or (plan is not None and not plan.fits)
        return draw_grounding(display, result.grounding, mark_center=False,
                              placement_blocked=blocked, show_warning=False,
                              show_pick=not getattr(result, "placement_only", False))

    def save(self, output_dir, image, result, *, depth_mm=None, K=None):
        directory = save_result(output_dir, image, result.grounding, depth_mm=depth_mm, K=K)
        if not cv2.imwrite(str(directory / "annotated.png"), self.draw(image, result)):
            raise OSError("无法保存抓取物与容器标注图")
        if result.segmentation is not None:
            if not cv2.imwrite(str(directory / "mask.png"), result.segmentation.mask.astype(np.uint8) * 255):
                raise OSError("无法保存目标掩码")
        if result.placement_segmentation is not None:
            placement_mask = result.placement_segmentation.mask.astype(np.uint8) * 255
            if not cv2.imwrite(str(directory / "placement_mask.png"), placement_mask):
                raise OSError("无法保存放置目标掩码")
        rejected_segmentation = getattr(result, "rejected_container_segmentation", None)
        if rejected_segmentation is not None:
            rejected_mask = rejected_segmentation.mask.astype(np.uint8) * 255
            if not cv2.imwrite(str(directory / "rejected_container_mask.png"), rejected_mask):
                raise OSError("无法保存不可用容器掩码")
        if result.segmentation is not None or result.placement_segmentation is not None or rejected_segmentation is not None:
            combined_mask = self.combined_mask(
                image.shape[:2], result.segmentation, result.placement_segmentation, rejected_segmentation,
            )
            if not cv2.imwrite(str(directory / "combined_mask.png"), combined_mask):
                raise OSError("无法保存抓取物与容器合并掩码")
        if result.placement_analysis is not None:
            interior_mask = result.placement_analysis.container.interior_mask.astype(np.uint8) * 255
            if not cv2.imwrite(str(directory / "placement_interior_mask.png"), interior_mask):
                raise OSError("无法保存容器内部轮廓掩码")
        g = result.grasp_result
        if g is not None:
            np.save(directory / "grasp_candidates.npy", g.grasps.grasp_group_array, allow_pickle=False)
        metadata = {
            "status": self.status(result), "elapsed_s": result.elapsed_s,
            "placement_only": bool(getattr(result, "placement_only", False)),
            "target_name": result.grounding.target_name, "bbox_xyxy": result.grounding.bbox_xyxy,
            "placement": result.grounding.to_dict()["placement"],
            "targets_complete": result.grounding.targets_complete,
            "grasp_available": g is not None and g.best is not None,
            "sam_area_px": result.segmentation.area_px if result.segmentation else 0,
            "rejected_container_sam_area_px": rejected_segmentation.area_px if rejected_segmentation else 0,
            "rejected_container_sam_error": getattr(result, "rejected_container_error", None),
            "candidate_count": len(g.grasps) if g is not None else 0,
            "filter_counts": getattr(g, "filter_counts", None) if g is not None else None,
            "best_grasp": grasp_record(g.best) if g is not None else None,
            "vlm_model": self.pipeline.client.settings.model,
            "sam_checkpoint": str(self.pipeline.segmenter.settings.checkpoint),
            "graspnet_constraints": self.pipeline.options,
            "open3d_grasps": self.open3d_grasps,
            "dashboard": {
                "enabled": self.dashboard,
                "tile_width": self.dashboard_tile_width,
                "tile_height": self.dashboard_tile_height,
                "grasp_cloud_focus_radius_m": self.grasp_cloud_focus_radius_m,
            },
            "placement_geometry": self.pipeline.placement_backend.placement_geometry_record(
                result.placement_analysis, result.placement_error,
            ),
        }
        (directory / "pipeline.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        placement_record = self.pipeline.placement_backend.placement_geometry_record(
            result.placement_analysis, result.placement_error,
        )
        (directory / "placement_geometry.json").write_text(
            json.dumps(placement_record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
        )
        if self._viewer is not None and self._shown_result is result:
            try:
                self._viewer.capture(directory / "grasp_cloud.png")
            except RuntimeError as exc:
                print(f"[3D] 点云截图保存失败：{exc}", flush=True)
        if self._placement_viewer is not None and self._shown_placement_result is result:
            try:
                self._placement_viewer.capture(directory / "placement_cloud.png")
            except RuntimeError as exc:
                print(f"[放置3D] 容器点云截图保存失败：{exc}", flush=True)
        if self.dashboard:
            # Saved diagnostics always contain all four panels even when the
            # operator currently has one panel enlarged in the live window.
            focused = self._focused_panel
            self._focused_panel = None
            try:
                dashboard = self.compose_display(self.draw(image, result))
            finally:
                self._focused_panel = focused
            if not cv2.imwrite(str(directory / "dashboard.png"), dashboard):
                raise OSError("无法保存四画面总览图")
        return directory

    def report(self, result, directory):
        if getattr(result, "placement_only", False):
            placement = result.grounding.placement
            print("\n[VLM] 抬高复查：本轮只识别容器，不重新识别或分割抓取物。")
            if placement is not None and placement.required and placement.found:
                relation = {"in": "放入", "on": "放到上面", "near": "放到旁边"}[placement.relation]
                print(f"[VLM] 放置目标：{placement.target_name}（{relation}）")
                print("[VLM] 放置框 [x1,y1,x2,y2]：", placement.bbox_xyxy)
                print("[VLM] 放置目标说明：", placement.reason)
            elif placement is not None:
                print("[VLM] 放置目标未确认：", placement.reason)
                if placement.rejected_container is not None:
                    rejected = placement.rejected_container
                    print(f"[VLM] 不可用容器：{rejected.target_name}；框={rejected.bbox_xyxy}；{rejected.reason}")
            print("[VLM] 图像、容器标注和 JSON 已保存：", directory, flush=True)
        else:
            report_result(result.grounding, directory)
        if result.segmentation is not None:
            print(f"[SAM] 目标掩码：{result.segmentation.area_px} 像素")
        if result.placement_segmentation is not None:
            print(f"[SAM] 放置目标掩码：{result.placement_segmentation.area_px} 像素")
        rejected_segmentation = getattr(result, "rejected_container_segmentation", None)
        if rejected_segmentation is not None:
            print(f"[SAM] 不可用容器掩码（仅供预览）：{rejected_segmentation.area_px} 像素")
        if getattr(result, "rejected_container_error", None):
            print(f"[SAM] {result.rejected_container_error}")
        if result.placement_error:
            print(f"[放置几何] {result.placement_error}")
        elif result.placement_analysis is not None:
            container = result.placement_analysis.container
            support_angle = (
                "未知" if container.floor_support_angle_deg is None
                else f"{container.floor_support_angle_deg:.1f}°"
            )
            print(
                f"[放置几何] 开口XY={container.inner_length_m * 100:.1f}×"
                f"{container.inner_width_m * 100:.1f} cm，可见底面="
                f"{container.visible_floor_length_m * 100:.1f}×"
                f"{container.visible_floor_width_m * 100:.1f} cm，开口来源={container.opening_source}，"
                f"方法={container.extraction_method}，"
                f"盒底/桌面夹角={support_angle}，平面RMSE={container.plane_rmse_m * 1000:.1f} mm"
            )
            plan = result.placement_analysis.plan
            if plan is None:
                print("[放置规划] 没有有效抓取位姿，暂不生成模拟放置姿态。")
            elif plan.fits:
                fit_region = {
                    "visible_floor": "可见盒底",
                    "rim_floor_fused": "盒缘与可见盒底融合区域",
                }.get(plan.fit_region, "开口")
                visibility = (
                    "；容器部分超出画面，按已观测区域保守规划"
                    if container.interior_truncated_by_image else ""
                )
                print(
                    f"[放置规划] XY/Yaw 在{fit_region}内可放入；平面旋转={plan.rotation_about_normal_deg:+.1f}°，"
                    f"稳健投影={plan.object_length_m * 100:.1f}×{plan.object_width_m * 100:.1f} cm，"
                    f"单侧边距=({plan.clearance_long_m * 100:.1f},"
                    f"{plan.clearance_short_m * 100:.1f}) cm，"
                    f"要求≥{plan.required_clearance_m * 100:.1f} cm{visibility}"
                )
            else:
                print(
                    f"[放置规划] 无可用放置位姿；最佳角度={plan.rotation_about_normal_deg:+.1f}°，"
                    f"剩余边距=({plan.clearance_long_m * 100:.1f},"
                    f"{plan.clearance_short_m * 100:.1f}) cm，"
                    f"要求≥{plan.required_clearance_m * 100:.1f} cm；原因={plan.status}"
                )
        if result.grasp_result is not None:
            print("[GraspNet]", result.grasp_result.status)
            counts = getattr(result.grasp_result, "filter_counts", None)
            if counts:
                width_count = counts.get("width", counts["final"])
                stages = (f"掩码内={counts['mask']}（框内掩码外={counts['bbox'] - counts['mask']}） "
                          f"→ 有效深度={counts['valid_depth']}（深度缺失={counts['mask'] - counts['valid_depth']}） "
                          f"→ 深度一致={counts['mask_depth']}"
                          f"（深度超差={counts['valid_depth'] - counts['mask_depth']}）") if "mask" in counts else (
                              f"掩码/深度={counts['mask_depth']}")
                print(f"[候选筛选] 解码={counts['decoded']} → 碰撞后={counts['collision_free']} "
                      f"→ 目标框={counts['bbox']} → {stages} → 夹爪宽度={width_count}")
            best = grasp_record(result.grasp_result.best)
            if best is None:
                print("[GraspNet] 无可用抓取位姿。")
            else:
                print("[GraspNet] 通过预览检查的最高分抓取（相机坐标系，米）：", best["translation_m"])
                print(f"[GraspNet] 分数={best['score']:.3f}，夹爪宽度={best['width_m'] * 100:.2f} cm")
        print(f"[第二阶段] 耗时 {result.elapsed_s:.2f}s；仅预览，没有机械臂动作。", flush=True)

    def status(self, result):
        if getattr(result, "placement_only", False):
            placement = result.grounding.placement
            if placement is None or not placement.found:
                return "PLACE RECHECK | NO VALID CONTAINER"
            if result.placement_error:
                return "PLACE RECHECK | GEOMETRY ERROR"
            if result.placement_analysis is not None and result.placement_analysis.plan is not None:
                return ("PLACE RECHECK | PLAN FIT" if result.placement_analysis.plan.fits
                        else "PLACE RECHECK | NO VALID PLACE POSE")
            return "PLACE RECHECK | TARGET FOUND"
        if not result.grounding.found:
            return "NO VISIBLE PICK TARGET"
        placement = result.grounding.placement
        if result.grasp_result is None or result.grasp_result.best is None:
            status = "NO VALID GRASP"
        else:
            status = f"GRASP PREVIEW PASSED | candidates={len(result.grasp_result.grasps)}"
        if placement is not None and placement.required:
            status += " | PLACE TARGET FOUND" if placement.found else " | PLACE TARGET NOT CONFIRMED"
            placement_analysis = getattr(result, "placement_analysis", None)
            placement_error = getattr(result, "placement_error", None)
            if placement_analysis is not None and placement_analysis.plan is not None:
                status += " | PLACE PLAN FIT" if placement_analysis.plan.fits else " | NO VALID PLACE POSE"
                if (placement_analysis.plan.fits
                        and placement_analysis.container.interior_truncated_by_image):
                    status += " | PARTIAL CONTAINER"
            elif placement_error:
                status += " | PLACE GEOMETRY ERROR"
        elif placement is not None and placement.selection_mode == "grasp_only":
            status += " | INVALID CONTAINER POSE | GRASP ONLY"
        return status

    def intro(self):
        return ("第二阶段：VLM 识别抓取/放置目标 → SAM 分割两类目标 → GraspNet 抓取预览 "
                "→ 容器几何与模拟放置预览，不连接机械臂。")

    def window_title(self):
        return "VLM SAM GraspNet - Stage 2"


def resolve_open3d_grasps(cfg, override=None):
    """Resolve CLI override or the two user-facing YAML display modes."""
    if override is not None:
        return override
    value = (cfg.get("vlm_grasp") or {}).get("open3d_grasps", "width")
    if not isinstance(value, str) or value not in OPEN3D_CONFIG_MODES:
        raise ValueError("vlm_grasp.open3d_grasps 只能设置为 best 或 width")
    return value


def resolve_dashboard(cfg):
    settings = ((cfg.get("vlm_grasp") or {}).get("dashboard") or {})
    enabled = bool(settings.get("enabled", True))
    width = int(settings.get("tile_width", 640))
    height = int(settings.get("tile_height", 360))
    focus_radius = float(settings.get("grasp_cloud_focus_radius_m", 0.12))
    if width < 160 or height < 120:
        raise ValueError("vlm_grasp.dashboard 每格尺寸至少为 160×120")
    if focus_radius <= 0:
        raise ValueError("vlm_grasp.dashboard.grasp_cloud_focus_radius_m 必须大于 0")
    return enabled, width, height, focus_radius


def parse_args():
    parser = argparse.ArgumentParser(description="VLM＋SAM＋GraspNet 同帧目标抓取预览，不连接机械臂")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config/default.yaml")
    parser.add_argument("--instruction", help="中文指令；省略时交互输入")
    parser.add_argument("--image", type=Path, help="已有 RGB 图像，必须与 --rgbd 配套")
    parser.add_argument("--rgbd", type=Path, help="同一快照的 rgbd.npz，包含 depth_mm、camera_matrix")
    parser.add_argument("--no-window", action="store_true", help="已有图像模式下不打开 GUI")
    parser.add_argument("--no-mask-window", action="store_true", help="关闭抓取物与容器合并掩码窗口，仍保存掩码图片")
    parser.add_argument("--no-open3d", action="store_true", help="关闭抓取候选和放置几何 Open3D 窗口")
    parser.add_argument("--top-k", type=int, help="bbox/pre-bbox 诊断模式下最多显示的候选数")
    parser.add_argument("--open3d-grasps", choices=("best", "width", "final", "bbox", "pre-bbox"),
                        help="临时覆盖 YAML：width=显示全部原始解码候选，final=显示最终候选，best=只显示最佳位姿")
    parser.add_argument("--check-config", action="store_true", help="检查配置、权重和 CUDA，不请求 API 或打开相机")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/vlm_grasp")
    parser.add_argument("--camera-type", choices=("realsense_d435i", "realsense_d405", "orbbec_gemini2"))
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--fps", type=int)
    parser.add_argument("--warmup", type=int, default=15)
    args = parser.parse_args()
    if args.top_k is not None and args.top_k < 1:
        parser.error("--top-k 必须大于 0")
    if not args.check_config:
        if args.image and not (args.rgbd and args.instruction and args.instruction.strip()):
            parser.error("--image 必须同时提供 --rgbd 和 --instruction")
        if args.rgbd and not args.image:
            parser.error("--rgbd 必须与 --image 一起使用")
        if args.no_window and not args.image:
            parser.error("--no-window 仅用于 --image + --rgbd")
        if not args.image and not args.instruction and not sys.stdin.isatty():
            parser.error("当前终端不能交互输入，请提供 --instruction")
    return args


def check_models(cfg):
    import torch
    from ultralytics import SAM  # noqa: F401
    from utils import graspnet_utils

    sam = SAMSettings.from_config(cfg)
    checkpoint = graspnet_utils.resolve_checkpoint_path((cfg.get("graspnet") or {}).get("checkpoint", "checkpoint-rs.tar"))
    if not checkpoint.is_file():
        raise ValueError(f"GraspNet 权重不存在：{checkpoint}")
    if not torch.cuda.is_available():
        raise ValueError("GraspNet 需要 CUDA；当前环境无法使用 CUDA，请在 rebotarm 环境检查 GPU")
    if sam.device != "cpu":
        device = torch.device(sam.device)
        if device.type != "cuda" or (device.index is not None and device.index >= torch.cuda.device_count()):
            raise ValueError("sam.device 必须是可用的 CUDA 设备或 cpu")
    print(f"[模型配置] SAM={sam.checkpoint}，device={sam.device}")
    print(f"[模型配置] GraspNet={checkpoint}，GPU={torch.cuda.get_device_name(0)}")



def main():
    args = parse_args()
    try:
        cfg = load_config(args.config)
        settings = VLMSettings.from_config(cfg)
        placement_settings = PlacementPreviewSettings.from_config(cfg)
        open3d_grasps = resolve_open3d_grasps(cfg, args.open3d_grasps)
        dashboard, tile_width, tile_height, focus_radius = resolve_dashboard(cfg)
        check_models(cfg)
        if args.check_config:
            print(f"[配置就绪] VLM={settings.model}，base_url={settings.base_url}，"
                  f"Open3D={open3d_grasps}，"
                  f"放置几何={'开启' if placement_settings.enabled else '关闭'}；尚未验证远程 API。")
            return 0
        client = VLMClient(settings)
        pipeline = VLMGraspPipeline(cfg, client)
        top_k = args.top_k if args.top_k is not None else int((cfg.get("graspnet") or {}).get("top_k", 50))
        if top_k < 1:
            raise ValueError("graspnet.top_k 必须大于 0")
        processor = GraspPreviewProcessor(
            pipeline, show_mask=not (args.no_window or args.no_mask_window),
            show_open3d=not (args.no_window or args.no_open3d), top_k=top_k,
            open3d_grasps=open3d_grasps,
            dashboard=dashboard,
            dashboard_tile_width=tile_width,
            dashboard_tile_height=tile_height,
            grasp_cloud_focus_radius_m=focus_radius,
        )
        if args.image:
            with np.load(args.rgbd, allow_pickle=False) as data:
                depth_mm, K = data["depth_mm"].copy(), data["camera_matrix"].copy()
            return run_image(args, client, processor=processor, depth_mm=depth_mm, K=K)
        return run_camera(args, cfg, client, processor=processor)
    except (VLMError, ValueError, OSError, RuntimeError, KeyError, ImportError, cv2.error) as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n退出预览。")
        raise SystemExit(130)
