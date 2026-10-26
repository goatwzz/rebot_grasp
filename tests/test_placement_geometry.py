"""Container geometry and preview-only placement planning tests."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import cv2
import numpy as np

from utils.placement_geometry import (
    PlacementPreviewSettings,
    _fit_plane_rectangle,
    _grow_seed_plane_mask,
    _plan_placement,
    _rim_opening_rectangle,
    analyze_placement_geometry,
    analyze_placement_geometry_from_points,
    draw_placement_preview,
    mask_points_camera,
    placement_geometry_record,
)
from utils.sam_utils import SegmentedTarget
from utils.vlm_grasp_pipeline import VLMGraspPipeline, VLMGraspResult
from utils.vlm_utils import PlacementGrounding, TargetGrounding


def synthetic_scene(*, object_size=(75, 20), container_size=(105, 65), layered=False,
                    object_angle=-20.0, container_angle=25.0):
    height, width = 240, 320
    K = np.array([[300.0, 0.0, 160.0], [0.0, 300.0, 120.0], [0.0, 0.0, 1.0]])
    color = np.zeros((height, width, 3), dtype=np.uint8)
    depth = np.zeros((height, width), dtype=np.uint16)
    pick = np.zeros((height, width), dtype=np.uint8)
    place = np.zeros_like(pick)
    pick_box = cv2.boxPoints(((80, 150), object_size, object_angle)).astype(np.int32)
    place_box = cv2.boxPoints(((240, 115), container_size, container_angle)).astype(np.int32)
    cv2.fillConvexPoly(pick, pick_box, 1)
    cv2.fillConvexPoly(place, place_box, 1)
    depth[pick != 0] = 700
    depth[place != 0] = 800
    if layered:
        depth[place != 0] = 700
        inner = np.zeros_like(place)
        inner_box = cv2.boxPoints(((240, 115), (container_size[0] - 24, container_size[1] - 22), 25)).astype(np.int32)
        cv2.fillConvexPoly(inner, inner_box, 1)
        depth[inner != 0] = 820
    center = np.array([(80 - 160) * 0.7 / 300, (150 - 120) * 0.7 / 300, 0.7])
    best = SimpleNamespace(translation=center, rotation_matrix=np.eye(3))
    return color, depth, K, pick.astype(bool), place.astype(bool), best


class PlacementGeometryTests(unittest.TestCase):
    def test_complete_rim_corrects_partial_floor_center(self):
        floor_center = np.array([0.0, 0.02, 0.5])
        rim_x = np.linspace(-0.10, 0.10, 160)
        rim_y = np.linspace(-0.06, 0.06, 120)
        rim = np.vstack([
            np.column_stack([rim_x, np.full(len(rim_x), side), np.full(len(rim_x), 0.45)])
            for side in (-0.06, 0.06)
        ] + [
            np.column_stack([np.full(len(rim_y), side), rim_y, np.full(len(rim_y), 0.45)])
            for side in (-0.10, 0.10)
        ])
        floor_x, floor_y = np.meshgrid(np.linspace(-0.09, 0.09, 30), np.linspace(-0.015, 0.055, 20))
        floor = np.column_stack([floor_x.ravel(), floor_y.ravel(), np.full(floor_x.size, 0.5)])
        settings = PlacementPreviewSettings(min_points=32)
        opening = _rim_opening_rectangle(
            np.vstack([rim, floor]), floor_center,
            np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, -1.0]), 0.18, 0.07, settings,
        )
        self.assertIsNotNone(opening)
        self.assertAlmostEqual(opening[0][1], 0.0, places=3)
        self.assertGreater(opening[4], 0.10)
        container = SimpleNamespace(
            relation="in", opening_source="rim_floor_fused",
            normal_camera=np.array([0.0, 0.0, -1.0]),
            center_camera_m=opening[0], long_axis_camera=opening[1],
            short_axis_camera=opening[2], inner_length_m=opening[3], inner_width_m=opening[4],
            interior_truncated_by_image=False,
        )
        object_x, object_y = np.meshgrid(np.linspace(-0.07, 0.07, 30), np.linspace(-0.025, 0.025, 10))
        object_points = np.column_stack([object_x.ravel(), object_y.ravel(), np.full(object_x.size, 0.45)])
        grasp = SimpleNamespace(translation=object_points.mean(axis=0), rotation_matrix=np.eye(3))
        plan = _plan_placement(object_points, grasp, container, settings)
        self.assertTrue(plan.fits)
        self.assertEqual(plan.fit_region, "rim_floor_fused")
        self.assertAlmostEqual(plan.object_center_on_plane_camera_m[1], 0.0, places=3)

    def test_missing_or_conflicting_rim_is_not_used(self):
        rim_x = np.linspace(-0.1, 0.1, 160)
        rim_y = np.linspace(-0.06, 0.06, 120)
        three_sides = np.vstack([
            np.column_stack([rim_x, np.full(len(rim_x), -0.06), np.full(len(rim_x), 0.45)]),
            np.column_stack([np.full(len(rim_y), -0.1), rim_y, np.full(len(rim_y), 0.45)]),
            np.column_stack([np.full(len(rim_y), 0.1), rim_y, np.full(len(rim_y), 0.45)]),
        ])
        settings = PlacementPreviewSettings(min_points=32)
        args = (np.array([0.0, 0.02, 0.5]), np.array([1.0, 0.0, 0.0]),
                np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, -1.0]), 0.18, 0.07, settings)
        self.assertIsNone(_rim_opening_rectangle(three_sides, *args))
        fourth_side = np.column_stack([
            rim_x, np.full(len(rim_x), 0.06), np.full(len(rim_x), 0.45),
        ])
        far_rim = np.vstack([three_sides, fourth_side])
        far_rim[:, 1] -= 0.15
        self.assertIsNone(_rim_opening_rectangle(far_rim, *args))

    def test_post_lift_geometry_uses_saved_points_without_pick_mask(self):
        _, depth, K, pick, place, best = synthetic_scene()
        points = mask_points_camera(pick, depth, K)
        result = analyze_placement_geometry_from_points(
            points,
            place,
            depth,
            K,
            gripper_pose_camera=best,
            target_name="纸盒",
            relation="in",
            settings=PlacementPreviewSettings(min_points=32),
        )
        self.assertIsNotNone(result.plan)
        self.assertTrue(result.plan.fits)
        self.assertGreater(len(result.candidate_angles_deg), 1)

    def test_preferred_support_normal_rejects_larger_side_wall_plane(self):
        floor_x, floor_y = np.meshgrid(
            np.linspace(-0.10, 0.10, 50), np.linspace(-0.05, 0.05, 30),
        )
        floor = np.column_stack([
            floor_x.ravel(), floor_y.ravel(), np.full(floor_x.size, 0.40),
        ])
        wall_x, wall_z = np.meshgrid(
            np.linspace(-0.10, 0.10, 50), np.linspace(0.30, 0.50, 80),
        )
        wall = np.column_stack([
            wall_x.ravel(), np.full(wall_x.size, 0.05), wall_z.ravel(),
        ])
        points = np.vstack([floor, wall])
        result = _fit_plane_rectangle(
            points, 0.002, 96,
            preferred_normal=np.array([0.0, 0.0, 1.0]),
            max_normal_deviation_deg=10.0,
        )
        self.assertGreater(abs(float(result[4] @ np.array([0.0, 0.0, 1.0]))), 0.99)
        self.assertAlmostEqual(result[5], 0.20, places=3)
        self.assertAlmostEqual(result[6], 0.10, places=3)

    def test_plane_growth_discards_disconnected_coplanar_outliers(self):
        height, width = 100, 140
        container = np.zeros((height, width), dtype=bool)
        seed = np.zeros_like(container)
        depth = np.zeros((height, width), dtype=np.uint16)
        container[20:80, 10:70] = True
        seed[20:80, 10:70] = True
        depth[20:80, 10:70] = 800
        # A non-coplanar bridge made the old depth candidate connected, while
        # the small far patch itself was close enough to stretch minAreaRect.
        container[45:55, 70:115] = True
        seed[45:55, 70:115] = True
        depth[45:55, 70:105] = 900
        depth[45:55, 105:115] = 800
        grown = _grow_seed_plane_mask(
            container,
            seed,
            depth,
            np.eye(3),
            np.array([0.0, 0.0, 0.8]),
            np.array([0.0, 0.0, 1.0]),
            PlacementPreviewSettings(min_points=32, morph_kernel_px=3),
        )
        self.assertTrue(grown[30, 30])
        self.assertFalse(grown[50, 110])

    def test_rotated_container_produces_fitting_aligned_object_plan(self):
        color, depth, K, pick, place, best = synthetic_scene()
        settings = PlacementPreviewSettings(min_points=32)
        result = analyze_placement_geometry(
            pick, place, depth, K, best_grasp=best, target_name="纸盒", relation="in", settings=settings,
        )
        self.assertGreater(result.container.inner_length_m, result.container.inner_width_m)
        self.assertLess(result.container.plane_rmse_m, 1e-5)
        self.assertEqual(result.container.extraction_method, "eroded_mask")
        self.assertTrue(result.plan.fits)
        self.assertGreaterEqual(result.plan.clearance_long_m, settings.wall_clearance_m)
        self.assertGreaterEqual(result.plan.clearance_short_m, settings.wall_clearance_m)
        relative = result.plan.simulated_object_points_camera - result.container.center_camera_m
        projected_long = relative @ result.container.long_axis_camera
        projected_short = relative @ result.container.short_axis_camera
        self.assertLessEqual(np.max(np.abs(projected_long)),
                             result.container.inner_length_m / 2 - settings.wall_clearance_m + 1e-9)
        self.assertLessEqual(np.max(np.abs(projected_short)),
                             result.container.inner_width_m / 2 - settings.wall_clearance_m + 1e-9)
        self.assertAlmostEqual(abs(np.dot(result.container.long_axis_camera,
                                          result.container.short_axis_camera)), 0.0, places=6)
        record = placement_geometry_record(result)
        self.assertTrue(record["available"])
        self.assertEqual(record["coordinate_frame"], "camera_optical")
        self.assertTrue(record["plan"]["fits"])
        self.assertEqual(record["plan"]["planned_dofs"], ["x", "y", "yaw"])
        self.assertFalse(record["validation_scope"]["container_depth_or_wall_height"])
        self.assertFalse(record["validation_scope"]["object_height_or_thickness"])
        display = draw_placement_preview(color, place, result, K)
        self.assertEqual(display.shape, color.shape)
        self.assertTrue(display.any())

    def test_depth_layer_extracts_container_interior(self):
        _, depth, K, pick, place, _ = synthetic_scene(layered=True)
        result = analyze_placement_geometry(
            pick, place, depth, K, best_grasp=None, target_name="纸盒", relation="in",
            settings=PlacementPreviewSettings(min_points=32),
        )
        self.assertEqual(result.container.extraction_method, "depth_quantile+plane_grow")
        self.assertGreater(result.container.depth_span_m, 0.018)
        self.assertLess(result.container.interior_mask.sum(), place.sum())
        self.assertGreater(result.container.inner_length_m, result.container.visible_floor_length_m)
        self.assertGreater(result.container.inner_width_m, result.container.visible_floor_width_m)
        record = placement_geometry_record(result)["container"]
        self.assertAlmostEqual(record["opening_length_m"], result.container.inner_length_m)
        self.assertAlmostEqual(record["visible_floor_length_m"], result.container.visible_floor_length_m)
        self.assertEqual(record["opening_source"], "projected_mask")
        self.assertIsNotNone(record["projected_mask_center_m"])
        self.assertIsNone(result.plan)

    def test_object_projection_ignores_sparse_depth_outlier(self):
        _, depth, K, pick, place, best = synthetic_scene(object_size=(65, 18))
        noisy_depth = depth.copy()
        noisy_pick = pick.copy()
        noisy_pick[15, 15] = True
        noisy_depth[15, 15] = 700
        robust = analyze_placement_geometry(
            noisy_pick, place, noisy_depth, K, best_grasp=best,
            target_name="纸盒", relation="in",
            settings=PlacementPreviewSettings(min_points=32, object_xy_trim_quantile=0.005),
        )
        untrimmed = analyze_placement_geometry(
            noisy_pick, place, noisy_depth, K, best_grasp=best,
            target_name="纸盒", relation="in",
            settings=PlacementPreviewSettings(min_points=32, object_xy_trim_quantile=0.0),
        )
        self.assertTrue(robust.plan.fits)
        self.assertGreater(untrimmed.plan.object_length_m, robust.plan.object_length_m + 0.02)
        self.assertEqual(robust.plan.object_footprint_quantile, 0.005)

    def test_one_degree_yaw_search_recovers_narrow_valid_angle(self):
        length, width = 0.164, 0.078
        x, y = np.meshgrid(
            np.linspace(-length / 2, length / 2, 41),
            np.linspace(-width / 2, width / 2, 21),
        )
        points = np.column_stack([x.ravel(), y.ravel(), np.zeros(x.size)])
        angle = np.deg2rad(18.0)
        initial_rotation = np.array([
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ])
        points = points @ initial_rotation.T
        container = SimpleNamespace(
            long_axis_camera=np.array([1.0, 0.0, 0.0]),
            short_axis_camera=np.array([0.0, 1.0, 0.0]),
            normal_camera=np.array([0.0, 0.0, 1.0]),
            inner_length_m=0.1888315,
            inner_width_m=0.1040199,
            center_camera_m=np.zeros(3),
            interior_truncated_by_image=False,
            relation="in",
        )
        grasp = SimpleNamespace(translation=np.zeros(3), rotation_matrix=np.eye(3))
        coarse = _plan_placement(
            points, grasp, container,
            PlacementPreviewSettings(
                min_points=32, yaw_step_deg=5.0, wall_clearance_m=0.012,
            ),
        )
        fine = _plan_placement(
            points, grasp, container,
            PlacementPreviewSettings(
                min_points=32, yaw_step_deg=1.0, wall_clearance_m=0.012,
            ),
        )
        self.assertFalse(coarse.fits)
        self.assertTrue(fine.fits)
        self.assertEqual(fine.rotation_about_normal_deg, -18.0)
        self.assertGreaterEqual(fine.clearance_long_m, 0.012)
        self.assertGreaterEqual(fine.clearance_short_m, 0.012)

    def test_oversized_object_is_not_forced_into_container(self):
        _, depth, K, pick, place, best = synthetic_scene(
            object_size=(120, 55), container_size=(72, 42),
        )
        result = analyze_placement_geometry(
            pick, place, depth, K, best_grasp=best, target_name="小盒子", relation="in",
            settings=PlacementPreviewSettings(min_points=32),
        )
        self.assertFalse(result.plan.fits)
        self.assertIn("无可用放置位姿", result.plan.status)
        self.assertFalse(placement_geometry_record(result)["plan"]["fits"])

    def test_asymmetric_object_footprint_is_centered_to_maximize_wall_clearance(self):
        _, depth, K, pick, place, best = synthetic_scene(container_size=(120, 90))
        depth[pick] = 0
        pick[:] = False
        triangle = np.array([[35, 175], [125, 175], [125, 125]], dtype=np.int32)
        cv2.fillConvexPoly(pick.view(np.uint8), triangle, 1)
        depth[pick] = 700
        result = analyze_placement_geometry(
            pick, place, depth, K, best_grasp=best, target_name="纸盒", relation="in",
            settings=PlacementPreviewSettings(min_points=32),
        )
        self.assertTrue(result.plan.fits)
        expected_long = (result.container.visible_floor_length_m - result.plan.object_length_m) / 2.0
        expected_short = (result.container.visible_floor_width_m - result.plan.object_width_m) / 2.0
        self.assertAlmostEqual(result.plan.clearance_long_m, expected_long, places=8)
        self.assertAlmostEqual(result.plan.clearance_short_m, expected_short, places=8)
        self.assertGreaterEqual(
            result.plan.clearance_long_m + 1e-9, result.plan.required_clearance_m,
        )
        self.assertGreaterEqual(
            result.plan.clearance_short_m + 1e-9, result.plan.required_clearance_m,
        )

    def test_insertion_uses_observed_floor_not_projected_outer_walls(self):
        _, depth, K, pick, place, best = synthetic_scene(layered=True, object_size=(75, 50))
        settings = PlacementPreviewSettings(min_points=32)
        result = analyze_placement_geometry(
            pick, place, depth, K, best_grasp=best, target_name="纸盒", relation="in", settings=settings,
        )
        self.assertGreater(
            result.container.inner_width_m,
            result.plan.object_width_m + 2 * settings.wall_clearance_m,
        )
        self.assertLess(
            result.container.visible_floor_width_m,
            result.plan.object_width_m + 2 * settings.wall_clearance_m,
        )
        self.assertFalse(result.plan.fits)
        self.assertEqual(result.plan.fit_region, "visible_floor")

    def test_fitting_object_is_centered_inside_observed_floor(self):
        _, depth, K, pick, place, best = synthetic_scene(layered=True)
        result = analyze_placement_geometry(
            pick, place, depth, K, best_grasp=best, target_name="纸盒", relation="in",
            settings=PlacementPreviewSettings(min_points=32),
        )
        self.assertTrue(result.plan.fits)
        floor_corners = result.container.visible_floor_corners_camera_m
        axis_long = (floor_corners[1] - floor_corners[0]) / result.container.visible_floor_length_m
        axis_short = (floor_corners[2] - floor_corners[1]) / result.container.visible_floor_width_m
        relative = result.plan.simulated_object_points_camera - result.container.visible_floor_center_camera_m
        self.assertLessEqual(np.max(np.abs(relative @ axis_long)),
                             result.container.visible_floor_length_m / 2 + 1e-9)
        self.assertLessEqual(np.max(np.abs(relative @ axis_short)),
                             result.container.visible_floor_width_m / 2 + 1e-9)

    def test_near_relation_is_rejected_and_truncated_container_uses_visible_region(self):
        _, depth, K, pick, place, best = synthetic_scene()
        near = analyze_placement_geometry(
            pick, place, depth, K, best_grasp=best, target_name="盒子", relation="near",
            settings=PlacementPreviewSettings(min_points=32),
        )
        self.assertFalse(near.plan.fits)
        self.assertIn("不是放入或放在表面", near.plan.status)

        truncated = np.zeros(place.shape, dtype=np.uint8)
        cv2.fillConvexPoly(
            truncated,
            cv2.boxPoints(((305, 115), (80, 60), 20)).astype(np.int32),
            1,
        )
        truncated_depth = depth.copy()
        truncated_depth[place] = 0
        truncated = truncated.astype(bool)
        truncated_depth[truncated] = 800
        clipped = analyze_placement_geometry(
            pick, truncated, truncated_depth, K, best_grasp=best, target_name="截断盒子", relation="in",
            settings=PlacementPreviewSettings(min_points=32),
        )
        self.assertTrue(clipped.container.interior_truncated_by_image)
        self.assertTrue(clipped.plan.fits)
        self.assertIn("已观测容器区域内满足安全间距", clipped.plan.status)
        self.assertGreaterEqual(clipped.plan.clearance_long_m, clipped.plan.required_clearance_m)
        self.assertGreaterEqual(clipped.plan.clearance_short_m, clipped.plan.required_clearance_m)
        record = placement_geometry_record(clipped)
        self.assertEqual(record["plan"]["container_visibility"], "partial_at_image_boundary")
        self.assertTrue(record["plan"]["uses_observed_container_geometry"])
        self.assertTrue(record["validation_scope"]["partial_container_uses_observed_geometry"])

    def test_invalid_settings_are_rejected(self):
        invalid = [
            {"min_points": 1}, {"interior_depth_quantile": 0.99},
            {"wall_clearance_m": -0.01}, {"yaw_step_deg": 0},
            {"floor_normal_tolerance_deg": 0}, {"floor_normal_tolerance_deg": 61},
            {"rim_inset_m": -0.001},
            {"container_xy_trim_quantile": 0.1}, {"object_xy_trim_quantile": -0.01},
            {"enabled": "yes"},
        ]
        for values in invalid:
            cfg = {"vlm_grasp": {"placement_preview": values}}
            with self.subTest(values=values), self.assertRaises(ValueError):
                PlacementPreviewSettings.from_config(cfg)


class PlacementPipelineTests(unittest.TestCase):
    def test_pick_and_destination_are_segmented_and_geometry_uses_best_grasp(self):
        color, depth, K, pick, place, best = synthetic_scene()
        placement = PlacementGrounding(
            True, True, "纸盒", (190, 70, 290, 160), "测试", "in", "纸盒", "explicit",
        )
        grounding = TargetGrounding(
            True, "香蕉", (35, 110, 125, 190), "测试", "放入盒子",
            (320, 240), (320, 240), "pixel", placement,
        )
        client = SimpleNamespace(ground=Mock(return_value=grounding))
        segmenter = SimpleNamespace(segment=Mock(side_effect=[
            SegmentedTarget(pick, grounding.bbox_xyxy, int(pick.sum())),
            SegmentedTarget(place, placement.bbox_xyxy, int(place.sum())),
        ]))
        grasp_result = SimpleNamespace(best=best, grasps=[])
        grasp_backend = SimpleNamespace(infer_frame=Mock(return_value=grasp_result))
        analysis = object()
        placement_backend = SimpleNamespace(analyze_placement_geometry=Mock(return_value=analysis))
        pipeline = VLMGraspPipeline(
            {"vlm_grasp": {"placement_preview": {"min_points": 32}}}, client,
            segmenter=segmenter, net=object(), grasp_backend=grasp_backend,
            placement_backend=placement_backend,
        )
        with redirect_stdout(io.StringIO()):
            result = pipeline.process("把香蕉放进盒子", color, depth, K)
        self.assertEqual(segmenter.segment.call_count, 2)
        self.assertIs(result.placement_segmentation.mask, place)
        self.assertIs(result.placement_analysis, analysis)
        call = placement_backend.analyze_placement_geometry.call_args
        self.assertIs(call.args[0], result.segmentation.mask)
        self.assertIs(call.args[1], result.placement_segmentation.mask)
        self.assertIs(call.kwargs["best_grasp"], best)
        self.assertEqual(call.kwargs["target_name"], "纸盒")

    def test_destination_geometry_failure_does_not_discard_grasp_result(self):
        color, depth, K, pick, place, best = synthetic_scene()
        placement = PlacementGrounding(True, True, "纸盒", (190, 70, 290, 160), "测试", "in")
        grounding = TargetGrounding(
            True, "香蕉", (35, 110, 125, 190), "测试", "放入盒子",
            (320, 240), (320, 240), "pixel", placement,
        )
        segmenter = SimpleNamespace(segment=Mock(side_effect=[
            SegmentedTarget(pick, grounding.bbox_xyxy, int(pick.sum())),
            SegmentedTarget(place, placement.bbox_xyxy, int(place.sum())),
        ]))
        grasp_result = SimpleNamespace(best=best, grasps=[])
        placement_backend = SimpleNamespace(
            analyze_placement_geometry=Mock(side_effect=ValueError("平面不足")),
        )
        pipeline = VLMGraspPipeline(
            {"vlm_grasp": {"placement_preview": {"min_points": 32}}},
            SimpleNamespace(ground=Mock(return_value=grounding)),
            segmenter=segmenter, net=object(),
            grasp_backend=SimpleNamespace(infer_frame=Mock(return_value=grasp_result)),
            placement_backend=placement_backend,
        )
        with redirect_stdout(io.StringIO()):
            result = pipeline.process("放进去", color, depth, K)
        self.assertIs(result.grasp_result, grasp_result)
        self.assertIsNone(result.placement_analysis)
        self.assertIn("平面不足", result.placement_error)


class PlacementPreviewOutputTests(unittest.TestCase):
    def make_result(self):
        color, depth, K, pick, place, best = synthetic_scene()
        placement = PlacementGrounding(True, True, "纸盒", (185, 65, 295, 165), "测试", "in")
        grounding = TargetGrounding(
            True, "香蕉", (30, 105, 130, 195), "测试", "放入盒子",
            (320, 240), (320, 240), "pixel", placement,
        )
        analysis = analyze_placement_geometry(
            pick, place, depth, K, best_grasp=best, target_name="纸盒", relation="in",
            settings=PlacementPreviewSettings(min_points=32),
        )
        result = VLMGraspResult(
            grounding,
            SegmentedTarget(pick, grounding.bbox_xyxy, int(pick.sum())),
            None,
            0.1,
            K,
            SegmentedTarget(place, placement.bbox_xyxy, int(place.sum())),
            analysis,
            None,
        )
        return color, depth, K, result

    def test_masks_geometry_json_and_annotated_plan_are_saved(self):
        from scripts.vlm_grasp_demo import GraspPreviewProcessor
        from utils import placement_geometry

        color, depth, K, result = self.make_result()
        pipeline = SimpleNamespace(
            placement_backend=placement_geometry,
            backend=SimpleNamespace(),
            client=SimpleNamespace(settings=SimpleNamespace(model="test-model")),
            segmenter=SimpleNamespace(settings=SimpleNamespace(checkpoint=Path("sam-test.pt"))),
            options={},
        )
        processor = GraspPreviewProcessor(pipeline, show_mask=False, show_open3d=False)
        with tempfile.TemporaryDirectory() as temp:
            directory = processor.save(Path(temp), color, result, depth_mm=depth, K=K)
            for name in ("combined_mask.png", "placement_mask.png", "placement_interior_mask.png",
                         "placement_geometry.json"):
                self.assertTrue((directory / name).is_file(), name)
            combined = cv2.imread(str(directory / "combined_mask.png"), cv2.IMREAD_GRAYSCALE)
            expected = result.segmentation.mask | result.placement_segmentation.mask
            np.testing.assert_array_equal(combined, expected.astype(np.uint8) * 255)
            record = json.loads((directory / "placement_geometry.json").read_text())
            self.assertTrue(record["available"])
            self.assertTrue(record["plan"]["fits"])
            annotated = cv2.imread(str(directory / "annotated.png"))
            self.assertIsNotNone(annotated)
            self.assertTrue(annotated.any())

    def test_main_preview_labels_both_targets_without_resizing(self):
        from scripts.vlm_grasp_demo import GraspPreviewProcessor
        from utils import placement_geometry

        color, _, _, result = self.make_result()
        pipeline = SimpleNamespace(placement_backend=placement_geometry, backend=SimpleNamespace())
        display = GraspPreviewProcessor(
            pipeline, show_mask=False, show_open3d=False,
        ).draw(color, result)
        np.testing.assert_array_equal(display[145, 30], [0, 255, 0])
        np.testing.assert_array_equal(display[65, 185], [0, 165, 255])
        self.assertEqual(display.shape, color.shape)

    def test_main_preview_marks_geometry_rejection_red_with_cross(self):
        from scripts.vlm_grasp_demo import GraspPreviewProcessor
        from utils import placement_geometry

        color, _, _, result = self.make_result()
        result.placement_analysis.plan.fits = False
        result.placement_analysis.plan.status = "物体超过容器安全空间"
        pipeline = SimpleNamespace(placement_backend=placement_geometry)
        display = GraspPreviewProcessor(pipeline, show_open3d=False).draw(color, result)
        np.testing.assert_array_equal(display[115, 185], [0, 0, 255])
        np.testing.assert_array_equal(display[115, 240], [0, 0, 255])
        np.testing.assert_array_equal(display[145, 30], [0, 255, 0])

    def test_main_preview_marks_geometry_error_as_unverified(self):
        from scripts.vlm_grasp_demo import GraspPreviewProcessor
        from utils import placement_geometry

        color, _, _, result = self.make_result()
        result.placement_analysis = None
        result.placement_error = "有效深度点不足"
        pipeline = SimpleNamespace(placement_backend=placement_geometry)
        display = GraspPreviewProcessor(pipeline, show_open3d=False).draw(color, result)
        np.testing.assert_array_equal(display[115, 240], [0, 0, 255])

    def test_placement_open3d_window_updates_clears_and_closes(self):
        from scripts.vlm_grasp_demo import GraspPreviewProcessor

        color, _, _, result = self.make_result()
        viewer = Mock()
        viewer.poll.return_value = True
        viewer.frame.return_value = np.full((90, 160, 3), 80, np.uint8)
        placement_backend = SimpleNamespace(Open3DPlacementWindow=Mock(return_value=viewer))
        pipeline = SimpleNamespace(placement_backend=placement_backend)
        processor = GraspPreviewProcessor(pipeline, show_mask=False, show_open3d=True)
        with redirect_stdout(io.StringIO()):
            processor.show_result(color, result)
        placement_backend.Open3DPlacementWindow.assert_called_once_with(visible=False)
        viewer.update.assert_called_once_with(result.placement_analysis)
        processor.reset_preview(color.shape[:2], "DISCARDED")
        viewer.clear.assert_called_once()
        viewer.poll.return_value = False
        processor.poll_preview()
        viewer.close.assert_called_once()
        self.assertIsNone(processor._placement_viewer)


if __name__ == "__main__":
    unittest.main()
