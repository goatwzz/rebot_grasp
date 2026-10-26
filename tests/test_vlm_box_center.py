"""Hardware-free checks for the VLM box-center placement baseline."""

from __future__ import annotations

from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from scripts import vlm_grasp_box_center_robot as baseline
from utils import placement_geometry
from utils.sam_utils import SegmentedTarget
from utils.vlm_utils import PlacementGrounding, TargetGrounding


class BoxCenterBaselineTests(unittest.TestCase):
    def setUp(self):
        self.color = np.zeros((100, 120, 3), np.uint8)
        self.depth = np.full((100, 120), 500, np.uint16)
        self.K = np.array([[100.0, 0.0, 60.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]])
        self.placement = PlacementGrounding(True, True, "box", (70, 20, 110, 80), "open", "in")
        self.grounding = TargetGrounding(
            True, "banana", (10, 20, 50, 80), "visible", "place banana in box",
            (120, 100), (120, 100), "pixel", self.placement,
        )

    def test_center_moves_with_vlm_box_and_uses_local_depth(self):
        self.depth[45:56, 85:96] = 600
        pixel, point, count = baseline.box_center_camera(self.placement.bbox_xyxy, self.depth, self.K)
        self.assertEqual(pixel, (90, 50))
        self.assertEqual(count, 121)
        np.testing.assert_allclose(point, [0.18, 0.0, 0.6])
        moved_pixel, moved_point, _ = baseline.box_center_camera((80, 20, 119, 80), self.depth, self.K)
        self.assertEqual(moved_pixel, (99, 50))
        self.assertGreater(moved_point[0], point[0])
        transform = np.eye(4)
        transform[:3, 3] = [0.2, -0.1, 0.3]
        release, preplace = baseline.fixed_place_poses(point, transform, 0.08)
        np.testing.assert_allclose(release, [0.38, -0.1, 0.08, 0.0, 1.2, 0.0])
        np.testing.assert_allclose(preplace, [0.38, -0.1, 0.16, 0.0, 1.2, 0.0])

    def test_missing_center_depth_is_not_replaced_by_unrelated_box_depth(self):
        self.depth[45:56, 85:96] = 0
        with self.assertRaisesRegex(ValueError, "中心附近没有"):
            baseline.box_center_camera(self.placement.bbox_xyxy, self.depth, self.K)

    def test_pipeline_skips_container_sam_and_pose_geometry(self):
        mask = np.zeros(self.depth.shape, bool)
        mask[20:80, 10:50] = True
        segmenter = SimpleNamespace(segment=Mock(return_value=SegmentedTarget(
            mask, self.grounding.bbox_xyxy, int(mask.sum()),
        )), settings=SimpleNamespace(checkpoint=Path("sam-test.pt")))
        grasp_result = SimpleNamespace(best=None, grasps=[])
        backend = SimpleNamespace(infer_frame=Mock(return_value=grasp_result))
        geometry = SimpleNamespace(analyze_placement_geometry=Mock(),
                                   placement_geometry_record=placement_geometry.placement_geometry_record)
        client = SimpleNamespace(ground=Mock(return_value=self.grounding),
                                 settings=SimpleNamespace(model="test-model"))
        pipeline = baseline.BoxCenterPipeline(
            {}, client, segmenter=segmenter, net=object(), grasp_backend=backend,
            placement_backend=geometry,
        )
        with redirect_stdout(io.StringIO()):
            result = pipeline.process("place banana in box", self.color, self.depth, self.K)
        self.assertFalse(pipeline.placement_settings.enabled)
        segmenter.segment.assert_called_once()
        self.assertEqual(segmenter.segment.call_args.args[1], self.grounding.bbox_xyxy)
        geometry.analyze_placement_geometry.assert_not_called()
        self.assertIsNone(result.placement_segmentation)
        self.assertIsNone(result.placement_analysis)
        self.assertIs(backend.infer_frame.call_args.kwargs["target_mask"], mask)

    def test_prepare_uses_fixed_pose_and_requires_ik(self):
        result = SimpleNamespace(
            grounding=self.grounding,
            segmentation=SimpleNamespace(mask=np.ones(self.depth.shape, bool)),
            grasp_result=SimpleNamespace(grasps=[object()]),
        )
        pending = SimpleNamespace(initial_place=None, recheck_after_lift=True)
        robot = SimpleNamespace(
            _combined_exclusion_masks=Mock(return_value=[]),
            _finish_prepare_pick=Mock(return_value=pending),
            _pose_checks=Mock(return_value=(True, "IK pass")),
            PendingPlace=lambda result, release, preplace: SimpleNamespace(
                result=result, release6d=release, preplace6d=preplace,
            ),
        )
        settings = SimpleNamespace(place_approach_offset_m=0.08)
        with patch.object(baseline.importlib, "import_module", return_value=robot), patch.object(
            baseline.placement_geometry, "estimate_table_normal", return_value=np.array([0., 0., 1.]),
        ):
            with redirect_stdout(io.StringIO()):
                prepared = baseline.prepare_box_center_pick(
                    "place", result, self.color, self.depth, self.K, np.eye(4),
                    object(), settings, {"min_base_z_m": 0.03},
                    placement_geometry.PlacementPreviewSettings(),
                )
        self.assertIs(prepared, pending)
        self.assertFalse(prepared.recheck_after_lift)
        self.assertEqual(prepared.initial_place.release6d[2:], (0.08, 0.0, 1.2, 0.0))
        self.assertFalse(result.box_center_plan["container_pose_estimated"])
        self.assertFalse(result.box_center_plan["fit_or_collision_checked"])
        robot._pose_checks.assert_called_once()

    def test_invalid_container_never_starts_pick(self):
        grounding = replace(self.grounding, placement=PlacementGrounding(
            False, False, "", None, "invalid", None, selection_mode="grasp_only",
        ))
        result = SimpleNamespace(grounding=grounding)
        with patch.object(baseline.importlib, "import_module") as imported:
            with redirect_stdout(io.StringIO()):
                prepared = baseline.prepare_box_center_pick(
                    "place", result, self.color, self.depth, self.K, np.eye(4),
                    object(), object(), {}, placement_geometry.PlacementPreviewSettings(),
                )
        self.assertIsNone(prepared)
        imported.return_value._finish_prepare_pick.assert_not_called()

    def test_failed_place_ik_prevents_pick(self):
        result = SimpleNamespace(
            grounding=self.grounding,
            segmentation=SimpleNamespace(mask=np.ones(self.depth.shape, bool)),
            grasp_result=SimpleNamespace(grasps=[object()]),
        )
        robot = SimpleNamespace(
            _combined_exclusion_masks=Mock(return_value=[]),
            _finish_prepare_pick=Mock(return_value=SimpleNamespace(initial_place=None)),
            _pose_checks=Mock(return_value=(False, "release IK failed")),
        )
        with patch.object(baseline.importlib, "import_module", return_value=robot), patch.object(
            baseline.placement_geometry, "estimate_table_normal", return_value=np.array([0., 0., 1.]),
        ):
            with redirect_stdout(io.StringIO()):
                prepared = baseline.prepare_box_center_pick(
                    "place", result, self.color, self.depth, self.K, np.eye(4), object(),
                    SimpleNamespace(place_approach_offset_m=0.08), {},
                    placement_geometry.PlacementPreviewSettings(),
                )
        self.assertIsNone(prepared)
        self.assertFalse(result.box_center_plan["ik_valid"])

    def test_baseline_record_is_saved_separately(self):
        mask = np.zeros(self.depth.shape, bool)
        mask[20:80, 10:50] = True
        result = SimpleNamespace(
            grounding=self.grounding,
            segmentation=SegmentedTarget(mask, self.grounding.bbox_xyxy, int(mask.sum())),
            placement_segmentation=None,
            placement_analysis=None,
            placement_error=None,
            grasp_result=None,
            elapsed_s=1.0,
            camera_matrix=self.K,
            box_center_plan={"ik_valid": True, "release_base_pose6d": [0.1, 0.2, 0.08, 0, 1.2, 0],
                             "pixel_xy": [90, 50]},
        )
        pipeline = SimpleNamespace(
            placement_backend=placement_geometry,
            client=SimpleNamespace(settings=SimpleNamespace(model="test-model")),
            segmenter=SimpleNamespace(settings=SimpleNamespace(checkpoint=Path("sam-test.pt"))),
            options={},
        )
        processor = baseline.BoxCenterPreviewProcessor(pipeline, show_open3d=False, dashboard=False)
        with tempfile.TemporaryDirectory() as temp:
            directory = processor.save(Path(temp), self.color, result)
            record = json.loads((directory / "box_center_baseline.json").read_text())
            self.assertTrue(record["ik_valid"])
            self.assertEqual(record["release_base_pose6d"][2], 0.08)
            self.assertFalse((directory / "placement_interior_mask.png").exists())
            annotated = cv2.imread(str(directory / "annotated.png"))
            self.assertTrue(annotated[50, 90].any())


if __name__ == "__main__":
    unittest.main()
