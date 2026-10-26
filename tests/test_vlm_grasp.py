"""Target identity/geometry and snapshot orchestration without GPU or hardware."""

from contextlib import redirect_stdout
import ast
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from utils.sam_utils import SAMSegmenter, SAMSettings, SegmentedTarget
from utils.target_mask_utils import target_grasp_keep_mask, validate_bbox, validate_mask, validate_rgbd
from utils.vlm_grasp_pipeline import VLMGraspPipeline
from utils.vlm_utils import TargetGrounding

ROOT = Path(__file__).resolve().parents[1]


def grasp_functions(names, namespace):
    tree = ast.parse((ROOT / "utils/graspnet_utils.py").read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *functions],
                        type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), "graspnet_utils.py", "exec"), namespace)
    return namespace


def scene():
    color = np.zeros((60, 80, 3), np.uint8)
    depth = np.full((60, 80), 500, np.uint16)
    K = np.diag([100., 100., 1.])
    mask = np.zeros((60, 80), bool)
    mask[20:40, 20:40] = True
    return color, depth, K, mask


class TargetMaskTests(unittest.TestCase):
    def test_reject_other_objects_and_same_pixel_at_wrong_depth(self):
        _, depth, K, mask = scene()
        points = [[.15, .15, .5], [.3, .15, .5], [.3, .3, 1.], [0., 0., 0.],
                  [np.nan, 0., .5], [0., 0., -.5], [np.inf, 0., .5]]
        np.testing.assert_array_equal(target_grasp_keep_mask(points, mask, K, depth),
                                      [True, False, False, False, False, False, False])

    def test_depth_holes_are_not_valid_target_candidates(self):
        _, depth, K, mask = scene()
        depth[30, 30] = 0
        self.assertFalse(target_grasp_keep_mask([[.15, .15, .5]], mask, K, depth)[0])

    def test_mask_and_depth_rejections_are_counted_separately(self):
        _, depth, K, mask = scene()
        depth[30, 30] = 0
        points = [[.125, .125, .5], [.3, .15, .5], [.15, .15, .5], [.15, .15, .56]]
        keep, counts = target_grasp_keep_mask(points, mask, K, depth, return_counts=True)
        np.testing.assert_array_equal(keep, [True, False, False, False])
        self.assertEqual(counts, {"mask": 3, "valid_depth": 2, "mask_depth": 1})
        np.testing.assert_array_equal(target_grasp_keep_mask(points, mask, K, depth), keep)

    def test_empty_candidates_have_zero_counts(self):
        _, depth, K, mask = scene()
        keep, counts = target_grasp_keep_mask([], mask, K, depth, return_counts=True)
        self.assertEqual(len(keep), 0)
        self.assertEqual(counts, {"mask": 0, "valid_depth": 0, "mask_depth": 0})

    def test_terminal_reports_each_rejection_reason(self):
        from scripts.vlm_grasp_demo import GraspPreviewProcessor

        grounding = TargetGrounding(True, "香蕉", (20, 20, 39, 39), "测试", "抓香蕉", (80, 60), (80, 60), "pixel")
        grasp_result = SimpleNamespace(
            status="test", best=None,
            filter_counts={"decoded": 8, "collision_free": 6, "bbox": 5,
                           "mask": 4, "valid_depth": 3, "mask_depth": 2, "width": 1, "final": 1},
        )
        result = SimpleNamespace(grounding=grounding, segmentation=None, placement_segmentation=None,
                                 placement_error=None, placement_analysis=None, grasp_result=grasp_result,
                                 elapsed_s=0.1)
        output = io.StringIO()
        with redirect_stdout(output):
            GraspPreviewProcessor(None).report(result, Path("/unused"))
        self.assertIn("掩码内=4（框内掩码外=1）", output.getvalue())
        self.assertIn("有效深度=3（深度缺失=1）", output.getvalue())
        self.assertIn("深度一致=2（深度超差=1）", output.getvalue())

    def test_bad_mask_or_misaligned_depth_is_rejected(self):
        color, depth, K, mask = scene()
        for invalid in (mask[:30], np.zeros_like(mask), mask.astype(float) * .5):
            with self.assertRaises(ValueError):
                validate_mask(invalid, color.shape[:2])
        with self.assertRaises(ValueError):
            validate_rgbd(color, depth[:30], K)
        for bbox in ((20, 20, 80, 40), (40, 20, 20, 40), (1.5, 2, 3, 4)):
            with self.assertRaises(ValueError):
                validate_bbox(bbox, color.shape[:2])


class SAMTests(unittest.TestCase):
    def test_box_prompt_returns_only_mask_within_requested_instance_box(self):
        color, _, _, mask = scene()
        mask[5:15, 5:15] = True  # A separate object must not leak into the target.
        tensor = Mock()
        tensor.detach.return_value.cpu.return_value.numpy.return_value = mask
        segmenter = SAMSegmenter.__new__(SAMSegmenter)
        segmenter.settings = SAMSettings(Path("unused.pt"), "cpu", 16)
        segmenter.model = Mock()
        segmenter.model.predict.return_value = [SimpleNamespace(masks=SimpleNamespace(data=[tensor]))]
        result = segmenter.segment(color, (20, 20, 39, 39))
        self.assertEqual(result.area_px, 400)
        self.assertFalse(result.mask[10, 10])
        self.assertEqual(segmenter.model.predict.call_args.kwargs["bboxes"], [[20, 20, 39, 39]])
        tensor.detach.return_value.cpu.return_value.numpy.return_value = mask[:30]
        with self.assertRaises(ValueError):
            segmenter.segment(color, (20, 20, 39, 39))
        segmenter.model.predict.return_value = []
        with self.assertRaises(ValueError):
            segmenter.segment(color, (20, 20, 39, 39))


class PointSamplingTests(unittest.TestCase):
    def setUp(self):
        self.color, self.depth, self.K, self.mask = scene()
        yy, xx = np.indices(self.depth.shape)
        cloud = np.dstack([xx, yy, self.depth / 1000.]).astype(np.float32)
        self.ns = grasp_functions({"build_end_points"}, {
            "np": np, "cv2": cv2, "validate_rgbd": validate_rgbd, "validate_mask": validate_mask,
            "CameraInfo": lambda *args: None,
            "create_point_cloud_from_depth_image": lambda *a, **kw: cloud,
            "torch": SimpleNamespace(from_numpy=lambda arr: SimpleNamespace(cuda=lambda **kw: arr)),
            "o3d": SimpleNamespace(geometry=SimpleNamespace(PointCloud=SimpleNamespace),
                                   utility=SimpleNamespace(Vector3dVector=np.asarray)),
        })
        grasp_functions({"_expanded_bbox_pixel_bounds"}, self.ns)

    def test_samples_only_target_but_retains_full_collision_scene(self):
        endpoints, _, raw = self.ns["build_end_points"](
            self.color, self.depth, self.K, 32, .05, 1., target_mask=self.mask,
        )
        sampled = endpoints["point_clouds"][0]
        self.assertTrue(((sampled[:, :2] >= 20) & (sampled[:, :2] <= 39)).all())
        self.assertEqual(len(raw), 60 * 80)

    def test_sparse_target_does_not_fall_back_to_background(self):
        self.depth[self.mask] = 0
        with self.assertRaisesRegex(ValueError, "不回退"):
            self.ns["build_end_points"](
                self.color, self.depth, self.K, 32, .05, 1., target_mask=self.mask,
            )

    def test_context_includes_nearby_surface_but_excludes_distant_objects(self):
        endpoints, _, raw = self.ns["build_end_points"](
            self.color, self.depth, self.K, 4800, .05, 1., target_mask=self.mask,
            bbox_xyxy=(20, 20, 39, 39), target_context_expand_ratio=1.35,
        )
        sampled = endpoints["point_clouds"][0]
        self.assertTrue(np.any((sampled[:, 0] < 20) | (sampled[:, 0] > 39)))
        self.assertTrue(((sampled[:, :2] >= 17) & (sampled[:, :2] <= 42)).all())
        self.assertEqual(len(raw), 4800)
        self.depth[self.mask] = 0
        with self.assertRaisesRegex(ValueError, "不回退"):
            self.ns["build_end_points"](
                self.color, self.depth, self.K, 32, .05, 1., target_mask=self.mask,
                target_context_expand_ratio=1.35,
            )


class FakeGrasps:
    def __init__(self, array=None):
        self.grasp_group_array = np.array(array if array is not None else [], dtype=float).reshape(-1, 3)
        self.translations = self.grasp_group_array
        self.widths = np.full(len(self), .05)

    def __len__(self):
        return len(self.grasp_group_array)

    def __getitem__(self, index):
        return FakeGrasps(self.grasp_group_array[index])


class GraspConstraintTests(unittest.TestCase):
    def test_best_preview_uses_the_selected_pose_without_mutating_it(self):
        ns = grasp_functions({"visualization_grasps"}, {
            "GraspGroup": lambda array=None: np.empty((0, 17)) if array is None else array,
        })
        row = np.arange(17, dtype=float)
        result = SimpleNamespace(best=SimpleNamespace(grasp_array=row), grasps=np.zeros((10, 17)))
        displayed = ns["visualization_grasps"](result, "best")
        self.assertEqual(displayed.shape, (1, 17))
        np.testing.assert_array_equal(displayed[0], row)
        displayed[:] = -1
        np.testing.assert_array_equal(row, np.arange(17))

    def test_best_preview_is_empty_when_no_pose_was_selected(self):
        ns = grasp_functions({"visualization_grasps"}, {
            "GraspGroup": lambda: np.empty((0, 17)),
        })
        result = SimpleNamespace(best=None, grasps=np.ones((10, 17)))
        self.assertEqual(len(ns["visualization_grasps"](result, "best")), 0)

    def test_width_preview_uses_filtered_grasp_candidates(self):
        ns = grasp_functions({"visualization_grasps"}, {
            "GraspGroup": lambda array=None: np.empty((0, 17)) if array is None else array,
        })
        candidates = np.arange(51, dtype=float).reshape(3, 17)
        result = SimpleNamespace(grasps=candidates)
        self.assertIs(ns["visualization_grasps"](result, "width"), candidates)

    def test_raw_preview_uses_unfiltered_decoded_candidates(self):
        ns = grasp_functions({"visualization_grasps"}, {
            "GraspGroup": lambda array=None: np.empty((0, 17)) if array is None else array,
        })
        raw = np.arange(51, dtype=float).reshape(3, 17)
        filtered = raw[:1].copy()
        result = SimpleNamespace(raw_grasps=raw, grasps=filtered, best=None)
        self.assertIs(ns["visualization_grasps"](result, "raw"), raw)
        self.assertIs(ns["visualization_grasps"](result, "final"), filtered)

    def test_raw_grasps_are_copied_before_collision_filtering(self):
        from contextlib import nullcontext

        decoded = np.array([[.1, .1, .5], [.2, .2, .5], [.3, .3, .5]])
        tensor = SimpleNamespace(detach=lambda: SimpleNamespace(
            cpu=lambda: SimpleNamespace(numpy=lambda: decoded),
        ))
        detector = SimpleNamespace(detect=lambda *args, **kwargs: np.array([False, True, False]))
        namespace = grasp_functions({"infer_grasps", "copy_grasp_group"}, {
            "torch": SimpleNamespace(no_grad=nullcontext), "GraspGroup": FakeGrasps, "np": np,
            "DEFAULT_VOXEL_SIZE": .01,
            "pred_decode": lambda endpoints: [tensor],
            "ModelFreeCollisionDetector": lambda *args, **kwargs: detector,
        })
        filtered, counts, raw = namespace["infer_grasps"](
            lambda endpoints: endpoints, {}, np.zeros((3, 3)), .03,
        )
        self.assertEqual(counts["decoded"], 3)
        self.assertEqual(counts["collision_removed"], 1)
        self.assertEqual(len(filtered), 2)
        np.testing.assert_array_equal(raw.grasp_group_array, decoded)
        self.assertIsNot(raw.grasp_group_array, decoded)

    def test_explicit_mask_filters_candidates_without_yolo(self):
        import time
        color, depth, K, mask = scene()
        grasps = FakeGrasps([[.15, .15, .5], [.3, .15, .5], [.3, .3, 1.]])
        builder = Mock(return_value=({}, None, np.zeros((4800, 3))))
        ns = grasp_functions({"infer_frame", "filter_grasps_by_mask", "filter_grasps_by_bbox", "filter_grasps_by_width"}, {
            "__package__": "utils", "np": np, "time": time, "DEFAULT_VOXEL_SIZE": .01,
            "validate_bbox": validate_bbox, "validate_mask": validate_mask, "validate_rgbd": validate_rgbd,
            "target_grasp_keep_mask": target_grasp_keep_mask, "build_end_points": builder,
            "infer_grasps": Mock(return_value=(grasps, {"decoded": 3, "collision_removed": 0},
                                                FakeGrasps(grasps.grasp_group_array.copy()))),
            "copy_grasp_group": lambda g: FakeGrasps(g.grasp_group_array.copy()),
            "select_best_grasp": lambda g: object() if len(g) else None,
            "GraspNetFrameResult": SimpleNamespace,
        })
        result = ns["infer_frame"](
            None, color, depth, K, num_point=32, min_depth=.05, max_depth=1., collision_thresh=.01,
            target_bbox_xyxy=(20, 20, 39, 39), target_mask=mask, target_name="指定物体",
        )
        self.assertEqual(len(result.pre_bbox_grasps), 3)
        self.assertEqual(len(result.raw_grasps), 3)
        self.assertEqual(len(result.bbox_grasps), 2)
        self.assertEqual(len(result.grasps), 1)
        self.assertEqual(result.detections, [])
        self.assertIsNotNone(result.best)
        self.assertEqual(result.filter_counts, {
            "decoded": 3, "collision_free": 3, "bbox": 2,
            "mask": 2, "valid_depth": 2, "mask_depth": 1, "width": 1, "final": 1,
        })
        self.assertTrue(builder.call_args.kwargs["strict_target"])
        with self.assertRaises(ValueError):
            ns["infer_frame"](
                None, color, depth, K, num_point=32, min_depth=.05, max_depth=1., collision_thresh=.01,
                target_mask=mask, yolo_model=object(),
            )


class PipelineTests(unittest.TestCase):
    def make_pipeline(self, found=True):
        color, depth, K, mask = scene()
        grounding = TargetGrounding(found, "红积木", (20, 20, 39, 39) if found else None,
                                    "测试", "积木", (80, 60), (80, 60), "pixel")
        client = SimpleNamespace(ground=Mock(return_value=grounding))
        segmenter = SimpleNamespace(segment=Mock(return_value=SegmentedTarget(mask, (20, 20, 39, 39), 400)))
        backend = SimpleNamespace(infer_frame=Mock(return_value=SimpleNamespace(best=None, grasps=[])))
        return VLMGraspPipeline({}, client, segmenter=segmenter, net=object(), grasp_backend=backend), color, depth, K

    def test_same_snapshot_and_exact_mask_are_forwarded(self):
        pipeline, color, depth, K = self.make_pipeline()
        with redirect_stdout(io.StringIO()):
            result = pipeline.process("积木", color, depth, K)
        call = pipeline.backend.infer_frame.call_args
        np.testing.assert_array_equal(call.args[1], color)
        np.testing.assert_array_equal(call.args[2], depth)
        np.testing.assert_array_equal(call.args[3], K)
        self.assertIs(call.kwargs["target_mask"], result.segmentation.mask)
        self.assertEqual(call.kwargs["target_bbox_xyxy"], result.grounding.bbox_xyxy)
        self.assertEqual(call.kwargs["target_context_expand_ratio"], 1.35)
        color[:] = 255
        depth[:] = 0
        K[:] = 0
        self.assertFalse(call.args[1].any())
        self.assertTrue((call.args[2] == 500).all())
        self.assertEqual(result.camera_matrix[0, 0], 100)

    def test_no_target_skips_sam_and_graspnet(self):
        pipeline, color, depth, K = self.make_pipeline(False)
        with redirect_stdout(io.StringIO()):
            result = pipeline.process("积木", color, depth, K)
        self.assertIsNone(result.grasp_result)
        pipeline.segmenter.segment.assert_not_called()
        pipeline.backend.infer_frame.assert_not_called()

    def test_bad_depth_rejected_before_remote_request(self):
        pipeline, color, depth, K = self.make_pipeline()
        with self.assertRaises(ValueError):
            pipeline.process("积木", color, depth[:30], K)
        pipeline.client.ground.assert_not_called()

    def test_sam_failure_never_runs_graspnet(self):
        pipeline, color, depth, K = self.make_pipeline()
        pipeline.segmenter.segment.side_effect = ValueError("分割失败")
        with redirect_stdout(io.StringIO()), self.assertRaises(ValueError):
            pipeline.process("积木", color, depth, K)
        pipeline.backend.infer_frame.assert_not_called()

    def test_preview_labels_pick_and_exports_matching_rgbd_mask_and_pose(self):
        from scripts.vlm_grasp_demo import GraspPreviewProcessor
        pipeline, color, depth, K = self.make_pipeline()
        best = SimpleNamespace(translation=np.array([.15, .15, .5]), rotation_matrix=np.eye(3),
                               score=.8, width=.05, depth=.02)
        pipeline.backend.infer_frame.return_value = SimpleNamespace(
            best=best, grasps=FakeGrasps([best.translation]), status="target grasp ready",
        )
        pipeline.backend.draw_best_grasp_projection = Mock()
        pipeline.client.settings = SimpleNamespace(model="test-vision-model")
        pipeline.segmenter.settings = SAMSettings(Path("test-sam.pt"))
        with redirect_stdout(io.StringIO()):
            result = pipeline.process("积木", color, depth, K)
        processor = GraspPreviewProcessor(pipeline)
        # The GUI draws before saving: K must come from this result, not mutable
        # processor state or the next camera frame.
        display = processor.draw(color, result)
        self.assertEqual(display.shape, color.shape)
        np.testing.assert_array_equal(display[30, 20], [0, 255, 0])
        self.assertFalse(color.any())
        pipeline.backend.draw_best_grasp_projection.assert_not_called()
        with tempfile.TemporaryDirectory() as temp:
            directory = processor.save(Path(temp), color, result, depth_mm=depth, K=K)
            mask = cv2.imread(str(directory / "mask.png"), cv2.IMREAD_GRAYSCALE)
            np.testing.assert_array_equal(mask != 0, result.segmentation.mask)
            with np.load(directory / "rgbd.npz") as data:
                np.testing.assert_array_equal(data["depth_mm"], depth)
                np.testing.assert_array_equal(data["camera_matrix"], K)
            metadata = json.loads((directory / "pipeline.json").read_text())
            self.assertEqual(metadata["candidate_count"], 1)
            self.assertEqual(metadata["best_grasp"]["translation_m"], [.15, .15, .5])
            self.assertEqual(metadata["best_grasp"]["coordinate_frame"], "camera_optical")
            self.assertEqual(np.load(directory / "grasp_candidates.npy").shape, (1, 3))

    def test_no_candidate_preview_does_not_draw_a_fake_grasp_at_box_center(self):
        from scripts.vlm_grasp_demo import GraspPreviewProcessor
        pipeline, color, depth, K = self.make_pipeline()
        pipeline.backend.draw_best_grasp_projection = Mock()
        with redirect_stdout(io.StringIO()):
            result = pipeline.process("积木", color, depth, K)
        processor = GraspPreviewProcessor(pipeline)
        self.assertEqual(processor.status(result), "NO VALID GRASP")
        display = processor.draw(color, result)
        self.assertFalse(np.any(np.all(display == [0, 0, 255], axis=2)))


class PreviewConfigTests(unittest.TestCase):
    def test_yaml_selects_all_candidates_or_best_grasp(self):
        from scripts.vlm_grasp_demo import resolve_open3d_grasps

        self.assertEqual(resolve_open3d_grasps({}), "width")
        self.assertEqual(resolve_open3d_grasps({"vlm_grasp": {"open3d_grasps": "width"}}), "width")
        self.assertEqual(resolve_open3d_grasps({"vlm_grasp": {"open3d_grasps": "best"}}), "best")
        self.assertEqual(resolve_open3d_grasps({"vlm_grasp": {"open3d_grasps": "best"}}, "bbox"), "bbox")
        for invalid in ("final", "all", "", 1, None):
            cfg = {"vlm_grasp": {"open3d_grasps": invalid}}
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                resolve_open3d_grasps(cfg)

    def test_dashboard_defaults_and_validates_equal_tile_size(self):
        from scripts.vlm_grasp_demo import resolve_dashboard

        self.assertEqual(resolve_dashboard({}), (True, 640, 360, 0.12))
        cfg = {"vlm_grasp": {"dashboard": {
            "enabled": False, "tile_width": 500, "tile_height": 300,
        }}}
        self.assertEqual(resolve_dashboard(cfg), (False, 500, 300, 0.12))
        with self.assertRaises(ValueError):
            resolve_dashboard({"vlm_grasp": {"dashboard": {"tile_width": 100}}})
        with self.assertRaises(ValueError):
            resolve_dashboard({"vlm_grasp": {"dashboard": {"grasp_cloud_focus_radius_m": 0}}})


class GraspCloudFocusTests(unittest.TestCase):
    def test_focus_cloud_removes_distant_scene_points_around_grasp(self):
        import open3d as o3d
        from utils.graspnet_utils import Open3DGraspWindow

        near = np.array([
            [-0.10, -0.04, 0.48], [0.10, 0.04, 0.52], [0.0, 0.0, 0.50],
        ])
        far = np.array([[1.2, 0.0, 1.5], [-1.0, 0.8, 1.2]])
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(np.vstack([np.repeat(near, 8, axis=0), far]))
        viewer = Open3DGraspWindow.__new__(Open3DGraspWindow)
        viewer._focus_radius_m = 0.18
        focused, center = viewer._focus_cloud(cloud, FakeGrasps([[0.0, 0.0, 0.5]]))
        self.assertEqual(len(focused.points), 24)
        np.testing.assert_allclose(center, [0.0, 0.0, 0.5])
        self.assertLess(np.max(np.abs(np.asarray(focused.points) - center)), 0.18 + 1e-9)

    def test_focus_cloud_keeps_container_depth_context(self):
        import open3d as o3d
        from utils.graspnet_utils import Open3DGraspWindow

        pick = np.repeat([[0.0, 0.0, 0.5]], 20, axis=0)
        box = np.repeat([[0.32, 0.02, 0.55]], 20, axis=0)
        irrelevant = np.array([[1.4, 1.0, 1.6]])
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(np.vstack([pick, box, irrelevant]))
        viewer = Open3DGraspWindow.__new__(Open3DGraspWindow)
        viewer._focus_radius_m = 0.12
        focused, center = viewer._focus_cloud(
            cloud,
            FakeGrasps([[0.0, 0.0, 0.5]]),
            context_points=box,
        )
        points = np.asarray(focused.points)
        self.assertEqual(len(points), 40)
        self.assertTrue(np.any(points[:, 0] > 0.3))
        self.assertLess(center[0], 0.3)

    def test_container_mask_supplies_depth_context_when_geometry_failed(self):
        import open3d as o3d
        from scripts.vlm_grasp_demo import GraspPreviewProcessor

        mask = np.zeros((10, 10), dtype=bool)
        mask[5, 5] = True
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(np.array([
            [0.25, 0.25, 0.5],  # projects to the container mask pixel
            [0.05, 0.05, 0.5],
        ]))
        result = SimpleNamespace(
            placement_analysis=None,
            placement_segmentation=SimpleNamespace(mask=mask),
            rejected_container_segmentation=None,
            camera_matrix=np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 1.0]]),
        )
        points = GraspPreviewProcessor._container_context_points(
            result, SimpleNamespace(o3d_cloud=cloud),
        )
        np.testing.assert_allclose(points, [[0.25, 0.25, 0.5]])

    def test_vlm_box_supplies_depth_context_without_container_sam(self):
        import open3d as o3d
        from scripts.vlm_grasp_demo import GraspPreviewProcessor
        from utils.graspnet_utils import Open3DGraspWindow

        pick = np.repeat([[0.0, 0.0, 0.5]], 20, axis=0)
        box = np.repeat([[0.32, 0.02, 0.55]], 20, axis=0)
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(np.vstack([pick, box]))
        result = SimpleNamespace(
            grounding=SimpleNamespace(placement=SimpleNamespace(
                required=True, found=True, bbox_xyxy=(5, 0, 8, 2), rejected_container=None,
            )),
            placement_analysis=None,
            placement_segmentation=None,
            rejected_container_segmentation=None,
            camera_matrix=np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 1.0]]),
        )
        context = GraspPreviewProcessor._container_context_points(
            result, SimpleNamespace(o3d_cloud=cloud),
        )
        np.testing.assert_allclose(context, box)
        viewer = Open3DGraspWindow.__new__(Open3DGraspWindow)
        viewer._focus_radius_m = 0.12
        focused, _ = viewer._focus_cloud(cloud, FakeGrasps([[0.0, 0.0, 0.5]]), context)
        self.assertEqual(len(focused.points), 40)


class PreviewWindowsTests(unittest.TestCase):
    def setUp(self):
        from scripts.vlm_grasp_demo import GraspPreviewProcessor
        pipeline, self.color, depth, K = PipelineTests().make_pipeline()
        self.viewer = Mock()
        self.viewer.poll.return_value = True
        self.viewer.frame.return_value = np.full((90, 160, 3), 80, np.uint8)
        pipeline.backend.Open3DGraspWindow = Mock(return_value=self.viewer)
        self.selected_grasps = FakeGrasps([[.15, .15, .5]])
        pipeline.backend.visualization_grasps = Mock(return_value=self.selected_grasps)
        pipeline.backend.draw_best_grasp_projection = Mock()
        pipeline.client.settings = SimpleNamespace(model="test-model")
        pipeline.segmenter.settings = SAMSettings(Path("test-sam.pt"))
        best = SimpleNamespace(translation=np.array([.15, .15, .5]), rotation_matrix=np.eye(3),
                               score=.8, width=.05, depth=.02)
        pipeline.backend.infer_frame.return_value = SimpleNamespace(
            best=best, grasps=FakeGrasps([[.15, .15, .5], [.16, .16, .5]]), o3d_cloud=object(), status="test",
        )
        with redirect_stdout(io.StringIO()):
            self.result = pipeline.process("积木", self.color, depth, K)
        self.processor = GraspPreviewProcessor(pipeline, top_k=20)

    def test_combined_white_mask_and_all_raw_grasps_are_displayed(self):
        with patch.object(cv2, "namedWindow"), patch.object(cv2, "resizeWindow"), \
             patch.object(cv2, "imshow") as showing, redirect_stdout(io.StringIO()):
            self.processor.show_result(self.color, self.result)
        showing.assert_not_called()
        np.testing.assert_array_equal(
            self.processor._mask_frame[:, :, 0],
            self.result.segmentation.mask.astype(np.uint8) * 255,
        )
        self.processor.pipeline.backend.visualization_grasps.assert_called_once_with(self.result.grasp_result, "raw")
        self.viewer.update.assert_called_once_with(
            self.result.grasp_result.o3d_cloud, self.selected_grasps, context_points=None,
        )
        self.processor.pipeline.backend.Open3DGraspWindow.assert_called_once_with(
            "VLM GraspNet - All Raw Decoded Grasps", None, apply_nms=False, visible=False,
            focus_radius_m=0.12,
        )
        self.viewer.frame.assert_called_once_with()

    def test_raw_grasps_still_display_when_final_selection_is_empty(self):
        self.result.grasp_result.best = None
        self.result.grasp_result.grasps = FakeGrasps()
        with patch.object(cv2, "namedWindow"), patch.object(cv2, "resizeWindow"), \
             patch.object(cv2, "imshow"), redirect_stdout(io.StringIO()):
            self.processor.show_result(self.color, self.result)
        self.processor.pipeline.backend.visualization_grasps.assert_called_once_with(self.result.grasp_result, "raw")
        self.viewer.update.assert_called_once_with(
            self.result.grasp_result.o3d_cloud, self.selected_grasps, context_points=None,
        )

    def test_discard_clears_old_3d_and_new_result_reuses_window(self):
        with patch.object(cv2, "namedWindow"), patch.object(cv2, "resizeWindow"), \
             patch.object(cv2, "imshow"), redirect_stdout(io.StringIO()):
            self.processor.show_result(self.color, self.result)
            self.processor.reset_preview(self.color.shape[:2], "PROCESSING")
            self.assertIsNone(self.processor._shown_result)
            self.viewer.clear.assert_called_once()
            self.processor.show_result(self.color, self.result)
        self.assertEqual(self.viewer.update.call_count, 2)
        self.assertEqual(self.processor.pipeline.backend.Open3DGraspWindow.call_count, 1)

    def test_not_found_clears_previous_mask_and_cloud(self):
        with patch.object(cv2, "namedWindow"), patch.object(cv2, "resizeWindow"), \
             patch.object(cv2, "imshow") as showing, redirect_stdout(io.StringIO()):
            self.processor.show_result(self.color, self.result)
            self.processor.show_result(self.color, SimpleNamespace(segmentation=None))
        self.viewer.clear.assert_called_once()
        self.assertIsNone(self.processor._shown_result)
        showing.assert_not_called()
        self.assertEqual(self.processor._mask_frame.shape[:2], self.color.shape[:2])
        self.assertFalse(np.array_equal(
            self.processor._mask_frame[:, :, 0],
            self.result.segmentation.mask.astype(np.uint8) * 255,
        ))

    def test_closed_3d_is_polled_and_resources_are_released(self):
        with patch.object(cv2, "namedWindow"), patch.object(cv2, "resizeWindow"), \
             patch.object(cv2, "imshow"), redirect_stdout(io.StringIO()):
            self.processor.show_result(self.color, self.result)
        self.viewer.poll.return_value = False
        with patch.object(cv2, "getWindowProperty", return_value=1), patch.object(cv2, "destroyWindow") as destroy:
            self.processor.poll_preview()
            self.processor.close_preview()
        self.viewer.close.assert_called_once()
        destroy.assert_not_called()
        self.assertIsNone(self.processor._viewer)

    def test_disabled_auxiliary_windows_do_not_create_gui(self):
        self.processor.show_mask = self.processor.show_open3d = False
        with patch.object(cv2, "namedWindow") as named, patch.object(cv2, "imshow") as showing:
            self.processor.show_result(self.color, self.result)
        named.assert_not_called()
        showing.assert_not_called()
        self.processor.pipeline.backend.Open3DGraspWindow.assert_not_called()

    def test_screenshot_is_only_saved_for_the_current_displayed_result(self):
        with patch.object(cv2, "namedWindow"), patch.object(cv2, "resizeWindow"), \
             patch.object(cv2, "imshow"), redirect_stdout(io.StringIO()):
            self.processor.show_result(self.color, self.result)
            with tempfile.TemporaryDirectory() as temp:
                directory = self.processor.save(Path(temp), self.color, self.result)
                self.viewer.capture.assert_called_once_with(directory / "grasp_cloud.png")
                self.assertTrue((directory / "dashboard.png").is_file())
                self.processor.reset_preview(self.color.shape[:2], "DISCARDED")
                self.processor.save(Path(temp), self.color, self.result)
                self.assertEqual(self.viewer.capture.call_count, 1)

    def test_dashboard_has_four_equal_tiles_and_configured_size(self):
        with redirect_stdout(io.StringIO()):
            self.processor.show_result(self.color, self.result)
        dashboard = self.processor.compose_display(self.processor.draw(self.color, self.result))
        self.assertEqual(dashboard.shape, (720, 1280, 3))
        self.assertEqual(self.processor.display_size(), (1280, 720))

    def test_dashboard_panels_can_be_focused_by_key_or_double_click(self):
        with redirect_stdout(io.StringIO()):
            self.processor.show_result(self.color, self.result)
        self.assertTrue(self.processor.handle_preview_key(ord("3")))
        self.assertEqual(self.processor._focused_panel, 2)
        focused = self.processor.compose_display(self.color)
        self.assertEqual(focused.shape, (720, 1280, 3))
        self.processor.handle_mouse(cv2.EVENT_LBUTTONDBLCLK, 10, 10, 0)
        self.assertIsNone(self.processor._focused_panel)
        self.processor.handle_mouse(cv2.EVENT_LBUTTONDBLCLK, 100, 500, 0)
        self.assertEqual(self.processor._focused_panel, 2)
        self.assertTrue(self.processor.handle_preview_key(ord("0")))
        self.assertIsNone(self.processor._focused_panel)

    def test_grasp_cloud_mouse_controls_hidden_open3d_view(self):
        with redirect_stdout(io.StringIO()):
            self.processor.show_result(self.color, self.result)
        self.viewer.frame.reset_mock()
        self.processor.handle_mouse(cv2.EVENT_LBUTTONDOWN, 100, 500, 0)
        self.processor.handle_mouse(cv2.EVENT_MOUSEMOVE, 112, 493, cv2.EVENT_FLAG_LBUTTON)
        self.processor.handle_mouse(cv2.EVENT_LBUTTONUP, 112, 493, 0)
        self.viewer.rotate.assert_called_once_with(12, -7)
        self.processor.handle_mouse(cv2.EVENT_RBUTTONDOWN, 100, 500, 0)
        self.processor.handle_mouse(cv2.EVENT_MOUSEMOVE, 95, 509, cv2.EVENT_FLAG_RBUTTON)
        self.processor.handle_mouse(cv2.EVENT_RBUTTONUP, 95, 509, 0)
        self.viewer.pan.assert_called_once_with(-5, 9)
        self.processor.handle_mouse(cv2.EVENT_MOUSEWHEEL, 100, 500, 120 << 16)
        self.viewer.zoom.assert_called_once_with(1.0)
        self.assertEqual(self.viewer.frame.call_count, 3)

    def test_dashboard_binds_one_opencv_mouse_callback(self):
        with patch.object(cv2, "setMouseCallback") as callback:
            self.processor.bind_window("dashboard")
        callback.assert_called_once_with("dashboard", self.processor.handle_mouse)


if __name__ == "__main__":
    unittest.main()
