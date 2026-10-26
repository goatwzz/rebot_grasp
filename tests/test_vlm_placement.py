"""Two-role task grounding, coordinate checks, display, and preview integration."""

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

from scripts.vlm_target_demo import TargetPreviewProcessor, report_result, save_result
from scripts.vlm_grasp_demo import GraspPreviewProcessor
from utils.vlm_utils import (
    VLMClient, VLMError, VLMSettings, _draw_detection_box, draw_grounding, parse_grounding,
)
from utils.vlm_grasp_pipeline import VLMGraspPipeline
from utils.sam_utils import SegmentedTarget


def task_payload(**placement_overrides):
    placement = {"required": True, "found": True, "target_name": "盘子", "bbox_xyxy": [60, 10, 90, 60],
                 "relation": "in", "reason": "指令要求放入右侧盘子", "requested_name": "盘子", "selection_mode": "explicit"}
    placement.update(placement_overrides)
    return {"found": True, "target_name": "香蕉", "bbox_xyxy": [10, 20, 40, 60],
            "coordinate_space": "pixel", "reason": "左侧黄色香蕉", "placement": placement}


def parse_task(data):
    return parse_grounding(json.dumps(data, ensure_ascii=False), instruction="把香蕉放到盘子里面",
                           image_size=(200, 160), request_image_size=(100, 80), coordinate_space="pixel",
                           require_placement_schema=True)


class PlacementGroundingTests(unittest.TestCase):
    def test_invalid_container_cross_is_three_times_the_previous_large_box_size(self):
        image = np.zeros((300, 300, 3), dtype=np.uint8)
        _draw_detection_box(image, (40, 40, 260, 260), "INVALID CONTAINER", (0, 0, 255), invalid=True)
        # The previous large-box radius was 22 px.  The new diagonal reaches
        # 66 px from the center while remaining inside the detection box.
        np.testing.assert_array_equal(image[90, 90], [0, 0, 255])
        np.testing.assert_array_equal(image[210, 210], [0, 0, 255])
        np.testing.assert_array_equal(image[80, 80], [0, 0, 0])

    def test_rejected_container_is_detected_but_never_a_placement_target(self):
        result = parse_task(task_payload(
            required=False, found=False, target_name="", bbox_xyxy=None, relation=None,
            requested_name="", selection_mode="grasp_only", reason="盒子侧翻，仅抓取",
            rejected_container={"target_name": "纸盒", "bbox_xyxy": [60, 10, 90, 60],
                                "reason": "盒子侧翻，开口朝向侧面"},
        ))
        placement = result.placement
        self.assertFalse(placement.required)
        self.assertFalse(placement.found)
        self.assertIsNone(placement.bbox_xyxy)
        self.assertEqual(placement.rejected_container.bbox_xyxy, (120, 20, 180, 120))
        image = np.zeros((160, 200, 3), np.uint8)
        annotated = draw_grounding(image, result, mark_center=False, show_warning=False)
        np.testing.assert_array_equal(annotated[70, 20], [0, 255, 0])
        np.testing.assert_array_equal(annotated[70, 120], [0, 0, 255])
        np.testing.assert_array_equal(annotated[70, 150], [0, 0, 255])
        self.assertFalse(image.any())
        with tempfile.TemporaryDirectory() as temp:
            directory = save_result(Path(temp), image, result)
            stored = json.loads((directory / "result.json").read_text())
            self.assertEqual(stored["placement"]["rejected_container"]["target_name"], "纸盒")
            self.assertEqual(stored["placement"]["rejected_container"]["bbox_xyxy"], [120, 20, 180, 120])

    def test_rejected_container_may_coexist_with_safe_fallback(self):
        result = parse_task(task_payload(
            selection_mode="fallback", requested_name="纸盒",
            rejected_container={"target_name": "纸盒", "bbox_xyxy": [42, 10, 55, 60], "reason": "纸盒封闭"},
        ))
        image = np.zeros((160, 200, 3), np.uint8)
        annotated = draw_grounding(image, result, mark_center=False)
        np.testing.assert_array_equal(annotated[70, 84], [0, 0, 255])
        np.testing.assert_array_equal(annotated[70, 120], [0, 165, 255])
        current = draw_grounding(image, parse_task(task_payload()), mark_center=False)
        np.testing.assert_array_equal(current[70, 84], [0, 0, 0])

    def test_rejected_container_validation_rejects_invalid_or_duplicate_boxes(self):
        valid = {"target_name": "纸盒", "bbox_xyxy": [42, 10, 55, 60], "reason": "纸盒封闭"}
        invalid = ([], {**valid, "target_name": ""}, {**valid, "reason": ""},
                   {**valid, "bbox_xyxy": [-1, 10, 55, 60]},
                   {**valid, "bbox_xyxy": [10, 20, 40, 60]},
                   {**valid, "bbox_xyxy": [60, 10, 90, 60]})
        for rejected in invalid:
            with self.subTest(rejected=rejected), self.assertRaises(VLMError):
                parse_task(task_payload(rejected_container=rejected))

    def test_rejected_container_uses_normalized_coordinates_too(self):
        data = task_payload(
            bbox_xyxy=[600, 200, 900, 800],
            rejected_container={"target_name": "纸盒", "bbox_xyxy": [400, 100, 550, 900], "reason": "纸盒封闭"},
        )
        data.update(bbox_xyxy=[100, 300, 300, 700], coordinate_space="normalized_1000")
        result = parse_grounding(json.dumps(data), instruction="放入盒子", image_size=(1280, 720),
                                 request_image_size=(640, 360), coordinate_space="normalized_1000")
        self.assertEqual(result.placement.rejected_container.bbox_xyxy, (512, 72, 704, 648))

    def test_stage_two_keeps_rgb_size_and_uses_english_labels(self):
        grounding = parse_task(task_payload(
            required=False, found=False, target_name="", bbox_xyxy=None, relation=None,
            requested_name="", selection_mode="grasp_only", reason="本轮仅抓取",
            rejected_container={"target_name": "纸盒", "bbox_xyxy": [60, 10, 90, 60],
                                "reason": "开口朝侧面，无法竖直放入"},
        ))
        image = np.zeros((160, 200, 3), np.uint8)
        result = SimpleNamespace(grounding=grounding, placement_segmentation=None,
                                 placement_analysis=None, placement_error=None)
        with patch("utils.vlm_utils.cv2.putText", wraps=cv2.putText) as put_text:
            display = GraspPreviewProcessor(None).draw(image, result)
        self.assertEqual(display.shape, image.shape)
        labels = [call.args[1] for call in put_text.call_args_list]
        self.assertEqual(labels, ["PICK", "INVALID"])
        np.testing.assert_array_equal(display[70, 150], [0, 0, 255])
        self.assertFalse(image.any())

    def test_valid_destination_labels_are_english(self):
        image = np.zeros((160, 200, 3), np.uint8)
        with patch("utils.vlm_utils.cv2.putText", wraps=cv2.putText) as put_text:
            display = draw_grounding(image, parse_task(task_payload()), mark_center=False)
        self.assertEqual(display.shape, image.shape)
        self.assertEqual([call.args[1] for call in put_text.call_args_list], ["PICK", "PLACE"])
        np.testing.assert_array_equal(display[70, 20], [0, 255, 0])
        np.testing.assert_array_equal(display[70, 120], [0, 165, 255])

    def test_pick_and_destination_are_mapped_independently_from_same_image(self):
        result = parse_task(task_payload())
        self.assertEqual(result.bbox_xyxy, (20, 40, 80, 120))
        self.assertEqual(result.placement.bbox_xyxy, (120, 20, 180, 120))
        self.assertEqual(result.placement.target_name, "盘子")
        self.assertEqual(result.placement.relation, "in")
        self.assertTrue(result.targets_complete)
        self.assertTrue(result.to_dict()["targets_complete"])

    def test_normalized_destination_uses_original_width_and_height(self):
        data = task_payload(bbox_xyxy=[600, 200, 900, 800])
        data.update(bbox_xyxy=[100, 300, 300, 700], coordinate_space="normalized_1000")
        result = parse_grounding(json.dumps(data), instruction="放入盒子", image_size=(1280, 720),
                                 request_image_size=(1280, 720), coordinate_space="normalized_1000",
                                 require_placement_schema=True)
        self.assertEqual(result.bbox_xyxy, (128, 216, 384, 504))
        self.assertEqual(result.placement.bbox_xyxy, (768, 144, 1152, 576))

    def test_missing_destination_is_a_partial_task_with_visible_pick(self):
        result = parse_task(task_payload(found=False, bbox_xyxy=None, reason="没有可定位的盘子或替代放置区域"))
        self.assertTrue(result.found)
        self.assertFalse(result.targets_complete)
        self.assertTrue(result.placement.required)
        self.assertIsNone(result.placement.bbox_xyxy)
        processor = TargetPreviewProcessor(None)
        self.assertIn("PLACE TARGET NOT CONFIRMED", processor.status(result))
        self.assertIn("PLACE TARGET NOT CONFIRMED", GraspPreviewProcessor(None).status(
            SimpleNamespace(grounding=result, grasp_result=None)))

    def test_automatic_destination_and_explicit_fallback_are_preserved(self):
        automatic = parse_task(task_payload(target_name="盒子", requested_name="", selection_mode="auto",
                                            reason="没有指定目的地，自动选择可容纳香蕉的盒子"))
        self.assertTrue(automatic.placement.required)
        self.assertEqual(automatic.placement.selection_mode, "auto")
        self.assertTrue(automatic.targets_complete)
        fallback = parse_task(task_payload(target_name="盒子", selection_mode="fallback",
                                           reason="没有找到盘子，选择现有盒子作为替代"))
        self.assertEqual(fallback.placement.requested_name, "盘子")
        self.assertEqual(fallback.placement.target_name, "盒子")
        output = io.StringIO()
        with redirect_stdout(output):
            report_result(fallback, Path("/unused"))
        self.assertIn("使用替代放置目标", output.getvalue())
        self.assertIn("原要求=盘子", output.getvalue())

    def test_placement_mode_cannot_silently_lose_the_requested_destination(self):
        for overrides in ({"selection_mode": "unknown"}, {"requested_name": ""},
                          {"selection_mode": "auto"}, {"selection_mode": "fallback", "requested_name": ""}):
            with self.subTest(overrides=overrides), self.assertRaises(VLMError):
                parse_task(task_payload(**overrides))

    def test_explicit_no_placement_clears_old_destination_overlay(self):
        full = parse_task(task_payload())
        pick_only = parse_task(task_payload(required=False, found=False, target_name="", bbox_xyxy=None,
                                           relation=None, reason="明确要求不放置", requested_name="", selection_mode="disabled"))
        self.assertTrue(pick_only.targets_complete)
        image = np.zeros((160, 200, 3), np.uint8)
        old = draw_grounding(image, full)
        current = draw_grounding(image, pick_only)
        np.testing.assert_array_equal(old[70, 120], [0, 165, 255])
        np.testing.assert_array_equal(current[70, 120], [0, 0, 0])

    def test_bad_container_pose_becomes_grasp_only_without_destination_overlay(self):
        result = parse_task(task_payload(
            required=False, found=False, target_name="", bbox_xyxy=None, relation=None,
            reason="盒子侧翻且开口朝向侧面，不适合放入物体；本轮仅抓取。",
            requested_name="", selection_mode="grasp_only",
        ))
        self.assertTrue(result.targets_complete)
        self.assertFalse(result.placement.required)
        self.assertEqual(result.placement.selection_mode, "grasp_only")
        image = np.zeros((160, 200, 3), np.uint8)
        annotated = draw_grounding(image, result)
        np.testing.assert_array_equal(annotated[70, 120], [0, 0, 0])
        self.assertIn("INVALID CONTAINER POSE", TargetPreviewProcessor(None).status(result))
        self.assertTrue(annotated[-40:, :].any())
        output = io.StringIO()
        with redirect_stdout(output):
            report_result(result, Path("/unused"))
        self.assertIn("错误的容器摆放姿态", output.getvalue())
        self.assertIn("本轮不执行放置", output.getvalue())

    def test_grasp_only_contract_rejects_stale_or_invalid_destination_fields(self):
        base = dict(required=False, found=False, target_name="", bbox_xyxy=None, relation=None,
                    reason="盒子倒扣，本轮仅抓取", requested_name="", selection_mode="grasp_only")
        invalid = (
            {"found": True}, {"target_name": "盒子"}, {"bbox_xyxy": [60, 10, 90, 60]},
            {"relation": "in"}, {"requested_name": "盒子"}, {"selection_mode": "fallback"},
        )
        for override in invalid:
            with self.subTest(override=override), self.assertRaises(VLMError):
                parse_task(task_payload(**{**base, **override}))

    def test_missing_schema_is_not_silently_treated_as_pick_only(self):
        data = task_payload()
        del data["placement"]
        with self.assertRaisesRegex(VLMError, "placement"):
            parse_task(data)
        legacy = parse_grounding(json.dumps(data), instruction="旧记录", image_size=(200, 160),
                                 request_image_size=(100, 80), coordinate_space="pixel")
        self.assertIsNone(legacy.placement)

    def test_invalid_destination_and_same_object_for_both_roles_are_rejected(self):
        for overrides in ({"bbox_xyxy": [60, 10, 101, 60]}, {"bbox_xyxy": [90, 10, 60, 60]},
                          {"bbox_xyxy": [10, 20, 40, 60]}, {"relation": "unknown"}, {"required": "true"},
                          {"required": False}, {"found": False}, {"target_name": ""}):
            with self.subTest(overrides=overrides), self.assertRaises(VLMError):
                parse_task(task_payload(**overrides))

    def test_annotation_terminal_and_saved_json_show_both_roles(self):
        result = parse_task(task_payload())
        image = np.zeros((160, 200, 3), np.uint8)
        annotated = draw_grounding(image, result)
        np.testing.assert_array_equal(annotated[70, 20], [0, 255, 0])
        np.testing.assert_array_equal(annotated[70, 120], [0, 165, 255])
        self.assertFalse(image.any())
        with tempfile.TemporaryDirectory() as temp:
            directory = save_result(Path(temp), image, result)
            stored = json.loads((directory / "result.json").read_text())
            self.assertEqual(stored["placement"]["target_name"], "盘子")
            self.assertEqual(stored["placement"]["bbox_xyxy"], [120, 20, 180, 120])
            output = io.StringIO()
            with redirect_stdout(output):
                report_result(result, directory)
            self.assertIn("抓取目标： 香蕉", output.getvalue())
            self.assertIn("放置目标：盘子", output.getvalue())

    def test_one_request_returns_both_targets(self):
        response = {"choices": [{"message": {"content": json.dumps(task_payload())}}]}
        settings = VLMSettings("https://example.invalid/v1", "test-model")
        with patch.dict("os.environ", {"VLM_API_KEY": "test-secret"}), patch("utils.vlm_utils.urlopen") as opening:
            opening.return_value.__enter__.return_value.read.return_value = json.dumps(response).encode()
            result = VLMClient(settings).ground("把香蕉放到盘子里", np.zeros((80, 100, 3), np.uint8))
        self.assertEqual(opening.call_count, 1)
        self.assertEqual(result.target_name, "香蕉")
        self.assertEqual(result.placement.target_name, "盘子")
        request_payload = json.loads(opening.call_args.args[0].data.decode("utf-8"))
        prompt = request_payload["messages"][0]["content"]
        self.assertIn("upside-down container", prompt)
        self.assertIn("lying on its side", prompt)
        self.assertIn("faces substantially upward", prompt)
        self.assertIn("selection_mode='grasp_only'", prompt)
        self.assertIn("move approximately straight downward from above", prompt)
        self.assertIn("require pushing the object horizontally", prompt)
        self.assertIn("Do NOT evade this check", prompt)
        self.assertIn("broad solid panel faces the camera", prompt)
        self.assertIn("DETECTION IS INDEPENDENT OF PLACEMENT SAFETY", prompt)
        self.assertIn("placement.rejected_container", prompt)
        self.assertNotIn("Do not return a box for an unusable container", prompt)

    def test_both_roles_are_segmented_but_only_pick_mask_is_sent_to_graspnet(self):
        grounding = parse_task(task_payload())
        color = np.zeros((160, 200, 3), np.uint8)
        depth = np.full((160, 200), 500, np.uint16)
        K = np.diag([100., 100., 1.])
        mask = np.zeros(depth.shape, bool)
        mask[40:121, 20:81] = True
        place_mask = np.zeros(depth.shape, bool)
        place_mask[20:121, 120:181] = True
        segmenter = SimpleNamespace(segment=Mock(side_effect=[
            SegmentedTarget(mask, grounding.bbox_xyxy, int(mask.sum())),
            SegmentedTarget(place_mask, grounding.placement.bbox_xyxy, int(place_mask.sum())),
        ]))
        backend = SimpleNamespace(infer_frame=Mock(return_value=SimpleNamespace(best=None, grasps=[])))
        pipeline = VLMGraspPipeline({}, SimpleNamespace(ground=Mock(return_value=grounding)),
                                    segmenter=segmenter, net=object(), grasp_backend=backend)
        with redirect_stdout(io.StringIO()):
            result = pipeline.process("把香蕉放到盘子里", color, depth, K)
        self.assertEqual(segmenter.segment.call_count, 2)
        self.assertEqual(segmenter.segment.call_args_list[0].args[1], grounding.bbox_xyxy)
        self.assertEqual(segmenter.segment.call_args_list[1].args[1], grounding.placement.bbox_xyxy)
        self.assertIs(backend.infer_frame.call_args.kwargs["target_mask"], mask)
        self.assertIs(result.placement_segmentation.mask, place_mask)
        self.assertEqual(result.grounding.placement.target_name, "盘子")
        self.assertIn("PLACE TARGET FOUND", GraspPreviewProcessor(pipeline).status(
            replace(result, grasp_result=SimpleNamespace(best=object(), grasps=[object()]))))

    def test_grasp_only_segments_rejected_container_without_placement_geometry(self):
        grounding = parse_task(task_payload(
            required=False, found=False, target_name="", bbox_xyxy=None, relation=None,
            reason="盒子倒扣，开口不可用；本轮仅抓取。", requested_name="", selection_mode="grasp_only",
            rejected_container={"target_name": "纸盒", "bbox_xyxy": [60, 10, 90, 60], "reason": "盒子倒扣"},
        ))
        color = np.zeros((160, 200, 3), np.uint8)
        depth = np.full((160, 200), 500, np.uint16)
        K = np.diag([100., 100., 1.])
        mask = np.zeros(depth.shape, bool)
        mask[40:121, 20:81] = True
        rejected_mask = np.zeros(depth.shape, bool)
        rejected_mask[30:111, 130:171] = True
        segmenter = SimpleNamespace(segment=Mock(side_effect=[
            SegmentedTarget(rejected_mask, grounding.placement.rejected_container.bbox_xyxy,
                            int(rejected_mask.sum())),
            SegmentedTarget(mask, grounding.bbox_xyxy, int(mask.sum())),
        ]), settings=SimpleNamespace(checkpoint=Path("sam-test.pt")))
        backend = SimpleNamespace(infer_frame=Mock(return_value=SimpleNamespace(
            best=object(), grasps=[object()],
        )))
        placement_backend = SimpleNamespace(analyze_placement_geometry=Mock())
        pipeline = VLMGraspPipeline(
            {}, SimpleNamespace(ground=Mock(return_value=grounding),
                                settings=SimpleNamespace(model="test-model")), segmenter=segmenter,
            net=object(), grasp_backend=backend, placement_backend=placement_backend,
        )
        with redirect_stdout(io.StringIO()):
            result = pipeline.process("把香蕉放进侧翻的盒子", color, depth, K)
        self.assertEqual(segmenter.segment.call_count, 2)
        self.assertEqual(segmenter.segment.call_args_list[0].args[1],
                         grounding.placement.rejected_container.bbox_xyxy)
        self.assertEqual(segmenter.segment.call_args_list[1].args[1], grounding.bbox_xyxy)
        backend.infer_frame.assert_called_once()
        self.assertIs(backend.infer_frame.call_args.kwargs["target_mask"], mask)
        placement_backend.analyze_placement_geometry.assert_not_called()
        self.assertIsNone(result.placement_segmentation)
        self.assertIsNone(result.placement_analysis)
        self.assertIs(result.rejected_container_segmentation.mask, rejected_mask)
        processor = GraspPreviewProcessor(pipeline, show_open3d=False)
        combined = processor.combined_mask(color.shape[:2], result.segmentation,
                                           result.placement_segmentation,
                                           result.rejected_container_segmentation)
        np.testing.assert_array_equal(combined, (mask | rejected_mask).astype(np.uint8) * 255)
        with patch.object(processor, "_display_mask") as display_mask:
            processor.show_result(color, result)
        np.testing.assert_array_equal(display_mask.call_args.args[0], combined)
        status = processor.status(result)
        self.assertIn("GRASP ONLY", status)
        self.assertIn("INVALID CONTAINER POSE", status)
        annotated = processor.draw(color, result)
        np.testing.assert_array_equal(annotated[70, 20], [0, 255, 0])
        np.testing.assert_array_equal(annotated[70, 120], [0, 0, 255])
        np.testing.assert_array_equal(annotated[70, 150], [0, 0, 255])
        self.assertTrue(annotated[-40:, :].any())
        from utils import placement_geometry
        pipeline.placement_backend = placement_geometry
        with tempfile.TemporaryDirectory() as temp:
            directory = processor.save(Path(temp), color, replace(result, grasp_result=None))
            for name, expected in (("rejected_container_mask.png", rejected_mask),
                                   ("combined_mask.png", mask | rejected_mask)):
                actual = cv2.imread(str(directory / name), cv2.IMREAD_GRAYSCALE)
                np.testing.assert_array_equal(actual, expected.astype(np.uint8) * 255)
            self.assertFalse((directory / "placement_mask.png").exists())
            metadata = json.loads((directory / "pipeline.json").read_text())
            self.assertEqual(metadata["rejected_container_sam_area_px"], int(rejected_mask.sum()))

    def test_rejected_container_segmentation_failure_preserves_grasp(self):
        grounding = parse_task(task_payload(
            required=False, found=False, target_name="", bbox_xyxy=None, relation=None,
            requested_name="", selection_mode="grasp_only", rejected_container={
                "target_name": "纸盒", "bbox_xyxy": [60, 10, 90, 60], "reason": "盒子倒扣",
            },
        ))
        color = np.zeros((160, 200, 3), np.uint8)
        depth = np.full((160, 200), 500, np.uint16)
        mask = np.zeros(depth.shape, bool)
        mask[40:121, 20:81] = True
        segmenter = SimpleNamespace(segment=Mock(side_effect=[
            ValueError("无有效掩码"), SegmentedTarget(mask, grounding.bbox_xyxy, int(mask.sum())),
        ]))
        backend = SimpleNamespace(infer_frame=Mock(return_value=SimpleNamespace(best=None, grasps=[])))
        pipeline = VLMGraspPipeline({}, SimpleNamespace(ground=Mock(return_value=grounding)),
                                    segmenter=segmenter, net=object(), grasp_backend=backend)
        with redirect_stdout(io.StringIO()):
            result = pipeline.process("抓香蕉", color, depth, np.eye(3))
        self.assertIsNotNone(result.segmentation)
        self.assertIsNone(result.rejected_container_segmentation)
        self.assertIn("无有效掩码", result.rejected_container_error)
        backend.infer_frame.assert_called_once()

    def test_rejected_and_valid_containers_are_both_masked_but_only_valid_is_planned(self):
        grounding = parse_task(task_payload(
            selection_mode="fallback", requested_name="纸盒",
            rejected_container={"target_name": "纸盒", "bbox_xyxy": [42, 10, 55, 60],
                                "reason": "纸盒封闭"},
        ))
        color = np.zeros((160, 200, 3), np.uint8)
        depth = np.full((160, 200), 500, np.uint16)
        masks = []
        for y1, y2, x1, x2 in ((30, 100, 90, 105), (40, 120, 20, 80), (20, 120, 120, 180)):
            mask = np.zeros(depth.shape, bool)
            mask[y1:y2, x1:x2] = True
            masks.append(mask)
        rejected_box = grounding.placement.rejected_container.bbox_xyxy
        segmenter = SimpleNamespace(segment=Mock(side_effect=[
            SegmentedTarget(masks[0], rejected_box, int(masks[0].sum())),
            SegmentedTarget(masks[1], grounding.bbox_xyxy, int(masks[1].sum())),
            SegmentedTarget(masks[2], grounding.placement.bbox_xyxy, int(masks[2].sum())),
        ]))
        backend = SimpleNamespace(infer_frame=Mock(return_value=SimpleNamespace(best=None, grasps=[])))
        placement_backend = SimpleNamespace(analyze_placement_geometry=Mock(return_value=None))
        pipeline = VLMGraspPipeline(
            {}, SimpleNamespace(ground=Mock(return_value=grounding)), segmenter=segmenter,
            net=object(), grasp_backend=backend, placement_backend=placement_backend,
        )
        with redirect_stdout(io.StringIO()):
            result = pipeline.process("把香蕉放到盘子里", color, depth, np.eye(3))
        self.assertEqual(segmenter.segment.call_count, 3)
        self.assertIs(backend.infer_frame.call_args.kwargs["target_mask"], masks[1])
        geometry_args = placement_backend.analyze_placement_geometry.call_args.args
        self.assertIs(geometry_args[0], masks[1])
        self.assertIs(geometry_args[1], masks[2])
        combined = GraspPreviewProcessor.combined_mask(
            depth.shape, result.segmentation, result.placement_segmentation,
            result.rejected_container_segmentation,
        )
        np.testing.assert_array_equal(combined, np.logical_or.reduce(masks).astype(np.uint8) * 255)

    def test_rejected_container_mask_is_shown_without_pick_target(self):
        payload = task_payload(
            required=False, found=False, target_name="", bbox_xyxy=None, relation=None,
            requested_name="", selection_mode="grasp_only",
            rejected_container={"target_name": "纸盒", "bbox_xyxy": [60, 10, 90, 60],
                                "reason": "盒子倒扣"},
        )
        payload.update(found=False, target_name="", bbox_xyxy=None)
        grounding = parse_task(payload)
        color = np.zeros((160, 200, 3), np.uint8)
        depth = np.full((160, 200), 500, np.uint16)
        rejected_mask = np.zeros(depth.shape, bool)
        rejected_mask[30:110, 130:170] = True
        segmenter = SimpleNamespace(segment=Mock(return_value=SegmentedTarget(
            rejected_mask, grounding.placement.rejected_container.bbox_xyxy,
            int(rejected_mask.sum()),
        )))
        backend = SimpleNamespace(infer_frame=Mock())
        pipeline = VLMGraspPipeline({}, SimpleNamespace(ground=Mock(return_value=grounding)),
                                    segmenter=segmenter, net=object(), grasp_backend=backend)
        with redirect_stdout(io.StringIO()):
            result = pipeline.process("抓香蕉", color, depth, np.eye(3))
        self.assertIsNone(result.segmentation)
        backend.infer_frame.assert_not_called()
        processor = GraspPreviewProcessor(pipeline, show_open3d=False)
        with patch.object(processor, "_display_mask") as display_mask:
            processor.show_result(color, result)
        np.testing.assert_array_equal(display_mask.call_args.args[0], rejected_mask.astype(np.uint8) * 255)


if __name__ == "__main__":
    unittest.main()
