"""Hardware-free validation of multi-step VLM inventory decisions."""

from __future__ import annotations

import json
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from utils.vlm_grasp_pipeline import VLMGraspPipeline
from utils.vlm_sequence_utils import SequencePipeline, SequenceProgress, parse_inventory
from utils.vlm_utils import PlacementGrounding, TargetGrounding, VLMError


def inventory(objects, *, clear=False, container=None, space="pixel", inside=None, held=None, uncertain=None):
    return json.dumps({
        "coordinate_space": space,
        "container": container or {"name": "盒子", "bbox_xyxy": [40, 30, 70, 65]},
        "outside_objects": objects,
        "inside_objects": inside or [],
        "held_objects": held or [],
        "uncertain_objects": uncertain or [],
        "scene_clear": clear,
    })


def parse(content, instruction="把香蕉全部放进盒子"):
    return parse_inventory(content, instruction=instruction, image_size=(100, 80),
                           request_size=(100, 80), coordinate_space="pixel")


class InventoryTests(unittest.TestCase):
    def test_empty_outside_list_is_completion(self):
        result = parse(inventory([], clear=True))
        self.assertEqual(result.remaining, 0)
        self.assertIsNone(result.target_box)

    def test_selected_target_follows_vlm_order(self):
        near = {"name": "香蕉", "bbox_xyxy": [22, 35, 32, 45]}
        far = {"name": "香蕉", "bbox_xyxy": [2, 35, 12, 45]}
        result = parse(inventory([near, far]), "先从最外围开始把香蕉放进盒子")
        self.assertEqual(result.target_box, (22, 35, 32, 45))
        self.assertEqual(result.remaining, 2)

    def test_vlm_outside_label_is_not_overridden_by_box_overlap(self):
        result = parse(inventory([{"name": "香蕉", "bbox_xyxy": [45, 35, 55, 45]}]))
        self.assertEqual(result.target_box, (45, 35, 55, 45))

    def test_inside_objects_are_not_selected_and_uncertain_objects_pause(self):
        inside = {"name": "香蕉", "bbox_xyxy": [45, 35, 55, 45]}
        self.assertEqual(parse(inventory([], clear=True, inside=[inside])).remaining, 0)
        with self.assertRaises(VLMError):
            parse(inventory([], uncertain=[inside]))

    def test_held_object_pauses_before_next_pick(self):
        held = {"name": "香蕉", "bbox_xyxy": [45, 35, 55, 45]}
        with self.assertRaisesRegex(VLMError, "夹爪仍持有"):
            parse(inventory([{"name": "香蕉", "bbox_xyxy": [2, 35, 12, 45]}], held=[held]))

    def test_rejects_inconsistent_empty_flag(self):
        with self.assertRaises(VLMError):
            parse(inventory([], clear=False))

    def test_missing_container_cannot_confirm_completion(self):
        content = json.dumps({"coordinate_space": "pixel", "container": None,
                              "outside_objects": [], "inside_objects": [],
                              "uncertain_objects": [], "scene_clear": True})
        with self.assertRaises(VLMError):
            parse(content)

    def test_rejects_invalid_coordinate_space_and_box(self):
        target = {"name": "香蕉", "bbox_xyxy": [2, 2, 12, 12]}
        with self.assertRaises(VLMError):
            parse(inventory([target], space="normalized_1000"))
        with self.assertRaises(VLMError):
            parse(inventory([{"name": "香蕉", "bbox_xyxy": [2, 2, 101, 12]}]))


class ProgressTests(unittest.TestCase):
    def test_two_bananas_then_double_empty_confirmation(self):
        progress = SequenceProgress()
        self.assertEqual(progress.observe(2, 5), "pick")
        progress.placed += 1
        self.assertEqual(progress.observe(1, 5), "pick")
        progress.placed += 1
        self.assertEqual(progress.observe(0, 5), "confirm_empty")
        self.assertEqual(progress.observe(0, 5), "complete")

    def test_unchanged_inventory_remains_pickable(self):
        progress = SequenceProgress()
        self.assertEqual(progress.observe(2, 5), "pick")
        for _ in range(3):
            progress.placed += 1
            self.assertEqual(progress.observe(2, 5), "pick")

    def test_added_objects_do_not_stop_sequence(self):
        progress = SequenceProgress()
        self.assertEqual(progress.observe(2, 5), "pick")
        progress.placed += 1
        self.assertEqual(progress.observe(3, 5), "pick")
        progress.placed += 1
        self.assertEqual(progress.observe(4, 5), "pick")

    def test_limit_allows_final_empty_check_but_not_next_pick(self):
        progress = SequenceProgress()
        self.assertEqual(progress.observe(1, 1), "pick")
        progress.placed += 1
        self.assertEqual(progress.observe(0, 1), "confirm_empty")
        self.assertEqual(progress.observe(0, 1), "complete")
        progress = SequenceProgress(placed=1)
        self.assertEqual(progress.observe(1, 1), "limit")

    def test_false_empty_recovers_without_skipping_progress_check(self):
        progress = SequenceProgress()
        self.assertEqual(progress.observe(2, 5), "pick")
        progress.placed += 1
        self.assertEqual(progress.observe(0, 5), "confirm_empty")
        self.assertEqual(progress.observe(1, 5), "pick")
        self.assertFalse(progress.empty_confirmation_pending)


class TargetLockTests(unittest.TestCase):
    def make_pipeline(self, placement_box=(40, 30, 70, 65)):
        selected = {"name": "香蕉", "bbox_xyxy": [2, 2, 12, 12]}
        grounding = TargetGrounding(
            True, "错误的香蕉", (20, 2, 30, 12), "", "指令", (100, 80), (100, 80), "pixel",
            PlacementGrounding(True, True, "盒子", placement_box, "", "in", "盒子", "explicit"),
        )
        client = SimpleNamespace(
            settings=SimpleNamespace(max_image_side=1280, coordinate_space="pixel"),
            chat=Mock(return_value=inventory([selected])),
            ground=Mock(return_value=grounding),
        )
        pipeline = object.__new__(SequencePipeline)
        pipeline.client = client
        return pipeline, client

    def test_pipeline_locks_inventory_target_not_ground_fallback(self):
        pipeline, client = self.make_pipeline()
        color = np.zeros((80, 100, 3), dtype=np.uint8)
        depth = np.full((80, 100), 500, dtype=np.uint16)
        K = np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]])

        def observe_grounding(instance, command, rgb, mm, intrinsics):
            return SimpleNamespace(grounding=instance.client.ground(command, rgb))

        with patch.object(VLMGraspPipeline, "process", observe_grounding):
            selected = pipeline.process("把香蕉放进盒子", color, depth, K)
        self.assertEqual(selected.grounding.bbox_xyxy, (2, 2, 12, 12))
        self.assertEqual(selected.grounding.target_name, "香蕉")
        self.assertEqual(pipeline.destination_name, "盒子")
        self.assertEqual(pipeline.destination_box, (40, 30, 70, 65))
        self.assertIs(pipeline.client, client)
        self.assertIn("new matching objects may have been added", client.chat.call_args.args[2])
        self.assertIn("never use bounding-box overlap alone", client.chat.call_args.args[2])
        self.assertIn("still held by it in held_objects", client.chat.call_args.args[2])

    def test_pipeline_rejects_different_destination(self):
        pipeline, _ = self.make_pipeline((72, 30, 95, 65))
        color = np.zeros((80, 100, 3), dtype=np.uint8)
        depth = np.full((80, 100), 500, dtype=np.uint16)
        K = np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]])

        def observe_grounding(instance, command, rgb, mm, intrinsics):
            return SimpleNamespace(grounding=instance.client.ground(command, rgb))

        with patch.object(VLMGraspPipeline, "process", observe_grounding):
            with self.assertRaises(VLMError):
                pipeline.process("把香蕉放进盒子", color, depth, K)

    def test_recheck_rejects_fallback_to_another_container(self):
        pipeline, _ = self.make_pipeline()
        pipeline.destination_name = "盒子"
        pipeline.destination_box = (40, 30, 70, 65)
        placement = PlacementGrounding(True, True, "盘子", (40, 30, 70, 65), "", "in", "盒子", "fallback")
        result = SimpleNamespace(grounding=SimpleNamespace(placement=placement))
        with patch.object(VLMGraspPipeline, "recheck_placement", return_value=result):
            with self.assertRaises(VLMError):
                pipeline.recheck_placement("把香蕉放进盒子", None, None, None,
                                           object_points_camera=None, gripper_pose_camera=None)

    def test_recheck_accepts_same_box_with_different_description(self):
        pipeline, _ = self.make_pipeline()
        pipeline.destination_name = "右侧敞口纸盒"
        pipeline.destination_box = (40, 30, 70, 65)
        placement = PlacementGrounding(True, True, "敞口纸盒", (41, 31, 71, 66), "", "in", "盒子", "explicit")
        result = SimpleNamespace(grounding=SimpleNamespace(placement=placement))
        with patch.object(VLMGraspPipeline, "recheck_placement", return_value=result):
            self.assertIs(
                pipeline.recheck_placement("把香蕉放进盒子", None, None, None,
                                           object_points_camera=None, gripper_pose_camera=None),
                result,
            )

    def test_recheck_rejects_other_box_even_with_same_name(self):
        pipeline, _ = self.make_pipeline()
        pipeline.destination_name = "敞口纸盒"
        pipeline.destination_box = (40, 30, 70, 65)
        placement = PlacementGrounding(True, True, "敞口纸盒", (72, 30, 95, 65), "", "in", "盒子", "explicit")
        result = SimpleNamespace(grounding=SimpleNamespace(placement=placement))
        with patch.object(VLMGraspPipeline, "recheck_placement", return_value=result):
            with self.assertRaises(VLMError):
                pipeline.recheck_placement("把香蕉放进盒子", None, None, None,
                                           object_points_camera=None, gripper_pose_camera=None)


if __name__ == "__main__":
    unittest.main()
