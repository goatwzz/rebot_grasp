"""Hardware-free pick/place checks; run with python -m unittest discover -s tests."""

from __future__ import annotations

import argparse
import ast
from contextlib import redirect_stdout
import io
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from utils import place_utils
from utils.camera_utils import compose_cam_to_base_transform


ROOT = Path(__file__).resolve().parents[1]


def load_functions(path, names, namespace):
    """Load orchestration without importing CUDA extensions or the motor SDK."""
    tree = ast.parse((ROOT / path).read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace


def box(name="box", conf=0.9, valid=True):
    return SimpleNamespace(
        class_name=name, conf=conf, is_valid=valid,
        position=np.array([0.02, 0.03, 0.4]), center_px=(30, 40),
    )


class PlaceSequenceTests(unittest.TestCase):
    def setUp(self):
        self.controller = SimpleNamespace(move_to_traj=Mock(return_value=True))
        self.driver = SimpleNamespace(
            open_gripper=Mock(), open_gripper_for_pick=Mock(), grasp=Mock(return_value=True),
            _state_lock=threading.Lock(), _STATE_POSITION="position",
        )
        self.grasp = (0.3, 0., 0.1, 0., 1.2, 0.)
        self.pre = (0.3, 0., 0.18, 0., 1.2, 0.)
        self.center = np.array([0.25, 0.1, 0.08])
        self.ready = {"x": 0.3, "y": 0., "z": 0.3, "pitch": 0.7, "duration": 3.}

    def run_sequence(self, dry_run=False):
        with patch.object(place_utils.time, "sleep"), redirect_stdout(io.StringIO()):
            return place_utils.execute_place_sequence(
                self.controller, self.driver, self.grasp, self.pre, self.center, self.ready, dry_run,
            )

    def test_original_release_coordinates_and_box_selection(self):
        chosen = box("cardboard box", 0.8)
        self.assertIs(place_utils.select_box_grasp([
            box("yellow banana", 0.99), box(conf=0.7), box(conf=0.99, valid=False), chosen,
        ]), chosen)
        T = np.array([[0., -1., 0., .3], [1., 0., 0., -.1], [0., 0., 1., .2], [0., 0., 0., 1.]])
        transformed = T @ np.append(chosen.position, 1.)
        np.testing.assert_allclose(
            place_utils.box_center_to_base(chosen, T), [transformed[0], transformed[1], 0.08],
        )
        self.assertEqual(place_utils.PLACE_RPY, (0., 1.2, 0.))
        self.assertIsNone(place_utils.select_box_grasp([]))

    def test_success_has_original_set_sequence(self):
        events = Mock()
        events.attach_mock(self.controller.move_to_traj, "move")
        events.attach_mock(self.driver.open_gripper, "open")
        events.attach_mock(self.driver.open_gripper_for_pick, "pick_open")
        events.attach_mock(self.driver.grasp, "close")
        self.assertTrue(self.run_sequence())
        self.assertEqual([c[0] for c in events.mock_calls], [
            "pick_open", "move", "move", "close", "move", "move", "open", "move",
        ])
        calls = self.controller.move_to_traj.call_args_list
        np.testing.assert_allclose(calls[3].args, [0.25, 0.1, 0.08, 0., 1.2, 0.])
        self.assertEqual(calls[3].kwargs, {"duration": 2.0})
        self.assertEqual(self.driver.open_gripper.call_args_list[0].kwargs, {"timeout": 0.5})
        self.assertEqual(self.driver._target_pos, 0.)
        self.assertEqual(self.driver._state, "position")

    def test_empty_grasp_never_moves_to_box_or_releases(self):
        self.driver.grasp.return_value = False
        self.assertFalse(self.run_sequence())
        self.assertEqual(self.controller.move_to_traj.call_count, 2)
        self.assertEqual(self.driver.open_gripper_for_pick.call_count, 1)

    def test_failed_move_never_releases_object(self):
        for failed_move in range(4):
            with self.subTest(failed_move=failed_move):
                self.controller.move_to_traj.reset_mock()
                self.controller.move_to_traj.side_effect = [True] * failed_move + [False]
                self.driver.open_gripper.reset_mock()
                self.driver.open_gripper_for_pick.reset_mock()
                self.assertFalse(self.run_sequence())
                self.assertEqual(self.driver.open_gripper_for_pick.call_count, 1)

    def test_motion_timeout_stops_before_closing(self):
        self.controller._send_thread = SimpleNamespace(is_alive=lambda: True, join=Mock())
        self.assertFalse(self.run_sequence())
        self.driver.grasp.assert_not_called()

    def test_dry_run_has_no_pick_or_place_commands(self):
        self.assertTrue(self.run_sequence(dry_run=True))
        self.controller.move_to_traj.assert_not_called()
        self.driver.open_gripper.assert_not_called()
        self.driver.open_gripper_for_pick.assert_not_called()
        self.driver.grasp.assert_not_called()


class GraspWorkflowTests(unittest.TestCase):
    def make_namespace(self, *, place=True, box_visible=True, place_reachable=True):
        ns = load_functions("scripts/grasp.py", {"parse_args", "main"}, {
            "argparse": argparse, "Path": Path, "PROJECT_ROOT": ROOT,
            "GRASPNET_ROOT": ROOT / "sdk/graspnet-baseline", "np": np, "time": time,
        })
        with patch.object(sys, "argv", ["demo", "--no-open3d", "--dry-run"]):
            args = ns["parse_args"](place_in_box=place)
        ns["parse_args"] = Mock(return_value=args)
        color = np.zeros((80, 100, 3), dtype=np.uint8)
        depth = np.full((80, 100), 400, dtype=np.uint16)
        camera = SimpleNamespace(open=Mock(), warm_up=Mock(), close=Mock(), K=np.eye(3),
                                 get_frame=Mock(return_value=(color, depth)))
        cfg = {"camera": {"type": "realsense_d435i"}, "robot": {},
               "grasp_pipeline": {"grasp": {"depth_quantile": 0.5}}}
        detection = box()
        detections = [detection] if box_visible else []
        result = SimpleNamespace(status="inferred", target_status="banana", detections=detections,
                                 selected_target=object(), best=object(), grasps=object())
        gu = SimpleNamespace(
            build_net=Mock(), infer_frame=Mock(return_value=result),
            select_target=Mock(), target_status_text=Mock(return_value="banana"),
            draw_status=Mock(side_effect=lambda image, *a, **k: image.copy()),
            draw_detections_overlay=Mock(side_effect=lambda image, *a, **k: image.copy()),
            visualization_grasps=Mock(return_value=[]), draw_best_grasp_projection=Mock(),
        )
        controller = SimpleNamespace(_running=True, end=Mock())
        driver = SimpleNamespace(start=Mock(), get_tcp_pose=Mock(return_value=np.eye(4)),
                                 release_gripper=Mock(), home_gripper_if_needed=Mock())
        checker = SimpleNamespace(check=Mock(return_value=(place_reachable, 0.)))
        pose = (0.3, 0., 0.1, 0., 1.2, 0.)
        cv = SimpleNamespace(
            WINDOW_NORMAL=0, WND_PROP_VISIBLE=0, MARKER_CROSS=0, namedWindow=Mock(), resizeWindow=Mock(),
            imshow=Mock(), waitKey=Mock(side_effect=[ord("g"), ord("q")]),
            getWindowProperty=Mock(return_value=1), destroyAllWindows=Mock(), drawMarker=Mock(),
        )
        ns.update(
            cv2=cv, configure_camera=Mock(side_effect=lambda config, args: config),
            load_config=Mock(return_value=cfg), make_camera=Mock(return_value=camera),
            load_hand_eye=Mock(return_value=(np.eye(4), "eye_in_hand")),
            load_yolo_from_config=Mock(return_value=(object(), {"infer_every": 3})),
            graspnet_utils=gu, selected_arm_config=Mock(return_value=SimpleNamespace(controller_mode="mit")),
            RebotArm=Mock(), RebotArmEndPose=Mock(return_value=controller),
            GraspDriver=Mock(return_value=driver), IkChecker=Mock(return_value=checker),
            _move_ready=Mock(), detect_objects=Mock(return_value=([object()], detections)),
            estimate_grasps=Mock(return_value=[detection] if box_visible else []),
            select_box_grasp=place_utils.select_box_grasp, box_center_to_base=place_utils.box_center_to_base,
            PLACE_RPY=place_utils.PLACE_RPY, compose_cam_to_base_transform=compose_cam_to_base_transform,
            _select_executable_grasp=Mock(return_value=(result.best, pose, pose, pose)),
            _print_grasp=Mock(), _execute_grasp=Mock(), execute_place_sequence=Mock(return_value=True),
            GRIPPER_MAX_DISTANCE_M=0.09,
        )
        return ns, driver, checker

    def test_grasp_only_uses_original_grasp_executor(self):
        ns, driver, _ = self.make_namespace(place=False)
        with redirect_stdout(io.StringIO()):
            self.assertEqual(ns["main"](), 0)
        ns["_execute_grasp"].assert_called_once()
        ns["execute_place_sequence"].assert_not_called()
        ns["estimate_grasps"].assert_not_called()
        driver.release_gripper.assert_not_called()
        driver.home_gripper_if_needed.assert_called_once()
        ns["RebotArmEndPose"].return_value.end.assert_called_once()

    def test_place_uses_snapshot_detections_and_original_box_executor(self):
        ns, driver, _ = self.make_namespace()
        with redirect_stdout(io.StringIO()):
            self.assertEqual(ns["main"](place_in_box=True), 0)
        ns["_execute_grasp"].assert_not_called()
        ns["execute_place_sequence"].assert_called_once()
        call = ns["execute_place_sequence"].call_args
        self.assertEqual(call.args[4][2], 0.08)
        self.assertTrue(call.kwargs["dry_run"])
        driver.get_tcp_pose.assert_called_once()
        driver.release_gripper.assert_not_called()
        driver.home_gripper_if_needed.assert_called_once()
        ns["RebotArmEndPose"].return_value.end.assert_called_once()
        self.assertEqual(ns["graspnet_utils"].infer_frame.call_args.kwargs["target_class"], "yellow banana")
        self.assertIs(ns["graspnet_utils"].infer_frame.call_args.kwargs["yolo_detections"],
                      ns["detect_objects"].return_value[1])

    def test_no_box_skips_graspnet_and_execution(self):
        ns, _, _ = self.make_namespace(box_visible=False)
        with redirect_stdout(io.StringIO()):
            ns["main"](place_in_box=True)
        ns["graspnet_utils"].infer_frame.assert_not_called()
        ns["execute_place_sequence"].assert_not_called()

    def test_unreachable_box_skips_grasp_execution(self):
        ns, _, checker = self.make_namespace(place_reachable=False)
        with redirect_stdout(io.StringIO()):
            ns["main"](place_in_box=True)
        checker.check.assert_called_once()
        ns["_select_executable_grasp"].assert_not_called()
        ns["execute_place_sequence"].assert_not_called()

    def test_place_rejects_no_yolo_before_startup(self):
        ns = load_functions("scripts/grasp.py", {"parse_args"}, {
            "argparse": argparse, "PROJECT_ROOT": ROOT, "GRASPNET_ROOT": ROOT,
        })
        with patch.object(sys, "argv", ["demo", "--no-yolo"]), redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as result, patch("sys.stderr", io.StringIO()):
                ns["parse_args"](place_in_box=True)
        self.assertEqual(result.exception.code, 2)


class DetectionReuseTests(unittest.TestCase):
    def test_supplied_empty_detections_do_not_trigger_another_yolo_pass(self):
        ns = load_functions("utils/graspnet_utils.py", {"infer_frame"}, {
            "__package__": "utils", "DEFAULT_VOXEL_SIZE": 0.01, "np": np,
            "select_target": Mock(return_value=None),
            "target_status_text": Mock(return_value="no target"),
            "GraspGroup": list, "GraspNetFrameResult": SimpleNamespace,
            "o3d": SimpleNamespace(geometry=SimpleNamespace(PointCloud=object)),
        })
        for supplied in ([], None):
            with self.subTest(supplied=supplied), patch("utils.yolo_utils.detect_objects") as detect:
                detect.return_value = ([], [])
                result = ns["infer_frame"](
                    None, np.zeros((2, 2, 3)), np.zeros((2, 2)), np.eye(3),
                    num_point=10, min_depth=0.05, max_depth=1.0, collision_thresh=0.01,
                    yolo_model=object(), yolo_detections=supplied,
                )
                self.assertIsNone(result.best)
                self.assertEqual(detect.call_count, 1 if supplied is None else 0)


if __name__ == "__main__":
    unittest.main()
