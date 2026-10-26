"""Hardware-free checks for the VLM real-robot safety path."""

from __future__ import annotations

import ast
from contextlib import redirect_stdout
import io
from pathlib import Path
import sys
import threading
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import yaml

from drivers.robot.grasp_driver import GraspDriver, SelectedArmConfig
from utils.placement_geometry import PlacementPreviewSettings, estimate_table_normal
from utils.sam_utils import SegmentedTarget
from utils.transforms import canonicalize_parallel_gripper_tcp_rotation, graspnet_rotation_to_rebot_tcp_rotation
from utils.vlm_grasp_pipeline import VLMGraspPipeline
from utils.vlm_robot_utils import (
    VLMRobotSettings,
    current_gripper_pose_camera,
    execute_pick_and_lift,
    execute_place,
    placement_is_executable,
    placement_poses_to_base,
    transport_carried_object_points,
)
from utils.vlm_utils import PlacementGrounding, TargetGrounding


ROOT = Path(__file__).resolve().parents[1]


class FakeGrasps:
    def __init__(self, rotations):
        self.rotation_matrices = np.asarray(rotations, dtype=np.float64).reshape(-1, 3, 3)

    def __len__(self):
        return len(self.rotation_matrices)

    def __getitem__(self, keep):
        return FakeGrasps(self.rotation_matrices[keep])


def direction_functions():
    tree = ast.parse((ROOT / "utils/graspnet_utils.py").read_text())
    names = {
        "approach_tilt_angles_deg", "filter_grasps_by_approach_tilt",
        "ready_rotation_angles_deg", "filter_grasps_by_ready_rotation",
    }
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes],
        type_ignores=[],
    )
    namespace = {
        "np": np,
        "canonicalize_parallel_gripper_tcp_rotation": canonicalize_parallel_gripper_tcp_rotation,
        "graspnet_rotation_to_rebot_tcp_rotation": graspnet_rotation_to_rebot_tcp_rotation,
    }
    exec(compile(ast.fix_missing_locations(module), "graspnet_utils.py", "exec"), namespace)
    return namespace


def prepare_pick_function(namespace):
    tree = ast.parse((ROOT / "scripts/vlm_grasp_robot.py").read_text())
    node = next(
        item for item in tree.body
        if isinstance(item, ast.FunctionDef) and item.name == "prepare_pick"
    )
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), "vlm_grasp_robot.py", "exec"), namespace)
    return namespace["prepare_pick"]


def capture_fresh_function():
    tree = ast.parse((ROOT / "scripts/vlm_grasp_robot.py").read_text())
    node = next(item for item in tree.body if isinstance(item, ast.FunctionDef) and item.name == "_capture_fresh")
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node],
        type_ignores=[],
    )
    namespace = {}
    exec(compile(ast.fix_missing_locations(module), "vlm_grasp_robot.py", "exec"), namespace)
    return namespace["_capture_fresh"]


def completion_home_function(move_ready):
    tree = ast.parse((ROOT / "scripts/vlm_grasp_robot.py").read_text())
    node = next(
        item for item in tree.body
        if isinstance(item, ast.FunctionDef) and item.name == "_return_ready_after_sequence"
    )
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node],
        type_ignores=[],
    )
    namespace = {"move_ready": move_ready}
    exec(compile(ast.fix_missing_locations(module), "vlm_grasp_robot.py", "exec"), namespace)
    return namespace["_return_ready_after_sequence"]


class SequenceCompletionTests(unittest.TestCase):
    def setUp(self):
        self.controller = object()
        self.ready_cfg = {"x": 0.25, "duration": 3.0}
        self.move_ready = Mock(return_value=True)
        self.finish = completion_home_function(self.move_ready)

    def test_completed_task_returns_ready_once(self):
        with redirect_stdout(io.StringIO()):
            self.assertTrue(self.finish(
                self.controller, self.ready_cfg,
                holding_object=False, verification_pending=False, dry_run=False,
            ))
        self.move_ready.assert_called_once_with(self.controller, self.ready_cfg)

    def test_holding_or_unverified_release_prevents_motion(self):
        for holding, pending in ((True, False), (False, True)):
            with self.subTest(holding=holding, pending=pending), redirect_stdout(io.StringIO()):
                self.assertFalse(self.finish(
                    self.controller, self.ready_cfg,
                    holding_object=holding, verification_pending=pending, dry_run=False,
                ))
        self.move_ready.assert_not_called()

    def test_dry_run_does_not_move(self):
        with redirect_stdout(io.StringIO()):
            self.assertTrue(self.finish(
                self.controller, self.ready_cfg,
                holding_object=False, verification_pending=False, dry_run=True,
            ))
        self.move_ready.assert_not_called()

    def test_failed_ready_motion_is_not_reported_as_success(self):
        self.move_ready.return_value = False
        with redirect_stdout(io.StringIO()):
            self.assertFalse(self.finish(
                self.controller, self.ready_cfg,
                holding_object=False, verification_pending=False, dry_run=False,
            ))

    def test_ready_motion_exception_is_not_reported_as_success(self):
        self.move_ready.side_effect = RuntimeError("controller stopped")
        with redirect_stdout(io.StringIO()):
            self.assertFalse(self.finish(
                self.controller, self.ready_cfg,
                holding_object=False, verification_pending=False, dry_run=False,
            ))


def rotation_with_tilt(degrees):
    angle = np.deg2rad(90.0 - degrees)
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


class DirectionSafetyTests(unittest.TestCase):
    def test_side_grasp_is_rejected_but_slanted_top_grasp_remains(self):
        functions = direction_functions()
        grasps = FakeGrasps([rotation_with_tilt(77.0), rotation_with_tilt(35.0)])
        filtered, angles = functions["filter_grasps_by_approach_tilt"](
            grasps, np.array([0.0, 0.0, 1.0]), 60.0,
        )
        np.testing.assert_allclose(angles, [77.0, 35.0], atol=1e-7)
        self.assertEqual(len(filtered), 1)
        np.testing.assert_allclose(filtered.rotation_matrices[0], grasps.rotation_matrices[1])

    def test_reverse_approach_is_rejected_even_when_tilt_matches(self):
        functions = direction_functions()
        grasps = FakeGrasps([rotation_with_tilt(35.0), rotation_with_tilt(145.0)])
        filtered, angles = functions["filter_grasps_by_approach_tilt"](
            grasps, np.array([0.0, 0.0, 1.0]), 50.0,
        )
        np.testing.assert_allclose(angles, [35.0, 145.0], atol=1e-7)
        self.assertEqual(len(filtered), 1)

    def test_large_ready_relative_rotation_is_rejected(self):
        functions = direction_functions()
        small = np.deg2rad(30.0)
        large = np.deg2rad(140.0)
        grasps = FakeGrasps([
            [[np.cos(small), -np.sin(small), 0], [np.sin(small), np.cos(small), 0], [0, 0, 1]],
            [[np.cos(large), -np.sin(large), 0], [np.sin(large), np.cos(large), 0], [0, 0, 1]],
        ])
        filtered, angles = functions["filter_grasps_by_ready_rotation"](
            grasps, np.eye(4), np.eye(3), 90.0,
        )
        np.testing.assert_allclose(angles, [30.0, 140.0], atol=1e-7)
        self.assertEqual(len(filtered), 1)

    def test_missing_table_plane_is_not_treated_as_a_valid_normal(self):
        depth = np.full((20, 30), 500, dtype=np.uint16)
        excluded = np.ones(depth.shape, dtype=bool)
        normal = estimate_table_normal(
            depth,
            np.array([[100.0, 0.0, 15.0], [0.0, 100.0, 10.0], [0.0, 0.0, 1.0]]),
            exclusion_masks=[excluded],
            settings=PlacementPreviewSettings(min_points=16),
        )
        self.assertIsNone(normal)

    def test_robot_settings_reject_ninety_degree_limit(self):
        self.assertEqual(VLMRobotSettings.from_config({}).max_approach_tilt_deg, 40.0)
        with self.assertRaises(ValueError):
            VLMRobotSettings.from_config({"vlm_robot": {"max_approach_tilt_deg": 90}})
        with self.assertRaises(ValueError):
            VLMRobotSettings.from_config({"vlm_robot": {"max_ready_rotation_deg": 181}})
        with self.assertRaises(ValueError):
            VLMRobotSettings.from_config({"vlm_robot": {"banana_end_fraction": 0.5}})
        with self.assertRaises(ValueError):
            VLMRobotSettings.from_config({"vlm_robot": {"banana_min_mask_clearance_m": -0.001}})

    def test_fresh_capture_does_not_reuse_one_old_frame(self):
        frame = (np.zeros((4, 5, 3), np.uint8), np.ones((4, 5), np.uint16))
        camera = SimpleNamespace(get_frame=Mock(side_effect=[frame, (None, None), (None, None), (None, None)]))
        with self.assertRaises(RuntimeError):
            capture_fresh_function()(camera)

    def test_fresh_capture_requires_consecutive_valid_frames(self):
        frame = (np.zeros((4, 5, 3), np.uint8), np.ones((4, 5), np.uint16))
        camera = SimpleNamespace(get_frame=Mock(side_effect=[frame, (None, None), (None, None), frame]))
        with self.assertRaises(RuntimeError):
            capture_fresh_function()(camera)


class CarriedGeometryTests(unittest.TestCase):
    def test_points_follow_tcp_and_are_reexpressed_in_new_camera(self):
        points = np.array([[0.1, -0.2, 0.4], [0.2, 0.0, 0.5]])
        lifted = np.eye(4)
        lifted[:3, 3] = [0.3, -0.1, 0.2]
        # Camera and gripper undergo the same rigid translation, so the held
        # object's coordinates in the eye-in-hand camera stay unchanged.
        actual = transport_carried_object_points(
            points, np.eye(4), (0.0, 0.0, 0.0, 0.0, 0.0, 0.0), lifted, lifted,
        )
        np.testing.assert_allclose(actual, points)
        pose = current_gripper_pose_camera(lifted, lifted)
        np.testing.assert_allclose(pose.translation, np.zeros(3))
        np.testing.assert_allclose(pose.rotation_matrix, np.eye(3))

    def test_release_and_preplace_offsets_follow_container_normal(self):
        plan = SimpleNamespace(
            fits=True,
            gripper_translation_camera_m=np.array([0.1, 0.2, 0.5]),
            gripper_rotation_camera=np.eye(3),
        )
        analysis = SimpleNamespace(
            plan=plan,
            container=SimpleNamespace(normal_camera=np.array([0.0, 0.0, -1.0])),
        )
        settings = VLMRobotSettings(place_release_clearance_m=0.08, place_approach_offset_m=0.06)
        release, preplace = placement_poses_to_base(analysis, np.eye(4), settings)
        np.testing.assert_allclose(release[:3], [0.1, 0.2, 0.42])
        np.testing.assert_allclose(preplace[:3], [0.1, 0.2, 0.36])


class MotionOutcomeTests(unittest.TestCase):
    def test_rs_holding_torque_increases_without_raising_closing_limit(self):
        config = yaml.safe_load((ROOT / "config/default.yaml").read_text(encoding="utf-8"))
        gripper = config["robot"]["gripper"]["rs"]
        self.assertEqual(gripper["default_force"], 0.40)
        self.assertEqual(gripper["close_torque"], gripper["tau_max"])

    def test_rs_opening_is_capped_below_hardware_limit(self):
        config = yaml.safe_load((ROOT / "config/default.yaml").read_text(encoding="utf-8"))
        gripper_group = SimpleNamespace(_jcfgs=[SimpleNamespace(name="gripper")], send_mit=Mock())
        arm = SimpleNamespace(
            groups={"arm": SimpleNamespace(num_joints=6), "gripper": gripper_group},
            has_gripper=True,
        )
        fake_kinematics = ModuleType("reBotArm_control_py.kinematics")
        fake_kinematics.compute_fk = Mock()
        fake_kinematics.load_robot_model = Mock(return_value=object())
        fake_kinematics.pad_q_for_model = Mock()
        with patch.dict(sys.modules, {"reBotArm_control_py.kinematics": fake_kinematics}), \
             patch("drivers.robot.grasp_driver.selected_arm_config", return_value=SelectedArmConfig("rs", "mit")):
            driver = GraspDriver(arm, object(), gripper_config=config["robot"]["gripper"])
            capped_driver = GraspDriver(arm, object())
            with self.assertRaises(ValueError):
                GraspDriver(arm, object(), gripper_config={"rs": {"max_open_angle_rad": 4.8}})
        self.assertAlmostEqual(driver._open_hi, 0.98 * 4.71)
        driver._send_gripper_mit(5.0)
        self.assertAlmostEqual(gripper_group.send_mit.call_args.args[0][0], 0.98 * 4.71)
        self.assertAlmostEqual(capped_driver._open_hi, 4.71)
        capped_driver._send_gripper_mit(10.0)
        self.assertAlmostEqual(gripper_group.send_mit.call_args.args[0][0], 4.71)

    def test_already_open_gripper_does_not_receive_another_open_command(self):
        driver = GraspDriver.__new__(GraspDriver)
        driver._angle_open = 5.0
        driver._open_soft_limit = 4.51
        driver._open_sign = 1.0
        driver._open_lo = 0.0
        driver._open_hi = 4.51
        driver._arrive_tol = 0.12
        driver._state_lock = threading.Lock()
        driver._state = driver._STATE_IDLE
        driver._ensure_running = Mock()
        driver.get_gripper_state = Mock(return_value=(4.45, 0.0, 0.0))
        driver._set_position_target = Mock()
        self.assertIsNone(driver.open_gripper())
        driver._set_position_target.assert_not_called()

    def test_pick_closes_observed_open_gripper_before_opening(self):
        driver = GraspDriver.__new__(GraspDriver)
        driver._open_sign = 1.0
        driver._open_soft_limit = 0.98 * 4.71
        driver._state_lock = threading.Lock()
        driver._state = driver._STATE_IDLE
        driver._ensure_running = Mock()
        driver.get_gripper_state = Mock(return_value=(4.382, 0.0, 0.0))
        driver._set_position_target = Mock()
        driver._wait_until = Mock(return_value=True)
        driver.open_gripper = Mock()
        driver.open_gripper_for_pick(timeout=3.0)
        driver._set_position_target.assert_called_once_with(0.0)
        driver.open_gripper.assert_called_once_with(timeout=3.0)

    def test_pick_does_not_close_when_initially_closed(self):
        driver = GraspDriver.__new__(GraspDriver)
        driver._open_sign = 1.0
        driver._open_soft_limit = 0.98 * 4.71
        driver._state_lock = threading.Lock()
        driver._state = driver._STATE_IDLE
        driver._ensure_running = Mock()
        driver.get_gripper_state = Mock(return_value=(0.2, 0.0, 0.0))
        driver._set_position_target = Mock()
        driver.open_gripper = Mock()
        driver.open_gripper_for_pick()
        driver._set_position_target.assert_not_called()
        driver.open_gripper.assert_called_once_with(timeout=3.0)

    def test_pick_does_not_open_when_preclose_fails(self):
        driver = GraspDriver.__new__(GraspDriver)
        driver._open_sign = 1.0
        driver._open_soft_limit = 0.98 * 4.71
        driver._state_lock = threading.Lock()
        driver._state = driver._STATE_IDLE
        driver._ensure_running = Mock()
        driver.get_gripper_state = Mock(return_value=(4.382, 0.0, 0.0))
        driver._set_position_target = Mock()
        driver._wait_until = Mock(return_value=False)
        driver._send_gripper_mit = Mock()
        driver.open_gripper = Mock()
        with self.assertRaisesRegex(RuntimeError, "归零未到位"):
            driver.open_gripper_for_pick()
        driver.open_gripper.assert_not_called()
        self.assertEqual(driver._state, driver._STATE_IDLE)
        driver._send_gripper_mit.assert_called_once_with(0.0, kp=0.0, kd=0.0, tau=0.0)

    def test_pick_does_not_preclose_held_object(self):
        driver = GraspDriver.__new__(GraspDriver)
        driver._open_sign = 1.0
        driver._open_soft_limit = 0.98 * 4.71
        driver._state_lock = threading.Lock()
        driver._state = driver._STATE_HOLDING
        driver._ensure_running = Mock()
        driver.get_gripper_state = Mock(return_value=(3.0, 0.0, 0.0))
        driver._set_position_target = Mock()
        driver.open_gripper = Mock()
        with self.assertRaisesRegex(RuntimeError, "持物状态"):
            driver.open_gripper_for_pick()
        driver._set_position_target.assert_not_called()
        driver.open_gripper.assert_not_called()

    def test_home_gripper_skips_when_feedback_is_at_zero(self):
        driver = GraspDriver.__new__(GraspDriver)
        driver._arrive_tol = 0.15
        driver._state_lock = threading.Lock()
        driver._state = driver._STATE_IDLE
        driver._ensure_running = Mock()
        driver.get_gripper_state = Mock(return_value=(0.04, 0.0, 0.0))
        driver._set_position_target = Mock()
        with redirect_stdout(io.StringIO()):
            driver.home_gripper_if_needed()
        driver._set_position_target.assert_not_called()

    def test_home_gripper_clears_stale_open_target_at_zero(self):
        driver = GraspDriver.__new__(GraspDriver)
        driver._arrive_tol = 0.15
        driver._state_lock = threading.Lock()
        driver._state = driver._STATE_POSITION
        driver._target_pos = 4.616
        driver._position_reached = False
        driver._ensure_running = Mock()
        driver.get_gripper_state = Mock(return_value=(0.04, 0.0, 0.0))
        driver._set_position_target = Mock()
        with redirect_stdout(io.StringIO()):
            driver.home_gripper_if_needed()
        self.assertEqual(driver._target_pos, 0.0)
        self.assertTrue(driver._position_reached)
        driver._set_position_target.assert_not_called()

    def test_home_gripper_closes_only_after_feedback_check(self):
        driver = GraspDriver.__new__(GraspDriver)
        driver._arrive_tol = 0.15
        driver._state_lock = threading.Lock()
        driver._state = driver._STATE_IDLE
        driver._ensure_running = Mock()
        driver.get_gripper_state = Mock(side_effect=[(2.0, 0.0, 0.0), (0.03, 0.0, 0.0)])
        driver._set_position_target = Mock()
        driver._wait_until = Mock(return_value=True)
        with redirect_stdout(io.StringIO()):
            driver.home_gripper_if_needed(timeout=4.0)
        driver._set_position_target.assert_called_once_with(0.0)
        driver._wait_until.assert_called_once_with(driver._position_done, 4.0)

    def test_home_gripper_rejects_held_object(self):
        driver = GraspDriver.__new__(GraspDriver)
        driver._state_lock = threading.Lock()
        driver._state = driver._STATE_HOLDING
        driver._ensure_running = Mock()
        driver.get_gripper_state = Mock(return_value=(2.0, 0.0, 0.0))
        driver._set_position_target = Mock()
        with self.assertRaisesRegex(RuntimeError, "可能持有物体"):
            driver.home_gripper_if_needed()
        driver._set_position_target.assert_not_called()

    def test_home_gripper_timeout_stops_drive(self):
        driver = GraspDriver.__new__(GraspDriver)
        driver._arrive_tol = 0.15
        driver._state_lock = threading.Lock()
        driver._state = driver._STATE_IDLE
        driver._ensure_running = Mock()
        driver.get_gripper_state = Mock(return_value=(2.0, 0.0, 0.0))
        driver._set_position_target = Mock()
        driver._wait_until = Mock(return_value=False)
        driver._send_gripper_mit = Mock()
        with redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "归零超时"):
            driver.home_gripper_if_needed()
        driver._send_gripper_mit.assert_called_once_with(0.0, kp=0.0, kd=0.0, tau=0.0)

    def test_repeated_opening_drops_unreachable_old_target(self):
        driver = GraspDriver.__new__(GraspDriver)
        driver._angle_open = 5.0
        driver._open_soft_limit = 4.51
        driver._open_sign = 1.0
        driver._open_lo = 0.0
        driver._open_hi = 4.51
        driver._arrive_tol = 0.12
        driver._state_lock = threading.Lock()
        driver._state = driver._STATE_POSITION
        driver._target_pos = 4.51
        driver._position_reached = False
        driver._ensure_running = Mock()
        driver.get_gripper_state = Mock(return_value=(4.45, 0.0, 0.0))
        driver._set_position_target = Mock()
        driver.open_gripper()
        self.assertEqual(driver._target_pos, 4.45)
        self.assertTrue(driver._position_reached)
        driver._set_position_target.assert_not_called()

    def test_gripper_open_timeout_stops_outward_drive(self):
        driver = GraspDriver.__new__(GraspDriver)
        driver._angle_open = 5.0
        driver._open_soft_limit = 4.51
        driver._open_sign = 1.0
        driver._open_lo = 0.0
        driver._open_hi = 4.51
        driver._arrive_tol = 0.12
        driver._state_lock = threading.Lock()
        driver._state = driver._STATE_IDLE
        driver._ensure_running = Mock()
        driver.get_gripper_state = Mock(side_effect=[(4.0, 0.0, 0.0), (4.2, 0.0, 0.0)])
        driver._wait_until = Mock(return_value=False)
        driver._send_gripper_mit = Mock()
        with self.assertRaisesRegex(RuntimeError, "已停止向外驱动"):
            driver.open_gripper(timeout=5.0)
        self.assertEqual(driver._state, driver._STATE_IDLE)
        self.assertEqual(driver._target_pos, driver._open_hi)
        driver._send_gripper_mit.assert_called_once_with(4.2, kp=0.0, kd=0.0, tau=0.0)
        driver._wait_until.assert_called_once_with(driver._position_done, 5.0)

    def test_release_opening_uses_feedback_motion_not_command_target(self):
        driver = GraspDriver.__new__(GraspDriver)
        driver._open_sign = -1.0
        driver.get_gripper_state = Mock(side_effect=[(-1.2, 0.0, 0.0), (-1.8, 0.0, 0.0)])
        driver.open_gripper = Mock()
        driver._wait_until = Mock(side_effect=lambda predicate, timeout: predicate())
        with redirect_stdout(io.StringIO()):
            self.assertTrue(driver.open_gripper_for_release(timeout=1.5, min_motion_rad=0.5))
        driver.open_gripper.assert_called_once_with(timeout=1.5)

    def test_release_open_timeout_does_not_continue_placement(self):
        driver = GraspDriver.__new__(GraspDriver)
        driver.get_gripper_state = Mock(return_value=(1.2, 0.0, 0.0))
        driver.open_gripper = Mock(side_effect=RuntimeError("stalled"))
        with redirect_stdout(io.StringIO()):
            self.assertFalse(driver.open_gripper_for_release(timeout=1.5, min_motion_rad=0.5))

    def setUp(self):
        self.settings = VLMRobotSettings()
        self.pose = (0.3, 0.0, 0.2, 0.0, 1.2, 0.0)
        self.controller = SimpleNamespace(move_to_traj=Mock(return_value=True))
        self.driver = SimpleNamespace(
            open_gripper=Mock(return_value=True),
            open_gripper_for_pick=Mock(return_value=True),
            open_gripper_for_release=Mock(return_value=True),
            grasp=Mock(return_value=True),
        )

    def test_dry_run_sends_no_motion_or_gripper_command(self):
        with redirect_stdout(io.StringIO()):
            pick = execute_pick_and_lift(
                self.controller, self.driver, self.pose, self.pose, self.pose, {}, self.settings, dry_run=True,
            )
            place = execute_place(
                self.controller, self.driver, self.pose, self.pose, {}, self.settings, dry_run=True,
            )
        self.assertFalse(pick.grasped)
        self.assertFalse(place.released)
        self.controller.move_to_traj.assert_not_called()
        self.driver.open_gripper_for_pick.assert_not_called()
        self.driver.grasp.assert_not_called()

    def test_grasp_state_is_preserved_when_ready_motion_fails(self):
        with patch("utils.vlm_robot_utils.wait_motion", return_value=True), \
             patch("utils.vlm_robot_utils.move_ready", return_value=False), \
             redirect_stdout(io.StringIO()):
            result = execute_pick_and_lift(
                self.controller, self.driver, self.pose, self.pose, self.pose, {}, self.settings, dry_run=False,
            )
        self.assertTrue(result.grasped)
        self.assertFalse(result.at_ready)

    def test_empty_grasp_returns_ready_without_claiming_success(self):
        self.driver.grasp.return_value = False
        with patch("utils.vlm_robot_utils.wait_motion", return_value=True), \
             patch("utils.vlm_robot_utils.move_ready", return_value=True), \
             redirect_stdout(io.StringIO()):
            result = execute_pick_and_lift(
                self.controller, self.driver, self.pose, self.pose, self.pose, {}, self.settings, dry_run=False,
            )
        self.assertFalse(result.grasped)
        self.assertTrue(result.at_ready)
        self.assertEqual(self.controller.move_to_traj.call_count, 3)
        self.driver.open_gripper_for_release.assert_not_called()

    def test_release_state_is_preserved_when_return_ready_fails(self):
        with patch("utils.vlm_robot_utils.wait_motion", return_value=True), \
             patch("utils.vlm_robot_utils.move_ready", return_value=False), \
             redirect_stdout(io.StringIO()):
            result = execute_place(
                self.controller, self.driver, self.pose, self.pose, {}, self.settings, dry_run=False,
            )
        self.assertTrue(result.released)
        self.assertFalse(result.at_ready)
        self.driver.open_gripper_for_release.assert_called_once_with(
            timeout=self.settings.gripper_open_timeout_s,
            min_motion_rad=self.settings.release_min_open_motion_rad,
        )

    def test_unconfirmed_release_stops_before_retreat_and_next_cycle(self):
        self.driver.open_gripper_for_release.return_value = False
        with patch("utils.vlm_robot_utils.wait_motion", return_value=True), \
             patch("utils.vlm_robot_utils.move_ready", return_value=True), \
             redirect_stdout(io.StringIO()):
            result = execute_place(
                self.controller, self.driver, self.pose, self.pose, {}, self.settings,
                dry_run=False,
            )
        self.assertFalse(result.released)
        self.assertFalse(result.at_ready)
        self.assertEqual(self.controller.move_to_traj.call_count, 2)

    def test_gripper_open_timeout_stops_pick_before_arm_motion(self):
        self.driver.open_gripper_for_pick.side_effect = RuntimeError("opening stalled")
        with patch("utils.vlm_robot_utils.wait_motion", return_value=True), \
             patch("utils.vlm_robot_utils.move_ready", return_value=True), \
             redirect_stdout(io.StringIO()):
            result = execute_pick_and_lift(
                self.controller, self.driver, self.pose, self.pose, self.pose,
                {}, self.settings, dry_run=False,
            )
        self.assertEqual(result.failure_stage, "gripper_open")
        self.controller.move_to_traj.assert_not_called()

    def test_pick_uses_fixed_open_command(self):
        with patch("utils.vlm_robot_utils.wait_motion", return_value=True), \
             patch("utils.vlm_robot_utils.move_ready", return_value=True), \
             redirect_stdout(io.StringIO()):
            execute_pick_and_lift(
                self.controller, self.driver, self.pose, self.pose, self.pose,
                {}, self.settings, dry_run=False,
            )
        self.driver.open_gripper_for_pick.assert_called_once_with(timeout=self.settings.gripper_open_timeout_s)

    def test_pick_retreat_failure_preserves_holding_state(self):
        self.controller.move_to_traj.side_effect = [True, True, False]
        with patch("utils.vlm_robot_utils.wait_motion", return_value=True), redirect_stdout(io.StringIO()):
            result = execute_pick_and_lift(
                self.controller, self.driver, self.pose, self.pose, self.pose,
                {}, self.settings, dry_run=False,
            )
        self.assertTrue(result.grasped)
        self.assertFalse(result.at_ready)

    def test_place_retreat_failure_prevents_next_cycle(self):
        self.controller.move_to_traj.side_effect = [True, True, False]
        with patch("utils.vlm_robot_utils.wait_motion", return_value=True), redirect_stdout(io.StringIO()):
            result = execute_place(
                self.controller, self.driver, self.pose, self.pose, {}, self.settings, dry_run=False,
            )
        self.assertTrue(result.released)
        self.assertFalse(result.at_ready)


class PostLiftRecheckTests(unittest.TestCase):
    def test_recheck_uses_new_rgbd_and_carried_points_without_graspnet(self):
        placement = PlacementGrounding(True, True, "纸盒", (40, 20, 80, 60), "开口朝上", "in")
        grounding = TargetGrounding(
            False, "", None, "物体已夹走", "复查", (100, 80), (100, 80), "pixel", placement,
        )
        mask = np.zeros((80, 100), dtype=bool)
        mask[20:60, 40:80] = True
        segmenter = SimpleNamespace(segment=Mock(return_value=SegmentedTarget(mask, placement.bbox_xyxy, int(mask.sum()))))
        grasp_backend = SimpleNamespace(build_net=Mock(), infer_frame=Mock())
        placement_backend = SimpleNamespace(analyze_placement_geometry_from_points=Mock(return_value=SimpleNamespace(
            plan=SimpleNamespace(fits=True),
        )))
        client = SimpleNamespace(ground=Mock(return_value=grounding))
        pipeline = VLMGraspPipeline(
            {"vlm_grasp": {"placement_preview": {"min_points": 16}}},
            client,
            segmenter=segmenter,
            net=object(),
            grasp_backend=grasp_backend,
            placement_backend=placement_backend,
        )
        color = np.zeros((80, 100, 3), dtype=np.uint8)
        depth = np.full((80, 100), 600, dtype=np.uint16)
        K = np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]])
        points = np.array([[0.0, 0.0, 0.5], [0.01, 0.0, 0.5]])
        gripper = SimpleNamespace(translation=np.zeros(3), rotation_matrix=np.eye(3))
        with redirect_stdout(io.StringIO()):
            result = pipeline.recheck_placement(
                "抬高后复查", color, depth, K,
                object_points_camera=points,
                gripper_pose_camera=gripper,
            )
        grasp_backend.infer_frame.assert_not_called()
        ground_args, ground_kwargs = client.ground.call_args
        self.assertEqual(ground_args[0], "抬高后复查")
        np.testing.assert_array_equal(ground_args[1], color)
        self.assertTrue(ground_kwargs["placement_only"])
        call = placement_backend.analyze_placement_geometry_from_points.call_args
        np.testing.assert_array_equal(call.args[0], points)
        self.assertIs(call.args[1], mask)
        np.testing.assert_array_equal(call.args[2], depth)
        np.testing.assert_array_equal(call.args[3], K)
        self.assertIs(result.placement_segmentation.mask, mask)
        self.assertTrue(placement_is_executable(result)[0])


class FirstSnapshotPlacementTests(unittest.TestCase):
    def make_call(self, *, selection_mode="auto", executable=True):
        selected_grasp = object()
        pending = SimpleNamespace(grasp=selected_grasp, initial_place=None, recheck_after_lift=True)
        placement = SimpleNamespace(
            required=selection_mode != "disabled",
            found=selection_mode == "auto",
            target_name="纸盒" if selection_mode == "auto" else "",
            relation="in" if selection_mode == "auto" else None,
            selection_mode=selection_mode,
        )
        result = SimpleNamespace(
            grounding=SimpleNamespace(placement=placement),
            segmentation=SimpleNamespace(mask=np.ones((4, 5), dtype=bool)),
            grasp_result=SimpleNamespace(grasps=[object()]),
            placement_segmentation=(
                SimpleNamespace(mask=np.ones((4, 5), dtype=bool))
                if placement.found else None
            ),
            placement_analysis=None,
            placement_error=None,
        )
        analysis = SimpleNamespace(plan=SimpleNamespace(fits=True))
        backend = SimpleNamespace(
            estimate_table_normal=Mock(return_value=np.array([0.0, 0.0, 1.0])),
            analyze_placement_geometry=Mock(return_value=analysis),
        )
        namespace = {
            "np": np,
            "placement_geometry": backend,
            "_combined_exclusion_masks": Mock(return_value=[]),
            "_finish_prepare_pick": Mock(return_value=pending),
            "placement_is_executable": Mock(return_value=(executable, "invalid container")),
            "placement_poses_to_base": Mock(return_value=((1, 2, 3, 4, 5, 6), (6, 5, 4, 3, 2, 1))),
            "_pose_checks": Mock(return_value=(True, "IK ok")),
            "_select_executable_placement": Mock(return_value=(
                (((1, 2, 3, 4, 5, 6), (6, 5, 4, 3, 2, 1)), "IK ok")
                if executable else (None, "invalid container")
            )),
            "PendingPlace": lambda result, release, preplace: SimpleNamespace(
                result=result, release6d=release, preplace6d=preplace,
            ),
        }
        function = prepare_pick_function(namespace)
        with redirect_stdout(io.StringIO()):
            returned = function(
                "抓香蕉", result, np.zeros((4, 5, 3), np.uint8),
                np.full((4, 5), 500, np.uint16), np.eye(3), np.eye(4),
                object(), VLMRobotSettings(), {"min_base_z_m": 0.03},
                PlacementPreviewSettings(min_points=16), np.eye(3),
            )
        return returned, result, backend, namespace, selected_grasp

    def test_valid_first_container_is_planned_with_the_executable_grasp(self):
        pending, result, backend, namespace, selected = self.make_call()
        self.assertIsNotNone(pending.initial_place)
        self.assertTrue(pending.recheck_after_lift)
        self.assertIs(result.placement_analysis, backend.analyze_placement_geometry.return_value)
        self.assertIs(backend.analyze_placement_geometry.call_args.kwargs["best_grasp"], selected)
        namespace["_select_executable_placement"].assert_called_once()

    def test_invalid_first_container_keeps_pick_and_requests_post_lift_recheck(self):
        pending, _, backend, namespace, _ = self.make_call(selection_mode="grasp_only", executable=False)
        self.assertIsNone(pending.initial_place)
        self.assertTrue(pending.recheck_after_lift)
        backend.analyze_placement_geometry.assert_not_called()
        namespace["_select_executable_placement"].assert_called_once()

    def test_explicit_grasp_only_does_not_recheck_or_place(self):
        pending, _, backend, namespace, _ = self.make_call(selection_mode="disabled", executable=False)
        self.assertIsNone(pending.initial_place)
        self.assertFalse(pending.recheck_after_lift)
        backend.analyze_placement_geometry.assert_not_called()
        namespace["placement_is_executable"].assert_not_called()


if __name__ == "__main__":
    unittest.main()
