"""Box localization and the pick/place sequence shared with scripts/set.py."""

from __future__ import annotations

import threading
import time
from typing import Any, Optional

import numpy as np

from .ordinary_grasp import GraspPose


# Keep the release position/orientation used by the original set.py demo.
PLACE_Z_M = 0.08
PLACE_RPY = (0.0, 1.2, 0.0)


def select_box_grasp(grasps: list[GraspPose]) -> Optional[GraspPose]:
    boxes = [
        grasp for grasp in grasps
        if grasp.is_valid and "box" in grasp.class_name.lower()
        and "banana" not in grasp.class_name.lower()
        and np.all(np.isfinite(grasp.position))
    ]
    return max(boxes, key=lambda grasp: grasp.conf, default=None)


def box_center_to_base(box: GraspPose, T_cam2base: np.ndarray) -> np.ndarray:
    """Use the original box estimate for XY and fixed base-frame Z=0.08 m."""
    box_base = T_cam2base @ np.append(box.position, 1.0)
    return np.array([float(box_base[0]), float(box_base[1]), PLACE_Z_M])


def wait_motion(controller: Any, duration: float, extra: float = 0.6) -> bool:
    thread = getattr(controller, "_send_thread", None)
    if thread is not None and thread.is_alive():
        thread.join(timeout=duration + extra + 2.0)
        if thread.is_alive():
            print("[Place] Motion thread timed out; stop sequence")
            return False
    else:
        time.sleep(duration + extra)
    return True


def move_ready(controller: Any, ready_cfg: dict[str, Any]) -> bool:
    duration = float(ready_cfg.get("duration", 3.0))
    if not controller.move_to_traj(
        x=float(ready_cfg.get("x", 0.25)),
        y=float(ready_cfg.get("y", 0.0)),
        z=float(ready_cfg.get("z", 0.35)),
        roll=float(ready_cfg.get("roll", 0.0)),
        pitch=float(ready_cfg.get("pitch", 1.2)),
        yaw=float(ready_cfg.get("yaw", 0.0)),
        duration=duration,
    ):
        print("[Place] Ready position IK failed")
        return False
    return wait_motion(controller, duration)


def execute_place_sequence(
    controller: Any,
    grasp_driver: Any,
    grasp6d: tuple[float, ...],
    pre6d: tuple[float, ...],
    box_center: np.ndarray,
    ready_cfg: dict[str, Any],
    dry_run: bool,
) -> bool:
    """Run set.py's grasp -> ready -> box -> release -> ready sequence."""
    print(f"[Step2] pregrasp xyz={pre6d[:3]} rpy={pre6d[3:]}")
    print(f"[Step2] grasp    xyz={grasp6d[:3]} rpy={grasp6d[3:]}")
    print(f"[Step4] box xyz={box_center.tolist()} rpy={PLACE_RPY}")
    if dry_run:
        print("[Place] dry run; skip pick/place motion")
        return True

    print("[Step2] Open gripper")
    grasp_driver.open_gripper_for_pick()
    for label, pose, duration in (("pregrasp", pre6d, 2.0), ("grasp", grasp6d, 1.5)):
        print(f"[Step2] Move to {label}")
        if not controller.move_to_traj(*pose, duration=duration):
            print(f"[Step2] {label} IK failed")
            return False
        if not wait_motion(controller, duration):
            return False

    print("[Step2] Closing gripper")
    if not grasp_driver.grasp():
        print("[Step2] Empty grasp; skip placement")
        return False
    print("[Step3] Lift to ready position with object")
    if not move_ready(controller, ready_cfg):
        return False

    print(f"[Step4] Move to box center: xyz={box_center.tolist()}")
    if not controller.move_to_traj(*box_center, *PLACE_RPY, duration=2.0):
        print("[Step4] Place position IK failed")
        return False
    if not wait_motion(controller, 2.0):
        return False
    print("[Step4] Release gripper - object falls into box")
    grasp_driver.open_gripper(timeout=0.5)

    # Preserve set.py's delayed close while the arm returns to ready.
    def close_gripper() -> None:
        time.sleep(0.8)
        with grasp_driver._state_lock:
            grasp_driver._target_pos = 0.0
            grasp_driver._state = grasp_driver._STATE_POSITION
            grasp_driver._position_reached = False

    close_thread = threading.Thread(target=close_gripper)
    close_thread.start()
    try:
        print("[Step5] Return to ready position")
        return move_ready(controller, ready_cfg)
    finally:
        close_thread.join()
