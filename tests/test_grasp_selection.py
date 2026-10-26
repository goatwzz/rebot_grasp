"""Hardware-free checks for ordered grasp selection."""

from __future__ import annotations

import ast
from contextlib import redirect_stdout
import io
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


class FakeGraspGroup:
    def __init__(self, grasps):
        self.grasp_group_array = np.asarray(grasps, dtype=object)

    def __len__(self):
        return len(self.grasp_group_array)

    def __getitem__(self, index):
        return self.grasp_group_array[index]


def selection_function():
    tree = ast.parse((ROOT / "scripts/grasp.py").read_text())
    names = {"_pose_z_ok", "_select_executable_grasp"}
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes],
        type_ignores=[],
    )
    namespace = {
        "GraspGroup": FakeGraspGroup,
        "graspnet_utils": SimpleNamespace(grasp_to_base_poses=lambda grasp, *_args: (
            (0.0, 0.0, grasp.pose_z, 0.0, 0.0, 0.0),
            (0.0, 0.0, grasp.pose_z + 0.1, 0.0, 0.0, 0.0),
            (0.0, 0.0, grasp.pose_z + 0.1, 0.0, 0.0, 0.0),
        )),
        "_rank_grasps": lambda grasps: FakeGraspGroup(
            sorted(grasps.grasp_group_array, key=lambda grasp: -grasp.score)
        ),
    }
    exec(compile(ast.fix_missing_locations(module), "grasp.py", "exec"), namespace)
    return namespace["_select_executable_grasp"]


class GraspSelectionTests(unittest.TestCase):
    def setUp(self):
        self.select = selection_function()
        self.central = SimpleNamespace(score=0.4, pose_z=0.1)
        self.end = SimpleNamespace(score=0.8, pose_z=0.2)
        self.grasps = FakeGraspGroup([self.central, self.end])

    def test_preordered_selection_preserves_middle_preference(self):
        checker = SimpleNamespace(check=lambda *_pose: (True, 0.0))
        with redirect_stdout(io.StringIO()):
            selected = self.select(checker, self.grasps, np.eye(4), 0.08, 0.08, 0.0, 0.03, preordered=True)
        self.assertIs(selected[0], self.central)

    def test_preordered_selection_tries_next_candidate_after_ik_failure(self):
        checker = SimpleNamespace(check=lambda *_pose: (_pose[2] != 0.1, 0.01))
        with redirect_stdout(io.StringIO()):
            selected = self.select(checker, self.grasps, np.eye(4), 0.08, 0.08, 0.0, 0.03, preordered=True)
        self.assertIs(selected[0], self.end)

    def test_default_selection_keeps_network_score_ranking(self):
        checker = SimpleNamespace(check=lambda *_pose: (True, 0.0))
        with redirect_stdout(io.StringIO()):
            selected = self.select(checker, self.grasps, np.eye(4), 0.08, 0.08, 0.0, 0.03)
        self.assertIs(selected[0], self.end)


if __name__ == "__main__":
    unittest.main()
