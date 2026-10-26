"""Offline checks for elongated-object grasp ranking."""

from __future__ import annotations

import unittest

import numpy as np

from utils.grasp_candidate_quality import rank_elongated_grasp_centers


class GraspCandidateQualityTests(unittest.TestCase):
    def setUp(self):
        self.mask = np.zeros((100, 200), dtype=bool)
        self.mask[30:70, 20:180] = True
        self.camera_matrix = np.array([[1000.0, 0.0, 0.0], [0.0, 1000.0, 0.0], [0.0, 0.0, 1.0]])

    def test_rejects_end_and_side_even_when_network_scores_are_high(self):
        pixels = np.array([[100, 50], [145, 50], [23, 50], [100, 34]])
        translations = np.column_stack((pixels / 2000.0, np.full(len(pixels), 0.5)))
        order, fractions, clearances = rank_elongated_grasp_centers(
            translations, np.array([0.51, 0.58, 0.99, 0.95]), self.mask, self.camera_matrix,
            min_clearance_m=0.006, end_fraction=0.15,
        )
        self.assertEqual(order.tolist(), [0, 1])
        self.assertLess(fractions[2], 0.15)
        self.assertLess(clearances[3], 0.006)

    def test_returns_no_candidate_instead_of_falling_back_to_unsafe_grasp(self):
        translations = np.array([[0.0115, 0.025, 0.5], [0.05, 0.017, 0.5]])
        order, _, _ = rank_elongated_grasp_centers(
            translations, np.array([0.9, 0.8]), self.mask, self.camera_matrix,
            min_clearance_m=0.006, end_fraction=0.15,
        )
        self.assertEqual(order.tolist(), [])

    def test_rejects_non_elongated_mask(self):
        square = np.ones((40, 40), dtype=bool)
        with self.assertRaises(ValueError):
            rank_elongated_grasp_centers(
                np.array([[0.01, 0.01, 0.5]]), np.array([0.8]), square, self.camera_matrix,
                min_clearance_m=0.006, end_fraction=0.15,
            )

    def test_more_vertical_candidate_wins_when_scores_are_close(self):
        translations = np.array([[0.05, 0.025, 0.5], [0.05, 0.025, 0.5]])
        order, _, _ = rank_elongated_grasp_centers(
            translations, np.array([0.60, 0.56]), self.mask, self.camera_matrix,
            min_clearance_m=0.006, end_fraction=0.15,
            approach_tilt_deg=np.array([40.0, 20.0]),
        )
        self.assertEqual(order.tolist(), [1, 0])

    def test_middle_candidate_precedes_shoulder_when_score_is_comparable(self):
        pixels = np.array([[52, 50], [108, 50]])
        translations = np.column_stack((pixels / 2000.0, np.full(len(pixels), 0.5)))
        order, _, _ = rank_elongated_grasp_centers(
            translations, np.array([0.6, 0.35]), self.mask, self.camera_matrix,
            min_clearance_m=0.006, end_fraction=0.15,
        )
        self.assertEqual(order.tolist(), [1, 0])

    def test_outside_mask_is_rejected_even_with_zero_clearance_limit(self):
        order, _, _ = rank_elongated_grasp_centers(
            np.array([[0.05, 0.045, 0.5]]), np.array([0.9]), self.mask, self.camera_matrix,
            min_clearance_m=0.0, end_fraction=0.15,
        )
        self.assertEqual(order.tolist(), [])


if __name__ == "__main__":
    unittest.main()
