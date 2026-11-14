"""Checks for weak_spots_e3: the sensitivity grid vote reproduces E3 and the reference vote."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from weak_spots import e3_vote
from weak_spots_e3 import grid_vote


class GridVoteTest(unittest.TestCase):
    iou = np.array([0.1, 0.25, 0.35, 0.59, 0.8, 0.1, 0.8])
    matched = np.array([False, False, False, False, True, False, True])
    cls = np.array(['person'] * 5 + ['cup', 'cup'])

    def test_grid_point_is_e3(self):
        np.testing.assert_array_equal(grid_vote(self.iou, self.matched, self.cls, 0.3, 0.25),
                                      e3_vote(self.iou, self.matched, self.cls))

    def test_weight_one_is_reference_vote(self):
        np.testing.assert_array_equal(grid_vote(self.iou, self.matched, self.cls, 0.3, 1.0),
                                      (~self.matched).astype(float))

    def test_threshold_moves_boundary(self):
        np.testing.assert_allclose(grid_vote(self.iou, self.matched, self.cls, 0.4, 0.0),
                                   [1, 1, 1, 0, 0, 1, 0])
        np.testing.assert_allclose(grid_vote(self.iou, self.matched, self.cls, 0.2, 0.5),
                                   [1, 0.5, 0.5, 0.5, 0, 1, 0])


if __name__ == '__main__':
    unittest.main()
