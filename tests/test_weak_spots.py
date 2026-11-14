"""Checks for weak_spots: leader file decomposition (D, V, P, L), rule E1, key E2, threshold E3."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from weak_spots import decompose, decompose_pose, e1_veto, e2_conf_term, e3_vote, formula, in_window


class DecomposeTest(unittest.TestCase):
    def setUp(self):
        # пять боксов: человек вне окна (P=1), человек у окна, кисть с вето, чашка, человек вне окна с L
        self.conf = np.array([0.91, 0.88, 0.70, 0.55, 0.90])
        self.d = np.array([0, 2, 6, 1, 0]) / 6
        self.v = np.array([0., 0., 1., 0., 0.])
        self.p = np.array([1., 0., 0., 0., 0.])
        self.l = np.array([0., 0., 0., 0., 1.])
        self.person_out = np.array([True, False, False, False, True])
        self.flip = (1 - self.conf + self.d + 2 * self.v) / 4
        self.pose = formula(1 - self.conf, self.d * 6, 6, self.p, self.v, self.l, self.person_out)

    def test_round_trip(self):
        d, v = decompose(self.flip, self.conf)
        np.testing.assert_allclose(d, self.d)
        np.testing.assert_array_equal(v, self.v)
        p, l = decompose_pose(self.pose, self.conf, d, v, self.person_out)
        np.testing.assert_array_equal(p, self.p)
        np.testing.assert_array_equal(l, self.l)
        rebuilt = formula(1 - self.conf, d * 6, 6, p, v, l, self.person_out)
        self.assertLess(np.abs(rebuilt - self.pose).max(), 1e-12)

    def test_pose_share_is_seven_way(self):
        # человек вне окна, D=0, P=1: D' = 1/7
        self.assertAlmostEqual(4 * self.pose[0] - (1 - self.conf[0]), 1 / 7)

    def test_off_grid_d_stops(self):
        with self.assertRaises(ValueError):
            decompose(self.flip + 0.01, self.conf)

    def test_inconsistent_pose_stops(self):
        d, v = decompose(self.flip, self.conf)
        broken = self.pose.copy()
        broken[0] += 0.05 / 4          # P = 7·D' − 6·D уходит с {0, 1}
        with self.assertRaises(ValueError):
            decompose_pose(broken, self.conf, d, v, self.person_out)
        broken = self.pose.copy()
        broken[3] += 1 / 24            # у чашки D' ≠ D
        with self.assertRaises(ValueError):
            decompose_pose(broken, self.conf, d, v, self.person_out)


class RulesTest(unittest.TestCase):
    def test_window(self):
        np.testing.assert_array_equal(in_window([0, 300, 400], [100, 361, 500]), [True, False, False])

    def test_e1_only_window_people_with_full_agreement(self):
        veto = [1, 1, 1, 1, 1]
        d = [0, 0, 1 / 6, 0, 0]
        cls = ['person', 'person', 'person', 'nude_hands', 'person']
        window = [True, False, True, True, True]
        np.testing.assert_array_equal(e1_veto(veto, d, cls, window), [0, 1, 1, 1, 0])
        np.testing.assert_array_equal(e1_veto(veto, d, cls, window, everywhere=True), [0, 0, 1, 0, 0])
        np.testing.assert_array_equal(e1_veto([0, 0], [0, 0], ['person'] * 2, [True, True]), [0, 0])

    def test_e2_key_people_only(self):
        key = e2_conf_term([0.9, 0.9, 0.9], [0.5, 0.0, 0.5], ['person', 'person', 'cup'])
        np.testing.assert_allclose(key, [0.3, 0.55, 0.1])

    def test_e3_two_level_vote(self):
        votes = e3_vote([0.1, 0.3, 0.59, 0.8, 0.1, 0.8],
                        [False, False, False, True, False, False],
                        ['person', 'person', 'person', 'person', 'cup', 'cup'])
        np.testing.assert_allclose(votes, [1, 0.25, 0.25, 0, 1, 1])


if __name__ == '__main__':
    unittest.main()
