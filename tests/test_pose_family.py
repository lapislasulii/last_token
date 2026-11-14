"""Checks for pose_family: D' by zone and class, P and L from the two leader files, only hall people change."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import person_vote as PV
from pose_family import CANDIDATE_W, LEADER_W, assert_only_hall_people, d_prime, pose_vote_at, score
from weak_spots import decompose, decompose_pose


class DPrimeTest(unittest.TestCase):
    def test_leader_and_candidate_for_hall_person(self):
        self.assertAlmostEqual(float(d_prime([2 / 6], [1], [True], LEADER_W)[0]), (6 * 2 / 6 + 1) / 7)
        self.assertAlmostEqual(float(d_prime([2 / 6], [1], [True], CANDIDATE_W)[0]), (2 / 6 + 1) / 2)

    def test_window_people_and_other_classes_keep_y(self):
        # у окна и у других классов person_out = False: D' = Y при любом весе и любом P
        for w in (LEADER_W, CANDIDATE_W, 2 / 3):
            np.testing.assert_allclose(d_prime([0.5, 1.0, 0.0], [1, 0, 1], [False, False, False], w), [0.5, 1.0, 0.0])

    def test_score_uses_max_of_veto_and_legs(self):
        s = score([0.9], [0.0], [0.0], [0.0], [1.0], [True], CANDIDATE_W)
        self.assertAlmostEqual(float(s[0]), (0.1 + 0 + 2) / 4)


class LeaderDecompositionTest(unittest.TestCase):
    def test_p_and_l_from_two_files(self):
        conf = np.array([0.9, 0.85, 0.7, 0.6, 0.88, 0.92])
        y = np.array([1, 0, 6, 3, 2, 0]) / 6
        v = np.array([0., 0., 1., 0., 0., 0.])
        p = np.array([0., 1., 0., 0., 1., 0.])
        l = np.array([0., 0., 0., 0., 0., 1.])
        out = np.array([True, True, False, False, True, True])   # люди вне окна; 3-й и 4-й — окно или другой класс
        flip = (1 - conf + y + 2 * v) / 4
        flip_pose = score(conf, y, p, v, l, out, LEADER_W)
        y2, v2 = decompose(flip, conf)
        p2, l2 = decompose_pose(flip_pose, conf, y2, v2, out)
        np.testing.assert_allclose(y2, y)
        np.testing.assert_array_equal(v2, v)
        np.testing.assert_array_equal(p2, p)
        np.testing.assert_array_equal(l2, l)
        self.assertLess(np.abs(score(conf, y2, p2, v2, l2, out, LEADER_W) - flip_pose).max(), 1e-12)
        changed = assert_only_hall_people(flip_pose, score(conf, y2, p2, v2, l2, out, CANDIDATE_W), out)
        np.testing.assert_array_equal(changed, out & (y != p))


class OnlyHallPeopleTest(unittest.TestCase):
    def test_change_outside_hall_people_fails(self):
        with self.assertRaises(AssertionError):
            assert_only_hall_people([0.1, 0.2], [0.1, 0.3], [True, False])
        np.testing.assert_array_equal(assert_only_hall_people([0.1, 0.2], [0.15, 0.2], [True, False]), [True, False])


class PoseVoteCopyTest(unittest.TestCase):
    def test_copy_matches_original_at_default_threshold(self):
        rows = pd.DataFrame({'id': ['a', 'b', 'c'], 'frame_id': ['f', 'f', 'f'],
                             'class_name': ['person', 'person', 'cup'],
                             'x1': [0, 200, 0], 'y1': [0, 0, 0], 'x2': [100, 300, 10], 'y2': [200, 200, 10]})
        pose = pd.DataFrame({'frame_id': ['f', 'f'], 'src': ['full', 'full'], 'conf': [0.9, 0.45],
                             'x1': [0, 200], 'y1': [0, 0], 'x2': [100, 300], 'y2': [210, 200]})
        np.testing.assert_array_equal(pose_vote_at(rows, pose, PV.POSE_CONF), PV.pose_vote(rows, pose))
        np.testing.assert_array_equal(pose_vote_at(rows, pose, 0.5), [0, 1, 0])
        np.testing.assert_array_equal(pose_vote_at(rows, pose, 0.4), [0, 0, 0])


if __name__ == '__main__':
    unittest.main()
