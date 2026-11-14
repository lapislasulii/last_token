"""Checks for weak_spots_e3 eval-e3: per-member E3 votes and the gate against person_vote with several members."""
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import redetect as R
from weak_spots import e3_vote
from weak_spots_e3 import member_votes, setup


class MemberVotesTest(unittest.TestCase):
    cls = np.array(['person', 'person', 'person', 'cup'])
    matched = np.array([[False, False, True, False],
                        [True, False, True, True]])
    iou = np.array([[0.1, 0.5, 0.9, 0.2],
                    [0.7, 0.2, 0.8, 0.9]])

    def test_one_member_is_e3(self):
        np.testing.assert_array_equal(member_votes(self.matched[0], self.iou[0], self.cls),
                                      e3_vote(self.iou[0], self.matched[0], self.cls))

    def test_votes_add_over_members(self):
        # человек 1: 1 + 0; человек 2: 0.25 + 1; человек 3: 0 + 0; чашка: 1 + 0
        np.testing.assert_allclose(member_votes(self.matched, self.iou, self.cls), [1, 1.25, 0, 1])

    def test_weight_one_is_reference(self):
        np.testing.assert_array_equal(member_votes(self.matched, self.iou, self.cls, 0.3, 1.0),
                                      (~self.matched).sum(axis=0))


@unittest.skipUnless((ROOT / 'outputs/expert_veto/train_pose.csv').exists()
                     and (ROOT / 'outputs/redetect_strong/pred_day_20261025.csv').exists(),
                     'нужны локальный кэш вето и дневные предсказания base')
class GateWithSeveralMembersTest(unittest.TestCase):
    def test_gate_and_votes_with_a_copy_of_base(self):
        R.WORK = ROOT / 'outputs' / 'redetect_strong'
        original = R.pred_path
        with mock.patch.object(R, 'pred_path', lambda name, member: original(name, 'base' if member == 'copy' else member)):
            df, matched, iou, reference = setup(['base', 'copy'])     # внутри — assert против person_vote.scores
        self.assertEqual(matched.shape[0], 2)
        np.testing.assert_array_equal(matched[0], matched[1])
        self.assertTrue((df.n_members == 2).all())
        # копия того же участника: без голоса позы скор тот же, что у одного base;
        # у людей вне окна вес детектора вдвое больше веса позы: D' = (2v + P) / 3 вместо (v + P) / 2
        single = setup(['base'])
        out = df.person_out.to_numpy()
        np.testing.assert_allclose(reference.to_numpy()[~out], single[3].to_numpy()[~out])
        v, p = single[0].votes.to_numpy()[out], df.P.to_numpy()[out]
        expected = (1 - df.confidence.to_numpy()[out] + (2 * v + p) / 3
                    + 2 * np.maximum(df.veto, df.L).to_numpy()[out]) / 4
        np.testing.assert_allclose(reference.to_numpy()[out], expected)
        np.testing.assert_allclose(member_votes(matched, iou, df.class_name) / 2,
                                   member_votes(single[1], single[2], df.class_name))


if __name__ == '__main__':
    unittest.main()
