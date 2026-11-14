"""Тесты разбивки ничьих: python -m unittest discover -s tests -p test_tie_break.py -v"""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from tie_break import (DATA, STRONG, discover_members, oof_report, order_preserved, safe_eps,  # noqa: E402
                       tie_break, tie_stats)


class TieBreakLogic(unittest.TestCase):
    def setUp(self):
        self.score = pd.Series([0.5, 0.5, 0.5, 0.25, 0.25, 1.0], index=list('abcdef'))
        self.key = pd.Series([0.1, 0.9, 0.5, 1.0, 0.0, 0.3], index=list('abcdef'))

    def test_orders_inside_groups_only(self):
        new, eps = tie_break(self.score, self.key)
        self.assertTrue(order_preserved(self.score, new))
        self.assertEqual(list(new.sort_values(ascending=False).index), ['f', 'b', 'c', 'a', 'd', 'e'])
        self.assertTrue(new.between(0, 1).all())
        self.assertEqual(tie_stats(new)['in_ties'], 0)

    def test_extreme_key_cannot_jump_groups(self):
        new, _ = tie_break(self.score, self.key.where(self.score != 0.25, 1.0).where(self.score != 0.5, 0.0))
        self.assertLess(new[['a', 'b', 'c']].max(), new['f'])
        self.assertGreater(new[['a', 'b', 'c']].min(), new[['d', 'e']].max())

    def test_eps_adapts_to_small_gaps_and_rejects_unsafe(self):
        close = pd.Series([0.5, 0.5002, 0.5004])
        self.assertLess(safe_eps(close), 0.0002 / 3 + 1e-12)
        with self.assertRaises(ValueError):
            safe_eps(close, 0.01)
        with self.assertRaises(ValueError):
            tie_break(self.score, self.key * 2)          # ключ вне [0, 1]

    def test_tie_stats(self):
        self.assertEqual(tie_stats(self.score), {'unique': 3, 'rows': 6, 'in_ties': 5, 'groups': 2, 'largest': 3})

    def test_discover_members_naming(self):
        with tempfile.TemporaryDirectory() as folder:
            for name in ['pred_all.csv', 'pred_all_s1.csv', 'pred_all_flip.csv', 'pred_all_s1_flip.csv',
                         'pred_all_st_strict.csv', 'pred_day_20261025.csv']:
                (Path(folder) / name).write_text('frame_id\n', encoding='utf-8')
            self.assertEqual(sorted(discover_members(Path(folder))), ['base', 'base_flip', 's1', 's1_flip', 'st_strict'])


@unittest.skipUnless((DATA / 'test.csv').exists() and (STRONG / 'oof_features.csv').exists(), 'нет датасета или признаков')
class TieBreakOnData(unittest.TestCase):
    def test_oof_gain_on_both_days(self):
        rows = [r for r in oof_report() if r['source'] == 'одиночный детектор base' and r['variant'] == 'формула']
        before, after = rows
        for day in [k for k in before if k.startswith('20')]:
            self.assertGreater(after[day], before[day])
        self.assertLess(after['in_ties'], before['in_ties'] / 3)

    def test_cli_writes_valid_file(self):
        import tie_break as tb
        from tie_break import member_agreement, test_boxes
        boxes = test_boxes()
        agreement = member_agreement(boxes, ['base'])
        score = (1 - boxes.confidence + agreement.D) / 2
        with tempfile.TemporaryDirectory() as folder:
            src = Path(folder) / 'leader.csv'
            pd.DataFrame({'id': boxes.id, 'score': score}).to_csv(src, index=False)
            out = tb.run_file(src, ['base'], None, None)
            result = pd.read_csv(out, dtype={'id': str})
        template = pd.read_csv(DATA / 'sample_submission.csv', dtype={'id': str})
        self.assertEqual(list(result.id), list(template.id))
        self.assertTrue(np.isfinite(result.score).all() and result.score.between(0, 1).all())
        merged = pd.Series(score.to_numpy(), index=boxes.id).loc[result.id]
        self.assertTrue(order_preserved(merged.reset_index(drop=True), result.score))


if __name__ == '__main__':
    unittest.main()
