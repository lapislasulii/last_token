"""Тесты аудитора сабмитов.

    python -m unittest discover -s tests -p test_submission_auditor.py -v

Правила формата проверяются на синтетическом шаблоне и не требуют датасета.
Второе мнение проверяется на реальных данных, если они лежат в data/rascar-hack-case-1/.
"""
import sys
import unittest
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from submission_auditor import DEFAULT_DATA, SubmissionAuditor, check_format  # noqa: E402

IDS = [f'id{i:03d}' for i in range(50)]


def rules(findings):
    return {f.rule for f in findings if f.level == 'error'}


class FormatRules(unittest.TestCase):
    def setUp(self):
        self.good = pd.DataFrame({'id': IDS, 'score': np.linspace(0, 1, len(IDS))})

    def test_correct_file_has_no_errors(self):
        findings, clean = check_format(self.good, IDS)
        self.assertEqual(rules(findings), set())
        self.assertEqual(len(clean), len(IDS))

    def test_missing_and_extra_ids(self):
        sub = self.good.iloc[2:].copy()
        sub.loc[len(IDS)] = ['stranger', 0.3]
        self.assertTrue({'missing_id', 'extra_id', 'row_count'} <= rules(check_format(sub, IDS)[0]))

    def test_duplicate_id(self):
        sub = pd.concat([self.good, self.good.iloc[[4]]], ignore_index=True)
        findings, clean = check_format(sub, IDS)
        self.assertIn('duplicate_id', rules(findings))
        self.assertNotIn('id004', clean.index)

    def test_nan_text_and_out_of_range(self):
        sub = self.good.astype({'score': object})
        sub.loc[0, 'score'], sub.loc[1, 'score'], sub.loc[2, 'score'] = 'nan', 'oops', 1.5
        sub.loc[3, 'score'] = -0.1
        findings, clean = check_format(sub, IDS)
        self.assertTrue({'not_finite', 'out_of_range'} <= rules(findings))
        self.assertEqual(len(clean), len(IDS) - 4)

    def test_constant_scores(self):
        self.assertIn('constant', rules(check_format(self.good.assign(score=0.5), IDS)[0]))

    def test_missing_column(self):
        self.assertIn('columns', rules(check_format(self.good.rename(columns={'score': 'pred'}), IDS)[0]))


@unittest.skipUnless((DEFAULT_DATA / 'test.csv').exists(), 'нет локального датасета')
class SecondOpinion(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        warnings.filterwarnings('ignore')
        cls.auditor = SubmissionAuditor()
        cls.ids = cls.auditor.template.id
        cls.conf = cls.auditor.test.loc[cls.ids, 'confidence'].to_numpy()
        cls.mini = cls.auditor.test.loc[cls.ids, 'mini'].to_numpy()

    def audit(self, scores):
        return self.auditor.audit(pd.DataFrame({'id': self.ids, 'score': scores}), 'case')

    def test_reasonable_model_is_ok(self):
        self.assertEqual(self.audit(0.5 * self.mini + 0.5 * (1 - self.conf)).verdict, 'ok')

    def test_baseline_copy_is_flagged(self):
        result = self.audit(1 - self.conf)
        self.assertEqual(result.verdict, 'warning')
        self.assertIn('same_as_baseline', {f.rule for f in result.findings})

    def test_inverted_scores_are_flagged(self):
        result = self.audit(self.conf)
        self.assertIn('inverted', {f.rule for f in result.findings})
        self.assertGreater(result.stats['missed_by_main'], 10)

    def test_shifted_merge_is_flagged(self):
        result = self.audit(np.roll(0.5 * self.mini + 0.5 * (1 - self.conf), 1))
        self.assertEqual(result.verdict, 'warning')
        self.assertIn('unrelated', {f.rule for f in result.findings})

    def test_mini_model_beats_constant_on_held_out_day(self):
        for day in self.auditor.validation['days'].values():
            self.assertGreater(day['ap_mini'], day['errors'] / day['boxes'])

    def test_class_breakdown_and_pushed_down_class(self):
        good = 0.5 * self.mini + 0.5 * (1 - self.conf)
        result = self.audit(good)
        classes = {c['class'] for c in result.stats['by_class']}
        self.assertEqual(classes, set(self.auditor.test.class_name))
        cups = (self.auditor.test.loc[self.ids, 'class_name'] == 'cup').to_numpy()
        lost = self.audit(np.where(cups, 0.0, good))
        pushed = [f for f in lost.findings if f.rule == 'class_pushed_down']
        self.assertEqual([f.class_name for f in pushed], ['cup'])

    def test_compare_identical_and_different(self):
        good = pd.DataFrame({'id': self.ids, 'score': 0.5 * self.mini + 0.5 * (1 - self.conf)})
        same = self.auditor.compare(good, good.assign(score=good.score * 0.9), 'a', 'b')
        self.assertAlmostEqual(same.stats['spearman'], 1.0)
        self.assertIn('почти одинаковые', ' '.join(same.notes))
        flipped = self.auditor.compare(good, good.assign(score=1 - good.score), 'a', 'b', top=4)
        self.assertLess(flipped.stats['spearman'], 0)
        self.assertEqual(len(flipped.movers), 8)
        self.assertTrue(all(m.ref_rank is not None for m in flipped.movers))

    def test_compare_survives_broken_file(self):
        good = pd.DataFrame({'id': self.ids, 'score': 1 - self.conf})
        broken = good.iloc[5:].copy()
        result = self.auditor.compare(broken, good, 'broken', 'good')
        self.assertIn('не проходит проверку формата', ' '.join(result.notes))
        self.assertEqual(result.stats['rows'], len(self.ids) - 5)

    def test_heuristic_fallback_still_audits(self):
        fallback = SubmissionAuditor(force_heuristics=True)
        self.assertIn('эвристики', fallback.mode)
        result = fallback.audit(pd.DataFrame({'id': self.ids, 'score': self.conf}), 'case')
        self.assertIn('inverted', {f.rule for f in result.findings})


if __name__ == '__main__':
    unittest.main()
