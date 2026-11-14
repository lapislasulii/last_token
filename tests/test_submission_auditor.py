"""Тесты аудитора сабмитов.

    python -m unittest discover -s tests -p test_submission_auditor.py -v

Правила формата и реестр хешей проверяются без датасета.
Второе мнение и опознание рецептов — на реальных данных, если они лежат в data/rascar-hack-case-1/.
"""
import sys
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from submission_auditor import (DEFAULT_DATA, DETECTOR_FEATURES, STRONG, SubmissionAuditor,  # noqa: E402
                                check_format, known_by_sha, sha256)

IDS = [f'id{i:03d}' for i in range(50)]


def errors(findings):
    return {f.rule for f in findings if f.level == 'error'}


def rules(result):
    return {f.rule for f in result.findings}


class FormatRules(unittest.TestCase):
    def setUp(self):
        self.good = pd.DataFrame({'id': IDS, 'score': np.linspace(0, 1, len(IDS))})

    def test_correct_file_has_no_errors(self):
        findings, clean = check_format(self.good, IDS)
        self.assertEqual(errors(findings), set())
        self.assertEqual(len(clean), len(IDS))

    def test_missing_extra_and_duplicate_ids(self):
        sub = pd.concat([self.good.iloc[2:], self.good.iloc[[4]]], ignore_index=True)
        sub.loc[len(sub)] = ['stranger', 0.3]
        findings, clean = check_format(sub, IDS)
        self.assertTrue({'missing_id', 'extra_id', 'duplicate_id'} <= errors(findings))
        self.assertNotIn('row_count', errors(findings))          # 50 строк: потеря двух скрыта дублем и чужим id
        self.assertNotIn('id004', clean.index)
        self.assertIn('row_count', errors(check_format(self.good.iloc[1:], IDS)[0]))

    def test_nan_text_and_out_of_range(self):
        sub = self.good.astype({'score': object})
        sub.loc[0, 'score'], sub.loc[1, 'score'], sub.loc[2, 'score'], sub.loc[3, 'score'] = 'nan', 'oops', 1.5, -0.1
        findings, clean = check_format(sub, IDS)
        self.assertTrue({'not_finite', 'out_of_range'} <= errors(findings))
        self.assertEqual(len(clean), len(IDS) - 4)

    def test_constant_and_missing_column(self):
        self.assertIn('constant', errors(check_format(self.good.assign(score=0.5), IDS)[0]))
        self.assertIn('columns', errors(check_format(self.good.rename(columns={'score': 'pred'}), IDS)[0]))

    def test_known_sha_registry(self):
        self.assertEqual(known_by_sha('16016509246975456ce264ef9fda459e6f6b22dab499f89eb07433e5b5e64474')[1], 0.89753)
        self.assertEqual(known_by_sha('5eb4d404c310' + '0' * 52)[1], 0.82471)   # в отчёте записан только префикс
        self.assertIsNone(known_by_sha('0' * 64))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'x.csv'
            path.write_text('id,score\n', encoding='utf-8')
            self.assertEqual(len(sha256(path)), 64)


@unittest.skipUnless((DEFAULT_DATA / 'test.csv').exists(), 'нет локального датасета')
class SecondOpinion(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        warnings.filterwarnings('ignore')
        cls.auditor = SubmissionAuditor()
        cls.ids = cls.auditor.template.id
        cls.recipes = {k: v[cls.ids].to_numpy() for k, v in cls.auditor.recipes.items()}
        cls.good = cls.recipes.get('ens3_1to1', cls.recipes['baseline'])
        test = cls.auditor.test.loc[cls.ids]
        cls.veto = ((test.confidence < 0.6) & (test.votes_missing > 0)).to_numpy() if 'votes_missing' in test else None

    def audit(self, scores, name='case'):
        return self.auditor.audit(pd.DataFrame({'id': self.ids, 'score': scores}), name)

    def test_mini_model_beats_baseline_on_held_out_days(self):
        for day in self.auditor.validation['days'].values():
            self.assertGreater(day['ap_mini'], day['ap_baseline'])

    def test_good_submission_is_ok(self):
        self.assertEqual(self.audit(self.good).verdict, 'ok')

    @unittest.skipUnless((STRONG / 'pred_all_s2.csv').exists(), 'нет предсказаний ансамбля')
    def test_every_recipe_is_identified_with_and_without_veto(self):
        for name, scores in self.recipes.items():
            for veto in (False, True):
                identity = self.audit((scores + self.veto) / 2 if veto else scores).identity
                self.assertEqual((identity.get('recipe'), identity.get('veto')), (name, veto), name)
        identity = self.audit((self.recipes['ens3_1to1'] + self.veto) / 2).identity
        self.assertEqual(identity['kaggle_public'], 0.90211)
        self.assertEqual(len(identity['veto_ids']), int(self.veto.sum()))

    @unittest.skipUnless((STRONG / 'pred_all_s2.csv').exists(), 'нет предсказаний ансамбля')
    def test_old_leader_is_flagged_and_differs_in_eight_rows(self):
        old = (self.recipes['ens3_any'] + self.veto) / 2
        self.assertIn('not_best', rules(self.audit(old)))
        frame = lambda s: pd.DataFrame({'id': self.ids, 'score': s})  # noqa: E731
        comparison = self.auditor.compare(frame((self.good + self.veto) / 2), frame(old), 'new', 'old')
        self.assertEqual(comparison.stats['changed_rows'], 8)          # как в reports/one_to_one.md
        self.assertIn('почти одинаковые', ' '.join(comparison.notes))

    @unittest.skipUnless((STRONG / 'pred_all_s2.csv').exists(), 'нет предсказаний ансамбля')
    def test_tie_broken_leader_is_identified_and_ties_reported(self):
        from tie_break import tie_break
        leader = pd.Series((self.good + self.veto) / 2, index=self.ids.values)
        plain = self.audit(leader.to_numpy())
        self.assertIn('ties', rules(plain))
        self.assertGreater(plain.stats['ties']['in_ties'], 300)
        broken, eps = tie_break(leader, 1 - self.auditor.test.loc[self.ids, 'iou_core'].clip(0, 1).set_axis(self.ids.values))
        result = self.audit(broken.loc[self.ids].to_numpy())
        identity = result.identity
        self.assertEqual((identity['recipe'], identity['veto'], identity.get('tie_break')), ('ens3_1to1', True, True))
        self.assertAlmostEqual(identity['tie_break_eps'], eps, places=8)
        self.assertNotIn('ties', rules(result))
        self.assertEqual(result.verdict, 'ok')

    def test_leak_note_and_clean_baseline_are_reported(self):
        if self.auditor.features is not DETECTOR_FEATURES:
            self.skipTest('без признаков детектора')
        self.assertIn('leak_note', self.auditor.validation)
        self.assertEqual(len(self.auditor.validation['temporal_only']), 2)

    def test_broken_queues_are_flagged(self):
        self.assertIn('same_as_baseline', rules(self.audit(self.recipes['baseline'])))
        self.assertIn('inverted', rules(self.audit(1 - self.good)))
        shifted = self.audit(np.roll(self.good, 1))
        self.assertIn('unrelated', rules(shifted))
        self.assertNotIn('inverted', rules(shifted))
        cups = (self.auditor.test.loc[self.ids, 'class_name'] == 'cup').to_numpy()
        pushed = [f.class_name for f in self.audit(np.where(cups, 0.0, self.good)).findings if f.rule == 'class_pushed_down']
        self.assertEqual(pushed, ['cup'])

    def test_queue_has_reasons_and_veto_explanation(self):
        scores = (self.good + self.veto) / 2 if self.veto is not None else self.good
        result = self.audit(scores)
        self.assertEqual([f.main_rank for f in result.queue], list(range(1, 16)))
        self.assertTrue(all(f.reasons for f in result.queue))
        if self.veto is not None:
            self.assertTrue(any('вето' in r for f in result.queue for r in f.reasons))

    def test_heuristic_fallback_still_audits(self):
        fallback = SubmissionAuditor(force_heuristics=True)
        self.assertIn('эвристики', fallback.mode)
        self.assertIn('inverted', rules(fallback.audit(pd.DataFrame({'id': self.ids, 'score': 1 - self.good}), 'x')))

    def test_without_detector_files_falls_back_to_temporal(self):
        with tempfile.TemporaryDirectory() as empty:
            auditor = SubmissionAuditor(strong=Path(empty))
        self.assertEqual(len(auditor.features), 5)
        self.assertTrue(auditor.notes)
        self.assertEqual(auditor.audit(pd.DataFrame({'id': self.ids, 'score': self.good}), 'x').verdict, 'ok')


if __name__ == '__main__':
    unittest.main()
