"""Тесты сравнения OOF: python -m unittest discover -s tests -p test_compare_oof.py -v"""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from compare_oof import compare, main, markdown, paired_bootstrap  # noqa: E402
from evaluate_predictions import DEFAULT_DATA  # noqa: E402


def toy():
    """Два дня по 3 кадра; ошибки — строки с error=1."""
    rows = []
    for day in ('2026-10-25', '2026-10-26'):
        for frame in range(3):
            for k in range(4):
                rows.append({'id': f'{day}-{frame}-{k}', 'day': day, 'frame_id': f'{day}-{frame}', 'class_name': 'cup',
                             'confidence': 0.5 + 0.1 * k, 'error': int(k < 2)})
    df = pd.DataFrame(rows)
    perfect = pd.Series(np.where(df.error == 1, 0.9, 0.1), index=df.id)   # ошибки выше всех правильных
    noisy = pd.Series(df.confidence.to_numpy(), index=df.id)              # ошибки (низкий confidence) — в конце
    return df, perfect, noisy


class CompareLogic(unittest.TestCase):
    def test_pass_and_fail_by_criterion(self):
        df, perfect, noisy = toy()
        self.assertTrue(compare(df, perfect, noisy, 0.0, 0.0, repeats=0)['passed'])
        failed = compare(df, noisy, perfect, 0.0, 0.0, repeats=0)
        self.assertFalse(failed['passed'])
        self.assertTrue(any('прирост' in r for r in failed['reasons']))
        self.assertTrue(any('хуже опорного' in r for r in failed['reasons']))

    def test_identical_files_have_zero_gain(self):
        df, _, noisy = toy()
        result = compare(df, noisy, noisy.copy(), 0.0, 0.0, repeats=50)
        self.assertAlmostEqual(result['gain'], 0.0)
        self.assertTrue(result['passed'])
        self.assertEqual(result['bootstrap']['positive_share'], 0.0)

    def test_bootstrap_is_paired_and_reproducible(self):
        df, perfect, noisy = toy()
        a = paired_bootstrap(df, perfect, noisy, 100, seed=1)
        b = paired_bootstrap(df, perfect, noisy, 100, seed=1)
        pd.testing.assert_frame_equal(a, b)
        self.assertTrue((a['mean'] >= -1e-12).all())        # идеальная очередь не хуже ни на одной выборке

    def test_markdown_mentions_verdict(self):
        df, perfect, noisy = toy()
        text = markdown(compare(df, perfect, noisy, 0.001, 0.0, repeats=20), 'new', 'ref')
        self.assertIn('**Пройден**', text)
        self.assertIn('| разница |', text)


@unittest.skipUnless((DEFAULT_DATA / 'train.csv').exists(), 'нет локального датасета')
class CompareCli(unittest.TestCase):
    def test_cli_exit_codes(self):
        train = pd.read_csv(DEFAULT_DATA / 'train.csv', dtype={'id': str})
        train = train[train.class_name != 'table']
        with tempfile.TemporaryDirectory() as folder:
            good = Path(folder) / 'good.csv'
            bad = Path(folder) / 'bad.csv'
            pd.DataFrame({'id': train.id, 'score': 1 - train.confidence}).to_csv(good, index=False)
            pd.DataFrame({'id': train.id, 'score': train.confidence}).to_csv(bad, index=False)
            self.assertEqual(main([str(good), 'baseline', '--bootstrap', '0']), 0)
            self.assertEqual(main([str(bad), 'baseline', '--bootstrap', '0']), 2)
            pd.DataFrame({'id': train.id[:5], 'score': 0.5}).to_csv(bad, index=False)
            self.assertEqual(main([str(bad), 'baseline', '--bootstrap', '0']), 1)


if __name__ == '__main__':
    unittest.main()
