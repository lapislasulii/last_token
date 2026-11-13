"""Add the expert veto on top of any score file: new = (score + veto) / 2.

veto (0 or 1) comes from tools/expert_veto.py, which must be run first for the split.
No weight is fitted. Input scores must already be in [0, 1].

Usage:
  python tools/blend_veto.py train oof/synthetic_rankblend_seed43.csv
      -> oof/synthetic_rankblend_seed43_veto.csv, AP by day against baseline, input and plain veto
  python tools/blend_veto.py test submissions/synthetic_rankblend_seed43.csv
      -> submissions/synthetic_rankblend_seed43_veto.csv (not uploaded)
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('split', choices=['train', 'test'])
    parser.add_argument('scores', type=Path, help='CSV with id,score')
    args = parser.parse_args()
    features = ROOT / f'outputs/expert_veto/{args.split}_features.csv'
    assert features.exists(), f'Run first: python tools/expert_veto.py {args.split}'
    rows = pd.read_csv(features, dtype={'id': str})
    scores = pd.read_csv(args.scores, dtype={'id': str})[['id', 'score']]
    assert scores.id.is_unique and set(scores.id) == set(rows.id), 'ids differ from the split'
    assert np.isfinite(scores.score).all() and scores.score.between(0, 1).all(), 'scores must be in [0, 1]'
    rows = rows.merge(scores, on='id', validate='one_to_one')
    rows['blend'] = (rows.score + rows.veto) / 2
    path = args.scores.with_name(args.scores.stem + '_veto.csv')
    rows[['id']].assign(score=rows.blend).to_csv(path, index=False)
    print(f'Wrote {path}; veto raised {int(rows.veto.sum())} of {len(rows)} boxes')
    if args.split == 'test':
        return
    day = rows.frame_id.str.extract(r'_(2026\d{4})_')[0]
    for name, value in (('baseline 1-conf', 1 - rows.confidence), ('baseline + veto', (1 - rows.confidence + rows.veto) / 2),
                        ('input', rows.score), ('input + veto', rows.blend)):
        by_day = [average_precision_score(rows.error[day == d], value[day == d]) for d in sorted(day.unique())]
        print(f'{name:16s} AP by day {np.round(by_day, 4)} mean {np.mean(by_day):.4f}')


if __name__ == '__main__':
    main()
