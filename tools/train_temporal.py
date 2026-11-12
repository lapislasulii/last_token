"""Fixed small experiment set; train, evaluate, and optionally publish selected models.

python tools/train_temporal.py
python tools/train_temporal.py --publish NAME [NAME ...]
python tools/train_temporal.py --predict model_releases/NAME.joblib
"""
import argparse
import hashlib
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from train_logreg import ROOT, GEOMETRY, features
from temporal_features import CORE, DETAIL, add_temporal

# Declared before evaluating. No automated hyperparameter search.
CONFIGS = {
    'temporal_core_c01': (CORE, False, .1, 'lr'),
    'temporal_core_c1': (CORE, False, 1., 'lr'),
    'temporal_class_c01': (CORE, True, .1, 'lr'),
    'temporal_class_c1': (CORE, True, 1., 'lr'),
    'temporal_full_c01': (CORE + DETAIL + GEOMETRY, True, .1, 'lr'),
    'temporal_full_c1': (CORE + DETAIL + GEOMETRY, True, 1., 'lr'),
    'temporal_trees': (CORE + DETAIL + GEOMETRY, True, 1., 'trees'),
}


def make_model(config):
    columns, categorical, c, kind = config
    steps = [('numeric', StandardScaler(), ['confidence'] + columns)]
    if categorical:
        steps.append(('class', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['class_name']))
    learner = (LogisticRegression(C=c, max_iter=3000, random_state=42) if kind == 'lr' else
               ExtraTreesClassifier(n_estimators=300, max_depth=3, min_samples_leaf=8, max_features=1., random_state=42, n_jobs=1))
    return Pipeline([('features', ColumnTransformer(steps)), ('model', learner)])


def write_submission(data, rows, scores, path):
    template = pd.read_csv(data / 'sample_submission.csv', dtype={'id': str})
    assert template.id.is_unique and rows.id.is_unique and set(template.id) == set(rows.id)
    assert np.isfinite(scores).all() and ((scores >= 0) & (scores <= 1)).all()
    result = template[['id']].merge(rows[['id']].assign(score=scores), on='id', validate='one_to_one')
    path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(path, index=False)


def score_bundle(bundle, rows):
    scores = bundle['estimator'].predict_proba(rows)[:, 1]
    return bundle['model_weight'] * scores + (1 - bundle['model_weight']) * (1 - rows.confidence.to_numpy())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT / 'data/rascar-hack-case-1')
    parser.add_argument('--publish', nargs='+', default=[], choices=list(CONFIGS) + [n+'_blend' for n in CONFIGS])
    parser.add_argument('--predict', type=Path)
    args = parser.parse_args()
    policy = json.loads((args.data / 'scoring_policy.json').read_text())
    test = add_temporal(features(args.data, 'test', policy), args.data, 'test', policy)
    if args.predict:
        bundle = joblib.load(args.predict)
        out = ROOT / 'submissions' / (args.predict.stem + '.csv')
        write_submission(args.data, test, score_bundle(bundle, test), out)
        print(out)
        return
    train = add_temporal(features(args.data, 'train', policy), args.data, 'train', policy)
    days = sorted(train.day.unique())
    assert len(days) == 2 and set(train.id).isdisjoint(test.id)
    for name in ['oof', 'models', 'reports']:
        (ROOT / name).mkdir(exist_ok=True)
    scores = {'baseline': 1 - train.confidence.to_numpy()}
    for name, config in CONFIGS.items():
        values = np.full(len(train), np.nan)
        for day in days:
            fit, valid = train[train.day != day], train[train.day == day]
            assert set(fit.frame_id).isdisjoint(valid.frame_id)
            estimator = make_model(config).fit(fit, fit.error)
            assert estimator.named_steps['features'].named_transformers_['numeric'].n_samples_seen_ == len(fit)
            values[valid.index] = estimator.predict_proba(valid)[:, 1]
        assert np.isfinite(values).all()
        scores[name] = values
        # Fixed equal-weight probability blend, evaluated as a separate candidate.
        scores[name+'_blend'] = .5 * values + .5 * scores['baseline']
    records = []
    for name, values in scores.items():
        train[['id', 'day']].assign(score=values).to_csv(ROOT / f'oof/{name}.csv', index=False)
        ap = {day: float(average_precision_score(train.loc[train.day == day, 'error'], values[train.day == day])) for day in days}
        records.append({'method': name, **ap, 'mean': np.mean(list(ap.values())), 'worst': min(ap.values())})
    metrics = pd.DataFrame(records).set_index('method')
    metrics['improves_both'] = (metrics[days] > metrics.loc['baseline', days]).all(axis=1)
    print(metrics.round(6).to_string())
    lines = ['# Temporal experiments', '',
             'Seven fixed configurations, each also evaluated with a fixed 50/50 baseline blend (14 candidates).',
             'Two UTC-day folds; table excluded; scaling/encoding fit only on training fold.',
             'Only supplied stream/context predictions used; no human boxes, pool, test labels or Kaggle feedback.',
             'Neighbors across UTC-day boundaries excluded. Missing neighbors explicitly encoded in full variants.',
             'CORE: maximum/minimum neighbor IoU, neighbor confidence difference, same-frame same-class overlap.',
             'FULL: CORE + geometry + missing flags, actual time gaps, area/center changes and same-class count.', '',
             '| Method | ' + ' | '.join(metrics.columns) + ' |', '|---|'+'---|'*len(metrics.columns)]
    for name, row in metrics.iterrows():
        lines.append('| '+name+' | '+' | '.join(str(v) if isinstance(v, (bool, np.bool_)) else f'{v:.6f}' for v in row)+' |')
    lines += ['', 'Mean is mean daily AP. Selecting among candidates on these two days introduces selection bias.',
              'Gains are exploratory, not independent validation or Kaggle scores. Prior logistic experiments also used these days.',
              'Inspect per-day performance, not only the mean. Final published estimators are refitted on all 198 labeled non-table boxes.']
    (ROOT / 'reports/temporal_results.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    for name in args.publish:
        base = name.removesuffix('_blend')
        estimator = make_model(CONFIGS[base]).fit(train, train.error)
        bundle = {'estimator': estimator, 'model_weight': .5 if name.endswith('_blend') else 1., 'name': name}
        destination = ROOT / 'model_releases' / (name+'.joblib')
        destination.parent.mkdir(exist_ok=True)
        joblib.dump(bundle, destination, compress=3)
        restored = joblib.load(destination)
        np.testing.assert_allclose(score_bundle(restored, test), score_bundle(bundle, test))
        write_submission(args.data, test, score_bundle(restored, test), ROOT / 'submissions' / (name+'.csv'))
        metadata = {'name': name, 'validation': metrics.loc[name].to_dict(), 'training_boxes': len(train),
                    'python': platform.python_version(), 'sklearn': sklearn.__version__, 'pandas': pd.__version__,
                    'numpy': np.__version__, 'joblib': joblib.__version__,
                    'sha256': hashlib.sha256(destination.read_bytes()).hexdigest(),
                    'training_inputs_sha256': {f: hashlib.sha256((args.data/f).read_bytes()).hexdigest() for f in
                        ['train.csv', 'train_frames.csv', 'train_context.csv', 'train_stream_predictions.csv', 'scoring_policy.json']}}
        destination.with_suffix('.json').write_text(json.dumps(metadata, indent=2)+'\n')
        print('Published locally:', destination, destination.stat().st_size, 'bytes')


if __name__ == '__main__':
    main()
