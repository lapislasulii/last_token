"""Reproduce two-day logistic-regression ablations. Run from any directory."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
GEOMETRY = ['width', 'height', 'area', 'aspect', 'cx', 'cy', 'bottom']


def features(data, split, policy):
    rows = pd.read_csv(data / f'{split}.csv', dtype={'id': str})
    frames = pd.read_csv(data / f'{split}_frames.csv')
    assert rows.id.is_unique
    rows['class_name'] = rows.class_name.replace(policy['aliases'])
    rows = rows[~rows.class_name.isin(policy['test_excluded_classes'])].copy()
    rows = rows.merge(frames[['frame_id', 'timestamp', 'image_width', 'image_height']], on='frame_id', how='left', validate='many_to_one')
    rows['day'] = pd.to_datetime(rows.timestamp, utc=True, format='mixed').dt.strftime('%Y-%m-%d')
    rows['width'] = (rows.x2 - rows.x1) / rows.image_width
    rows['height'] = (rows.y2 - rows.y1) / rows.image_height
    assert (rows[['width', 'height']] > 0).all().all()
    rows['area'] = rows.width * rows.height
    rows['aspect'] = (rows.x2 - rows.x1) / (rows.y2 - rows.y1)
    rows['cx'] = (rows.x1 + rows.x2) / (2 * rows.image_width)
    rows['cy'] = (rows.y1 + rows.y2) / (2 * rows.image_height)
    rows['bottom'] = rows.y2 / rows.image_height
    assert rows.day.notna().all()
    assert np.isfinite(rows[['confidence'] + GEOMETRY]).all().all()
    return rows


def model(variant):
    numeric = ['confidence'] + (GEOMETRY if variant == 'geometry' else [])
    transformers = [('numeric', StandardScaler(), numeric)]
    if variant != 'confidence':
        transformers.append(('class', OneHotEncoder(handle_unknown='ignore'), ['class_name']))
    return Pipeline([('features', ColumnTransformer(transformers)),
                     ('model', LogisticRegression(C=1.0, max_iter=2000, random_state=42))])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT / 'data/rascar-hack-case-1')
    args = parser.parse_args()
    policy = json.loads((args.data / 'scoring_policy.json').read_text())
    train, test = [features(args.data, s, policy) for s in ['train', 'test']]
    days = sorted(train.day.unique())
    assert len(days) == 2, 'Expected exactly two labeled days'
    assert set(train.id).isdisjoint(test.id)
    results = []
    for directory in ['oof', 'models', 'submissions', 'reports']:
        (ROOT / directory).mkdir(exist_ok=True)
    for variant in ['baseline', 'confidence', 'class', 'geometry']:
        predictions = []
        for day in days:
            fit, valid = train[train.day != day], train[train.day == day]
            assert set(fit.frame_id).isdisjoint(valid.frame_id)
            if variant == 'baseline':
                score = 1 - valid.confidence.to_numpy()
            else:
                estimator = model(variant).fit(fit, fit.error)
                score = estimator.predict_proba(valid)[:, 1]
                # Ensure scaling was fitted only on the training fold.
                scaler = estimator.named_steps['features'].named_transformers_['numeric']
                assert scaler.n_samples_seen_ == len(fit)
                assert estimator.named_steps['model'].n_iter_.max() < 2000
            assert np.isfinite(score).all() and ((score >= 0) & (score <= 1)).all()
            predictions.append(valid[['id', 'day']].assign(score=score))
            results.append({'method': variant, 'day': day, 'boxes': len(valid), 'errors': int(valid.error.sum()), 'AP': average_precision_score(valid.error, score)})
        oof = pd.concat(predictions)
        assert oof.id.is_unique and set(oof.id) == set(train.id)
        oof.to_csv(ROOT / f'oof/logreg_{variant}.csv', index=False)
        if variant == 'baseline':
            score = 1 - test.confidence.to_numpy()
        else:
            estimator = model(variant).fit(train, train.error)
            joblib.dump(estimator, ROOT / f'models/logreg_{variant}.joblib')
            restored = joblib.load(ROOT / f'models/logreg_{variant}.joblib')
            score = restored.predict_proba(test)[:, 1]
            np.testing.assert_allclose(score, estimator.predict_proba(test)[:, 1])
        template = pd.read_csv(args.data / 'sample_submission.csv', dtype={'id': str})
        assert template.id.is_unique and set(template.id) == set(test.id)
        submission = template[['id']].merge(test[['id']].assign(score=score), on='id', validate='one_to_one')
        assert submission.score.between(0, 1).all() and np.isfinite(submission.score).all()
        submission.to_csv(ROOT / f'submissions/logreg_{variant}.csv', index=False)
    metrics = pd.DataFrame(results)
    pivot = metrics.pivot(index='method', columns='day', values='AP').reindex(['baseline', 'confidence', 'class', 'geometry'])
    pivot['mean'] = pivot.mean(axis=1)
    pivot['worst'] = pivot[days].min(axis=1)
    print(pivot.round(6).to_string())
    lines = ['# Logistic regression: two-day validation', '',
             'Fixed C=1.0; no hyperparameter search. Scaling and class encoding fitted inside each training fold.',
             'Table excluded. Each day is predicted using only labels from the other day. No machine pool or manual-box features.', '',
             '| Method | ' + ' | '.join(pivot.columns) + ' |', '|---|' + '---:|' * len(pivot.columns)]
    for name, row in pivot.iterrows():
        lines.append('| ' + name + ' | ' + ' | '.join(f'{v:.6f}' for v in row) + ' |')
    lines += ['', 'confidence: confidence only; class: + one-hot class; geometry: + normalized size, area, center, bottom and pixel aspect ratio.',
              'Mean is mean daily AP, not pooled AP. These small two-day estimates are not Kaggle scores.',
              'OOF: oof/logreg_*.csv. Models: models/logreg_*.joblib. Test predictions: submissions/logreg_*.csv.',
              'Final models refitted on all non-table train rows; each submission is validated against template IDs.',
              f'Environment: sklearn {sklearn.__version__}, pandas {pd.__version__}, numpy {np.__version__}.',
              f'Train: {len(train)} boxes. Test: {len(test)} boxes.']
    (ROOT / 'reports/logreg_results.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
