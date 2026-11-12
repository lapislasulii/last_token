"""Predict weak error reasons, then aggregate their probability into error score.

python tools/label_error_reasons.py
python tools/train_reason_logreg.py
python tools/train_reason_logreg.py --predict models/reason_coarse.joblib
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from train_logreg import ROOT, features
from temporal_features import CORE, add_temporal
from train_temporal import write_submission

# Explicit allowlist: no error/reason/human-box/OOF columns enter the model.
INPUT_COLUMNS = ['confidence'] + CORE


def model():
    return make_pipeline(StandardScaler(), LogisticRegression(C=1., max_iter=3000, random_state=42))


def error_probability(estimator, x):
    correct_index = list(estimator.classes_).index('correct')
    probabilities = estimator.predict_proba(x)
    assert np.allclose(probabilities.sum(axis=1), 1.)
    return np.clip(1. - probabilities[:, correct_index], 0., 1.)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT/'data/rascar-hack-case-1')
    parser.add_argument('--labels', type=Path, default=ROOT/'outputs/error_reasons/train_reason_labels.csv')
    parser.add_argument('--predict', type=Path)
    args = parser.parse_args()
    policy = json.loads((args.data/'scoring_policy.json').read_text())
    test = add_temporal(features(args.data, 'test', policy), args.data, 'test', policy)
    x_test = test[INPUT_COLUMNS]
    if args.predict:
        bundle = joblib.load(args.predict)
        assert bundle['input_columns'] == INPUT_COLUMNS
        score = error_probability(bundle['estimator'], x_test)
        destination = ROOT/'submissions'/f'{args.predict.stem}.csv'
        write_submission(args.data, test, score, destination)
        print(destination)
        return
    train = add_temporal(features(args.data, 'train', policy), args.data, 'train', policy)
    x = train[INPUT_COLUMNS]
    if not args.labels.exists():
        parser.error('Generate weak labels first: python tools/label_error_reasons.py')
    reasons = pd.read_csv(args.labels, dtype={'id':str}).set_index('id')
    assert reasons.index.is_unique and set(train.id) <= set(reasons.index)
    reasons = reasons.loc[train.id].reset_index()
    np.testing.assert_array_equal(reasons.error, train.error)
    assert reasons.matching_agrees.all(), 'Resolve matching disagreements before training reason targets'
    assert (reasons.reason.eq('correct').astype(int) == 1-train.error).all()
    days = sorted(train.day.unique())
    assert len(days) == 2 and set(train.id).isdisjoint(test.id)
    targets = {'coarse': reasons.reason.copy(), 'fine': reasons.reason.copy()}
    geometry = reasons.reason.eq('geometry_candidate')
    targets['fine'].loc[geometry] = 'geometry_' + reasons.loc[geometry, 'geometry_subtype']
    scores = {'baseline': 1-train.confidence.to_numpy()}
    binary = np.full(len(train), np.nan)
    for day in days:
        fit, valid = train.day != day, train.day == day
        estimator = model().fit(x.loc[fit], train.loc[fit, 'error'])
        binary[valid] = estimator.predict_proba(x.loc[valid])[:, 1]
    scores['binary_temporal'] = binary
    scores['binary_temporal_blend'] = .5*binary + .5*scores['baseline']
    unseen = []
    for directory in ['models', 'oof', 'reports']:
        (ROOT/directory).mkdir(exist_ok=True)
    for name, target in targets.items():
        values = np.full(len(train), np.nan)
        predicted_reasons = np.empty(len(train), dtype=object)
        for day in days:
            fit, valid = train.day != day, train.day == day
            assert set(train.loc[fit,'frame_id']).isdisjoint(train.loc[valid,'frame_id'])
            estimator = model().fit(x.loc[fit], target.loc[fit])
            assert estimator.named_steps['standardscaler'].n_samples_seen_ == int(fit.sum())
            assert estimator.named_steps['logisticregression'].n_iter_.max() < 3000
            values[valid] = error_probability(estimator, x.loc[valid])
            predicted_reasons[valid] = estimator.predict(x.loc[valid])
            absent = sorted(set(target.loc[valid])-set(target.loc[fit]))
            unseen.append({'target':name,'validation_day':day,'absent_in_training':', '.join(absent) or 'none'})
        scores[f'reason_{name}'] = values
        scores[f'reason_{name}_blend'] = .5*values + .5*scores['baseline']
        train[['id','day']].assign(reason_target=target, predicted_reason=predicted_reasons, score=values).to_csv(ROOT/f'oof/reason_{name}_diagnostics.csv', index=False)
        final = model().fit(x, target)
        bundle = {'estimator':final,'input_columns':INPUT_COLUMNS,'target':name,'aggregation':'1-P(correct)'}
        path = ROOT/f'models/reason_{name}.joblib'
        joblib.dump(bundle,path,compress=3)
        restored = joblib.load(path)
        test_score = error_probability(restored['estimator'],x_test)
        np.testing.assert_allclose(test_score,error_probability(final,x_test))
        write_submission(args.data,test,test_score,ROOT/f'submissions/reason_{name}.csv')
        write_submission(args.data,test,.5*test_score+.5*(1-test.confidence.to_numpy()),ROOT/f'submissions/reason_{name}_blend.csv')
    metrics = []
    for name, value in scores.items():
        assert np.isfinite(value).all() and ((value>=0)&(value<=1)).all()
        train[['id','day']].assign(score=value).to_csv(ROOT/f'oof/{name}.csv',index=False)
        row = {'method':name}
        for day in days:
            mask=train.day==day
            row[day]=average_precision_score(train.loc[mask,'error'],value[mask])
        row['mean']=np.mean([row[day] for day in days])
        metrics.append(row)
    metrics=pd.DataFrame(metrics).set_index('method')
    print(metrics.round(6).to_string())
    lines=['# Регрессия с обучением по причинам ошибок','',
           'Две цели: coarse — correct/geometry/class mismatch/duplicate/unresolved; fine — геометрия дополнительно разделена на 4 подтипа.',
           'Причины используются только как обучающие цели, а не входные признаки. Вход одинаков во всех моделях: confidence и четыре CORE-признака соседних/пересекающихся машинных боксов.',
           'C=1, StandardScaler fit только на обучающем дне. Два UTC-дня проверены по очереди. Итоговый score = 1 − P(correct). Смесь с baseline фиксирована 50/50. Подбора гиперпараметров в этом эксперименте нет.', '',
           '| Метод | '+' | '.join(metrics.columns)+' |','|---|'+'---:|'*len(metrics.columns)]
    for name,row in metrics.iterrows():
        lines.append('| '+name+' | '+' | '.join(f'{v:.6f}' for v in row)+' |')
    lines+=['','## Причины, отсутствующие на обучающем дне','']
    for row in unseen:
        lines.append(f'- {row["target"]}, проверка {row["validation_day"]}: {row["absent_in_training"]}.')
    lines+=['',
            'Отсутствующую при обучении причину классификатор не умеет отдельно предсказывать. Её примеры всё ещё могут получить высокий общий score ошибки через другие категории.',
            'Причины эвристические; корректное восстановление error не доказывает правильность названия причины. Редкие причины и неоднозначность ограничивают этот способ обучения.',
            'Все метрики — на тех же днях, уже использованных для выбора предыдущих моделей; это не независимое подтверждение и не Kaggle.',
            'Артефакты локально: models/reason_*.joblib; oof/reason_*.csv; submissions/reason_*.csv. Финальные модели обучены на всех 198 строках.',
            'Повторить: python tools/label_error_reasons.py && python tools/train_reason_logreg.py. Предсказание без ручных боксов: python tools/train_reason_logreg.py --predict models/reason_coarse.joblib.']
    (ROOT/'reports/reason_logreg_results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':
    main()
