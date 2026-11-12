"""Generate diagnostic weak reason labels for TRAIN only, using human geometry.

python tools/label_error_reasons.py
Never use these human-derived columns as test-time features.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
XY = ['x1', 'y1', 'x2', 'y2']


def overlap(a, b):
    a, b = np.asarray(a, float).reshape(-1, 4), np.asarray(b, float).reshape(-1, 4)
    intersection = np.maximum(0, np.minimum(a[:, None, 2:], b[None, :, 2:]) - np.maximum(a[:, None, :2], b[None, :, :2])).prod(axis=2)
    area_a = np.maximum(0, a[:, 2:] - a[:, :2]).prod(axis=1)
    area_b = np.maximum(0, b[:, 2:] - b[:, :2]).prod(axis=1)
    union = area_a[:, None] + area_b[None, :] - intersection
    return intersection / np.maximum(union, 1e-12), intersection, area_a, area_b


def match_boxes(iou, threshold):
    """Maximize valid pair count first, total IoU second; allow unmatched rows."""
    n, m = iou.shape
    if not n or not m:
        return {}
    # One extra match outweighs every possible difference in total IoU.
    bonus = min(n, m) + 1
    utility = np.zeros((n, m + n))
    utility[:, :m] = np.where(iou >= threshold, bonus + iou, -1.)
    rows, columns = linear_sum_assignment(utility, maximize=True)
    return {int(r): int(c) for r, c in zip(rows, columns) if c < m and iou[r, c] >= threshold}


def annotate_frame(predictions, humans, threshold=.6, geometry_threshold=.2):
    """Return one row per prediction, preserving official error and explaining evidence."""
    predictions, humans = predictions.reset_index(drop=True), humans.reset_index(drop=True)
    iou, intersection, pred_area, human_area = overlap(predictions[XY].to_numpy(), humans[XY].to_numpy())
    assigned = {}
    for cls in predictions.class_name.unique():
        pi = np.flatnonzero(predictions.class_name.to_numpy() == cls)
        hi = np.flatnonzero((humans.class_name.to_numpy() == cls) & humans.match_eligible.to_numpy())
        local = match_boxes(iou[np.ix_(pi, hi)], threshold)
        assigned.update({int(pi[p]): int(hi[h]) for p, h in local.items()})
    owner = {h: p for p, h in assigned.items()}
    results = []
    for i, row in predictions.iterrows():
        same = np.flatnonzero((humans.class_name.to_numpy() == row.class_name) & humans.match_eligible.to_numpy())
        other = np.flatnonzero(humans.class_name.to_numpy() != row.class_name)
        best = int(same[np.argmax(iou[i, same])]) if len(same) else None
        other_best = int(other[np.argmax(iou[i, other])]) if len(other) else None
        best_iou = float(iou[i, best]) if best is not None else 0.
        other_iou = float(iou[i, other_best]) if other_best is not None else 0.
        record = {'id': row.id, 'reason': 'correct', 'geometry_subtype': '',
                  'reconstructed_error': int(i not in assigned),
                  'matching_agrees': int(row.error) == int(i not in assigned),
                  'best_same_iou': best_iou, 'best_other_iou': other_iou,
                  'best_same_human_row': humans.iloc[best].human_row if best is not None else None,
                  'best_other_class': humans.iloc[other_best].class_name if other_best is not None else '',
                  'best_other_human_row': humans.iloc[other_best].human_row if other_best is not None else None,
                  'matched_human_row': humans.iloc[assigned[i]].human_row if i in assigned else None,
                  'competing_detection_id': '', 'needs_visual_review': False,
                  'borderline_iou': geometry_threshold <= best_iou < threshold and best_iou >= threshold-.05,
                  'competing_class_evidence': best_iou >= geometry_threshold and other_iou >= threshold}
        if best is not None:
            h = humans.iloc[best]
            record.update(pred_to_human_area=float(pred_area[i]/max(human_area[best], 1e-12)),
                          prediction_covered=float(intersection[i,best]/max(pred_area[i], 1e-12)),
                          human_covered=float(intersection[i,best]/max(human_area[best], 1e-12)))
            pc = (row[XY[:2]].to_numpy(float) + row[XY[2:]].to_numpy(float))/2
            hc = (humans[XY[:2]].to_numpy(float) + humans[XY[2:]].to_numpy(float))/2
            distances = np.linalg.norm(hc[same]-pc, axis=1)
            nearest = int(same[np.argmin(distances)])
            record['nearest_same_human_row'] = humans.iloc[nearest].human_row
            record['nearest_same_center_distance_px'] = float(distances.min())
            record['same_class_target_ambiguous'] = bool(((iou[i, same] >= geometry_threshold) & (iou[i, same] >= best_iou-.05)).sum() > 1)
        else:
            record.update(pred_to_human_area=np.nan, prediction_covered=np.nan, human_covered=np.nan,
                          nearest_same_human_row=None, nearest_same_center_distance_px=np.nan, same_class_target_ambiguous=False)
        if not record['matching_agrees']:
            record.update(reason='matching_disagreement', needs_visual_review=True)
        elif int(row.error) == 1:
            if best_iou >= threshold:
                # A label of error despite a valid same-class pair requires an occupied GT.
                if best in owner and owner[best] != i:
                    record.update(reason='duplicate_candidate', competing_detection_id=str(predictions.iloc[owner[best]].id))
                else:
                    record.update(reason='matching_disagreement', needs_visual_review=True)
            elif best_iou >= geometry_threshold:
                record['reason'] = 'geometry_candidate'
                ratio = record['pred_to_human_area']
                if record['prediction_covered'] >= .8 and ratio < .75:
                    record['geometry_subtype'] = 'too_small_or_partial'
                elif record['human_covered'] >= .8 and ratio > 4/3:
                    record['geometry_subtype'] = 'too_large'
                elif .75 <= ratio <= 4/3:
                    record['geometry_subtype'] = 'shift_or_shape'
                else:
                    record['geometry_subtype'] = 'mixed_geometry'
                record['needs_visual_review'] = bool(record['same_class_target_ambiguous'] or record['competing_class_evidence'] or record['borderline_iou'])
            elif other_iou >= threshold:
                record.update(reason='class_mismatch_candidate', needs_visual_review=True)
            else:
                record.update(reason='unresolved', needs_visual_review=True)
        results.append(record)
    return pd.DataFrame(results)


def markdown_table(df):
    lines = ['| '+' | '.join(map(str, df.columns))+' |', '|'+ '---|'*len(df.columns)]
    for row in df.itertuples(index=False, name=None):
        lines.append('| '+' | '.join(f'{x:.3f}' if isinstance(x, float) else str(x) for x in row)+' |')
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT/'data/rascar-hack-case-1')
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/error_reasons')
    parser.add_argument('--geometry-threshold', type=float, default=.2)
    args = parser.parse_args()
    data = args.data
    policy = json.loads((data/'scoring_policy.json').read_text())
    if not 0 < args.geometry_threshold < policy['correct_iou']:
        parser.error('geometry-threshold must lie between 0 and official IoU threshold')
    pred = pd.read_csv(data/'train.csv', dtype={'id':str})
    human = pd.read_csv(data/'human_boxes_train.csv')
    frames = pd.read_csv(data/'train_frames.csv')
    assert pred.id.is_unique and frames.frame_id.is_unique
    for df in [pred,human]:
        df['class_name'] = df.class_name.replace(policy['aliases'])
    human['human_row'] = np.arange(len(human))
    human['match_eligible'] = human.class_name.isin(policy['classes']) & human.evaluated_class.astype(str).str.lower().eq('true')
    human = human.merge(frames[['frame_id','image_width','image_height']],on='frame_id',validate='many_to_one')
    for column,center,extent,sign,size in [('x1','cx','w',-1,'image_width'),('x2','cx','w',1,'image_width'),('y1','cy','h',-1,'image_height'),('y2','cy','h',1,'image_height')]:
        human[column]=(human[center]+sign*human[extent]/2)*human[size]
    outputs=[]
    for fid, group in pred.groupby('frame_id'):
        outputs.append(annotate_frame(group,human[human.frame_id==fid],policy['correct_iou'],args.geometry_threshold))
    labels=pred.merge(pd.concat(outputs,ignore_index=True),on='id',validate='one_to_one')
    assert len(labels)==len(pred) and np.array_equal(labels.error,pred.error)
    labels=labels.merge(frames[['frame_id','timestamp']],on='frame_id',validate='many_to_one')
    labels['day']=pd.to_datetime(labels.timestamp,utc=True,format='mixed').dt.strftime('%Y-%m-%d')
    labels['in_test_scope']=~labels.class_name.isin(policy['test_excluded_classes'])
    labels['reason_source']='geometry_heuristic_v1'
    labels['baseline_score']=1-labels.confidence
    scoped=labels[labels.in_test_scope].copy()
    oof=ROOT/'oof/temporal_core_c1_blend.csv'
    if oof.exists():
        scores=pd.read_csv(oof,dtype={'id':str})[['id','score']].rename(columns={'score':'audit_score'})
        assert scores.id.is_unique and set(scores.id)==set(scoped.id)
        labels=labels.merge(scores,on='id',how='left',validate='one_to_one')
        scope_mask=labels.in_test_scope
        for col in ['baseline_score','audit_score']:
            labels.loc[scope_mask,col+'_queue_fraction']=labels.loc[scope_mask].groupby('day')[col].rank(ascending=False,method='average',pct=True)
        scoped=labels[labels.in_test_scope].copy()
    args.output.mkdir(parents=True,exist_ok=True)
    labels.to_csv(args.output/'train_reason_labels.csv',index=False)
    errors=scoped[scoped.error==1]
    unresolved=errors[errors.needs_visual_review].copy()
    if 'audit_score_queue_fraction' in unresolved:
        unresolved=unresolved.sort_values('audit_score_queue_fraction',ascending=False)
    unresolved.to_csv(args.output/'review_queue.csv',index=False)
    by_day=errors.groupby(['day','reason']).size().unstack(fill_value=0).reset_index()
    subtype=errors[errors.reason=='geometry_candidate'].groupby(['class_name','geometry_subtype']).size().rename('count').reset_index()
    text=['# Предполагаемые причины ошибок train','',
          'Автоматическая диагностическая разметка, не новые официальные метки. `error` не изменён.',
          f'Обработано {len(labels)} детекций; в области test {len(scoped)}, ошибок {len(errors)}.',
          f'Расхождений с восстановленным matching: {int((~labels.matching_agrees).sum())}.',
          '', '## Ошибки по дням', '',markdown_table(by_day),'','## Подтипы геометрии','',markdown_table(subtype),
          '', '## Правила и ограничения', '',
          '- Matching: тот же класс, IoU из policy; сначала максимальное число пар, затем сумма IoU. Добавлены фиктивные колонки для несопоставленных предсказаний.',
          '- duplicate_candidate: машинный бокс мог бы совпасть с ручным, но тот уже занят другим предсказанием; ID конкурента сохранён. Это конфликт сопоставления, не доказательство причины по изображению.',
          f'- geometry_candidate: лучший IoU с эталоном того же класса от {args.geometry_threshold:g} до {policy["correct_iou"]:g}. Целевой эталон всё ещё может быть выбран неверно; неоднозначность отмечается отдельно.',
          '- too_small_or_partial: ≥80% предсказания внутри эталона, площадь <75% эталона. too_large: ≥80% эталона внутри предсказания, площадь >4/3 эталона. shift_or_shape: отношение площадей от 0,75 до 4/3; остальные — mixed_geometry.',
          '- class_mismatch_candidate: IoU с другим классом ≥ официального порога, а с тем же классом ниже геометрического порога. Нужен просмотр: близость не доказывает неверный класс.',
          '- unresolved: геометрия не дала уверенного объяснения. Это могут быть ложные срабатывания, сильно смещённые/частичные боксы, ошибки класса или особенности эталона. Не называем всю группу false positive.',
          '- borderline_iou: до официального порога не хватает не более 0,05. Это флаг поверх причины, а не её замена.',
          '- best_same_human_row и nearest_same_human_row могут различаться: ближайший центр не обязательно соответствует нужному объекту. Номера human_row — с нуля по исходному human_boxes_train.csv.',
          '- matching_disagreement: восстановленный статус расходится с официальным. Такие случаи требуют разбора, официальный error остаётся прежним.',
          '', '## Использование', '',
          '`python tools/label_error_reasons.py` создаёт `outputs/error_reasons/train_reason_labels.csv` и `review_queue.csv`; изображения и индивидуальные метки остаются локальными.',
          'Колонки причин вычислены с использованием ручного эталона. Их нельзя передавать аудитору как доступные на test признаки. Можно использовать для диагностики, выбора экспериментов или как слабые цели обучения будущего классификатора причин.',
          'Если обучать классификатор причин, в каждом разбиении используются причины только обучающего дня. Его прогнозы на отложенном дне затем проверяются по официальному error и AP; причина остаётся предположительной.',
          'Прежде чем обучать CV-компонент, просмотреть очередь unresolved и проверить качество новых слабых меток на небольшом наборе train-примеров.']
    if 'audit_score_queue_fraction' in errors:
        diagnostic=errors.groupby('reason').agg(errors=('id','size'),baseline_median_queue=('baseline_score_queue_fraction','median'),audit_median_queue=('audit_score_queue_fraction','median')).reset_index()
        text+=['','## Где ошибки находятся в очереди OOF','',markdown_table(diagnostic),'',
               'Доля очереди измерена среди всех нестоловых детекций дня, затем агрегирована по причинам: ближе к 1 — проверяются позже. Это диагностическая статистика, не AP по причинам и не независимое доказательство прироста.']
    report=ROOT/'reports/error_reasons.md'
    report.write_text('\n'.join(text)+'\n',encoding='utf-8')
    print(by_day.to_string(index=False))
    print('Matching disagreements:',int((~labels.matching_agrees).sum()))
    print('Visual review queue:',len(unresolved))
    print('Saved:',args.output,report)


if __name__=='__main__':
    main()
