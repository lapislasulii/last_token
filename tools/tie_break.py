"""Разбивка ничьих в сабмите по непрерывному согласию детекторов-контролёров. Без обучения.

Балл лидера собран из грубых частей (D: 0, 1/3, 2/3, 1; вето: 0/1; confidence с двумя знаками),
поэтому две трети очереди стоят в группах с одинаковым баллом, и порядок внутри них для AP случаен.
Инструмент упорядочивает боксы ВНУТРИ каждой группы по 1 − iou_same (среднее по участникам ансамбля:
чем хуже лучший IoU с нашими детекторами, тем раньше), не меняя порядок между группами:

    новый = score · (1 − eps) + eps · (1 − iou_same),   eps меньше трети минимального зазора между группами.

Запуск из корня репозитория:
  python tools/tie_break.py submissions/redetect_strong_ens-base-s1-s2_1to1_veto_veto.csv
      -> submissions/..._tiebreak.csv + отчёт; на Kaggle ничего не отправляется
  python tools/tie_break.py --oof          # эффект на отложенных днях train (одиночный детектор; с вето, если оно есть локально)
  python tools/tie_break.py FILE --members base s1 s2 --eps 0.001

Предсказания участников берутся из outputs/redetect_strong/pred_all*.csv (в git), признаки train — из oof_features*.csv.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parent
sys.path.insert(0, str(TOOLS))

STRONG = ROOT / 'outputs' / 'redetect_strong'
DATA = ROOT / 'data' / 'rascar-hack-case-1'
DEFAULT_MEMBERS = ['base', 's1', 's2']     # состав лидера (Public 0.90211)
DEFAULT_EPS = 1e-3


# ---------------------------------------------------------------- участники

def discover_members(strong: Path = STRONG) -> dict[str, Path]:
    """pred_all.csv → base, pred_all_s1.csv → s1, pred_all_flip.csv → base_flip, pred_all_s1_flip.csv → s1_flip."""
    found = {}
    for path in sorted(strong.glob('pred_all*.csv')):
        rest = path.stem[len('pred_all'):]                 # '', '_s1', '_flip', '_s1_flip', '_st_strict'
        if rest and not rest.startswith('_'):
            continue
        flip = rest.endswith('_flip')
        rest = rest[:-len('_flip')] if flip else rest
        found[(rest.lstrip('_') or 'base') + ('_flip' if flip else '')] = path
    return found


def member_agreement(boxes: pd.DataFrame, members: list[str], strong: Path = STRONG) -> pd.DataFrame:
    """iou_same и пара «один к одному» каждого участника для исходных боксов (функция из redetect.py)."""
    import redetect
    files = discover_members(strong)
    missing = [m for m in members if m not in files]
    if missing:
        raise FileNotFoundError(f'нет предсказаний участников {missing} в {strong}; есть: {sorted(files)}')
    cols = ['id', 'frame_id', 'class_name', 'x1', 'y1', 'x2', 'y2']
    out = boxes[['id']].copy()
    for m in members:
        feats = redetect.agreement_features(boxes[cols], pd.read_csv(files[m]))
        feats = boxes[['id']].merge(feats, on='id', how='left', validate='one_to_one')
        out[f'iou_{m}'] = feats.iou_same.to_numpy()
        out[f'matched_{m}'] = feats.matched.astype(bool).to_numpy()
    out['iou_same'] = out[[f'iou_{m}' for m in members]].mean(axis=1)
    out['D'] = 1 - out[[f'matched_{m}' for m in members]].mean(axis=1)
    return out


def test_boxes(data: Path = DATA) -> pd.DataFrame:
    import redetect
    return redetect.load('test.csv', dtype={'id': str})


# ---------------------------------------------------------------- разбивка

def tie_stats(score: pd.Series) -> dict:
    counts = score.round(12).value_counts()
    return {'unique': int(len(counts)), 'rows': int(len(score)), 'in_ties': int(counts[counts > 1].sum()),
            'groups': int((counts > 1).sum()), 'largest': int(counts.max())}


def safe_eps(score: pd.Series, eps: float | None = None) -> float:
    """eps, при котором порядок между группами гарантированно не меняется."""
    levels = np.unique(score.round(12).to_numpy())
    gap = float(np.diff(levels).min()) if len(levels) > 1 else 1.0
    limit = gap / 3
    if eps is None:
        return min(DEFAULT_EPS, limit)
    if eps >= limit:
        raise ValueError(f'eps={eps} слишком велик: минимальный зазор между группами {gap:.6f}, нужно eps < {limit:.6f}')
    return eps


def tie_break(score: pd.Series, key: pd.Series, eps: float | None = None) -> tuple[pd.Series, float]:
    """key в [0, 1], больше = подозрительнее. Возвращает новый балл в [0, 1] и использованный eps."""
    key = key.reindex(score.index)
    if key.isna().any() or not key.between(0, 1).all():
        raise ValueError('ключ разбивки должен быть задан для всех строк и лежать в [0, 1]')
    eps = safe_eps(score, eps)
    return score * (1 - eps) + eps * key, eps


def order_preserved(old: pd.Series, new: pd.Series) -> bool:
    """Каждый бокс группы с большим исходным баллом остаётся выше любого бокса группы с меньшим."""
    frame = pd.DataFrame({'old': old.round(12), 'new': new})
    by_level = frame.groupby('old').new.agg(['min', 'max']).sort_index()
    return bool((by_level['max'].to_numpy()[:-1] < by_level['min'].to_numpy()[1:]).all())


# ---------------------------------------------------------------- отложенные дни

def oof_report(strong: Path = STRONG, veto_path: Path = ROOT / 'outputs/expert_veto/train_features.csv') -> list[dict]:
    """Формула лидера и та же формула с разбивкой на отложенных днях train."""
    from sklearn.metrics import average_precision_score as ap
    tables = {'одиночный детектор base': strong / 'oof_features.csv'}
    tables.update({f'ансамбль {p.stem.split("_ens-")[1].replace("-", "+")}': p for p in sorted(strong.glob('oof_features_ens-*.csv'))})
    veto = pd.read_csv(veto_path, dtype={'id': str})[['id', 'veto']] if veto_path.exists() else None
    rows = []
    for name, path in tables.items():
        df = pd.read_csv(path, dtype={'id': str, 'day': str})
        if 'error' not in df:
            continue
        disagree = df.D if 'D' in df else (~df.matched.astype(bool)).astype(float) if 'matched' in df else (df.iou_same < .6).astype(float)
        base = (1 - df.confidence + disagree) / 2
        variants = {'формула': base}
        if veto is not None:
            v = df[['id']].merge(veto, on='id', how='left').veto.fillna(0).to_numpy()
            variants['формула + вето'] = (base + v) / 2
        for label, score in variants.items():
            score = pd.Series(score.to_numpy(), index=df.index)
            broken, eps = tie_break(score, 1 - df.iou_same.clip(0, 1))
            for kind, s in (('как есть', score), ('с разбивкой ничьих', broken)):
                days = {d: float(ap(df.error[df.day == d], s[df.day == d])) for d in sorted(df.day.unique())}
                person = df.class_name == 'person'
                rows.append({'source': name, 'variant': label, 'kind': kind, **days, 'mean': float(np.mean(list(days.values()))),
                             'person': float(ap(df.error[person], s[person])), 'in_ties': tie_stats(s)['in_ties'], 'rows': len(s)})
    return rows


def print_oof(rows: list[dict]) -> None:
    print('Отложенные дни train: тот же балл как есть и с разбивкой ничьих по 1 − iou_same.')
    print('Признаки детектора на train получены моделями, не видевшими проверочный день; разбивка ничего не обучает.')
    last = None
    for r in rows:
        if (r['source'], r['variant']) != last:
            print(f"\n{r['source']} — {r['variant']}:")
            last = (r['source'], r['variant'])
        days = '  '.join(f'{k} {v:.4f}' for k, v in r.items() if k.startswith('20'))
        print(f"  {r['kind']:20s} {days}  среднее {r['mean']:.4f}  люди {r['person']:.3f}  в ничьих {r['in_ties']}/{r['rows']}")
    if not any(r['variant'].endswith('вето') for r in rows):
        print('\nВето не найдено (outputs/expert_veto/train_features.csv): строки «+ вето» появятся у того, кто его считал.')


# ---------------------------------------------------------------- CLI

def run_file(path: Path, members: list[str], eps: float | None, out: Path | None) -> Path:
    from submission_auditor import check_format, read_submission
    boxes = test_boxes()
    findings, clean = check_format(read_submission(path), pd.read_csv(DATA / 'sample_submission.csv', dtype={'id': str}).id)
    errors = [f for f in findings if f.level == 'error']
    if errors:
        raise SystemExit('файл не проходит проверку формата: ' + '; '.join(f.message for f in errors[:5]))
    agreement = member_agreement(boxes, members).set_index('id')
    score = clean.loc[agreement.index]
    new, eps = tie_break(score, 1 - agreement.iou_same.clip(0, 1), eps)
    assert order_preserved(score, new), 'порядок между группами изменился'
    assert np.isfinite(new).all() and new.between(0, 1).all()
    before, after = tie_stats(score), tie_stats(new)
    out = out or path.with_name(path.stem + '_tiebreak.csv')
    template = pd.read_csv(DATA / 'sample_submission.csv', dtype={'id': str})
    template[['id']].assign(score=new.loc[template.id].to_numpy()).to_csv(out, index=False)

    classes = boxes.set_index('id').class_name.loc[score.index]
    moved = (score.rank(ascending=False, method='first') - new.rank(ascending=False, method='first')).abs()
    print(f'Файл: {path.name}; участники для разбивки: {"+".join(members)}; eps = {eps:g}')
    print(f'  ничьи до:    {before["in_ties"]} боксов из {before["rows"]} в {before["groups"]} группах, крупнейшая {before["largest"]}')
    print(f'  ничьи после: {after["in_ties"]} боксов в {after["groups"]} группах, крупнейшая {after["largest"]}')
    print('  порядок между исходными группами сохранён: да; новый балл в [0, 1]: да')
    print(f'  мест сдвинулось внутри групп: медиана {moved.median():.0f}, максимум {moved.max():.0f}')
    print('  по классам, боксов в ничьих до → после: ' + ', '.join(
        f'{c} {int(score[classes == c].round(12).duplicated(keep=False).sum())}→{int(new[classes == c].duplicated(keep=False).sum())}'
        for c in classes.value_counts().index))
    print(f'Записан {out}. На Kaggle не отправлялся.')
    print(f'Проверка: python tools/submission_auditor.py {out} --vs {path}')
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('file', nargs='?', type=Path, help='сабмит id,score')
    parser.add_argument('--members', nargs='+', default=DEFAULT_MEMBERS, help='участники для iou_same (по умолчанию base s1 s2)')
    parser.add_argument('--eps', type=float, help=f'вес разбивки (по умолчанию {DEFAULT_EPS} или меньше, если зазоры малы)')
    parser.add_argument('--out', type=Path, help='куда записать (по умолчанию <имя>_tiebreak.csv рядом)')
    parser.add_argument('--oof', action='store_true', help='показать эффект на отложенных днях train')
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if not args.file and not args.oof:
        parser.error('укажите файл сабмита или --oof')
    if args.oof:
        print_oof(oof_report())
    if args.file:
        if args.oof:
            print()
        run_file(args.file, args.members, args.eps, args.out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
