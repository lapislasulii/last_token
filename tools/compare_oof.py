"""Сравнение двух OOF-файлов и проверка заранее заданного критерия эксперимента.

Дополняет tools/evaluate_predictions.py (проверка файла, AP по дням и классам берутся оттуда) тем, чего там нет:
  - разница нового и опорного варианта по каждому дню и в среднем;
  - парный bootstrap по кадрам внутри дня (как в check_frame_stability.py, но для любых двух файлов);
  - вердикт по критерию в стиле задач команды: «среднее ≥ опорного + X и ни один день не хуже Y»;
  - доля боксов в ничьих у обоих файлов;
  - готовая markdown-таблица для reports/.

Запуск из корня репозитория:
  python tools/compare_oof.py oof/new.csv oof/redetect_strong_veto_veto.csv --min-gain 0.003 --max-day-drop 0.005
  python tools/compare_oof.py oof/new.csv baseline              # опорный вариант 1 − confidence
  python tools/compare_oof.py oof/new.csv oof/ref.csv --markdown reports/my_experiment_table.md

Код выхода: 0 — критерий пройден, 2 — не пройден, 1 — ошибка во входных файлах.
Bootstrap описывает чувствительность к составу известных кадров; новых дней он не создаёт.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))

from evaluate_predictions import DEFAULT_DATA, EvaluationError, evaluate, load_dataset, validate_prediction_file  # noqa: E402


def load_scores(spec: str, df: pd.DataFrame) -> tuple[pd.Series, str]:
    if spec == 'baseline':
        return pd.Series(1 - df.confidence.to_numpy(), index=df.id.astype(str)), 'baseline 1 − confidence'
    path = Path(spec)
    scores, _ = validate_prediction_file(path, df.id)
    return scores, path.stem


def tie_share(scores: pd.Series) -> tuple[int, int]:
    rounded = scores.round(12)
    return int(rounded.duplicated(keep=False).sum()), len(rounded)


def paired_bootstrap(df: pd.DataFrame, new: pd.Series, ref: pd.Series, repeats: int, seed: int) -> pd.DataFrame:
    """Перевыбор целых кадров с повторением внутри каждого дня; одинаковые кадры для обоих вариантов."""
    rng = np.random.default_rng(seed)
    days = sorted(df.day.unique())
    groups = {d: {f: g.index.to_numpy() for f, g in df[df.day == d].groupby('frame_id')} for d in days}
    y, a, b = df.error.to_numpy(), new.loc[df.id].to_numpy(), ref.loc[df.id].to_numpy()
    rows = []
    for _ in range(repeats):
        diffs = []
        for d in days:
            frames = list(groups[d])
            chosen = rng.choice(len(frames), len(frames), replace=True)
            idx = np.concatenate([groups[d][frames[i]] for i in chosen])
            if len(np.unique(y[idx])) < 2:
                break
            diffs.append(average_precision_score(y[idx], a[idx]) - average_precision_score(y[idx], b[idx]))
        if len(diffs) == len(days):
            rows.append([*diffs, float(np.mean(diffs))])
    return pd.DataFrame(rows, columns=[*days, 'mean'])


def compare(df: pd.DataFrame, new: pd.Series, ref: pd.Series, min_gain: float, max_day_drop: float,
            repeats: int = 2000, seed: int = 42) -> dict:
    a, b = evaluate(df, new), evaluate(df, ref)
    days = [{'day': x['day'], 'new': x['ap'], 'ref': y['ap'], 'delta': x['ap'] - y['ap'], 'boxes': x['n_boxes'],
             'errors': x['n_errors']} for x, y in zip(a['per_day'], b['per_day'])]
    classes = [{'class': x['class_name'], 'boxes': x['n_boxes'], 'new': x['ap'], 'ref': y['ap'], 'delta': x['ap'] - y['ap']}
               for x, y in zip(a['per_class'], b['per_class'])]
    mean_new, mean_ref = a['summary']['mean_ap'], b['summary']['mean_ap']
    gain = mean_new - mean_ref
    worst_drop = max(-d['delta'] for d in days)
    reasons = []
    if gain < min_gain - 1e-12:
        reasons.append(f'прирост среднего {gain:+.4f} меньше требуемого {min_gain:+.4f}')
    for d in days:
        if -d['delta'] > max_day_drop + 1e-12:
            reasons.append(f'день {d["day"]} хуже опорного на {-d["delta"]:.4f} (допустимо {max_day_drop:.4f})')
    boot = paired_bootstrap(df, new, ref, repeats, seed) if repeats else pd.DataFrame()
    result = {'days': days, 'classes': classes, 'mean_new': mean_new, 'mean_ref': mean_ref, 'gain': gain,
              'worst_day_drop': worst_drop, 'criterion': {'min_gain': min_gain, 'max_day_drop': max_day_drop},
              'passed': not reasons, 'reasons': reasons, 'ties_new': tie_share(new), 'ties_ref': tie_share(ref)}
    if len(boot):
        result['bootstrap'] = {'repeats': len(boot), 'q05': float(boot['mean'].quantile(.05)), 'median': float(boot['mean'].median()),
                               'q95': float(boot['mean'].quantile(.95)), 'positive_share': float((boot['mean'] > 0).mean()),
                               'positive_share_by_day': {d: float((boot[d] > 0).mean()) for d in boot.columns if d != 'mean'}}
    return result


def markdown(result: dict, new_name: str, ref_name: str) -> str:
    c = result['criterion']
    lines = [f'| Вариант | ' + ' | '.join(d['day'] for d in result['days']) + ' | Среднее |',
             '|---|' + '---:|' * (len(result['days']) + 1),
             f'| {ref_name} (опорный) | ' + ' | '.join(f'{d["ref"]:.4f}' for d in result['days']) + f' | {result["mean_ref"]:.4f} |',
             f'| **{new_name}** | ' + ' | '.join(f'**{d["new"]:.4f}**' for d in result['days']) + f' | **{result["mean_new"]:.4f}** |',
             f'| разница | ' + ' | '.join(f'{d["delta"]:+.4f}' for d in result['days']) + f' | {result["gain"]:+.4f} |', '',
             f'Критерий, заданный заранее: среднее ≥ опорного {c["min_gain"]:+.4f}, ни один день не хуже опорного больше чем на '
             f'{c["max_day_drop"]:.4f}. **{"Пройден" if result["passed"] else "Не пройден"}**'
             + ('' if result['passed'] else ': ' + '; '.join(result['reasons'])) + '.']
    if 'bootstrap' in result:
        b = result['bootstrap']
        lines.append(f'Парный bootstrap по кадрам ({b["repeats"]} повторов): разница среднего AP 5–95% от {b["q05"]:+.4f} '
                     f'до {b["q95"]:+.4f}, медиана {b["median"]:+.4f}, доля повторов с приростом {b["positive_share"]:.1%}. '
                     'Это чувствительность к составу известных кадров, не проверка на новых днях.')
    return '\n'.join(lines)


def print_result(result: dict, new_name: str, ref_name: str) -> None:
    print(f'=== {new_name}  против  {ref_name} ===')
    for d in result['days']:
        print(f'  {d["day"]}: {d["new"]:.4f} против {d["ref"]:.4f}  ({d["delta"]:+.4f}); боксов {d["boxes"]}, ошибок {d["errors"]}')
    print(f'  среднее: {result["mean_new"]:.4f} против {result["mean_ref"]:.4f}  ({result["gain"]:+.4f})')
    print('  по классам: ' + ', '.join(f'{c["class"]} {c["new"]:.3f} ({c["delta"]:+.3f})' for c in result['classes']))
    for label, (tied, total) in (('новый', result['ties_new']), ('опорный', result['ties_ref'])):
        print(f'  ничьи, {label}: {tied} из {total} боксов')
    if 'bootstrap' in result:
        b = result['bootstrap']
        print(f'  bootstrap по кадрам ({b["repeats"]}): разница среднего 5–95% [{b["q05"]:+.4f}; {b["q95"]:+.4f}], '
              f'прирост в {b["positive_share"]:.1%} повторов; по дням ' +
              ', '.join(f'{d} {s:.0%}' for d, s in b['positive_share_by_day'].items()))
    c = result['criterion']
    print(f'  КРИТЕРИЙ (среднее ≥ опорного {c["min_gain"]:+.4f}, день не хуже на > {c["max_day_drop"]:.4f}): '
          + ('ПРОЙДЕН' if result['passed'] else 'НЕ ПРОЙДЕН — ' + '; '.join(result['reasons'])))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('new', help='OOF-файл id,score нового варианта')
    parser.add_argument('ref', help='опорный OOF-файл или слово baseline')
    parser.add_argument('--min-gain', type=float, default=0.0, help='минимальный прирост среднего AP (по умолчанию 0)')
    parser.add_argument('--max-day-drop', type=float, default=0.0, help='допустимое ухудшение любого дня (по умолчанию 0)')
    parser.add_argument('--bootstrap', type=int, default=2000, help='повторов bootstrap по кадрам, 0 — не считать')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--markdown', type=Path, help='записать таблицу и вывод в markdown-файл')
    parser.add_argument('--data', type=Path, default=DEFAULT_DATA)
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    try:
        df = load_dataset(args.data)
        new, new_name = load_scores(args.new, df)
        ref, ref_name = load_scores(args.ref, df)
    except EvaluationError as exc:
        print(f'ОШИБКА: {exc}', file=sys.stderr)
        return 1
    result = compare(df, new, ref, args.min_gain, args.max_day_drop, args.bootstrap, args.seed)
    print_result(result, new_name, ref_name)
    table = markdown(result, new_name, ref_name)
    print('\n' + table)
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(table + '\n', encoding='utf-8')
        print(f'\nMarkdown: {args.markdown}')
    return 0 if result['passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
