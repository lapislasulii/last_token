"""Демо аудитора сабмитов: одна команда, понятный отчёт.

    python audit_demo.py                 # сабмиты из submissions/ + восстановленные рецепты + типичные баги
    python audit_demo.py path/to/my.csv  # добавить свои файлы к демо

Консольный отчёт печатается сразу, HTML и JSON сохраняются в outputs/audit/
(каталог в .gitignore: в HTML есть превью кадров под NDA).
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'tools'))

from submission_auditor import (OUTPUT_DIR, SubmissionAuditor, format_comparison_text, format_text,  # noqa: E402
                                read_submission, save_reports, sha256)


def demo_cases(auditor: SubmissionAuditor):
    """Восстановленные из git рецепты и типичные ошибки последнего часа перед дедлайном."""
    ids = auditor.template.id
    frame = lambda scores: pd.DataFrame({'id': ids, 'score': np.asarray(scores, dtype=float)})  # noqa: E731
    recipes = {name: auditor.recipes[name][ids].to_numpy() for name in auditor.recipes}
    if 'ens3_1to1' in recipes:
        good, good_name = recipes['ens3_1to1'], 'ансамбль один к одному, без вето'
    else:
        good, good_name = 0.5 * auditor.test.loc[ids, 'mini'].to_numpy() + 0.5 * recipes['baseline'], 'смесь мини-модели и baseline'
    yield 'demo_leader_no_veto', f'восстановлено из git: {good_name} (признаков вето в git нет)', frame(good)

    if 'ens3_1to1' in recipes:
        # Вето смоделировано правилом только для проверки распознавания: настоящих признаков вето в git нет.
        sim = ((auditor.test.loc[ids, 'confidence'] < 0.6) & (auditor.test.loc[ids, 'votes_missing'] > 0)).to_numpy()
        yield 'demo_leader_sim_veto', 'имитация лидера: ансамбль один к одному + вето (вето смоделировано правилом)', \
            frame((good + sim) / 2)
        from tie_break import tie_break
        leader = pd.Series((good + sim) / 2, index=ids.values)
        broken, _ = tie_break(leader, 1 - auditor.test.loc[ids, 'iou_core'].clip(0, 1).set_axis(ids.values))
        yield 'demo_leader_tiebreak', 'тот же лидер после tools/tie_break.py (ничьи разбиты по iou_same ансамбля)', \
            frame(broken.loc[ids].to_numpy())
        yield 'demo_old_matching', 'не тот файл: ансамбль со старым сопоставлением «любое совпадение» + то же вето', \
            frame((recipes['ens3_any'] + sim) / 2)
        yield 'demo_single_detector', 'не тот файл: одиночный детектор без вето', frame(recipes['single_1to1'])

    yield 'demo_shifted_merge', 'баг merge: баллы сдвинуты на одну строку относительно id (формат проходит)', \
        frame(np.roll(good, 1))
    yield 'demo_inverted', 'перепутан смысл: score = 1 − правильный score', frame(1 - good)

    broken = frame(good).astype({'score': object}).drop(index=[0, 1, 2]).reset_index(drop=True)
    broken.loc[len(broken)] = [broken.id[5], 0.4]                         # дубль id
    broken.loc[10, 'score'] = 'nan'                                       # NaN после деления на ноль
    broken.loc[11, 'score'] = 1.3                                         # забыли обрезать в [0, 1]
    yield 'demo_broken_file', 'битый файл: потеряны строки, дубль, NaN, балл вне [0, 1]', broken

    yield 'demo_constant', 'все баллы одинаковые (упала модель, подставилось значение по умолчанию)', frame(np.full(len(ids), 0.5))
    cups = (auditor.test.loc[ids, 'class_name'] == 'cup').to_numpy()
    yield 'demo_cups_lost', 'баг фильтра: всем чашкам выставлен минимальный балл', frame(np.where(cups, 0.0, good))


def main() -> int:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    warnings.filterwarnings('ignore')
    print('Считаю признаки и обучаю мини-модель...')
    auditor = SubmissionAuditor()
    for note in auditor.notes:
        print(f'! {note}')
    v = auditor.validation
    print(f'\n=== Можно ли доверять второму мнению: {auditor.mode} ===')
    print('    (учимся на одном дне, проверяемся на другом; детекторы на train не видели проверочный день)')
    for day, d in v['days'].items():
        print(f'  {day}: AP мини-модели {d["ap_mini"]:.3f}  vs  baseline 1-confidence {d["ap_baseline"]:.3f}')
    print(f'  в топ-20% мини-модели ошибочны {v["error_rate_mini_top20"]:.0%} боксов (в среднем {v["error_rate_all"]:.0%})')
    if 'ap_person_mini' in v:
        print(f'  люди: AP {v["ap_person_mini"]:.3f} против {v["ap_person_baseline"]:.3f} у baseline')

    files = sorted((ROOT / 'submissions').glob('*.csv')) + [Path(p) for p in sys.argv[1:]]
    cases = [(f.stem, f'файл {f.name}', read_submission(f), sha256(f)) for f in files]
    cases += [(name, description, frame, None) for name, description, frame in demo_cases(auditor)]

    results = []
    for name, description, frame, digest in cases:
        result = auditor.audit(frame, name, top=5, digest=digest)
        results.append(result)
        print(f'\n# {description}')
        print(format_text(result, limit=2, queue=3 if name == 'demo_leader_sim_veto' else 0))

    frames = {name: frame for name, _, frame, _ in cases}
    pairs = [('demo_leader_tiebreak', 'demo_leader_sim_veto'), ('demo_leader_sim_veto', 'demo_old_matching'),
             ('demo_single_detector', 'demo_leader_no_veto')]
    comparisons = [auditor.compare(frames[a], frames[b], a, b, top=4) for a, b in pairs if a in frames and b in frames]
    for c in comparisons:
        print()
        print(format_comparison_text(c, limit=2))

    print('\n=== Итог ===')
    for r in results:
        ident = r.identity
        what = ident.get('recipe', '') + ('+вето' if ident.get('veto') else '') + ('+разбивка' if ident.get('tie_break') else '')
        print(f'  {r.verdict.upper():8} {r.name:24} {what:18} {"Public " + str(ident["kaggle_public"]) if ident.get("kaggle_public") else ""}')
    path = save_reports(results, auditor, OUTPUT_DIR, 'audit_demo', comparisons=comparisons)
    print(f'\nHTML-отчёт для жюри: {path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
