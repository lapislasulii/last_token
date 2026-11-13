"""Демо аудитора сабмитов: одна команда, понятный отчёт.

    python audit_demo.py                 # реальные сабмиты из submissions/ + примеры типичных багов
    python audit_demo.py path/to/my.csv  # добавить свои файлы к демо

Консольный отчёт печатается сразу, HTML и JSON сохраняются в outputs/audit/
(каталог в .gitignore: в HTML есть превью кадров под NDA).
"""
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'tools'))

from submission_auditor import (OUTPUT_DIR, SubmissionAuditor, format_comparison_text, format_text,  # noqa: E402
                                read_submission, save_reports)


def demo_cases(auditor: SubmissionAuditor):
    """Типичные ошибки последнего часа перед дедлайном. Основа — честный сабмит без ошибок."""
    template = auditor.template[['id']].copy()
    good = template.assign(score=0.5 * auditor.test.loc[template.id, 'mini'].to_numpy()
                           + 0.5 * (1 - auditor.test.loc[template.id, 'confidence'].to_numpy()))
    yield 'demo_good_blend', 'корректный сабмит: смесь мини-модели и baseline', good

    shifted = good.copy()
    shifted['score'] = np.roll(shifted.score.to_numpy(), 1)
    yield 'demo_shifted_merge', 'баг merge: баллы сдвинуты на одну строку относительно id (формат проходит)', shifted

    yield 'demo_inverted', 'перепутан смысл: score = confidence («уверен» вместо «подозрителен»)', \
        template.assign(score=auditor.test.loc[template.id, 'confidence'].to_numpy())

    broken = good.copy().astype({'score': object})
    broken = broken.drop(index=[0, 1, 2]).reset_index(drop=True)            # потеряны 3 id
    broken.loc[len(broken)] = [broken.id[5], 0.4]                           # дубль id
    broken.loc[10, 'score'] = 'nan'                                         # NaN после деления на ноль
    broken.loc[11, 'score'] = 1.3                                           # забыли обрезать в [0, 1]
    yield 'demo_broken_file', 'битый файл: потеряны строки, дубль, NaN, балл вне [0, 1]', broken

    yield 'demo_constant', 'все баллы одинаковые (упала модель, подставилось значение по умолчанию)', \
        template.assign(score=0.5)

    cups = (auditor.test.loc[template.id, 'class_name'] == 'cup').to_numpy()
    yield 'demo_cups_lost', 'баг фильтра: всем чашкам выставлен минимальный балл (класс выпал из очереди)', \
        good.assign(score=np.where(cups, 0.0, good.score))


def main() -> int:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    warnings.filterwarnings('ignore')
    print('Обучаю мини-модель и считаю признаки...')
    auditor = SubmissionAuditor()
    v = auditor.validation
    print('\n=== Можно ли доверять второму мнению (учимся на одном дне, проверяемся на другом) ===')
    for day, d in v['days'].items():
        print(f'  {day}: AP мини-модели {d["ap_mini"]:.3f}  vs  baseline 1-confidence {d["ap_baseline"]:.3f}')
    print(f'  в топ-20% мини-модели ошибочны {v["error_rate_mini_top20"]:.0%} боксов '
          f'(в среднем {v["error_rate_all"]:.0%}, в её спокойных 20% — {v["error_rate_mini_bottom20"]:.0%})')

    files = sorted((ROOT / 'submissions').glob('*.csv')) + [Path(p) for p in sys.argv[1:]]
    cases = [(f.stem, f'файл команды {f.relative_to(ROOT) if f.is_relative_to(ROOT) else f}', read_submission(f))
             for f in files]
    cases += list(demo_cases(auditor))

    results = []
    for name, description, frame in cases:
        result = auditor.audit(frame, name, top=5)
        results.append(result)
        print(f'\n# {description}')
        print(format_text(result, limit=3))

    # Сравнение очередей помогает выбрать два финальных сабмита.
    frames = {name: frame for name, _, frame in cases}
    pairs = [(a, b) for a, b in [('logreg_geometry', 'logreg_class'), ('logreg_confidence', 'logreg_baseline'),
                                 ('demo_good_blend', 'logreg_class')] if a in frames and b in frames]
    comparisons = [auditor.compare(frames[a], frames[b], a, b, top=5) for a, b in pairs]
    for c in comparisons:
        print()
        print(format_comparison_text(c, limit=2))

    print('\n=== Итог ===')
    for r in results:
        print(f'  {r.verdict.upper():8} {r.name}')
    for c in comparisons:
        print(f'  {"VS":8} {c.new} против {c.ref}: Spearman {c.stats["spearman"]:+.3f}')
    path = save_reports(results, auditor, OUTPUT_DIR, 'audit_demo', comparisons=comparisons)
    print(f'\nHTML-отчёт для жюри: {path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
