"""Аудитор сабмитов основной системы: мини-модель как второе мнение.

Проверяет любой файл id,score перед отправкой на Kaggle:
  1. жёсткие правила формата (error);
  2. второе мнение мини-модели — логистической регрессии на 5 числах (warning/info);
  3. эвристики, которые объясняют словами, почему бокс подозрителен (и служат запасным
     вариантом, если мини-модель обучить не удалось).

Запуск из корня репозитория:
  python tools/submission_auditor.py submissions/my.csv
  python tools/submission_auditor.py submissions/my.csv other.csv --top 15

Код основной системы не меняется: признаки берутся из tools/train_logreg.py и
tools/temporal_features.py.
"""
from __future__ import annotations

import argparse
import html
import json
import os
import sys
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parent
sys.path.insert(0, str(TOOLS))

DEFAULT_DATA = ROOT / 'data' / 'rascar-hack-case-1'
OUTPUT_DIR = ROOT / 'outputs' / 'audit'   # каталог в .gitignore: в отчёте превью кадров под NDA

# 5 входов мини-модели: тот же набор, что у temporal_core_c1.
MINI_FEATURES = ['confidence', 'neighbor_iou_max', 'neighbor_iou_min', 'neighbor_conf_delta', 'same_frame_iou']

# Пороги расхождения рангов (доля очереди: 0 = начало, 1 = конец).
MINI_TOP = 0.10         # мини-модель считает бокс одним из 10% самых подозрительных
MAIN_LOW = 0.50         # а основная система поставила его во вторую половину очереди
MINI_CALM = 0.10        # мини-модель уверена, что бокс в порядке (последние 10%)
MAIN_HIGH = 0.10        # а основная система поставила его в первые 10%
DISAGREE_SHARE = 0.05   # доля сильных расхождений, после которой сабмит получает warning
BASELINE_CORR = 0.995   # ранговая корреляция с 1-confidence: «модель ничего не добавила»
UNRELATED_CORR = 0.30   # ниже — очередь почти не связана со вторым мнением (у моделей команды > 0.8)


@dataclass
class Finding:
    level: str                  # error | warning | info
    rule: str
    message: str
    id: str | None = None
    frame_id: str | None = None
    class_name: str | None = None
    reasons: list[str] = field(default_factory=list)
    main_rank: int | None = None
    mini_rank: int | None = None
    ref_rank: int | None = None


@dataclass
class Comparison:
    new: str
    ref: str
    notes: list[str]
    stats: dict
    movers: list[Finding]

    def to_dict(self):
        return {'new': self.new, 'ref': self.ref, 'notes': self.notes, 'stats': self.stats,
                'movers': [asdict(f) for f in self.movers]}


@dataclass
class AuditResult:
    name: str
    verdict: str                # ok | warning | error
    findings: list[Finding]
    stats: dict

    def to_dict(self):
        return {'name': self.name, 'verdict': self.verdict, 'stats': self.stats,
                'findings': [asdict(f) for f in self.findings]}


# ---------------------------------------------------------------- правила формата

def check_format(sub: pd.DataFrame, expected_ids) -> tuple[list[Finding], pd.Series]:
    """Жёсткие правила. Возвращает находки и корректные баллы, индексированные id."""
    findings: list[Finding] = []
    expected = pd.Index(pd.Series(expected_ids, dtype=str))
    for column in ('id', 'score'):
        if column not in sub.columns:
            findings.append(Finding('error', 'columns', f'нет обязательной колонки «{column}»'))
    if findings:
        return findings, pd.Series(dtype=float)

    ids = sub['id'].astype(str).reset_index(drop=True)
    raw = sub['score'].reset_index(drop=True)
    score = pd.to_numeric(raw, errors='coerce')

    if len(sub) != len(expected):
        findings.append(Finding('error', 'row_count', f'строк {len(sub)}, а в шаблоне {len(expected)}'))
    for value in ids[ids.duplicated()].unique()[:10]:
        findings.append(Finding('error', 'duplicate_id', f'id встречается в {int((ids == value).sum())} строках', id=value))
    missing = expected.difference(ids)
    for value in missing[:10]:
        findings.append(Finding('error', 'missing_id', 'id из шаблона отсутствует в сабмите', id=value))
    if len(missing) > 10:
        findings.append(Finding('error', 'missing_id', f'и ещё {len(missing) - 10} отсутствующих id'))
    for value in pd.Index(ids).difference(expected)[:10]:
        findings.append(Finding('error', 'extra_id', 'id нет в шаблоне', id=value))

    values = score.to_numpy(dtype=float)
    bad_number = ~np.isfinite(values)
    for i in np.flatnonzero(bad_number)[:10]:
        findings.append(Finding('error', 'not_finite', f'score = «{raw.iloc[i]}» — не конечное число', id=ids.iloc[i]))
    out_of_range = ~bad_number & ((np.nan_to_num(values) < 0) | (np.nan_to_num(values) > 1))
    for i in np.flatnonzero(out_of_range)[:10]:
        findings.append(Finding('error', 'out_of_range', f'score = {values[i]:g} вне [0, 1]', id=ids.iloc[i]))

    valid = ~bad_number & ~out_of_range & ~ids.duplicated(keep=False).to_numpy() & ids.isin(expected).to_numpy()
    clean = pd.Series(values[valid], index=ids.to_numpy()[valid], dtype=float)
    if len(clean) > 1 and clean.nunique() == 1:
        findings.append(Finding('error', 'constant', f'все баллы равны {clean.iloc[0]:g}: очередь не упорядочена'))
    elif len(clean) > 20 and clean.nunique() < 0.1 * len(clean):
        findings.append(Finding('warning', 'few_unique',
                                f'всего {clean.nunique()} разных баллов на {len(clean)} строк: много ничьих в очереди'))
    return findings, clean


# ---------------------------------------------------------------- данные и мини-модель

def load_rows(data: Path, split: str) -> pd.DataFrame:
    from temporal_features import add_temporal
    from train_logreg import features
    policy = json.loads((data / 'scoring_policy.json').read_text(encoding='utf-8'))
    rows = add_temporal(features(data, split, policy), data, split, policy)
    frames = pd.read_csv(data / f'{split}_frames.csv', usecols=['frame_id', 'image_path'])
    return rows.merge(frames, on='frame_id', how='left', validate='many_to_one')


def make_mini_model():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    return Pipeline([('scale', StandardScaler()),
                     ('model', LogisticRegression(C=1.0, max_iter=3000, random_state=42))])


def heuristics(row) -> list[tuple[str, str]]:
    """Эвристики: (короткое имя правила, объяснение человеческими словами)."""
    found = []
    if row.same_frame_iou >= 0.5:
        found.append(('дубль на кадре', f'дубль: на кадре есть ещё {row.class_name} с IoU {row.same_frame_iou:.2f}'))
    if row.confidence < 0.5:
        found.append(('низкий confidence', f'низкий confidence детектора ({row.confidence:.2f})'))
    if not (row.previous_missing and row.next_missing) and row.neighbor_iou_max < 0.3:
        found.append(('нет опоры во времени', 'нет опоры во времени: на соседних кадрах похожего бокса нет '
                                               f'(лучший IoU {row.neighbor_iou_max:.2f})'))
    if row.class_name == 'person' and row.height < 0.2:
        found.append(('низкий бокс человека', f'низкий бокс человека ({row.height:.0%} высоты кадра): '
                                               'возможно, обрезан торс за столом'))
    if row.x1 < 2 or row.x2 > row.image_width - 2:
        found.append(('бокс у края кадра', 'бокс упирается в край кадра'))
    if row.area < 0.0008:
        found.append(('крошечный бокс', f'крошечный бокс ({row.area:.2%} площади кадра)'))
    return found


def explain(row) -> list[str]:
    return [text for _, text in heuristics(row)]


def heuristic_score(rows: pd.DataFrame) -> np.ndarray:
    """Запасной вариант без обучения: 1-confidence + штраф за каждое сработавшее правило."""
    flags = rows.apply(heuristics, axis=1).map(len).to_numpy()
    return np.clip(0.7 * (1 - rows.confidence.to_numpy()) + 0.1 * flags, 0, 1)


def validate_mini_model(train: pd.DataFrame) -> dict:
    """Честная проверка мини-модели: учимся на одном дне, проверяемся на другом."""
    from sklearn.metrics import average_precision_score
    result = {'days': {}}
    oof = pd.Series(np.nan, index=train.index)
    for day in sorted(train.day.unique()):
        fit, hold = train[train.day != day], train[train.day == day]
        if fit.empty or fit.error.nunique() < 2 or hold.error.nunique() < 2:
            continue
        oof[hold.index] = make_mini_model().fit(fit[MINI_FEATURES], fit.error).predict_proba(hold[MINI_FEATURES])[:, 1]
        result['days'][day] = {
            'boxes': len(hold), 'errors': int(hold.error.sum()),
            'ap_mini': float(average_precision_score(hold.error, oof[hold.index])),
            'ap_baseline': float(average_precision_score(hold.error, 1 - hold.confidence)),
        }
    if result['days']:
        result['ap_mini_mean'] = float(np.mean([d['ap_mini'] for d in result['days'].values()]))
        result['ap_baseline_mean'] = float(np.mean([d['ap_baseline'] for d in result['days'].values()]))
    scored = train.assign(mini=oof).dropna(subset=['mini'])
    if len(scored):
        result['error_rate_all'] = float(scored.error.mean())
        result['error_rate_mini_top20'] = float(scored[scored.mini.rank(ascending=False, pct=True) <= 0.2].error.mean())
        result['error_rate_mini_bottom20'] = float(scored[scored.mini.rank(pct=True) <= 0.2].error.mean())
    # Точность каждой эвристики на train: как часто сработавшее правило указывает на настоящую ошибку.
    rules = {}
    for found, error in zip(train.apply(heuristics, axis=1), train.error):
        for key, _ in found:
            hit = rules.setdefault(key, [0, 0])
            hit[0] += 1
            hit[1] += int(error)
    result['rules'] = {k: {'fired': n, 'error_rate': e / n}
                       for k, (n, e) in sorted(rules.items(), key=lambda x: -x[1][0])}
    return result


class SubmissionAuditor:
    """Обученная мини-модель + данные теста. Создаётся один раз и проверяет сколько угодно сабмитов."""

    def __init__(self, data: Path = DEFAULT_DATA, force_heuristics: bool = False):
        self.data = data
        self.template = pd.read_csv(data / 'sample_submission.csv', dtype={'id': str})
        self.train = load_rows(data, 'train')
        self.test = load_rows(data, 'test').set_index('id')
        self.validation = validate_mini_model(self.train)
        self.model = None
        try:
            if force_heuristics:
                raise RuntimeError('эвристики выбраны вручную')
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                self.model = make_mini_model().fit(self.train[MINI_FEATURES], self.train.error)
            self.test['mini'] = self.model.predict_proba(self.test[MINI_FEATURES])[:, 1]
            self.mode = 'мини-модель (логистическая регрессия, 5 признаков)'
        except Exception as exc:  # запасной вариант — правила без обучения
            self.model = None
            self.test['mini'] = heuristic_score(self.test)
            self.mode = f'только эвристики ({exc})'
        self.test['reasons'] = self.test.apply(explain, axis=1)

    def weights(self) -> dict:
        if self.model is None:
            return {}
        return dict(zip(MINI_FEATURES, map(float, self.model.named_steps['model'].coef_[0])))

    def audit(self, sub: pd.DataFrame, name: str, top: int = 10) -> AuditResult:
        findings, clean = check_format(sub, self.template.id)
        stats = {'rows': int(len(sub)), 'valid_rows': int(len(clean)), 'mode': self.mode}
        rows = self.test.loc[self.test.index.intersection(clean.index)].copy()
        if len(rows) > 20 and clean.nunique() > 1:
            findings += self._second_opinion(rows.assign(score=clean[rows.index]), stats, top)
        levels = {f.level for f in findings}
        verdict = 'error' if 'error' in levels else 'warning' if 'warning' in levels else 'ok'
        order = {'error': 0, 'warning': 1, 'info': 2}
        findings.sort(key=lambda f: (order[f.level], f.rule if f.level == 'info' else ''))
        return AuditResult(name, verdict, findings, stats)

    def _second_opinion(self, rows: pd.DataFrame, stats: dict, top: int) -> list[Finding]:
        findings = []
        n = len(rows)
        main = rows.score.rank(ascending=False)
        mini = rows.mini.rank(ascending=False)
        corr_mini = float(rows.score.rank().corr(rows.mini.rank()))
        corr_base = float(rows.score.rank().corr((1 - rows.confidence).rank()))
        k = max(1, int(0.2 * n))
        overlap = len(set(main.nsmallest(k).index) & set(mini.nsmallest(k).index)) / k
        stats.update(spearman_vs_mini=corr_mini, spearman_vs_baseline=corr_base, top20_overlap_with_mini=overlap)

        if corr_mini < 0:
            findings.append(Finding('warning', 'inverted',
                                    f'очередь направлена против мини-модели (Spearman {corr_mini:+.2f}): похоже, '
                                    'большой score означает «бокс верный», а не «проверить первым»'))
        elif corr_mini < UNRELATED_CORR:
            findings.append(Finding('warning', 'unrelated',
                                    f'очередь почти не связана с мини-моделью (Spearman {corr_mini:+.2f}); у всех рабочих '
                                    'моделей команды > +0.8. Типичная причина — баллы сдвинуты относительно id при merge'))
        if corr_base > BASELINE_CORR:
            findings.append(Finding('warning', 'same_as_baseline',
                                    f'порядок почти совпадает с 1-confidence (Spearman {corr_base:.3f}): '
                                    'основная модель ничего не добавила к baseline'))

        missed = rows[(mini / n <= MINI_TOP) & (main / n > MAIN_LOW)]
        overflag = rows[(mini / n > 1 - MINI_CALM) & (main / n <= MAIN_HIGH)]
        stats.update(missed_by_main=len(missed), overflagged_by_main=len(overflag))
        stats['by_class'] = self._by_class(rows, main, mini, set(missed.index) | set(overflag.index))
        for c in stats['by_class']:
            if c['boxes'] >= 10 and c['mini_top20'] >= 0.3 and c['main_top20'] < c['mini_top20'] / 3:
                findings.append(Finding('warning', 'class_pushed_down',
                                        f'класс {c["class"]} отодвинут в конец очереди: в топ-20% системы '
                                        f'{c["main_top20"]:.0%} его боксов, у мини-модели {c["mini_top20"]:.0%}',
                                        class_name=c['class']))
        share = (len(missed) + len(overflag)) / n
        if share > DISAGREE_SHARE:
            findings.append(Finding('warning', 'disagreement',
                                    f'сильных расхождений с мини-моделью {len(missed) + len(overflag)} из {n} '
                                    f'({share:.0%}) — больше порога {DISAGREE_SHARE:.0%}'))
        for rid, row in missed.sort_values('mini', ascending=False).head(top).iterrows():
            findings.append(self._detection(rid, row, main, mini, 'missed',
                                            'мини-модель считает бокс одним из самых подозрительных, а в очереди он далеко'))
        for rid, row in overflag.sort_values('mini').head(top).iterrows():
            findings.append(self._detection(rid, row, main, mini, 'overflagged',
                                            'бокс в начале очереди, хотя мини-модель и правила не видят проблем'))
        return findings

    @staticmethod
    def _by_class(rows, main, mini, disagreements) -> list[dict]:
        """Разбивка по классам: где система и мини-модель видят проблемы по-разному."""
        n = len(rows)
        top_main, top_mini = main / n <= 0.2, mini / n <= 0.2
        table = []
        for cls, group in rows.groupby('class_name'):
            idx = group.index
            table.append({'class': cls, 'boxes': len(group),
                          'main_top20': float(top_main[idx].mean()), 'mini_top20': float(top_mini[idx].mean()),
                          'mean_rank_main': float(main[idx].mean() / n), 'mean_rank_mini': float(mini[idx].mean() / n),
                          'disagreements': int(sum(i in disagreements for i in idx))})
        return sorted(table, key=lambda c: -c['boxes'])

    @staticmethod
    def _detection(rid, row, main, mini, rule, message, ref=None):
        reasons = list(row.reasons) or ['явных эвристических причин нет — стоит посмотреть кадр глазами']
        return Finding('info', rule, message, id=rid, frame_id=row.frame_id, class_name=row.class_name,
                       reasons=reasons, main_rank=int(round(main[rid])), mini_rank=int(round(mini[rid])),
                       ref_rank=int(round(ref[rid])) if ref is not None else None)

    # ------------------------------------------------------------ сравнение двух сабмитов
    def compare(self, new: pd.DataFrame, ref: pd.DataFrame, new_name: str, ref_name: str, top: int = 8) -> Comparison:
        """Чем новая очередь отличается от опорной (например, лучшей на Public)."""
        (new_errors, a), (ref_errors, b) = check_format(new, self.template.id), check_format(ref, self.template.id)
        notes = []
        for name, errors in ((new_name, new_errors), (ref_name, ref_errors)):
            if any(f.level == 'error' for f in errors):
                notes.append(f'«{name}» не проходит проверку формата — сравнение по общим корректным строкам')
        common = self.test.index.intersection(a.index).intersection(b.index)
        if len(common) < 20 or a[common].nunique() < 2 or b[common].nunique() < 2:
            return Comparison(new_name, ref_name, notes + ['слишком мало корректных строк для сравнения'], {}, [])
        rows = self.test.loc[common]
        n = len(rows)
        rank_new = a[common].rank(ascending=False)
        rank_ref = b[common].rank(ascending=False)
        rank_mini = rows.mini.rank(ascending=False)
        k = max(1, int(0.2 * n))
        top_new, top_ref = set(rank_new.nsmallest(k).index), set(rank_ref.nsmallest(k).index)
        spearman = float(rank_new.corr(rank_ref))
        shift = rank_ref - rank_new                     # > 0: в новой очереди бокс поднялся ближе к началу
        stats = {'rows': n, 'spearman': spearman, 'top20_overlap': len(top_new & top_ref) / k,
                 'entered_top20': len(top_new - top_ref), 'left_top20': len(top_ref - top_new),
                 'median_abs_shift': float(shift.abs().median()),
                 'new_vs_mini': float(rank_new.corr(rank_mini)), 'ref_vs_mini': float(rank_ref.corr(rank_mini))}
        by_class = []
        for cls, group in rows.groupby('class_name'):
            idx = group.index
            by_class.append({'class': cls, 'boxes': len(group),
                             'new_top20': float(rank_new[idx].le(k).mean()), 'ref_top20': float(rank_ref[idx].le(k).mean()),
                             'mean_shift': float(shift[idx].mean())})
        stats['by_class'] = sorted(by_class, key=lambda c: -c['boxes'])

        if spearman > 0.98:
            notes.append(f'очереди почти одинаковые (Spearman {spearman:.3f}): как второй финальный сабмит '
                         'такая пара почти не страхует от ошибки выбора')
        elif spearman < 0.6:
            notes.append(f'очереди сильно различаются (Spearman {spearman:.2f}): разнообразная пара для двух '
                         'финальных слотов, если обе сильны на Public')
        else:
            notes.append(f'очереди похожи, но не совпадают (Spearman {spearman:.2f}); '
                         f'в топ-20% сменилось {stats["entered_top20"]} из {k} боксов')
        moved = max((c for c in stats['by_class'] if c['boxes'] >= 10), key=lambda c: abs(c['mean_shift']), default=None)
        if moved and abs(moved['mean_shift']) > 0.05 * n:
            direction = 'ближе к началу' if moved['mean_shift'] > 0 else 'ближе к концу'
            notes.append(f'сильнее всего сдвинулся класс {moved["class"]}: в среднем на {abs(moved["mean_shift"]):.0f} '
                         f'мест {direction} очереди')

        movers = []
        for rid in shift.nlargest(top).index:
            movers.append(self._detection(rid, rows.loc[rid], rank_new, rank_mini, 'moved_up',
                                          f'поднялся на {shift[rid]:.0f} мест', rank_ref))
        for rid in shift.nsmallest(top).index:
            movers.append(self._detection(rid, rows.loc[rid], rank_new, rank_mini, 'moved_down',
                                          f'опустился на {-shift[rid]:.0f} мест', rank_ref))
        return Comparison(new_name, ref_name, notes, stats, movers)

    def box(self, rid: str) -> dict:
        row = self.test.loc[rid]
        return {k: float(row[k]) for k in ('x1', 'y1', 'x2', 'y2', 'image_width', 'image_height')} | {
            'image_path': str(row.image_path)}


# ---------------------------------------------------------------- отчёты

LABEL = {'ok': 'OK', 'warning': 'WARNING', 'error': 'ERROR'}


def format_text(result: AuditResult, limit: int = 6) -> str:
    s = result.stats
    lines = [f'=== Аудит «{result.name}»: {LABEL[result.verdict]} ===',
             f"строк {s['rows']}, корректных {s['valid_rows']}; второе мнение: {s['mode']}"]
    if 'spearman_vs_mini' in s:
        lines.append(f"согласие с мини-моделью: Spearman {s['spearman_vs_mini']:+.2f}, "
                     f"общий топ-20% {s['top20_overlap_with_mini']:.0%}; с baseline: {s['spearman_vs_baseline']:+.3f}")
        lines.append(f"пропущено подозрительных: {s['missed_by_main']}, лишних в начале очереди: {s['overflagged_by_main']}")
    if s.get('by_class'):
        lines.append('  по классам (доля боксов класса в топ-20% очереди: система / мини-модель; расхождений):')
        lines += [f"    {c['class']:12} {c['boxes']:4} боксов  {c['main_top20']:4.0%} / {c['mini_top20']:4.0%}  "
                  f"{c['disagreements']}" for c in s['by_class']]
    shown = 0
    for f in result.findings:
        if f.level == 'info':
            if shown >= limit:
                continue
            shown += 1
        where = f' id={f.id}' if f.id else ''
        rank = f' {f.class_name}, место {f.main_rank} у системы / {f.mini_rank} у мини-модели.' if f.main_rank else ''
        lines.append(f'  {f.level.upper():7} {f.rule:16}{where}{rank} {f.message}')
        lines += [f'{"":26}- {reason}' for reason in f.reasons]
    hidden = sum(f.level == 'info' for f in result.findings) - shown
    if hidden > 0:
        lines.append(f'  ... остальные находки уровня info ({hidden}) — в HTML-отчёте')
    return '\n'.join(lines)


def format_comparison_text(c: Comparison, limit: int = 3) -> str:
    lines = [f'=== Сравнение «{c.new}» с «{c.ref}» ===']
    s = c.stats
    if s:
        lines.append(f"Spearman {s['spearman']:+.3f}; общий топ-20% {s['top20_overlap']:.0%} "
                     f"(вошли {s['entered_top20']}, вышли {s['left_top20']}); медианный сдвиг {s['median_abs_shift']:.0f} мест")
        lines.append(f"согласие с мини-моделью: новый {s['new_vs_mini']:+.2f}, опорный {s['ref_vs_mini']:+.2f}")
        lines.append('  по классам (доля в топ-20%: новый / опорный; средний сдвиг, + = ближе к началу):')
        lines += [f"    {k['class']:12} {k['boxes']:4} боксов  {k['new_top20']:4.0%} / {k['ref_top20']:4.0%}  "
                  f"{k['mean_shift']:+6.1f}" for k in s['by_class']]
    lines += [f'  ВЫВОД   {note}' for note in c.notes]
    for rule in ('moved_up', 'moved_down'):
        for f in [m for m in c.movers if m.rule == rule][:limit]:
            lines.append(f'  {rule:10} id={f.id} {f.class_name}: место {f.ref_rank} -> {f.main_rank} '
                         f'(мини-модель {f.mini_rank}); {"; ".join(f.reasons)}')
    return '\n'.join(lines)


def _class_table(rows: list[dict], comparison: bool) -> str:
    if comparison:
        head = '<th>Класс</th><th>Боксов</th><th>В топ-20%: новый</th><th>опорный</th><th>Средний сдвиг</th>'
        body = ''.join(f"<tr><td>{html.escape(c['class'])}</td><td>{c['boxes']}</td><td>{c['new_top20']:.0%}</td>"
                       f"<td>{c['ref_top20']:.0%}</td><td>{c['mean_shift']:+.1f}</td></tr>" for c in rows)
    else:
        head = '<th>Класс</th><th>Боксов</th><th>В топ-20%: система</th><th>мини-модель</th><th>Расхождений</th>'
        body = ''.join(f"<tr><td>{html.escape(c['class'])}</td><td>{c['boxes']}</td><td>{c['main_top20']:.0%}</td>"
                       f"<td>{c['mini_top20']:.0%}</td><td>{c['disagreements']}</td></tr>" for c in rows)
    return f'<div class="scroll"><table><tr>{head}</tr>{body}</table></div>'


CSS = """
:root{--bg:#fbfaf7;--fg:#1d1d1b;--muted:#6b6a65;--card:#fff;--line:#e4e1d8;--ok:#2f7d4f;--warn:#a66a00;--err:#b3261e;--info:#3b5ba5}
@media (prefers-color-scheme:dark){:root{--bg:#161614;--fg:#ecebe6;--muted:#a3a19a;--card:#21211e;--line:#34332f;--ok:#2f8f58;--warn:#b07400;--err:#c2392f;--info:#4a68b5}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1100px;margin:0 auto;padding:24px 16px 64px}h1{font-size:26px;margin:0 0 4px}h2{font-size:18px;margin:32px 0 10px}
.muted{color:var(--muted)}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px;margin:16px 0}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px}.card b{display:block;font-size:22px}
.badge{display:inline-block;padding:2px 10px;border-radius:99px;font-weight:600;font-size:13px;color:#fff}
.ok{background:var(--ok)}.warning{background:var(--warn)}.error{background:var(--err)}.info{background:var(--info)}
.scroll{overflow-x:auto}table{width:100%;border-collapse:collapse;background:var(--card);border:1px solid var(--line)}
td,th{padding:8px 10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}th{font-size:13px;color:var(--muted)}
.det{display:grid;grid-template-columns:minmax(0,320px) 1fr;gap:14px;background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px;margin:10px 0}
.det svg{width:100%;height:auto;border-radius:6px;background:var(--line)}.det ul{margin:6px 0 0;padding-left:18px}code{font-size:13px;word-break:break-all}
@media (max-width:640px){.det{grid-template-columns:1fr}}
"""


def format_html(results: list[AuditResult], auditor: SubmissionAuditor, out_dir: Path,
                comparisons: list[Comparison] = ()) -> str:
    v = auditor.validation
    esc = html.escape
    parts = ['<!doctype html><html lang="ru"><head><meta charset="utf-8">'
             '<meta name="viewport" content="width=device-width,initial-scale=1">'
             f'<title>Аудит сабмитов</title><style>{CSS}</style></head><body><main>',
             '<h1>Аудит сабмитов перед отправкой</h1>',
             '<p class="muted">Мини-модель (логистическая регрессия на 5 числах) и простые правила проверяют очередь '
             'основной системы. Код основной системы не менялся.</p>',
             '<h2>Сводка</h2><div class="scroll"><table><tr><th>Сабмит</th><th>Вердикт</th><th>Ошибок формата</th>'
             '<th>Spearman с мини-моделью</th><th>Пропущено / лишних</th></tr>']
    for r in results:
        s = r.stats
        corr = f"{s['spearman_vs_mini']:+.2f}" if 'spearman_vs_mini' in s else '—'
        dis = f"{s['missed_by_main']} / {s['overflagged_by_main']}" if 'missed_by_main' in s else '—'
        parts.append(f'<tr><td><a href="#{esc(r.name)}">{esc(r.name)}</a></td>'
                     f'<td><span class="badge {r.verdict}">{r.verdict}</span></td>'
                     f'<td>{sum(f.level == "error" for f in r.findings)}</td><td>{corr}</td><td>{dis}</td></tr>')
    parts.append('</table></div>')

    parts.append('<h2>Можно ли доверять второму мнению</h2>'
                 '<p class="muted">Честная проверка на train: учимся на одном дне, проверяемся на другом.</p><div class="grid">')
    for day, d in v.get('days', {}).items():
        parts.append(f'<div class="card"><span class="muted">AP мини-модели, {day}</span><b>{d["ap_mini"]:.3f}</b>'
                     f'<span class="muted">baseline {d["ap_baseline"]:.3f}</span></div>')
    if 'error_rate_mini_top20' in v:
        parts.append(f'<div class="card"><span class="muted">Ошибок в её топ-20%</span><b>{v["error_rate_mini_top20"]:.0%}</b>'
                     f'<span class="muted">в среднем {v["error_rate_all"]:.0%}</span></div>'
                     f'<div class="card"><span class="muted">Ошибок в её спокойных 20%</span>'
                     f'<b>{v["error_rate_mini_bottom20"]:.0%}</b></div>')
    parts.append('</div><div class="scroll"><table><tr><th>Эвристика</th><th>Сработала на train</th><th>Из них ошибок</th></tr>')
    for rule, d in v.get('rules', {}).items():
        parts.append(f'<tr><td>{esc(rule)}</td><td>{d["fired"]}</td><td>{d["error_rate"]:.0%}</td></tr>')
    parts.append('</table></div>')
    if auditor.weights():
        parts.append('<p class="muted">Веса мини-модели (после масштабирования): ' + ', '.join(
            f'<code>{k}</code> {w:+.2f}' for k, w in auditor.weights().items()) + '</p>')

    for r in results:
        parts.append(f'<h2 id="{esc(r.name)}">{esc(r.name)} <span class="badge {r.verdict}">{r.verdict}</span></h2>')
        plain = [f for f in r.findings if f.level != 'info']
        if plain:
            parts.append('<div class="scroll"><table><tr><th>Уровень</th><th>Правило</th><th>Где</th><th>Причина</th></tr>')
            parts += [f'<tr><td><span class="badge {f.level}">{f.level}</span></td><td><code>{f.rule}</code></td>'
                      f'<td><code>{esc(f.id or "весь файл")}</code></td><td>{esc(f.message)}</td></tr>' for f in plain]
            parts.append('</table></div>')
        elif r.verdict == 'ok':
            parts.append('<p>Формат корректен, очередь согласуется с мини-моделью. Можно отправлять.</p>')
        if r.stats.get('by_class'):
            parts.append('<p class="muted">По классам: какая доля боксов класса попала в первые 20% очереди.</p>')
            parts.append(_class_table(r.stats['by_class'], comparison=False))
        parts += [_detection_card(f, auditor, out_dir) for f in r.findings if f.level == 'info']

    if comparisons:
        parts.append('<h2>Сравнение сабмитов</h2><p class="muted">Помогает выбрать два финальных сабмита: '
                     'насколько новая очередь отличается от опорной и какие боксы переехали сильнее всего.</p>')
    for c in comparisons:
        s = c.stats
        parts.append(f'<h2>{esc(c.new)} <span class="muted">против</span> {esc(c.ref)}</h2>')
        if s:
            parts.append(f'<div class="grid"><div class="card"><span class="muted">Spearman</span><b>{s["spearman"]:+.3f}</b></div>'
                         f'<div class="card"><span class="muted">Общий топ-20%</span><b>{s["top20_overlap"]:.0%}</b>'
                         f'<span class="muted">вошли {s["entered_top20"]}, вышли {s["left_top20"]}</span></div>'
                         f'<div class="card"><span class="muted">Медианный сдвиг</span><b>{s["median_abs_shift"]:.0f} мест</b></div>'
                         f'<div class="card"><span class="muted">Согласие с мини-моделью</span>'
                         f'<b>{s["new_vs_mini"]:+.2f}</b><span class="muted">опорный {s["ref_vs_mini"]:+.2f}</span></div></div>')
        parts.append('<ul>' + ''.join(f'<li>{esc(note)}</li>' for note in c.notes) + '</ul>')
        if s:
            parts.append(_class_table(s['by_class'], comparison=True))
        parts += [_detection_card(f, auditor, out_dir) for f in c.movers]
    parts.append('</main></body></html>')
    return '\n'.join(parts)


def _detection_card(f: Finding, auditor: SubmissionAuditor, out_dir: Path) -> str:
    esc = html.escape
    b = auditor.box(f.id)
    image = auditor.data / b['image_path']
    try:
        src = Path(os.path.relpath(image, out_dir)).as_posix()
    except ValueError:  # другой диск в Windows
        src = image.resolve().as_uri()
    w, h = b['image_width'], b['image_height']
    svg = (f'<svg viewBox="0 0 {w:g} {h:g}" role="img" aria-label="кадр {esc(f.frame_id)}">'
           f'<image href="{esc(src)}" width="{w:g}" height="{h:g}"/>'
           f'<rect x="{b["x1"]:g}" y="{b["y1"]:g}" width="{b["x2"] - b["x1"]:g}" height="{b["y2"] - b["y1"]:g}" '
           f'fill="none" stroke="#ffcc00" stroke-width="5"/></svg>')
    title = {'missed': 'Проверить раньше', 'overflagged': 'Можно проверить позже',
             'moved_up': 'Поднялся в очереди', 'moved_down': 'Опустился в очереди'}.get(f.rule, f.rule)
    reasons = ''.join(f'<li>{esc(x)}</li>' for x in f.reasons)
    ranks = (f'Место: опорный <b>{f.ref_rank}</b> → новый <b>{f.main_rank}</b>, мини-модель <b>{f.mini_rank}</b>'
             if f.ref_rank is not None else
             f'Место в очереди: система <b>{f.main_rank}</b>, мини-модель <b>{f.mini_rank}</b>')
    return (f'<div class="det">{svg}<div><span class="badge info">{f.rule}</span> <b>{title}</b>'
            f'<div class="muted"><code>id={esc(f.id)}</code> · {esc(f.class_name)} · <code>{esc(f.frame_id)}</code></div>'
            f'<div>{ranks}</div>'
            f'<div>{esc(f.message)}</div><ul>{reasons}</ul></div></div>')


def save_reports(results: list[AuditResult], auditor: SubmissionAuditor, out_dir: Path = OUTPUT_DIR,
                 stem: str = 'audit_report', comparisons: list[Comparison] = ()) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f'{stem}.html'
    path.write_text(format_html(results, auditor, out_dir, comparisons), encoding='utf-8')
    payload = {'validation': auditor.validation, 'mini_model_weights': auditor.weights(),
               'audits': [r.to_dict() for r in results], 'comparisons': [c.to_dict() for c in comparisons]}
    (out_dir / f'{stem}.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    return path


def read_submission(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={'id': str, 'score': str}, keep_default_na=False)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('files', nargs='+', type=Path, help='CSV id,score')
    parser.add_argument('--data', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--top', type=int, default=10, help='сколько детекций показывать в каждой группе')
    parser.add_argument('--out', type=Path, default=OUTPUT_DIR)
    parser.add_argument('--heuristics-only', action='store_true', help='не обучать мини-модель, только правила')
    parser.add_argument('--vs', type=Path, help='опорный сабмит (например, лучший на Public) для сравнения очередей')
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    auditor = SubmissionAuditor(args.data, force_heuristics=args.heuristics_only)
    subs = {f.stem: read_submission(f) for f in args.files}
    results = [auditor.audit(frame, name, args.top) for name, frame in subs.items()]
    for r in results:
        print(format_text(r), end='\n\n')
    comparisons = []
    if args.vs:
        ref = read_submission(args.vs)
        comparisons = [auditor.compare(frame, ref, name, args.vs.stem, args.top) for name, frame in subs.items()]
        for c in comparisons:
            print(format_comparison_text(c), end='\n\n')
    print(f'HTML-отчёт: {save_reports(results, auditor, args.out, comparisons=comparisons)}')
    return 1 if any(r.verdict == 'error' for r in results) else 0


if __name__ == '__main__':
    raise SystemExit(main())
