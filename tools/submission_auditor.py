"""Аудитор сабмитов основной системы: проверка перед отправкой, опознание файла, второе мнение, очередь оператора.

Проверяет любой файл id,score:
  1. жёсткие правила формата (error);
  2. опознание: SHA256 известных отправок и распознавание рецепта по формуле, включая вето экспертов;
  3. второе мнение мини-модели — логистической регрессии на 7 признаках согласия сильного детектора
     и соседних кадров (warning/info);
  4. объяснение каждого бокса словами и очередь оператора с превью кадров (HTML).
Если признаков сильного детектора нет, мини-модель откатывается на 5 временных признаков, а если и она
не обучилась — на эвристики.

Запуск из корня репозитория:
  python tools/submission_auditor.py submissions/my.csv
  python tools/submission_auditor.py submissions/a.csv submissions/b.csv --vs submissions/best.csv

Код основной системы не меняется: признаки берутся из tools/train_logreg.py, tools/temporal_features.py
и tools/redetect.py; предсказания детекторов — из outputs/redetect_strong/ (в git).
"""
from __future__ import annotations

import argparse
import hashlib
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
STRONG = ROOT / 'outputs' / 'redetect_strong'      # предсказания и OOF-признаки сильного детектора (в git)
OUTPUT_DIR = ROOT / 'outputs' / 'audit'             # в .gitignore: в HTML превью кадров под NDA
MEMBERS = {'base': 'pred_all.csv', 's1': 'pred_all_s1.csv', 's2': 'pred_all_s2.csv'}

TEMPORAL_FEATURES = ['confidence', 'neighbor_iou_max', 'neighbor_iou_min', 'neighbor_conf_delta', 'same_frame_iou']
DETECTOR_FEATURES = ['confidence', 'iou_same', 'match_conf', 'unmatched',
                     'neighbor_iou_max', 'neighbor_iou_min', 'same_frame_iou']

# Отправки Kaggle, записанные в reports/ (SHA256 файла → что это и какой Public).
KNOWN_SHA = {
    '16016509246975456ce264ef9fda459e6f6b22dab499f89eb07433e5b5e64474':
        ('ансамбль base+s1+s2 + вето, любое совпадение', 0.89753, 'reports/redetect_ensemble.md'),
    'a027294245f862bd4be822c6a8f9abbd17cbd6c130699af0b9f3db074357a84a':
        ('ансамбль base+s1+s2+st_strict + вето', 0.89118, 'reports/self_training_submissions.json'),
    'e0aa2f22ab11ebcb0c5fc180f978b3f13b6d8985b47f6a550b51992bb7e8588a':
        ('ученик st_strict + вето', 0.80732, 'reports/self_training_submissions.json'),
    '5eb4d404c310': ('одиночный YOLO11l + вето, любое совпадение', 0.82471, 'reports/redetect_strong.md'),
}
# Рецепты, которые аудитор умеет воспроизвести; Public известных отправок (рецепт, с вето?) → AP.
RECIPE_NAMES = {
    'baseline': '1 − confidence',
    'single_any': 'одиночный YOLO11l, любое совпадение',
    'single_1to1': 'одиночный YOLO11l, один к одному',
    'ens3_any': 'ансамбль base+s1+s2, любое совпадение',
    'ens3_1to1': 'ансамбль base+s1+s2, один к одному',
}
KAGGLE = {('baseline', False): 0.64114, ('baseline', True): 0.70514, ('single_any', True): 0.82471,
          ('ens3_any', True): 0.89753, ('ens3_1to1', True): 0.90211}
BEST_PUBLIC = max(KAGGLE.values())

# Пороги расхождения рангов (доля очереди: 0 = начало, 1 = конец).
MINI_TOP, MAIN_LOW = 0.10, 0.50       # мини-модель: топ-10%, система: вторая половина → «пропущен»
MINI_CALM, MAIN_HIGH = 0.10, 0.10     # мини-модель: последние 10%, система: первые 10% → «лишний»
DISAGREE_SHARE = 0.08                 # доля сильных расхождений, после которой сабмит получает warning
BASELINE_CORR = 0.995                 # «модель ничего не добавила к 1-confidence»
UNRELATED_CORR = 0.30                 # очередь почти не связана со вторым мнением
FORMULA_TOL = 5e-6                    # рецепты различаются минимум на 1/12, а CSV может хранить 6 знаков


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
class AuditResult:
    name: str
    verdict: str                # ok | warning | error
    findings: list[Finding]
    stats: dict
    identity: dict = field(default_factory=dict)
    queue: list[Finding] = field(default_factory=list)

    def to_dict(self):
        return {'name': self.name, 'verdict': self.verdict, 'identity': self.identity, 'stats': self.stats,
                'findings': [asdict(f) for f in self.findings], 'queue': [asdict(f) for f in self.queue]}


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
    values = pd.to_numeric(raw, errors='coerce').to_numpy(dtype=float)

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

    bad_number = ~np.isfinite(values)
    for i in np.flatnonzero(bad_number)[:10]:
        findings.append(Finding('error', 'not_finite', f'score = «{raw.iloc[i]}» — не конечное число', id=ids.iloc[i]))
    safe = np.nan_to_num(values)
    out_of_range = ~bad_number & ((safe < 0) | (safe > 1))
    for i in np.flatnonzero(out_of_range)[:10]:
        findings.append(Finding('error', 'out_of_range', f'score = {values[i]:g} вне [0, 1]', id=ids.iloc[i]))

    valid = ~bad_number & ~out_of_range & ~ids.duplicated(keep=False).to_numpy() & ids.isin(expected).to_numpy()
    clean = pd.Series(values[valid], index=ids.to_numpy()[valid], dtype=float)
    if len(clean) > 1 and clean.nunique() == 1:
        findings.append(Finding('error', 'constant', f'все баллы равны {clean.iloc[0]:g}: очередь не упорядочена'))
    elif len(clean) > 20 and clean.nunique() < 0.03 * len(clean):
        findings.append(Finding('warning', 'few_unique',
                                f'всего {clean.nunique()} разных баллов на {len(clean)} строк: много ничьих в очереди'))
    return findings, clean


# ---------------------------------------------------------------- данные

def load_rows(data: Path, split: str) -> pd.DataFrame:
    from temporal_features import add_temporal
    from train_logreg import features
    policy = json.loads((data / 'scoring_policy.json').read_text(encoding='utf-8'))
    rows = add_temporal(features(data, split, policy), data, split, policy)
    frames = pd.read_csv(data / f'{split}_frames.csv', usecols=['frame_id', 'image_path'])
    return rows.merge(frames, on='frame_id', how='left', validate='many_to_one')


def add_detector_features(train: pd.DataFrame, test: pd.DataFrame, strong: Path = STRONG):
    """Признаки согласия сильного детектора. train — честные OOF из git, test — по предсказаниям участников."""
    import redetect
    oof = pd.read_csv(strong / 'oof_features.csv', dtype={'id': str})
    if 'matched' not in oof:
        raise FileNotFoundError('в oof_features.csv нет колонки matched (нужна версия с сопоставлением один к одному)')
    keep = ['id', 'iou_same', 'match_conf', 'iou_other', 'matched']
    train = train.merge(oof[keep], on='id', how='left', validate='one_to_one')
    if train[keep[1:]].isna().any().any():
        raise ValueError('OOF-признаки детектора есть не для всех train-боксов')
    train['unmatched'] = (~train.matched.astype(bool)).astype(float)
    train['votes_missing'], train['n_members'] = train.unmatched, 1

    boxes = test[['id', 'frame_id', 'class_name', 'x1', 'y1', 'x2', 'y2']]
    members = {m: f for m, f in MEMBERS.items() if (strong / f).exists()}
    if 'base' not in members:
        raise FileNotFoundError(f'нет {strong / MEMBERS["base"]}')
    feats = {m: boxes[['id']].merge(redetect.agreement_features(boxes, pd.read_csv(strong / f)), on='id', how='left')
             for m, f in members.items()}
    base = feats['base']
    test = test.merge(base[['id', 'iou_same', 'match_conf', 'iou_other', 'matched']], on='id', how='left')
    test['unmatched'] = (~test.matched.astype(bool)).astype(float)
    test['n_members'] = len(feats)
    test['votes_missing'] = np.sum([~f.set_index('id').loc[test.id, 'matched'].astype(bool).to_numpy() for f in feats.values()], axis=0)
    # Доли D для рецептов: один к одному и прежнее «любое совпадение» (iou_same < 0.6).
    test['D_1to1'] = test.votes_missing / len(feats)
    test['D_any'] = np.mean([(f.set_index('id').loc[test.id, 'iou_same'] < 0.6).to_numpy() for f in feats.values()], axis=0)
    test['single_any'] = (base.set_index('id').loc[test.id, 'iou_same'] < 0.6).to_numpy().astype(float)
    return train, test, sorted(feats)


def make_mini_model():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    return Pipeline([('scale', StandardScaler()),
                     ('model', LogisticRegression(C=1.0, max_iter=3000, random_state=42))])


# ---------------------------------------------------------------- объяснения

def heuristics(row) -> list[tuple[str, str]]:
    """Эвристики: (короткое имя правила, объяснение человеческими словами)."""
    found = []
    if 'votes_missing' in row and not pd.isna(row.votes_missing):
        n, k = int(row.n_members), int(row.votes_missing)
        if k == n:
            who = 'детектор-контролёр не нашёл' if n == 1 else f'ни один из {n} детекторов-контролёров не нашёл'
            found.append(('детектор не подтвердил', f'{who} пары по правилу один к одному (лучший IoU {row.iou_same:.2f})'))
        elif k > 0:
            verb = 'не нашёл' if k == 1 else 'не нашли'
            found.append(('детекторы разошлись', f'{k} из {n} детекторов-контролёров {verb} пары: спорный бокс'))
        if row.iou_other >= 0.6 and row.iou_same < 0.6:
            found.append(('похоже на другой класс', f'на этом месте детектор видит объект другого класса '
                                                     f'(IoU {row.iou_other:.2f}): возможно, неверный класс'))
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


def validate_mini_model(train: pd.DataFrame, features: list[str]) -> dict:
    """Честная проверка мини-модели: учимся на одном дне, проверяемся на другом."""
    from sklearn.metrics import average_precision_score
    result = {'features': features, 'days': {}}
    oof = pd.Series(np.nan, index=train.index)
    for day in sorted(train.day.unique()):
        fit, hold = train[train.day != day], train[train.day == day]
        if fit.empty or fit.error.nunique() < 2 or hold.error.nunique() < 2:
            continue
        oof[hold.index] = make_mini_model().fit(fit[features], fit.error).predict_proba(hold[features])[:, 1]
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
        persons = scored[scored.class_name == 'person']
        if persons.error.nunique() == 2:
            result['ap_person_mini'] = float(average_precision_score(persons.error, persons.mini))
            result['ap_person_baseline'] = float(average_precision_score(persons.error, 1 - persons.confidence))
    rules = {}
    for found, error in zip(train.apply(heuristics, axis=1), train.error):
        for key, _ in found:
            hit = rules.setdefault(key, [0, 0])
            hit[0] += 1
            hit[1] += int(error)
    result['rules'] = {k: {'fired': n, 'error_rate': e / n}
                       for k, (n, e) in sorted(rules.items(), key=lambda x: -x[1][0])}
    return result


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def known_by_sha(digest: str | None):
    if not digest:
        return None
    for key, value in KNOWN_SHA.items():
        if digest.startswith(key):
            return value
    return None


# ---------------------------------------------------------------- аудитор

class SubmissionAuditor:
    """Создаётся один раз (≈5 с) и проверяет сколько угодно сабмитов."""

    def __init__(self, data: Path = DEFAULT_DATA, force_heuristics: bool = False, strong: Path = STRONG):
        self.data = data
        self.template = pd.read_csv(data / 'sample_submission.csv', dtype={'id': str})
        train, test = load_rows(data, 'train'), load_rows(data, 'test')
        self.members, self.notes = [], []
        try:
            train, test, self.members = add_detector_features(train, test, strong)
            self.features = DETECTOR_FEATURES
        except Exception as exc:
            self.features = TEMPORAL_FEATURES
            self.notes.append(f'признаков сильного детектора нет ({exc}); второе мнение — только временные признаки')
        self.train, self.test = train, test.set_index('id')
        self.validation = validate_mini_model(self.train, self.features)
        self.model = None
        try:
            if force_heuristics:
                raise RuntimeError('эвристики выбраны вручную')
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                self.model = make_mini_model().fit(self.train[self.features], self.train.error)
            self.test['mini'] = self.model.predict_proba(self.test[self.features])[:, 1]
            self.mode = f'мини-модель (логистическая регрессия, {len(self.features)} признаков)'
        except Exception as exc:
            self.model = None
            self.test['mini'] = heuristic_score(self.test)
            self.mode = f'только эвристики ({exc})'
        self.test['reasons'] = self.test.apply(explain, axis=1)
        self.recipes = self._recipes()

    def _recipes(self) -> dict[str, pd.Series]:
        t = self.test
        recipes = {'baseline': 1 - t.confidence}
        if 'unmatched' in t:
            recipes['single_any'] = (1 - t.confidence + t.single_any) / 2
            recipes['single_1to1'] = (1 - t.confidence + t.unmatched) / 2
        if len(self.members) == 3:
            recipes['ens3_any'] = (1 - t.confidence + t.D_any) / 2
            recipes['ens3_1to1'] = (1 - t.confidence + t.D_1to1) / 2
        return recipes

    def weights(self) -> dict:
        if self.model is None:
            return {}
        return dict(zip(self.features, map(float, self.model.named_steps['model'].coef_[0])))

    # ------------------------------------------------------------ опознание
    def identify(self, clean: pd.Series, digest: str | None = None) -> dict:
        identity = {'sha256': digest}
        known = known_by_sha(digest)
        if known:
            identity.update(known_submission=known[0], kaggle_public=known[1], source=known[2])
        ids = self.test.index.intersection(clean.index)
        if len(ids) < len(self.test) or not len(ids) or clean[ids].nunique() < 2:
            return identity
        score = clean[ids]
        # Сначала точные совпадения без вето: «одиночный детектор» по форме совпадает с «baseline + вето» при V = D,
        # и более простое объяснение (без вето) должно побеждать.
        for name, recipe in self.recipes.items():
            if np.abs(score - recipe[ids]).max() < FORMULA_TOL:
                identity.update(recipe=name, veto=False)
                break
        else:
            for name, recipe in self.recipes.items():
                x = 2 * score - recipe[ids]
                v = x.round()
                if np.abs(x - v).max() < FORMULA_TOL and v.isin([0, 1]).all() and v.sum() > 0:
                    identity.update(recipe=name, veto=True, veto_ids=list(v[v == 1].index))
                    break
        if 'recipe' in identity:
            public = KAGGLE.get((identity['recipe'], identity['veto']))
            identity['recipe_public'] = public
            identity.setdefault('kaggle_public', public)
        else:
            corr = {n: float(score.rank().corr(r[ids].rank())) for n, r in self.recipes.items()}
            best = max(corr, key=corr.get)
            identity.update(nearest_recipe=best, nearest_corr=corr[best])
        return identity

    def _identity_findings(self, identity: dict) -> list[Finding]:
        out = []
        if 'known_submission' in identity:
            out.append(Finding('info', 'known_file', f'SHA256 совпадает с отправленным файлом: '
                                                     f'{identity["known_submission"]}, Public {identity["kaggle_public"]}'))
        if 'recipe' in identity:
            name = RECIPE_NAMES[identity['recipe']] + (' + вето экспертов' if identity['veto'] else '')
            extra = f', вето сработало на {len(identity["veto_ids"])} боксах' if identity['veto'] else ''
            public = identity.get('recipe_public')
            known = f'; на Kaggle такой рецепт дал Public {public}' if public else '; такой рецепт на Kaggle не отправлялся'
            out.append(Finding('info', 'recipe', f'файл точно воспроизводится формулой: {name}{extra}{known}'))
        elif 'nearest_recipe' in identity:
            out.append(Finding('info', 'recipe', f'формулой не воспроизводится; ближе всего к рецепту '
                                                 f'«{RECIPE_NAMES[identity["nearest_recipe"]]}» (Spearman {identity["nearest_corr"]:+.3f})'))
        public = identity.get('kaggle_public')
        if public is not None and public < BEST_PUBLIC:
            out.append(Finding('warning', 'not_best', f'у этого файла Public {public}, а лучший известный результат '
                                                      f'команды — {BEST_PUBLIC} (ансамбль один к одному + вето)'))
        return out

    # ------------------------------------------------------------ аудит
    def audit(self, sub: pd.DataFrame, name: str, top: int = 8, digest: str | None = None, queue: int = 15) -> AuditResult:
        findings, clean = check_format(sub, self.template.id)
        stats = {'rows': int(len(sub)), 'valid_rows': int(len(clean)), 'mode': self.mode}
        identity = self.identify(clean, digest)
        findings += self._identity_findings(identity)
        rows = self.test.loc[self.test.index.intersection(clean.index)].copy()
        veto = set(identity.get('veto_ids', []))
        rows['reasons'] = [list(r) + (['вето экспертов: публичные COCO/поза-модели не подтвердили объект'] if i in veto else [])
                           for i, r in zip(rows.index, rows.reasons)]
        result_queue = []
        if len(rows) > 20 and clean.nunique() > 1:
            rows = rows.assign(score=clean[rows.index])
            findings += self._second_opinion(rows, stats, top, veto)
            result_queue = self._queue(rows, queue)
        levels = {f.level for f in findings}
        verdict = 'error' if 'error' in levels else 'warning' if 'warning' in levels else 'ok'
        order = {'error': 0, 'warning': 1, 'info': 2}
        findings.sort(key=lambda f: (order[f.level], f.main_rank is not None, f.rule))
        return AuditResult(name, verdict, findings, stats, identity, result_queue)

    def audit_file(self, path: Path, top: int = 8, queue: int = 15) -> AuditResult:
        return self.audit(read_submission(path), path.stem, top, sha256(path), queue)

    def _second_opinion(self, rows: pd.DataFrame, stats: dict, top: int, veto: set) -> list[Finding]:
        findings = []
        n = len(rows)
        main = rows.score.rank(ascending=False)
        mini = rows.mini.rank(ascending=False)
        corr_mini = float(rows.score.rank().corr(rows.mini.rank()))
        corr_base = float(rows.score.rank().corr((1 - rows.confidence).rank()))
        k = max(1, int(0.2 * n))
        overlap = len(set(main.nsmallest(k).index) & set(mini.nsmallest(k).index)) / k
        stats.update(spearman_vs_mini=corr_mini, spearman_vs_baseline=corr_base, top20_overlap_with_mini=overlap)

        if corr_mini < -UNRELATED_CORR:
            findings.append(Finding('warning', 'inverted',
                                    f'очередь направлена против мини-модели (Spearman {corr_mini:+.2f}): похоже, '
                                    'большой score означает «бокс верный», а не «проверить первым»'))
        elif corr_mini < UNRELATED_CORR:
            findings.append(Finding('warning', 'unrelated',
                                    f'очередь почти не связана с мини-моделью (Spearman {corr_mini:+.2f}); у всех рабочих '
                                    'сабмитов команды > +0.8. Типичная причина — баллы сдвинуты относительно id при merge'))
        if corr_base > BASELINE_CORR:
            findings.append(Finding('warning', 'same_as_baseline',
                                    f'порядок почти совпадает с 1-confidence (Spearman {corr_base:.3f}): '
                                    'основная модель ничего не добавила к baseline'))

        missed = rows[(mini / n <= MINI_TOP) & (main / n > MAIN_LOW)]
        # Бокс, поднятый вето, мини-модель не обязана видеть: вето использует модели, которых у неё нет.
        overflag = rows[(mini / n > 1 - MINI_CALM) & (main / n <= MAIN_HIGH) & ~rows.index.isin(veto)]
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
        stats['person'] = self._person(rows, main, mini)
        return findings

    def _queue(self, rows: pd.DataFrame, size: int) -> list[Finding]:
        """Очередь оператора: первые боксы сабмита с причинами подозрения."""
        main = rows.score.rank(ascending=False, method='first')
        mini = rows.mini.rank(ascending=False)
        return [self._detection(rid, rows.loc[rid], main, mini, 'queue', f'score {rows.score[rid]:.3f}')
                for rid in main.nsmallest(size).index]

    @staticmethod
    def _person(rows, main, mini) -> dict:
        n = len(rows)
        persons = rows[rows.class_name == 'person']
        if persons.empty:
            return {}
        watch = persons.assign(main_rank=main[persons.index], mini_rank=mini[persons.index])
        watch = watch[(watch.mini_rank / n <= 0.3) & (watch.main_rank / n > 0.5)].sort_values('mini_rank')
        return {'boxes': len(persons), 'main_top20': float((main[persons.index] / n <= 0.2).mean()),
                'mini_top20': float((mini[persons.index] / n <= 0.2).mean()),
                'low_box': int((persons.height < 0.2).sum()),
                'watch': [{'id': i, 'main_rank': int(r.main_rank), 'mini_rank': int(r.mini_rank), 'reasons': list(r.reasons)}
                          for i, r in watch.head(8).iterrows()]}

    @staticmethod
    def _by_class(rows, main, mini, disagreements) -> list[dict]:
        n = len(rows)
        top_main, top_mini = main / n <= 0.2, mini / n <= 0.2
        table = []
        for cls, group in rows.groupby('class_name'):
            idx = group.index
            table.append({'class': cls, 'boxes': len(group),
                          'main_top20': float(top_main[idx].mean()), 'mini_top20': float(top_mini[idx].mean()),
                          'disagreements': int(sum(i in disagreements for i in idx))})
        return sorted(table, key=lambda c: -c['boxes'])

    @staticmethod
    def _detection(rid, row, main, mini, rule, message, ref=None):
        reasons = list(row.reasons) or ['явных эвристических причин нет — стоит посмотреть кадр глазами']
        return Finding('info', rule, message, id=rid, frame_id=row.frame_id, class_name=row.class_name,
                       reasons=reasons, main_rank=int(round(main[rid])), mini_rank=int(round(mini[rid])),
                       ref_rank=int(round(ref[rid])) if ref is not None else None)

    # ------------------------------------------------------------ сравнение двух сабмитов
    def compare(self, new: pd.DataFrame, ref: pd.DataFrame, new_name: str, ref_name: str, top: int = 6) -> Comparison:
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
        rank_new, rank_ref = a[common].rank(ascending=False), b[common].rank(ascending=False)
        rank_mini = rows.mini.rank(ascending=False)
        k = max(1, int(0.2 * n))
        top_new, top_ref = set(rank_new.nsmallest(k).index), set(rank_ref.nsmallest(k).index)
        spearman = float(rank_new.corr(rank_ref))
        shift = rank_ref - rank_new                     # > 0: в новой очереди бокс поднялся ближе к началу
        stats = {'rows': n, 'spearman': spearman, 'top20_overlap': len(top_new & top_ref) / k,
                 'entered_top20': len(top_new - top_ref), 'left_top20': len(top_ref - top_new),
                 'changed_rows': int((np.abs(a[common] - b[common]) > 1e-9).sum()),
                 'median_abs_shift': float(shift.abs().median()),
                 'new_vs_mini': float(rank_new.corr(rank_mini)), 'ref_vs_mini': float(rank_ref.corr(rank_mini))}
        stats['by_class'] = sorted(({'class': cls, 'boxes': len(g),
                                     'new_top20': float(rank_new[g.index].le(k).mean()),
                                     'ref_top20': float(rank_ref[g.index].le(k).mean()),
                                     'mean_shift': float(shift[g.index].mean())}
                                    for cls, g in rows.groupby('class_name')), key=lambda c: -c['boxes'])
        if spearman > 0.98:
            notes.append(f'очереди почти одинаковые (Spearman {spearman:.3f}, изменено {stats["changed_rows"]} строк): '
                         'как второй финальный сабмит такая пара почти не страхует от ошибки выбора')
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
        movers = [self._detection(rid, rows.loc[rid], rank_new, rank_mini, 'moved_up', f'поднялся на {shift[rid]:.0f} мест', rank_ref)
                  for rid in shift[shift > 0].nlargest(top).index]
        movers += [self._detection(rid, rows.loc[rid], rank_new, rank_mini, 'moved_down', f'опустился на {-shift[rid]:.0f} мест', rank_ref)
                   for rid in shift[shift < 0].nsmallest(top).index]
        return Comparison(new_name, ref_name, notes, stats, movers)

    def box(self, rid: str) -> dict:
        row = self.test.loc[rid]
        return {k: float(row[k]) for k in ('x1', 'y1', 'x2', 'y2', 'image_width', 'image_height')} | {
            'image_path': str(row.image_path)}


def find_reference(folder: Path = ROOT / 'submissions') -> Path | None:
    """Лучший известный сабмит в папке: по SHA256 или по имени файла лидера."""
    best, best_score = None, -1.0
    for path in sorted(folder.glob('*.csv')):
        known = known_by_sha(sha256(path))
        score = known[1] if known else (BEST_PUBLIC if path.name == 'redetect_strong_ens-base-s1-s2_1to1_veto_veto.csv' else -1)
        if score > best_score:
            best, best_score = path, score
    return best


# ---------------------------------------------------------------- отчёты

LABEL = {'ok': 'OK', 'warning': 'WARNING', 'error': 'ERROR'}


def format_text(result: AuditResult, limit: int = 4, queue: int = 5) -> str:
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
        if f.main_rank is not None:
            if shown >= limit:
                continue
            shown += 1
        where = f' id={f.id}' if f.id else ''
        rank = f' {f.class_name}, место {f.main_rank} у системы / {f.mini_rank} у мини-модели.' if f.main_rank else ''
        lines.append(f'  {f.level.upper():7} {f.rule:17}{where}{rank} {f.message}')
        lines += [f'{"":27}- {reason}' for reason in f.reasons]
    hidden = sum(f.main_rank is not None for f in result.findings) - shown
    if hidden > 0:
        lines.append(f'  ... остальные расхождения ({hidden}) — в HTML-отчёте')
    if result.queue and queue:
        lines.append(f'  очередь оператора (первые {min(queue, len(result.queue))} из {len(result.queue)} в HTML):')
        for f in result.queue[:queue]:
            lines.append(f'    {f.main_rank:3}. {f.class_name:10} id={f.id}  {"; ".join(f.reasons)}')
    return '\n'.join(lines)


def format_comparison_text(c: Comparison, limit: int = 3) -> str:
    lines = [f'=== Сравнение «{c.new}» с «{c.ref}» ===']
    s = c.stats
    if s:
        lines.append(f"Spearman {s['spearman']:+.3f}; изменено строк {s['changed_rows']}; общий топ-20% {s['top20_overlap']:.0%} "
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


CSS = """
:root{--bg:#fbfaf7;--fg:#1d1d1b;--muted:#6b6a65;--card:#fff;--line:#e4e1d8;--ok:#2f7d4f;--warn:#a66a00;--err:#b3261e;--info:#3b5ba5}
@media (prefers-color-scheme:dark){:root{--bg:#161614;--fg:#ecebe6;--muted:#a3a19a;--card:#21211e;--line:#34332f;--ok:#2f8f58;--warn:#b07400;--err:#c2392f;--info:#4a68b5}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1100px;margin:0 auto;padding:24px 16px 64px}h1{font-size:26px;margin:0 0 4px}h2{font-size:18px;margin:32px 0 10px}h3{font-size:16px;margin:20px 0 8px}
.muted{color:var(--muted)}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px;margin:16px 0}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px}.card b{display:block;font-size:22px}
.badge{display:inline-block;padding:2px 10px;border-radius:99px;font-weight:600;font-size:13px;color:#fff}
.ok{background:var(--ok)}.warning{background:var(--warn)}.error{background:var(--err)}.info{background:var(--info)}
.scroll{overflow-x:auto}table{width:100%;border-collapse:collapse;background:var(--card);border:1px solid var(--line)}
td,th{padding:8px 10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}th{font-size:13px;color:var(--muted)}
.det{display:grid;grid-template-columns:minmax(0,320px) 1fr;gap:14px;background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px;margin:10px 0}
.det svg{width:100%;height:auto;border-radius:6px;background:var(--line)}.det ul{margin:6px 0 0;padding-left:18px}code{font-size:13px;word-break:break-all}
details{margin:10px 0}summary{cursor:pointer;color:var(--muted)}
@media (max-width:640px){.det{grid-template-columns:1fr}}
"""


def _class_table(rows: list[dict], comparison: bool) -> str:
    esc = html.escape
    if comparison:
        head = '<th>Класс</th><th>Боксов</th><th>В топ-20%: новый</th><th>опорный</th><th>Средний сдвиг</th>'
        body = ''.join(f"<tr><td>{esc(c['class'])}</td><td>{c['boxes']}</td><td>{c['new_top20']:.0%}</td>"
                       f"<td>{c['ref_top20']:.0%}</td><td>{c['mean_shift']:+.1f}</td></tr>" for c in rows)
    else:
        head = '<th>Класс</th><th>Боксов</th><th>В топ-20%: система</th><th>мини-модель</th><th>Расхождений</th>'
        body = ''.join(f"<tr><td>{esc(c['class'])}</td><td>{c['boxes']}</td><td>{c['main_top20']:.0%}</td>"
                       f"<td>{c['mini_top20']:.0%}</td><td>{c['disagreements']}</td></tr>" for c in rows)
    return f'<div class="scroll"><table><tr>{head}</tr>{body}</table></div>'


def format_html(results: list[AuditResult], auditor: SubmissionAuditor, out_dir: Path,
                comparisons: list[Comparison] = ()) -> str:
    v = auditor.validation
    esc = html.escape
    parts = ['<!doctype html><html lang="ru"><head><meta charset="utf-8">'
             '<meta name="viewport" content="width=device-width,initial-scale=1">'
             f'<title>Аудит сабмитов</title><style>{CSS}</style></head><body><main>',
             '<h1>Аудит сабмитов перед отправкой</h1>',
             f'<p class="muted">Проверка формата, опознание файла, второе мнение мини-модели ({esc(auditor.mode)}) '
             'и очередь оператора с объяснениями. Код основной системы не менялся.</p>']
    parts += [f'<p class="muted">⚠ {esc(n)}</p>' for n in auditor.notes]
    parts.append('<h2>Сводка</h2><div class="scroll"><table><tr><th>Сабмит</th><th>Вердикт</th><th>Что это за файл</th>'
                 '<th>Public</th><th>Spearman с мини-моделью</th><th>Пропущено / лишних</th></tr>')
    for r in results:
        s, ident = r.stats, r.identity
        what = ident.get('known_submission') or (RECIPE_NAMES[ident['recipe']] + (' + вето' if ident['veto'] else '')
                                                 if 'recipe' in ident else 'не опознан')
        public = ident.get('kaggle_public')
        corr = f"{s['spearman_vs_mini']:+.2f}" if 'spearman_vs_mini' in s else '—'
        dis = f"{s['missed_by_main']} / {s['overflagged_by_main']}" if 'missed_by_main' in s else '—'
        parts.append(f'<tr><td><a href="#{esc(r.name)}">{esc(r.name)}</a></td>'
                     f'<td><span class="badge {r.verdict}">{r.verdict}</span></td><td>{esc(what)}</td>'
                     f'<td>{public if public else "—"}</td><td>{corr}</td><td>{dis}</td></tr>')
    parts.append('</table></div>')

    parts.append('<h2>Можно ли доверять второму мнению</h2>'
                 '<p class="muted">Честная проверка на train: учимся на одном дне, проверяемся на другом. '
                 'Признаки детектора на train — от моделей, не видевших этот день.</p><div class="grid">')
    for day, d in v.get('days', {}).items():
        parts.append(f'<div class="card"><span class="muted">AP мини-модели, {day}</span><b>{d["ap_mini"]:.3f}</b>'
                     f'<span class="muted">baseline {d["ap_baseline"]:.3f}</span></div>')
    if 'error_rate_mini_top20' in v:
        parts.append(f'<div class="card"><span class="muted">Ошибок в её топ-20%</span><b>{v["error_rate_mini_top20"]:.0%}</b>'
                     f'<span class="muted">в среднем {v["error_rate_all"]:.0%}</span></div>')
    if 'ap_person_mini' in v:
        parts.append(f'<div class="card"><span class="muted">AP на людях</span><b>{v["ap_person_mini"]:.3f}</b>'
                     f'<span class="muted">baseline {v["ap_person_baseline"]:.3f}</span></div>')
    parts.append('</div><details><summary>Эвристики и веса мини-модели</summary>'
                 '<div class="scroll"><table><tr><th>Эвристика</th><th>Сработала на train</th><th>Из них ошибок</th></tr>')
    for rule, d in v.get('rules', {}).items():
        parts.append(f'<tr><td>{esc(rule)}</td><td>{d["fired"]}</td><td>{d["error_rate"]:.0%}</td></tr>')
    parts.append('</table></div>')
    if auditor.weights():
        parts.append('<p class="muted">Веса (после масштабирования): ' + ', '.join(
            f'<code>{k}</code> {w:+.2f}' for k, w in auditor.weights().items()) + '</p>')
    parts.append('</details>')

    for r in results:
        parts.append(f'<h2 id="{esc(r.name)}">{esc(r.name)} <span class="badge {r.verdict}">{r.verdict}</span></h2>')
        plain = [f for f in r.findings if f.main_rank is None]
        if plain:
            parts.append('<div class="scroll"><table><tr><th>Уровень</th><th>Правило</th><th>Где</th><th>Что</th></tr>')
            parts += [f'<tr><td><span class="badge {f.level}">{f.level}</span></td><td><code>{f.rule}</code></td>'
                      f'<td><code>{esc(f.id or "весь файл")}</code></td><td>{esc(f.message)}</td></tr>' for f in plain]
            parts.append('</table></div>')
        if r.stats.get('by_class'):
            parts.append('<h3>По классам</h3><p class="muted">Какая доля боксов класса попала в первые 20% очереди.</p>')
            parts.append(_class_table(r.stats['by_class'], comparison=False))
        p = r.stats.get('person')
        if p:
            parts.append(f'<h3>Люди — слабое место решения</h3><p>{p["boxes"]} боксов person; в топ-20% системы '
                         f'{p["main_top20"]:.0%}, у мини-модели {p["mini_top20"]:.0%}; низких боксов (&lt;20% высоты кадра, '
                         f'возможно обрезан торс): {p["low_box"]}.</p>')
            if p['watch']:
                parts.append('<p class="muted">Люди, которых мини-модель ставит в первые 30%, а система — во вторую половину:</p><ul>')
                parts += [f'<li><code>{esc(w["id"])}</code>: система {w["main_rank"]}, мини-модель {w["mini_rank"]} — '
                          f'{esc("; ".join(w["reasons"]) or "без явных причин")}</li>' for w in p['watch']]
                parts.append('</ul>')
        cards = [f for f in r.findings if f.main_rank is not None]
        if cards:
            parts.append('<h3>Расхождения с мини-моделью</h3>')
            parts += [_detection_card(f, auditor, out_dir) for f in cards]
        if r.queue:
            parts.append(f'<details><summary>Очередь оператора: первые {len(r.queue)} боксов с причинами</summary>')
            parts += [_detection_card(f, auditor, out_dir) for f in r.queue]
            parts.append('</details>')

    if comparisons:
        parts.append('<h2>Сравнение сабмитов</h2><p class="muted">Помогает выбрать два финальных сабмита: '
                     'насколько очереди отличаются и какие боксы переехали сильнее всего.</p>')
    for c in comparisons:
        s = c.stats
        parts.append(f'<h3>{esc(c.new)} <span class="muted">против</span> {esc(c.ref)}</h3>')
        if s:
            parts.append(f'<div class="grid"><div class="card"><span class="muted">Spearman</span><b>{s["spearman"]:+.3f}</b>'
                         f'<span class="muted">изменено строк: {s["changed_rows"]}</span></div>'
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
    title = {'missed': 'Проверить раньше', 'overflagged': 'Можно проверить позже', 'queue': f'Место {f.main_rank} в очереди',
             'moved_up': 'Поднялся в очереди', 'moved_down': 'Опустился в очереди'}.get(f.rule, f.rule)
    reasons = ''.join(f'<li>{esc(x)}</li>' for x in f.reasons)
    ranks = (f'Место: опорный <b>{f.ref_rank}</b> → новый <b>{f.main_rank}</b>, мини-модель <b>{f.mini_rank}</b>'
             if f.ref_rank is not None else
             f'Место в очереди: система <b>{f.main_rank}</b>, мини-модель <b>{f.mini_rank}</b>')
    return (f'<div class="det">{svg}<div><span class="badge info">{f.rule}</span> <b>{title}</b>'
            f'<div class="muted"><code>id={esc(f.id)}</code> · {esc(f.class_name)} · <code>{esc(f.frame_id)}</code></div>'
            f'<div>{ranks}</div><div>{esc(f.message)}</div><ul>{reasons}</ul></div></div>')


def save_reports(results: list[AuditResult], auditor: SubmissionAuditor, out_dir: Path = OUTPUT_DIR,
                 stem: str = 'audit_report', comparisons: list[Comparison] = ()) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f'{stem}.html'
    path.write_text(format_html(results, auditor, out_dir, comparisons), encoding='utf-8')
    payload = {'validation': auditor.validation, 'mini_model_weights': auditor.weights(), 'notes': auditor.notes,
               'audits': [r.to_dict() for r in results], 'comparisons': [c.to_dict() for c in comparisons]}
    (out_dir / f'{stem}.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    return path


def read_submission(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={'id': str, 'score': str}, keep_default_na=False)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('files', nargs='+', type=Path, help='CSV id,score')
    parser.add_argument('--data', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--top', type=int, default=8, help='сколько расхождений показывать в каждой группе')
    parser.add_argument('--queue', type=int, default=15, help='длина очереди оператора в HTML')
    parser.add_argument('--out', type=Path, default=OUTPUT_DIR)
    parser.add_argument('--heuristics-only', action='store_true', help='не обучать мини-модель, только правила')
    parser.add_argument('--vs', type=Path, help='опорный сабмит; по умолчанию — лучший известный файл из submissions/')
    parser.add_argument('--no-vs', action='store_true', help='не сравнивать с опорным сабмитом')
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    warnings.filterwarnings('ignore')
    auditor = SubmissionAuditor(args.data, force_heuristics=args.heuristics_only)
    for note in auditor.notes:
        print(f'! {note}')
    results = [auditor.audit_file(f, args.top, args.queue) for f in args.files]
    for r in results:
        print(format_text(r), end='\n\n')
    ref = None if args.no_vs else (args.vs or find_reference())
    comparisons = []
    if ref is not None:
        frame = read_submission(ref)
        comparisons = [auditor.compare(read_submission(f), frame, f.stem, ref.stem, args.top)
                       for f in args.files if f.resolve() != ref.resolve()]
        for c in comparisons:
            print(format_comparison_text(c), end='\n\n')
    print(f'HTML-отчёт: {save_reports(results, auditor, args.out, comparisons=comparisons)}')
    return 1 if any(r.verdict == 'error' for r in results) else 0


if __name__ == '__main__':
    raise SystemExit(main())
