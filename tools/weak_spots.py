"""Слабые места метода: люди в нижнем ярусе и конфликт вето с детекторами. Без обучения.

Опора — лидер tools/person_vote.py с --skip-window --legs:
    score = (1 − confidence + D' + 2·max(V, L)) / 4,
    D' = (голоса детекторов + P) / (число детекторов + 1) у людей вне окна (центр рамки < 330 px — окно), иначе D.

Эксперименты заданы до запуска; каждый сравнивается с опорой отдельно, комбинация — только из прошедших:
  E1  конфликт вето у окна: у person в зоне окна при D = 0 (все детекторы подтверждают) вето V := 0; L не трогаем.
      Довод из механизма: вето людей — та же pose-модель, которой у окна мешает засвет (P там уже исключён).
  E1c контроль, только для понимания: то же правило для всех классов везде.
  E2  порядок людей внутри яруса: у person (1 − confidence) -> ((1 − confidence) + (1 − match_conf)) / 2,
      match_conf — уверенность нашего детектора в паре с IoU >= 0.6 (0, если пары нет), среднее по участникам.
  E3  двухуровневый голос детектора для person: 1 при iou_same < 0.3; 0.25 при iou_same >= 0.3 без пары
      один к одному; 0 при паре. Только train: на test нет flip-предсказаний по участникам.

Критерий (для каждого): ни один день не хуже опоры больше чем на 0.003; среднее выше минимум на +0.003;
при исключении любого одного кадра среднее не ниже опоры. Для E1 ещё: среди конфликтных людей у окна
на train ошибок не больше половины (если их меньше 3 — «на train нечем проверить»).

Сборка test без кэша вето: файлы лидера из origin/main раскладываются обратно на D, V, P, L
(x = 4·score − (1 − confidence); V = 1[x > 1.5]; D = x − 2V; у людей вне окна P = 7·D' − 6·D).

Команды (из корня репозитория; нужен outputs/expert_veto/train_{pose,features}.csv):
  python tools/weak_spots.py eval
  git show origin/main:submissions/redetect_strong_ens-base-s1-s2-base_flip-s1_flip-s2_flip_veto_veto.csv \\
      > outputs/weak_spots/leader_flip.csv
  git show origin/main:submissions/redetect_strong_ens-base-s1-s2-base_flip-s1_flip-s2_flip_pvote_nowin_legs_veto.csv \\
      > outputs/weak_spots/leader_flip_pose.csv
  python tools/weak_spots.py build --experiments E1        # только прошедшие E1/E2; в Kaggle не отправляется
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

import person_vote as PV
import redetect as R

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "outputs" / "weak_spots"
LEADER_FLIP = WORK / "leader_flip.csv"
LEADER_FLIP_POSE = WORK / "leader_flip_pose.csv"
N_LEADER = 6                       # участников в файлах лидера: base s1 s2 и их отражения
TEST_MEMBERS = ["base", "s1", "s2"]  # для match_conf на test: flip-предсказаний test нет
LOW_IOU, MID_VOTE = 0.3, 0.25
GRID_TOL, MATCH_TOL = 1e-6, 1e-9
EXPERIMENTS = ["E1", "E1c", "E2", "E3"]


# ---------------------------------------------------------------- правила (чистые функции, покрыты тестами)
def in_window(x1, x2):
    return (np.asarray(x1, dtype=float) + np.asarray(x2, dtype=float)) / 2 < PV.WINDOW_X


def e1_veto(veto, d, cls, window, everywhere=False):
    """E1: у person в зоне окна при D = 0 вето снимается. everywhere=True — контроль E1c (все классы, везде)."""
    veto, d = np.asarray(veto, dtype=float), np.asarray(d, dtype=float)
    where = np.ones(len(veto), dtype=bool) if everywhere else (np.asarray(cls) == "person") & np.asarray(window)
    return np.where(where & (d == 0), 0.0, veto)


def e2_conf_term(confidence, match_conf, cls):
    """E2: член уверенности в формуле; у person — среднее (1 − conf заказчика) и (1 − conf нашего детектора)."""
    own = 1 - np.asarray(confidence, dtype=float)
    ours = 1 - np.asarray(match_conf, dtype=float)
    return np.where(np.asarray(cls) == "person", (own + ours) / 2, own)


def e3_vote(iou_same, matched, cls):
    """E3: голос одного детектора. У person: 1 при IoU < 0.3, 0.25 при IoU >= 0.3 без пары, 0 при паре;
    у остальных классов — обычный голос «нет пары один к одному»."""
    iou_same, matched = np.asarray(iou_same, dtype=float), np.asarray(matched, dtype=bool)
    person = np.where(matched, 0.0, np.where(iou_same < LOW_IOU, 1.0, MID_VOTE))
    return np.where(np.asarray(cls) == "person", person, (~matched).astype(float))


def formula(conf_term, votes, n_members, P, veto, L, person_out):
    """Скор лидера из частей: D' у людей вне окна включает голос позы P."""
    votes, P, person_out = np.asarray(votes, dtype=float), np.asarray(P, dtype=float), np.asarray(person_out)
    d = np.where(person_out, (votes + P) / (n_members + 1), votes / n_members)
    return (np.asarray(conf_term, dtype=float) + d + 2 * np.maximum(veto, L)) / 4


def decompose(score, confidence, n=N_LEADER):
    """Файл «детекторы + вето»: score = (1 − conf + D + 2V) / 4 -> D (сетка 1/n), V."""
    x = 4 * np.asarray(score, dtype=float) - (1 - np.asarray(confidence, dtype=float))
    veto = (x > 1.5).astype(float)
    d = x - 2 * veto
    off = np.abs(d * n - np.round(d * n))
    if off.max() > GRID_TOL or d.min() < -GRID_TOL or d.max() > 1 + GRID_TOL:
        raise ValueError(f"D не на сетке 1/{n}: отклонение до {off.max():.2e}, диапазон [{d.min():.4f}, {d.max():.4f}]")
    return np.round(d * n) / n, veto


def decompose_pose(score_pose, confidence, d, veto, person_out, n=N_LEADER):
    """Файл лидера с позой -> P (у людей вне окна), L. У остальных D' обязан совпасть с D, а V' >= V."""
    x = 4 * np.asarray(score_pose, dtype=float) - (1 - np.asarray(confidence, dtype=float))
    veto_p = (x > 1.5).astype(float)
    d_p = x - 2 * veto_p
    d, veto = np.asarray(d, dtype=float), np.asarray(veto, dtype=float)
    person_out = np.asarray(person_out, dtype=bool)
    p = np.where(person_out, (n + 1) * d_p - n * d, 0.0)
    if np.abs(p - np.round(p)).max() > GRID_TOL or not np.isin(np.round(p), [0, 1]).all():
        raise ValueError("P вне {0, 1}: файлы лидера не согласуются")
    if np.abs(d_p - d)[~person_out].max(initial=0) > GRID_TOL:
        raise ValueError("D' ≠ D у боксов без голоса позы")
    if (veto_p < veto).any():
        raise ValueError("V' < V: вето с позой не может быть меньше вето без позы")
    return np.round(p), ((veto_p == 1) & (veto == 0)).astype(float)


# ---------------------------------------------------------------- оценка на train
def train_table():
    """198 оцениваемых боксов train с сигналами лидера на участнике base и признаками нашего детектора."""
    train = R.load("train.csv", dtype={"id": str})
    train = train[train.class_name != "table"].copy()
    train["day"] = train.frame_id.map(R.frame_day("train"))
    member = R.oof_features(train, "base")
    df = PV.add_signals(train, "train", [member])
    feats = member.set_index("id").reindex(df.id)
    df["iou_same"], df["match_conf"] = feats.iou_same.to_numpy(), feats.match_conf.to_numpy()
    df["matched"] = feats.matched.to_numpy(dtype=bool)
    df["window"] = in_window(df.x1, df.x2)
    df["person_out"] = (df.class_name == "person") & ~df.window
    return df


def variants(df):
    """Скоры опоры и экспериментов на одной таблице."""
    own = 1 - df.confidence
    base = dict(conf_term=own, votes=df.votes, n_members=df.n_members, P=df.P, veto=df.veto, L=df.L,
                person_out=df.person_out)
    out = {"опора": formula(**base)}
    _, check = PV.scores(df, legs=True, skip_window=True)
    assert np.allclose(out["опора"], check), "опора разошлась с person_vote.scores"
    out["E1"] = formula(**{**base, "veto": e1_veto(df.veto, df.D, df.class_name, df.window)})
    out["E1c"] = formula(**{**base, "veto": e1_veto(df.veto, df.D, df.class_name, df.window, everywhere=True)})
    out["E2"] = formula(**{**base, "conf_term": e2_conf_term(df.confidence, df.match_conf, df.class_name)})
    out["E3"] = formula(**{**base, "votes": e3_vote(df.iou_same, df.matched, df.class_name)})
    return {k: pd.Series(np.asarray(v, dtype=float), index=df.index) for k, v in out.items()}


def combine(df, names):
    """Комбинация прошедших экспериментов (E1c — контроль, в комбинацию не входит)."""
    veto = e1_veto(df.veto, df.D, df.class_name, df.window) if "E1" in names else df.veto
    conf = e2_conf_term(df.confidence, df.match_conf, df.class_name) if "E2" in names else 1 - df.confidence
    votes = e3_vote(df.iou_same, df.matched, df.class_name) if "E3" in names else df.votes
    return pd.Series(formula(conf, votes, df.n_members, df.P, veto, df.L, df.person_out), index=df.index)


def day_ap(df, score, mask=None):
    mask = np.ones(len(df), dtype=bool) if mask is None else mask
    return [average_precision_score(df.error[mask & (df.day == d)], score[mask & (df.day == d)]) for d in R.DAYS]


def leave_one_frame(df, score, reference):
    """Минимальный прирост среднего AP к опоре при исключении одного кадра (по всем кадрам train)."""
    gains = []
    for frame in df.frame_id.unique():
        keep = (df.frame_id != frame).to_numpy()
        gains.append(np.mean(day_ap(df, score, keep)) - np.mean(day_ap(df, reference, keep)))
    return float(np.min(gains)), int(np.argmin(gains))


def check(df, name, score, reference):
    """Печать AP, bootstrap, исключения кадров; возвращает, прошёл ли критерий."""
    ref_days, days = day_ap(df, reference), day_ap(df, score)
    gain = np.mean(days) - np.mean(ref_days)
    worst, _ = leave_one_frame(df, score, reference)
    for day in R.DAYS:
        mask = df.day == day
        diffs = PV.frame_bootstrap(df[mask], reference[mask], score[mask])
        low, high = np.quantile(diffs, [0.05, 0.95])
        print(f"  {name} {day}: bootstrap {diffs.mean():+.4f}, 5–95% [{low:+.4f}, {high:+.4f}], "
              f"доля повторов с приростом {(diffs > 0).mean():.2f}")
    ok_days = all(d >= r - 0.003 for d, r in zip(days, ref_days))
    passed = ok_days and gain >= 0.003 and worst >= 0
    print(f"  {name}: дни не хуже −0.003: {'да' if ok_days else 'нет'}; среднее {gain:+.4f} "
          f"(нужно ≥ +0.003); минимум при исключении кадра {worst:+.4f} (нужно ≥ 0) -> "
          f"{'ПРОШЁЛ' if passed else 'не прошёл'}")
    return passed


def person_ap(df, scores):
    """AP внутри person по дням и прирост к опоре: эксперименты меняют людей, прирост должен идти отсюда."""
    rows = []
    person = (df.class_name == "person").to_numpy()
    ref = day_ap(df, scores["опора"], person)
    for name, score in scores.items():
        days = day_ap(df, score, person)
        rows.append(dict(формула=name, **{d: round(v, 4) for d, v in zip(R.DAYS, days)},
                         **{f"прирост {d}": round(v - r, 4) for d, v, r in zip(R.DAYS, days, ref)},
                         оба_дня=round(average_precision_score(df.error[person], score[person]), 4),
                         строк_изменено=int((np.abs(score - scores["опора"]) > 1e-12).sum()),
                         из_них_людей=int(((np.abs(score - scores["опора"]) > 1e-12) & person).sum())))
    return pd.DataFrame(rows)


def evaluate(args):
    df = train_table()
    scores = variants(df)
    print("Участник: base (других дневных предсказаний на этой машине нет); опора — person_vote --skip-window --legs")
    print("\n1. AP по дням")
    table = pd.DataFrame([dict(формула=k, **{d: round(v, 4) for d, v in zip(R.DAYS, day_ap(df, s))},
                               среднее=round(float(np.mean(day_ap(df, s))), 4)) for k, s in scores.items()])
    print(table.to_string(index=False))

    print("\n5. Конфликт вето и детектора: класс × зона × V × D (вето без L; D — наш детектор)")
    zone = np.where(df.window, "окно", "зал")
    print(df.groupby(["class_name", zone, "veto", "D"]).error.agg(боксов="size", ошибок="sum").to_string())
    conflict = df[(df.veto == 1) & (df.D == 0)]
    print("Конфликтные боксы (V = 1, D = 0):")
    print(conflict.groupby(["class_name", np.where(conflict.window, "окно", "зал")]).error
          .agg(боксов="size", ошибок="sum").to_string() if len(conflict) else "  нет")
    window_people = conflict[(conflict.class_name == "person") & conflict.window]

    print("\n6. Люди нижнего яруса лидера (D' = 0, max(V, L) = 0): AUC порядка")
    d_lead = np.where(df.person_out, (df.votes + df.P) / (df.n_members + 1), df.D)
    low = df[(df.class_name == "person") & (d_lead == 0) & (np.maximum(df.veto, df.L) == 0)]
    for day, part in low.groupby("day"):
        key = e2_conf_term(part.confidence, part.match_conf, part.class_name)
        if 0 < part.error.sum() < len(part):
            print(f"  {day}: боксов {len(part)}, ошибок {int(part.error.sum())}; AUC 1 − confidence "
                  f"{roc_auc_score(part.error, 1 - part.confidence):.3f}, AUC ключа E2 {roc_auc_score(part.error, key):.3f}")
        else:
            print(f"  {day}: боксов {len(part)}, ошибок {int(part.error.sum())}: AUC не определён")

    print("\n2–3. Bootstrap по кадрам (3000 повторов) и исключение по одному кадру; критерий")
    passed = []
    for name in EXPERIMENTS:
        if check(df, name, scores[name], scores["опора"]) and name != "E1c":
            passed.append(name)
    print(f"\nE1, доп. условие: конфликтных людей у окна на train {len(window_people)}, "
          f"из них ошибок {int(window_people.error.sum())}")
    if len(window_people) < 3:
        print("  -> на train нечем проверить (меньше 3 боксов); остаётся довод из механизма (засвет у окна)")
    elif window_people.error.sum() > len(window_people) / 2:
        print("  -> ошибок больше половины: E1 не проходит")
        passed = [p for p in passed if p != "E1"]
    if len(passed) > 1:
        name = "+".join(passed)
        scores[name] = combine(df, passed)
        print(f"\nКомбинация прошедших: {name}")
        check(df, name, scores[name], scores["опора"])

    print("\n4. AP внутри класса person")
    print(person_ap(df, scores).to_string(index=False))
    print(f"\nПрошли критерий: {', '.join(passed) if passed else 'ни один'}")

    for name, score in scores.items():
        path = ROOT / "oof" / f"weak_spots_{'ref' if name == 'опора' else name.replace('+', '_')}.csv"
        if path.exists():
            print(f"{path} уже есть, не перезаписываю")
        else:
            df[["id"]].assign(score=score.to_numpy()).to_csv(path, index=False)
            print("OOF:", path)


# ---------------------------------------------------------------- сборка test поверх файлов лидера
def leader_parts():
    """Тест-боксы с D, V, P, L, восстановленными из файлов лидера; ворота — точное совпадение пересборки."""
    for path in (LEADER_FLIP, LEADER_FLIP_POSE):
        if not path.exists():
            raise SystemExit(f"нет {path}: выполните команды git show из docstring")
    test = R.load("test.csv", dtype={"id": str})
    flip = pd.read_csv(LEADER_FLIP, dtype={"id": str}).rename(columns={"score": "s_flip"})
    pose = pd.read_csv(LEADER_FLIP_POSE, dtype={"id": str}).rename(columns={"score": "s_pose"})
    df = test.merge(flip, on="id", validate="one_to_one").merge(pose, on="id", validate="one_to_one")
    assert len(df) == len(test) == 362
    df["window"] = in_window(df.x1, df.x2)
    df["person_out"] = (df.class_name == "person") & ~df.window
    df["D"], df["V"] = decompose(df.s_flip, df.confidence)
    df["P"], df["L"] = decompose_pose(df.s_pose, df.confidence, df.D, df.V, df.person_out)
    df["votes"] = df.D * N_LEADER
    rebuilt = formula(1 - df.confidence, df.votes, N_LEADER, df.P, df.V, df.L, df.person_out)
    gap = np.abs(rebuilt - df.s_pose).max()
    if gap > MATCH_TOL:
        raise SystemExit(f"Ворота сборки не пройдены: пересборка лидера отличается на {gap:.2e}")
    print(f"Ворота сборки: лидер с позой пересобран из D, V, P, L, максимум расхождения {gap:.1e}")
    print(f"  V = 1 у {int(df.V.sum())}, L = 1 у {int(df.L.sum())}, P = 1 у {int(df.P.sum())} "
          f"(людей вне окна {int(df.person_out.sum())}); конфликтов V = 1, D = 0: {int(((df.V == 1) & (df.D == 0)).sum())}")
    return df


def test_match_conf(df):
    """match_conf на test — среднее по участникам base s1 s2 (как agreement_features в person_vote.submit)."""
    test = R.load("test.csv", dtype={"id": str})
    values = []
    for member in TEST_MEMBERS:
        feats = R.agreement_features(test, pd.read_csv(R.pred_path("all", member)))
        values.append(feats.set_index("id").match_conf.reindex(df.id).to_numpy())
    return np.mean(values, axis=0)


def build(args):
    R.WORK = ROOT / "outputs" / "redetect_strong"
    df = leader_parts()
    names = sorted(set(args.experiments))
    veto = e1_veto(df.V, df.D, df.class_name, df.window) if "E1" in names else df.V
    conf = 1 - df.confidence
    if "E2" in names:
        df["match_conf"] = test_match_conf(df)
        conf = e2_conf_term(df.confidence, df.match_conf, df.class_name)
    df["score"] = formula(conf, df.votes, N_LEADER, df.P, veto, df.L, df.person_out)
    changed = df[np.abs(df.score - df.s_pose) > 1e-12]
    print(f"\n{'+'.join(names)}: изменилось строк {len(changed)} из {len(df)}; по классам "
          f"{changed.class_name.value_counts().to_dict()}")
    for row in changed.itertuples():
        print(f"  {row.id} {row.class_name:10s} {'окно' if row.window else 'зал '} "
              f"D={row.D:.3f} V={int(row.V)} L={int(row.L)} {row.s_pose:.4f} -> {row.score:.4f}")
    rank_old = df.s_pose.rank(ascending=False, method="average")
    rank_new = df.score.rank(ascending=False, method="average")
    print(f"Сравнение с лидером: Spearman {spearmanr(df.s_pose, df.score).statistic:.4f}; "
          f"сменили позицию больше чем на 20 мест: {int((np.abs(rank_new - rank_old) > 20).sum())}")
    template = pd.read_csv(R.DATA / "sample_submission.csv", dtype={"id": str})
    out = template[["id"]].merge(df[["id", "score"]], on="id", validate="one_to_one")
    assert len(out) == 362 and out.id.is_unique and set(out.id) == set(template.id)
    assert np.isfinite(out.score).all() and out.score.between(0, 1).all()
    path = ROOT / "submissions" / f"weak_spots_{'_'.join(names)}_on_flip_pose.csv"
    if path.exists():
        raise FileExistsError(f"{path} уже есть: не перезаписываю")
    path.parent.mkdir(exist_ok=True)
    out.to_csv(path, index=False)
    print("Файл:", path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cmd", choices=["eval", "build"])
    parser.add_argument("--experiments", nargs="+", choices=["E1", "E2"], default=["E1"],
                        help="build: прошедшие эксперименты (E3 на test не собрать)")
    a = parser.parse_args()
    R.WORK = ROOT / "outputs" / "redetect_strong"
    {"eval": evaluate, "build": build}[a.cmd](a)
