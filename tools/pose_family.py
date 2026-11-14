"""Голос позы у людей в зале: «каждая семья моделей весит поровну». Без обучения.

Сейчас у людей вне окна (центр рамки >= 330 px) D' = (6·Y + P) / 7: Y — доля шести YOLO-голосов без пары
один к одному (IoU >= 0.6), P — голос pose-модели (tools/person_vote.py). Шесть YOLO — одна архитектура
с разными seed и отражениями, ошибаются согласованно; поза — независимая модель. Кандидат (зафиксирован
до запуска): у людей вне окна D' = (Y + P) / 2, то есть вес позы w = 1/2. Всё остальное как у лидера:
у окна и у других классов D' = Y, вето max(V, L), score = (1 − confidence + D' + 2·max(V, L)) / 4.

Урок RF-DETR: локально выигрыш на людях, а на test он переставил кисти и подносы. Поэтому и на train,
и на test проверяется, что меняются только люди вне окна.

Критерий: ни один день не хуже опоры больше чем на 0.003; среднее выше минимум на +0.003; минимум при
исключении одного кадра >= 0; при P с порогом уверенности позы 0.4 и 0.6 прирост среднего тоже > 0.

Команды (из корня репозитория; нужны outputs/expert_veto/ и дневные предсказания шести участников):
  python tools/pose_family.py eval
  python tools/pose_family.py build          # только если критерий пройден; в Kaggle не отправляется
"""
import argparse
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import person_vote as PV
import redetect as R
import weak_spots as W

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "outputs" / "pose_family"
MEMBERS = ["base", "s1", "s2", "base_flip", "s1_flip", "s2_flip"]
LEADER_W = 1 / 7                 # (6·Y + P) / 7
CANDIDATE_W = 1 / 2              # (Y + P) / 2
GRID_W = [1 / 7, 1 / 4, 1 / 3, 1 / 2, 2 / 3]
POSE_CONFS = [0.4, 0.5, 0.6]
PUBLISHED = (0.92579, 0.92565)   # лидер на шести участниках, опубликованные числа команды
LEADER_FILES = {
    "leader_flip.csv": "submissions/redetect_strong_ens-base-s1-s2-base_flip-s1_flip-s2_flip_veto_veto.csv",
    "leader_flip_pose.csv": "submissions/redetect_strong_ens-base-s1-s2-base_flip-s1_flip-s2_flip_pvote_nowin_legs_veto.csv",
}
OOF = ROOT / "oof" / "pose_family_w050.csv"
OUT = ROOT / "submissions" / "pose_family_w050_on_flip_pose.csv"
BOX = ["x1", "y1", "x2", "y2"]


# ---------------------------------------------------------------- формула (чистые функции, покрыты тестами)
def d_prime(y, p, person_out, w):
    """D' с весом позы w у людей вне окна: (1 − w)·Y + w·P; у остальных D' = Y. w = 1/7 — лидер."""
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    return np.where(np.asarray(person_out, dtype=bool), (1 - w) * y + w * p, y)


def score(confidence, y, p, veto, legs, person_out, w):
    return (1 - np.asarray(confidence, dtype=float) + d_prime(y, p, person_out, w)
            + 2 * np.maximum(veto, legs)) / 4


def assert_only_hall_people(old, new, person_out, tol=1e-12):
    """Урок RF-DETR: скор может меняться только у людей вне окна. Возвращает маску изменённых строк."""
    changed = np.abs(np.asarray(new, dtype=float) - np.asarray(old, dtype=float)) > tol
    stray = changed & ~np.asarray(person_out, dtype=bool)
    assert not stray.any(), f"изменились строки не людей вне окна: {int(stray.sum())}"
    return changed


def pose_vote_at(rows, pose, conf):
    """Копия person_vote.pose_vote с параметром порога уверенности позы (оригинал — POSE_CONF = 0.5).
    P = 1, если рамка person не получила пару (один к одному, IoU >= 0.6) среди рамок pose-модели
    на полном кадре с уверенностью >= conf; для остальных классов P = 0."""
    pose = pose[(pose.src == "full") & (pose.conf >= conf)]
    vote = pd.Series(0.0, index=rows.id)
    people = rows[rows.class_name == "person"]
    for frame, group in people.groupby("frame_id"):
        theirs = pose[pose.frame_id == frame]
        iou = R.iou_matrix(group[BOX].to_numpy(), theirs[BOX].to_numpy())
        matched = R.one_to_one(iou, np.ones_like(iou, dtype=bool))
        vote[group.id.to_numpy()] = (~matched).astype(float)
    return vote.to_numpy()


# ---------------------------------------------------------------- оценка на train
def train_table():
    for member in MEMBERS:
        for day in R.DAYS:
            if not R.pred_path(f"day_{day}", member).exists():
                raise SystemExit(f"нет {R.pred_path(f'day_{day}', member)}: подтяните из origin/main (git show)")
    train = R.load("train.csv", dtype={"id": str})
    train = train[train.class_name != "table"].copy()
    train["day"] = train.frame_id.map(R.frame_day("train"))
    df = PV.add_signals(train, "train", [R.oof_features(train, m) for m in MEMBERS])
    df["window"] = W.in_window(df.x1, df.x2)
    df["person_out"] = (df.class_name == "person") & ~df.window
    return df


def scored(df, w, p=None):
    p = df.P if p is None else p
    return pd.Series(score(df.confidence, df.D, p, df.veto, df.L, df.person_out, w), index=df.index)


def mean_ap(df, s):
    return float(np.mean(W.day_ap(df, s)))


def tier(y):
    return np.where(y == 0, "0", np.where(y == 1, "1", "дробный"))


def evaluate(args):
    df = train_table()
    reference, candidate = scored(df, LEADER_W), scored(df, CANDIDATE_W)
    _, check = PV.scores(df, legs=True, skip_window=True)
    assert np.allclose(reference.to_numpy(), check.to_numpy(), atol=1e-12), "опора разошлась с PV.scores"
    ref_days = W.day_ap(df, reference)
    assert all(abs(a - b) <= 1e-4 for a, b in zip(ref_days, PUBLISHED)), f"ворота: опора {ref_days} ≠ {PUBLISHED}"
    print(f"Участники: {' '.join(MEMBERS)}")
    print(f"Ворота: опора w = 1/7 даёт {ref_days[0]:.5f} / {ref_days[1]:.5f} (опубликовано {PUBLISHED[0]} / "
          f"{PUBLISHED[1]}, допуск 1e-4) и совпадает с person_vote.scores --skip-window --legs")

    changed = assert_only_hall_people(reference, candidate, df.person_out)
    print(f"Изменено строк train: {int(changed.sum())}, все — люди вне окна "
          f"(людей вне окна {int(df.person_out.sum())}, у них Y ≠ P: {int((df.person_out & (df.D != df.P)).sum())})")

    print("\n1. AP по дням")
    rows = [dict(формула=name, **{d: round(v, 4) for d, v in zip(R.DAYS, W.day_ap(df, s))},
                 среднее=round(mean_ap(df, s), 4)) for name, s in (("опора w=1/7", reference), ("кандидат w=1/2", candidate))]
    print(pd.DataFrame(rows).to_string(index=False))
    print("\n2. Bootstrap (3000 повторов), исключение по одному кадру, критерий по AP")
    passed_ap = W.check(df, "w=1/2", candidate, reference)

    print("\n3. Сетка по весу позы w (диагностика)")
    grid = []
    for w in GRID_W:
        s = scored(df, w)
        days = W.day_ap(df, s)
        worst, _ = W.leave_one_frame(df, s, reference)
        grid.append(dict(w=f"{w:.4f}", **{d: round(v, 4) for d, v in zip(R.DAYS, days)},
                         **{f"прирост {d}": round(v - r, 4) for d, v, r in zip(R.DAYS, days, ref_days)},
                         прирост_среднего=round(float(np.mean(days) - np.mean(ref_days)), 4),
                         мин_искл_кадра=round(worst, 4)))
    print(pd.DataFrame(grid).to_string(index=False))

    print("\n4. Чувствительность к порогу уверенности позы в P (опора и кандидат с одним и тем же P)")
    pose = pd.read_csv(PV.VETO / "train_pose.csv")
    p_rows, p_gain = [], {}
    for conf in POSE_CONFS:
        p = pose_vote_at(df, pose, conf)
        if conf == PV.POSE_CONF:
            assert np.array_equal(p, df.P.to_numpy()), "копия pose_vote разошлась с оригиналом"
        ref_c, cand_c = scored(df, LEADER_W, p), scored(df, CANDIDATE_W, p)
        r_days, c_days = W.day_ap(df, ref_c), W.day_ap(df, cand_c)
        p_gain[conf] = float(np.mean(c_days) - np.mean(r_days))
        p_rows.append(dict(conf=conf, P1_людей_вне_окна=int(p[df.person_out].sum()),
                           опора=" / ".join(f"{v:.4f}" for v in r_days), кандидат=" / ".join(f"{v:.4f}" for v in c_days),
                           прирост_по_дням=" / ".join(f"{c - r:+.4f}" for c, r in zip(c_days, r_days)),
                           прирост_среднего=round(p_gain[conf], 4)))
    print(pd.DataFrame(p_rows).to_string(index=False))

    print("\n5. Люди вне окна × день × Y × P: боксов / ошибок")
    hall = df[df.person_out].assign(Y=lambda t: tier(t.D))
    print(hall.groupby(["day", "Y", "P"]).error.agg(боксов="size", ошибок="sum").to_string())
    both = hall.groupby("P").error.agg(боксов="size", ошибок="sum")
    print("Оба дня: " + "; ".join(f"P = {int(k)}: ошибок {int(r.ошибок)} из {int(r.боксов)}" for k, r in both.iterrows()))

    print("\n6. AP внутри person по дням")
    print(W.person_ap(df, {"опора": reference, "кандидат": candidate}).to_string(index=False))

    passed_p = all(p_gain[c] > 0 for c in (0.4, 0.6))
    print(f"\nКритерий: по AP {'да' if passed_ap else 'нет'}; прирост при P conf 0.4 и 0.6 "
          f"{p_gain[0.4]:+.4f} / {p_gain[0.6]:+.4f}: {'да' if passed_p else 'нет'} -> "
          f"{'ПРОШЁЛ' if passed_ap and passed_p else 'не прошёл'}")
    if OOF.exists():
        print(f"{OOF} уже есть, не перезаписываю")
    else:
        df[["id"]].assign(score=candidate.to_numpy()).to_csv(OOF, index=False)
        print("OOF:", OOF)


# ---------------------------------------------------------------- сборка test
def leader_copies():
    """Локальные копии файлов лидера из origin/main в outputs/pose_family/ (существующие не трогаем)."""
    WORK.mkdir(parents=True, exist_ok=True)
    for name, path in LEADER_FILES.items():
        target = WORK / name
        if not target.exists():
            target.write_bytes(subprocess.run(["git", "show", f"origin/main:{path}"], cwd=ROOT,
                                              check=True, capture_output=True).stdout)
    W.LEADER_FLIP, W.LEADER_FLIP_POSE = WORK / "leader_flip.csv", WORK / "leader_flip_pose.csv"


def build(args):
    if OUT.exists():
        raise FileExistsError(f"{OUT} уже есть: не перезаписываю")
    leader_copies()
    df = W.leader_parts()   # Y (= D на сетке 1/6), V, P, L + ворота: пересобранный лидер совпадает с файлом (1e-9)
    leader = score(df.confidence, df.D, df.P, df.V, df.L, df.person_out, LEADER_W)
    assert np.abs(leader - df.s_pose.to_numpy()).max() <= 1e-9, "ворота сборки: лидер не пересобирается"
    df["score"] = score(df.confidence, df.D, df.P, df.V, df.L, df.person_out, CANDIDATE_W)
    changed = assert_only_hall_people(df.s_pose, df.score, df.person_out)
    expected = (df.person_out & (df.D != df.P)).to_numpy()
    assert np.array_equal(changed, expected), "изменились не ровно люди вне окна с Y ≠ P"
    up = int((df.score > df.s_pose + 1e-12).sum())
    print(f"\nКандидат w = 1/2: изменилось строк {int(changed.sum())} (все — люди вне окна с Y ≠ P); "
          f"поднялось {up}, опустилось {int(changed.sum()) - up}")
    rank_old = df.s_pose.rank(ascending=False)
    rank_new = df.score.rank(ascending=False)
    moved = df[np.abs(rank_new - rank_old) > 20]
    print(f"Spearman с лидером {spearmanr(df.s_pose, df.score).statistic:.4f}; "
          f"сдвинулись больше чем на 20 мест: {len(moved)}")
    if len(moved):
        print(moved.groupby(["class_name", np.where(moved.window, "окно", "зал")]).size().to_string())
    assert moved.person_out.all(), "больше чем на 20 мест сдвинулись не только люди в зале"
    for row in df[changed].itertuples():
        print(f"  {row.id} Y={row.D:.3f} P={int(row.P)} V={int(row.V)} L={int(row.L)} "
              f"{row.s_pose:.4f} -> {row.score:.4f}")
    template = pd.read_csv(R.DATA / "sample_submission.csv", dtype={"id": str})
    out = template[["id"]].merge(df[["id", "score"]], on="id", validate="one_to_one")
    assert len(out) == 362 and out.id.is_unique and set(out.id) == set(template.id)
    assert np.isfinite(out.score).all() and out.score.between(0, 1).all()
    out.to_csv(OUT, index=False)
    print("Файл:", OUT)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cmd", choices=["eval", "build"])
    a = parser.parse_args()
    R.WORK = ROOT / "outputs" / "redetect_strong"
    {"eval": evaluate, "build": build}[a.cmd](a)
