"""E3 (двухуровневый голос детектора для людей): чувствительность к порогу и весу и точная сборка test.

E3 из tools/weak_spots.py: у person голос одного детектора — 1 при iou_same < 0.3, 0.25 при iou_same >= 0.3
без пары один к одному, 0 при паре. Кандидат — ровно эти 0.3 / 0.25; сетка ниже только диагностика.

  sensitivity       порог 0.2 / 0.3 / 0.4 × вес 0 / 0.25 / 0.5 на train (участник base, локальный кэш вето),
                    AP по дням и прирост к опоре person_vote --skip-window --legs.
  build-e3 --mode exact
                    test поверх файлов лидера из origin/main. Ворота: D шести участников
                    (base s1 s2 base_flip s1_flip s2_flip) из outputs/redetect_strong/pred_all*.csv с обычным
                    правилом совпадает с разложением файла лидера, а пересобранный скор равен лидеру с позой
                    (допуск 1e-9). Затем E3 для людей по всем шести участникам; P, V, L — из разложения
                    файла лидера с позой. Выход: submissions/weak_spots_e3_exact_on_flip_pose.csv.

  eval-e3 --members ...
                    E3 на train для набора участников: каждый — R.oof_features (не видел день бокса), голос
                    для людей как в E3, у остальных классов обычный; D — среднее голосов; P, V, L — PV.add_signals
                    с тем же списком. Ворота: опора без E3 совпадает с person_vote.py eval --skip-window --legs.

Команды (из корня репозитория):
  python tools/weak_spots_e3.py sensitivity                # то же, что сетка из eval-e3 --members base
  python tools/weak_spots_e3.py eval-e3 --members base s1 s2 base_flip s1_flip s2_flip
  python tools/weak_spots_e3.py build-e3 --mode exact      # в Kaggle не отправляется
Файлы pred_all*_flip.csv берутся из origin/main через git show (см. reports/weak_spots.md), в git ветки не входят.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import person_vote as PV
import redetect as R
import weak_spots as W

ROOT = Path(__file__).resolve().parents[1]
MEMBERS = ["base", "s1", "s2", "base_flip", "s1_flip", "s2_flip"]   # состав лидера
THRESHOLDS, WEIGHTS = (0.2, 0.3, 0.4), (0.0, 0.25, 0.5)
OUT = ROOT / "submissions" / "weak_spots_e3_exact_on_flip_pose.csv"
TOL = 1e-9


def grid_vote(iou_same, matched, cls, threshold, weight):
    """Голос одного детектора для сетки: у person 1 при IoU < порога, weight без пары при IoU >= порога, 0 при паре.
    threshold=0.3, weight=0.25 — ровно E3 (weak_spots.e3_vote); weight=1 — обычный голос опоры."""
    iou_same, matched = np.asarray(iou_same, dtype=float), np.asarray(matched, dtype=bool)
    person = np.where(matched, 0.0, np.where(iou_same < threshold, 1.0, weight))
    return np.where(np.asarray(cls) == "person", person, (~matched).astype(float))


def member_votes(matched, iou_same, cls, threshold=W.LOW_IOU, weight=W.MID_VOTE):
    """Сумма голосов участников (строки — участники). У person — двухуровневый голос, у остальных — обычный."""
    matched, iou_same = np.atleast_2d(matched), np.atleast_2d(iou_same)
    return sum(grid_vote(iou_same[k], matched[k], cls, threshold, weight) for k in range(len(matched)))


def train_members(members):
    """Боксы train с P, V, L по тому же списку участников (PV.add_signals) и признаки каждого участника.
    Каждый участник взят из R.oof_features: модель не видела день бокса."""
    train = R.load("train.csv", dtype={"id": str})
    train = train[train.class_name != "table"].copy()
    train["day"] = train.frame_id.map(R.frame_day("train"))
    frames = [R.oof_features(train, m) for m in members]
    df = PV.add_signals(train, "train", frames)
    feats = [f.set_index("id").reindex(df.id) for f in frames]
    matched = np.stack([f.matched.to_numpy(dtype=bool) for f in feats])
    iou_same = np.stack([f.iou_same.to_numpy(dtype=float) for f in feats])
    df["window"] = W.in_window(df.x1, df.x2)
    df["person_out"] = (df.class_name == "person") & ~df.window
    return df, matched, iou_same


def leader_score(df, votes):
    return pd.Series(W.formula(1 - df.confidence, votes, df.n_members, df.P, df.veto, df.L, df.person_out),
                     index=df.index)


def grid_table(df, matched, iou_same, reference):
    """Сетка порог × вес: AP по дням и прирост к опоре (диагностика; кандидат — 0.3 / 0.25)."""
    ref_days = W.day_ap(df, reference)
    person = (df.class_name == "person").to_numpy()
    ref_person = W.day_ap(df, reference, person)
    rows = []
    for threshold in THRESHOLDS:
        for weight in WEIGHTS:
            score = leader_score(df, member_votes(matched, iou_same, df.class_name, threshold, weight))
            days, pdays = W.day_ap(df, score), W.day_ap(df, score, person)
            rows.append({"порог": threshold, "вес": weight,
                         R.DAYS[0]: round(days[0], 4), R.DAYS[1]: round(days[1], 4),
                         "среднее": round(float(np.mean(days)), 4),
                         "прирост": round(float(np.mean(days) - np.mean(ref_days)), 4),
                         f"прирост {R.DAYS[0]}": round(days[0] - ref_days[0], 4),
                         f"прирост {R.DAYS[1]}": round(days[1] - ref_days[1], 4),
                         "люди: прирост по дням": f"{pdays[0] - ref_person[0]:+.4f} / {pdays[1] - ref_person[1]:+.4f}",
                         "E3": "<-" if (threshold, weight) == (0.3, 0.25) else ""})
    return pd.DataFrame(rows)


def setup(members):
    """Таблица, опора и ворота: опора без E3 обязана совпасть с person_vote eval --skip-window --legs."""
    df, matched, iou_same = train_members(members)
    assert np.array_equal((~matched).sum(axis=0), df.votes.to_numpy()), "голоса участников разошлись с add_signals"
    reference = leader_score(df, (~matched).sum(axis=0))
    _, check = PV.scores(df, legs=True, skip_window=True)
    assert np.allclose(reference.to_numpy(), check.to_numpy(), atol=1e-12), \
        "опора разошлась с person_vote.py eval --skip-window --legs"
    return df, matched, iou_same, reference


def sensitivity(args):
    df, matched, iou_same, reference = setup(["base"])
    ref_days = W.day_ap(df, reference)
    print("Участник base, локальный кэш вето; опора — person_vote --skip-window --legs "
          f"({ref_days[0]:.4f} / {ref_days[1]:.4f}). Диагностика: кандидат остаётся 0.3 / 0.25.")
    print(grid_table(df, matched, iou_same, reference).to_string(index=False))


def eval_e3(args):
    """E3 на train для любого набора участников с дневными предсказаниями."""
    members = args.members
    missing = [m for m in members if not all(R.pred_path(f"day_{d}", m).exists() for d in R.DAYS)]
    if missing:
        raise SystemExit(f"нет дневных предсказаний для: {' '.join(missing)}")
    df, matched, iou_same, reference = setup(members)
    print(f"Участники: {' '.join(members)}; опора — person_vote --skip-window --legs "
          f"(ворота: совпадение с person_vote.scores пройдено)")
    scores = {"опора": reference, "E3": leader_score(df, member_votes(matched, iou_same, df.class_name))}
    print("\nAP по дням")
    print(pd.DataFrame([dict(формула=k, **{d: round(v, 4) for d, v in zip(R.DAYS, W.day_ap(df, s))},
                             среднее=round(float(np.mean(W.day_ap(df, s))), 4)) for k, s in scores.items()])
          .to_string(index=False))
    print("\nBootstrap по кадрам (3000 повторов), исключение по одному кадру, критерий")
    W.check(df, "E3", scores["E3"], reference)
    print("\nAP внутри person")
    print(W.person_ap(df, scores).to_string(index=False))
    print("\nСетка чувствительности порог × вес (диагностика; кандидат 0.3 / 0.25)")
    print(grid_table(df, matched, iou_same, reference).to_string(index=False))


def member_table(test, member):
    feats = R.agreement_features(test, pd.read_csv(R.pred_path("all", member)))
    return feats.set_index("id").reindex(test.id)


def build_e3(args):
    if OUT.exists():
        raise FileExistsError(f"{OUT} уже есть: не перезаписываю")
    df = W.leader_parts()                                  # D, V, P, L из файлов лидера + ворота пересборки
    test = R.load("test.csv", dtype={"id": str}).set_index("id").loc[df.id].reset_index()
    for member in MEMBERS:
        if not R.pred_path("all", member).exists():
            raise SystemExit(f"нет {R.pred_path('all', member)}: подтяните из origin/main (git show)")
    feats = [member_table(test, m) for m in MEMBERS]
    matched = np.stack([f.matched.to_numpy(dtype=bool) for f in feats])
    iou_same = np.stack([f.iou_same.to_numpy(dtype=float) for f in feats])

    d6 = (~matched).mean(axis=0)
    gap_d = np.abs(d6 - df.D.to_numpy()).max()
    print(f"Ворота 1: D шести участников против разложения лидера — расхождение до {gap_d:.1e}, "
          f"совпало у {int((np.abs(d6 - df.D.to_numpy()) <= TOL).sum())} из {len(df)}")
    if gap_d > TOL:
        raise SystemExit("ворота 1 не пройдены: D участников не совпадает с файлом лидера")
    rebuilt = W.formula(1 - df.confidence, (~matched).sum(axis=0), len(MEMBERS), df.P, df.V, df.L, df.person_out)
    gap_s = np.abs(rebuilt - df.s_pose.to_numpy()).max()
    print(f"Ворота 2: пересобранный скор против лидера с позой — расхождение до {gap_s:.1e}")
    if gap_s > TOL:
        raise SystemExit("ворота 2 не пройдены")

    votes = sum(W.e3_vote(iou_same[k], matched[k], df.class_name) for k in range(len(MEMBERS)))
    df["score"] = W.formula(1 - df.confidence, votes, len(MEMBERS), df.P, df.V, df.L, df.person_out)
    changed = df[np.abs(df.score - df.s_pose) > 1e-12]
    print(f"\nE3 exact: изменилось строк {len(changed)} из {len(df)}; по классам "
          f"{changed.class_name.value_counts().to_dict()}; у окна {int(changed.window.sum())}")
    low = (iou_same < W.LOW_IOU) & ~matched
    df["n_low"], df["n_mid"] = low.sum(axis=0), ((~matched) & ~low).sum(axis=0)
    for row in df.loc[changed.index].itertuples():
        print(f"  {row.id} {'окно' if row.window else 'зал '} D={row.D:.3f} (IoU<0.3: {row.n_low}, без пары: {row.n_mid}) "
              f"P={int(row.P)} V={int(row.V)} L={int(row.L)} {row.s_pose:.4f} -> {row.score:.4f}")
    rank_old = df.s_pose.rank(ascending=False)
    rank_new = df.score.rank(ascending=False)
    print(f"Spearman с лидером {spearmanr(df.s_pose, df.score).statistic:.4f}; "
          f"сменили позицию больше чем на 20 мест: {int((np.abs(rank_new - rank_old) > 20).sum())}")

    template = pd.read_csv(R.DATA / "sample_submission.csv", dtype={"id": str})
    out = template[["id"]].merge(df[["id", "score"]], on="id", validate="one_to_one")
    assert len(out) == 362 and out.id.is_unique and set(out.id) == set(template.id)
    assert np.isfinite(out.score).all() and out.score.between(0, 1).all()
    OUT.parent.mkdir(exist_ok=True)
    out.to_csv(OUT, index=False)
    print("Файл:", OUT)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cmd", choices=["sensitivity", "eval-e3", "build-e3"])
    parser.add_argument("--mode", choices=["exact"], default="exact", help="build-e3: только точный режим")
    parser.add_argument("--members", nargs="+", default=["base"],
                        help="eval-e3: участники с дневными предсказаниями, например base s1 s2 base_flip s1_flip s2_flip")
    a = parser.parse_args()
    R.WORK = ROOT / "outputs" / "redetect_strong"
    {"sensitivity": sensitivity, "eval-e3": eval_e3, "build-e3": build_e3}[a.cmd](a)
