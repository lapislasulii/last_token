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

Команды (из корня репозитория):
  python tools/weak_spots_e3.py sensitivity
  python tools/weak_spots_e3.py build-e3 --mode exact      # в Kaggle не отправляется
Файлы pred_all*_flip.csv берутся из origin/main через git show (см. reports/weak_spots.md), в git ветки не входят.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

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


def sensitivity(args):
    df = W.train_table()
    base = dict(conf_term=1 - df.confidence, n_members=df.n_members, P=df.P, veto=df.veto, L=df.L,
                person_out=df.person_out)
    reference = pd.Series(W.formula(votes=df.votes, **base), index=df.index)
    ref_days = W.day_ap(df, reference)
    person = (df.class_name == "person").to_numpy()
    ref_person = W.day_ap(df, reference, person)
    print("Участник base, локальный кэш вето; опора — person_vote --skip-window --legs "
          f"({ref_days[0]:.4f} / {ref_days[1]:.4f}). Диагностика: кандидат остаётся 0.3 / 0.25.")
    rows = []
    for threshold in THRESHOLDS:
        for weight in WEIGHTS:
            score = pd.Series(W.formula(votes=grid_vote(df.iou_same, df.matched, df.class_name, threshold, weight),
                                        **base), index=df.index)
            days, pdays = W.day_ap(df, score), W.day_ap(df, score, person)
            rows.append({"порог": threshold, "вес": weight,
                         R.DAYS[0]: round(days[0], 4), R.DAYS[1]: round(days[1], 4),
                         "среднее": round(float(np.mean(days)), 4),
                         "прирост": round(float(np.mean(days) - np.mean(ref_days)), 4),
                         f"прирост {R.DAYS[0]}": round(days[0] - ref_days[0], 4),
                         f"прирост {R.DAYS[1]}": round(days[1] - ref_days[1], 4),
                         "люди: прирост по дням": f"{pdays[0] - ref_person[0]:+.4f} / {pdays[1] - ref_person[1]:+.4f}",
                         "E3": "<-" if (threshold, weight) == (0.3, 0.25) else ""})
    print(pd.DataFrame(rows).to_string(index=False))


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
    parser.add_argument("cmd", choices=["sensitivity", "build-e3"])
    parser.add_argument("--mode", choices=["exact"], default="exact", help="build-e3: только точный режим")
    a = parser.parse_args()
    R.WORK = ROOT / "outputs" / "redetect_strong"
    {"sensitivity": sensitivity, "build-e3": build_e3}[a.cmd](a)
