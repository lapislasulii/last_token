"""Голос pose-модели для людей: ещё один «участник ансамбля» без обучения на наших метках.

Зачем. У людей наш дообученный детектор плохо переносит через день конвенцию ручной разметки
(«рамка до ног, только если ноги видны»). Публичная pose-модель YOLO11x-pose рисует рамку видимой части
человека, и это ближе к ручной рамке. Мы сверяем рамки заказчика с её рамками по официальному правилу
метки (один к одному, IoU >= 0.6) и добавляем результат как ещё один голос в долю D — только для person.

Скор:  (1 - confidence + D' + 2 * V') / 4
  D' для person = (сумма голосов наших детекторов + P) / (число детекторов + 1); для остальных классов D' = D.
  P  = 1, если у рамки заказчика нет пары среди рамок pose-модели на полном кадре (conf >= 0.5).
  С флагом --skip-window голос P не учитывается у окна слева (центр рамки левее 330 px): там контровой
       свет, pose-модель на полном кадре человека часто не видит, и на train её голос в этой зоне пустой.
  V' = max(V, L) с флагом --legs; V — вето экспертов.
  L  = 1, если у скелета, чьи плечи внутри рамки, уверенное колено или лодыжка лежит ниже нижней границы
       рамки больше чем на 0.35 её высоты: рамка обрезает видимые ноги.

Новых прогонов моделей нет: берётся кэш tools/expert_veto.py (outputs/expert_veto/{split}_pose.csv).
Веса: https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x-pose.pt

Команды (из корня репозитория):
  python tools/person_vote.py eval   --tag strong --members base [--legs] [--skip-window]
      AP по дням против текущей формулы, bootstrap по кадрам, таблица ошибок людей; OOF в oof/
  python tools/person_vote.py submit --tag strong --members base s1 s2 [--legs] [--skip-window]
      файл для test в submissions/ (существующий файл не перезаписывается; в Kaggle не отправляется)
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

import redetect as R

ROOT = Path(__file__).resolve().parents[1]
VETO = ROOT / "outputs" / "expert_veto"
BOX = ["x1", "y1", "x2", "y2"]
POSE_CONF = 0.5      # рамки pose-модели, которые считаем «её разметкой»
LEG_BELOW = 0.35     # насколько ниже рамки должна быть точка ноги, в высотах рамки
KEYPOINT_CONF, SHOULDER_CONF, SKELETON_CONF = 0.5, 0.3, 0.15
LEG_POINTS = ("lkne", "rkne", "lank", "rank")
WINDOW_X = 330       # центр рамки левее этой границы — места у окна слева (контровой свет)


def pose_vote(rows, pose):
    """P: 1, если рамка person не получила пару (один к одному, IoU >= 0.6) среди рамок pose-модели.

    Берём только прогон на полном кадре: тайлы обрезают людей, и обрезанная рамка «подтверждает»
    обрезанную рамку заказчика. Для остальных классов P = 0.
    """
    pose = pose[(pose.src == "full") & (pose.conf >= POSE_CONF)]
    vote = pd.Series(0.0, index=rows.id)
    people = rows[rows.class_name == "person"]
    for frame, group in people.groupby("frame_id"):
        theirs = pose[pose.frame_id == frame]
        iou = R.iou_matrix(group[BOX].to_numpy(), theirs[BOX].to_numpy())
        matched = R.one_to_one(iou, np.ones_like(iou, dtype=bool))
        vote[group.id.to_numpy()] = (~matched).astype(float)
    return vote.to_numpy()


def legs_below(rows, pose):
    """На сколько высот рамки ниже её нижней границы лежит самая нижняя уверенная точка ноги
    у скелета, чьи плечи внутри рамки. Тайлы (увеличение вдвое) здесь нужны: на них видно ступни.
    -9, если таких точек нет."""
    pose = pose[pose.conf >= SKELETON_CONF].copy()
    pose["shx"] = (pose.lsho_x + pose.rsho_x) / 2
    pose["shy"] = (pose.lsho_y + pose.rsho_y) / 2
    pose = pose[np.minimum(pose.lsho_c, pose.rsho_c) > SHOULDER_CONF]
    by_frame = dict(tuple(pose.groupby("frame_id")))
    result = []
    for row in rows.itertuples():
        best = -9.0
        skeletons = by_frame.get(row.frame_id)
        if row.class_name == "person" and skeletons is not None:
            own = skeletons[(skeletons.shx > row.x1) & (skeletons.shx < row.x2)
                            & (skeletons.shy > row.y1) & (skeletons.shy < row.y2)]
            for point in LEG_POINTS:
                seen = own[own[f"{point}_c"] > KEYPOINT_CONF]
                if len(seen):
                    best = max(best, float(((seen[f"{point}_y"] - row.y2) / (row.y2 - row.y1)).max()))
        result.append(best)
    return np.array(result)


def add_signals(df, split, member_frames):
    """Собрать голоса детекторов, P, вето V и флаг ног L в одну таблицу."""
    pose = pd.read_csv(VETO / f"{split}_pose.csv")
    veto = pd.read_csv(VETO / f"{split}_features.csv", dtype={"id": str})[["id", "veto"]]
    df = df.merge(veto, on="id", how="left", validate="one_to_one")
    votes = np.stack([(~frame.set_index("id").matched.reindex(df.id)).to_numpy(dtype=float) for frame in member_frames])
    df["votes"], df["n_members"] = votes.sum(axis=0), len(member_frames)
    df["D"] = votes.mean(axis=0)
    df["P"] = pose_vote(df, pose)
    df["L"] = (legs_below(df, pose) > LEG_BELOW).astype(float)
    return df


def scores(df, legs, skip_window=False):
    """Текущая формула и формула с голосом pose-модели."""
    person = df.class_name == "person"
    if skip_window:  # у окна оставляем прежнюю долю D
        person &= (df.x1 + df.x2) / 2 >= WINDOW_X
    d_new = np.where(person, (df.votes + df.P) / (df.n_members + 1), df.D)
    veto_new = np.maximum(df.veto, df.L) if legs else df.veto
    current = (1 - df.confidence + df.D + 2 * df.veto) / 4
    new = (1 - df.confidence + d_new + 2 * veto_new) / 4
    return current, new


def frame_bootstrap(day_rows, current, new, repeats=3000, seed=0):
    """Парный bootstrap по кадрам: разность AP новой и текущей формулы на одном дне."""
    rng = np.random.default_rng(seed)
    frames = day_rows.frame_id.unique()
    index = {frame: np.where(day_rows.frame_id.to_numpy() == frame)[0] for frame in frames}
    error, a, b = day_rows.error.to_numpy(), new.to_numpy(), current.to_numpy()
    diffs = []
    for _ in range(repeats):
        take = np.concatenate([index[frame] for frame in rng.choice(frames, len(frames))])
        if error[take].min() != error[take].max():
            diffs.append(average_precision_score(error[take], a[take]) - average_precision_score(error[take], b[take]))
    return np.array(diffs)


def variant(args):
    """Часть имени файла: pvote, pvote_nowin, pvote_legs, pvote_nowin_legs."""
    return "pvote" + ("_nowin" if args.skip_window else "") + ("_legs" if args.legs else "")


def evaluate(args):
    train = R.load("train.csv", dtype={"id": str})
    train = train[train.class_name != "table"].copy()
    train["day"] = train.frame_id.map(R.frame_day("train"))
    df = add_signals(train, "train", [R.oof_features(train, member) for member in args.members])
    current, new = scores(df, args.legs, args.skip_window)
    print(f"Участники: {' '.join(args.members)}; флаг ног: {'да' if args.legs else 'нет'}; "
          f"зона окна: {'пропущена' if args.skip_window else 'учтена'}")
    for name, value in (("текущая формула", current), ("с голосом pose-модели", new)):
        by_day = [average_precision_score(df.error[df.day == d], value[df.day == d]) for d in R.DAYS]
        print(f"{name:24s} AP {by_day[0]:.4f} {by_day[1]:.4f} среднее {np.mean(by_day):.4f}")
    for day in R.DAYS:
        mask = df.day == day
        diffs = frame_bootstrap(df[mask], current[mask], new[mask])
        low, high = np.quantile(diffs, [0.05, 0.95])
        print(f"{day}: прирост по bootstrap {diffs.mean():+.4f}, интервал 5–95% [{low:+.4f}, {high:+.4f}], "
              f"доля повторов с приростом {(diffs > 0).mean():.2f}")
    people = df[(df.class_name == "person") & (df.veto == 0)].copy()
    people["зона"] = np.where((people.x1 + people.x2) / 2 < WINDOW_X, "окно", "зал")
    table = people.groupby(["зона", "day", "D", "P"]).error.agg(боксов="size", ошибок="sum")
    print("\nЛюди без вето: голос нашего детектора D и голос pose-модели P")
    print(table.to_string())
    fired = df[df.L == 1]
    print(f"\nФлаг ног: {len(fired)} боксов, из них ошибок {int(fired.error.sum())}; "
          f"по дням {fired.groupby('day').size().to_dict()}")
    path = ROOT / "oof" / f"{R.WORK.name}{R.members_suffix(args.members)}_{variant(args)}_veto.csv"
    if path.exists():
        print(f"{path} уже есть, не перезаписываю")
    else:
        df[["id"]].assign(score=new).to_csv(path, index=False)
        print("OOF:", path)


def submit(args):
    test = R.load("test.csv", dtype={"id": str})
    frames = [test.merge(R.agreement_features(test, pd.read_csv(R.pred_path("all", member))), on="id")
              for member in args.members]
    df = add_signals(test, "test", frames)
    current, new = scores(df, args.legs, args.skip_window)
    people = df[df.class_name == "person"]
    window = (people.x1 + people.x2) / 2 < WINDOW_X
    print(f"Людей {len(people)}; P=1 у {int(people.P.sum())}, из них у окна {int(people.P[window].sum())}; "
          f"флаг ног у {int(df.L.sum())}; "
          f"строк с изменённым скором {int((np.abs(new - current) > 1e-12).sum())} из {len(df)}")
    template = pd.read_csv(R.DATA / "sample_submission.csv", dtype={"id": str})
    assert set(template.id) == set(df.id) and df.id.is_unique
    path = ROOT / "submissions" / f"{R.WORK.name}{R.members_suffix(args.members)}_{variant(args)}_veto.csv"
    if path.exists():
        raise FileExistsError(f"{path} уже есть: удалите его вручную, если нужна пересборка")
    out = template[["id"]].merge(df[["id"]].assign(score=new), on="id", validate="one_to_one")
    assert np.isfinite(out.score).all() and out.score.between(0, 1).all()
    out.to_csv(path, index=False)
    print("Файл:", path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cmd", choices=["eval", "submit"])
    parser.add_argument("--tag", default="strong", help="рабочая папка outputs/redetect_<tag>")
    parser.add_argument("--members", nargs="+", default=["base"], help="участники ансамбля, как в redetect.py")
    parser.add_argument("--legs", action="store_true", help="добавить к вето флаг «ноги видны ниже рамки»")
    parser.add_argument("--skip-window", action="store_true", help="не учитывать голос pose-модели у окна слева")
    a = parser.parse_args()
    R.WORK = ROOT / "outputs" / (f"redetect_{a.tag}" if a.tag else "redetect")
    {"eval": evaluate, "submit": submit}[a.cmd](a)
