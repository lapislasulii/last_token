"""«Второе мнение»: дообучаем YOLO на ручных боксах train и сверяем с ним боксы исходного детектора.

Идея: error = «бокс детектора не сопоставился с ручным боксом того же класса при IoU >= 0.6».
Если наш детектор учится рисовать боксы по конвенции разметчиков, расхождение с ним — сигнал ошибки.

Честная проверка по дням: детектор, обученный на одном UTC-дне, оценивает боксы другого дня.

Команды (из корня репо, через .venv/bin/python):
  build    — собрать YOLO-датасеты: day_20261025, day_20261026, all (оба дня, для test)
  train    — дообучить детектор на каждом наборе
  predict  — прогнать детекторы по нужным кадрам, сохранить детекции
  eval     — признаки согласия + AP по дням против бейзлайна, OOF-файл для evaluate_predictions.py
  submit   — предсказания для test детектором, обученным на обоих днях (набор all)

Сильный рецепт (NVIDIA): --tag strong --extra-classes --weights yolo11x.pt --epochs 250 --batch 4 --device 0.
--tag кладёт всё в outputs/redetect_<tag>, старые результаты не затираются.
--extra-classes добавляет в обучение столы и «отвлекающие» ручные классы (телефоны, тарелки, вещи):
детектор учится отличать их от чашек и кистей; в согласии участвуют только оцениваемые классы.
"""
import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "rascar-hack-case-1"
WORK = ROOT / "outputs" / "redetect"
DAYS = ["20261025", "20261026"]
CLASSES = ["person", "nude_hands", "cup", "tray"]  # оцениваемые классы, которые реально встречаются
TRAIN_CLASSES = CLASSES  # с --extra-classes сюда добавляется EXTRA
EXTRA = ["table", "phone", "empty_plate", "full_plate", "personal_items", "bottle", "hair", "uniform_not_ok"]

policy = json.loads((DATA / "scoring_policy.json").read_text())
ALIASES = policy["aliases"]


def load(name, **kw):
    df = pd.read_csv(DATA / name, **kw)
    if "class_name" in df:
        df["class_name"] = df["class_name"].replace(ALIASES)
    return df


def frame_day(split):
    fr = load(f"{split}_frames.csv", dtype={"day": str})
    return fr.set_index("frame_id")["day"]


# ---------------------------------------------------------------- build
def build():
    human = load("human_boxes_train.csv")
    human = human[human.class_name.isin(TRAIN_CLASSES)]
    day_of = frame_day("train")
    targets = load("train_context.csv").frame_id  # кадры с ручной разметкой
    sets = {f"day_{d}": [f for f in targets if day_of[f] == d] for d in DAYS}
    sets["all"] = list(targets)
    for name, frames in sets.items():
        root = WORK / "yolo" / name
        for sub in ("images", "labels"):
            (root / sub).mkdir(parents=True, exist_ok=True)
        for fid in frames:
            img = root / "images" / f"{fid}.jpg"
            if not img.exists():
                try:
                    os.symlink(DATA / f"{fid}.jpg", img)
                except OSError:
                    # Windows may disallow symlinks without Developer Mode.
                    shutil.copy2(DATA / f"{fid}.jpg", img)
            rows = human[human.frame_id == fid]
            (root / "labels" / f"{fid}.txt").write_text("".join(
                f"{TRAIN_CLASSES.index(r.class_name)} {r.cx:.6f} {r.cy:.6f} {r.w:.6f} {r.h:.6f}\n" for r in rows.itertuples()))
        # val = тот же набор: другой день нельзя использовать для отбора эпох
        (root / "data.yaml").write_text(
            f"path: {root}\ntrain: images\nval: images\nnames:\n" + "".join(f"  {i}: {c}\n" for i, c in enumerate(TRAIN_CLASSES)))
        print(name, len(frames), "кадров,", int(human.frame_id.isin(frames).sum()), "боксов")


# ---------------------------------------------------------------- train
def train(args):
    from ultralytics import YOLO
    for name in args.sets:
        if args.resume:
            checkpoint = WORK / "runs" / name / "weights" / "last.pt"
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            YOLO(checkpoint).train(resume=True, device=args.device)
            continue
        model = YOLO(args.weights)  # публичные COCO-веса Ultralytics, скачиваются автоматически
        model.train(data=str(WORK / "yolo" / name / "data.yaml"), epochs=args.epochs, imgsz=args.imgsz,
                    batch=args.batch, device=args.device, project=str(WORK / "runs"), name=name, exist_ok=True,
                    seed=0, deterministic=True, val=False, patience=0, plots=False, flipud=0.0, verbose=False, workers=2)


# ---------------------------------------------------------------- predict
def predict(args):
    from ultralytics import YOLO
    jobs = {  # какой детектор на каких кадрах
        "day_20261025": ("train", "20261026"),
        "day_20261026": ("train", "20261025"),
        "all": ("test", None),
    }
    for name in args.sets:
        split, day = jobs[name]
        frames = load(f"{split}_frames.csv", dtype={"day": str})
        if day:
            frames = frames[frames.day == day]
        model = YOLO(WORK / "runs" / name / "weights" / "last.pt")
        out = []
        paths = [str(DATA / p) for p in frames.image_path]
        # Bound memory even when the backend interprets a list as one batch.
        for start in range(0, len(paths), args.predict_batch):
            results = model.predict(paths[start:start + args.predict_batch], imgsz=args.imgsz,
                                    conf=0.01, iou=0.6, device=args.device,
                                    stream=True, verbose=False, max_det=200)
            for fid, res in zip(frames.frame_id.iloc[start:start + args.predict_batch], results):
                b = res.boxes
                for xyxy, c, k in zip(b.xyxy.cpu().numpy(), b.conf.cpu().numpy(), b.cls.cpu().numpy().astype(int)):
                    out.append((fid, model.names[int(k)], float(c), *map(float, xyxy)))
            print(f"{name}: {min(start + args.predict_batch, len(paths))}/{len(paths)} frames", flush=True)
        df = pd.DataFrame(out, columns=["frame_id", "class_name", "confidence", "x1", "y1", "x2", "y2"])
        df.to_csv(WORK / f"pred_{name}.csv", index=False)
        print(name, "->", len(frames), "кадров,", len(df), "детекций")


# ---------------------------------------------------------------- features
def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ix = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    iy = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = ix * iy
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def agreement_features(boxes, ours, min_conf=0.05):
    """Для каждого исходного бокса — насколько с ним согласен наш детектор."""
    ours = ours[ours.confidence >= min_conf]
    feats = []
    xy = ["x1", "y1", "x2", "y2"]
    for fid, g in boxes.groupby("frame_id"):
        o = ours[ours.frame_id == fid]
        m = iou_matrix(g[xy].to_numpy(), o[xy].to_numpy())
        same = g.class_name.to_numpy()[:, None] == o.class_name.to_numpy()[None, :]
        oc = o.confidence.to_numpy()
        for i, r in enumerate(g.itertuples()):
            ms = np.where(same[i], m[i], 0.0) if len(o) else np.zeros(0)
            mo = np.where(~same[i], m[i], 0.0) if len(o) else np.zeros(0)
            j = int(ms.argmax()) if ms.size else -1
            ok = ms >= 0.6
            feats.append(dict(
                id=r.id,
                iou_same=ms.max() if ms.size else 0.0,                         # лучший IoU с нашим боксом того же класса
                conf_at_best=oc[j] if j >= 0 and ms[j] > 0 else 0.0,            # уверенность нашего в этом боксе
                match_conf=(oc * ok).max() if ms.size else 0.0,                 # макс. уверенность среди «пар» с IoU>=0.6
                iou_other=mo.max() if mo.size else 0.0,                         # совпадение с нашим боксом ДРУГОГО класса
                other_conf=oc[int(mo.argmax())] if mo.size and mo.max() > 0 else 0.0,
                n_same_03=int((ms >= 0.3).sum()) if ms.size else 0,             # сколько наших боксов рядом
            ))
    return pd.DataFrame(feats)


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def rank01(x):
    return pd.Series(x).rank(pct=True).to_numpy()


# ---------------------------------------------------------------- eval
def evaluate(args):
    train_df = load("train.csv", dtype={"id": str})
    train_df = train_df[train_df.class_name != "table"].copy()
    train_df["day"] = train_df.frame_id.map(frame_day("train"))
    parts = []
    for det_day, eval_day in (("20261025", "20261026"), ("20261026", "20261025")):
        ours = pd.read_csv(WORK / f"pred_day_{det_day}.csv")
        g = train_df[train_df.day == eval_day]
        parts.append(g.merge(agreement_features(g, ours), on="id"))
    df = pd.concat(parts, ignore_index=True)
    df.to_csv(WORK / "oof_features.csv", index=False)

    base = 1 - df.confidence
    agree = df.match_conf  # 0, если наш детектор не нарисовал совпадающий бокс
    candidates = {
        "baseline 1-conf": base,
        "1 - iou_same": 1 - df.iou_same,
        "1 - match_conf": 1 - agree,
        "1 - iou_same*conf_at_best": 1 - df.iou_same * df.conf_at_best,
        "rank-avg(base, 1-match_conf)": (rank01(base) + rank01(1 - agree)) / 2,
        "rank-avg(base, 1-iou_same)": (rank01(base) + rank01(1 - df.iou_same)) / 2,
        "veto: (base + [iou_same<0.6]) / 2": primary_score(df),
        "veto, только person": (base + ((df.iou_same < 0.6) & (df.class_name == "person"))) / 2,
    }
    rows = []
    for name, s in candidates.items():
        s = pd.Series(np.asarray(s, dtype=float), index=df.index)
        aps = {d: average_precision_score(df.error[df.day == d], s[df.day == d]) for d in DAYS}
        rows.append(dict(method=name, **{f"AP {d}": v for d, v in aps.items()}, mean=np.mean(list(aps.values()))))
    res = pd.DataFrame(rows)
    print(res.round(4).to_string(index=False))
    print("\nAP по классам (1 - match_conf vs baseline):")
    for c, g in df.groupby("class_name"):
        if 0 < g.error.sum() < len(g):
            print(f"  {c:11s} n={len(g):3d}  base={average_precision_score(g.error, 1 - g.confidence):.3f}"
                  f"  ours={average_precision_score(g.error, 1 - g.match_conf):.3f}")
    res.to_csv(WORK / "eval_heuristics.csv", index=False)
    oof = ROOT / "oof" / f"{WORK.name}_veto.csv"
    oof.parent.mkdir(exist_ok=True)
    df[["id"]].assign(score=primary_score(df)).to_csv(oof, index=False)
    print("OOF основного варианта:", oof)


def primary_score(df):
    """Заранее выбранный вариант: бокс без пары IoU>=0.6 у нашего детектора поднимается над остальными."""
    return ((1 - df.confidence) + (df.iou_same < 0.6)) / 2


# ---------------------------------------------------------------- submit
def submit():
    test = load("test.csv", dtype={"id": str})
    ours = pd.read_csv(WORK / "pred_all.csv")
    df = test.merge(agreement_features(test, ours), on="id")
    df.to_csv(WORK / "test_features.csv", index=False)
    template = pd.read_csv(DATA / "sample_submission.csv", dtype={"id": str})
    assert set(template.id) == set(df.id) and df.id.is_unique
    path = ROOT / "submissions" / f"{WORK.name}_veto.csv"
    path.parent.mkdir(exist_ok=True)
    template[["id"]].merge(df[["id"]].assign(score=primary_score(df)), on="id", validate="one_to_one").to_csv(path, index=False)
    print("Доля боксов без пары по классам:")
    print((df.iou_same < 0.6).groupby(df.class_name).mean().round(2).to_string())
    print("Файл:", path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "train", "predict", "eval", "submit"])
    ap.add_argument("--sets", nargs="+", default=["day_20261025", "day_20261026"])
    ap.add_argument("--weights", default="yolo11s.pt")
    ap.add_argument("--resume", action="store_true", help="продолжить выбранные наборы с сохранённого last.pt")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--predict-batch", type=int, default=4)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--tag", default="", help="суффикс рабочей папки: outputs/redetect_<tag>")
    ap.add_argument("--extra-classes", action="store_true", help="учить ещё столы и отвлекающие классы")
    a = ap.parse_args()
    if a.tag:
        WORK = ROOT / "outputs" / f"redetect_{a.tag}"
    if a.extra_classes:
        TRAIN_CLASSES = CLASSES + EXTRA
    WORK.mkdir(parents=True, exist_ok=True)
    {"build": lambda: build(), "train": lambda: train(a), "predict": lambda: predict(a), "eval": lambda: evaluate(a),
     "submit": lambda: submit()}[a.cmd]()
