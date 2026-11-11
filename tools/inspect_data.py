"""Сводка по данным кейса 1: пишет reports/data_overview.md и контактные листы reports/*.png.

Запуск: python3 tools/inspect_data.py [путь к папке с данными]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "rascar-hack-case-1"
OUT = ROOT / "reports"
OUT.mkdir(exist_ok=True)

lines: list[str] = []


def h(title):
    lines.append(f"\n## {title}\n")


def md(df, index=True):
    """Markdown-таблица без зависимости от tabulate."""
    if isinstance(df, pd.Series):
        df = df.to_frame()
    if index:
        df = df.reset_index()
    fmt = lambda v: f"{v:.3f}".rstrip("0").rstrip(".") if isinstance(v, float) else str(v)
    lines.append("| " + " | ".join(map(str, df.columns)) + " |")
    lines.append("|" + "---|" * len(df.columns))
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(fmt(v) for v in row) + " |")
    lines.append("")


def p(text=""):
    lines.append(str(text))


def iou_matrix(a, b):
    """IoU между боксами a (n×4) и b (m×4) в формате x1,y1,x2,y2."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


# ---------- загрузка ----------
policy = json.loads((DATA / "scoring_policy.json").read_text())
aliases = policy.get("aliases", {})
train = pd.read_csv(DATA / "train.csv", dtype={"id": str})
test = pd.read_csv(DATA / "test.csv", dtype={"id": str})
human = pd.read_csv(DATA / "human_boxes_train.csv")
stream = {s: pd.read_csv(DATA / f"{s}_stream_predictions.csv", dtype={"detection_id": str}) for s in ("train", "test")}
frames = {s: pd.read_csv(DATA / f"{s}_frames.csv") for s in ("train", "test")}
context = {s: pd.read_csv(DATA / f"{s}_context.csv") for s in ("train", "test")}
mframes = pd.read_csv(DATA / "machine_frames.csv")
mpred = pd.read_csv(DATA / "machine_predictions.csv")
sample = pd.read_csv(DATA / "sample_submission.csv", dtype={"id": str})

for df in (train, test, human, mpred, *stream.values()):
    df["class_name"] = df["class_name"].replace(aliases)
for fr in (*frames.values(), mframes):
    fr["ts"] = pd.to_datetime(fr["timestamp"], utc=True, format="mixed")
    fr["day"] = fr["ts"].dt.strftime("%Y-%m-%d")

for name, df in (("train", train), ("test", test)):
    df["day"] = df.frame_id.map(frames[name].set_index("frame_id")["day"])
    df["w"] = df.x2 - df.x1
    df["h"] = df.y2 - df.y1
    df["area"] = df.w * df.h
    df["aspect"] = df.w / df.h.clip(lower=1)
    df["cy_bottom"] = df.y2

p("# Обзор данных: кейс 1 (аудит детектора)")
p(f"Источник: `{DATA.relative_to(ROOT) if DATA.is_relative_to(ROOT) else DATA}`. Отчёт сгенерирован `tools/inspect_data.py`.")

# ---------- размеры ----------
h("Размеры")
sizes = pd.DataFrame([
    ("train.csv (детекции целевых кадров)", len(train), train.frame_id.nunique()),
    ("test.csv", len(test), test.frame_id.nunique()),
    ("sample_submission.csv", len(sample), None),
    ("human_boxes_train.csv", len(human), human.frame_id.nunique()),
    ("train_stream_predictions.csv", len(stream["train"]), stream["train"].frame_id.nunique()),
    ("test_stream_predictions.csv", len(stream["test"]), stream["test"].frame_id.nunique()),
    ("machine_predictions.csv", len(mpred), mpred.frame_id.nunique()),
], columns=["файл", "строк", "кадров"])
md(sizes, index=False)
p(f"- id из sample_submission совпадают с test.csv: **{set(sample.id) == set(test.id)}**")
p(f"- Все детекции train.csv есть в train_stream (по id): **{train.id.isin(stream['train'].detection_id).mean():.3f}**")
p(f"- Все детекции test.csv есть в test_stream (по id): **{test.id.isin(stream['test'].detection_id).mean():.3f}**")
missing_img = sum(not (DATA / ip).exists() for fr in (*frames.values(), mframes) for ip in fr.image_path)
p(f"- Отсутствующих изображений: **{missing_img}**")
p(f"- Классы в оценке: {policy['classes']}; порог IoU {policy['correct_iou']}; исключены в test: {policy['test_excluded_classes']}")

# ---------- метка ----------
h("Метка error в train (без table)")
tr = train[train.class_name != "table"]
by_day = tr.groupby("day").agg(кадров=("frame_id", "nunique"), боксов=("id", "size"), ошибок=("error", "sum"), доля=("error", "mean"))
by_day["AP 1-conf"] = [average_precision_score(g.error, 1 - g.confidence) for _, g in tr.groupby("day")]
md(by_day.round(3))
p(f"Всего: {len(tr)} боксов, ошибок {int(tr.error.sum())} ({tr.error.mean():.3f}). Случайная очередь даёт AP ≈ доле ошибок. Table в train: {int((train.class_name=='table').sum())} боксов, ошибок {int(train[train.class_name=='table'].error.sum())}.")

h("По классам: train vs test")
cls = tr.groupby("class_name").agg(train_боксов=("id", "size"), ошибок=("error", "sum"), доля_ошибок=("error", "mean"), conf_ср=("confidence", "mean"))
cls["AP 1-conf"] = [average_precision_score(g.error, 1 - g.confidence) if 0 < g.error.sum() < len(g) else np.nan for _, g in tr.groupby("class_name")]
cls = cls.join(test.groupby("class_name").size().rename("test_боксов"), how="outer").fillna(0)
md(cls.sort_values("train_боксов", ascending=False).round(3))

h("Confidence vs ошибка (train без table)")
bins = pd.cut(tr.confidence, [0, .3, .4, .5, .6, .7, .8, .9, 1.0])
md(tr.groupby(bins, observed=True).agg(боксов=("id", "size"), доля_ошибок=("error", "mean")).round(3))
p(f"Минимальный confidence в train: {train.confidence.min():.2f}, в test: {test.confidence.min():.2f}, в пуле: {mpred.confidence.min():.2f}")

# ---------- геометрия ----------
h("Геометрия боксов по классам (медианы) и ошибка")
geo = tr.groupby(["class_name", "error"])[["w", "h", "area", "aspect", "cy_bottom"]].median().round(1)
md(geo)

# ---------- почему ошибки: сопоставление с ручной разметкой ----------
h("Разбор ошибок train: на что похожа ошибочная детекция")
hb = human.copy()
fw, fh = 1280, 720
hb["x1"] = (hb.cx - hb.w / 2) * fw
hb["x2"] = (hb.cx + hb.w / 2) * fw
hb["y1"] = (hb.cy - hb.h / 2) * fh
hb["y2"] = (hb.cy + hb.h / 2) * fh
kinds = []
for fid, g in tr.groupby("frame_id"):
    hf = hb[hb.frame_id == fid]
    preds = g[["x1", "y1", "x2", "y2"]].to_numpy()
    m_all = iou_matrix(preds, hf[["x1", "y1", "x2", "y2"]].to_numpy())
    same_cls = (g.class_name.to_numpy()[:, None] == hf.class_name.to_numpy()[None, :])
    m_pp = iou_matrix(preds, preds)
    np.fill_diagonal(m_pp, 0)
    same_pp = g.class_name.to_numpy()[:, None] == g.class_name.to_numpy()[None, :]
    for i, row in enumerate(g.itertuples()):
        if row.error == 0:
            continue
        best_same = (m_all[i] * same_cls[i]).max() if m_all.shape[1] else 0
        best_other = (m_all[i] * ~same_cls[i]).max() if m_all.shape[1] else 0
        dup = (m_pp[i] * same_pp[i]).max() if len(g) > 1 else 0
        if dup >= 0.5 and best_same >= 0.6:
            k = "дубль (рядом свой бокс того же класса, эталон занят)"
        elif best_same >= 0.6:
            k = "IoU≥0.6, но эталон отдан другому предсказанию"
        elif best_same >= 0.2:
            k = "геометрия (тот же класс, 0.2≤IoU<0.6)"
        elif best_other >= 0.5:
            k = "чужой класс (IoU≥0.5 с эталоном другого класса)"
        else:
            k = "ложное срабатывание (рядом эталона нет)"
        kinds.append((row.class_name, k))
kinds = pd.DataFrame(kinds, columns=["class_name", "тип"])
md(kinds.groupby(["тип"]).size().rename("ошибок").sort_values(ascending=False).to_frame())
md(pd.crosstab(kinds.class_name, kinds.тип))
p("Типы определены эвристикой по ручным боксам (не официальные), нужны для понимания, что ловить.")
p(f"\nИсточник ручных боксов: {human.source.value_counts().to_dict()}; evaluated_class: {human.evaluated_class.value_counts().to_dict()}")

# ---------- время ----------
h("Время и контекст")
for name in ("train", "test"):
    fr = frames[name].sort_values("ts")
    ctx = context[name]
    p(f"**{name}**: кадров в frames {len(fr)}, целевых {len(ctx)}, дни {fr.day.value_counts().sort_index().to_dict()}; "
      f"нет previous: {ctx.previous.isna().sum()}, нет next: {ctx.next.isna().sum()}")
    tgt = fr[fr.frame_id.isin(ctx.frame_id)]
    p(f"  целевые кадры по часам UTC: {tgt.ts.dt.hour.value_counts().sort_index().to_dict()}")
gaps = mframes.sort_values("ts").groupby("day").ts.diff().dt.total_seconds().dropna()
p(f"\nПул: {len(mframes)} кадров, дни {mframes.day.value_counts().sort_index().to_dict()}; интервалы, с: медиана {gaps.median():.1f}, "
  f"доля ровно ~30с {((gaps>25)&(gaps<35)).mean():.3f}, >60с: {(gaps>60).sum()} раз, макс {gaps.max():.0f}")
overlap = set(frames["train"].frame_id) & set(mframes.frame_id)
p(f"Пересечение кадров train(+контекст) с пулом: {len(overlap)}")

# ---------- временная устойчивость ----------
h("Временная устойчивость: есть ли похожий бокс того же класса на соседних кадрах")
sp = stream["train"]
nbr = context["train"].set_index("frame_id")
rows = []
for r in tr.itertuples():
    best = []
    for col in ("previous", "next"):
        nf = nbr.at[r.frame_id, col] if r.frame_id in nbr.index else np.nan
        if pd.isna(nf):
            best.append(np.nan)
            continue
        cand = sp[(sp.frame_id == nf) & (sp.class_name == r.class_name)]
        m = iou_matrix(np.array([[r.x1, r.y1, r.x2, r.y2]]), cand[["x1", "y1", "x2", "y2"]].to_numpy())
        best.append(m.max() if m.size else 0.0)
    rows.append((r.id, r.error, *best))
tmp = pd.DataFrame(rows, columns=["id", "error", "iou_prev", "iou_next"])
tmp["iou_nb_max"] = tmp[["iou_prev", "iou_next"]].max(axis=1)
tmp["iou_nb_min"] = tmp[["iou_prev", "iou_next"]].min(axis=1)
bins = pd.cut(tmp.iou_nb_max, [-0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0])
md(tmp.groupby(bins, observed=True).agg(боксов=("id", "size"), доля_ошибок=("error", "mean")).round(3))
for col in ("iou_nb_max", "iou_nb_min"):
    v = tmp.dropna(subset=[col])
    p(f"- AP по `1 - {col}`: {average_precision_score(v.error, 1 - v[col]):.3f} (на {len(v)} боксах)")

h("Дубли внутри кадра: макс IoU с другим боксом того же класса")
dups = []
for fid, g in tr.groupby("frame_id"):
    pr = g[["x1", "y1", "x2", "y2"]].to_numpy()
    m = iou_matrix(pr, pr)
    np.fill_diagonal(m, 0)
    m = m * (g.class_name.to_numpy()[:, None] == g.class_name.to_numpy()[None, :])
    dups += list(zip(g.error, m.max(axis=1)))
dups = pd.DataFrame(dups, columns=["error", "max_iou_same"])
md(dups.groupby(pd.cut(dups.max_iou_same, [-0.01, 0.01, 0.3, 0.5, 0.7, 1.0]), observed=True).agg(боксов=("error", "size"), доля_ошибок=("error", "mean")).round(3))

# ---------- контактные листы ----------
h("Контактные листы")
COL = {0: (40, 200, 70), 1: (230, 40, 40)}


def sheet(fids, name, df, show_human=False, n_cols=3, scale=0.45):
    tiles = []
    for fid in fids:
        path = DATA / f"{fid}.jpg"
        if not path.exists():
            continue
        im = Image.open(path).convert("RGB")
        d = ImageDraw.Draw(im)
        if show_human:
            for r in hb[(hb.frame_id == fid) & (hb.class_name != "table")].itertuples():
                d.rectangle([r.x1, r.y1, r.x2, r.y2], outline=(60, 140, 255), width=2)
        for r in df[df.frame_id == fid].itertuples():
            c = COL.get(getattr(r, "error", 0), (255, 200, 0))
            d.rectangle([r.x1, r.y1, r.x2, r.y2], outline=c, width=3)
            d.text((r.x1 + 3, r.y1 + 2), f"{r.class_name} {r.confidence:.2f}", fill=c)
        d.text((8, 8), fid[-15:], fill=(255, 255, 0))
        tiles.append(im.resize((int(im.width * scale), int(im.height * scale))))
    if not tiles:
        return
    tw, th = tiles[0].size
    rows_n = (len(tiles) + n_cols - 1) // n_cols
    canvas = Image.new("RGB", (tw * n_cols, th * rows_n))
    for i, t in enumerate(tiles):
        canvas.paste(t, ((i % n_cols) * tw, (i // n_cols) * th))
    canvas.save(OUT / name, quality=85)
    p(f"- `reports/{name}`: {len(tiles)} кадров")


err_frames = tr.groupby("frame_id").error.sum().sort_values(ascending=False)
sheet(list(err_frames.index[:9]), "sheet_train_most_errors.jpg", tr, show_human=True)
rng = np.random.default_rng(0)
sheet(list(rng.choice(tr.frame_id.unique(), 9, replace=False)), "sheet_train_random.jpg", tr, show_human=True)
sheet(list(rng.choice(test.frame_id.unique(), 6, replace=False)), "sheet_test_random.jpg", test)
p("Цвета: зелёный — верная детекция, красный — ошибочная, синий — ручной бокс, жёлтый — test (метки нет).")

(OUT / "data_overview.md").write_text("\n".join(lines), encoding="utf-8")
print("written", OUT / "data_overview.md")
