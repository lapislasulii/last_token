"""SAM как «второй разметчик» краёв боксов: голос G за то, что рамка заказчика проведена не по объекту.

Зачем. Доля D и вето ловят «объекта нет», но плохо ловят «объект есть, края не те»: кисти с IoU 0.50–0.59
против ручной рамки, люди с рамкой, обрезанной снизу. SAM (Segment Anything) по подсказке-рамке рисует маску
объекта; плотная рамка маски — оценка того, как разметчик обвёл бы видимый объект. На наших метках SAM не
обучается, поэтому его голос честно проверяется сразу на обоих днях train, как вето.

  G = 1[iou_sam < 0.6],  iou_sam = IoU(рамка заказчика, рамка маски SAM).

Подсказки (заданы до просмотра результатов):
  кисти, чашки — квадратная вырезка 4× от большей стороны бокса, подносы — 2×; вырезка увеличивается
                 до 1024 px по длинной стороне, подсказка — бокс в координатах вырезки;
  люди         — полный кадр, бокс расширен вниз на 60% и вверх на 10% высоты (чтобы маска дотянулась
                 до видимых ног) плюс положительная точка в центре исходного бокса.
  Маска: порог 0.5 (Ultralytics режет логиты по 0); из нескольких компонент берётся та, что содержит центр
  исходного бокса, иначе самая большая. Пустая маска -> iou_sam = 0.

Варианты формулы (заданы заранее, новых не добавлять). Опора — лидер tools/person_vote.py
с --skip-window --legs: score = (1 − confidence + D' + 2·max(V, L)) / 4.
  A — G как ещё один участник в D' для person, nude_hands, cup, tray;
  B — то же, только для person и nude_hands;
  C — мягкий голос g = clip((0.75 − iou_sam) / 0.3, 0, 1) для person, nude_hands, cup.
  d_new = (votes + P·[человек вне окна] + g) / (n_members + [P учтён] + 1) для классов варианта.

Веса (публичные): https://github.com/ultralytics/assets/releases/download/v8.4.0/sam2_t.pt  (положить в weights/)

Команды (из корня репозитория):
  python tools/sam_geometry.py run    --split train --device cpu [--model weights/sam2_t.pt] [--rerun]
      кэш outputs/sam_geometry/{split}_sam.csv
  python tools/sam_geometry.py review                       # контактный лист в outputs/sam_geometry/review/
  python tools/sam_geometry.py diag   --members base         # ранний сигнал без вето: (1 − conf + D) / 2 против + G
  python tools/sam_geometry.py eval   --members base s1 s2   # полная проверка против лидера; нужен outputs/expert_veto/
  python tools/sam_geometry.py submit --members base s1 s2 --variant A
      файл в submissions/ (существующий не перезаписывается; в Kaggle не отправляется)
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy import ndimage
from sklearn.metrics import average_precision_score

import person_vote as PV
import redetect as R

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "sam_geometry"
BOX = ["x1", "y1", "x2", "y2"]
SAM_BOX = ["sx1", "sy1", "sx2", "sy2"]
CLASSES = ["person", "nude_hands", "cup", "tray"]
CROP_FACTOR = {"nude_hands": 4.0, "cup": 4.0, "tray": 2.0}
CROP_SIDE = 1024
PERSON_DOWN, PERSON_UP = 0.60, 0.10
G_IOU = 0.6
VARIANTS = {
    "A": (("person", "nude_hands", "cup", "tray"), "hard"),
    "B": (("person", "nude_hands"), "hard"),
    "C": (("person", "nude_hands", "cup"), "soft"),
}


# ---------------------------------------------------------------- геометрия (чистые функции, покрыты тестами)
def box_iou(a, b):
    """IoU двух рамок x1, y1, x2, y2."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def crop_window(box, factor, width, height):
    """Квадрат со стороной factor × большая сторона бокса, центр в центре бокса, обрезка по кадру.
    Возвращает целые x0, y0, x1, y1 (x1, y1 не включаются)."""
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    half = factor * max(box[2] - box[0], box[3] - box[1]) / 2
    x0, y0 = max(0, int(np.floor(cx - half))), max(0, int(np.floor(cy - half)))
    x1, y1 = min(width, int(np.ceil(cx + half))), min(height, int(np.ceil(cy + half)))
    return x0, y0, x1, y1


def crop_scale(window, side=CROP_SIDE):
    """Во сколько раз увеличить вырезку, чтобы длинная сторона стала side."""
    return side / max(window[2] - window[0], window[3] - window[1])


def to_crop(box, window, scale):
    """Рамка из координат кадра в координаты увеличенной вырезки."""
    return [(box[0] - window[0]) * scale, (box[1] - window[1]) * scale,
            (box[2] - window[0]) * scale, (box[3] - window[1]) * scale]


def from_crop(box, window, scale):
    """Рамка из координат увеличенной вырезки обратно в координаты кадра."""
    return [box[0] / scale + window[0], box[1] / scale + window[1],
            box[2] / scale + window[0], box[3] / scale + window[1]]


def person_prompt(box, width, height):
    """Подсказка для человека: бокс, расширенный вниз на 60% и вверх на 10% высоты, и точка в центре."""
    h = box[3] - box[1]
    prompt = [max(0.0, box[0]), max(0.0, box[1] - PERSON_UP * h),
              min(float(width), box[2]), min(float(height), box[3] + PERSON_DOWN * h)]
    point = [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]
    return prompt, point


def mask_to_box(mask, center):
    """Плотная рамка маски: компонента, содержащая центр (x, y), иначе самая большая.
    Возвращает (рамка x1, y1, x2, y2 с x2, y2 не включительно, площадь компоненты в пикселях) или (None, 0)."""
    mask = np.asarray(mask) > 0.5
    if not mask.any():
        return None, 0
    labels, count = ndimage.label(mask)
    cx = int(np.clip(np.floor(center[0]), 0, mask.shape[1] - 1))
    cy = int(np.clip(np.floor(center[1]), 0, mask.shape[0] - 1))
    keep = labels[cy, cx]
    if keep == 0:
        sizes = ndimage.sum_labels(mask, labels, index=np.arange(1, count + 1))
        keep = int(np.argmax(sizes)) + 1
    ys, xs = np.nonzero(labels == keep)
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)], int(len(xs))


def soft_vote(iou_sam):
    """Мягкий голос варианта C: 0 при IoU >= 0.75, 1 при IoU <= 0.45."""
    return np.clip((0.75 - np.asarray(iou_sam, dtype=float)) / 0.3, 0.0, 1.0)


# ---------------------------------------------------------------- прогон SAM
def load_split(split):
    df = R.load(f"{split}.csv", dtype={"id": str})
    return df[df.class_name.isin(CLASSES)].reset_index(drop=True)


def sam_row(row, sam_box, area):
    """Строка кэша: рамка SAM, площадь маски, iou_sam, насколько рамка SAM ниже рамки заказчика."""
    box = [row.x1, row.y1, row.x2, row.y2]
    if sam_box is None:
        return dict(id=row.id, sx1=np.nan, sy1=np.nan, sx2=np.nan, sy2=np.nan, mask_area=0,
                    iou_sam=0.0, bottom_ext=np.nan)
    return dict(id=row.id, sx1=sam_box[0], sy1=sam_box[1], sx2=sam_box[2], sy2=sam_box[3], mask_area=area,
                iou_sam=box_iou(box, sam_box), bottom_ext=(sam_box[3] - row.y2) / (row.y2 - row.y1))


def predict_masks(model, image_bgr, bboxes, points, device):
    """Маски SAM по подсказкам-рамкам (и положительным точкам). Пустые маски Ultralytics может отбросить."""
    extra = {} if points is None else dict(points=[[p] for p in points], labels=[[1]] * len(points))
    result = model.predict(image_bgr, bboxes=bboxes, verbose=False, device=device, **extra)[0]
    return [] if result.masks is None else list(result.masks.data.cpu().numpy())


def first_mask(masks, shape):
    """Маска единственной подсказки или пустая маска нужного размера."""
    return masks[0] if len(masks) else np.zeros(shape, dtype=bool)


def run(args):
    path = OUT / f"{args.split}_sam.csv"
    if path.exists() and not args.rerun:
        print(f"{path} уже есть; пересчёт только с --rerun")
        return
    from ultralytics import SAM
    model = SAM(str(ROOT / args.model))
    rows = load_split(args.split)
    frames = rows.frame_id.unique()
    out, empty, start = [], [], time.time()
    for number, frame in enumerate(frames):
        with Image.open(R.DATA / f"{frame}.jpg") as source:
            image = source.convert("RGB")
        width, height = image.size
        full_bgr = np.ascontiguousarray(np.array(image)[:, :, ::-1])
        group = rows[rows.frame_id == frame]
        people = group[group.class_name == "person"]
        if len(people):  # все люди кадра одним вызовом: одно кодирование кадра
            prompts = [person_prompt([r.x1, r.y1, r.x2, r.y2], width, height) for r in people.itertuples()]
            masks = predict_masks(model, full_bgr, [p[0] for p in prompts], [p[1] for p in prompts], args.device)
            if len(masks) != len(prompts):  # Ultralytics отбросил пустую маску: по одному, чтобы не спутать порядок
                masks = [first_mask(predict_masks(model, full_bgr, [p[0]], [p[1]], args.device), (height, width))
                         for p in prompts]
            for row, (_, point), mask in zip(people.itertuples(), prompts, masks):
                sam_box, area = mask_to_box(mask, point)
                empty += [row.id] if sam_box is None else []
                out.append(sam_row(row, sam_box, area))
        for row in group[group.class_name != "person"].itertuples():
            box = [row.x1, row.y1, row.x2, row.y2]
            window = crop_window(box, CROP_FACTOR[row.class_name], width, height)
            scale = crop_scale(window)
            size = (round((window[2] - window[0]) * scale), round((window[3] - window[1]) * scale))
            crop = image.crop(window).resize(size, Image.BICUBIC)
            prompt = to_crop(box, window, scale)
            mask = first_mask(predict_masks(model, np.ascontiguousarray(np.array(crop)[:, :, ::-1]), [prompt],
                                            None, args.device), size[::-1])
            center = [(prompt[0] + prompt[2]) / 2, (prompt[1] + prompt[3]) / 2]
            crop_box, area = mask_to_box(mask, center)
            sam_box = None if crop_box is None else from_crop(crop_box, window, scale)
            empty += [row.id] if sam_box is None else []
            out.append(sam_row(row, sam_box, int(round(area / scale ** 2))))
        if (number + 1) % 5 == 0 or number + 1 == len(frames):
            print(f"{args.split}: кадр {number + 1}/{len(frames)}, боксов {len(out)}/{len(rows)}, "
                  f"{time.time() - start:.0f} с", flush=True)
    if empty:
        print(f"Пустая маска у {len(empty)} боксов (iou_sam = 0): {', '.join(map(str, empty))}")
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out)[["id", *SAM_BOX, "mask_area", "iou_sam", "bottom_ext"]].to_csv(path, index=False)
    (OUT / f"{args.split}_sam.json").write_text(json.dumps(
        dict(model=args.model, device=str(args.device), seconds=round(time.time() - start), boxes=len(out),
             empty_masks=len(empty)), ensure_ascii=False, indent=1))
    print("Кэш:", path)


def with_sam(df, split):
    sam = pd.read_csv(OUT / f"{split}_sam.csv", dtype={"id": str})
    df = df.merge(sam, on="id", how="left", validate="one_to_one")
    assert df.iou_sam.notna().all(), f"нет SAM для части боксов {split}: запустите run --split {split}"
    df["G"] = (df.iou_sam < G_IOU).astype(float)
    df["window"] = (df.x1 + df.x2) / 2 < PV.WINDOW_X
    return df


# ---------------------------------------------------------------- контактный лист
def human_boxes():
    human = R.load("human_boxes_train.csv")
    with Image.open(R.DATA / f"{human.frame_id.iloc[0]}.jpg") as image:
        width, height = image.size
    human["x1"], human["x2"] = (human.cx - human.w / 2) * width, (human.cx + human.w / 2) * width
    human["y1"], human["y2"] = (human.cy - human.h / 2) * height, (human.cy + human.h / 2) * height
    return human


def review(args):
    df = with_sam(load_split("train"), "train")
    rng = np.random.default_rng(0)
    picks = []
    for cls in CLASSES:
        for error in (1, 0):
            pool = df[(df.class_name == cls) & (df.error == error)]
            picks.append(pool.iloc[rng.choice(len(pool), min(2, len(pool)), replace=False)])
    picks = pd.concat(picks)
    human = human_boxes()
    tile, tiles = 320, []
    for row in picks.itertuples():
        with Image.open(R.DATA / f"{row.frame_id}.jpg") as source:
            image = source.convert("RGB")
        theirs = human[(human.frame_id == row.frame_id) & (human.class_name == row.class_name)]
        box = [row.x1, row.y1, row.x2, row.y2]
        ious = [box_iou(box, h) for h in theirs[BOX].to_numpy()]
        best = theirs[BOX].to_numpy()[int(np.argmax(ious))] if len(ious) else None
        boxes = [box, [row.sx1, row.sy1, row.sx2, row.sy2]] + ([list(best)] if best is not None else [])
        allb = np.array([b for b in boxes if np.isfinite(b).all()])
        lo, hi = allb[:, :2].min(axis=0), allb[:, 2:].max(axis=0)
        pad = 0.25 * (hi - lo).max() + 10
        window = (int(max(0, lo[0] - pad)), int(max(0, lo[1] - pad)),
                  int(min(image.width, hi[0] + pad)), int(min(image.height, hi[1] + pad)))
        scale = tile / max(window[2] - window[0], window[3] - window[1])
        view = image.crop(window).resize((round((window[2] - window[0]) * scale),
                                          round((window[3] - window[1]) * scale)), Image.BICUBIC)
        canvas = Image.new("RGB", (tile, tile + 30), "black")
        canvas.paste(view, (0, 0))
        draw = ImageDraw.Draw(canvas)
        for b, color, w in zip(boxes, ("yellow", "cyan", "lime"), (2, 2, 1)):
            if np.isfinite(b).all():
                draw.rectangle(to_crop(b, window, scale), outline=color, width=w)
        draw.text((4, tile + 2), f"{row.class_name} err={row.error} iou_sam={row.iou_sam:.2f} "
                                 f"hum={max(ious) if ious else 0:.2f}", fill="white")
        draw.text((4, tile + 15), "yellow=customer cyan=SAM green=human", fill="gray")
        tiles.append(canvas)
    sheet = Image.new("RGB", (4 * tile, 4 * (tile + 30)), "black")
    for k, canvas in enumerate(tiles):
        sheet.paste(canvas, ((k % 4) * tile, (k // 4) * (tile + 30)))
    path = OUT / "review" / "sheet_train_seed0.jpg"
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, quality=90)
    print(picks[["id", "class_name", "error", "iou_sam", "bottom_ext"]].to_string(index=False))
    print("Лист:", path)


# ---------------------------------------------------------------- оценка
def train_rows(members):
    """Оцениваемые боксы train с днём и таблицы согласия участников (каждый не видел день бокса)."""
    train = load_split("train")
    train["day"] = train.frame_id.map(R.frame_day("train"))
    return train, [R.oof_features(train, member) for member in members]


def detector_votes(df, frames):
    """votes, n_members, D — как в person_vote.add_signals, но без вето и позы."""
    votes = np.stack([(~f.set_index("id").matched.reindex(df.id)).to_numpy(dtype=float) for f in frames])
    return df.assign(votes=votes.sum(axis=0), n_members=len(frames), D=votes.mean(axis=0))


def variant_d(df, name, base_num, base_den):
    """D с голосом SAM для классов варианта; у остальных классов D не меняется."""
    classes, kind = VARIANTS[name]
    use = df.class_name.isin(classes).to_numpy()
    g = soft_vote(df.iou_sam) if kind == "soft" else df.G.to_numpy()
    return np.where(use, (base_num + g) / (base_den + 1), base_num / base_den)


def tier(d):
    return np.where(d == 0, "0", np.where(d == 1, "1", "дробный"))


def ap_table(df, formulas):
    rows = []
    for name, value in formulas.items():
        by_day = [average_precision_score(df.error[df.day == d], value[df.day == d]) for d in R.DAYS]
        rows.append(dict(формула=name, **{d: round(v, 4) for d, v in zip(R.DAYS, by_day)},
                         среднее=round(float(np.mean(by_day)), 4)))
    return pd.DataFrame(rows)


def bootstrap_lines(df, reference, formulas):
    for name, value in formulas.items():
        for day in R.DAYS:
            mask = df.day == day
            diffs = PV.frame_bootstrap(df[mask], reference[mask], value[mask])
            low, high = np.quantile(diffs, [0.05, 0.95])
            print(f"{name} {day}: прирост {diffs.mean():+.4f}, 5–95% [{low:+.4f}, {high:+.4f}], "
                  f"доля повторов с приростом {(diffs > 0).mean():.2f}")


def class_ap(df, formulas):
    rows = []
    for cls in CLASSES:
        part = df[df.class_name == cls]
        row = dict(класс=cls, боксов=len(part), ошибок=int(part.error.sum()))
        for name, value in formulas.items():
            v = value[part.index]
            row[name] = round(average_precision_score(part.error, v), 4) if 0 < part.error.sum() < len(part) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def chain_ids(df):
    """Цепочки боксов одного объекта по соседним кадрам: тот же класс, IoU >= 0.5 с боксом предыдущего кадра.
    Нужны, чтобы проверить, что G на test не срабатывает на одном и том же человеке подряд."""
    frames = sorted(df.frame_id.unique())
    chain, next_id, previous = {}, 0, None
    for frame in frames:
        group = df[df.frame_id == frame]
        for row in group.itertuples():
            best, best_iou = None, 0.5
            if previous is not None:
                for prev in previous[previous.class_name == row.class_name].itertuples():
                    iou = box_iou([row.x1, row.y1, row.x2, row.y2], [prev.x1, prev.y1, prev.x2, prev.y2])
                    if iou >= best_iou:
                        best, best_iou = chain[prev.id], iou
            if best is None:
                best, next_id = next_id, next_id + 1
            chain[row.id] = best
        previous = group
    return df.id.map(chain)


def test_firing():
    test = with_sam(load_split("test"), "test")
    print("\nG на test (без меток): срабатываний / боксов")
    table = test.groupby(["class_name", np.where(test.window, "окно", "зал")]).G.agg(G="sum", боксов="size")
    print(table.astype(int).to_string())
    fired = test[test.G == 1].copy()
    fired["chain"] = chain_ids(test)[fired.index]
    for cls, part in fired.groupby("class_name"):
        top = part.chain.value_counts()
        print(f"{cls}: срабатываний {len(part)}, крупнейшая цепочка {top.iloc[0]} "
              f"({top.iloc[0] / len(part):.0%}), цепочек {len(top)}")
    top = fired.chain.value_counts()
    print(f"всего: срабатываний {len(fired)}, крупнейшая цепочка {top.iloc[0]} ({top.iloc[0] / len(fired):.0%})")
    return test


def diag(args):
    """Ранний сигнал без вето и позы: (1 − conf + D) / 2 против того же с голосом SAM (A, B, C)."""
    train, frames = train_rows(args.members)
    df = with_sam(detector_votes(train, frames), "train")
    print(f"Участники: {' '.join(args.members)}; вето и поза не учитываются (диагностика)")
    reference = (1 - df.confidence + df.D) / 2
    formulas = {"опора": reference}
    for name in VARIANTS:
        formulas[name] = (1 - df.confidence + variant_d(df, name, df.votes, df.n_members)) / 2
    print(ap_table(df, formulas).to_string(index=False))
    bootstrap_lines(df, reference, {k: v for k, v in formulas.items() if k != "опора"})
    print("\nКласс × день × ярус D × G: боксов / ошибок")
    df["ярус"] = tier(df.D)
    print(df.groupby(["class_name", "day", "ярус", "G"]).error.agg(боксов="size", ошибок="sum").to_string())
    print("\nAP внутри класса (оба дня)")
    print(class_ap(df, formulas).to_string(index=False))
    print("\niou_sam: медиана по классу и метке")
    print(df.groupby(["class_name", "error"]).iou_sam.median().round(3).unstack().to_string())
    if (OUT / "test_sam.csv").exists():
        test_firing()


def leader(df):
    """Числитель и знаменатель D' лидера и его вето: P только у людей вне окна, вето = max(V, L)."""
    pose_on = ((df.class_name == "person") & ~df.window).to_numpy()
    num = df.votes.to_numpy() + df.P.to_numpy() * pose_on
    den = df.n_members.to_numpy() + pose_on
    veto = np.maximum(df.veto, df.L).to_numpy()
    return num, den, veto


def scored(df):
    """Опора и варианты A, B, C на одной таблице с сигналами лидера."""
    num, den, veto = leader(df)
    reference = pd.Series((1 - df.confidence.to_numpy() + num / den + 2 * veto) / 4, index=df.index)
    current, check = PV.scores(df, legs=True, skip_window=True)
    assert np.allclose(check.to_numpy(), reference.to_numpy()), "опора разошлась с person_vote.scores"
    formulas = {"опора": reference}
    for name in VARIANTS:
        formulas[name] = pd.Series((1 - df.confidence.to_numpy() + variant_d(df, name, num, den) + 2 * veto) / 4,
                                   index=df.index)
    return formulas, num / den, veto


def name_for(members, variant):
    return f"{R.WORK.name}{R.members_suffix(members)}_pvote_nowin_legs_sam{variant}_veto.csv"


def evaluate(args):
    train, frames = train_rows(args.members)
    df = with_sam(PV.add_signals(train, "train", frames), "train")
    formulas, d_leader, veto = scored(df)
    print(f"Участники: {' '.join(args.members)}; опора — person_vote --skip-window --legs")
    print(ap_table(df, formulas).to_string(index=False))
    bootstrap_lines(df, formulas["опора"], {k: v for k, v in formulas.items() if k != "опора"})
    df["ярус"], df["V"] = tier(d_leader), veto
    print("\nКласс × день × ярус D' лидера × вето × G: боксов / ошибок")
    print(df.groupby(["class_name", "day", "ярус", "V", "G"]).error.agg(боксов="size", ошибок="sum").to_string())
    print("\nAP внутри класса (оба дня)")
    print(class_ap(df, formulas).to_string(index=False))
    test_firing()
    for name in VARIANTS:
        path = ROOT / "oof" / name_for(args.members, name)
        if path.exists():
            print(f"{path} уже есть, не перезаписываю")
        else:
            df[["id"]].assign(score=formulas[name].to_numpy()).to_csv(path, index=False)
            print("OOF:", path)


def submit(args):
    test = R.load("test.csv", dtype={"id": str})  # как в person_vote.submit; столов в test нет
    frames = [test.merge(R.agreement_features(test, pd.read_csv(R.pred_path("all", m))), on="id")
              for m in args.members]
    df = with_sam(PV.add_signals(test, "test", frames), "test")
    formulas, _, _ = scored(df)
    new, reference = formulas[args.variant], formulas["опора"]
    template = pd.read_csv(R.DATA / "sample_submission.csv", dtype={"id": str})
    leader_path = ROOT / "submissions" / f"{R.WORK.name}{R.members_suffix(args.members)}_pvote_nowin_legs_veto.csv"
    lead = pd.read_csv(leader_path, dtype={"id": str}) if leader_path.exists() else None
    out = template[["id"]].merge(df[["id"]].assign(score=new.to_numpy()), on="id", validate="one_to_one")
    assert len(out) == 362 and out.id.is_unique and set(out.id) == set(template.id)
    assert np.isfinite(out.score).all() and out.score.between(0, 1).all()
    changed = df[np.abs(new - reference) > 1e-12]
    print(f"Изменилось относительно опоры: {len(changed)} строк; по классам {changed.class_name.value_counts().to_dict()}")
    if lead is not None:
        diff = out.merge(lead, on="id", suffixes=("", "_lead"))
        print(f"Относительно файла лидера {leader_path.name}: {int((np.abs(diff.score - diff.score_lead) > 1e-9).sum())} строк")
    path = ROOT / "submissions" / name_for(args.members, args.variant)
    if path.exists():
        raise FileExistsError(f"{path} уже есть: удалите его вручную, если нужна пересборка")
    out.to_csv(path, index=False)
    print("Файл:", path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cmd", choices=["run", "review", "diag", "eval", "submit"])
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--device", default="cpu", help="cpu или номер GPU, например 0")
    parser.add_argument("--model", default="weights/sam2_t.pt", help="путь к весам SAM от корня репозитория")
    parser.add_argument("--rerun", action="store_true", help="пересчитать кэш SAM")
    parser.add_argument("--tag", default="strong", help="рабочая папка outputs/redetect_<tag>")
    parser.add_argument("--members", nargs="+", default=["base"], help="участники ансамбля, как в redetect.py")
    parser.add_argument("--variant", choices=list(VARIANTS), default="A")
    a = parser.parse_args()
    R.WORK = ROOT / "outputs" / (f"redetect_{a.tag}" if a.tag else "redetect")
    {"run": run, "review": review, "diag": diag, "eval": evaluate, "submit": submit}[a.cmd](a)
