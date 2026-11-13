"""Browse local detector predictions and human annotations.

python tools/view_frame.py                    # local browser viewer
python tools/view_frame.py --split train --frame FRAME_ID --output reports/frame.png
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
COLORS = {"correct": "#22c55e", "error": "#ef4444", "unknown": "#eab308", "human": "#3b82f6", "table": "#ffffff"}


class Dataset:
    def __init__(self, folder: Path, scores: dict | None = None, test_detections: Path | None = None):
        self.folder = folder.resolve()
        self.test_detections = {}
        if test_detections:
            detections = pd.read_csv(test_detections)
            self.test_detections = dict(tuple(detections.groupby('frame_id')))
        self.scores = {}       # split -> {id: (score, место в очереди, размер очереди)}
        self.worst = {}        # split -> кадры, отсортированные по числу промахов модели
        self.score_files = {}
        policy = json.loads((folder / "scoring_policy.json").read_text(encoding="utf-8"))
        aliases = policy.get("aliases", {})
        self.excluded = set(policy.get("test_excluded_classes", []))
        self.frames = {}
        self.predictions = {}
        self.context = {}
        self.frame_lists = {}
        self.preferred_frames = {}
        for split in ("train", "test", "machine"):
            frames = pd.read_csv(folder / f"{split}_frames.csv", dtype={"frame_id": str})
            frames = frames.sort_values(["timestamp", "frame_id"])
            self.frame_lists[split] = frames.frame_id.tolist()
            self.frames[split] = frames.set_index("frame_id").to_dict("index")
            name = "machine_predictions.csv" if split == "machine" else f"{split}_stream_predictions.csv"
            predictions = pd.read_csv(folder / name, dtype={"detection_id": str, "frame_id": str})
            predictions["class_name"] = predictions.class_name.replace(aliases)
            if split == "train":
                labels = pd.read_csv(folder / "train.csv", dtype={"id": str, "frame_id": str})
                if not labels.id.is_unique:
                    raise ValueError("Duplicate IDs in train.csv")
                # Only existing official labels, never inferred from confidence or IoU.
                predictions = predictions.merge(labels[["id", "error"]], left_on="detection_id", right_on="id", how="left", validate="one_to_one")
            self.predictions[split] = {key: group for key, group in predictions.groupby("frame_id")}
            candidates = predictions[~predictions.class_name.isin(self.excluded)]
            if split == "train":
                candidates = candidates[candidates.error.notna()]
                counts = candidates.groupby("frame_id").error.sum()
            else:
                counts = candidates.groupby("frame_id").size()
            self.preferred_frames[split] = counts.idxmax() if not counts.empty else self.frame_lists[split][0]
            if split != "machine":
                ctx = pd.read_csv(folder / f"{split}_context.csv").fillna("")
                self.context[split] = ctx.set_index("frame_id").to_dict("index")
        human = pd.read_csv(folder / "human_boxes_train.csv")
        human["class_name"] = human.class_name.replace(aliases)
        self.human = {key: group for key, group in human.groupby("frame_id")}
        for split, path in (scores or {}).items():
            self.load_scores(split, path)

    def load_scores(self, split, path: Path):
        """Score модели (id,score). Очередь — как в метрике: train по UTC-дням, test целиком."""
        s = pd.read_csv(path, dtype={"id": str})[["id", "score"]]
        rows = pd.read_csv(self.folder / f"{split}.csv", dtype={"id": str, "frame_id": str})
        rows = rows[~rows.class_name.replace({}).isin(self.excluded)].merge(s, on="id", how="inner")
        rows["day"] = rows.frame_id.str.extract(r"_(2026\d{4})_")[0] if split == "train" else "all"
        rows["rank"] = rows.groupby("day").score.rank(ascending=False, method="min").astype(int)
        rows["n"] = rows.groupby("day").score.transform("size")
        self.scores[split] = {r.id: (float(r.score), int(r.rank), int(r.n)) for r in rows.itertuples()}
        self.score_files[split] = path.name
        if "error" in rows:
            # промах: ошибка в нижней половине очереди или верный бокс в верхней четверти
            pct = rows["rank"] / rows.n
            rows["miss"] = ((rows.error == 1) & (pct > 0.5)) | ((rows.error == 0) & (pct <= 0.25))
            counts = rows.groupby("frame_id").miss.sum()
            self.worst[split] = [f for f, c in counts.sort_values(ascending=False).items() if c > 0]

    def image_path(self, split, frame_id):
        row = self.frames[split][frame_id]
        path = (self.folder / row["image_path"]).resolve()
        if not path.is_relative_to(self.folder):
            raise ValueError("Image path outside dataset")
        return path

    def payload(self, split, frame_id):
        row = self.frames[split][frame_id]
        with Image.open(self.image_path(split, frame_id)) as image:
            width, height = image.size
        # Machine-pool coordinates refer to the original 1280 x 720 canvas.
        coordinate_width = 1280 if split == "machine" else float(row["image_width"])
        coordinate_height = 720 if split == "machine" else float(row["image_height"])
        sx, sy = width / coordinate_width, height / coordinate_height
        boxes = []
        predictions = self.predictions[split].get(frame_id)
        if predictions is not None:
            for r in predictions.to_dict("records"):
                error = r.get("error")
                status = "unknown" if error is None or pd.isna(error) else ("error" if int(error) else "correct")
                box = {"kind": "machine", "class_name": r["class_name"], "id": str(r["detection_id"]), "confidence": float(r["confidence"]), "status": status, "excluded": r["class_name"] in self.excluded,
                       "xyxy": [float(r["x1"]) * sx, float(r["y1"]) * sy, float(r["x2"]) * sx, float(r["y2"]) * sy]}
                scored = self.scores.get(split, {}).get(str(r["detection_id"]))
                if scored:
                    box["score"], box["rank"], box["queue"] = scored
                boxes.append(box)
        if split == 'test' and frame_id in self.test_detections:
            for r in self.test_detections[frame_id].itertuples():
                boxes.append(dict(kind='redetect', class_name=r.class_name,
                                  confidence=float(r.confidence), status='unknown',
                                  excluded=r.class_name in self.excluded, source='YOLO11l base',
                                  xyxy=[float(r.x1)*sx, float(r.y1)*sy, float(r.x2)*sx, float(r.y2)*sy]))
        humans = self.human.get(frame_id) if split == "train" else None
        if humans is not None:
            for r in humans.to_dict("records"):
                scored = bool(r.get("evaluated_class", True)) and r["class_name"] not in self.excluded
                boxes.append({"kind": "human", "class_name": r["class_name"], "status": "human", "source": r.get("source", ""), "excluded": r["class_name"] in self.excluded, "scored": scored,
                              "xyxy": [(float(r["cx"]) - float(r["w"])/2)*width, (float(r["cy"]) - float(r["h"])/2)*height, (float(r["cx"]) + float(r["w"])/2)*width, (float(r["cy"]) + float(r["h"])/2)*height]})
        ctx = self.context.get(split, {}).get(frame_id)
        if ctx is None and split == "machine":
            # Nearest available pool frames; actual timestamps are shown, no 30s assumption.
            keys = self.frame_lists[split]
            i = keys.index(frame_id)
            ctx = {"previous": keys[i-1] if i else "", "next": keys[i+1] if i+1 < len(keys) else ""}
        return {"frame_id": frame_id, "split": split, "timestamp": str(row["timestamp"]), "width": width, "height": height, "boxes": boxes, "context": ctx or {}, "has_human": humans is not None, "is_target": frame_id in self.context.get(split, {})}


def box_label(box):
    if box["kind"] == "human":
        return box["class_name"]
    return f"{box['class_name']} {box['confidence']:.2f}"


def export_image(dataset, split, frame_id, output, humans=True, machines=True, tables=False, labels=False, other=False):
    """Те же цвета, что в браузере: синий — ручной, зелёный/красный — детектор верно/ошибка, белый — стол."""
    data = dataset.payload(split, frame_id)
    with Image.open(dataset.image_path(split, frame_id)) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=14)
    for box in sorted(data["boxes"], key=lambda b: (b["kind"] != "human", b["excluded"])):
        is_table = box["excluded"]
        if box["kind"] == "human" and (not humans or is_table or (not box["scored"] and not other)):
            continue
        if box["kind"] == "machine" and (not machines or (is_table and not tables)):
            continue
        color = COLORS["table"] if is_table else COLORS[box["status"]]
        coords = box["xyxy"]
        draw.rectangle(coords, outline=color, width=2 if is_table else (4 if box["kind"] == "human" else 3))
        if labels and not is_table:
            label = box_label(box)
            bounds = draw.textbbox((0, 0), label, font=font)
            x = max(0, min(coords[0], image.width - bounds[2]))
            y = max(0, coords[1] - 18)
            draw.rectangle((x, y, x + bounds[2], y + 18), fill="#111827")
            draw.text((x, y), label, font=font, fill=color)
    banner = Image.new("RGB", (image.width, image.height + 65), "#111827")
    banner.paste(image, (0, 65))
    d = ImageDraw.Draw(banner)
    d.text((8, 4), f"{split}: {frame_id}", font=font, fill="white")
    d.text((8, 25), "blue = human box | green = detector error=0 | red = error=1 | yellow = unknown | white = table", font=font, fill="white")
    d.text((8, 45), data["timestamp"], font=font, fill="white")
    output.parent.mkdir(parents=True, exist_ok=True)
    banner.save(output)
    print(f"Saved: {output.resolve()}")


def serve(dataset, split, frame_id, port, open_browser):
    page = Path(__file__).with_name("frame_viewer.html").read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            url = urlsplit(self.path)
            params = parse_qs(url.query)
            try:
                if url.path == "/":
                    body, content_type = page, "text/html; charset=utf-8"
                elif url.path == "/api/frames":
                    body = json.dumps({"frames": dataset.frame_lists, "preferred_frames": dataset.preferred_frames, "initial_split": split, "initial_frame": frame_id,
                                       "worst": dataset.worst, "score_files": dataset.score_files}).encode()
                    content_type = "application/json"
                elif url.path in ("/api/frame", "/image"):
                    s, f = params["split"][0], params["frame"][0]
                    if url.path == "/api/frame":
                        body = json.dumps(dataset.payload(s, f), allow_nan=False).encode()
                        content_type = "application/json"
                    else:
                        path = dataset.image_path(s, f)
                        body, content_type = path.read_bytes(), mimetypes.guess_type(path)[0] or "image/jpeg"
                else:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.end_headers()
                self.wfile.write(body)
            except (KeyError, ValueError, FileNotFoundError) as exc:
                self.send_error(404, str(exc))

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{server.server_address[1]}"
    print(f"Viewer: {url} (Ctrl+C to stop)", flush=True)
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/rascar-hack-case-1")
    parser.add_argument("--split", choices=["train", "test", "machine"], default="train")
    parser.add_argument("--frame", help="Frame ID, or image filename with .jpg")
    parser.add_argument("--output", type=Path, help="Save annotated PNG/JPEG instead of starting viewer")
    parser.add_argument("--no-human", action="store_true")
    parser.add_argument("--no-machine", action="store_true")
    parser.add_argument("--tables", action="store_true", help="Include table boxes in exported image")
    parser.add_argument("--labels", action="store_true", help="Short labels in exported image")
    parser.add_argument("--other", action="store_true", help="Also show unscored human classes (hair, uniform, phone…)")
    parser.add_argument("--scores", type=Path, help="OOF score модели для train (id,score), например oof/redetect_strong_veto_veto.csv")
    parser.add_argument("--test-scores", type=Path, help="Сабмит для test (id,score)")
    parser.add_argument("--test-detections", type=Path, help="Боксы нашего детектора на test: pred_all.csv")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    try:
        scores = {k: v for k, v in (("train", args.scores), ("test", args.test_scores)) if v}
        dataset = Dataset(args.data, scores, args.test_detections)
        frame_id = Path(args.frame).stem if args.frame else None
        if frame_id is None:
            # Start on an informative labeled frame rather than an empty night scene.
            frame_id = dataset.preferred_frames[args.split]
        if frame_id not in dataset.frames[args.split]:
            parser.error(f"Unknown frame in {args.split}: {frame_id}")
        if args.output:
            export_image(dataset, args.split, frame_id, args.output, not args.no_human, not args.no_machine, args.tables,
                         args.labels, args.other)
        else:
            serve(dataset, args.split, frame_id, args.port, not args.no_browser)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
