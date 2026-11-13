"""Expert veto: public COCO detector and pose model as zero-shot second opinions.

No training on our labels. Score = (1 - confidence + veto) / 2, where veto = 1 if
  cup        : no COCO cup-like box (cup/bowl/bottle/wine glass/vase) with IoU >= CUP_IOU,
  nude_hands : no pose wrist within HAND_DIST box sizes of the box centre,
  person     : no pose skeleton whose box has IoU > PERSON_IOU with the detection.

Weights (public): https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x.pt
                  https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x-pose.pt
Put them into weights/. Each frame is run full-size and as 3x3 overlapping tiles upscaled 2x.

Usage:
  python tools/expert_veto.py train   # OOF-style check by day -> oof/expert_veto.csv
  python tools/expert_veto.py test    # -> submissions/expert_veto.csv (not uploaded)
Expert outputs are cached in outputs/expert_veto/. Ultralytics warns about a pose bug on
Apple MPS; with torch 2.15 MPS and CPU outputs were identical on 18 train frames.
On an NVIDIA card pass --det-device 0 --pose-device 0.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/rascar-hack-case-1'
OUT = ROOT / 'outputs/expert_veto'
CUP_LIKE = ['cup', 'bowl', 'bottle', 'wine glass', 'vase']
CUP_IOU, HAND_DIST, PERSON_IOU = 0.3, 0.6, 0.2   # fixed, not tuned per day
POSE_MIN_CONF, WRIST_MIN_CONF = 0.15, 0.1
KEYPOINTS = ['nose', 'leye', 'reye', 'lear', 'rear', 'lsho', 'rsho', 'lelb', 'relb', 'lwri', 'rwri',
             'lhip', 'rhip', 'lkne', 'rkne', 'lank', 'rank']
VIEWS = [('full', 0, 0, 1280, 720, 1.0)] + [('tile', x0, y0, 640, 360, 2.0)
                                              for x0 in (0, 320, 640) for y0 in (0, 180, 360)]
BOX = ['x1', 'y1', 'x2', 'y2']


def run_experts(split, det_device, pose_device):
    """Run both experts on every frame of the split that has a scored detection."""
    from ultralytics import YOLO
    rows = pd.read_csv(DATA / f'{split}.csv')
    frames = rows[rows.class_name != 'table'].frame_id.unique()
    det, pose = YOLO(ROOT / 'weights/yolo11x.pt'), YOLO(ROOT / 'weights/yolo11x-pose.pt')
    det_rows, pose_rows = [], []
    for number, frame in enumerate(frames):
        with Image.open(DATA / f'{frame}.jpg') as source:
            image = source.convert('RGB')
        for src, x0, y0, w, h, scale in VIEWS:
            view = image if src == 'full' else image.crop((x0, y0, x0 + w, y0 + h)).resize((1280, 720), Image.BICUBIC)
            result = det.predict(view, imgsz=1280, conf=0.03, iou=0.6, verbose=False, device=det_device)[0]
            for box, conf, cls in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy(),
                                      result.boxes.cls.cpu().numpy()):
                det_rows.append((frame, src, x0, y0, det.names[int(cls)], conf,
                                 box[0] / scale + x0, box[1] / scale + y0, box[2] / scale + x0, box[3] / scale + y0))
            result = pose.predict(view, imgsz=1280, conf=0.05, iou=0.6, verbose=False, device=pose_device)[0]
            if result.keypoints is None or not len(result.boxes):
                continue
            for box, conf, points in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy(),
                                         result.keypoints.data.cpu().numpy()):
                points = points.copy()
                points[:, 2] = np.where((points[:, 0] == 0) & (points[:, 1] == 0), 0, points[:, 2])  # missing point
                points[:, 0] = points[:, 0] / scale + x0
                points[:, 1] = points[:, 1] / scale + y0
                pose_rows.append((frame, src, x0, y0, conf, box[0] / scale + x0, box[1] / scale + y0,
                                  box[2] / scale + x0, box[3] / scale + y0, *points.reshape(-1)))
        print(f'{split}: {number + 1}/{len(frames)} frames', flush=True)
    head = ['frame_id', 'src', 'tx', 'ty']
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(det_rows, columns=head + ['cls', 'conf'] + BOX).to_csv(OUT / f'{split}_det.csv', index=False)
    pd.DataFrame(pose_rows, columns=head + ['conf'] + BOX + [f'{k}_{a}' for k in KEYPOINTS for a in 'xyc']
                 ).to_csv(OUT / f'{split}_pose.csv', index=False)


def iou_many(box, boxes):
    if not len(boxes):
        return np.zeros(0)
    ix = np.clip(np.minimum(box[2], boxes[:, 2]) - np.maximum(box[0], boxes[:, 0]), 0, None)
    iy = np.clip(np.minimum(box[3], boxes[:, 3]) - np.maximum(box[1], boxes[:, 1]), 0, None)
    inter = ix * iy
    return inter / ((box[2] - box[0]) * (box[3] - box[1]) + (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) - inter)


def features(split):
    rows = pd.read_csv(DATA / f'{split}.csv', dtype={'id': str})
    rows = rows[rows.class_name != 'table'].reset_index(drop=True)
    det = pd.read_csv(OUT / f'{split}_det.csv')
    pose = pd.read_csv(OUT / f'{split}_pose.csv')
    # A tile box touching an inner tile border is a truncated object: drop it.
    tile = det.src == 'tile'
    cut = tile & ((((det.x1 - det.tx) < 3) & (det.tx > 0)) | (((det.tx + 640 - det.x2) < 3) & (det.tx < 640))
                  | (((det.y1 - det.ty) < 3) & (det.ty > 0)) | (((det.ty + 360 - det.y2) < 3) & (det.ty < 360)))
    det = det[~cut & det.cls.isin(CUP_LIKE)]
    pose = pose[pose.conf > POSE_MIN_CONF]
    det_by_frame = {f: g[BOX].to_numpy() for f, g in det.groupby('frame_id')}
    pose_by_frame = dict(tuple(pose.groupby('frame_id')))
    cup_iou, wrist_dist, pose_iou = [], [], []
    for row in rows.itertuples():
        box = np.array([row.x1, row.y1, row.x2, row.y2], dtype=float)
        people = pose_by_frame.get(row.frame_id)
        ious = iou_many(box, det_by_frame.get(row.frame_id, np.zeros((0, 4))))
        cup_iou.append(ious.max() if len(ious) else 0.0)
        ious = iou_many(box, people[BOX].to_numpy()) if people is not None else np.zeros(0)
        pose_iou.append(ious.max() if len(ious) else 0.0)
        best = 9.0
        if people is not None:
            cx, cy, size = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2, max(box[2] - box[0], box[3] - box[1])
            for wrist in ('lwri', 'rwri'):
                seen = people[people[f'{wrist}_c'] > WRIST_MIN_CONF]
                if len(seen):
                    best = min(best, (np.hypot(seen[f'{wrist}_x'] - cx, seen[f'{wrist}_y'] - cy) / size).min())
        wrist_dist.append(best)
    rows['cup_iou'], rows['wrist_dist'], rows['pose_iou'] = cup_iou, wrist_dist, pose_iou
    rows['veto'] = (((rows.class_name == 'cup') & (rows.cup_iou < CUP_IOU))
                    | ((rows.class_name == 'nude_hands') & (rows.wrist_dist > HAND_DIST))
                    | ((rows.class_name == 'person') & (rows.pose_iou <= PERSON_IOU))).astype(int)
    rows['score'] = (1 - rows.confidence + rows.veto) / 2
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('split', choices=['train', 'test'])
    parser.add_argument('--det-device', default='mps')
    parser.add_argument('--pose-device', default='mps')
    parser.add_argument('--rerun', action='store_true', help='ignore cached expert outputs')
    args = parser.parse_args()
    if args.rerun or not (OUT / f'{args.split}_pose.csv').exists():
        run_experts(args.split, args.det_device, args.pose_device)
    rows = features(args.split)
    rows.drop(columns=['score']).to_csv(OUT / f'{args.split}_features.csv', index=False)
    print(rows.groupby('class_name').veto.agg(['sum', 'size']).rename(columns={'sum': 'veto', 'size': 'boxes'}))
    if args.split == 'test':
        template = pd.read_csv(DATA / 'sample_submission.csv', dtype={'id': str})
        assert set(template.id) == set(rows.id) and rows.id.is_unique
        path = ROOT / 'submissions/expert_veto.csv'
        path.parent.mkdir(exist_ok=True)
        template[['id']].merge(rows[['id', 'score']], on='id', validate='one_to_one').to_csv(path, index=False)
        print(f'Wrote {path}')
        return
    day = rows.frame_id.str.extract(r'_(2026\d{4})_')[0]
    print(rows[rows.veto == 1].groupby('class_name').error.agg(['sum', 'size']).rename(
        columns={'sum': 'errors', 'size': 'veto'}))
    for name, score in (('baseline', 1 - rows.confidence), ('expert veto', rows.score)):
        by_day = [average_precision_score(rows.error[day == d], score[day == d]) for d in sorted(day.unique())]
        print(f'{name:12s} AP by day {np.round(by_day, 4)} mean {np.mean(by_day):.4f}')
    path = ROOT / 'oof/expert_veto.csv'
    path.parent.mkdir(exist_ok=True)
    rows[['id', 'score']].to_csv(path, index=False)
    print(f'Wrote {path}')


if __name__ == '__main__':
    main()
