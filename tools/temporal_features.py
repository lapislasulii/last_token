"""Label-free features from supplied prediction streams and neighboring frames."""
import numpy as np
import pandas as pd

CORE = ['neighbor_iou_max', 'neighbor_iou_min', 'neighbor_conf_delta', 'same_frame_iou']
DETAIL = ['previous_missing', 'next_missing', 'previous_seconds', 'next_seconds',
          'neighbor_area_change', 'neighbor_center_shift', 'same_class_count']


def ious(box, candidates):
    if len(candidates) == 0:
        return np.array([])
    intersection = np.maximum(0, np.minimum(box[2:], candidates[:, 2:]) - np.maximum(box[:2], candidates[:, :2])).prod(axis=1)
    area = np.maximum(0, box[2:] - box[:2]).prod()
    areas = np.maximum(0, candidates[:, 2:] - candidates[:, :2]).prod(axis=1)
    return intersection / np.maximum(area + areas - intersection, 1e-12)


def add_temporal(rows, data, split, policy):
    stream = pd.read_csv(data / f'{split}_stream_predictions.csv', dtype={'detection_id': str})
    stream['class_name'] = stream.class_name.replace(policy['aliases'])
    groups = {key: group for key, group in stream.groupby(['frame_id', 'class_name'])}
    context = pd.read_csv(data / f'{split}_context.csv').fillna('').set_index('frame_id')
    frames = pd.read_csv(data / f'{split}_frames.csv')
    timestamps = dict(zip(frames.frame_id, pd.to_datetime(frames.timestamp, utc=True, format='mixed')))
    coords = ['x1', 'y1', 'x2', 'y2']
    output = []
    for row in rows.itertuples():
        box = np.array([row.x1, row.y1, row.x2, row.y2])
        area = np.prod(box[2:] - box[:2])
        same = groups.get((row.frame_id, row.class_name))
        other = same[same.detection_id != row.id] if same is not None else pd.DataFrame(columns=coords)
        overlap = ious(box, other[coords].to_numpy(dtype=float))
        record = {'same_frame_iou': float(overlap.max()) if overlap.size else 0., 'same_class_count': len(other)}
        matches, confidences, area_changes, shifts = [], [], [], []
        for side in ['previous', 'next']:
            neighbor = context.at[row.frame_id, side] if row.frame_id in context.index else ''
            # Do not use frames across the day boundary in either validation direction.
            available = bool(neighbor) and neighbor in timestamps and timestamps[neighbor].date() == timestamps[row.frame_id].date()
            record[f'{side}_missing'] = int(not available)
            record[f'{side}_seconds'] = abs((timestamps[neighbor] - timestamps[row.frame_id]).total_seconds()) if available else 0.
            candidates = groups.get((neighbor, row.class_name)) if available else None
            if candidates is None or candidates.empty:
                matches.append(0.)
                continue
            candidate_boxes = candidates[coords].to_numpy(dtype=float)
            overlap = ious(box, candidate_boxes)
            best = int(overlap.argmax())
            matches.append(float(overlap[best]))
            if overlap[best] > 0:
                confidences.append(float(candidates.iloc[best].confidence))
                area_changes.append(abs(np.log(np.prod(candidate_boxes[best, 2:] - candidate_boxes[best, :2]) / area)))
                shift = ((candidate_boxes[best, :2] + candidate_boxes[best, 2:]) - (box[:2] + box[2:])) / 2
                shifts.append(float(np.linalg.norm(shift / [row.image_width, row.image_height])))
        record.update(neighbor_iou_max=max(matches), neighbor_iou_min=min(matches),
                      neighbor_conf_delta=float(np.mean(confidences) - row.confidence) if confidences else 0.,
                      neighbor_area_change=float(np.mean(area_changes)) if area_changes else 0.,
                      neighbor_center_shift=float(np.mean(shifts)) if shifts else 0.)
        output.append(record)
    result = pd.concat([rows.reset_index(drop=True), pd.DataFrame(output)], axis=1)
    if not np.isfinite(result[CORE + DETAIL]).all().all():
        raise ValueError('Nonfinite temporal feature')
    return result
