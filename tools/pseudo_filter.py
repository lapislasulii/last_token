"""Conservative pseudo-label screening, not ground-truth reconstruction."""
import numpy as np

TARGETS = {'person', 'nude_hands', 'cup', 'tray'}


def rejection_reasons(names, confidence, boxes, original_people, low=.25, high=.5, person_iou=.3):
    confidence = np.asarray(confidence)
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    original_people = np.asarray(original_people, dtype=float).reshape(-1, 4)
    reasons = ['ambiguous_' + name for name in sorted({
        n for n, c in zip(names, confidence) if n in TARGETS and low <= c < high})]
    teachers = boxes[np.array([n == 'person' and c >= high for n, c in zip(names, confidence)], dtype=bool)]
    # Both inputs must use the original 1280x720 coordinate system.
    if len(original_people):
        if not len(teachers):
            reasons.append('unconfirmed_original_person')
        else:
            lo = np.maximum(original_people[:, None, :2], teachers[None, :, :2])
            hi = np.minimum(original_people[:, None, 2:], teachers[None, :, 2:])
            inter = np.clip(hi-lo, 0, None).prod(axis=2)
            area = lambda b: np.clip(b[:, 2:]-b[:, :2], 0, None).prod(axis=1)
            iou = inter / np.maximum(area(original_people)[:, None]+area(teachers)[None, :]-inter, 1e-9)
            if (iou.max(axis=1) < person_iou).any():
                reasons.append('unconfirmed_original_person')
    return reasons
