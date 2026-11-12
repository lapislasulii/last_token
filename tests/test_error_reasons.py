import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from label_error_reasons import XY, annotate_frame, match_boxes


def humans(boxes, classes=None):
    frame = pd.DataFrame(boxes, columns=XY)
    frame['class_name'] = classes if classes is not None else ['cup']*len(frame)
    frame['match_eligible'] = True
    frame['human_row'] = np.arange(len(frame))
    return frame


def predictions(boxes, errors, classes=None):
    frame = pd.DataFrame(boxes, columns=XY)
    frame['class_name'] = classes if classes is not None else ['cup']*len(frame)
    frame['id'] = [f'p{i}' for i in range(len(frame))]
    frame['error'] = errors
    return frame


class ErrorReasonsTest(unittest.TestCase):
    def test_cardinality_before_iou(self):
        self.assertEqual(match_boxes(np.array([[.9,.6],[.6,0]]),.6), {0:1,1:0})
        self.assertEqual(match_boxes(np.array([[.9,.7],[.7,.9]]),.6), {0:0,1:1})
        self.assertEqual(match_boxes(np.zeros((2,0)),.6), {})
        self.assertEqual(match_boxes(np.array([[.59],[.6]]),.6), {1:0})

    def test_duplicate(self):
        pred=predictions([[0,0,10,10],[0,0,9,10]],[0,1])
        result=annotate_frame(pred,humans([[0,0,10,10]]))
        self.assertEqual(result.reason.tolist(),['correct','duplicate_candidate'])
        self.assertEqual(result.iloc[1].competing_detection_id,'p0')
        self.assertTrue(result.matching_agrees.all())

    def test_geometry_subtypes(self):
        for box, expected in [([0,0,5,10],'too_small_or_partial'),
                              ([0,0,20,10],'too_large'),([3,0,13,10],'shift_or_shape')]:
            result=annotate_frame(predictions([box],[1]),humans([[0,0,10,10]])).iloc[0]
            self.assertEqual(result.reason,'geometry_candidate')
            self.assertEqual(result.geometry_subtype,expected)

    def test_class_mismatch_and_unresolved(self):
        pred=predictions([[0,0,10,10]],[1])
        result=annotate_frame(pred,humans([[0,0,10,10]],['bottle'])).iloc[0]
        self.assertEqual(result.reason,'class_mismatch_candidate')
        self.assertEqual(result.best_other_class,'bottle')
        self.assertTrue(result.needs_visual_review)
        for human in [humans([]),humans([[100,100,110,110]])]:
            self.assertEqual(annotate_frame(pred,human).iloc[0].reason,'unresolved')

    def test_official_label_not_overwritten(self):
        pred=predictions([[0,0,10,10]],[1])
        result=annotate_frame(pred,humans([[0,0,10,10]])).iloc[0]
        self.assertEqual(result.reason,'matching_disagreement')
        self.assertFalse(result.matching_agrees)
        self.assertEqual(pred.iloc[0].error,1)


if __name__=='__main__':
    unittest.main()
