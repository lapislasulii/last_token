"""Independent checks for IoU, unavailable/cross-day context and label isolation."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from temporal_features import CORE, DETAIL, add_temporal, ious


class TemporalFeaturesTest(unittest.TestCase):
    def test_iou(self):
        actual = ious(np.array([0., 0., 10., 10.]), np.array([[0., 0., 10., 10.], [5., 0., 15., 10.], [20., 20., 30., 30.]]))
        np.testing.assert_allclose(actual, [1., 1/3, 0.])
        self.assertEqual(ious(np.zeros(4), np.empty((0, 4))).size, 0)

    def test_context_and_labels(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            pd.DataFrame([
                ['a', '2026-10-25T23:59:00Z'],
                ['b', '2026-10-25T23:59:30Z'],
                ['c', '2026-10-26T00:00:00Z'],
            ], columns=['frame_id', 'timestamp']).to_csv(folder/'train_frames.csv', index=False)
            pd.DataFrame([['a', '', 'b'], ['b', 'a', 'c']], columns=['frame_id', 'previous', 'next']).to_csv(folder/'train_context.csv', index=False)
            pd.DataFrame([
                ['one', 'a', 'cup', .8, 0, 0, 10, 10],
                ['two', 'b', 'cup', .6, 0, 0, 10, 10],
                ['three', 'c', 'cup', .9, 0, 0, 10, 10],
            ], columns=['detection_id', 'frame_id', 'class_name', 'confidence', 'x1', 'y1', 'x2', 'y2']).to_csv(folder/'train_stream_predictions.csv', index=False)
            rows = pd.DataFrame([
                ['one', 'a', 'cup', .8, 0, 0, 10, 10, 100, 100, 0],
                ['two', 'b', 'cup', .6, 0, 0, 10, 10, 100, 100, 1],
            ], columns=['id', 'frame_id', 'class_name', 'confidence', 'x1', 'y1', 'x2', 'y2', 'image_width', 'image_height', 'error'])
            result = add_temporal(rows, folder, 'train', {'aliases': {}})
            self.assertEqual(result.loc[0, 'previous_missing'], 1)
            self.assertEqual(result.loc[0, 'next_seconds'], 30.)
            self.assertEqual(result.loc[0, 'neighbor_iou_max'], 1.)
            self.assertEqual(result.loc[0, 'same_frame_iou'], 0.)  # self excluded
            self.assertAlmostEqual(result.loc[0, 'neighbor_conf_delta'], -.2)
            self.assertEqual(result.loc[1, 'next_missing'], 1)  # next UTC day excluded
            self.assertEqual(result.loc[1, 'next_seconds'], 0.)
            changed = add_temporal(rows.assign(error=1-rows.error), folder, 'train', {'aliases': {}})
            pd.testing.assert_frame_equal(result[CORE+DETAIL], changed[CORE+DETAIL])
            shuffled = add_temporal(rows.iloc[::-1], folder, 'train', {'aliases': {}})
            pd.testing.assert_frame_equal(result.set_index('id')[CORE+DETAIL].sort_index(), shuffled.set_index('id')[CORE+DETAIL].sort_index())


if __name__ == '__main__':
    unittest.main()
