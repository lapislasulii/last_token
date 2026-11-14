"""Checks for SAM geometry helpers: mask -> box, component choice, crop coordinates, prompts."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from sam_geometry import (box_iou, crop_scale, crop_window, from_crop, mask_to_box, person_prompt,
                          soft_vote, to_crop)


class MaskToBoxTest(unittest.TestCase):
    def test_tight_box_is_exclusive_on_the_right(self):
        mask = np.zeros((20, 30))
        mask[5:9, 10:16] = 1.0
        box, area = mask_to_box(mask, (12, 6))
        self.assertEqual(box, [10.0, 5.0, 16.0, 9.0])
        self.assertEqual(area, 24)

    def test_threshold_half(self):
        mask = np.full((10, 10), 0.4)
        mask[2:4, 2:4] = 0.9
        box, _ = mask_to_box(mask, (0, 0))
        self.assertEqual(box, [2.0, 2.0, 4.0, 4.0])

    def test_empty_mask(self):
        self.assertEqual(mask_to_box(np.zeros((5, 5)), (2, 2)), (None, 0))

    def test_component_with_centre_wins_over_larger(self):
        mask = np.zeros((40, 40), dtype=bool)
        mask[0:20, 0:20] = True        # большая чужая компонента (например, соседняя чашка)
        mask[30:34, 30:34] = True      # маленькая компонента под центром бокса
        box, area = mask_to_box(mask, (31.5, 31.5))
        self.assertEqual(box, [30.0, 30.0, 34.0, 34.0])
        self.assertEqual(area, 16)

    def test_largest_component_when_centre_is_empty(self):
        mask = np.zeros((40, 40), dtype=bool)
        mask[0:5, 0:5] = True
        mask[20:30, 20:35] = True
        box, _ = mask_to_box(mask, (12, 12))
        self.assertEqual(box, [20.0, 20.0, 35.0, 30.0])


class CropTest(unittest.TestCase):
    def test_window_is_square_and_centred(self):
        window = crop_window([100, 100, 130, 120], 4.0, 1280, 720)
        self.assertEqual(window, (55, 50, 175, 170))     # сторона 4 × 30 = 120, центр (115, 110)

    def test_window_is_clipped_by_frame(self):
        window = crop_window([0, 690, 30, 720], 4.0, 1280, 720)
        self.assertEqual(window, (0, 645, 75, 720))

    def test_round_trip(self):
        window = crop_window([400.5, 300.25, 431.0, 337.75], 4.0, 1280, 720)
        scale = crop_scale(window)
        self.assertAlmostEqual(scale * max(window[2] - window[0], window[3] - window[1]), 1024)
        box = [400.5, 300.25, 431.0, 337.75]
        inside = to_crop(box, window, scale)
        self.assertTrue(all(0 <= v <= 1024 for v in inside))
        np.testing.assert_allclose(from_crop(inside, window, scale), box)

    def test_crop_box_maps_to_frame(self):
        window, scale = (100, 200, 228, 328), 8.0      # вырезка 128 px, увеличена до 1024
        np.testing.assert_allclose(from_crop([0, 0, 1024, 512], window, scale), [100, 200, 228, 264])


class PromptAndVoteTest(unittest.TestCase):
    def test_person_prompt_extends_down_and_up(self):
        prompt, point = person_prompt([100, 100, 200, 300], 1280, 720)
        self.assertEqual(prompt, [100, 80.0, 200, 420.0])
        self.assertEqual(point, [150.0, 200.0])

    def test_person_prompt_clipped(self):
        prompt, _ = person_prompt([10, 5, 50, 700], 1280, 720)
        self.assertEqual(prompt[1], 0.0)
        self.assertEqual(prompt[3], 720.0)

    def test_iou_and_soft_vote(self):
        self.assertAlmostEqual(box_iou([0, 0, 10, 10], [5, 0, 15, 10]), 1 / 3)
        self.assertEqual(box_iou([0, 0, 1, 1], [2, 2, 3, 3]), 0.0)
        np.testing.assert_allclose(soft_vote([0.8, 0.75, 0.6, 0.45, 0.1]), [0, 0, 0.5, 1, 1])


if __name__ == '__main__':
    unittest.main()
