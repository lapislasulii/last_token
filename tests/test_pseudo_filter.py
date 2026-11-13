import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from pseudo_filter import rejection_reasons


class PseudoFilterTests(unittest.TestCase):
    def test_missing_person(self):
        self.assertIn('unconfirmed_original_person', rejection_reasons([], [], [], [[0,0,10,10]]))

    def test_wrong_person_location(self):
        self.assertIn('unconfirmed_original_person', rejection_reasons(['person'], [.9], [[20,20,30,30]], [[0,0,10,10]]))

    def test_ambiguous_hand(self):
        self.assertEqual(rejection_reasons(['nude_hands'], [.4], [[0,0,5,5]], []), ['ambiguous_nude_hands'])

    def test_confident_person(self):
        self.assertEqual(rejection_reasons(['person'], [.5], [[0,0,10,10]], [[0,0,10,10]]), [])

    def test_empty_scene(self):
        self.assertEqual(rejection_reasons([], [], [], []), [])
