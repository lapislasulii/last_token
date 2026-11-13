import sys,unittest
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from train_synthetic_boxes import max_overlap,crop_image


class SyntheticBoxesTest(unittest.TestCase):
    def test_shift_to_other_person_is_not_false_negative(self):
        # A box moved away from its source may match a different annotated person.
        self.assertEqual(max_overlap([20,0,30,10],[[0,0,10,10],[20,0,30,10]]),1.)

    def test_missing_class_and_partial_box(self):
        self.assertEqual(max_overlap([0,0,10,10],[]),0.)
        self.assertAlmostEqual(max_overlap([0,0,10,5],[[0,0,10,10]]),.5)

    def test_crop_clipped_at_frame_edge(self):
        im=Image.new('RGB',(128,72),'white')
        c=crop_image(im,[0,0,100,200],1280,720)
        self.assertEqual(c.size,(224,224));self.assertTrue(np.isfinite(np.asarray(c)).all())


if __name__=='__main__':unittest.main()
