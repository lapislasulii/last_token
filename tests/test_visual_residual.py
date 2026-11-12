"""Check direction and baseline-offset behavior on independent synthetic data."""
import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from train_visual_residual import fit,predict


class VisualResidualTest(unittest.TestCase):
    def test_learns_error_direction_on_new_rows(self):
        rng=np.random.default_rng(42)
        x=rng.normal(size=(400,16))
        y=(x[:,0]>0).astype(int)
        model=fit(x,np.full(400,.5),y,.1)
        heldout=rng.normal(size=(200,16))
        scores=predict(model,heldout,np.full(200,.5))
        self.assertGreater(np.mean((scores>.5)==(heldout[:,0]>0)),.9)

    def test_zero_correction_preserves_baseline(self):
        rng=np.random.default_rng(7)
        x=rng.normal(size=(80,16))
        model=fit(x,np.full(80,.5),np.arange(80)%2,1.)
        model['coef'][:]=0
        confidence=np.linspace(.01,.99,80)
        np.testing.assert_allclose(predict(model,x,confidence),1-confidence,atol=1e-12)


if __name__=='__main__':
    unittest.main()
