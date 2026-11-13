import sys,unittest
from pathlib import Path
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from prepare_pseudo_labels import filter_fold


class PseudoLabelsTest(unittest.TestCase):
    def test_excludes_validation_frames_and_teacher_leakage(self):
        r=pd.DataFrame({'id':['same_day','teacher_leak','safe','uncertain','missing'],
            'day':['2026-10-25','2026-10-24','2026-10-24','2026-10-24','2026-10-24'],
            'annotation_train_days':[[],['2026-10-25'],['2026-10-26'],[],[]],
            'review_status':['accepted','accepted','accepted','uncertain','accepted'],
            'pseudo_error':[1,1,0,1,float('nan')]})
        self.assertEqual(filter_fold(r,'2026-10-25').id.tolist(),['safe'])


if __name__=='__main__':unittest.main()
