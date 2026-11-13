"""Use a saved synthetic-box head with the existing marked CLIP test cache."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from train_logreg import ROOT,features
from train_synthetic_boxes import CLASSES,MODEL,REVISION
from train_temporal import write_submission


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('model',type=Path);p.add_argument('--output',type=Path);a=p.parse_args()
    b=joblib.load(a.model);assert b['encoder']==MODEL and b['revision']==REVISION
    directory=ROOT/'outputs/marked_context_clip'
    metadata=json.loads((directory/'encoder.json').read_text());assert metadata['model']==MODEL and metadata['revision']==REVISION
    data=ROOT/'data/rascar-hack-case-1';rows=features(data,'test',json.loads((data/'scoring_policy.json').read_text()))
    z=np.load(directory/'test_features.npz');assert np.array_equal(z['ids'],rows.id)
    x=np.column_stack([z['features'],rows.class_name.map(CLASSES.index)])
    visual=b['estimator'].predict_proba(x)[:,1];score=np.zeros(len(rows))
    for day,g in rows.groupby('day'):
        score[g.index]=.5*pd.Series(visual[g.index]).rank(pct=True).to_numpy()+.5*(1-g.confidence).rank(pct=True).to_numpy()
    out=a.output or ROOT/f'submissions/synthetic_rankblend_seed{b["seed"]}.csv'
    write_submission(data,rows,score,out);print(out)


if __name__=='__main__':main()
