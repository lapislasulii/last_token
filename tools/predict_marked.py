"""Restore a published marked head and predict using its extracted test cache."""
import argparse,json
from pathlib import Path
import joblib
import numpy as np
from train_logreg import ROOT,features
from train_visual_residual import predict
from train_temporal import write_submission


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model',type=Path)
    args=parser.parse_args()
    bundle=joblib.load(args.model)
    name=args.model.stem
    directory=ROOT/'outputs'/name.removesuffix('_l1')
    meta_path=directory/'encoder.json'
    if meta_path.exists():
        meta=json.loads(meta_path.read_text())
        assert meta['model']==bundle['encoder'] and meta['revision']==bundle['revision']
    else:
        raise ValueError('First run train_marked_context.py for this encoder to create its cache.')
    z=np.load(directory/'test_features.npz')
    data=ROOT/'data/rascar-hack-case-1'
    rows=features(data,'test',json.loads((data/'scoring_policy.json').read_text()))
    assert np.array_equal(rows.id,z['ids'])
    score=predict(bundle,z['features'],rows.confidence.to_numpy())
    output=ROOT/f'submissions/{name}.csv'
    write_submission(data,rows,score,output)
    print(output)


if __name__=='__main__':main()
