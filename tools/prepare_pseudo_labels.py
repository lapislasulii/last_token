"""Validate reviewed machine-pool labels and produce leakage-aware day-fold manifests.

This does not train a model or claim that pseudo-labels are ground truth.
"""
import argparse,json
from pathlib import Path
import pandas as pd
from train_logreg import ROOT


def filter_fold(rows,day):
    valid=rows.day.ne(day)&rows.annotation_train_days.map(lambda ds:day not in ds)
    return rows[valid&rows.review_status.eq('accepted')&rows.pseudo_error.notna()].copy()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('csv',type=Path)
    args=parser.parse_args();data=ROOT/'data/rascar-hack-case-1'
    r=pd.read_csv(args.csv,dtype={'id':str,'frame_id':str})
    required={'id','frame_id','pseudo_error','pseudo_confidence','review_status','annotator_model','annotation_train_days'}
    assert required<=set(r.columns),f'Missing: {required-set(r.columns)}'
    assert r.id.notna().all() and r.id.is_unique and r.frame_id.notna().all()
    assert r.pseudo_error.dropna().isin([0,1]).all()
    assert r.pseudo_confidence.between(0,1).all()
    assert r.review_status.isin(['accepted','rejected','uncertain']).all()
    assert r.annotator_model.fillna('').str.len().gt(0).all()
    r['annotation_train_days']=r.annotation_train_days.map(json.loads)
    assert r.annotation_train_days.map(lambda x:isinstance(x,list) and all(isinstance(d,str) and pd.Timestamp(d).strftime('%Y-%m-%d')==d for d in x)).all()
    machine=pd.read_csv(data/'machine_predictions.csv',dtype={'detection_id':str,'frame_id':str})
    machine=machine.rename(columns={'detection_id':'id'})
    joined=r.merge(machine[['id','frame_id']],on=['id','frame_id'],how='left',indicator=True,validate='one_to_one')
    assert joined._merge.eq('both').all(),'Unknown machine-pool ID/frame pair'
    frames=pd.read_csv(data/'machine_frames.csv')
    days=pd.to_datetime(frames.timestamp,utc=True,format='mixed').dt.strftime('%Y-%m-%d')
    r['day']=r.frame_id.map(dict(zip(frames.frame_id,days)));assert r.day.notna().all()
    # Test and official train/context frames are never auxiliary training examples.
    forbidden=set()
    for split in ('train','test'):forbidden.update(pd.read_csv(data/f'{split}_frames.csv').frame_id)
    r=r[~r.frame_id.isin(forbidden)].copy()
    out=ROOT/'outputs/pseudo_ready';out.mkdir(parents=True,exist_ok=True)
    summary={}
    for day in ['2026-10-25','2026-10-26']:
        ready=filter_fold(r,day);summary[day]={'rows':len(ready),'frames':ready.frame_id.nunique(),'pseudo_errors':int(ready.pseudo_error.sum())}
        ready['annotation_train_days']=ready.annotation_train_days.map(json.dumps)
        ready.to_csv(out/f'fit_excluding_{day}.csv',index=False)
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
