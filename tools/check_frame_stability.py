"""Sensitivity of fixed day-OOF AP to removal/resampling of evaluation frames.

Does not retrain. Paired frame bootstrap is descriptive, not new-day validation.
"""
import json
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from train_logreg import ROOT,features

NAMES=['baseline','marked_context_clip_l1','clip_inside_outside']


def metrics(g):
    return {n:float(average_precision_score(g.error,g[n])) for n in NAMES}


def main():
    data=ROOT/'data/rascar-hack-case-1'
    r=features(data,'train',json.loads((data/'scoring_policy.json').read_text()))
    r['baseline']=1-r.confidence
    for name in NAMES[1:]:
        s=pd.read_csv(ROOT/f'oof/{name}.csv',dtype={'id':str})
        assert s.id.is_unique and set(s.id)==set(r.id)
        r[name]=s.set_index('id').loc[r.id,'score'].to_numpy()
        assert np.isfinite(r[name]).all() and r[name].between(0,1).all()
        assert (s.set_index('id').loc[r.id,'day'].to_numpy()==r.day.to_numpy()).all()
    out=ROOT/'outputs/frame_stability';out.mkdir(parents=True,exist_ok=True)
    full={day:metrics(g) for day,g in r.groupby('day')};records=[]
    for day,g in r.groupby('day'):
        for frame,removed in g.groupby('frame_id'):
            keep=g[g.frame_id!=frame];m=metrics(keep)
            gap=m[NAMES[2]]-m[NAMES[1]]
            mean={n:np.mean([m[n] if d==day else full[d][n] for d in full]) for n in NAMES}
            records.append({'day':day,'frame_id':frame,'removed_boxes':len(removed),'removed_errors':int(removed.error.sum()),
                'classes':','.join(f'{c}:{len(h)}' for c,h in removed.groupby('class_name')),
                **m,'new_minus_current':gap,'gap_change':gap-(full[day][NAMES[2]]-full[day][NAMES[1]]),
                'mean_gap':mean[NAMES[2]]-mean[NAMES[1]],'mean_current_minus_baseline':mean[NAMES[1]]-mean['baseline']})
    loo=pd.DataFrame(records);loo.to_csv(out/'leave_one_frame_out.csv',index=False)
    print('FULL',json.dumps(full));print('REMOVAL SUMMARY')
    print(loo.groupby('day').agg(n=('frame_id','size'),min_gap=('new_minus_current','min'),max_gap=('new_minus_current','max'),reversals=('new_minus_current',lambda x:int((x<=0).sum())),min_mean_gap=('mean_gap','min'),max_mean_gap=('mean_gap','max')).to_string())
    print('INFLUENTIAL');print(loo.loc[loo.gap_change.abs().sort_values(ascending=False).index].head(8).to_string(index=False))
    # Same resampled frames for all methods. Preserve number of frames in each day.
    rng=np.random.default_rng(42);boot=[]
    days=list(full)
    for iteration in range(1000):
        daily=[]
        for day in days:
            g=r[r.day==day];frames=g.frame_id.unique()
            sample=pd.concat([g[g.frame_id==f] for f in rng.choice(frames,len(frames),replace=True)],ignore_index=True)
            if sample.error.nunique()<2:break
            m=metrics(sample);daily.append(m[NAMES[2]]-m[NAMES[1]])
        if len(daily)==len(days):boot.append([*daily,float(np.mean(daily))])
    b=pd.DataFrame(boot,columns=[*days,'mean']);b.to_csv(out/'paired_frame_bootstrap.csv',index=False)
    print('BOOTSTRAP descriptive quantiles');print(b.quantile([.025,.5,.975]).to_string());print('positive fraction',b.gt(0).mean().to_dict())
    (out/'summary.json').write_text(json.dumps({'full':full,'bootstrap_valid':len(b),'bootstrap_mean_quantiles':b['mean'].quantile([.025,.5,.975]).to_dict(),'bootstrap_mean_positive_fraction':float(b['mean'].gt(0).mean()),'retraining':False},indent=2))


if __name__=='__main__':main()
