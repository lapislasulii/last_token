"""Descriptive held-out diagnostics; no fitting or Kaggle submissions."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score as ap

ROOT = Path(__file__).resolve().parents[1]
out = ROOT / 'outputs/redetect_strong/diagnostics'
out.mkdir(exist_ok=True)
f = pd.read_csv(ROOT / 'outputs/redetect_strong/oof_features.csv', dtype={'id':str,'day':str})
v = pd.read_csv(ROOT / 'outputs/expert_veto/train_features.csv', dtype={'id':str})
f = f.merge(v[['id','veto']],on='id',validate='one_to_one')
f['D'] = (f.iou_same < .6).astype(int)
f['baseline'] = 1-f.confidence
f['expert'] = (f.baseline+f.veto)/2
f['combined'] = (f.baseline+f.D+2*f.veto)/4
f['cups_hands'] = (f.baseline+f.D*f.class_name.isin(['cup','nude_hands'])+2*f.veto)/4
metrics=[]
for scope, group in [('all',f)]+[(str(k),g) for k,g in f.groupby('day')]+[(str(k),g) for k,g in f.groupby('class_name')]:
    row=dict(scope=scope,n=len(group),errors=int(group.error.sum()))
    row.update({s:ap(group.error,group[s]) for s in ['baseline','expert','combined','cups_hands']})
    metrics.append(row)
print(pd.DataFrame(metrics).round(5).to_string(index=False))
pd.DataFrame(metrics).to_csv(out/'metrics.csv',index=False)
print('\nFLAGS BY CLASS')
flags=f.groupby(['class_name','veto','D']).error.agg(['size','sum']).reset_index()
print(flags.to_string(index=False));flags.to_csv(out/'flags.csv',index=False)
for day,g in f.groupby('day'):
    ordered=g.sort_values(['combined','id'],ascending=[False,True])
    print('DAY',day,'top20 errors',int(ordered.head(20).error.sum()),'of',int(g.error.sum()))
# Each inversion is a correct box scored above an error; ties contribute one half.
errors=f[f.error==1].copy();correct=f[f.error==0].copy()
errors['correct_above']= [sum((correct.day==r.day)&(correct.combined>r.combined))+.5*sum((correct.day==r.day)&(correct.combined==r.combined)) for r in errors.itertuples()]
correct['errors_below']=[sum((errors.day==r.day)&(errors.combined<r.combined))+.5*sum((errors.day==r.day)&(errors.combined==r.combined)) for r in correct.itertuples()]
errors.sort_values('correct_above',ascending=False).to_csv(out/'missed_errors.csv',index=False)
correct.sort_values('errors_below',ascending=False).to_csv(out/'false_alarms.csv',index=False)
reasons=pd.read_csv(ROOT/'outputs/error_reasons/train_reason_labels.csv',dtype={'id':str})
merged=errors.merge(reasons,on='id',suffixes=('','_reason'),validate='one_to_one')
merged.to_csv(out/'missed_with_reasons.csv',index=False)
print('\nREASON COLUMNS',list(reasons.columns))
print('\nMOST BURIED ERRORS')
cols=['id','class_name','day','combined','correct_above']+[c for c in reasons.columns if 'reason' in c and c in merged.columns]
print(merged.sort_values('correct_above',ascending=False)[cols].head(15).to_string(index=False))
print('\nFALSE ALARMS')
print(correct.sort_values('errors_below',ascending=False)[['id','class_name','day','veto','D','errors_below']].head(10).to_string(index=False))
