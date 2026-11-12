"""OOF-only person geometry specialist; fixed four combinations, no test labels."""
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from train_logreg import ROOT,features,GEOMETRY


def main():
    data=ROOT/'data/rascar-hack-case-1'
    rows=features(data,'train',json.loads((data/'scoring_policy.json').read_text()))
    reasons=pd.read_csv(ROOT/'outputs/error_reasons/train_reason_labels.csv',dtype={'id':str}).set_index('id')
    rows['geometry_target']=((rows.error==1)&rows.id.map(reasons.reason).eq('geometry_candidate')).astype(int)
    old=pd.read_csv(ROOT/'oof/visual_residual_context_l1.csv',dtype={'id':str}).set_index('id')
    assert set(old.index)==set(rows.id)
    baseline=old.loc[rows.id,'score'].to_numpy()
    z=np.load(ROOT/'outputs/visual_features/dinov2_small.npz')
    assert np.array_equal(z['train_ids'],rows.id)
    x=np.column_stack([z['train_box'],z['train_context']])
    numeric=rows[GEOMETRY].copy()
    numeric['edge_left']=rows.x1/rows.image_width
    numeric['edge_right']=1-rows.x2/rows.image_width
    numeric['edge_top']=rows.y1/rows.image_height
    numeric['edge_bottom']=1-rows.y2/rows.image_height
    person=(rows.class_name=='person').to_numpy()
    scores={'current_context':baseline};targets=[]
    # 8 visual PCA components; no human-derived input features.
    for c in (.01,.1):
        probability=np.zeros(len(rows))
        for day in sorted(rows.day.unique()):
            fit=person&(rows.day!=day).to_numpy(); valid=person&(rows.day==day).to_numpy()
            scaler=StandardScaler().fit(x[fit]);pca=PCA(n_components=8,whiten=True,svd_solver='full').fit(scaler.transform(x[fit]))
            ns=StandardScaler().fit(numeric.loc[fit])
            a=np.column_stack([pca.transform(scaler.transform(x[fit])),ns.transform(numeric.loc[fit])])
            b=np.column_stack([pca.transform(scaler.transform(x[valid])),ns.transform(numeric.loc[valid])])
            model=LogisticRegression(C=c,max_iter=2000,random_state=42).fit(a,rows.loc[fit,'geometry_target'])
            probability[valid]=model.predict_proba(b)[:,1]
            targets.append({'C':c,'day':day,'fit_person':int(fit.sum()),'fit_geometry':int(rows.loc[fit,'geometry_target'].sum()),'valid_geometry':int(rows.loc[valid,'geometry_target'].sum()),'geometry_ap':average_precision_score(rows.loc[valid,'geometry_target'],probability[valid])})
        for weight in (.25,.5):
            # Only raise scores toward specialist estimate, never lower other errors.
            s=baseline.copy();s[person]+=weight*np.maximum(0,probability[person]-baseline[person])
            scores[f'person_geometry_c{c:g}_w{weight:g}']=s
    output=ROOT/'outputs/person_geometry';output.mkdir(parents=True,exist_ok=True)
    records=[]
    for name,s in scores.items():
        assert np.isfinite(s).all() and ((s>=0)&(s<=1)).all()
        rows[['id','day']].assign(score=s).to_csv(ROOT/f'oof/{name}.csv',index=False)
        for day,g in rows.groupby('day'):
            v=g.assign(score=s[g.index]).sort_values(['score','id'],ascending=[False,True])
            half=v.iloc[:len(v)//2]; top=v.iloc[:20]
            records.append({'method':name,'day':day,'AP':average_precision_score(g.error,s[g.index]),
                'person_AP':average_precision_score(g.loc[g.class_name=='person','error'],s[g.index[g.class_name=='person']]),
                'geometry_caught_top_half':int(half.loc[half.class_name=='person','geometry_target'].sum()),
                'other_errors_caught_top_half':int(((half.error==1)&~((half.class_name=='person')&(half.geometry_target==1))).sum()),
                'correct_person_top_half':int(((half.class_name=='person')&(half.error==0)).sum()),
                'errors_top20':int(top.error.sum())})
    metrics=pd.DataFrame(records);metrics.to_csv(output/'metrics.csv',index=False)
    pd.DataFrame(targets).to_csv(output/'specialist_metrics.csv',index=False)
    print(metrics.round(5).to_string(index=False));print(pd.DataFrame(targets).round(5).to_string(index=False))


if __name__=='__main__':main()
