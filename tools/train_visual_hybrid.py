"""Joint residual: frozen context PCA plus class/geometry, optional temporal cues."""
import json
import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit,logit
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.metrics import average_precision_score
from train_logreg import ROOT,features,GEOMETRY
from train_temporal import write_submission
from temporal_features import add_temporal,CORE


def design_fit(x,rows,temporal):
    visual=make_pipeline(StandardScaler(),PCA(n_components=16,whiten=True,svd_solver='full'))
    numeric=GEOMETRY+(CORE if temporal else [])
    tabular=ColumnTransformer([('numeric',StandardScaler(),numeric),('class',OneHotEncoder(handle_unknown='ignore',sparse_output=False),['class_name'])])
    a=np.column_stack([visual.fit_transform(x),tabular.fit_transform(rows)])
    return visual,tabular,a


def fit(x,rows,temporal,strength):
    visual,tabular,a=design_fit(x,rows,temporal)
    offset=logit(np.clip(1-rows.confidence.to_numpy(),1e-5,1-1e-5))
    y=rows.error.to_numpy()
    def objective(w):
        z=offset+a@w
        return np.mean(np.logaddexp(0,z)-y*z)+strength*np.dot(w,w)/2,a.T@(expit(z)-y)/len(y)+strength*w
    r=minimize(objective,np.zeros(a.shape[1]),jac=True,method='L-BFGS-B',options={'maxiter':1000,'gtol':1e-8})
    assert r.success,r.message
    return {'visual':visual,'tabular':tabular,'coef':r.x,'temporal':temporal,'strength':strength}


def predict(b,x,rows):
    a=np.column_stack([b['visual'].transform(x),b['tabular'].transform(rows)])
    return expit(logit(np.clip(1-rows.confidence.to_numpy(),1e-5,1-1e-5))+a@b['coef'])


def main():
    data=ROOT/'data/rascar-hack-case-1'
    policy=json.loads((data/'scoring_policy.json').read_text())
    train,test=[add_temporal(features(data,s,policy),data,s,policy) for s in ('train','test')]
    z=np.load(ROOT/'outputs/visual_features/dinov2_small.npz')
    assert np.array_equal(z['train_ids'],train.id) and np.array_equal(z['test_ids'],test.id)
    x,xt=z['train_context'],z['test_context']
    days=sorted(train.day.unique())
    records=[]
    for temporal in (False,True):
        for strength in (1.,10.):
            name=f'visual_hybrid_{"temporal" if temporal else "geometry"}_l{strength:g}'
            oof=np.full(len(train),np.nan)
            for day in days:
                mask=(train.day!=day).to_numpy()
                b=fit(x[mask],train.loc[mask],temporal,strength)
                oof[~mask]=predict(b,x[~mask],train.loc[~mask])
            train[['id','day']].assign(score=oof).to_csv(ROOT/f'oof/{name}.csv',index=False)
            ap={d:average_precision_score(train.loc[train.day==d,'error'],oof[train.day==d]) for d in days}
            records.append({'method':name,**ap,'mean':np.mean(list(ap.values()))})
            b=fit(x,train,temporal,strength)
            b['encoder']=json.loads((ROOT/'outputs/visual_features/metadata.json').read_text())
            joblib.dump(b,ROOT/f'models/{name}.joblib')
            score=predict(b,xt,test)
            np.testing.assert_allclose(score,predict(joblib.load(ROOT/f'models/{name}.joblib'),xt,test))
            write_submission(data,test,score,ROOT/f'submissions/{name}.csv')
    metrics=pd.DataFrame(records).set_index('method')
    metrics.to_csv(ROOT/'outputs/visual_features/hybrid_metrics.csv')
    print(metrics.round(6).to_string())


if __name__=='__main__':
    main()
