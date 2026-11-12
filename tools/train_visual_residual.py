"""Regularized image correction to fixed detector log odds; two day folds."""
import json
import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score
from train_logreg import ROOT, features
from train_temporal import write_submission


def fit(x, confidence, y, strength):
    scaler = StandardScaler().fit(x)
    # Reduce degrees of freedom; both transforms fitted on training day only.
    pca = PCA(n_components=16, whiten=True, svd_solver='full').fit(scaler.transform(x))
    a = pca.transform(scaler.transform(x))
    offset = logit(np.clip(1-confidence,1e-5,1-1e-5))
    def objective(w):
        z = offset+a@w
        return np.mean(np.logaddexp(0,z)-y*z)+strength*np.dot(w,w)/2, a.T@(expit(z)-y)/len(y)+strength*w
    result = minimize(objective,np.zeros(a.shape[1]),jac=True,method='L-BFGS-B',options={'maxiter':1000,'gtol':1e-8})
    assert result.success, result.message
    return {'scaler':scaler,'pca':pca,'coef':result.x,'strength':strength}


def predict(bundle,x,confidence):
    a = bundle['pca'].transform(bundle['scaler'].transform(x))
    return expit(logit(np.clip(1-confidence,1e-5,1-1e-5))+a@bundle['coef'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predict',type=Path,help='Saved residual head; requires extracted test features')
    args=parser.parse_args()
    data=ROOT/'data/rascar-hack-case-1'
    policy=json.loads((data/'scoring_policy.json').read_text())
    z=np.load(ROOT/'outputs/visual_features/dinov2_small.npz')
    test=features(data,'test',policy)
    if args.predict:
        assert np.array_equal(z['test_ids'],test.id)
        b=joblib.load(args.predict)
        metadata=json.loads((ROOT/'outputs/visual_features/metadata.json').read_text())
        for key in ('model','revision','preprocessing','fingerprint'):
            assert b['encoder'][key]==metadata[key], f'Encoder/cache mismatch: {key}'
        xt=np.concatenate([z['test_box'],z['test_context']],axis=1) if b['view']=='both' else z['test_'+b['view']]
        output=ROOT/f'submissions/{args.predict.stem}.csv'
        write_submission(data,test,predict(b,xt,test.confidence.to_numpy()),output)
        print(output)
        return
    train=features(data,'train',policy)
    for folder in ('models','oof','submissions'):
        (ROOT/folder).mkdir(exist_ok=True)
    assert np.array_equal(z['train_ids'],train.id) and np.array_equal(z['test_ids'],test.id)
    days=sorted(train.day.unique())
    records=[]
    # Fixed six candidates. Higher penalty means a smaller visual correction.
    for view in ('box','context','both'):
        x,xt=[np.concatenate([z[f'{s}_box'],z[f'{s}_context']],axis=1) if view=='both' else z[f'{s}_{view}'] for s in ('train','test')]
        for strength in (1.,10.):
            name=f'visual_residual_{view}_l{strength:g}'
            oof=np.full(len(train),np.nan)
            for day in days:
                fitmask=(train.day!=day).to_numpy()
                b=fit(x[fitmask],train.confidence.to_numpy()[fitmask],train.error.to_numpy()[fitmask],strength)
                oof[~fitmask]=predict(b,x[~fitmask],train.confidence.to_numpy()[~fitmask])
            train[['id','day']].assign(score=oof).to_csv(ROOT/f'oof/{name}.csv',index=False)
            ap={d:average_precision_score(train.loc[train.day==d,'error'],oof[train.day==d]) for d in days}
            records.append({'method':name,**ap,'mean':np.mean(list(ap.values()))})
            b=fit(x,train.confidence.to_numpy(),train.error.to_numpy(),strength)
            b.update(view=view,encoder=json.loads((ROOT/'outputs/visual_features/metadata.json').read_text()))
            joblib.dump(b,ROOT/f'models/{name}.joblib')
            score=predict(b,xt,test.confidence.to_numpy())
            np.testing.assert_allclose(score,predict(joblib.load(ROOT/f'models/{name}.joblib'),xt,test.confidence.to_numpy()))
            write_submission(data,test,score,ROOT/f'submissions/{name}.csv')
    metrics=pd.DataFrame(records).set_index('method')
    metrics.to_csv(ROOT/'outputs/visual_features/residual_metrics.csv')
    print(metrics.round(6).to_string())


if __name__=='__main__':
    main()
