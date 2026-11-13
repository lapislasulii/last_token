"""Fixed 10%/25% second-opinion blends for cup/hands only. No training or upload."""
import json
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from train_logreg import ROOT,features


def expected_top_errors(g,column,k=20):
    k=min(k,len(g));threshold=g[column].nlargest(k).min()
    higher=g[g[column]>threshold];tied=g[g[column]==threshold]
    return float(higher.error.sum()+(k-len(higher))*tied.error.mean())


def main():
    data=ROOT/'data/rascar-hack-case-1'
    r=features(data,'train',json.loads((data/'scoring_policy.json').read_text())).set_index('id')
    clip=pd.read_csv(ROOT/'oof/marked_context_clip_l1.csv',dtype={'id':str}).set_index('id')
    second=pd.read_csv(ROOT/'outputs/redetect/oof_features.csv',dtype={'id':str,'day':str}).set_index('id')
    for table in (clip,second):assert table.index.is_unique and set(table.index)==set(r.index)
    clip=clip.loc[r.index];second=second.loc[r.index]
    assert np.array_equal(clip.day,r.day)
    assert np.array_equal(second.day,r.day.str.replace('-',''))
    assert np.array_equal(second.error,r.error) and np.array_equal(second.frame_id,r.frame_id)
    assert np.array_equal(second.class_name,r.class_name)
    r['clip']=clip.score
    r['second']=1-second.match_conf
    assert r[['clip','second']].apply(lambda s:np.isfinite(s).all() and s.between(0,1).all()).all()
    mask=r.class_name.isin(['cup','nude_hands'])
    names=['clip']
    for weight in (.1,.25):
        name=f'clip_redetect_w{weight:g}';names.append(name)
        r[name]=r['clip'];r.loc[mask,name]=(1-weight)*r.loc[mask,'clip']+weight*r.loc[mask,'second']
        np.testing.assert_array_equal(r.loc[~mask,name],r.loc[~mask,'clip'])
        r[['day',name]].rename(columns={name:'score'}).to_csv(ROOT/f'oof/{name}.csv')
    records=[]
    for day,g in r.groupby('day'):
        for cls,h in [('ALL',g),*[(c,g[g.class_name==c]) for c in ['cup','nude_hands']]]:
            for name in names:
                records.append({'day':day,'group':cls,'method':name,'n':len(h),'AP':average_precision_score(h.error,h[name]),'top20_errors':expected_top_errors(h,name) if cls=='ALL' else None})
    out=ROOT/'outputs/clip_redetect_blend';out.mkdir(parents=True,exist_ok=True)
    m=pd.DataFrame(records);m.to_csv(out/'metrics.csv',index=False)
    r.to_csv(out/'joined_oof.csv')
    print(m.round(6).to_string(index=False))
    full=m[m.group=='ALL'].pivot(index='method',columns='day',values='AP')
    full['mean']=full.mean(axis=1);print('SUMMARY');print(full.round(6).to_string())
    # Only proceed to deletion sensitivity if a blend beats CLIP on both days.
    days=sorted(r.day.unique())
    qualifying=[n for n in names[1:] if (full.loc[n,days]>full.loc['clip',days]).all()]
    print('Improves both days:',qualifying)
    sensitivity=[]
    for name in qualifying:
        for day,g in r.groupby('day'):
            for frame in g.frame_id.unique():
                h=g[g.frame_id!=frame]
                sensitivity.append({'method':name,'day':day,'frame_removed':frame,'AP_delta':average_precision_score(h.error,h[name])-average_precision_score(h.error,h['clip'])})
    if sensitivity:pd.DataFrame(sensitivity).to_csv(out/'frame_sensitivity.csv',index=False)


if __name__=='__main__':main()
