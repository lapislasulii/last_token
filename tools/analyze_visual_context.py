"""Analyze held-out visual scores, including ties and geometric reason groups."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parents[1]


def main():
    out=ROOT/'outputs/visual_context_analysis'
    out.mkdir(parents=True,exist_ok=True)
    reasons=pd.read_csv(ROOT/'outputs/error_reasons/train_reason_labels.csv',dtype={'id':str})
    scores=pd.read_csv(ROOT/'oof/visual_residual_context_l1.csv',dtype={'id':str})
    df=scores.rename(columns={'score':'visual_score'}).merge(reasons.drop(columns=['day']),on='id',validate='one_to_one')
    assert len(df)==198 and df.id.is_unique
    df['baseline_score']=1-df.confidence
    records=[]
    for day,g in df.groupby('day'):
        for col in ('baseline_score','visual_score'):
            df.loc[g.index,col+'_rank']=g[col].rank(ascending=False,method='average')
        neg=g[g.error==0]
        # Number of correctly ordered error/correct pairs, half credit for ties.
        for i,row in g[g.error==1].iterrows():
            credit=[]
            for col in ('baseline_score','visual_score'):
                diff=row[col]-neg[col].to_numpy()
                credit.append(np.sum(diff>0)+.5*np.sum(diff==0))
            df.loc[i,'pair_gain']=credit[1]-credit[0]
        for cls,h in [('ALL',g),*list(g.groupby('class_name'))]:
            records.append({'day':day,'class':cls,'n':len(h),'errors':int(h.error.sum()),
                'baseline_ap':average_precision_score(h.error,h.baseline_score),'visual_ap':average_precision_score(h.error,h.visual_score)})
        for col in ('baseline_score','visual_score'):
            # Expected errors in top K under uniform random tie breaking.
            for k in (10,20):
                cutoff=g[col].nlargest(k).min()
                above=g[g[col]>cutoff]; tied=g[g[col]==cutoff]
                expected=above.error.sum()+(k-len(above))*tied.error.mean()
                print(day,col,'top',k,'expected errors',round(expected,3))
    df['rank_gain']=df.baseline_score_rank-df.visual_score_rank
    df.to_csv(out/'per_detection.csv',index=False)
    metrics=pd.DataFrame(records)
    metrics['delta']=metrics.visual_ap-metrics.baseline_ap
    metrics.to_csv(out/'class_metrics.csv',index=False)
    group=df[df.error==1].groupby(['day','reason']).agg(n=('id','size'),pair_gain=('pair_gain','sum'),mean_rank_gain=('rank_gain','mean'))
    group.to_csv(out/'reason_metrics.csv')
    print(metrics.round(5).to_string(index=False));print(group.round(2).to_string())
    chosen=pd.concat([df[df.error==1].nlargest(4,'pair_gain'),df[df.error==1].nsmallest(4,'pair_gain'),df[df.error==0].nlargest(4,'rank_gain')]).drop_duplicates('id')
    canvas=Image.new('RGB',(1280,300*((len(chosen)+1)//2)),'white')
    draw=ImageDraw.Draw(canvas)
    for j,row in enumerate(chosen.itertuples()):
        image=Image.open(ROOT/'data/rascar-hack-case-1'/f'{row.frame_id}.jpg').convert('RGB')
        image.thumbnail((640,240))
        x,y=(j%2)*640,(j//2)*300
        canvas.paste(image,(x,y+60))
        sx,sy=image.width/1280,image.height/720
        draw.rectangle((x+row.x1*sx,y+60+row.y1*sy,x+row.x2*sx,y+60+row.y2*sy),outline='red' if row.error else 'green',width=3)
        draw.text((x+4,y+3),f'{row.id} {row.class_name} error={row.error}\n{row.day} rank {row.baseline_score_rank:g} -> {row.visual_score_rank:g}\n{row.reason} {row.geometry_subtype}',fill='black')
    canvas.save(out/'examples.jpg')
    chosen.to_csv(out/'examples.csv',index=False)


if __name__=='__main__':main()
