"""Descriptive distribution comparison; never infers or annotates test error labels."""
import json
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import ks_2samp
from train_logreg import ROOT,features,GEOMETRY
from temporal_features import add_temporal,CORE,DETAIL


def main():
    data=ROOT/'data/rascar-hack-case-1';policy=json.loads((data/'scoring_policy.json').read_text())
    out=ROOT/'outputs/train_test_shift';out.mkdir(parents=True,exist_ok=True)
    groups={};images=[];inventory=[]
    for split in ('train','test'):
        rows=add_temporal(features(data,split,policy),data,split,policy)
        rows['hour']=pd.to_datetime(rows.timestamp,utc=True,format='mixed').dt.hour
        rows['split']=split
        groups[split]=rows
        frames=pd.read_csv(data/f'{split}_frames.csv').set_index('frame_id')
        raw=pd.read_csv(data/f'{split}.csv')
        inventory.append({'split':split,'raw_rows':len(raw),'scored_rows':len(rows),'raw_target_frames':raw.frame_id.nunique(),'active_frames':rows.frame_id.nunique(),'frames_including_context':len(frames),'cameras':str(frames.camera_id.unique().tolist()),'locations':str(frames.location_id.unique().tolist())})
        # Only frames with scored detections, one record per frame (not per box).
        for frame in rows.frame_id.unique():
            f=frames.loc[frame]
            with Image.open(data/f.image_path) as im:
                width,height=im.size
                a=np.asarray(im.convert('RGB').resize((160,90)),dtype=float)/255
            brightness=a.mean(axis=2)
            images.append({'frame_id':frame,'split':split,'day':str(pd.to_datetime(f.timestamp,utc=True).date()),'width':width,'height':height,'brightness':brightness.mean(),'dark_fraction':(brightness<.1).mean(),'bright_fraction':(brightness>.9).mean(),'channel_difference':np.abs(a[:,:,0]-a[:,:,1]).mean()+np.abs(a[:,:,1]-a[:,:,2]).mean()})
    rows=pd.concat(groups.values(),ignore_index=True);im=pd.DataFrame(images)
    rows.drop(columns=['error'],errors='ignore').to_csv(out/'unlabeled_features.csv',index=False)
    im.to_csv(out/'frame_statistics.csv',index=False)
    print('INVENTORY');print(pd.DataFrame(inventory).to_string(index=False))
    print('CLASS COUNTS');print(pd.crosstab(rows.day,rows.class_name).to_string())
    print('CLASS DISTRIBUTION');print(pd.crosstab(rows.split,rows.class_name,normalize='index').round(3).to_string())
    print('CLASS MEANS');print(rows.groupby(['split','class_name'])[['confidence','area','width','height','neighbor_iou_max']].mean().round(3).to_string())
    print('DAY OVERVIEW');print(rows.groupby(['split','day']).agg(boxes=('id','size'),frames=('frame_id','nunique'),confidence=('confidence','mean'),hour=('hour','mean')).round(3).to_string())
    print('IMAGES');print(im.groupby(['split','day'])[['brightness','dark_fraction','bright_fraction','channel_difference']].mean().round(3).to_string())
    bins=[0,6,12,18,24]
    print('HOURS UTC ACTIVE FRAMES');f=rows.drop_duplicates('frame_id');print(pd.crosstab(f.day,pd.cut(f.hour,bins,right=False)).to_string())
    records=[]
    for cls in ['ALL',*sorted(rows.class_name.unique())]:
        a,b=[g if cls=='ALL' else g[g.class_name==cls] for g in groups.values()]
        for col in ['confidence',*GEOMETRY,*CORE,*DETAIL]:
            records.append({'class':cls,'feature':col,'train_mean':a[col].mean(),'test_mean':b[col].mean(),'train_median':a[col].median(),'test_median':b[col].median(),'KS_distance':ks_2samp(a[col],b[col]).statistic})
    stats=pd.DataFrame(records);stats.to_csv(out/'feature_comparison.csv',index=False)
    print('TOP DIFFERENCES DESCRIPTIVE KS (NO INDEPENDENCE CLAIM)');print(stats.sort_values('KS_distance',ascending=False).head(18).round(3).to_string(index=False))
    pd.DataFrame(inventory).to_csv(out/'inventory.csv',index=False)


if __name__=='__main__':main()
