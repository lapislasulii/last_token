"""Synthetic box geometry supervision. Train-only generation, day-held-out scoring."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image,ImageOps,ImageDraw
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score,roc_auc_score
import joblib
from train_logreg import ROOT,features
from train_temporal import write_submission
from temporal_features import ious

CLASSES=['person','nude_hands','cup','tray']
MODEL='openai/clip-vit-large-patch14'
REVISION='32bd64288804d66eefd0ccbe215aa642df71cc41'
CONFUSERS={'cup':['empty_plate','phone','personal_items','nude_hands','carton'],
 'nude_hands':['phone','cup','bottle','bracelet','uniform_not_ok'],
 'tray':['empty_plate','phone'],'person':['personal_items']}


def max_overlap(box,candidates):
    return float(ious(np.asarray(box),np.asarray(candidates).reshape(-1,4)).max()) if len(candidates) else 0.


def crop_image(im,box,width,height):
    sx,sy=im.width/width,im.height/height
    x1,y1,x2,y2=np.array(box)*[sx,sy,sx,sy];w,h=x2-x1,y2-y1
    bounds=(max(0,int(np.floor(x1-w*.25))),max(0,int(np.floor(y1-h*.25))),min(im.width,int(np.ceil(x2+w*.25))),min(im.height,int(np.ceil(y2+h*.25))))
    raw=im.crop(bounds);fit=ImageOps.contain(raw,(224,224),method=Image.Resampling.BICUBIC)
    crop=ImageOps.pad(raw,(224,224),method=Image.Resampling.BICUBIC,color=(124,116,104))
    ox,oy=round((224-fit.width)/2),round((224-fit.height)/2);ax,ay=fit.width/raw.width,fit.height/raw.height
    ImageDraw.Draw(crop).rectangle((ox+(x1-bounds[0])*ax,oy+(y1-bounds[1])*ay,ox+(x2-bounds[0])*ax,oy+(y2-bounds[1])*ay),outline=(255,0,0),width=2)
    return crop


def generate(data,policy,seed):
    rng=np.random.default_rng(seed)
    frames=pd.read_csv(data/'train_frames.csv').set_index('frame_id')
    human=pd.read_csv(data/'human_boxes_train.csv');human.class_name=human.class_name.replace(policy['aliases'])
    human=human.merge(frames[['image_width','image_height','timestamp']],left_on='frame_id',right_index=True,validate='many_to_one')
    for name,value in [('x1',(human.cx-human.w/2)*human.image_width),('x2',(human.cx+human.w/2)*human.image_width),('y1',(human.cy-human.h/2)*human.image_height),('y2',(human.cy+human.h/2)*human.image_height)]:human[name]=value
    xy=['x1','y1','x2','y2'];groups={(f,c):g[xy].to_numpy() for (f,c),g in human.groupby(['frame_id','class_name'])}
    records=[]
    def add(r,b,cls,kind,negative_only=False):
        b=np.clip(b,[0,0,0,0],[r.image_width,r.image_height,r.image_width,r.image_height])
        if b[2]-b[0]<2 or b[3]-b[1]<2:return
        overlap=max_overlap(b,groups.get((r.frame_id,cls),[]))
        if negative_only and overlap>=policy['correct_iou']:return
        records.append(dict(frame_id=r.frame_id,day=str(pd.to_datetime(r.timestamp,utc=True).date()),class_name=cls,
            x1=b[0],y1=b[1],x2=b[2],y2=b[3],error=int(overlap<policy['correct_iou']),iou=overlap,kind=kind,
            image_width=r.image_width,image_height=r.image_height))
    for r in human.itertuples():
        original=np.array([r.x1,r.y1,r.x2,r.y2]);w,h=r.x2-r.x1,r.y2-r.y1
        if r.class_name in CLASSES:
            for i in range(30):
                b=original.copy()
                if i<9:kind='small';b+=rng.uniform(-.08,.08,4)*[w,h,w,h]
                elif i<21 or r.class_name!='person':
                    kind='large_noise';cx,cy=(r.x1+r.x2)/2,(r.y1+r.y2)/2
                    cx+=rng.choice([-1,1])*rng.uniform(.05,.25)*w;cy+=rng.choice([-1,1])*rng.uniform(.05,.25)*h
                    nw,nh=rng.uniform(.7,1.4,2)*[w,h];b=np.array([cx-nw/2,cy-nh/2,cx+nw/2,cy+nh/2])
                elif i<25:
                    kind='cut_bottom';b[3]=r.y1+h*rng.uniform(.4,.8);b[[0,2]]+=rng.uniform(-.05,.05,2)*w
                else:
                    kind='enlarge';side=rng.choice([0,2,3]);factor=rng.uniform(1.4,2.3)
                    b[side]+=(-1 if side==0 else 1)*(factor-1)*(h if side==3 else w)
                add(r,b,r.class_name,kind)
        for cls,confusers in CONFUSERS.items():
            if r.class_name in confusers and (cls!='person' or (w>=60 and h>=60)):
                for _ in range(5):add(r,original+rng.uniform(-.08,.08,4)*[w,h,w,h],cls,'confuser',True)
    rows=pd.DataFrame(records)
    # Deterministic equal-class-label sampling within each day, using synthetic labels only.
    balanced=[]
    for _,g in rows.groupby(['day','class_name']):
        n=int(g.error.value_counts().min());assert g.error.nunique()==2
        for label in [0,1]:balanced.append(g[g.error==label].sample(n=n,random_state=seed))
    return rows,pd.concat(balanced).reset_index(drop=True),frames


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,default=42);parser.add_argument('--generate-only',action='store_true');parser.add_argument('--batch-size',type=int,default=16);args=parser.parse_args()
    data=ROOT/'data/rascar-hack-case-1';policy=json.loads((data/'scoring_policy.json').read_text())
    out=ROOT/f'outputs/synthetic_boxes/seed{args.seed}';out.mkdir(parents=True,exist_ok=True)
    raw,rows,frames=generate(data,policy,args.seed);raw.to_csv(out/'generated.csv',index=False);rows.to_csv(out/'training.csv',index=False)
    (out/'metadata.json').write_text(json.dumps({'encoder':MODEL,'revision':REVISION,'seed':args.seed,'batch_size':args.batch_size,
        'balanced_rows':len(rows),'source_frames':int(rows.frame_id.nunique()),'balance':'downsample majority independently per day/class',
        'source_sha256':{name:hashlib.sha256((data/name).read_bytes()).hexdigest() for name in ['human_boxes_train.csv','train_frames.csv','scoring_policy.json']}},indent=2))
    print('Generated',len(raw),'balanced training',len(rows),flush=True);print(rows.groupby(['day','class_name']).error.agg(['size','mean']).to_string(),flush=True)
    images={}
    def get_crop(r):
        if r.frame_id not in images:
            with Image.open(data/frames.loc[r.frame_id,'image_path']) as im:images[r.frame_id]=im.convert('RGB')
        return crop_image(images[r.frame_id],[r.x1,r.y1,r.x2,r.y2],r.image_width,r.image_height)
    for cls,g in rows.groupby('class_name'):
        canvas=Image.new('RGB',(224*5,250*4),'white');draw=ImageDraw.Draw(canvas)
        for i,r in enumerate(g.sample(n=min(20,len(g)),random_state=args.seed).itertuples()):
            x,y=i%5*224,i//5*250;canvas.paste(get_crop(r),(x,y+26));draw.text((x+2,y+2),f'{r.kind} IoU={r.iou:.4f} error={r.error}',fill='black')
        canvas.save(out/f'{cls}_examples.jpg')
    if args.generate_only:return
    import torch
    from transformers import CLIPVisionModel,AutoImageProcessor
    torch.set_num_threads(4);device='cuda' if torch.cuda.is_available() else 'cpu'
    kwargs=dict(revision=REVISION,cache_dir=ROOT/'weights/hf',local_files_only=True)
    encoder=CLIPVisionModel.from_pretrained(MODEL,**kwargs).eval().to(device);encoder.requires_grad_(False)
    processor=AutoImageProcessor.from_pretrained(MODEL,**kwargs);chunks=[];batch=[]
    for i,r in enumerate(rows.itertuples()):
        a=np.asarray(get_crop(r),dtype=np.float32).transpose(2,0,1)/255
        a=(a-np.array(processor.image_mean,dtype=np.float32)[:,None,None])/np.array(processor.image_std,dtype=np.float32)[:,None,None];batch.append(torch.from_numpy(a))
        if len(batch)==args.batch_size or i==len(rows)-1:
            with torch.inference_mode():chunks.append(encoder(pixel_values=torch.stack(batch).to(device)).pooler_output.cpu().numpy())
            batch=[]
        if (i+1)%500==0:print('Encoded',i+1,'/',len(rows),flush=True)
    emb=np.concatenate(chunks);assert np.isfinite(emb).all();np.savez_compressed(out/'features.npz',features=emb)
    metadata=json.loads((ROOT/'outputs/marked_context_clip/encoder.json').read_text());assert metadata['model']==MODEL and metadata['revision']==REVISION
    real=features(data,'train',policy);cache=np.load(ROOT/'outputs/marked_context_clip/features.npz');assert np.array_equal(cache['ids'],real.id)
    x=np.column_stack([emb,rows.class_name.map(CLASSES.index)]);xr=np.column_stack([cache['features'],real.class_name.map(CLASSES.index)])
    def estimator():return make_pipeline(ColumnTransformer([('visual',make_pipeline(StandardScaler(),PCA(n_components=64,svd_solver='randomized',random_state=42)),list(range(1024))),('class',OneHotEncoder(handle_unknown='ignore',sparse_output=False),[1024])]),LogisticRegression(C=1,max_iter=2000,random_state=42))
    scores=np.full(len(real),np.nan);blend=np.full(len(real),np.nan)
    for day in sorted(real.day.unique()):
        fit=rows.day!=day;valid=real.day==day;assert set(rows.loc[fit,'frame_id']).isdisjoint(real.loc[valid,'frame_id'])
        model=estimator().fit(x[fit],rows.loc[fit,'error']);assert model[-1].n_iter_.max()<2000
        scores[valid]=model.predict_proba(xr[valid])[:,1]
        blend[valid]=.5*pd.Series(scores[valid]).rank(pct=True).to_numpy()+.5*(1-real.loc[valid,'confidence']).rank(pct=True).to_numpy()
    records=[]
    for name,s in [('synthetic_visual',scores),('synthetic_rankblend',blend)]:
        real[['id','day']].assign(score=s).to_csv(ROOT/f'oof/{name}_seed{args.seed}.csv',index=False)
        for day,g in real.groupby('day'):records.append({'method':name,'day':day,'AP':average_precision_score(g.error,s[g.index])})
    m=pd.DataFrame(records);m.to_csv(out/'metrics.csv',index=False);print(m.to_string(index=False),flush=True)
    auc=[]
    for day,g in real.groupby('day'):
        for cls,h in g.groupby('class_name'):
            if h.error.nunique()==2:auc.append({'day':day,'class':cls,'n':len(h),'visual_auc':roc_auc_score(h.error,scores[h.index]),'baseline_auc':roc_auc_score(h.error,1-h.confidence)})
    pd.DataFrame(auc).to_csv(out/'auc.csv',index=False);print(pd.DataFrame(auc).round(4).to_string(index=False),flush=True)
    base=[average_precision_score(g.error,1-g.confidence) for _,g in real.groupby('day')];v=m[m.method=='synthetic_rankblend'].AP.to_numpy()
    success=bool(v.mean()>=.843 and (v>=base).all());print('SUCCESS_CRITERION',success,flush=True)
    if success:
        model=estimator().fit(x,rows.error);joblib.dump({'estimator':model,'encoder':MODEL,'revision':REVISION,'seed':args.seed},out/'model.joblib')
        test=features(data,'test',policy);z=np.load(ROOT/'outputs/marked_context_clip/test_features.npz');assert np.array_equal(z['ids'],test.id)
        p=model.predict_proba(np.column_stack([z['features'],test.class_name.map(CLASSES.index)]))[:,1];s=np.zeros(len(test))
        for day,g in test.groupby('day'):s[g.index]=.5*pd.Series(p[g.index]).rank(pct=True).to_numpy()+.5*(1-g.confidence).rank(pct=True).to_numpy()
        write_submission(data,test,s,ROOT/f'submissions/synthetic_rankblend_seed{args.seed}.csv')


if __name__=='__main__':main()
