"""Three fixed CLIP views: unmarked context, box+context, inside/outside patches."""
import json
import numpy as np
import pandas as pd
import joblib
from PIL import Image,ImageOps
from sklearn.metrics import average_precision_score
from train_logreg import ROOT,features
from train_visual_residual import fit,predict
from train_temporal import write_submission

MODEL='openai/clip-vit-large-patch14'
REVISION='32bd64288804d66eefd0ccbe215aa642df71cc41'


def main():
    import torch
    from transformers import CLIPVisionModel,AutoImageProcessor
    data=ROOT/'data/rascar-hack-case-1';policy=json.loads((data/'scoring_policy.json').read_text())
    train,test=[features(data,s,policy) for s in ('train','test')]
    rows=pd.concat([train,test],ignore_index=True)
    out=ROOT/'outputs/clip_spatial';out.mkdir(parents=True,exist_ok=True)
    paths=pd.concat([pd.read_csv(data/f'{s}_frames.csv') for s in ('train','test')]).drop_duplicates('frame_id').set_index('frame_id').image_path
    torch.set_num_threads(4);device='cuda' if torch.cuda.is_available() else 'cpu'
    kwargs=dict(revision=REVISION,cache_dir=ROOT/'weights/hf',local_files_only=True)
    model=CLIPVisionModel.from_pretrained(MODEL,**kwargs).eval().to(device);model.requires_grad_(False)
    processor=AutoImageProcessor.from_pretrained(MODEL,**kwargs)
    result={k:[] for k in ('context','box','inside','outside')};empty=0
    preview=Image.new('RGB',(448,224*4))
    for i,row in enumerate(rows.itertuples()):
        with Image.open(data/paths[row.frame_id]) as source: im=source.convert('RGB')
        sx,sy=im.width/row.image_width,im.height/row.image_height
        x1,y1,x2,y2=row.x1*sx,row.y1*sy,row.x2*sx,row.y2*sy;w,h=x2-x1,y2-y1
        imgs=[];masks=None
        for view,expand in [('context',.25),('box',0.)]:
            bounds=(max(0,int(np.floor(x1-w*expand))),max(0,int(np.floor(y1-h*expand))),min(im.width,int(np.ceil(x2+w*expand))),min(im.height,int(np.ceil(y2+h*expand))))
            raw=im.crop(bounds);fitted=ImageOps.contain(raw,(224,224),method=Image.Resampling.BICUBIC)
            crop=ImageOps.pad(raw,(224,224),method=Image.Resampling.BICUBIC,color=(124,116,104))
            if i<4:preview.paste(crop,(0 if view=='context' else 224,i*224))
            if view=='context':
                ox,oy=round((224-fitted.width)/2),round((224-fitted.height)/2)
                xx,yy=np.meshgrid(np.arange(16)*14+7,np.arange(16)*14+7)
                xx=(xx-ox)*raw.width/fitted.width+bounds[0];yy=(yy-oy)*raw.height/fitted.height+bounds[1]
                valid=(xx>=bounds[0])&(xx<bounds[2])&(yy>=bounds[1])&(yy<bounds[3])
                inside=valid&(xx>=x1)&(xx<x2)&(yy>=y1)&(yy<y2)
                masks=(inside.ravel(),(valid&~inside).ravel())
            a=np.asarray(crop,dtype=np.float32).transpose(2,0,1)/255
            a=(a-np.array(processor.image_mean,dtype=np.float32)[:,None,None])/np.array(processor.image_std,dtype=np.float32)[:,None,None]
            imgs.append(torch.from_numpy(a))
        with torch.inference_mode():
            prediction=model(pixel_values=torch.stack(imgs).to(device))
            cls=prediction.pooler_output.cpu().numpy();patch=prediction.last_hidden_state[0,1:].cpu().numpy()
        result['context'].append(cls[0]);result['box'].append(cls[1])
        for name,mask in zip(('inside','outside'),masks):
            if not mask.any():empty+=1
            result[name].append(patch[mask].mean(axis=0) if mask.any() else np.zeros(patch.shape[1],dtype=np.float32))
        if (i+1)%100==0:print(f'Extracted {i+1}/{len(rows)}',flush=True)
    result={k:np.asarray(v) for k,v in result.items()}
    assert all(np.isfinite(v).all() for v in result.values())
    preview.save(out/'views.jpg')
    np.savez_compressed(out/'features.npz',ids=rows.id.to_numpy(dtype=str),**result)
    (out/'metadata.json').write_text(json.dumps({'encoder':MODEL,'revision':REVISION,'empty_patch_groups':empty,'inside_outside':'Mean contextual patch tokens; padding excluded by patch centers. Not independent masked-image embeddings.'},indent=2))
    records=[]
    for name,x in [('clip_unmarked',result['context']),('clip_box_context',np.column_stack([result['box'],result['context']])),('clip_inside_outside',np.column_stack([result['inside'],result['outside']]))]:
        xt=x[len(train):];x=x[:len(train)];oof=np.full(len(train),np.nan)
        for day in sorted(train.day.unique()):
            mask=(train.day!=day).to_numpy();b=fit(x[mask],train.confidence.to_numpy()[mask],train.error.to_numpy()[mask],1.)
            oof[~mask]=predict(b,x[~mask],train.confidence.to_numpy()[~mask])
        train[['id','day']].assign(score=oof).to_csv(ROOT/f'oof/{name}.csv',index=False)
        for day,g in train.groupby('day'):records.append({'method':name,'day':day,'AP':average_precision_score(g.error,oof[g.index])})
        b=fit(x,train.confidence.to_numpy(),train.error.to_numpy(),1.);b.update(encoder=MODEL,revision=REVISION,view=name)
        joblib.dump(b,ROOT/f'models/{name}.joblib');s=predict(b,xt,test.confidence.to_numpy())
        np.testing.assert_allclose(s,predict(joblib.load(ROOT/f'models/{name}.joblib'),xt,test.confidence.to_numpy()))
        write_submission(data,test,s,ROOT/f'submissions/{name}.csv')
    m=pd.DataFrame(records);m.to_csv(out/'metrics.csv',index=False);print(m.to_string(index=False),flush=True)


if __name__=='__main__':main()
