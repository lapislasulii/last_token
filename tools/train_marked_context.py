"""Controlled OOF experiment: mark the tested box in the 1.5x crop.

Red is a constant marker, never a label. Also fits final lambda=1 head and test CSV.
Does not upload to Kaggle.
"""
import json
import argparse
import joblib
import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageDraw
from sklearn.metrics import average_precision_score
from train_logreg import ROOT, features
from train_visual import MODEL, REVISION
from train_visual_residual import fit, predict
from train_temporal import write_submission


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--encoder", choices=["small","base","large","clip"], default="small")
    parser.add_argument("--batch-size", type=int, default=8)
    args=parser.parse_args()
    encoder_name="openai/clip-vit-large-patch14" if args.encoder=="clip" else f"facebook/dinov2-{args.encoder}"
    revision={"small":REVISION,"base":"f9e44c814b77203eaa57a6bdbbd535f21ede1415","large":"47b73eefe95e8d44ec3623f8890bd894b6ea2d6c","clip":"32bd64288804d66eefd0ccbe215aa642df71cc41"}[args.encoder]
    prefix="marked_context" if args.encoder=="small" else f"marked_context_{args.encoder}"
    import torch
    from transformers import AutoModel, CLIPVisionModel, AutoImageProcessor
    data=ROOT/'data/rascar-hack-case-1'
    rows=features(data,'train',json.loads((data/'scoring_policy.json').read_text()))
    test=features(data,'test',json.loads((data/'scoring_policy.json').read_text()))
    assert set(rows.id).isdisjoint(test.id)
    out=ROOT/'outputs'/prefix;out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model_class=CLIPVisionModel if args.encoder=="clip" else AutoModel
    processor=AutoImageProcessor.from_pretrained(encoder_name,revision=revision,cache_dir=ROOT/"weights/hf") if args.encoder=="clip" else None
    mean=processor.image_mean if processor else [.485,.456,.406]
    std=processor.image_std if processor else [.229,.224,.225]
    encoder=model_class.from_pretrained(encoder_name,revision=revision,cache_dir=ROOT/'weights/hf').eval().to(device)
    encoder.requires_grad_(False)
    (out/"encoder.json").write_text(json.dumps({"model":encoder_name,"revision":revision,"batch_size":args.batch_size,"mean":mean,"std":std,"pooling":"post-layernorm CLS" if args.encoder=="clip" else "CLS"},indent=2))
    print(f"Loaded {encoder_name} {revision}",flush=True)
    paths=pd.concat([pd.read_csv(data/f'{s}_frames.csv') for s in ('train','test')]).drop_duplicates('frame_id').set_index('frame_id').image_path
    def encode(batch):
        result=encoder(pixel_values=batch)
        return result.pooler_output if args.encoder=="clip" else result.last_hidden_state[:,0]
    tensors=[];chunks=[];previews=[]
    for row in pd.concat([rows,test]).itertuples():
        with Image.open(data/paths[row.frame_id]) as source: im=source.convert('RGB')
        sx,sy=im.width/row.image_width,im.height/row.image_height
        x1,y1,x2,y2=row.x1*sx,row.y1*sy,row.x2*sx,row.y2*sy
        w,h=x2-x1,y2-y1
        bounds=(max(0,int(np.floor(x1-w*.25))),max(0,int(np.floor(y1-h*.25))),min(im.width,int(np.ceil(x2+w*.25))),min(im.height,int(np.ceil(y2+h*.25))))
        raw=im.crop(bounds)
        crop=ImageOps.pad(raw,(224,224),method=Image.Resampling.BICUBIC,color=(124,116,104))
        # Map to the actual rounded size used by ImageOps.contain/pad.
        fitted=ImageOps.contain(raw,(224,224),method=Image.Resampling.BICUBIC)
        ox,oy=round((224-fitted.width)*.5),round((224-fitted.height)*.5)
        ax,ay=fitted.width/raw.width,fitted.height/raw.height
        if len(previews)<6: before=crop.copy()
        ImageDraw.Draw(crop).rectangle((ox+(x1-bounds[0])*ax,oy+(y1-bounds[1])*ay,ox+(x2-bounds[0])*ax,oy+(y2-bounds[1])*ay),outline=(255,0,0),width=2)
        if len(previews)<6: previews.append((before,crop.copy()))
        a=np.asarray(crop,dtype=np.float32).transpose(2,0,1)/255
        a=(a-np.array(mean,dtype=np.float32)[:,None,None])/np.array(std,dtype=np.float32)[:,None,None]
        tensors.append(torch.from_numpy(a))
        if len(tensors)==args.batch_size:
            with torch.inference_mode(): chunks.append(encode(torch.stack(tensors).to(device)).cpu().numpy())
            tensors=[]
    if tensors:
        with torch.inference_mode(): chunks.append(encode(torch.stack(tensors).to(device)).cpu().numpy())
    all_x=np.concatenate(chunks);assert all_x.shape==(len(rows)+len(test),encoder.config.hidden_size) and np.isfinite(all_x).all()
    x,xt=all_x[:len(rows)],all_x[len(rows):]
    canvas=Image.new('RGB',(448,224*len(previews)))
    for i,(a,b) in enumerate(previews):canvas.paste(a,(0,224*i));canvas.paste(b,(224,224*i))
    canvas.save(out/'before_after.jpg')
    np.savez_compressed(out/'features.npz',ids=rows.id.to_numpy(dtype=str),features=x)
    np.savez_compressed(out/'test_features.npz',ids=test.id.to_numpy(dtype=str),features=xt)
    records=[]
    for strength in (1.,10.):
        score=np.full(len(rows),np.nan)
        for day in sorted(rows.day.unique()):
            mask=(rows.day!=day).to_numpy()
            model=fit(x[mask],rows.confidence.to_numpy()[mask],rows.error.to_numpy()[mask],strength)
            score[~mask]=predict(model,x[~mask],rows.confidence.to_numpy()[~mask])
        name=f'{prefix}_l{strength:g}'
        rows[['id','day']].assign(score=score).to_csv(ROOT/f'oof/{name}.csv',index=False)
        for day,g in rows.groupby('day'):
            records.append({'method':name,'day':day,'ap':average_precision_score(g.error,score[g.index])})
    metrics=pd.DataFrame(records);metrics.to_csv(out/'metrics.csv',index=False);print(metrics.to_string(index=False),flush=True)
    model=fit(x,rows.confidence.to_numpy(),rows.error.to_numpy(),1.)
    model.update(view='marked_context',encoder=encoder_name,revision=revision,preprocessing='1.5x crop, RGB124/116/104 pad224, red rectangle 2px',normalization_mean=mean,normalization_std=std)
    target=ROOT/f'models/{prefix}_l1.joblib';joblib.dump(model,target)
    score=predict(joblib.load(target),xt,test.confidence.to_numpy())
    np.testing.assert_allclose(score,predict(model,xt,test.confidence.to_numpy()))
    write_submission(data,test,score,ROOT/f'submissions/{prefix}_l1.csv')


if __name__=='__main__':main()
