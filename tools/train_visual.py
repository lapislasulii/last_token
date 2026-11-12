"""Frozen public DINOv2 crop features and day-held-out logistic heads.

Run: python tools/train_visual.py. No test labels or manual boxes are used.
"""
import json
import hashlib
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score

from train_logreg import ROOT, features
from train_temporal import write_submission

MODEL = 'facebook/dinov2-small'
REVISION = 'ed25f3a31f01632728cabb09d1542f84ab7b0056'
# Fixed before validation; each also tested with an equal baseline blend.
CONFIGS = [(view, c) for view in ('box', 'context', 'both') for c in (.01, .1)]


def extract(data, rows, split, encoder, device):
    import torch
    paths = pd.read_csv(data / f'{split}_frames.csv').set_index('frame_id').image_path
    result = {}
    for view, expansion in [('box', 1.), ('context', 1.5)]:
        tensors, chunks = [], []
        for row in rows.itertuples():
            with Image.open(data / paths[row.frame_id]) as source:
                im = source.convert('RGB')
            sx, sy = im.width / row.image_width, im.height / row.image_height
            cx, cy = (row.x1 + row.x2) * sx / 2, (row.y1 + row.y2) * sy / 2
            w, h = (row.x2-row.x1)*sx*expansion, (row.y2-row.y1)*sy*expansion
            bounds = (max(0, int(np.floor(cx-w/2))), max(0, int(np.floor(cy-h/2))),
                      min(im.width, int(np.ceil(cx+w/2))), min(im.height, int(np.ceil(cy+h/2))))
            assert bounds[2] > bounds[0] and bounds[3] > bounds[1]
            crop = ImageOps.pad(im.crop(bounds), (224, 224), method=Image.Resampling.BICUBIC,
                                color=(124, 116, 104))
            a = np.asarray(crop, dtype=np.float32).transpose(2, 0, 1) / 255
            a = (a-np.array([.485,.456,.406], dtype=np.float32)[:,None,None]) / np.array([.229,.224,.225], dtype=np.float32)[:,None,None]
            tensors.append(torch.from_numpy(a))
            if len(tensors) == 32:
                with torch.inference_mode():
                    chunks.append(encoder(pixel_values=torch.stack(tensors).to(device)).last_hidden_state[:,0].cpu().numpy())
                tensors = []
        if tensors:
            with torch.inference_mode():
                chunks.append(encoder(pixel_values=torch.stack(tensors).to(device)).last_hidden_state[:,0].cpu().numpy())
        result[view] = np.concatenate(chunks)
        assert result[view].shape == (len(rows), 384) and np.isfinite(result[view]).all()
        print(f'Extracted {split} {view}: {result[view].shape}', flush=True)
    return result


def main():
    for folder in ('models', 'oof', 'submissions'):
        (ROOT/folder).mkdir(exist_ok=True)
    data = ROOT / 'data/rascar-hack-case-1'
    policy = json.loads((data/'scoring_policy.json').read_text())
    train, test = [features(data, s, policy) for s in ('train','test')]
    directory = ROOT/'outputs/visual_features'
    directory.mkdir(parents=True, exist_ok=True)
    cache = directory/'dinov2_small.npz'
    # Cache invalidates if inputs change; all images hashed, no labels enter encoder.
    digest = hashlib.sha256()
    for split in ('train','test'):
        for name in (f'{split}.csv', f'{split}_frames.csv'):
            digest.update((data/name).read_bytes())
        paths = pd.read_csv(data/f'{split}_frames.csv').set_index('frame_id').image_path
        for frame in sorted((train if split=='train' else test).frame_id.unique()):
            digest.update((data/paths[frame]).read_bytes())
    fingerprint = digest.hexdigest()
    if cache.exists():
        metadata = json.loads((directory/'metadata.json').read_text())
        assert metadata['revision'] == REVISION and metadata['model'] == MODEL
        z = np.load(cache)
        assert str(z['fingerprint']) == fingerprint, 'Changed data: remove feature cache explicitly'
        assert np.array_equal(z['train_ids'], train.id) and np.array_equal(z['test_ids'], test.id)
        arrays = {s:{v:z[f'{s}_{v}'] for v in ('box','context')} for s in ('train','test')}
    else:
        import torch
        from transformers import AutoModel
        torch.set_num_threads(4)
        revision = REVISION
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        encoder = AutoModel.from_pretrained(MODEL, revision=revision, cache_dir=ROOT/'weights/hf').eval().to(device)
        encoder.requires_grad_(False)
        print(f'Frozen {MODEL} {revision} on {device}', flush=True)
        arrays = {s:extract(data,r,s,encoder,device) for s,r in [('train',train),('test',test)]}
        np.savez_compressed(cache, fingerprint=fingerprint, train_ids=train.id.to_numpy(dtype=str),
                            test_ids=test.id.to_numpy(dtype=str), **{f'{s}_{v}':a for s,d in arrays.items() for v,a in d.items()})
        (directory/'metadata.json').write_text(json.dumps({'model':MODEL,'revision':revision,
            'url':f'https://huggingface.co/{MODEL}/tree/{revision}', 'fingerprint':fingerprint,
            'preprocessing':'clipped crop, box or 1.5x context; aspect preserving bicubic pad 224, RGB fill124/116/104; ImageNet normalization; CLS384',
            'torch':torch.__version__}, indent=2))
    records = []
    days = sorted(train.day.unique())
    assert len(days)==2 and set(train.id).isdisjoint(test.id)
    baseline = 1-train.confidence.to_numpy()
    scores = {'baseline':baseline}
    for view,c in CONFIGS:
        name = f'visual_{view}_c{c:g}'
        x, xt = [np.concatenate([a['box'],a['context']],axis=1) if view=='both' else a[view] for a in (arrays['train'],arrays['test'])]
        # Confidence is present in each head; scaling is learned inside each fold.
        x = np.column_stack([x,train.confidence])
        xt = np.column_stack([xt,test.confidence])
        oof = np.full(len(train), np.nan)
        for day in days:
            fit,valid = (train.day!=day).to_numpy(), (train.day==day).to_numpy()
            estimator = make_pipeline(StandardScaler(),LogisticRegression(C=c,max_iter=3000,random_state=42))
            estimator.fit(x[fit],train.error.to_numpy()[fit])
            assert estimator[0].n_samples_seen_==fit.sum()
            assert estimator[-1].n_iter_.max()<3000
            oof[valid] = estimator.predict_proba(x[valid])[:,1]
        scores[name],scores[name+'_blend'] = oof,.5*oof+.5*baseline
        estimator.fit(x,train.error)
        bundle = {'estimator':estimator,'view':view,'encoder':json.loads((directory/'metadata.json').read_text()),'confidence':True}
        joblib.dump(bundle,ROOT/f'models/{name}.joblib')
        prediction = estimator.predict_proba(xt)[:,1]
        np.testing.assert_allclose(joblib.load(ROOT/f'models/{name}.joblib')['estimator'].predict_proba(xt)[:,1],prediction)
        for suffix, values in [('',prediction),('_blend',.5*prediction+.5*(1-test.confidence.to_numpy()))]:
            write_submission(data,test,values,ROOT/f'submissions/{name}{suffix}.csv')
    for name,values in scores.items():
        assert np.isfinite(values).all()
        train[['id','day']].assign(score=values).to_csv(ROOT/f'oof/{name}.csv',index=False)
        ap = {d:average_precision_score(train.loc[train.day==d,'error'],values[train.day==d]) for d in days}
        records.append({'method':name,**ap,'mean':np.mean(list(ap.values()))})
    metrics = pd.DataFrame(records).set_index('method')
    metrics.to_csv(directory/'metrics.csv')
    print(metrics.round(6).to_string(),flush=True)


if __name__=='__main__':
    main()
