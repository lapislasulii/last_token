"""Render a fixed sample of held-out train frames, human vs second-detector people.

python tools/review_redetect.py --tag strong
Images stay under outputs/. This never loads test annotations.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--per-day', type=int, default=7)
    parser.add_argument('--display-confidence', type=float, default=0.25,
                        help='Display only; metric uses its original 0.05 threshold.')
    args = parser.parse_args()
    data = ROOT / 'data/rascar-hack-case-1'
    work = ROOT / f'outputs/redetect_{args.tag}'
    destination = work / 'review'
    destination.mkdir(parents=True, exist_ok=True)
    human = pd.read_csv(data / 'human_boxes_train.csv')
    frames = pd.read_csv(data / 'train_frames.csv', dtype={'day': str}).set_index('frame_id')
    manifest = []
    pictures = []
    for heldout, trained in [('20261025', '20261026'), ('20261026', '20261025')]:
        predictions = pd.read_csv(work / f'pred_day_{trained}.csv')
        candidates = sorted(f for f in human.loc[human.class_name == 'person', 'frame_id'].unique()
                            if frames.loc[f, 'day'] == heldout)
        indices = np.unique(np.linspace(0, len(candidates) - 1, min(args.per_day, len(candidates))).astype(int))
        for index in indices:
            fid = candidates[index]
            original = Image.open(data / frames.loc[fid, 'image_path']).convert('RGB')
            width, height = original.size
            panel = Image.new('RGB', (width * 2, height + 48), 'white')
            panel.paste(original, (0, 48))
            panel.paste(original, (width, 48))
            draw = ImageDraw.Draw(panel)
            draw.text((8, 4), f'{fid} | HELD OUT {heldout}, trained {trained}', fill='black')
            draw.text((8, 24), 'Human person boxes (blue)', fill='blue')
            draw.text((width + 8, 24), f'New detector (orange), display confidence >= {args.display_confidence}', fill='black')
            for row in human[(human.frame_id == fid) & (human.class_name == 'person')].itertuples():
                box = ((row.cx-row.w/2)*width, (row.cy-row.h/2)*height+48,
                       (row.cx+row.w/2)*width, (row.cy+row.h/2)*height+48)
                draw.rectangle(box, outline='blue', width=3)
            shown = predictions[(predictions.frame_id == fid) & (predictions.class_name == 'person') &
                                (predictions.confidence >= args.display_confidence)]
            for row in shown.itertuples():
                box = (row.x1+width, row.y1+48, row.x2+width, row.y2+48)
                draw.rectangle(box, outline='#ff8800', width=3)
                draw.text((box[0]+3, box[1]+3), f'{row.confidence:.2f}', fill='#ff8800')
            path = destination / f'{fid}.jpg'
            panel.save(path, quality=92)
            pictures.append(panel.resize((1280, 384)))
            manifest.append(dict(frame_id=fid, heldout_day=heldout, trained_day=trained,
                                 display_confidence=args.display_confidence, image=str(path)))
    pd.DataFrame(manifest).to_csv(destination / 'manifest.csv', index=False)
    for start in range(0, len(pictures), 4):
        batch = pictures[start:start+4]
        page = Image.new('RGB', (1280, 384*len(batch)), 'white')
        for offset, picture in enumerate(batch):
            page.paste(picture, (0, offset*384))
        page.save(destination / f'contact_{start//4+1}.jpg', quality=92)
    print(f'Rendered {len(manifest)} held-out frames to {destination}')


if __name__ == '__main__':
    main()
