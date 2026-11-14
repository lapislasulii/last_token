"""Audit RF-DETR voting on people using saved day-held-out and test predictions."""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score as ap

import person_vote as pv

R = pv.R
MEMBERS = ["base", "s1", "s2", "base_flip", "s1_flip", "s2_flip"]


def evaluate(split):
    boxes = R.load(f"{split}.csv").query("class_name != 'table'").copy()
    boxes["day"] = boxes.frame_id.map(R.frame_day(split))

    def features(member):
        if split == "train":
            return R.oof_features(boxes, member)
        predictions = pd.read_csv(R.pred_path("all", member))
        return boxes.merge(R.agreement_features(boxes, predictions), on="id", validate="1:1")

    g = pv.add_signals(boxes, split, [features(member) for member in MEMBERS])
    g["rf"] = (~features("rfdetr1120").set_index("id").matched.reindex(g.id)).astype(int).to_numpy()
    g["yolo"] = (g.D >= 0.5).astype(int)
    person = g.class_name == "person"
    disagreements = g[g.yolo != g.rf].copy()
    disagreements["group"] = np.where(disagreements.class_name == "person", "person", "other")
    keys = ["group", "yolo", "rf"]
    if split == "train":
        keys.insert(0, "day")
    cells = disagreements.groupby(keys).agg(boxes=("id", "size"), frames=("frame_id", "nunique"))
    if split == "train":
        cells["errors"] = disagreements.groupby(keys).error.sum()
    print(split, cells)
    g["base"] = pv.scores(g, True, True)[1]
    gate = person & ((g.x1 + g.x2) / 2 >= pv.WINDOW_X)
    d = (g.D + g.rf) / 2
    g["family"] = (1 - g.confidence + np.where(gate, (6*d + g.P)/7, d) + 2*np.maximum(g.veto,g.L))/4
    g["candidate"] = np.where(person, g.family, g.base)
    g["pose_family"] = np.where(gate, (1-g.confidence+(g.D+g.P)/2+2*np.maximum(g.veto,g.L))/4,g.base)
    return g


def gain(group, col):
    return ap(group.error, group[col]) - ap(group.error, group.base)


if __name__ == "__main__":
    R.WORK = R.ROOT / "outputs/redetect_strong"
    train = evaluate("train")
    test = evaluate("test")
    for day, g in train.groupby("day"):
        print(day, {c:round(ap(g.error,g[c]),6) for c in ["base","family","candidate","pose_family"]})
        p = g[g.class_name=="person"]
        print("person AP", ap(p.error,p.base), ap(p.error,p.candidate))
        boot = pv.frame_bootstrap(g,g.base,g.candidate)
        print("bootstrap", np.mean(boot>0))
    for col in ["candidate","pose_family"]:
        loo = [np.mean([gain(g,col) for _,g in train[train.frame_id!=fid].groupby("day")]) for fid in train.frame_id.unique()]
        print(col,"worst LOO", min(loo),"test changed",sum(abs(test[col]-test.base)>1e-12))

