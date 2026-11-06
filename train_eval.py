"""Train YOLO11n on one synthetic dataset, evaluate on every eval set, write results/<name>.json.
usage: python train_eval.py <name> <train_dir> [--aug default|off] [--seed 0] [--epochs 30]
Run from the repo root; datasets live in data/<set>/{images,labels}."""
import argparse
import json
import os
from pathlib import Path

from ultralytics import YOLO

from yolo_labels import CLASSES

ap = argparse.ArgumentParser()
ap.add_argument("name")
ap.add_argument("train_dir")
ap.add_argument("--aug", choices=["default", "off"], default="default")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--epochs", type=int, default=30)
ap.add_argument("--evals", nargs="+", default=["data/rand_val", "data/fixed_val", "data/shift_test"])
a = ap.parse_args()

ROOT = Path(__file__).resolve().parent


def yaml_for(train_dir, val_dir):
    p = ROOT / "data" / ("_%s__%s.yaml" % (Path(train_dir).name, Path(val_dir).name))
    p.write_text("train: %s/images\nval: %s/images\nnames:\n%s" % (
        (ROOT / train_dir), (ROOT / val_dir), "".join("  %d: %s\n" % kv for kv in enumerate(CLASSES))))
    return str(p)


# "off" removes colour jitter, mosaic, scale/translate jitter and flips so the training set alone differs.
off = dict(hsv_h=0, hsv_s=0, hsv_v=0, mosaic=0, scale=0, translate=0, fliplr=0, erasing=0, auto_augment=None)
model = YOLO("yolo11n.pt")
model.train(data=yaml_for(a.train_dir, a.evals[0]), epochs=a.epochs, imgsz=640, batch=32, workers=4,
            seed=a.seed, project=str(ROOT / "runs"), name=a.name, exist_ok=True, plots=False,
            val=False, **(off if a.aug == "off" else {}))
best = YOLO(str(ROOT / "runs" / a.name / "weights" / "last.pt"))
res = {"name": a.name, "train": a.train_dir, "aug": a.aug, "seed": a.seed, "epochs": a.epochs, "eval": {}}
for ev in a.evals:
    m = best.val(data=yaml_for(a.train_dir, ev), imgsz=640, batch=32, workers=4, plots=False, verbose=False,
                 project=str(ROOT / "runs"), name="val_%s_%s" % (a.name, Path(ev).name), exist_ok=True)
    res["eval"][Path(ev).name] = {"mAP50": round(float(m.box.map50), 4), "mAP50-95": round(float(m.box.map), 4)}
    print(ev, res["eval"][Path(ev).name])
os.makedirs(ROOT / "results", exist_ok=True)
json.dump(res, open(ROOT / "results" / (a.name + ".json"), "w"), indent=1)
