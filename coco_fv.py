"""
coco_fv.py — loads COCO-Search18 (search) and COCO-FreeView (free-view) fixation data.

For one image it reads the fixation JSON, crops the image letterbox and shifts the fixations
into the cropped frame, returning the cropped image and the pooled fixation arrays:
  - search : search fixations for that image's target category (TA or TP)
  - free   : free-view fixations on the same image
The density maps themselves are built offline in analysis/ from the saved points.

Data (see data/COCO-Search18/readme.txt):
  - TA-fixations/coco_search18_fixations_TA_trainval.json  (search, target-absent)
  - TP-fixations/coco_search18_fixations_TP_*.json         (search, target-present)
  - COCOFreeView_fixations_trainval.json                   (free-view)
  - {TA,TP}-images/<task>/<name>.jpg  — 1680x1050, zero-padded (aspect kept).
The image files are the display frame, so fixations are already in image pixels; the
loader crops the letterbox and shifts them into the cropped frame.

Each record carries X, Y, T lists (first entry = forced central fixation), 'correct',
'task', and 'name'. Pure CPU / numpy; uses gaze_metrics for the letterbox crop.
"""

import os
import json
import collections

import numpy as np
from PIL import Image

import gaze_metrics as ev


def load_records(path):
    """`path` is a json file, OR a list/tuple of json files whose records are concatenated
    (used for the full TP set, which ships split across train + validation files)."""
    if isinstance(path, (list, tuple)):
        recs = []
        for p in path:
            with open(p) as f:
                recs.extend(json.load(f))
        return recs
    with open(path) as f:
        return json.load(f)


def index_by_name(records, correct_only=True, task=None):
    """name -> list of trial records (optionally filtered to correct / one task)."""
    idx = collections.defaultdict(list)
    for r in records:
        if correct_only and r.get("correct", 1) != 1:
            continue
        if task is not None and r.get("task") != task:
            continue
        idx[r["name"]].append(r)
    return idx


def pool_fixations(records, drop_first=True):
    """Pool all trials' fixations into one N x 2 (x,y) array in image pixels.

    drop_first removes each trial's forced central start fixation (index 0).
    """
    xs, ys = [], []
    for r in records:
        s = 1 if drop_first else 0
        xs.extend(r["X"][s:])
        ys.extend(r["Y"][s:])
    if not xs:
        return np.zeros((0, 2), np.float32)
    return np.stack([np.asarray(xs, np.float32), np.asarray(ys, np.float32)], axis=1)


# Fixation-file selection. TA ships as one combined trainval file (2489 images). TP is split
# across train_split1 and validation_split1: condition "tp" loads the validation split (326
# images) by default, or both files (2489 images) when VLM_GAZE_TP_FULL is set. Free-view uses
# the same COCOFreeView JSON for both conditions.
_FIX_JSON = {"ta": ("TA-fixations", "coco_search18_fixations_TA_trainval.json"),
             "tp": ("TP-fixations", "coco_search18_fixations_TP_validation_split1.json")}
_FIX_JSON_TP_FULL = [("TP-fixations", "coco_search18_fixations_TP_train_split1.json"),
                     ("TP-fixations", "coco_search18_fixations_TP_validation_split1.json")]
_IMG_DIR = {"ta": "TA-images", "tp": "TP-images"}


def fixations_path(data_dir, condition="ta"):
    if condition == "tp" and os.environ.get("VLM_GAZE_TP_FULL"):
        return [os.path.join(data_dir, *p) for p in _FIX_JSON_TP_FULL]   # full TP trainval
    return os.path.join(data_dir, *_FIX_JSON[condition])


def image_path(data_dir, task, name, condition="ta"):
    return os.path.join(data_dir, _IMG_DIR[condition], task, name)


def load_image_and_fixations(name, task, ta_idx, fv_idx, data_dir, drop_first=True, condition="ta"):
    """Cropped image + pooled human fixations in the cropped frame, for one image (TA or TP).

    ta_idx : index_by_name(search records, task=task)
    fv_idx : index_by_name(free-view records)
    Returns a dict:
      image  : cropped PIL image
      size   : (iw, ih)
      fix    : {'search': Nx2, 'free': Nx2} pooled fixations, cropped to content
    """
    image = Image.open(image_path(data_dir, task, name, condition)).convert("RGB")
    fix = {"search": pool_fixations(ta_idx.get(name, []), drop_first),
           "free":   pool_fixations(fv_idx.get(name, []), drop_first)}
    image, fix, (iw, ih) = ev.crop_letterbox(image, fix)
    return {"image": image, "size": (iw, ih), "fix": fix, "task": task, "name": name}
