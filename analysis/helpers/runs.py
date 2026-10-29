"""Run-ID registry for the paper's analysis cells.

Every reproduced number traces back to a set of cluster run IDs. They are enumerated here
once (RUNS), so a notebook names a cell — (model, condition, prompt) — not a bare number. Each cell records the POOLED set as {"val": [...], "scaleup": [...]}:

  - val     — the 326-image validation split,
  - scaleup — the train-minus-smoke split (~2090 TA / ~2150 TP; disjoint from val AND smoke).

They are disjoint image sets, so pooling = concatenation (n ~= 2413 per TA cell, 2474 per TP
cell). A few cells are sharded (a job hit the walltime and finished as extra shards, or the val
set was run in chunks); every shard is listed and pooling dedups naturally by (task,name).

The cells = 2 models × 2 conditions {tp, ta} × 2 prompts {search, free}.

Data files are read by `helpers.util`; this module intentionally only stores constants.

Per-run files:
  points_<run>.csv        mode,condition,task,name,iw,ih,seq,rank,x1000,y1000,x_px,y_px,region
  human_points_<run>.csv  task,name,iw,ih,kind,x_px,y_px        (kind in {search,free})
  answers_<run>.jsonl     raw model generations (provenance)
"""
import os
from pathlib import Path

DATA_DIR = Path(os.environ.get("VLM_GAZE_ANALYSIS_DATA",
                               Path(__file__).resolve().parent.parent / "data"))
# live fallback for files not bundled under data/ (same default as the repo config)
OUTPUT_ROOT = (os.environ.get("VLM_GAZE_OUTPUT_ROOT")
               or os.path.expanduser("~/vlm-gaze/output"))

FROZEN_TOPN = 5          # top-N ranked model points kept per sample (config.FROZEN_TOPN)

MODEL_LABEL = {"qwen-instruct": "Qwen3-VL-32B-Instruct", "gemma": "Gemma-4-26B-A4B-it"}
MODELS = ["qwen-instruct", "gemma"]
CONDITIONS = ["tp", "ta"]

# (model, condition, prompt) -> {"val": [...], "scaleup": [...]}
RUNS = {
    # ============================ Qwen3-VL-32B-Instruct ============================
    # ---- prompts: search / free-view ----
    ("qwen-instruct", "tp", "search"): {"val": ["63889206"], "scaleup": ["63889207"]},
    ("qwen-instruct", "tp", "free"):   {"val": ["63889208"], "scaleup": ["63889209"]},
    ("qwen-instruct", "ta", "search"): {"val": ["63889202"], "scaleup": ["63889203"]},
    ("qwen-instruct", "ta", "free"):   {"val": ["63889204"], "scaleup": ["63889205"]},

    # ============================ Gemma-4-26B-A4B-it ============================
    # ---- prompts: search / free-view ----
    #   TP search val is stitched: 57202064 (321/326, hit walltime) + 57260139 (final 5 images).
    #   TA free val is sharded: five HF chunks covering the 326 val images (66+66+66+66+62).
    ("gemma", "tp", "search"): {"val": ["57202064", "57260139"], "scaleup": ["58387155"]},
    ("gemma", "tp", "free"):   {"val": ["58411643"], "scaleup": ["58387156"]},
    ("gemma", "ta", "search"): {"val": ["57202063"], "scaleup": ["58387153"]},
    ("gemma", "ta", "free"):   {"val": ["57707609", "57707610", "57707611",
                                                  "57707612", "57707613"], "scaleup": ["58387154"]},
}
