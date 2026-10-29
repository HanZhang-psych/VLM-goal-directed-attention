"""Run-ID registry for the paper's analysis cells.

Every reproduced number traces back to a set of cluster run IDs. They are enumerated here
once (RUNS), so a notebook names a cell — (model, condition, arm, readout) — not a bare
number. Each cell records the POOLED set as {"val": [...], "scaleup": [...]}:

  - val     — the 326-image validation split (where the frozen readouts were confirmed),
  - scaleup — the train-minus-smoke split (~2090 TA / ~2150 TP; disjoint from val AND smoke).

They are disjoint image sets, so pooling = concatenation -> the paper's definitive "pooled
trainval-smoke" dataset (n ~= 2413 per TA cell, 2474 per TP cell). A few cells are sharded
(a job hit the walltime and finished as extra shards, or the val set was run in chunks);
every shard is listed and pooling dedups naturally by (task,name).

The cells = models × 2 conditions {tp, ta} × the OUTPUT arm × 2 prompts {search, free}.
The revision's models are Qwen3-VL-32B-Instruct and Gemma-4-26B-A4B-it; the original
Qwen3-VL-32B-Thinking cells stay registered for provenance.

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

MODEL_LABEL = {"qwen-thinking": "Qwen3-VL-32B-Thinking", "qwen-instruct": "Qwen3-VL-32B-Instruct",
               "gemma": "Gemma-4-26B-A4B-it"}
MODELS = ["qwen-instruct", "gemma"]
CONDITIONS = ["tp", "ta"]

# (model, condition, arm, readout) -> {"val": [...], "scaleup": [...]}
RUNS = {
    # ============================ Qwen3-VL-32B-Thinking ============================
    # ---- OUTPUT arm (prompts: search / free-view) ----
    ("qwen-thinking", "tp", "output", "search"): {"val": ["58031647"], "scaleup": ["58094437"]},
    ("qwen-thinking", "tp", "output", "free"):   {"val": ["58094796"], "scaleup": ["58094446"]},
    ("qwen-thinking", "ta", "output", "search"): {"val": ["58031597"], "scaleup": ["58094212"]},
    ("qwen-thinking", "ta", "output", "free"):   {"val": ["58031679"], "scaleup": ["58094237"]},

    # ============================ Qwen3-VL-32B-Instruct ============================
    # The revision's Qwen model: same vLLM protocol, prompts, sampler and image sets as the
    # Thinking cells (QWEN_VLLM_ID override), run 2026-10-13/14. The causal (ablation) arm is run
    # on this model, so the alignment and the ablation results share one model and one sampler.
    # ---- OUTPUT arm (prompts: search / free-view) ----
    ("qwen-instruct", "tp", "output", "search"): {"val": ["63889206"], "scaleup": ["63889207"]},
    ("qwen-instruct", "tp", "output", "free"):   {"val": ["63889208"], "scaleup": ["63889209"]},
    ("qwen-instruct", "ta", "output", "search"): {"val": ["63889202"], "scaleup": ["63889203"]},
    ("qwen-instruct", "ta", "output", "free"):   {"val": ["63889204"], "scaleup": ["63889205"]},

    # ============================ Gemma-4-26B-A4B-it (Instruct) ============================
    # ---- OUTPUT arm (prompts: search / free-view) ----
    #   TP search val is stitched: 57202064 (321/326, hit walltime) + 57260139 (final 5 images).
    #   TA free val is sharded: five HF chunks covering the 326 val images (66+66+66+66+62).
    ("gemma", "tp", "output", "search"): {"val": ["57202064", "57260139"], "scaleup": ["58387155"]},
    ("gemma", "tp", "output", "free"):   {"val": ["58411643"], "scaleup": ["58387156"]},
    ("gemma", "ta", "output", "search"): {"val": ["57202063"], "scaleup": ["58387153"]},
    ("gemma", "ta", "output", "free"):   {"val": ["57707609", "57707610", "57707611",
                                                  "57707612", "57707613"], "scaleup": ["58387154"]},
}
