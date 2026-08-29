"""
config.py — shared constants for the VLM-gaze runners: the model ids, the output root,
the Slurm job id, the frozen prompts, and the frozen readout choices (output length cap,
per-model attention band, per-model memory layer).
"""
import os

# The two models. Qwen runs in Thinking mode (keeps a <think> trace); Gemma in instruct
# (non-thinking) mode. Both are used for both readout arms.
MODEL_IDS = {
    "qwen-thinking": "Qwen/Qwen3-VL-32B-Thinking",   # Qwen, Thinking mode
    "gemma": "google/gemma-4-26B-A4B-it",            # Gemma, instruct (non-thinking) mode
}
_ID_TO_KEY = {v: k for k, v in MODEL_IDS.items()}


def model_key(name):
    """Resolve a friendly key ('gemma') OR a full HF id to the friendly key, for config lookups."""
    return name if name in MODEL_IDS else _ID_TO_KEY.get(name, name)


# Default model used when --model is not passed.
DEFAULT_MODEL = MODEL_IDS["qwen-thinking"]

DEFAULT_OUTPUT_ROOT = (os.environ.get("VLM_GAZE_OUTPUT_ROOT")
                       or os.path.expanduser("~/vlm-gaze/output"))
SLURM_JOB_ID = os.environ.get("SLURM_JOB_ID")

# ---------------------------------------------------------------------------
# The search prompt, used for both readout arms. FROZEN_SYSTEM is the system prompt.
FROZEN_SYSTEM = "You are a helpful assistant."
FROZEN_PROMPT = (
    "You are searching for a {t} in this image. If a {t} is clearly visible, point to "
    "the {t}; otherwise, point out all locations in this image you would examine before "
    "you can decide whether a {t} is present. Rank the locations from most to least "
    "likely. Give one location per line in exactly this format:\n"
    "REGION: <a short noun phrase for the place> | POINT: (x, y)\n"
    "where x and y are integers from 0 to 1000, x left-to-right and y top-to-bottom.")

# ---------------------------------------------------------------------------
# Output length cap: scoring keeps the top-N ranked points per sampled scanpath. N=5 is the
# median human target-absent search scanpath length, applied to every model, condition, and
# arm. analysis/helpers/runs.py mirrors this value.
FROZEN_TOPN = 5

# ---------------------------------------------------------------------------
# The free-view / memory prompt. Its wording matches the COCO-FreeView memory-encoding instruction.
FREEVIEW_PROMPT = (
    "You are viewing this image to remember it for a later memory test. Point out the "
    "locations in this image you would naturally look at while studying it. Rank the "
    "locations from most to least likely to draw your attention. Give one location per line "
    "in exactly this format:\n"
    "REGION: <a short noun phrase for the place> | POINT: (x, y)\n"
    "where x and y are integers from 0 to 1000, x left-to-right and y top-to-bottom.")

# ---------------------------------------------------------------------------
# Search-arm attention readout. ATTN_BANDS maps a band name to a fraction-of-depth range.
# FROZEN_ATTN_BAND is the per-model band pooled for the search readout (Qwen mid, Gemma early).
ATTN_BANDS = {"early": (0.1, 0.4), "mid": (0.4, 0.7), "late": (0.5, 0.9), "verylate": (0.7, 1.0)}

FROZEN_ATTN_BAND = {"qwen-thinking": "mid", "gemma": "early"}
DEFAULT_ATTN_BAND = "mid"   # fallback for a model not in the table

# ---------------------------------------------------------------------------
# Memory-arm attention readout: a single frozen layer per model (Qwen L23, Gemma L28), read from
# the goal word "remember" in the free-view prompt. One layer rather than a band because the
# memory signal sits at a single depth, unlike the search readout's contiguous band.
FROZEN_MEM_LAYER = {"gemma": 28, "qwen-thinking": 23}
FROZEN_MEM_QUERY = "remember"
