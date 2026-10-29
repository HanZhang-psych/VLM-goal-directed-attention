"""Small notebook utilities shared across data analyses."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
import gaze_metrics as gm  # noqa: E402  (repo root: shared with inference/)

from . import runs as R  # noqa: E402


def get_run_ids(model, condition, goal):
    """Return val + scaleup run IDs for one registry cell (goal = the prompt: search / free)."""
    run_groups = R.RUNS.get((model, condition, goal), {"val": [], "scaleup": []})
    return list(run_groups["val"]) + list(run_groups["scaleup"])


def get_ablation_run_ids(model, condition, ablation):
    """Return val + scaleup run IDs for one ablation cell (ablation = goal / format)."""
    run_groups = R.ABLATIONS.get((model, condition, ablation), {"val": [], "scaleup": []})
    return list(run_groups["val"]) + list(run_groups["scaleup"])


def get_run_file(run_id, prefix):
    """Locate one bundled data file, falling back to the live output root."""
    extension = "jsonl" if prefix == "answers" else "csv"
    filename = f"{prefix}_{run_id}.{extension}"
    local_path = R.DATA_DIR / str(run_id) / filename
    live_path = Path(R.OUTPUT_ROOT) / str(run_id) / filename

    if local_path.exists():
        return local_path
    if live_path.exists():
        return live_path

    raise FileNotFoundError(
        f"{filename} not found under {R.DATA_DIR}/{run_id}/ or {R.OUTPUT_ROOT}/{run_id}/."
    )


def load_fixations(run_ids, mode):
    """Load VLM point predictions or human fixations across runs.

    mode:
      - "vlm" reads points_<run>.csv
      - "human" reads human_points_<run>.csv
    """
    file_prefix = {"vlm": "points", "human": "human_points"}[mode]
    dataframes = []

    for run in run_ids:
        dataframe = pd.read_csv(get_run_file(run, file_prefix))
        dataframe["run"] = run
        dataframes.append(dataframe)

    return pd.concat(dataframes, ignore_index=True)


def density_map(points, iw, ih, scale=4):
    """Compute a density map from model points with optional downscaling."""
    scaled_points = np.asarray(points, np.float32).reshape(-1, 2) / scale
    scaled_width = max(1, iw // scale)
    scaled_height = max(1, ih // scale)

    return gm.density_map(scaled_points, scaled_width, scaled_height)


def density_maps_from_points(data, mode, goal=None, scale=4):
    """Build one VLM density map per picture from top-ranked predicted points."""
    density_maps = {}
    groupby_key = ["task", "name"] # bottle, *.jpg

    if mode == "vlm":
        data = data[data["rank"] < R.FROZEN_TOPN].reset_index(drop=True)
    if mode == "human":
        if goal is None:
            raise ValueError("Goal must be specified for human fixations.")
        data = data[data["kind"] == goal].reset_index(drop=True)

    for image, image_points in data.groupby(groupby_key, sort=False):
        iw = int(image_points["iw"].iloc[0])
        ih = int(image_points["ih"].iloc[0])
        points = image_points[["x_px", "y_px"]].to_numpy(np.float32)
        density_maps[image] = density_map(points, iw, ih, scale=scale)

    return density_maps



def format_p_value(p_value):
    """Format a p-value compactly for plot annotations."""
    if p_value < 0.001:
        return "p < .001"
    return f"p = {p_value:.3f}"


