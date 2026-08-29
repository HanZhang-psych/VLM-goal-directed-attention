"""Small notebook utilities shared across data analyses."""
import collections
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from . import gaze_metrics as gm
from . import runs as R


def get_run_ids(model, condition, readout, goal):
    """Return val + scaleup run IDs for one registry cell."""
    run_groups = R.RUNS.get((model, condition, readout, goal), {"val": [], "scaleup": []})
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


def load_model_scanpaths(run_ids, max_rank=None):
    """Load ordered model scanpaths across points_<run>.csv files."""
    points = load_fixations(run_ids, mode="vlm")
    if max_rank is not None:
        points = points[points["rank"] < max_rank].reset_index(drop=True)

    return {
        image: [
            scanpath.sort_values("rank")[["x_px", "y_px"]].to_numpy(float)
            for _, scanpath in image_points.groupby(["run", "seq"], sort=False)
        ]
        for image, image_points in points.groupby(["task", "name"], sort=False)
    }


def load_human_scanpaths(coco_dir, image_keys, goals=("search", "free"), drop_first=True):
    """Load per-trial TA human scanpaths for fixation-order analyses."""
    coco_dir = Path(coco_dir)
    search_path = coco_dir / "TA-fixations" / "coco_search18_fixations_TA_trainval.json"
    free_path = coco_dir / "COCOFreeView_fixations_trainval.json"
    image_dir = coco_dir / "TA-images"

    def read_json_records(path):
        with open(path) as file:
            return json.load(file)

    search_index = collections.defaultdict(list)
    for record in read_json_records(search_path):
        if record.get("correct", 1) == 1:
            search_index[(record["task"], record["name"])].append(record)

    free_index = collections.defaultdict(list)
    for record in read_json_records(free_path):
        if record.get("correct", 1) == 1:
            free_index[record["name"]].append(record)

    scanpaths_by_goal = {goal: {} for goal in goals}
    for task, name in sorted(set(image_keys)):
        image = Image.open(image_dir / task / name).convert("RGB")
        left, top, right, bottom = gm.detect_letterbox(image)
        cropped_width = right - left
        cropped_height = bottom - top

        for goal in goals:
            records = search_index.get((task, name), []) if goal == "search" else free_index.get(name, [])
            scanpaths = []

            for record in records:
                x_values = np.asarray(record["X"], float)
                y_values = np.asarray(record["Y"], float)
                start = 1 if drop_first else 0
                points = np.stack([x_values[start:] - left, y_values[start:] - top], axis=1)
                in_bounds = (
                    (points[:, 0] >= 0)
                    & (points[:, 0] < cropped_width)
                    & (points[:, 1] >= 0)
                    & (points[:, 1] < cropped_height)
                )
                points = points[in_bounds]

                if len(points):
                    scanpaths.append(points)

            scanpaths_by_goal[goal][(task, name)] = (scanpaths, (cropped_width, cropped_height))

    return scanpaths_by_goal



def density_map(points, iw, ih, scale=4):
    """Compute an output-arm density map with optional downscaling."""
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


def get_npz_paths(model, condition, readout):
    """Map each picture key to its saved data/<run>/attn_maps_<run> grid file."""
    map_index = {}

    for run in get_run_ids(model, condition, "attention", readout):
        local_dir = R.DATA_DIR / str(run) / f"attn_maps_{run}"
        live_dir = Path(R.OUTPUT_ROOT) / str(run) / f"attn_maps_{run}"
        map_dir = local_dir if local_dir.exists() else live_dir

        if not map_dir.exists():
            continue

        for map_path in sorted(map_dir.glob("*.npz")):
            target, name = map_path.name.removesuffix(".npz").rsplit("_", 1)
            map_index[(target, name)] = map_path

    return map_index


def density_map_from_attention(map_path, scale=4):
    """Render one saved attention grid to a density map, optionally downscaled."""
    saved_map = np.load(map_path, allow_pickle=True)
    width = int(saved_map["iw"])
    height = int(saved_map["ih"])
    if scale != 1:
        width = max(1, width // scale)
        height = max(1, height // scale)

    return gm.grid_to_density(
        np.asarray(saved_map["grid"], np.float64),
        width,
        height,
        fc=gm.FROZEN_FC,
    )


def reorganization_map(search_map, free_map):
    """Compute z(search map) - z(free-view map) for one image."""
    def standardize(density_map):
        values = np.asarray(density_map, float)
        values = values - values.mean()
        standard_deviation = values.std()
        return values / standard_deviation if standard_deviation > 0 else values

    return standardize(search_map) - standardize(free_map)


def reorganization_fidelity(model_search_map, model_free_map, human_search_map, human_free_map):
    """Compute corr(z(M_search) - z(M_free), z(H_search) - z(H_free))."""
    model_reorganization = reorganization_map(model_search_map, model_free_map)
    human_reorganization = reorganization_map(human_search_map, human_free_map)

    return gm.cc(model_reorganization, human_reorganization)


def paired_trace_cosines(cosines, value_columns=("d1_mean", "d2_mean")):
    """Return one row per image with search and memory cosine columns side by side."""
    paired = cosines.pivot_table(
        index=["task", "name"],
        columns="arm",
        values=list(value_columns),
        aggfunc="first",
    )
    paired.columns = [f"{value}_{arm}" for value, arm in paired.columns]
    paired = paired.reset_index()

    required_columns = []
    for value_column in value_columns:
        required_columns.extend([f"{value_column}_search", f"{value_column}_memory"])

    return paired.dropna(subset=required_columns).reset_index(drop=True)
