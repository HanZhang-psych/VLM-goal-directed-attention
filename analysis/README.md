# analysis/ — reproducing the paper results

Self-contained reproduction of the paper's results.
Everything needed to re-derive the numbers lives in this folder: the scoring code
(`helpers/`), the raw per-run data (`data/`, gitignored — see below), and the notebooks that
assemble each result. **Nothing here imports from the repo root** — the helpers are copies of
the codebase's own scorers, so `analysis/` can be moved or shared on its own.

## Layout

```
analysis/
  helpers/
    gaze_metrics.py   density maps (FFT-Gaussian draw_heatmap) + Pearson CC
    runs.py           run-ID registry (RUNS) + CSV loaders (model points, human fixations)
    util.py           notebook helpers: scanpath/density loaders, reorganization, trace pivots
  data/<run>/         per-run inputs (gitignored; download the data release — see top-level README)
  notebooks/
    01_output_crossover.ipynb        output-arm 2x2 dissociation
    02_attention_crossover.ipynb     attention-arm 2x2 dissociation
    03_output_reorganization.ipynb   per-scene reorganization fidelity
    04_thinking_trace_analysis.ipynb reasoning-trace semantic dissociation
    05_fixation_order.ipynb          fixation-order match
```

## Running

```bash
# 1. download the data release into analysis/data/ (see the top-level README)
# 2. open the notebooks (they use only helpers/ + data/)
jupyter lab notebooks/
```

The first notebook run builds `.mapcache/` (the FFT-Gaussian densities); later runs reload it.

## Data model

Each result is scored from the per-run files — no COCO JSON or images
needed, because the pooled human fixations travel with each run:

- **OUTPUT arm** — `points_<run>.csv` (model-emitted ranked points) and
  `human_points_<run>.csv` (pooled human search + free fixations, already letterbox-cropped,
  `correct_only`/`drop_first` applied). `answers_<run>.jsonl` + the `.log` are bundled for
  provenance but are not used by the 2x2.
- **ATTENTION arm** — `notebooks/02_attention_crossover.ipynb` reads the saved
  `attn_maps_<run>/<task>_<name>.npz` files with `helpers.util.get_npz_paths()` and
  `helpers.util.density_map_from_attention()`, then recomputes correlations against human density maps
  in the notebook.

  **Density maps for the attention arm.** `rescore_attn` now persists the frozen-readout density map
  per image to `attn_maps_<run>/<task>_<name>.npz` (the compact vision-token grid + `iw,ih`; ~3 KB).
  `grid_to_density(grid, iw, ih, fc=6)` = `draw_heatmap` rebuilds the exact full-resolution map that
  is correlated and visualized. The raw `attn_raw_<run>` dumps are large (~15 MB/image) and are
  **not** bundled; the saved maps are.

## Method (must match the canonical scorer)

The OUTPUT 2x2 is built from the raw points as follows:
top-5 ranked model points per sample (`FROZEN_TOPN`), densities on an image/4 grid
(`DOWNSCALE=4`; CC is scale-robust) FFT-Gaussian smoothed at `fc=6` — identical smoothing for
humans and models. The 2x2 is prompt (search vs free-view) × human map (H_search vs H_free);
each cell is the mean per-image Pearson CC. The **prompt × map interaction is the double
dissociation**. Reported as a 2x2 repeated-measures ANOVA: for a within-image 2x2, each
effect's F(1, n−1) = t² of a one-sample test on the per-image contrast.

## Run-ID provenance — full 16-cell matrix

Pooled trainval−smoke = validation split + train-minus-smoke scale-up (disjoint image sets, so
pooling = concatenation), n ≈ 2474 per TP cell / 2413 per TA cell. 16 cells = 2 models × 2
conditions × 2 arms × 2 readouts. Output readouts = the two prompts {search, free}; attention
readouts = the two frozen readouts {search band, memory layer}. Authoritative list (with `val` /
`scaleup` split and stitch shards): `helpers/runs.py`. `helpers.runs.all_runs()` → all 37 run ids.

**Qwen3-VL-32B-Thinking** (attn: search band `mid`, memory layer L23)

| cond | arm · readout | val | scale-up |
|---|---|---|---|
| TP | output · search | 58031647 | 58094437 |
| TP | output · free | 58094796 | 58094446 |
| TP | attention · search | 58031828 | 58309309 |
| TP | attention · memory | 58306779 | 58309310 |
| TA | output · search | 58031597 | 58094212 |
| TA | output · free | 58031679 | 58094237 |
| TA | attention · search | 58031823 | 58309304 |
| TA | attention · memory | 58031836 | 58309306 |

**Gemma-4-26B-A4B-it** (attn: search band `early`, memory layer L28)

| cond | arm · readout | val | scale-up |
|---|---|---|---|
| TP | output · search | 57202064 + 57260139 (stitch) | 58387155 |
| TP | output · free | 58411643 | 58387156 |
| TP | attention · search | 57251467 | 58380383 |
| TP | attention · memory | 58411644 | 58380384 |
| TA | output · search | 57202063 | 58387153 |
| TA | output · free | 57707609–57707613 (5 shards) | 58387154 |
| TA | attention · search | 57251466 | 58380381 |
| TA | attention · memory | 57889360 | 58380382 |

Every cell above was verified to reproduce the paper's reported numbers (cell means to 3 dp; attention
and Qwen output exact; Gemma output exact once the stitch shard / val shards are included).
