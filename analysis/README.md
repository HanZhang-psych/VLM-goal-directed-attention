# analysis/ — reproducing the paper results

The notebooks that assemble the paper's numbers from the per-run data (`data/`, gitignored — see
the top-level README), with the small helper modules it needs. The density-map code is the repo-root
`gaze_metrics.py`, shared with `inference/`; `helpers/util.py` puts the repo root on the path.

## Layout

```
analysis/
  helpers/
    runs.py           run-ID registries: RUNS (model, condition, prompt) and ABLATIONS (model, condition, goal/format)
    util.py           CSV loaders (model points, human fixations) and density-map helpers
  data/<run>/         per-run inputs (gitignored; download the data release — see top-level README)
  notebooks/
    01_output_crossover.ipynb          2x2 goal-manipulation dissociation, target-present and target-absent
    02_goal_vs_format_ablation.ipynb   goal-row vs format-row ablation (intact references hard-coded from notebook 01)
```

## Running

```bash
# 1. download the data release into analysis/data/ (see the top-level README)
# 2. open the notebook
jupyter lab notebooks/
```

## Data model

Each cell is scored from the per-run files — no COCO JSON needed, because the pooled human
fixations travel with each run: `points_<run>.csv` (model-emitted ranked points) and
`human_points_<run>.csv` (pooled human search + free fixations, already letterbox-cropped,
`correct_only`/`drop_first` applied). `answers_<run>.jsonl` is bundled for provenance but is not
used by the 2x2. The one example-image figure reads a COCO image from `data/COCO-Search18/`.

## Method

Top-5 ranked model points per sample (`FROZEN_TOPN`), densities on an image/4 grid (CC is
scale-robust) FFT-Gaussian smoothed at `fc=6` — identical smoothing for humans and models. The
2x2 is prompt (search vs free-view) × human map (H_search vs H_free); each cell is the mean
per-image Pearson CC. The **prompt × map interaction is the double dissociation**, reported as a
2x2 repeated-measures ANOVA with paired follow-ups within each prompt.

## Run-ID provenance

Pooled = validation split + train-minus-smoke scale-up (disjoint image sets, so pooling =
concatenation). Images scored per 2x2: 2474 (TP) and 2413 (TA) for Qwen; 2474 (TP) and 2364 (TA) for
Gemma, whose target-absent search prompt yields no parseable `POINT` line in any of the 20 samples on
about 2% of images. Authoritative list: `helpers/runs.py`.

| model | cond | prompt | val | scale-up |
|---|---|---|---|---|
| Qwen3-VL-32B-Instruct | TP | search | 63889206 | 63889207 |
| Qwen3-VL-32B-Instruct | TP | free | 63889208 | 63889209 |
| Qwen3-VL-32B-Instruct | TA | search | 63889202 | 63889203 |
| Qwen3-VL-32B-Instruct | TA | free | 63889204 | 63889205 |
| Gemma-4-26B-A4B-it | TP | search | 64991404 | 58387155 |
| Gemma-4-26B-A4B-it | TP | free | 58411643 | 58387156 |
| Gemma-4-26B-A4B-it | TA | search | 64991402 | 58387153 |
| Gemma-4-26B-A4B-it | TA | free | 64991403 | 58387154 |

**Ablation cells** (notebook 02; HF causal hook, same prompts/sampler/image sets):

| model | cond | ablation | val | scale-up |
|---|---|---|---|---|
| Qwen3-VL-32B-Instruct | TP | goal-blind | 63953881 | 63998069 |
| Qwen3-VL-32B-Instruct | TP | format-blind | 63953880 | 63998068 |
| Qwen3-VL-32B-Instruct | TA | goal-blind | 63953877 | 63998067 |
| Qwen3-VL-32B-Instruct | TA | format-blind | 63953876 | 63998066 |
| Gemma-4-26B-A4B-it | TP | goal-blind | 64279305 | 64317440 |
| Gemma-4-26B-A4B-it | TP | format-blind | 64279304 + 64348378 | 64348381 |
| Gemma-4-26B-A4B-it | TA | goal-blind | 64279303 | 64317438 |
| Gemma-4-26B-A4B-it | TA | format-blind | 64279302 | 64317437 |
