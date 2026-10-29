# Emergent Goal-Directed Attention in Large Vision-Language Models

Code and data for the paper. We read a **goal-conditioned spatial priority map** out of frozen
vision–language models' emitted coordinates and compare it to human gaze on **COCO-Search18 /
COCO-FreeView**. Two architecturally distinct VLMs are treated as subjects — **Qwen3-VL-32B-Instruct**
and **Gemma-4-26B-A4B-it** — and the goal is manipulated on fixed images (search for {target} vs.
memorize the scene). The headline result is the goal-manipulation crossover (a double dissociation)
in the models' emitted coordinates.

> 📄 Paper: `<PAPER_URL>` · 📦 Data: [doi.org/10.5281/zenodo.22151952](https://doi.org/10.5281/zenodo.22151952) (see [Data](#data))

## Repository layout

```
config.py, coco_fv.py, gaze_metrics.py   shared: frozen prompts, output length cap, COCO loader, gaze metrics
output/                                  sample coordinate scanpaths via vLLM
image_sets/                              the image lists fed to each run (task,name CSVs)
analysis/                                self-contained reproduction of the paper numbers (notebook 01)
```

`output/README.md` has the exact commands to generate the model outputs.

## The readout

Each model is prompted with the *same* frozen search or free-view prompt (`config.py`) and asked for
ranked `REGION: … | POINT: (x, y)` locations; the sampled coordinate scanpaths are rendered as
density maps and scored against the human search and free-view fixation maps.
`output/run_coord_sample_{qwen_vllm,gemma_vllm,gemma}.py` (engines `output/*_vllm_readout.py`)
write the raw `points_`/`human_points_` files, which `analysis/` scores and assembles into the
paper's results.

## Data

The reproduction data is hosted on Zenodo —
**[doi.org/10.5281/zenodo.22151952](https://doi.org/10.5281/zenodo.22151952)**
(`VLM-gaze-reproduction-data.tar.gz`, CC-BY-4.0) — not in this repository.

1. **Reproduction data** — the per-run model outputs the analysis notebook reads
   (`points_`/`human_points_` CSVs). Download and extract it into `analysis/data/` (gitignored):
   ```bash
   # after downloading/extracting the Zenodo archive:
   #   analysis/data/<run>/points_<run>.csv, human_points_<run>.csv
   ```
   Then run the notebook — it reads only `helpers/` + `analysis/data/`, nothing else.
2. **Raw provenance** (not in the Zenodo archive) — the full model generations (`answers_*.jsonl`)
   are available on request; the cluster job logs are not distributed.

**Human gaze / images.** To re-run the inference pipeline (not needed for the analysis) you also need
**COCO-Search18** and **COCO-FreeView** from their official source
(https://sites.google.com/view/cocosearch/), placed under `data/COCO-Search18/`. The analysis data
already bundles the pooled human fixations per run, so **the notebook reproduces the numbers without
the raw COCO data** (only the example-image figure reads one COCO image).

## Reproduce the paper numbers

```bash
# 1. download the reproduction data into analysis/data/ (see Data above)
# 2. open the notebook (it uses only analysis/helpers + analysis/data)
jupyter lab analysis/notebooks/
```

| Notebook | Result |
|---|---|
| `01_output_crossover` | goal-manipulation double dissociation (2×2 prompt × human-map RM-ANOVA), target-present and target-absent, both models |

## Regenerate the data from scratch (optional; needs 2 GPUs)

The analysis above needs none of this — it runs entirely from the released reproduction data. To
re-run the inference instead:

1. **Environment.** A vLLM env (Gemma-4 needs vLLM ≥ 0.27.2).
2. **Weights** (~64 GB). Download the model weights from Hugging Face and point `HF_HOME` at the
   cache. Gemma is gated — accept the license and set `HF_TOKEN`.
3. **Run outputs.** Set `VLM_GAZE_OUTPUT_ROOT` to where runs should be written (default in `config.py`).
4. **COCO images** — COCO-Search18 / COCO-FreeView from https://sites.google.com/view/cocosearch/,
   under `data/COCO-Search18/`.

Then run the entrypoints directly, with the repo root on `PYTHONPATH` so the scripts import the
shared root modules. Subset any `image_sets/*.csv` for a quick check.

```bash
PYTHONPATH=. python output/run_coord_sample_qwen_vllm.py --run 002 --condition ta --images-csv image_sets/ta_val.csv
PYTHONPATH=. python output/run_coord_sample_gemma_vllm.py --run 003 --condition ta --images-csv image_sets/ta_val.csv
```

`output/README.md` lists the full set of entrypoints and flags.
