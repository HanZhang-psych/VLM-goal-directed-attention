# Emergent Goal-Directed Attention in Large Vision-Language Models

Code and data for the paper. We read a **goal-conditioned spatial priority map** out of frozen
vision–language models and compare it to human gaze on **COCO-Search18 / COCO-FreeView**. Two
architecturally distinct VLMs are treated as subjects — **Qwen3-VL-32B-Thinking** and
**Gemma-4-26B-A4B** — and the goal is manipulated on fixed images (search for {target} vs. memorize
the scene). The headline result is the goal-manipulation crossover (a double dissociation) in both
the models' emitted coordinates and their readable attention.

> 📄 Paper: `<PAPER_URL>` · 📦 Data: [doi.org/10.5281/zenodo.22151952](https://doi.org/10.5281/zenodo.22151952) (see [Data](#data))

## Repository layout

```
config.py, coco_fv.py, gaze_metrics.py   shared: frozen prompts/readout choices, COCO loader, gaze metrics
output/                                  OUTPUT arm — sample coordinate scanpaths via vLLM
attention/                               ATTENTION arm — capture eager attention, score offline
trace/                                   THINKING-TRACE arm — reasoning-trace semantic analysis
image_sets/                              the image lists fed to each run (task,name CSVs)
analysis/                                self-contained reproduction of the paper numbers (notebooks 01–05)
```

Each arm folder has its own `README.md` with the exact commands to run that arm.

## Two readout arms

Each model is read out two ways, both conditioned on the *same* frozen prompt (`config.py`):

- **Attention arm** — HF eager attention: the target-token → image-patch attention map.
  `attention/run_attn_{qwen,gemma}.py` (engines `attention/{qwen,gemma}_attn_readout.py`) dump raw
  per-layer attention; `attention/rescore_attn.py` scores it offline (no GPU). vLLM does not expose
  attention weights, so this arm uses the plain transformers path.
- **Output arm** — the emitted coordinate scanpaths, sampled via vLLM for throughput.
  `output/run_coord_sample_{qwen_vllm,gemma_vllm,gemma}.py` (engines `output/*_vllm_readout.py`). The
  raw `points_`/`human_points_` files are scored and assembled into the paper's results in `analysis/`.

## Data

The reproduction data is hosted on Zenodo —
**[doi.org/10.5281/zenodo.22151952](https://doi.org/10.5281/zenodo.22151952)**
(`VLM-gaze-reproduction-data.tar.gz`, CC-BY-4.0) — not in this repository.

1. **Reproduction data** — the per-run model outputs the analysis notebooks read
   (`points_`/`human_points_` CSVs, `attn_maps_*/*.npz`, `cosines_pooled.jsonl`). Download and
   extract it into `analysis/data/` (gitignored):
   ```bash
   # after downloading/extracting the Zenodo archive:
   #   analysis/data/<run>/...   and   analysis/data/cosines_pooled.jsonl
   ```
   Then run the notebooks — they read only `helpers/` + `analysis/data/`, nothing else.
2. **Raw provenance** (not in the Zenodo archive) — the full model generations (`answers_*.jsonl`,
   incl. the Qwen-Thinking traces) are available on request; the cluster job logs are not distributed.

**Human gaze / images.** To re-run the inference pipeline (not needed for the analysis) you also need
**COCO-Search18** and **COCO-FreeView** from their official source
(https://sites.google.com/view/cocosearch/), placed under `data/COCO-Search18/`. The analysis data
already bundles the pooled human fixations per run, so **the notebooks reproduce the numbers without
the raw COCO data.**

## Reproduce the paper numbers

```bash
# 1. download the reproduction data into analysis/data/ (see Data above)
# 2. open the notebooks (they use only analysis/helpers + analysis/data)
jupyter lab analysis/notebooks/
```

| Notebook | Result |
|---|---|
| `01_output_crossover`, `02_attention_crossover` | goal-manipulation double dissociation (2×2 prompt × human-map RM-ANOVA), output + attention arms |
| `03_output_reorganization` | per-scene reorganization fidelity (pervasiveness of the crossover) |
| `05_fixation_order` | fixation-order match (scasim-inspired + Sequence Score, vs shuffle floor / human ceiling) |
| `04_thinking_trace_analysis` | reasoning-trace semantic dissociation (Qwen-Thinking `<think>` traces; producers in `trace/`) |

## Regenerate the data from scratch (optional; needs 2 GPUs)

The analysis above needs none of this — it runs entirely from the released reproduction data. To
re-run the inference instead:

1. **Environments.** Create three Python environments: an HF/transformers env (attention arm), a
   vLLM env (output arm; Gemma-4 needs vLLM ≥ 0.27.2), and a CPU env with `sentence-transformers` +
   spaCy (trace arm).
2. **Weights** (~64 GB). Download the model weights from Hugging Face and point `HF_HOME` at the
   cache. Gemma is gated — accept the license and set `HF_TOKEN`.
3. **Run outputs.** Set `VLM_GAZE_OUTPUT_ROOT` to where runs should be written (default in `config.py`).
4. **COCO images** (only for a real run) — COCO-Search18 / COCO-FreeView from
   https://sites.google.com/view/cocosearch/, under `data/COCO-Search18/`.

Then run the entrypoints directly, with the repo root on `PYTHONPATH` so the arm scripts import the
shared root modules. Subset any `image_sets/*.csv` for a quick check.

```bash
# attention arm (GPU): dump raw attention, then score offline (CPU)
PYTHONPATH=. python attention/run_attn_qwen.py --run 001 --condition ta --images-csv image_sets/ta_val.csv
PYTHONPATH=. python attention/rescore_attn.py  --raw-dir "$VLM_GAZE_OUTPUT_ROOT/001/attn_raw_001" --condition ta --model qwen

# output arm (GPU, vLLM): sample coordinate scanpaths
PYTHONPATH=. python output/run_coord_sample_qwen_vllm.py --run 002 --condition ta --images-csv image_sets/ta_val.csv
```

Each arm folder's `README.md` lists the full set of entrypoints and flags.
