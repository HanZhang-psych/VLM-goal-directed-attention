# attention/ — ATTENTION arm (prompt-attention readout)

One forward pass per image capturing prompt-token attention; the raw per-layer dumps are scored
**offline** at the frozen band/layer into `attn_summary_/attn_points_/attn_maps_<run>` (the files
the `analysis/` attention notebook reads).

Run the entrypoints directly with the repo root on `PYTHONPATH`, so they import the shared root
modules (`config.py`, `coco_fv.py`, `gaze_metrics.py`) by bare name, alongside `qwen_attn_readout.py`
and `gemma_attn_readout.py` here in this folder. The forward passes need 2 GPUs; the scorer is CPU-only:

```bash
# GPU: dump raw per-layer attention for one run (writes under $VLM_GAZE_OUTPUT_ROOT)
PYTHONPATH=. python attention/run_attn_qwen.py  --run 001 --condition ta --images-csv image_sets/ta_val.csv
PYTHONPATH=. python attention/run_attn_gemma.py --run 001 --condition ta --images-csv image_sets/ta_val.csv
# CPU: score the raw dumps offline at the frozen band/layer → the analysis inputs
PYTHONPATH=. python attention/rescore_attn.py --raw-dir "$VLM_GAZE_OUTPUT_ROOT/001/attn_raw_001" --condition ta --model qwen
```

| file | role |
|---|---|
| `run_attn_qwen.py`, `run_attn_gemma.py` | forward pass, raw per-layer/head attention dumps |
| `qwen_attn_readout.py` | loads Qwen3-VL (HF eager) and captures raw attention (`load` + `capture_raw`); imported by `run_attn_qwen.py` |
| `gemma_attn_readout.py` | loads Gemma-4 (HF eager) and captures raw attention (`load` + `capture_raw`); imported by `run_attn_gemma.py` |
| `rescore_attn.py` | offline scorer: raw dumps → `attn_summary_/attn_points_/attn_maps_` at the frozen band/layer |
| `mem_band_scan.py` | selection scan for the frozen memory layer/band (provenance for `config.FROZEN_MEM_LAYER`) |
