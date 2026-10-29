# inference/ — model coordinate generation

The output readout: prompt the model for ranked `REGION: … | POINT: (x, y)` locations and write the
shared `points_/human_points_/answers_<run>` schema that the `analysis/` notebooks score.

Run the entrypoints directly with the repo root on `PYTHONPATH`, so they import the shared root
modules (`config.py`, `coco_fv.py`, `gaze_metrics.py`) by bare name. The vLLM paths need 2 GPUs:

```bash
PYTHONPATH=. python inference/run_coord_sample_qwen_vllm.py  --run 001 --condition ta --images-csv image_sets/ta_val.csv  # Qwen, vLLM
PYTHONPATH=. python inference/run_coord_sample_gemma_vllm.py --run 001 --condition ta --images-csv image_sets/ta_val.csv  # Gemma, vLLM
```

| file | role |
|---|---|
| `coord_sample.py` | writes the `points_/human_points_/answers` schema; shared by every runner |
| `qwen_vllm_readout.py`, `gemma_vllm_readout.py` | vLLM decoding readouts; each also holds its model's output parsing  |
| `run_coord_sample_qwen_vllm.py`, `…_gemma_vllm.py` | per-model entrypoints |
