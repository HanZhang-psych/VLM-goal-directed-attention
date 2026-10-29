# output/ — OUTPUT arm (model coordinate generation)

The output readout: prompt the model for ranked `REGION: … | POINT: (x, y)` locations and write the
shared `points_/human_points_/answers_<run>` schema that the `analysis/` notebooks score.

Run the entrypoints directly with the repo root on `PYTHONPATH`, so they import the shared root
modules (`config.py`, `coco_fv.py`, `gaze_metrics.py`) by bare name. The vLLM paths need 2 GPUs:

```bash
QWEN_VLLM_ID=Qwen/Qwen3-VL-32B-Instruct PYTHONPATH=. python output/run_coord_sample_qwen_vllm.py --run 001 --condition ta --images-csv image_sets/ta_val.csv  # Qwen-Instruct, vLLM (revision; omit QWEN_VLLM_ID for the original Thinking runs)
PYTHONPATH=. python output/run_coord_sample_gemma_vllm.py --run 001 --condition ta --images-csv image_sets/ta_val.csv  # Gemma, vLLM (scale-up)
PYTHONPATH=. python output/run_coord_sample_gemma.py      --run 001 --condition ta --images-csv image_sets/ta_val.csv  # Gemma, HF (val)
```

| file | role |
|---|---|
| `output_arm.py` | writes the `points_/human_points_/answers` schema; shared by every runner |
| `qwen_vllm_readout.py`, `gemma_vllm_readout.py` | vLLM decoding readouts; each also holds its model's output parsing (`qwen_vllm_readout` the Qwen parsers; `gemma_vllm_readout` the `GemmaReadout` HF base + `GemmaVllmReadout` subclass) |
| `run_coord_sample_qwen_vllm.py`, `…_gemma_vllm.py`, `…_gemma.py` | per-model entrypoints |
