"""run_coord_sample_qwen_vllm.py — OUTPUT arm for Qwen3-VL-32B-Thinking via vLLM.

Hands output_arm.run a QwenVllmReadout (vLLM continuous batching + chunked cross-image submission)
and emits the standard points_/human_points/answers schema; scoring is done in analysis/.
RUN IN A vLLM env (glm-vllm-env has vLLM; 32B needs 2 GPUs).

Usage:
  python run_coord_sample_qwen_vllm.py --run <jobid> --condition ta --images-csv image_sets/ta_val.csv
"""
import argparse

import output_arm
from config import DEFAULT_OUTPUT_ROOT, SLURM_JOB_ID
from qwen_vllm_readout import QwenVllmReadout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=SLURM_JOB_ID)
    ap.add_argument("--condition", choices=["ta", "tp"], default="ta")
    ap.add_argument("--images-csv", required=True, help="CSV with task,name columns")
    ap.add_argument("--coco-dir", default="data/COCO-Search18")
    ap.add_argument("--n", type=int, default=20, help="sampled scanpaths per image")
    ap.add_argument("--chunk", type=int, default=16,
                    help="images per vLLM generate() call (cross-image continuous batching). "
                         "0 = per-image. Lower if OOM.")
    ap.add_argument("--mode", choices=["point", "freeview"], default="point",
                    help="point = shared REGION|POINT prompt (default); freeview = TARGET-FREE "
                         "baseline (config.FREEVIEW_PROMPT)")
    args = ap.parse_args()
    if not args.run or not str(args.run).isdigit():
        ap.error("--run must be the numeric run/job id")
    output_arm.run(QwenVllmReadout(mode=args.mode), args.images_csv, args.coco_dir,
                   args.condition, args.run, DEFAULT_OUTPUT_ROOT, n=args.n, chunk=args.chunk)


if __name__ == "__main__":
    main()
