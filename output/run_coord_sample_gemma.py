"""run_coord_sample_gemma.py — coordinate readout for Gemma-4-26B-A4B-it (HF).

Thin driver: builds a GemmaReadout and hands it to the model-agnostic `coord_sample.run`, which
emits the standard points_/human_points/answers schema; scoring is done in analysis/.
The Gemma output readout (GemmaReadout) lives in gemma_vllm_readout.py.

Usage:
  python run_coord_sample_gemma.py --run <jobid> --condition ta --images-csv image_sets/_smoke_ta.csv
"""
import argparse

import coord_sample
from config import DEFAULT_OUTPUT_ROOT, SLURM_JOB_ID
from gemma_vllm_readout import GemmaReadout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=SLURM_JOB_ID)
    ap.add_argument("--condition", choices=["ta", "tp"], default="ta")
    ap.add_argument("--images-csv", required=True, help="CSV with task,name columns")
    ap.add_argument("--coco-dir", default="data/COCO-Search18")
    ap.add_argument("--n", type=int, default=20, help="sampled scanpaths per image")
    ap.add_argument("--batch", type=int, default=4,
                    help="num_return_sequences per forward pass (lower if OOM)")
    ap.add_argument("--min-new-tokens", type=int, default=64,
                    help="suppress EOS until this many tokens (forces Gemma past its canned "
                         "'no detections' bail-out into the REGION|POINT list); 0 = off")
    ap.add_argument("--mode", choices=["point", "box", "freeview"], default="point",
                    help="point = shared REGION|POINT prompt (default); box = Gemma-native JSON "
                         "box_2d detection (y-first), box centres — ablation for flaky yield")
    args = ap.parse_args()
    if not args.run or not str(args.run).isdigit():
        ap.error("--run must be the numeric run/job id")
    coord_sample.run(GemmaReadout(mode=args.mode, batch=args.batch,
                                min_new_tokens=args.min_new_tokens),
                   args.images_csv, args.coco_dir,
                   args.condition, args.run, DEFAULT_OUTPUT_ROOT, n=args.n)


if __name__ == "__main__":
    main()
