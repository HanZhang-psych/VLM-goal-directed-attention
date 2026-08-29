"""run_attn_gemma.py — PROMPT-attention arm for Gemma-4, raw capture-and-dump.

One forward pass per image over the SAME frozen prompt the Gemma output arm uses, saving raw
per-layer/per-head target->image attention (+ neutral baseline) in fp32 for the shared offline
scorer (rescore_attn.py). All layers are dumped; rescore_attn scores at the frozen band.

The residency guard aborts if ANY decoder layer offloaded to meta/CPU, since the dump covers all
layers (the Qwen search run guards only its frozen band).

Usage:
  python run_attn_gemma.py --run <jobid> --condition ta --images-csv image_sets/_smoke_ta_large.csv
Score:
  python rescore_attn.py --raw-dir output/<run>/attn_raw_<run> --condition ta --model gemma --sweep
"""
import argparse
import csv
import os
import re

import numpy as np

import coco_fv as cf
import gemma_attn_readout as gr
from config import DEFAULT_OUTPUT_ROOT, SLURM_JOB_ID, FROZEN_PROMPT, MODEL_IDS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=SLURM_JOB_ID)
    ap.add_argument("--model", default=MODEL_IDS["gemma"])
    ap.add_argument("--coco-dir", default="data/COCO-Search18")
    ap.add_argument("--condition", choices=["ta", "tp"], default="ta")
    ap.add_argument("--images-csv", required=True)
    ap.add_argument("--prompt", choices=["search", "memory"], default="search",
                    help="search = FROZEN_PROMPT target-token capture (default); memory = the "
                         "free-view/memory prompt with the query FROZEN to the goal word "
                         "(config.FROZEN_MEM_QUERY, 'remember'), captured PER HEAD in the SAME "
                         "schema as search so the memory arm gets the identical per-head difference "
                         "+ concentration weighting. Score with `rescore_attn.py --memory`.")
    ap.add_argument("--max-pixels", type=int, default=768 * 768)
    args = ap.parse_args()
    if not args.run or not str(args.run).isdigit():
        ap.error("--run must be the numeric run/job id")

    outdir = os.path.join(DEFAULT_OUTPUT_ROOT, str(args.run))
    rawdir = os.path.join(outdir, f"attn_raw_{args.run}")
    os.makedirs(rawdir, exist_ok=True)

    images = [(r["task"], r["name"]) for r in csv.DictReader(open(args.images_csv))]
    ta = cf.load_records(cf.fixations_path(args.coco_dir, args.condition))
    fv_idx = cf.index_by_name(cf.load_records(
        os.path.join(args.coco_dir, "COCOFreeView_fixations_trainval.json")))

    print(f"[i] loading {args.model} (capture=True) | condition={args.condition} "
          f"| {len(images)} images -> {rawdir}", flush=True)
    model, processor = gr.load(args.model, max_pixels=args.max_pixels, capture=True)

    L = gr.num_text_layers(model)
    meta = sorted({int(x.group(1)) for n, p in model.named_parameters() if p.device.type == "meta"
                   for x in [re.search(r"layers\.(\d+)\.", n)] if x})
    if meta:
        raise SystemExit(f"[!!] decoder layers {meta} offloaded to meta/CPU; the full-layer sweep "
                         "would be corrupted. Re-run fully resident (more GPUs / lower max-pixels).")
    print(f"[i] text_layers={L}; all resident (full-layer sweep OK).", flush=True)

    from config import FREEVIEW_PROMPT, FROZEN_MEM_QUERY
    for k, (task, name) in enumerate(images):
        m = cf.build_maps(name, task, cf.index_by_name(ta, task=task), fv_idx,
                          args.coco_dir, condition=args.condition, make_maps=False)
        if args.prompt == "memory":
            # Same free-view prompt, queried like the search arm: goal word = query, baseline =
            # every other post-image token. Head axis preserved (search schema).
            raw = gr.capture_raw(model, processor, m["image"],
                                 FREEVIEW_PROMPT, FROZEN_MEM_QUERY)
            if not raw["target_matched"]:
                raise SystemExit(
                    f"[!!] goal word {FROZEN_MEM_QUERY!r} not found in the tokenized free-view "
                    "prompt; _find_target_spans fell back to the last 4 prompt tokens, which would "
                    "silently score the WRONG query. Fix the query/tokenizer before continuing.")
            np.savez_compressed(
                os.path.join(rawdir, f"{task}_{name}.npz"),
                target_all=raw["target_all"], target_first=raw["target_first"],
                baseline=raw["baseline"], grid=np.array(raw["grid"]),
                iw=m["size"][0], ih=m["size"][1], n_occ=raw["n_occ"],
                task=np.array(task), name=np.array(name))
        else:
            raw = gr.capture_raw(model, processor, m["image"],
                                 FROZEN_PROMPT.format(t=task), task)
            np.savez_compressed(
                os.path.join(rawdir, f"{task}_{name}.npz"),
                target_all=raw["target_all"], target_first=raw["target_first"],
                baseline=raw["baseline"], grid=np.array(raw["grid"]),
                iw=m["size"][0], ih=m["size"][1], n_occ=raw["n_occ"],
                task=np.array(task), name=np.array(name))
        if (k + 1) % 10 == 0:
            print(f"  [{k+1}/{len(images)}]", flush=True)

    print(f"[✓] dumped {len(images)} raw attention files to {rawdir}/. "
          f"Score offline: python rescore_attn.py --raw-dir {rawdir} "
          f"--condition {args.condition} --model gemma --sweep", flush=True)


if __name__ == "__main__":
    main()
