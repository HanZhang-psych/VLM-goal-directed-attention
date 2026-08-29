"""
run_attn_qwen.py — the ATTENTION arm of the paired output-vs-attention comparison, as a raw
capture-and-dump. For every image, ONE forward pass (no generation) over the SAME frozen
prompt the output run uses (config.FROZEN_PROMPT + FROZEN_SYSTEM), and it saves the raw
per-layer, per-head target->image attention (all-mention and first-mention query sets) plus
the neutral baseline, in fp32. Nothing is aggregated here.

Every readout choice — layer band, head_power, clip, drop-corners, first-vs-all mention —
is then a cheap CPU sweep over the dumps (rescore_attn.py). This mirrors the output arm
(save raw output, score offline): the readout config is applied offline, not baked into the
GPU run.

capture=True (eager attention) is memory-heavy, so the attention arm is a SEPARATE process from
the vLLM output run

Usage:
  python run_attn_qwen.py --run <jobid> --condition ta --images-csv image_sets/ta_val.csv
  python run_attn_qwen.py --run <jobid> --condition tp --images-csv image_sets/tp_val.csv
  (for a quick check, point --images-csv at a small CSV and score it with rescore_attn.py)
"""
import argparse
import csv
import os
import re

import numpy as np

import coco_fv as cf
import qwen_attn_readout as qr
from config import (DEFAULT_MODEL, DEFAULT_OUTPUT_ROOT, SLURM_JOB_ID,
                    FROZEN_PROMPT, FROZEN_SYSTEM, FREEVIEW_PROMPT,
                    FROZEN_MEM_QUERY)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=SLURM_JOB_ID)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--coco-dir", default="data/COCO-Search18")
    ap.add_argument("--condition", choices=["ta", "tp"], default="ta")
    ap.add_argument("--images-csv", required=True)
    ap.add_argument("--prompt", choices=["search", "memory"], default="search",
                    help="search = FROZEN_PROMPT target-token capture (default); memory = the "
                         "free-view/memory prompt with the query FROZEN to the goal word "
                         "(config.FROZEN_MEM_QUERY, 'remember'), captured PER HEAD in the SAME "
                         "schema as search (target_first/target_all/baseline) so the memory arm "
                         "gets the identical per-head difference + concentration weighting. Score "
                         "with `rescore_attn.py --memory`.")
    ap.add_argument("--max-pixels", type=int, default=768 * 768)
    ap.add_argument("--shard", default=None, metavar="i/N",
                    help="SCALE-UP: process only stride images[i::N] (i in 0..N-1). Point every shard "
                         "at the SAME RUN so they share attn_raw_<RUN>/; skip-existing makes it "
                         "resumable. Default (none) = all images, unchanged for val runs.")
    args = ap.parse_args()
    if not args.run or not str(args.run).isdigit():
        ap.error("--run must be the numeric run/job id")

    outdir = os.path.join(DEFAULT_OUTPUT_ROOT, str(args.run))
    rawdir = os.path.join(outdir, f"attn_raw_{args.run}")
    os.makedirs(rawdir, exist_ok=True)

    images = [(r["task"], r["name"]) for r in csv.DictReader(open(args.images_csv))]
    if args.shard:                                    # strided partition for multi-job scale-up
        si, sn = (int(x) for x in args.shard.split("/"))
        if not (0 <= si < sn):
            ap.error(f"--shard i/N needs 0<=i<N (got {args.shard})")
        images = images[si::sn]
        print(f"[i] shard {si}/{sn}: {len(images)} of this csv's images", flush=True)

    ta = cf.load_records(cf.fixations_path(args.coco_dir, args.condition))
    fv_idx = cf.index_by_name(cf.load_records(
        os.path.join(args.coco_dir, "COCOFreeView_fixations_trainval.json")))

    print(f"[i] loading {args.model} (capture=True) | condition={args.condition} "
          f"| {len(images)} images -> {rawdir}", flush=True)
    model, processor = qr.load(args.model, max_pixels=args.max_pixels, capture=True)
    meta = [n for n, p in model.named_parameters() if p.device.type == "meta"]
    meta_layers = sorted({int(x.group(1)) for n in meta
                          for x in [re.search(r"layers\.(\d+)\.", n)] if x})
    if args.prompt == "search":
        # SEARCH reads only the frozen band, so only the band must be resident.
        rb = qr.readout_band(model, args.model)
        band = [l for l in meta_layers if l in rb]
        if band:
            raise SystemExit(f"[!!] readout-band layers {band} offloaded to meta/CPU; "
                             "the attention would be corrupted. Re-run fully resident.")
    elif meta_layers:
        # memory dumps are captured at EVERY layer (the frozen memory layer is selected offline),
        # so ANY offloaded decoder layer corrupts the capture.
        raise SystemExit(f"[!!] decoder layers {meta_layers} offloaded to meta/CPU; the full-layer "
                         "memory capture would be corrupted. Re-run fully resident.")

    n_skip = 0
    for k, (task, name) in enumerate(images):
        npz_path = os.path.join(rawdir, f"{task}_{name}.npz")
        if os.path.exists(npz_path) and os.path.getsize(npz_path) > 0:   # resume: already dumped
            n_skip += 1
            continue
        m = cf.build_maps(name, task, cf.index_by_name(ta, task=task), fv_idx,
                          args.coco_dir, condition=args.condition, make_maps=False)
        # Atomic write (tmp -> os.replace) so a killed job can't leave a truncated .npz that
        # skip-existing would later treat as complete. Pass a file handle so numpy keeps the exact name.
        tmp = npz_path + ".tmp"
        if args.prompt == "memory":
            # Same free-view prompt, but queried like the search arm: the goal word is the query,
            # and the baseline is every other post-image token. Head axis preserved.
            raw = qr.capture_raw(model, processor, m["image"],
                                 FREEVIEW_PROMPT, FROZEN_MEM_QUERY, system=FROZEN_SYSTEM)
            if not raw["target_matched"]:
                raise SystemExit(
                    f"[!!] goal word {FROZEN_MEM_QUERY!r} not found in the tokenized free-view "
                    "prompt; _find_target_spans fell back to the last 4 prompt tokens, which would "
                    "silently score the WRONG query. Fix the query/tokenizer before continuing.")
            with open(tmp, "wb") as f:
                np.savez_compressed(
                    f, target_all=raw["target_all"], target_first=raw["target_first"],
                    baseline=raw["baseline"], grid=np.array(raw["grid"]),
                    iw=m["size"][0], ih=m["size"][1], n_occ=raw["n_occ"],
                    task=np.array(task), name=np.array(name))
        else:
            raw = qr.capture_raw(model, processor, m["image"],
                                 FROZEN_PROMPT.format(t=task), task, system=FROZEN_SYSTEM)
            with open(tmp, "wb") as f:
                np.savez_compressed(
                    f, target_all=raw["target_all"], target_first=raw["target_first"],
                    baseline=raw["baseline"], grid=np.array(raw["grid"]),
                    iw=m["size"][0], ih=m["size"][1], n_occ=raw["n_occ"],
                    task=np.array(task), name=np.array(name))
        os.replace(tmp, npz_path)
        if (k + 1) % 25 == 0:
            print(f"  [{k+1}/{len(images)}]  (skipped {n_skip} already-dumped)", flush=True)

    print(f"[✓] this shard: {len(images) - n_skip} newly dumped, {n_skip} skipped (already present) "
          f"-> {rawdir}/. Score offline once ALL shards finish: python rescore_attn.py "
          f"--raw-dir {rawdir} --condition {args.condition} --sweep", flush=True)


if __name__ == "__main__":
    main()
