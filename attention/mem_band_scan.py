"""mem_band_scan.py — MEMORIZATION-arm readout selection under the SYMMETRIC (head-resolved) capture.

Selection scan for the `--prompt memory` dumps. Those dumps use the SEARCH
schema (target_first/target_all/baseline, [L, heads, HW]) with the query frozen to the goal word
'remember', so the memory arm can be scored through the SAME aggregation the search arm uses:
per-head baseline subtraction -> floor -> concentration-weighted average -> corner-zero -> rescale
(rescore_attn.attn_map_from_raw) — the same concentration-weighted pooling the search arm uses.

The point of the scan: the original "the memory signal lives at ONE layer, not a band" finding was
established under head-averaging, where pooling a band averages diffuse per-layer maps together and
cancels the signal. Concentration weighting downweights diffuse heads, so a band MAY now survive
pooling. This script tests exactly that by scoring, on the same images:

  * every named band in config.ATTN_BANDS (pooled, head-weighted), and
  * every single layer (head-weighted, so it is comparable to the bands),

reporting CC against the human FREE and SEARCH maps. Selection rule (unchanged):
among GOAL-SELECTIVE readouts (cc_free > cc_search), take the MAX cc_free.

Run this on the blind smoke set ONLY — it is a selection scan, and whatever it picks has to be
frozen before the test set is scored.

  python mem_band_scan.py --raw-dir output/<run>/attn_raw_<run> --condition ta
"""
import argparse
import glob
import os

import numpy as np

import coco_fv as cf
import gaze_metrics as ev
from config import ATTN_BANDS
from rescore_attn import attn_map_from_raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--condition", choices=["ta", "tp"], default="ta")
    ap.add_argument("--coco-dir", default="data/COCO-Search18")
    ap.add_argument("--key", default="target_first", choices=["target_first", "target_all"],
                    help="query array; 'remember' occurs once in the free-view prompt so the two "
                         "are identical there — kept for parity with the search arm's readout.")
    ap.add_argument("--head-power", type=float, default=4.0,
                    help="FROZEN at the search arm's value (4.0). Do not tune this here: applying "
                         "the search arm's already-frozen weighting unchanged IS the symmetry "
                         "claim; re-optimizing it would add a new free parameter.")
    ap.add_argument("--clip", type=float, default=99.0)
    ap.add_argument("--fc", type=float, default=ev.FROZEN_FC)
    ap.add_argument("--no-layers", action="store_true", help="skip the per-layer scan (bands only)")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.raw_dir, "*.npz")))
    if not files:
        raise SystemExit(f"no .npz in {args.raw_dir}")
    d0 = np.load(files[0], allow_pickle=True)
    if "per_token" in d0.files:
        raise SystemExit("this is an OLD freeview dump (per_token schema, head-averaged). Use "
                         "re-capture with --prompt memory to score it head-weighted.")
    if "target_first" not in d0.files:
        raise SystemExit(f"unrecognized dump schema: {d0.files}")
    nL = int(d0["target_first"].shape[0])
    nH = int(d0["target_first"].shape[1])
    print(f"[i] {len(files)} images, {nL} layers x {nH} heads, key={args.key}, "
          f"head_power={args.head_power}, clip={args.clip}, fc={args.fc}", flush=True)

    # readouts: (label, kwargs for attn_map_from_raw)
    jobs = [(f"band:{b}", {"layer_band": ATTN_BANDS[b]}) for b in ATTN_BANDS]
    if not args.no_layers:
        jobs += [(f"L{i}", {"layers": [i]}) for i in range(nL)]

    ta = cf.load_records(cf.fixations_path(args.coco_dir, args.condition))
    fv = cf.index_by_name(cf.load_records(
        os.path.join(args.coco_dir, "COCOFreeView_fixations_trainval.json")))

    agg = {lab: [] for lab, _ in jobs}
    for fp in files:
        d = np.load(fp, allow_pickle=True)
        task, name = str(d["task"]), str(d["name"])
        try:
            m = cf.build_maps(name, task, cf.index_by_name(ta, task=task), fv,
                              args.coco_dir, fc=args.fc, condition=args.condition)
        except Exception:                      # noqa: BLE001 - missing human map -> skip image
            continue
        iw, ih = m["size"]
        hs, hf = m["human"]["search"], m["human"]["free"]
        for lab, kw in jobs:
            amap = attn_map_from_raw(d, key=args.key, head_power=args.head_power,
                                     sink_clip_pct=args.clip, **kw)
            ar = ev.grid_to_density(amap, iw, ih, fc=args.fc)
            agg[lab].append((ev.cc(ar, hs), ev.cc(ar, hf)))

    rows = []
    for lab, _ in jobs:
        v = np.array(agg[lab])
        v = v[np.isfinite(v).all(axis=1)] if len(v) else v
        if not len(v):
            continue
        rows.append((lab, len(v), v[:, 0].mean(), v[:, 1].mean()))

    def show(title, subset):
        if not subset:
            return
        print(f"\n{title}")
        print(f"{'readout':>12s} {'n':>4s} {'cc_search':>10s} {'cc_free':>9s} {'free-search':>12s}")
        for lab, n, ccs, ccf in subset:
            print(f"{lab:>12s} {n:4d} {ccs:+10.3f} {ccf:+9.3f} {ccf - ccs:+12.3f}")

    show("BANDS (pooled, concentration-weighted)", [r for r in rows if r[0].startswith("band:")])
    show("SINGLE LAYERS (concentration-weighted)", [r for r in rows if not r[0].startswith("band:")])

    sel = [r for r in rows if r[3] > r[2]]     # goal-selective: cc_free > cc_search
    if not sel:
        print("\n[SELECT] NO goal-selective readout (none has cc_free > cc_search) — under the "
              "symmetric capture the memory arm is null. Report that; do not tune to rescue it.")
        return
    best = max(sel, key=lambda r: r[3])
    print(f"\n[SELECT] {best[0]}: cc_free {best[3]:+.3f} > cc_search {best[2]:+.3f} "
          f"(max cc_free among {len(sel)} goal-selective readouts)")
    bands_sel = [r for r in sel if r[0].startswith("band:")]
    if bands_sel:
        bb = max(bands_sel, key=lambda r: r[3])
        lyr = [r for r in sel if not r[0].startswith("band:")]
        bl = max(lyr, key=lambda r: r[3]) if lyr else None
        print(f"[i] best BAND {bb[0]} cc_free {bb[3]:+.3f}"
              + (f" vs best LAYER {bl[0]} cc_free {bl[3]:+.3f} "
                 f"-> {'BAND' if bb[3] >= bl[3] else 'LAYER'} wins" if bl else ""))

if __name__ == "__main__":
    main()
