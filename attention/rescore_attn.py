"""
rescore_attn.py — offline map / CC / Sequence-Score / layer-band sweep for the attention
arm, from the raw per-layer/per-head attention dumped by run_attn_qwen.py / run_attn_gemma.py
(via the attention readouts' capture_raw). No GPU: every readout choice — layer band, head_power,
clip, drop-corners, first-vs-all mention — is a cheap CPU sweep here rather than a re-run.

`attn_map_from_raw` performs the frozen aggregation:
  per (layer, head) in the band: (target - baseline), clamp>=0; weight each unit by
  (top-5% mass share)**head_power; weighted-sum; clip at the sink_clip_pct percentile;
  zero the two structural corners; min-max to [0,1].
The resulting coarse vision-token grid is then rendered to full image resolution with
gaze_metrics.grid_to_density — the SAME fc-Gaussian smoothing used for the point-based
human/output maps — before scoring, so all arms are smoothed on the identical foveal scale.

Usage:
  python rescore_attn.py --raw-dir $VLM_GAZE_OUTPUT_ROOT/<run>/attn_raw_<run> \
      --condition ta --sweep       # first-mention CC per layer-band, + default-config SS
"""
import argparse, csv, glob, os
import numpy as np

import coco_fv as cf
import gaze_metrics as ev
from config import (ATTN_BANDS as BANDS, FROZEN_ATTN_BAND, DEFAULT_ATTN_BAND,
                    FROZEN_MEM_LAYER, FROZEN_MEM_QUERY, MODEL_IDS, model_key)

# Layer bands and the per-model FROZEN readout band live in config.py (single source of truth):
# the single-band SS/points output uses config.FROZEN_ATTN_BAND[--model], not a free --band flag.


def attn_map_from_raw(d, key="target_all", layer_band=(0.5, 0.9), head_power=4.0,
                      sink_clip_pct=99.0, drop_corners=True, suppress_sinks=True, layers=None):
    """Aggregate the raw dump into an HxW map in [0,1] (the frozen readout).

    `key` selects the query array: target_all / target_first (prompt readout, run_attn_qwen.py).
    `layers` (optional) is an explicit list of 0-indexed layer indices that OVERRIDES layer_band —
    used by mem_band_scan.py to score single layers exactly, without going through the fractional
    band arithmetic (i/L*L does not round-trip to i for every L). None = use layer_band, unchanged.
    """
    T = d[key]                                                   # [L, heads, HW]
    B = d["baseline"]
    grid_t, H, W = [int(x) for x in d["grid"]]
    L = T.shape[0]
    lo, hi = int(layer_band[0] * L), int(layer_band[1] * L)
    if layers is not None:
        chosen = [int(i) for i in layers]
    else:
        chosen = range(lo, hi) if hi > lo else range(L)

    sigs = []
    for li in chosen:
        sig = T[li] - B[li] if suppress_sinks else T[li]         # [heads, HW] minus-neutral
        sigs.append(np.clip(sig, 0, None))
    S = np.concatenate(sigs, axis=0).astype(np.float64)          # [units, HW]

    tot = S.sum(axis=1)
    K = max(1, int(round(0.05 * S.shape[1])))
    topK = np.sort(S, axis=1)[:, -K:].sum(axis=1)
    conc = np.where(tot > 1e-6, topK / (tot + 1e-8), 0.0)
    w = conc ** float(head_power) if head_power and head_power > 0 else np.ones_like(conc)
    w = w / (w.sum() + 1e-8)
    m = (w[:, None] * S).sum(axis=0).reshape(H, W)

    if suppress_sinks and 0.0 < sink_clip_pct < 100.0:
        hi_clip = float(np.percentile(m, sink_clip_pct))
        if hi_clip > 0:
            m = np.minimum(m, hi_clip)
    if drop_corners:
        m[0, 0] = 0.0; m[H - 1, W - 1] = 0.0
    m = m - m.min()
    if m.max() > 0:
        m = m / m.max()
    return m


def wta_ior(m, k_max, ior_cells):
    """Winner-take-all + inhibition-of-return -> ranked [(row, col), ...]."""
    m = m.astype(np.float64).copy(); H, W = m.shape
    yy, xx = np.ogrid[:H, :W]; peaks = []
    for _ in range(k_max):
        r, c = divmod(int(np.argmax(m)), W)
        if not np.isfinite(m[r, c]) or m[r, c] <= 0:
            break
        peaks.append((r, c)); m[(yy - r) ** 2 + (xx - c) ** 2 <= ior_cells ** 2] = -np.inf
    return peaks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True, help="dir of <task>_<name>.npz from run_attn")
    ap.add_argument("--coco-dir", default="data/COCO-Search18")
    ap.add_argument("--condition", choices=["ta", "tp"], required=True)
    ap.add_argument("--head-power", type=float, default=4.0)
    ap.add_argument("--clip", type=float, default=99.0)
    ap.add_argument("--model", default="qwen-thinking", choices=list(MODEL_IDS),
                    help="friendly model key; selects the FROZEN readout via config: search band "
                         "(qwen-thinking->mid, gemma->early) and memory layer FROZEN_MEM_LAYER "
                         "(qwen-thinking->L23, gemma->L28). DEFAULT = qwen-thinking (canonical).")
    ap.add_argument("--sweep", action="store_true", help="also report CC over all named bands")
    ap.add_argument("--memory", action="store_true",
                    help="score the dump as the MEMORIZATION arm instead of the search arm: the "
                         "single frozen layer config.FROZEN_MEM_LAYER[--model], with the SAME "
                         "concentration-weighted head aggregation as the search arm "
                         "(attn_map_from_raw, head_power). Use for `--prompt memory` dumps (query "
                         "frozen to the goal word 'remember' at capture). Both arms share one schema "
                         "(target_first/baseline), so the arm cannot be auto-detected — pass this "
                         "flag to select the memory readout.")
    ap.add_argument("--k-max", type=int, default=8)
    ap.add_argument("--ior-frac", type=float, default=0.06)
    ap.add_argument("--fc", type=float, default=ev.FROZEN_FC,
                    help="FFT-Gaussian cutoff freq for the full-resolution human maps (frozen fc=6)")
    args = ap.parse_args()

    run = os.path.basename(os.path.normpath(args.raw_dir))
    run = run.replace("attn_raw_", "")
    outdir = os.path.dirname(os.path.normpath(args.raw_dir)) or "."
    ta = cf.load_records(cf.fixations_path(args.coco_dir, args.condition))
    fv = cf.index_by_name(cf.load_records(
        os.path.join(args.coco_dir, "COCOFreeView_fixations_trainval.json")))
    files = sorted(glob.glob(os.path.join(args.raw_dir, "*.npz")))
    if not files:
        raise SystemExit(f"no .npz in {args.raw_dir}")
    mk = model_key(args.model)
    # Arm selection. Both arms share one dump schema (target_first/baseline), so the arm is chosen by
    # the --memory flag, not the schema:
    #   --memory : the single frozen layer config.FROZEN_MEM_LAYER[--model], scored with the SAME
    #              concentration-weighted head aggregation as the search arm (attn_map_from_raw,
    #              layers=[mem_layer]) -> the two attention arms are identical in pooling, differing
    #              only in query word ('remember' vs the target category) and single-layer-vs-band.
    #   default  : the SEARCH arm (frozen band + target-token).
    mem = bool(args.memory)
    if mem:
        if mk not in FROZEN_MEM_LAYER:
            raise SystemExit(f"no frozen memorization layer for model '{mk}' "
                             f"(config.FROZEN_MEM_LAYER = {FROZEN_MEM_LAYER}).")
        if "target_first" not in np.load(files[0]).files:
            raise SystemExit("--memory needs a `--prompt memory` dump (target_first/baseline "
                             f"schema); this dump has {np.load(files[0]).files}.")
        mem_layer = int(FROZEN_MEM_LAYER[mk])
        print(f"[i] MEMORIZATION arm: model={mk} -> frozen layer L{mem_layer}, "
              f"query={FROZEN_MEM_QUERY!r}, "
              f"pooling=concentration-weighted (head_power={args.head_power})", flush=True)
    else:
        pairs = [("first", "target_first")]    # search arm: strictly the first target mention
        band_name = FROZEN_ATTN_BAND.get(mk, DEFAULT_ATTN_BAND)
        bands = BANDS if args.sweep else {band_name: BANDS[band_name]}
        print(f"[i] SEARCH arm: model={args.model} -> frozen band '{band_name}' {BANDS[band_name]}"
              + (" (+ full sweep)" if args.sweep else ""), flush=True)

    sf = open(os.path.join(outdir, f"attn_summary_{run}.csv"), "w")
    sf.write("key,band,condition,task,name,n_pool,cc_search,cc_free,nss\n")
    pf = open(os.path.join(outdir, f"attn_points_{run}.csv"), "w")
    pf.write("mode,condition,task,name,iw,ih,seq,rank,x1000,y1000,x_px,y_px,region\n")
    # PERSIST the density map for the FROZEN readout (previously discarded after scoring). We save
    # the compact vision-token grid `amap`; grid_to_density(grid, iw, ih, fc) reconstructs the EXACT
    # full-resolution map `ar` that is correlated and visualized (bit-exact), at ~8 KB/image instead
    # of ~3 MB. So downstream can read maps + compute correlations without the ~15 MB/image raw dump.
    mapsdir = os.path.join(outdir, f"attn_maps_{run}")
    os.makedirs(mapsdir, exist_ok=True)

    agg = {}
    for fp in files:
        d = np.load(fp)
        task, name = str(d["task"]), str(d["name"])
        npool = int(d["n_occ"]) if "n_occ" in d else 0
        m = cf.build_maps(name, task, cf.index_by_name(ta, task=task), fv, args.coco_dir,
                          fc=args.fc, condition=args.condition)
        iw, ih = m["size"]
        hs, hf = m["human"]["search"], m["human"]["free"]
        # jobs: (label, band_label, coarse_map, is_frozen_readout). We report the raw crossover
        # alignments cc_search / cc_free (goal CC is retired); NSS is read at the arm's MATCHED
        # fixations (search -> H_search fixations; memory -> H_free fixations).
        if mem:                       # single frozen layer, concentration-weighted (as search arm)
            amap = attn_map_from_raw(d, key="target_first", layers=[mem_layer],
                                     head_power=args.head_power, sink_clip_pct=args.clip)
            jobs = [("memory", f"L{mem_layer}", amap, True)]
            nss_fix = m["fix"]["free"]
        else:
            jobs = []
            for lab, arr in pairs:
                for bname, band in bands.items():
                    amap = attn_map_from_raw(d, key=arr, layer_band=band,
                                             head_power=args.head_power, sink_clip_pct=args.clip)
                    jobs.append((lab, bname, amap, lab == pairs[0][0] and bname == band_name))
            nss_fix = m["fix"]["search"]
        for lab, bname, amap, frozen in jobs:
            # render the coarse token grid to full resolution with the SAME fc-Gaussian used for the
            # human/output maps (identical smoothing across BOTH arms) — replaces bilinear upsampling.
            ar = ev.grid_to_density(amap, iw, ih, fc=args.fc)
            ccs, ccf = ev.cc(ar, hs), ev.cc(ar, hf)
            nss = ev.nss(ar, nss_fix, iw, ih)
            sf.write(f"{lab},{bname},{args.condition},{task},{name},{npool},"
                     f"{ccs:.4f},{ccf:.4f},{nss:.4f}\n")
            agg.setdefault((lab, bname), []).append((ccs, ccf))
            # Save the frozen-readout density map (token grid + size) so the correlation/visualization
            # can be rebuilt with grid_to_density, no raw dump needed. fc is fixed at ev.FROZEN_FC.
            if frozen:
                np.savez_compressed(os.path.join(mapsdir, f"{task}_{name}.npz"),
                                    grid=amap.astype(np.float32), iw=iw, ih=ih,
                                    readout=lab, band=bname)
            # WTA+IOR scanpath for the frozen readout only (search: first key @ frozen band; memory: the layer)
            if frozen:
                Hh, Ww = amap.shape
                ior = args.ior_frac * float(np.hypot(Hh, Ww))
                for rank, (r, c) in enumerate(wta_ior(amap, args.k_max, ior)):
                    xpx = (c + 0.5) / Ww * iw; ypx = (r + 0.5) / Hh * ih
                    pf.write(f"scanpath,{args.condition},{task},{name},{iw},{ih},0,{rank},"
                             f"{int(round(xpx/iw*1000))},{int(round(ypx/ih*1000))},"
                             f"{xpx:.1f},{ypx:.1f},peak{rank}\n")
    sf.close(); pf.close()

    print(f"{'key':14s} {'band':9s} {'n':>4s} {'cc_search':>10s} {'cc_free':>9s}")
    for (lab, b), v in agg.items():
        v = np.array(v); v = v[np.isfinite(v).all(axis=1)]
        print(f"{lab:14s} {b:9s} {len(v):4d} {v[:, 0].mean():+10.3f} {v[:, 1].mean():+9.3f}")
    scan = f"layer L{mem_layer}, query '{FROZEN_MEM_QUERY}'" if mem else f"key 'first', band '{band_name}'"
    print(f"\nwrote attn_summary_{run}.csv, attn_points_{run}.csv, attn_maps_{run}/ "
          f"(frozen density maps; scanpath = {scan})")


if __name__ == "__main__":
    main()
