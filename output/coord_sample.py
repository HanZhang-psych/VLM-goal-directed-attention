"""coord_sample.py — model-agnostic harness for the coordinate readout.

Purpose: run the coordinate/output readout for ANY model while emitting one shared schema,
so the scoring (in analysis/) is identical across models:

  points_<run>.csv        mode,condition,task,name,iw,ih,seq,rank,x1000,y1000,x_px,y_px,region
  human_points_<run>.csv  task,name,iw,ih,kind,x_px,y_px
  answers_<run>.jsonl     the full model output per sample (provenance)

A model supplies a `readout` object with all of its model-specific I/O:
    readout.name                       -> str, for logging
    readout.load()                     -> load model+processor (idempotent)
    readout.prompt_for(task)           -> str, the model-appropriate frozen prompt
    readout.sample_answers(image, prompt, n) -> list[str]   (n sampled generations)
    readout.parse(text, iw, ih)        -> list[(x1000, y1000, x_px, y_px, region)]
                                          ranked; coords already converted to the
                                          letterbox-cropped image frame (x_px in [0,iw))
Everything model-specific (prompt phrasing, coordinate convention, output parsing)
lives in the readout; everything here is identical for every model.

OPTIONAL batched mode (chunk>0): a readout may also expose
    readout.sample_answers_batch(pairs, n) -> list[list[str]]   pairs = [(image, prompt), ...]
returning n generations per pair, aligned to the input order. When present and chunk>0, run()
submits `chunk` images at once — for engines with cross-request batching (vLLM continuous
batching) this keeps the pipeline full instead of one image at a time. Falls back to the
per-image path otherwise; the emitted files are byte-identical either way.
"""
import csv
import json
import os
import re

import numpy as np

import coco_fv as cf

# Reasoning ("thinking") models wrap a chain-of-thought in <think>...</think> before the answer;
# capture it as its own field so the trace is never lost (empty for other models). Some models
# (e.g. Qwen3-VL-Thinking) have the chat template supply the OPENING <think> in the prompt, so only the
# CLOSING </think> appears in the generated text — make the opener optional and grab everything up
# to the first </think> either way.
_THINK = re.compile(r"(?:<think>)?\s*(.*?)</think>", re.S)


def run(readout, images_csv, coco_dir, condition, run, outdir_root, n=20, chunk=0):
    outdir = os.path.join(outdir_root, str(run))
    os.makedirs(outdir, exist_ok=True)
    images = [(r["task"], r["name"]) for r in csv.DictReader(open(images_csv))]

    ta = cf.load_records(cf.fixations_path(coco_dir, condition))
    fv_idx = cf.index_by_name(cf.load_records(
        os.path.join(coco_dir, "COCOFreeView_fixations_trainval.json")))
    # letterbox-cropped image + human fixations on the same frame; density maps unused
    # here (all scoring is offline), but this guarantees the model sees the SAME frame
    # the human fixations live in, so emitted coordinates and gaze are comparable.
    cache = {(t, nm): cf.build_maps(nm, t, cf.index_by_name(ta, task=t), fv_idx,
                                    coco_dir, condition=condition, make_maps=False)
             for t, nm in images}

    # RAW human fixations so the human maps can be rebuilt at any smoothing/grid offline.
    with open(os.path.join(outdir, f"human_points_{run}.csv"), "w") as f:
        f.write("task,name,iw,ih,kind,x_px,y_px\n")
        for (task, name), m in cache.items():
            iw, ih = m["size"]
            for kind in ("search", "free"):
                for x, y in m["fix"][kind]:
                    f.write(f"{task},{name},{iw},{ih},{kind},{x:.1f},{y:.1f}\n")

    readout.load()
    batched = chunk and chunk > 0 and hasattr(readout, "sample_answers_batch")
    print(f"[i] {readout.name} | condition={condition} | {len(images)} images | n={n}"
          + (f" | chunk={chunk} (batched)" if batched else ""), flush=True)

    # stream (flush each image / each chunk) so a job timeout keeps everything written so far.
    pf = open(os.path.join(outdir, f"points_{run}.csv"), "w")
    pf.write("mode,condition,task,name,iw,ih,seq,rank,x1000,y1000,x_px,y_px,region\n")
    # ALWAYS capture the full model output per sample (one JSON line each), so what the model
    # actually said is never lost — essential for diagnosing parsing/yield across models.
    af = open(os.path.join(outdir, f"answers_{run}.jsonl"), "w")

    def emit(task, name, iw, ih, answers):
        """Parse + write one image's n answers (points + answers jsonl). Returns seqlens."""
        seqlens = []
        for seq_idx, ans in enumerate(answers):
            pts = readout.parse(ans, iw, ih)
            seqlens.append(len(pts))
            _th = _THINK.search(ans)
            af.write(json.dumps({"condition": condition, "task": task, "name": name,
                                 "seq": seq_idx, "n_points": len(pts),
                                 "thinking": _th.group(1).strip() if _th else "",
                                 "text": ans}) + "\n")
            for rank, (x1000, y1000, xpx, ypx, reg) in enumerate(pts):
                reg = (reg or "").replace(",", " ").replace("\n", " ")
                pf.write(f"scanpath,scanpath,{task},{name},{iw},{ih},{seq_idx},{rank},"
                         f"{int(x1000)},{int(y1000)},{xpx:.1f},{ypx:.1f},{reg}\n")
        return seqlens

    def report(idx, task, name, seqlens):
        npts = int(sum(seqlens)); nseq = sum(1 for s in seqlens if s > 0)
        avg = float(np.mean(seqlens)) if seqlens else 0.0
        print(f"  [{idx}/{len(images)}] {task:10s} {name}: {npts} pts / {nseq}/{n} seq "
              f"(~{avg:.1f}/seq)", flush=True)

    if batched:
        done = 0
        for s in range(0, len(images), chunk):
            group = images[s:s + chunk]
            pairs = [(cache[(t, nm)]["image"], readout.prompt_for(t)) for t, nm in group]
            results = readout.sample_answers_batch(pairs, n)   # list[list[str]], aligned to group
            for (task, name), answers in zip(group, results):
                iw, ih = cache[(task, name)]["size"]
                seqlens = emit(task, name, iw, ih, answers)
                done += 1
                report(done, task, name, seqlens)
            pf.flush(); af.flush()                              # checkpoint after each chunk
    else:
        for k, (task, name) in enumerate(images):
            m = cache[(task, name)]
            iw, ih = m["size"]
            answers = readout.sample_answers(m["image"], readout.prompt_for(task), n)
            if os.environ.get("OUTPUT_DEBUG") and answers:     # optional live view; file is source of truth
                print(f"      [dbg] ans0: {answers[0][:240]!r}", flush=True)
            seqlens = emit(task, name, iw, ih, answers)
            pf.flush(); af.flush()
            report(k + 1, task, name, seqlens)
    pf.close()
    af.close()
    print(f"[✓] wrote {outdir}/points_{run}.csv, human_points_{run}.csv, "
          f"answers_{run}.jsonl. Scoring is done in analysis/.", flush=True)
