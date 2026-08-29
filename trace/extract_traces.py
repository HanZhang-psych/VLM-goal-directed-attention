"""STAGE 1 (local) — thinking traces -> minimal cleaned SENTENCE dataset.

Splits each trace into sentences (spaCy), MASKS the target surface-forms (the search
prompt supplies the target, so ~40% of search sentences name it vs ~1% of memory —
masking removes that tautological target-axis boost), and pools per image per arm.
Writes ONE small JSON: {salience_seed, cells:[{arm,task,name,sents:[masked sentences]}]}.

The SAME sentences are projected onto BOTH axes in Stage 2 (target axis / salience axis)
for the semantic double dissociation. No embeddings here.

Input, either:
  --traces traces.jsonl                       # {arm,task,name,seq,think} per line
  --search-runs ... --free-runs ... --root R  # raw answers_<run>.jsonl (arm = search|memory)
"""
import argparse
import collections
import json
import os
import re

import spacy

# frozen echo-clean attentional-salience seed (embedded in Stage 2; excludes the memory
# prompt's "draw your attention"). Lives here so the units file is self-describing.
SALIENCE_SEED = ["salient", "conspicuous", "prominent", "striking", "noticeable",
                 "dominant", "stands out"]

MIN_CONTENT_WORDS = 3   # drop sentences left too short after masking

# generic scene/meta nouns that do NOT count as a real scene object
STOP_NOUNS = set("""image picture photo photograph scene background foreground area areas
region regions location locations place places spot spots side sides corner corners center
centre middle part parts section view point points thing things object objects item items
format answer question coordinate coordinates position positions row column top bottom left
right edge edges space step steps way ways list rank order number level focus element
elements setup detail details context something anything everything nothing lot
task tasks problem problems goal goals instruction instructions prompt model user phrase
phrases noun nouns sense approach idea ideas guess option options choice possibility
attention priority""".split())

# answer-format / coordinate lines the model emits inside the trace (REGION:|POINT:|(x, y))
COORD_RE = re.compile(r"REGION\s*:|POINT\s*:|\(\s*\d+\s*,\s*\d+\s*\)", re.I)


def target_forms(task):
    tw = {task, task + "s", task.rstrip("s")} | set(task.split())
    if task == "potted plant":
        tw |= {"plant", "plants", "pot", "potted"}
    if task == "stop sign":
        tw |= {"sign", "signs", "stop"}
    return {w for w in tw if w}


def has_scene_noun(sent, tgt_forms):
    """True if the sentence names a real scene object (a NOUN/PROPN that is neither a
    generic scene/meta word nor the target). Drops procedural filler, coordinate lines,
    and masking stubs in one principled rule."""
    for tok in sent:
        if tok.pos_ in ("NOUN", "PROPN"):
            lem = tok.lemma_.lower()
            if lem.isalpha() and len(lem) >= 3 and lem not in STOP_NOUNS and lem not in tgt_forms:
                return True
    return False


def target_masker(task):
    parts = sorted((re.escape(w) for w in target_forms(task)), key=len, reverse=True)
    pat = re.compile(r"\b(?:" + "|".join(parts) + r")\b", re.I)
    return pat


def mask_sentence(text, pat):
    t = pat.sub(" ", text)
    t = " ".join(t.split())
    return t


def _think(r):
    th = r.get("thinking") or ""
    if th.strip():
        return th
    tx = r.get("text") or ""
    return tx.split("</think>")[0] if "</think>" in tx else ""


def load_records(args):
    if args.traces:
        for line in open(args.traces):
            yield json.loads(line)
        return
    for arm, runs in (("search", args.search_runs), ("memory", args.free_runs)):
        for run in runs:
            p = os.path.join(args.root, str(run), f"answers_{run}.jsonl")
            if not os.path.exists(p):
                print(f"[skip] missing {p}")
                continue
            for line in open(p):
                r = json.loads(line)
                yield {"arm": arm, "task": r["task"], "name": r["name"],
                       "seq": r.get("seq"), "think": _think(r)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces")
    ap.add_argument("--search-runs", nargs="+", default=[])
    ap.add_argument("--free-runs", nargs="+", default=[])
    ap.add_argument("--root", default=".")
    ap.add_argument("--max-seq", type=int, default=20)
    ap.add_argument("--n-process", type=int, default=2,
                    help="spaCy parallelism (perf only; does not change output)")
    ap.add_argument("--out", default="trace_units.json")
    args = ap.parse_args()

    # parser for sentence boundaries; lemmatizer for the scene-noun stop-list match
    nlp = spacy.load("en_core_web_sm", disable=["ner"])
    records = [r for r in load_records(args) if (r.get("seq") or 0) < args.max_seq]
    maskers = {t: target_masker(t) for t in {r["task"] for r in records}}

    pooled = collections.defaultdict(list)
    for i, (r, doc) in enumerate(zip(records, nlp.pipe((r["think"] for r in records),
                                                       batch_size=64, n_process=args.n_process))):
        pat = maskers[r["task"]]
        tw = target_forms(r["task"])
        cell = pooled[(r["arm"], r["task"], r["name"])]
        for s in doc.sents:
            if COORD_RE.search(s.text):                 # drop answer-format / coordinate lines
                continue
            if not has_scene_noun(s, tw):               # keep only sentences about a scene object
                continue
            masked = mask_sentence(s.text, pat)
            if sum(1 for w in re.findall(r"[a-zA-Z]+", masked) if len(w) >= 2) >= MIN_CONTENT_WORDS:
                cell.append(masked)
        if (i + 1) % 4000 == 0:
            print(f"  ...{i+1}/{len(records)} parsed", flush=True)

    cells = [{"arm": a, "task": t, "name": n, "sents": sents}
             for (a, t, n), sents in pooled.items()]
    with open(args.out, "w") as fh:
        json.dump({"salience_seed": SALIENCE_SEED, "cells": cells}, fh)
    n_sent = sum(len(c["sents"]) for c in cells)
    print(f"[i] {len(records)} traces -> {len(cells)} cells, {n_sent} masked sentences -> {args.out}")


if __name__ == "__main__":
    main()
