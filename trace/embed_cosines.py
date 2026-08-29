"""STAGE 2 (cluster) — masked SENTENCES -> MiniLM cosine values on BOTH axes.

Reads trace_units.json (Stage 1), embeds every unique sentence ONCE with MiniLM
(all-MiniLM-L6-v2), and projects each sentence onto two axes:
  d1 = cos(sentence, TARGET)             target axis  (goal-relevance)
  d2 = cos(sentence, SALIENCE centroid)  salience axis (stimulus-driven framing)
Writes per-image MEANS over the cell's sentences: {arm,task,name,d1_mean,d1_n,d2_mean,d2_n}.

The same sentences feed both axes -> the 2x2 (arm x axis) double dissociation in Stage 3.
Needs HuggingFace -> run on the cluster.
"""
import argparse
import json

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", default="trace_units.json")
    ap.add_argument("--out", default="cosines.jsonl")
    ap.add_argument("--model", default="all-MiniLM-L6-v2")
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer

    data = json.load(open(args.units))
    cells, seed = data["cells"], data["salience_seed"]

    # unique strings to embed: all sentences + targets + seed
    sents = sorted({s for c in cells for s in c["sents"]})
    tasks = sorted({c["task"] for c in cells})
    refs = sorted(set(seed) | set(tasks))
    all_strings = sents + refs

    model = SentenceTransformer(args.model)
    V = model.encode(all_strings, normalize_embeddings=True, batch_size=256,
                     show_progress_bar=True)
    E = {s: v for s, v in zip(all_strings, V)}

    cen = np.mean([E[w] for w in seed if w in E], axis=0)
    cen = cen / np.linalg.norm(cen)

    def mean_cos(strings, ref):
        cs = [float(E[s] @ ref) for s in strings if s in E]
        return (float(np.mean(cs)) if cs else None, len(cs))

    with open(args.out, "w") as fh:
        for c in cells:
            tv = E.get(c["task"])
            d1, n1 = mean_cos(c["sents"], tv) if tv is not None else (None, 0)
            d2, n2 = mean_cos(c["sents"], cen)
            fh.write(json.dumps({"arm": c["arm"], "task": c["task"], "name": c["name"],
                                 "d1_mean": d1, "d1_n": n1,
                                 "d2_mean": d2, "d2_n": n2}) + "\n")
    print(f"[i] embedded {len(all_strings)} strings "
          f"({len(sents)} sentences) -> {len(cells)} cells -> {args.out}")


if __name__ == "__main__":
    main()
