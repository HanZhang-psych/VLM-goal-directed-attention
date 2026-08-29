# trace/ — THINKING-TRACE arm (reasoning-mode semantic analysis)

Two-stage CPU pipeline over the search + free/memory `answers_<run>.jsonl` that produces the
per-image cosines consumed by `analysis/notebooks/04_thinking_trace_analysis.ipynb`
(pooled to `analysis/data/cosines_pooled.jsonl`).

Run the two stages directly. They import only third-party packages (`sentence-transformers`, spaCy),
so no `PYTHONPATH` wiring is needed:

```bash
# Stage 1: split masked thinking-trace sentences into MiniLM input units
python trace/extract_traces.py --search-runs <ids…> --free-runs <ids…> --out trace_units.json
# Stage 2: embed with all-MiniLM-L6-v2, write per-image target/salience cosines
python trace/embed_cosines.py --units trace_units.json --out cosines.jsonl
```

| file | role |
|---|---|
| `extract_traces.py` | split masked thinking-trace sentences into MiniLM input units (spaCy) |
| `embed_cosines.py` | embed units with `all-MiniLM-L6-v2`, write per-image target/salience cosines |

Stage 3 (the statistics) lives in `analysis/notebooks/04_thinking_trace_analysis.ipynb`.
