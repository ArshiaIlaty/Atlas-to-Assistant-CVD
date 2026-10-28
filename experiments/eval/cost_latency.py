"""Cost & latency summary from a benchmark run (paper Section 4.7 — no human needed).

Reports per-condition mean/SD tokens, mean/SD end-to-end latency, and an estimated
$/query, and compares to the paper's reported figures.

    python3 -I experiments/eval/cost_latency.py experiments/results/<run>.json
"""
from __future__ import annotations
import sys, json
import numpy as np

# Approx GPT-4.1 pricing (USD per 1M tokens) during the study period; adjust as needed.
PRICE_IN, PRICE_OUT = 2.00, 8.00
PAPER = ("Paper §4.7: 900–1,000 total tokens (mean 940, SD 60); "
         "$0.005–$0.02/query; RAG latency 6.8s (SD 0.5); no-retrieval 6–7s.")


def main():
    path = sys.argv[1]
    recs = json.load(open(path))
    conds = {}
    for r in recs:
        u = r.get("usage") or {}
        conds.setdefault(r["condition"], {"tok": [], "pin": [], "pout": [], "lat": []})
        c = conds[r["condition"]]
        if u.get("total_tokens"):
            c["tok"].append(u["total_tokens"]); c["pin"].append(u.get("prompt_tokens") or 0)
            c["pout"].append(u.get("completion_tokens") or 0)
        if r.get("latency_s"):
            c["lat"].append(r["latency_s"])

    print(f"Cost & latency from {path}\n")
    print(f"{'condition':<14}{'tokens (mean±sd)':>20}{'$ / query':>12}{'latency s (mean±sd)':>22}")
    for cond, c in conds.items():
        tok = np.array(c["tok"], float); lat = np.array(c["lat"], float)
        cost = (np.array(c["pin"]) * PRICE_IN + np.array(c["pout"]) * PRICE_OUT) / 1e6
        tstr = f"{tok.mean():.0f}±{tok.std():.0f}" if len(tok) else "n/a"
        cstr = f"${cost.mean():.4f}" if len(cost) else "n/a"
        lstr = f"{lat.mean():.2f}±{lat.std():.2f}" if len(lat) else "n/a"
        print(f"{cond:<14}{tstr:>20}{cstr:>12}{lstr:>22}")
    print(f"\n{PAPER}")
    print(f"(pricing assumed: ${PRICE_IN}/1M in, ${PRICE_OUT}/1M out)")


if __name__ == "__main__":
    main()
