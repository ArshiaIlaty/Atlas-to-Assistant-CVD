"""Score a benchmark run with the LLM judge and compare the three conditions.

Pipeline:  run_benchmark.py  ->  results/run_all_*.json  ->  THIS  ->  scored + stats

For every answer it records:
  - citation_correctness  (keyless; share of cited PMIDs present in the context)
  - LLM-judge FC / CR / SR (0-9) + claim-level citation_faithfulness (needs a key)
Then it runs the paired inferential tests (experiments/eval/stats.py) for each
condition pair, aligned by question id.

The LLM judge is a SCALABLE COMPLEMENT to the blinded human raters, not a
substitute: on a subset, compare judge vs human with stats.cohens_kappa and
report it (reviewers R1#1 / R1#7 / R2#1).

Usage:
  python3 -I experiments/eval/judge_run.py --results experiments/results/run_all_hybrid_gen.json
  python3 -I experiments/eval/judge_run.py --results <run.json> --no-llm   # citation only
"""
from __future__ import annotations
import os, sys, json, argparse
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import stats as S          # noqa: E402
import metrics as M        # noqa: E402

JUDGE_METRICS = ["factual_correctness", "clinical_relevance", "safety", "citation_faithfulness"]
PAIRS = [("rag", "alone"), ("rag_taxonomy", "alone"), ("rag_taxonomy", "rag")]


def score_records(records, use_llm=True, checkpoint_path=None):
    for i, r in enumerate(records):
        ans = r.get("answer")
        if ans is None or r.get("scores"):          # skip already-scored (resume support)
            continue
        r.setdefault("scores", {})
        cc = M.citation_correctness(ans, r.get("context", ""))
        r["scores"]["citation_correctness"] = cc["correctness"]
        if use_llm:
            r["scores"].update(
                {k: v for k, v in M.llm_judge(r["question"], ans, r.get("context", "")).items()
                 if k in JUDGE_METRICS})
        if checkpoint_path and i % 15 == 0:          # periodic checkpoint (instance-loss resilience)
            json.dump(records, open(checkpoint_path, "w"), indent=2)
    return records


def aggregate_and_compare(records, metric_names):
    """records need: qid, condition, scores{metric: value}. Returns printable rows."""
    by = defaultdict(dict)  # by[condition][qid] = {metric: val}
    for r in records:
        if "scores" in r:
            by[r["condition"]][r["qid"]] = r["scores"]

    report = []
    for metric in metric_names:
        means = {c: _mean([v.get(metric) for v in by[c].values()]) for c in by}
        comps = []
        for hi, lo in PAIRS:
            if hi in by and lo in by:
                qids = sorted(set(by[hi]) & set(by[lo]))
                a = [by[hi][q].get(metric) for q in qids]
                b = [by[lo][q].get(metric) for q in qids]
                if any(x is not None for x in a):
                    comps.append((f"{hi} vs {lo}", S.compare_conditions(a, b, hi, lo)))
        report.append((metric, means, comps))
    return report


def _mean(xs):
    xs = [x for x in xs if x is not None and x == x]  # drop None and NaN
    return sum(xs) / len(xs) if xs else float("nan")


def print_report(report):
    for metric, means, comps in report:
        print(f"\n### {metric}")
        print("  means: " + ", ".join(f"{c}={m:.2f}" for c, m in means.items()))
        for name, r in comps:
            p = r["wilcoxon"]["p_value"]
            pstr = f"{p:.2e}" if p == p else "n/a"
            print(f"  {name:<22} Δ={r['mean_diff']:+.2f} "
                  f"CI[{r['ci_low']:+.2f},{r['ci_high']:+.2f}] "
                  f"Wilcoxon p={pstr} dz={r['cohens_dz']:.2f} (n={r['n']})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--no-llm", action="store_true", help="citation_correctness only (no key)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    records = json.load(open(args.results))
    if args.limit:
        keep = set(sorted({r["qid"] for r in records})[:args.limit])
        records = [r for r in records if r["qid"] in keep]

    import config
    if not args.no_llm and config.llm_backend() == "none":
        sys.exit("No LLM credentials for judging. Set a key or pass --no-llm.")

    out = args.out or args.results.replace(".json", "_scored.json")
    # resume: if a partial scored file exists, carry over its scores
    if os.path.exists(out):
        prev = {(x["qid"], x["condition"]): x.get("scores") for x in json.load(open(out))}
        for r in records:
            if not r.get("scores") and prev.get((r.get("qid"), r.get("condition"))):
                r["scores"] = prev[(r["qid"], r["condition"])]
    records = score_records(records, use_llm=not args.no_llm, checkpoint_path=out)
    json.dump(records, open(out, "w"), indent=2)
    print(f"wrote {out}")

    names = ["citation_correctness"] + ([] if args.no_llm else JUDGE_METRICS)
    print_report(aggregate_and_compare(records, names))


if __name__ == "__main__":
    main()
