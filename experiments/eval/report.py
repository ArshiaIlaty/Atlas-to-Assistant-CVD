"""Turn a run/scored JSON into reviewable local files: a flat CSV and an HTML
report (per-question answers + the three-way stats table when scores exist).

    python3 -I experiments/eval/report.py experiments/results/<run>.json
"""
from __future__ import annotations
import os, sys, json, html, argparse
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import judge_run as JR  # noqa: E402

SCORE_COLS = ["citation_correctness", "factual_correctness", "clinical_relevance",
              "safety", "citation_faithfulness"]


def to_rows(records):
    rows = []
    for r in records:
        row = {
            "qid": r.get("qid"), "condition": r.get("condition"),
            "question": r.get("question"), "answer": r.get("answer"),
            "citations": ";".join(r.get("citations", []) or []),
            "retrieved_pmids": ";".join(r.get("retrieved_pmids", []) or []),
        }
        for k in SCORE_COLS:
            row[k] = (r.get("scores") or {}).get(k)
        rows.append(row)
    return rows


def write_csv(records, path):
    pd.DataFrame(to_rows(records)).to_csv(path, index=False)


def write_html(records, path):
    have_scores = any("scores" in r for r in records)
    parts = ["<!doctype html><meta charset=utf-8>",
             "<title>CVD RAG — results</title>",
             "<style>body{font:14px/1.5 system-ui,Arial;margin:24px;max-width:1100px}"
             "table{border-collapse:collapse;margin:12px 0}td,th{border:1px solid #ccc;padding:4px 8px}"
             ".q{background:#f4f6fb;padding:8px 12px;border-radius:6px;margin-top:22px}"
             ".cond{font-weight:600;color:#1a4d8f}.ans{white-space:pre-wrap;margin:6px 0 14px}"
             ".pmid{color:#557}</style>"]
    parts.append(f"<h1>CVD RAG results — {len(records)} records</h1>")

    if have_scores:
        names = ["citation_correctness", "factual_correctness", "clinical_relevance",
                 "safety", "citation_faithfulness"]
        names = [n for n in names if any((r.get("scores") or {}).get(n) is not None for r in records)]
        rep = JR.aggregate_and_compare(records, names)
        parts.append("<h2>Condition comparison (paired stats)</h2>")
        parts.append("<table><tr><th>Metric</th><th>means</th><th>pairwise Δ (Wilcoxon p, dz)</th></tr>")
        for metric, means, comps in rep:
            ms = ", ".join(f"{c}={m:.2f}" for c, m in means.items())
            cs = "<br>".join(
                f"{n}: Δ={r['mean_diff']:+.2f} [{r['ci_low']:+.2f},{r['ci_high']:+.2f}] "
                f"p={r['wilcoxon']['p_value']:.1e} dz={r['cohens_dz']:.2f}" for n, r in comps)
            parts.append(f"<tr><td>{metric}</td><td>{ms}</td><td>{cs}</td></tr>")
        parts.append("</table>")

    by_q = {}
    for r in records:
        by_q.setdefault(r.get("qid"), []).append(r)
    parts.append("<h2>Answers by question</h2>")
    for qid in sorted(by_q, key=lambda x: (x is None, x)):
        recs = by_q[qid]
        parts.append(f"<div class=q><b>Q{qid}.</b> {html.escape(str(recs[0].get('question','')))}</div>")
        for r in recs:
            sc = r.get("scores") or {}
            sc_str = " ".join(f"{k.split('_')[0]}={sc[k]:.2f}" for k in SCORE_COLS
                              if sc.get(k) is not None)
            parts.append(f"<div class=cond>{r.get('condition')} "
                         f"<span class=pmid>[{sc_str}]</span></div>")
            parts.append(f"<div class=ans>{html.escape(str(r.get('answer') or '(no answer)'))}</div>")
    open(path, "w").write("\n".join(parts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--out-prefix", default=None)
    args = ap.parse_args()
    records = json.load(open(args.results))
    prefix = args.out_prefix or os.path.splitext(args.results)[0]
    write_csv(records, prefix + ".csv")
    write_html(records, prefix + ".html")
    print(f"wrote {prefix}.csv and {prefix}.html")


if __name__ == "__main__":
    main()
