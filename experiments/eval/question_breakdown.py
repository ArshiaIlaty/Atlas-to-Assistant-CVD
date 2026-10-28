"""Per-question-type and per-Domain breakdown of the 50-question results
(reviewers R1#3 / R2#3): adds the missing axes by classifying each benchmark
question (via GPT) into one of five cognitive types (Bloom-style, per the paper's
benchmark design) and one of the paper's three Domains, then reports the judged
scores per axis and condition.

    python3 -I experiments/eval/question_breakdown.py --scored experiments/results/azure_v3_scored.json
"""
from __future__ import annotations
import os, sys, json, re, argparse
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd  # noqa: E402
import config, llm    # noqa: E402

TYPES = ["Knowledge/Recall", "Comprehension/Fact-Extraction", "Mechanism-Exploration",
         "Evidence-Based-Evaluation", "Study-Oriented-Clinical-Inquiry"]
DOMAINS = ["Basic & Translational Cardiovascular Science",
           "Cardiovascular Diseases & Interventions",
           "Population Health, Prevention & Policy"]
METRICS = ["factual_correctness", "clinical_relevance", "safety", "citation_faithfulness"]
SYS = ("Classify the cardiovascular research question. Return strict JSON "
       '{"qtype": <one of the listed types>, "qdomain": <one of the listed domains>}.')


def classify(q):
    user = (f"Types: {TYPES}\nDomains: {DOMAINS}\n\nQuestion: {q}\n\nJSON:")
    try:
        raw = llm.chat(SYS, user, max_tokens=2000)
        j = json.loads(re.search(r"\{.*\}", raw, re.S).group(0))
        t = j.get("qtype", ""); d = j.get("qdomain", "")
        return (t if t in TYPES else "Other"), (d if d in DOMAINS else "Other")
    except Exception:
        return "Other", "Other"


def load_questions():
    df = pd.read_excel(config.BENCHMARK_XLSX, sheet_name=0, header=0)
    return [str(q).strip() for q in df.iloc[:, 0].tolist() if str(q).strip() and str(q) != "nan"]


def breakdown(recs, qmeta, axis):
    """Mean of each metric per (axis-value, condition)."""
    agg = defaultdict(lambda: defaultdict(list))
    for r in recs:
        if not r.get("scores"):
            continue
        key = qmeta.get(r["qid"], {}).get(axis, "Other")
        for m in METRICS:
            v = r["scores"].get(m)
            if v is not None:
                agg[(key, r["condition"])][m].append(v)
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", default=os.path.join(config.RESULTS_DIR, "azure_v3_scored.json"))
    args = ap.parse_args()
    recs = json.load(open(args.scored))
    qs = load_questions()
    print(f"classifying {len(qs)} questions ...", file=sys.stderr, flush=True)
    qmeta = {}
    for i, q in enumerate(qs):
        t, d = classify(q)
        qmeta[i] = {"qtype": t, "qdomain": d}
    json.dump(qmeta, open(os.path.join(config.RESULTS_DIR, "question_labels.json"), "w"), indent=2)

    from collections import Counter
    print("\nQuestion-type counts:", dict(Counter(m["qtype"] for m in qmeta.values())))
    print("Question-domain counts:", dict(Counter(m["qdomain"] for m in qmeta.values())))

    for axis in ["qtype", "qdomain"]:
        print(f"\n===== Factual correctness by {axis} x condition =====")
        agg = breakdown(recs, qmeta, axis)
        keys = sorted({k for (k, _c) in agg})
        print(f"{'':<34}{'alone':>8}{'rag':>8}{'rag_tax':>9}  (n)")
        for k in keys:
            row = {}
            n = 0
            for c in ["alone", "rag", "rag_taxonomy"]:
                vals = agg.get((k, c), {}).get("factual_correctness", [])
                row[c] = sum(vals) / len(vals) if vals else float("nan")
                n = max(n, len(vals))
            print(f"{k[:33]:<34}{row['alone']:>8.2f}{row['rag']:>8.2f}{row['rag_taxonomy']:>9.2f}  ({n})")


if __name__ == "__main__":
    main()
