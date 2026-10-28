"""Public-benchmark evaluation: PubMedQA (expert-labeled, pqa_labeled / ori_pqal).

Situates our generation model against an established biomedical QA benchmark
(reviewer R1#10) with a fully automatic metric — no human rater. Uses PubMedQA's
own provided contexts (the benchmark's standard "reasoning-required" setting):
given the question + context abstracts, predict yes / no / maybe, scored against
the expert gold label with accuracy and macro-F1.

Data (download once into its own dir, then pass the path):
  mkdir -p experiments/data/pubmedqa
  curl -sL -o experiments/data/pubmedqa/ori_pqal.json \
    https://raw.githubusercontent.com/pubmedqa/pubmedqa/master/data/ori_pqal.json

  python3 -I experiments/eval/pubmedqa.py --data experiments/data/pubmedqa/ori_pqal.json --n 200
"""
from __future__ import annotations
import os, sys, json, argparse, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import llm  # noqa: E402

SYSTEM = ("You are a biomedical QA expert. Using ONLY the provided context, answer the "
          "research question with exactly one word: yes, no, or maybe. Answer 'maybe' if "
          "the context is inconclusive.")
LABELS = ["yes", "no", "maybe"]


def classify(answer):
    m = re.search(r"\b(yes|no|maybe)\b", (answer or "").lower())
    return m.group(1) if m else "maybe"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    data = json.load(open(args.data))
    items = list(data.items())
    import random
    random.Random(args.seed).shuffle(items)
    items = items[: args.n]

    rows, gold, pred = [], [], []
    for pmid, rec in items:
        ctx = "\n".join(rec.get("CONTEXTS", []))
        q = rec["QUESTION"]
        g = rec["final_decision"].lower().strip()
        user = f"Context:\n{ctx}\n\nQuestion: {q}\n\nAnswer (yes/no/maybe):"
        try:
            ans = llm.chat(SYSTEM, user, max_tokens=2000)
        except Exception as e:
            ans = f"(error: {type(e).__name__})"
        p = classify(ans)
        gold.append(g); pred.append(p)
        rows.append({"pmid": pmid, "gold": g, "pred": p, "correct": int(g == p)})

    from sklearn.metrics import accuracy_score, f1_score
    acc = accuracy_score(gold, pred)
    f1 = f1_score(gold, pred, labels=LABELS, average="macro", zero_division=0)
    out = args.out or os.path.join(os.path.dirname(args.data), "pubmedqa_results.json")
    json.dump({"n": len(rows), "accuracy": round(acc, 4), "macro_f1": round(f1, 4),
               "rows": rows}, open(out, "w"), indent=2)

    print(f"PubMedQA (reasoning-required, n={len(rows)})")
    print(f"  accuracy : {acc:.3f}")
    print(f"  macro-F1 : {f1:.3f}")
    from collections import Counter
    print(f"  gold dist: {dict(Counter(gold))}")
    print(f"  pred dist: {dict(Counter(pred))}")
    print(f"  (published reference: GPT-4-class models ~0.75 accuracy on this split)")
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
