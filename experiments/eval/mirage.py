"""Public biomedical QA benchmarks via the MIRAGE/MedRAG suite (reviewer R1#10):
MedQA, MedMCQA, BioASQ, MMLU-Med, PubMedQA — multiple-choice accuracy, fully
automatic (no human rater). Situates our generation model against established,
externally-labeled benchmarks.

Data (download once into its own dir):
  mkdir -p experiments/data/mirage
  curl -sL -o experiments/data/mirage/benchmark.json \
    https://raw.githubusercontent.com/Teddy-XiongGZ/MIRAGE/main/benchmark.json

  python3 -I experiments/eval/mirage.py --dataset bioasq --n 200
  python3 -I experiments/eval/mirage.py --dataset medqa --n 200
"""
from __future__ import annotations
import os, sys, json, re, argparse, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import llm  # noqa: E402

SYSTEM = ("You are a medical exam expert. Choose the single best option. "
          "Respond with ONLY the option letter (e.g. A).")
REFS = {"medqa": "~0.78–0.86", "medmcqa": "~0.70–0.74", "bioasq": "~0.88–0.93",
        "mmlu": "~0.87–0.90", "pubmedqa": "~0.60–0.78"}


def pick(ans, letters):
    m = re.search(r"\b([A-E])\b", (ans or "").upper())
    return m.group(1) if (m and m.group(1) in letters) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="experiments/data/mirage/benchmark.json")
    ap.add_argument("--dataset", required=True,
                    choices=["medqa", "medmcqa", "bioasq", "mmlu", "pubmedqa"])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    items = list(json.load(open(args.data))[args.dataset].items())
    random.Random(args.seed).shuffle(items)
    items = items[: args.n]

    gold, pred, rows = [], [], []
    for qid, rec in items:
        opts = rec["options"]
        letters = list(opts.keys())
        block = "\n".join(f"{k}. {v}" for k, v in opts.items())
        user = f"Question: {rec['question']}\n\nOptions:\n{block}\n\nAnswer (letter only):"
        try:
            out = llm.chat(SYSTEM, user, max_tokens=2000)
        except Exception as e:
            out = f"(error {type(e).__name__})"
        p = pick(out, letters)
        g = rec["answer"].strip().upper()
        gold.append(g); pred.append(p or "?")
        rows.append({"id": qid, "gold": g, "pred": p, "correct": int(p == g)})

    acc = sum(r["correct"] for r in rows) / len(rows)
    out_path = os.path.join(os.path.dirname(args.data), f"mirage_{args.dataset}_results.json")
    json.dump({"dataset": args.dataset, "n": len(rows), "accuracy": round(acc, 4), "rows": rows},
              open(out_path, "w"), indent=2)
    print(f"MIRAGE/{args.dataset}: n={len(rows)}  accuracy={acc:.3f}  "
          f"(published GPT-4-class ref {REFS.get(args.dataset,'?')})")
    print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
