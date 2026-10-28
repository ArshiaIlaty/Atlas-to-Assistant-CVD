"""GPT taxonomy labeling via the API (reviewer R1#13: replaces the non-reproducible
UI-based prompting of GPT-4.5/o1 with versioned, logged API calls), then computes
the hierarchical distribution (paper Section 4.4: Domain / Subdomain %).

The allowed hierarchy (the set of Domain > Subdomain > Topic paths) is derived from
the audited taxonomy CSV, so the model must assign each record to an existing path.

    python3 -I experiments/taxonomy/label.py --n 1000
    python3 -I experiments/taxonomy/label.py --n 5 --smoke     # quick check
"""
from __future__ import annotations
import os, sys, json, re, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd  # noqa: E402
import config, corpus, llm  # noqa: E402

TAXCSV = os.path.join(config.REPO, "Taxonomy", "API_Inconsistency_o1.csv")
OUT = os.path.join(config.REPO, "experiments", "results", "taxonomy_labels.csv")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

SYSTEM = ("You are a cardiovascular research librarian assigning papers to a fixed "
          "three-level taxonomy. Choose the SINGLE best path from the provided list. "
          "Return strict JSON: {\"domain\":..., \"subdomain\":..., \"topic\":...} using "
          "labels EXACTLY as listed.")


def allowed_paths():
    d = pd.read_csv(TAXCSV, dtype=str, keep_default_na=False)
    paths = (d[["Cat_L1", "Cat_L2", "Cat_L3"]].drop_duplicates()
             .sort_values(["Cat_L1", "Cat_L2", "Cat_L3"]))
    tree = {}
    for _, r in paths.iterrows():
        tree.setdefault(r.Cat_L1, {}).setdefault(r.Cat_L2, []).append(r.Cat_L3)
    lines = []
    for l1, subs in tree.items():
        lines.append(f"- {l1}")
        for l2, tops in subs.items():
            lines.append(f"    - {l2}: {', '.join(sorted(set(tops)))}")
    return "\n".join(lines), paths


def label_one(text, hierarchy_str):
    user = f"TAXONOMY:\n{hierarchy_str}\n\nPAPER:\n{text[:3000]}\n\nAssign the best path as JSON:"
    raw = llm.chat(SYSTEM, user, max_tokens=2000)
    m = re.search(r"\{.*\}", raw or "", re.S)
    try:
        j = json.loads(m.group(0)) if m else {}
        return {"domain": j.get("domain", ""), "subdomain": j.get("subdomain", ""),
                "topic": j.get("topic", "")}
    except json.JSONDecodeError:
        return {"domain": "", "subdomain": "", "topic": ""}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=os.path.join(config.REPO, "ETL", "pubmed_corpus_stratified.csv"))
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if config.llm_backend() == "none":
        sys.exit("needs Azure/OpenAI credentials")
    hierarchy_str, _ = allowed_paths()
    docs = corpus.load_corpus(path=args.corpus, limit=args.n)
    print(f"labeling {len(docs)} records with {config.AZURE_CHAT_DEPLOYMENT} ...", file=sys.stderr, flush=True)

    rows = []
    for i, d in enumerate(docs):
        lab = label_one(d["text"], hierarchy_str)
        rows.append({"pmid": d["pmid"], "year": d["year"], **lab})
        if i % 25 == 0:
            pd.DataFrame(rows).to_csv(OUT, index=False)   # checkpoint
            print(f"  {i+1}/{len(docs)}", file=sys.stderr, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(f"wrote {OUT} ({len(df)} labeled)", file=sys.stderr, flush=True)

    # Section 4.4 distribution
    print("\n=== Section 4.4 — Hierarchical distribution (API GPT labeling) ===")
    print(f"Domain % (n={len(df)}):")
    for k, v in (df["domain"].value_counts(normalize=True) * 100).round(2).items():
        print(f"  {v:>6}%  {k}")
    print("Top Subdomain %:")
    for k, v in (df["subdomain"].value_counts(normalize=True) * 100).round(2).head(8).items():
        print(f"  {v:>6}%  {k}")


if __name__ == "__main__":
    main()
