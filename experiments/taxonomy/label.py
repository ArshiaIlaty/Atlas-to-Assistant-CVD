"""GPT taxonomy labeling via the API (reviewer R1#13: replaces non-reproducible
UI prompting with versioned, logged API calls), then the hierarchical distribution
(paper Section 4.4) at BOTH the 7-domain (audit) granularity and the paper's
3-domain grouping.

The allowed hierarchy (Domain > Subdomain > Topic paths) is derived from the audited
taxonomy CSV, so the model must pick an existing path.

    python3 -I experiments/taxonomy/label.py --n 2000 --random --out taxonomy_labels_random2000.csv
    python3 -I experiments/taxonomy/label.py --n 0 --out taxonomy_labels_full.csv   # all 20,881
"""
from __future__ import annotations
import os, sys, json, re, argparse, random
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd  # noqa: E402
import config, corpus, llm  # noqa: E402

TAXCSV = os.path.join(config.REPO, "Taxonomy", "API_Inconsistency_o1.csv")
RESULTS = os.path.join(config.REPO, "experiments", "results")

SYSTEM = ("You are a cardiovascular research librarian assigning papers to a fixed "
          "three-level taxonomy. Choose the SINGLE best path from the provided list. "
          "Return strict JSON: {\"domain\":..., \"subdomain\":..., \"topic\":...} using "
          "labels EXACTLY as listed.")

# Map the 7 audit-taxonomy domains to the paper's 3-domain Section-4.4 grouping.
# Unmapped labels fall back to Cardiovascular Diseases & Interventions.
DOMAIN3 = {
    "Basic / Translational Research": "Basic & Translational Cardiovascular Science",
    "Clinical Cardiology": "Cardiovascular Diseases & Interventions",
    "Diagnostics & Imaging": "Cardiovascular Diseases & Interventions",
    "Interventional & Surgical": "Cardiovascular Diseases & Interventions",
    "Pharmacology & Therapeutics": "Cardiovascular Diseases & Interventions",
    "Epidemiology & Risk Factors": "Population Health, Prevention & Policy",
}
D3_DEFAULT = "Cardiovascular Diseases & Interventions"


def allowed_paths():
    d = pd.read_csv(TAXCSV, dtype=str, keep_default_na=False)
    tree = {}
    for _, r in d[["Cat_L1", "Cat_L2", "Cat_L3"]].drop_duplicates().iterrows():
        tree.setdefault(r.Cat_L1, {}).setdefault(r.Cat_L2, []).append(r.Cat_L3)
    lines = []
    for l1, subs in tree.items():
        lines.append(f"- {l1}")
        for l2, tops in subs.items():
            lines.append(f"    - {l2}: {', '.join(sorted(set(tops)))}")
    return "\n".join(lines)


def label_one(text, hierarchy_str):
    user = f"TAXONOMY:\n{hierarchy_str}\n\nPAPER:\n{text[:3000]}\n\nAssign the best path as JSON:"
    try:
        raw = llm.chat(SYSTEM, user, max_tokens=2000)
        m = re.search(r"\{.*\}", raw or "", re.S)
        j = json.loads(m.group(0)) if m else {}
        return {"domain": j.get("domain", ""), "subdomain": j.get("subdomain", ""),
                "topic": j.get("topic", "")}
    except Exception:
        return {"domain": "", "subdomain": "", "topic": ""}


def distribution(df):
    print(f"\n=== Section 4.4 distribution (n={len(df)}) ===")
    print("7-domain (audit taxonomy):")
    for k, v in (df["domain"].value_counts(normalize=True) * 100).round(2).items():
        print(f"  {v:>6}%  {k}")
    d3 = df["domain"].map(lambda x: DOMAIN3.get(x, D3_DEFAULT))
    print("3-domain (paper grouping) — paper ref: 39.0 / 37.6 / 23.4%:")
    for k, v in (d3.value_counts(normalize=True) * 100).round(2).items():
        print(f"  {v:>6}%  {k}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=os.path.join(config.REPO, "ETL", "pubmed_corpus_stratified.csv"))
    ap.add_argument("--n", type=int, default=2000, help="0 = all records")
    ap.add_argument("--random", action="store_true", help="random sample across years (vs first-N)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default="taxonomy_labels.csv")
    args = ap.parse_args()
    if config.llm_backend() == "none":
        sys.exit("needs credentials")

    out = args.out if os.path.isabs(args.out) else os.path.join(RESULTS, args.out)
    hierarchy = allowed_paths()
    docs = corpus.load_corpus(path=args.corpus)
    if args.n and args.n < len(docs):
        docs = random.Random(args.seed).sample(docs, args.n) if args.random else docs[: args.n]
    print(f"labeling {len(docs)} records ({'random' if args.random else 'head'}), "
          f"{args.workers} workers -> {out}", file=sys.stderr, flush=True)

    results = [None] * len(docs)
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(label_one, d["text"], hierarchy): i for i, d in enumerate(docs)}
        for fut in as_completed(futs):
            results[futs[fut]] = fut.result()
            done += 1
            if done % 50 == 0:
                part = [{"pmid": docs[j]["pmid"], "year": docs[j]["year"], **results[j]}
                        for j in range(len(docs)) if results[j]]
                pd.DataFrame(part).to_csv(out, index=False)
                print(f"  {done}/{len(docs)}", file=sys.stderr, flush=True)

    df = pd.DataFrame([{"pmid": d["pmid"], "year": d["year"], **r} for d, r in zip(docs, results)])
    df.to_csv(out, index=False)
    print(f"wrote {out} ({len(df)})", file=sys.stderr, flush=True)
    distribution(df)


if __name__ == "__main__":
    main()
