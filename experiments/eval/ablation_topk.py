"""Retrieval ablation (R1#5 / R2#6) and top-k sensitivity (R1#11).

Generation-only (no judge) so it stays cheap: for the RAG condition it varies the
retrieval mode (bm25 / dense / hybrid) and the retrieval depth k, and reports the
automatic citation metrics per variant — share of answers that cite, refusal rate,
and citation correctness (cited PMIDs actually present in the retrieved context).

    python3 -I experiments/eval/ablation_topk.py --backend azure --n 50
"""
from __future__ import annotations
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config, conditions  # noqa: E402
import metrics as M        # noqa: E402

REFUSE = "the information is not available"


def load_questions(n):
    import pandas as pd
    df = pd.read_excel(config.BENCHMARK_XLSX, sheet_name=0, header=0)
    qs = [str(q).strip() for q in df.iloc[:, 0].tolist() if str(q).strip() and str(q) != "nan"]
    return qs[:n] if n else qs


def summarize(recs):
    cc = [M.citation_correctness(r["answer"], r.get("context", ""))["correctness"] for r in recs]
    cc = [x for x in cc if x == x]
    cited = sum(1 for r in recs if r.get("citations"))
    refused = sum(1 for r in recs if REFUSE in (r.get("answer") or "").lower())
    return {"n": len(recs), "cited": cited, "refused": refused,
            "mean_citation_correctness": round(sum(cc) / len(cc), 3) if cc else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["azure", "local"], default="azure")
    ap.add_argument("--n", type=int, default=50)
    args = ap.parse_args()
    if config.llm_backend() == "none":
        sys.exit("needs credentials")

    if args.backend == "azure":
        import azure_search
        retr = azure_search.AzureSearchRetriever(); tagger = None
    else:
        import corpus, retrieval, taxonomy_tagger
        retr = retrieval.HybridRetriever(corpus.load_corpus(), cache_key="corpusall")
        tagger = taxonomy_tagger.TaxonomyTagger()

    qs = load_questions(args.n)
    out = {"ablation": {}, "topk": {}}

    print("=== Retrieval ablation (RAG, k=5) ===")
    for mode in ["hybrid", "bm25", "dense"]:
        recs = [conditions.run_condition(q, "rag", retriever=retr, tagger=tagger, k=5,
                                         retrieval_mode=mode) for q in qs]
        out["ablation"][mode] = summarize(recs)
        print(f"  {mode:<7} {out['ablation'][mode]}")

    print("=== Top-k sensitivity (RAG, hybrid) ===")
    for k in [1, 3, 5, 10]:
        recs = [conditions.run_condition(q, "rag", retriever=retr, tagger=tagger, k=k,
                                         retrieval_mode="hybrid") for q in qs]
        out["topk"][k] = summarize(recs)
        print(f"  k={k:<3} {out['topk'][k]}")

    path = os.path.join(config.RESULTS_DIR, "ablation_topk.json")
    json.dump(out, open(path, "w"), indent=2)
    print("wrote", path)


if __name__ == "__main__":
    main()
