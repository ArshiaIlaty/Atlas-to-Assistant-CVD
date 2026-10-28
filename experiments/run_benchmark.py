"""Run the benchmark questions through one or all three conditions.

Examples
--------
# Offline: inspect what each condition retrieves, no API key needed
python3 -I experiments/run_benchmark.py --condition rag --dry-run --limit 3

# Full generation once a key is set (OPENAI_API_KEY or AZURE_OPENAI_*)
python3 -I experiments/run_benchmark.py --condition all --k 5 --out results/run1.json

# Retrieval ablation (reviewer R1#5 / R2#6)
python3 -I experiments/run_benchmark.py --condition rag --retrieval-mode bm25 --dry-run
"""
from __future__ import annotations
import os, sys, json, argparse, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config, corpus, conditions  # noqa: E402


def load_questions(limit=None):
    import pandas as pd
    df = pd.read_excel(config.BENCHMARK_XLSX, sheet_name=0, header=0)
    qs = [str(q).strip() for q in df.iloc[:, 0].tolist() if str(q).strip() and str(q) != "nan"]
    return qs[:limit] if limit else qs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", choices=conditions.CONDITIONS + ["all"], default="all")
    ap.add_argument("--backend", choices=["auto", "local", "azure"], default="auto",
                    help="retrieval backend; auto = azure if AZURE_SEARCH_* set, else local")
    ap.add_argument("--k", type=int, default=config.TOP_K)
    ap.add_argument("--retrieval-mode", choices=["hybrid", "dense", "bm25"], default="hybrid")
    ap.add_argument("--limit", type=int, default=None, help="use first N questions")
    ap.add_argument("--corpus-limit", type=int, default=None, help="index only first N docs (dev)")
    ap.add_argument("--dry-run", action="store_true", help="retrieval only, no LLM call")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    conds = conditions.CONDITIONS if args.condition == "all" else [args.condition]
    needs_retrieval = any(c != "alone" for c in conds)
    needs_tax = "rag_taxonomy" in conds

    if not args.dry_run and config.llm_backend() == "none":
        sys.exit("No LLM credentials set. Add OPENAI_API_KEY or AZURE_OPENAI_* , or pass --dry-run.")

    backend = args.backend
    if backend == "auto":
        backend = "azure" if config.AZURE_SEARCH_ENDPOINT and config.AZURE_SEARCH_KEY else "local"

    retriever = tagger = None
    if needs_retrieval:
        if backend == "azure":
            print("Using Azure AI Search retriever", file=sys.stderr)
            import azure_search
            retriever = azure_search.AzureSearchRetriever()
        else:
            print("Loading corpus + building local retriever ...", file=sys.stderr)
            docs = corpus.load_corpus(limit=args.corpus_limit)
            print(f"  {len(docs)} docs with abstracts", file=sys.stderr)
            import retrieval
            retriever = retrieval.HybridRetriever(
                docs, cache_key=f"corpus{args.corpus_limit or 'all'}")
    if needs_tax and backend != "azure":
        # Azure docs already carry taxonomy; only the local path needs the tagger.
        import taxonomy_tagger
        tagger = taxonomy_tagger.TaxonomyTagger()

    questions = load_questions(limit=args.limit)
    print(f"Running {len(questions)} questions x {len(conds)} condition(s)", file=sys.stderr)

    out = []
    t0 = time.time()
    for qi, q in enumerate(questions):
        for c in conds:
            r = conditions.run_condition(
                q, c, retriever=retriever, tagger=tagger, k=args.k,
                retrieval_mode=args.retrieval_mode, dry_run=args.dry_run)
            r["qid"] = qi
            out.append(r)
        print(f"  [{qi + 1}/{len(questions)}] done", file=sys.stderr)

    path = args.out or os.path.join(
        config.RESULTS_DIR,
        f"run_{args.condition}_{args.retrieval_mode}_{'dry' if args.dry_run else 'gen'}.json")
    if not os.path.isabs(path):
        path = os.path.join(config.REPO, "experiments", path)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {path}  ({len(out)} records, {time.time() - t0:.1f}s)", file=sys.stderr)


if __name__ == "__main__":
    main()
