# Revision experiments — three-condition cardiovascular Q&A

Rebuilds the validation pipeline lost with the old Azure deployment, and extends
the paper's 2-condition comparison into the three conditions planned for the
major revision (HISC-D-26-03329):

| # | Condition      | Description                                             |
|---|----------------|---------------------------------------------------------|
| 1 | `alone`        | GPT-4 with no retrieved context                         |
| 2 | `rag`          | GPT-4 + hybrid retrieval of PubMed abstracts (PMID-cited) |
| 3 | `rag_taxonomy` | condition 2 + Domain/Subdomain/Topic taxonomy metadata  |

It is designed to **run offline today** (local SBERT retrieval, no API key) and
to **swap in Azure** (AI Search + Azure OpenAI GPT-4.1) by setting env vars only.

## Layout
```
experiments/
  config.py          env-driven paths + backends
  corpus.py          load/normalize ETL/pubmed_dataset.csv
  embeddings.py      SBERT (local) or API embeddings, cached to .cache/
  retrieval.py       BM25 + dense fused (RRF); mode = hybrid|dense|bm25  (ablation)
  taxonomy_tagger.py nearest-neighbor Domain/Subdomain/Topic tagging
  prompts.py         the paper's A.2 prompt + taxonomy variant + alone prompt
  conditions.py      the switchable 3-condition API  (run_condition)
  llm.py             OpenAI / Azure OpenAI chat
  run_benchmark.py   CLI: run questions x conditions -> results/*.json
  eval/
    stats.py           Wilcoxon, bootstrap CIs, Cohen's dz, Cohen's kappa
    analyze_existing.py recompute Table 3 WITH inferential stats (runs now)
    metrics.py         citation correctness (no key) + LLM-judge (needs key)
    judge_run.py       score a run with the LLM judge + compare all 3 conditions
```

## LLM-as-judge (scalable complement to human raters)
```bash
# score a generated run and get the 3-way pairwise stats
python3 -I experiments/eval/judge_run.py --results experiments/results/run_all_hybrid_gen.json
# keyless: citation-correctness only (no judge model)
python3 -I experiments/eval/judge_run.py --results <run.json> --no-llm
```
The judge is **blind to condition**. Use a **different model family** as judge than
the generator (`OPENAI_JUDGE_MODEL` / `AZURE_OPENAI_JUDGE_DEPLOYMENT`) to curb
self-preference bias, and **validate it against the blinded human raters** on a
subset with `stats.cohens_kappa` before relying on it — the reviewers specifically
objected to LLM-judging-LLM without a human gold standard (R1#1 / R1#7 / R2#1).

## Quick start
```bash
pip install -r experiments/requirements.txt

# 1) Reviewer deliverable you can run right now (R1#2/R2#2): inferential stats
#    on the existing 50-question scores.
python3 -I experiments/eval/analyze_existing.py

# 2) Offline: see what each condition retrieves (no API key)
python3 -I experiments/run_benchmark.py --condition all --dry-run --limit 3

# 3) Retrieval ablation (R1#5 / R2#6): hybrid vs dense vs bm25
python3 -I experiments/run_benchmark.py --condition rag --retrieval-mode bm25 --dry-run
```

## Generation + judging (needs credentials)
Set **either** OpenAI **or** Azure OpenAI, then drop `--dry-run`:
```bash
# OpenAI
export OPENAI_API_KEY=sk-...            # OPENAI_CHAT_MODEL defaults to gpt-4.1
# Azure OpenAI (matches the paper's deployment)
export AZURE_OPENAI_ENDPOINT=https://<resource>.openai.azure.com
export AZURE_OPENAI_API_KEY=...
export AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-4.1

python3 -I experiments/run_benchmark.py --condition all --k 5 --out results/run1.json
```

## Azure AI Search (production retrieval) — implemented
`experiments/azure_search.py` builds a hybrid (BM25 + `text-embedding-3-large`
vector) index with taxonomy metadata, and `AzureSearchRetriever` is a drop-in for
the local retriever. After setting `AZURE_SEARCH_ENDPOINT` + `AZURE_SEARCH_KEY`:
```bash
# build the index from a 3k PubMed sample
python3 -I experiments/azure_search.py --build --sample 3000
# sanity-check a query
python3 -I experiments/azure_search.py --query "statins in heart failure" --mode hybrid
# run the benchmark against Azure Search (auto-selected when AZURE_SEARCH_* is set)
python3 -I experiments/run_benchmark.py --condition all --backend azure --out results/azure_run.json
```
`--backend {auto,local,azure}` controls retrieval; `auto` picks Azure when its
env vars are present, else the local index. The three APIs in `conditions.py` are
backend-agnostic, so nothing else changes.

## What still needs data / decisions (not code)
- **2nd blinded rater** + inter-rater kappa (`eval/stats.cohens_kappa`) — R1#1/R2#1.
- **External / clinician-authored questions** to reduce benchmark circularity — R1#3/R2#3.
- **Azure credentials** for GPT-4.1 generation + the AI Search index.
- Final **question-set size** (plan says 10–100) and the **blind-survey platform**.

## Env vars
`CORPUS_CSV`, `TAXONOMY_CSV`, `EMBED_BACKEND` (local|openai|azure), `EMBED_MODEL`,
`TOP_K`, `RRF_K0`, `OPENAI_API_KEY`, `OPENAI_CHAT_MODEL`, `AZURE_OPENAI_ENDPOINT`,
`AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_API_VERSION`, `AZURE_OPENAI_CHAT_DEPLOYMENT`,
`AZURE_SEARCH_ENDPOINT`, `AZURE_SEARCH_KEY`, `AZURE_SEARCH_INDEX`.
