"""The three experimental conditions, switchable behind one entry point.

    1. alone         -> GPT-4 with no retrieved context
    2. rag           -> GPT-4 + hybrid retrieval of PubMed abstracts (PMID-cited)
    3. rag_taxonomy  -> condition 2 + Domain/Subdomain/Topic taxonomy metadata

This is the switchable API the plan calls for (also usable by Hossein's broader
work). `run_condition` returns the answer plus everything needed to score it:
retrieved PMIDs, the exact context, and the taxonomy paths.
"""
from __future__ import annotations
import re
import time
import prompts
import llm

CONDITIONS = ["alone", "rag", "rag_taxonomy"]


def _citations(text):
    return sorted(set(re.findall(r"PMID:\s*(\d+)", text or "")))


def run_condition(question, condition, retriever=None, tagger=None, k=5,
                  retrieval_mode="hybrid", dry_run=False):
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}; choose from {CONDITIONS}")

    result = {"question": question, "condition": condition, "k": k,
              "retrieval_mode": retrieval_mode}
    t0 = time.perf_counter()

    if condition == "alone":
        result["retrieved_pmids"] = []
        result["answer"] = None if dry_run else llm.chat(prompts.ALONE_SYSTEM, question)
        result["citations"] = _citations(result["answer"])
        result["usage"] = None if dry_run else llm.LAST_USAGE
        result["latency_s"] = round(time.perf_counter() - t0, 3)
        return result

    docs = retriever.search(question, k=k, mode=retrieval_mode)
    use_tax = condition == "rag_taxonomy"
    if use_tax and tagger is not None and any("taxonomy" not in d for d in docs):
        # Local retriever returns untagged docs; Azure Search already carries taxonomy.
        docs = tagger.tag_docs(docs)
    result["retrieved_pmids"] = [d["pmid"] for d in docs]
    if use_tax:
        result["retrieved_taxonomy"] = [d.get("taxonomy") for d in docs]

    context = prompts.format_context(docs, include_taxonomy=use_tax)
    result["context"] = context
    system = prompts.RAG_TAXONOMY_SYSTEM if use_tax else prompts.RAG_SYSTEM
    user = prompts.rag_user_prompt(question, context)
    result["answer"] = None if dry_run else llm.chat(system, user)
    result["citations"] = _citations(result["answer"])
    result["usage"] = None if dry_run else llm.LAST_USAGE
    result["latency_s"] = round(time.perf_counter() - t0, 3)
    return result
