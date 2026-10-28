"""Hybrid retrieval: BM25 (lexical) + dense (SBERT) fused with Reciprocal Rank
Fusion. Retrieval mode is switchable (hybrid | dense | bm25) so the reviewer-
requested retrieval ablation (R1#5 / R2#6) is a one-flag change.

Dense search uses brute-force cosine over L2-normalized vectors (exact; fast at
this corpus size). A FAISS index is used automatically if the corpus is large.
Azure AI Search can replace this class in production (see README).
"""
from __future__ import annotations
import re
import numpy as np
import config
import embeddings

_TOK = re.compile(r"[a-z0-9]+")


def tokenize(text):
    return _TOK.findall(text.lower())


def _rrf(rankings, k0, top_k):
    """Reciprocal Rank Fusion over several ranked ID lists."""
    scores = {}
    for ranked in rankings:
        for rank, idx in enumerate(ranked):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k0 + rank)
    order = sorted(scores, key=scores.get, reverse=True)[:top_k]
    return [(i, scores[i]) for i in order]


class HybridRetriever:
    def __init__(self, docs, cache_key="corpus"):
        from rank_bm25 import BM25Okapi
        self.docs = docs
        self.bm25 = BM25Okapi([tokenize(d["text"]) for d in docs])
        self.emb = embeddings.embed_cached([d["text"] for d in docs], cache_key)
        self._faiss = None
        if len(docs) > 50000:
            import faiss
            self._faiss = faiss.IndexFlatIP(self.emb.shape[1])
            self._faiss.add(self.emb)

    def _dense_rank(self, query, n):
        q = embeddings.embed([query])[0]
        if self._faiss is not None:
            _, idx = self._faiss.search(q[None, :], n)
            return list(idx[0])
        sims = self.emb @ q
        return list(np.argsort(-sims)[:n])

    def _bm25_rank(self, query, n):
        scores = self.bm25.get_scores(tokenize(query))
        return list(np.argsort(-scores)[:n])

    def search(self, query, k=None, mode="hybrid"):
        k = k or config.TOP_K
        pool = max(k * 10, 50)
        if mode == "dense":
            ranked = [(i, None) for i in self._dense_rank(query, k)]
        elif mode == "bm25":
            ranked = [(i, None) for i in self._bm25_rank(query, k)]
        else:  # hybrid
            ranked = _rrf([self._dense_rank(query, pool), self._bm25_rank(query, pool)],
                          config.RRF_K0, k)
        out = []
        for i, score in ranked:
            d = dict(self.docs[int(i)])
            d["retrieval_score"] = score
            out.append(d)
        return out
