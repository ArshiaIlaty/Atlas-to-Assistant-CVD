"""Attach a Domain/Subdomain/Topic taxonomy path to arbitrary documents.

The audited taxonomy exists only for the 200-record sample (Taxonomy/*.csv) and
does not title-match the retrieval corpus, so labels are propagated by embedding
nearest-neighbor: embed the labeled abstracts once, then tag any retrieved doc
with the path of its closest labeled neighbor. This provides the taxonomy
metadata used by the RAG+taxonomy condition and keeps the ±taxonomy ablation honest.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import config
import embeddings


class TaxonomyTagger:
    def __init__(self, path=None):
        path = path or config.TAXONOMY_CSV
        df = pd.read_csv(path, dtype=str, keep_default_na=False).drop_duplicates(subset=["Title"])
        df = df[df["Abstract"].str.strip().str.len() > 0].reset_index(drop=True)
        self.l1 = df["Cat_L1"].tolist()
        self.l2 = df["Cat_L2"].tolist()
        self.l3 = df["Cat_L3"].tolist()
        texts = (df["Title"].str.strip() + ". " + df["Abstract"].str.strip()).tolist()
        self.emb = embeddings.embed_cached(texts, "taxonomy")

    def _path(self, i):
        return {"domain": self.l1[i], "subdomain": self.l2[i], "topic": self.l3[i]}

    def tag(self, text):
        q = embeddings.embed([text])[0]
        return self._path(int(np.argmax(self.emb @ q)))

    def tag_by_embeddings(self, doc_vecs):
        """Vectorized NN tagging from precomputed doc embeddings (no API calls).
        doc_vecs must be in the same embedding space as the taxonomy (same model)."""
        nn = np.asarray(doc_vecs) @ self.emb.T
        return [self._path(int(i)) for i in nn.argmax(1)]

    def tag_docs(self, docs):
        # few docs (e.g. a query's top-k): embed each. For bulk tagging with
        # precomputed vectors use tag_by_embeddings() instead.
        for d in docs:
            d["taxonomy"] = self.tag(d["text"])
        return docs
