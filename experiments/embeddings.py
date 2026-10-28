"""Text embedding with a local SBERT default and an optional API backend.

Local backend (all-MiniLM-L6-v2) needs no API key and matches the model the
paper used for clustering, so retrieval and taxonomy tagging run offline today.
Swap to Azure/OpenAI text-embedding-3-large by setting EMBED_BACKEND + EMBED_MODEL.
"""
from __future__ import annotations
import os
import numpy as np
import config

_LOCAL_MODEL = None


def _local(texts):
    global _LOCAL_MODEL
    if _LOCAL_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _LOCAL_MODEL = SentenceTransformer(config.EMBED_MODEL)
    return np.asarray(_LOCAL_MODEL.encode(list(texts), batch_size=64,
                                          show_progress_bar=False, normalize_embeddings=True),
                      dtype=np.float32)


def _api(texts, batch=64):
    """Batched API embeddings (avoids per-request size/token limits)."""
    import llm
    client, _ = llm.get_client()
    model = config.EMBED_MODEL
    texts = list(texts)
    out = []
    for i in range(0, len(texts), batch):
        resp = client.embeddings.create(model=model, input=texts[i:i + batch])
        out.extend(d.embedding for d in resp.data)
    vecs = np.asarray(out, dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.clip(norms, 1e-8, None)


def embed(texts):
    """Return L2-normalized embedding matrix (n, d) for a list of strings."""
    if config.EMBED_BACKEND == "local":
        return _local(texts)
    return _api(texts)


def embed_cached(texts, cache_key):
    """Embed with an on-disk cache keyed by (backend, model, count, cache_key)."""
    fn = os.path.join(
        config.CACHE_DIR,
        f"emb_{config.EMBED_BACKEND}_{config.EMBED_MODEL.replace('/', '_')}_{len(texts)}_{cache_key}.npy",
    )
    if os.path.exists(fn):
        return np.load(fn)
    vecs = embed(texts)
    np.save(fn, vecs)
    return vecs
