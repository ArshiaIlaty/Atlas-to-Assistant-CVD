"""Azure AI Search backend: build a hybrid (BM25 + vector) index of a PubMed
sample, and retrieve from it with the same interface as the local retriever.

This is the production retrieval path described in the paper. Documents carry the
text-embedding-3-large vector plus taxonomy metadata, so `AzureSearchRetriever`
is a drop-in replacement for `retrieval.HybridRetriever` in conditions.py.

Build (needs AZURE_SEARCH_ENDPOINT + AZURE_SEARCH_KEY and Azure OpenAI embeddings):
    python3 -I experiments/azure_search.py --build --sample 3000
"""
from __future__ import annotations
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex, SimpleField, SearchableField, SearchField, SearchFieldDataType,
    VectorSearch, HnswAlgorithmConfiguration, VectorSearchProfile,
)
from azure.search.documents.models import VectorizedQuery

import config, corpus, embeddings

VECTOR_DIM = 3072          # text-embedding-3-large
VECTOR_FIELD = "contentVector"
PROFILE = "hnsw-profile"


def _require_creds():
    if not (config.AZURE_SEARCH_ENDPOINT and config.AZURE_SEARCH_KEY):
        sys.exit("Set AZURE_SEARCH_ENDPOINT and AZURE_SEARCH_KEY (in .env) first.")
    return AzureKeyCredential(config.AZURE_SEARCH_KEY)


def _index_schema():
    return SearchIndex(
        name=config.AZURE_SEARCH_INDEX,
        fields=[
            SimpleField(name="pmid", type=SearchFieldDataType.String, key=True),
            SearchableField(name="title", type=SearchFieldDataType.String),
            SearchableField(name="abstract", type=SearchFieldDataType.String),
            SearchableField(name="journal", type=SearchFieldDataType.String),
            SimpleField(name="authors", type=SearchFieldDataType.String),
            SimpleField(name="year", type=SearchFieldDataType.String, filterable=True, facetable=True),
            SimpleField(name="domain", type=SearchFieldDataType.String, filterable=True, facetable=True),
            SimpleField(name="subdomain", type=SearchFieldDataType.String, filterable=True, facetable=True),
            SimpleField(name="topic", type=SearchFieldDataType.String, filterable=True, facetable=True),
            SearchField(name=VECTOR_FIELD,
                        type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                        searchable=True, vector_search_dimensions=VECTOR_DIM,
                        vector_search_profile_name=PROFILE),
        ],
        vector_search=VectorSearch(
            algorithms=[HnswAlgorithmConfiguration(name="hnsw")],
            profiles=[VectorSearchProfile(name=PROFILE, algorithm_configuration_name="hnsw")],
        ),
    )


def build_index(sample=3000, recreate=True):
    cred = _require_creds()
    if config.EMBED_BACKEND != "azure":
        print("WARN: EMBED_BACKEND is not 'azure'; index vectors must match query vectors.",
              file=sys.stderr)
    idx_client = SearchIndexClient(config.AZURE_SEARCH_ENDPOINT, cred)
    if recreate:
        try:
            idx_client.delete_index(config.AZURE_SEARCH_INDEX)
        except Exception:
            pass
    idx_client.create_index(_index_schema())
    print(f"created index '{config.AZURE_SEARCH_INDEX}'", file=sys.stderr)

    docs = corpus.load_corpus(limit=sample)
    print(f"embedding {len(docs)} docs with {config.EMBED_MODEL} (batched) ...", file=sys.stderr)
    vecs = embeddings.embed_cached([d["text"] for d in docs], f"azsample{sample}")
    # taxonomy tags reuse the SAME doc vectors — no per-doc API calls
    import taxonomy_tagger
    tagger = taxonomy_tagger.TaxonomyTagger()
    tags = tagger.tag_by_embeddings(vecs)
    for d, t in zip(docs, tags):
        d["taxonomy"] = t

    payload = []
    for d, v in zip(docs, vecs):
        if not d["pmid"]:
            continue
        payload.append({
            "pmid": d["pmid"], "title": d["title"], "abstract": d["abstract"],
            "journal": d["journal"], "authors": d["authors"], "year": d["year"],
            "domain": d["taxonomy"]["domain"], "subdomain": d["taxonomy"]["subdomain"],
            "topic": d["taxonomy"]["topic"], VECTOR_FIELD: v.tolist(),
        })
    sc = SearchClient(config.AZURE_SEARCH_ENDPOINT, config.AZURE_SEARCH_INDEX, cred)
    for i in range(0, len(payload), 500):
        sc.upload_documents(documents=payload[i:i + 500])
    print(f"uploaded {len(payload)} documents", file=sys.stderr)


class AzureSearchRetriever:
    def __init__(self):
        cred = _require_creds()
        self.sc = SearchClient(config.AZURE_SEARCH_ENDPOINT, config.AZURE_SEARCH_INDEX, cred)

    def search(self, query, k=None, mode="hybrid"):
        k = k or config.TOP_K
        kw = {"top": k, "select": ["pmid", "title", "abstract", "year", "journal",
                                    "domain", "subdomain", "topic"]}
        if mode in ("hybrid", "dense"):
            qv = embeddings.embed([query])[0].tolist()
            kw["vector_queries"] = [VectorizedQuery(vector=qv, k_nearest_neighbors=k,
                                                    fields=VECTOR_FIELD)]
        kw["search_text"] = None if mode == "dense" else query
        out = []
        for r in self.sc.search(**kw):
            out.append({
                "pmid": r["pmid"], "title": r.get("title", ""), "abstract": r.get("abstract", ""),
                "year": r.get("year", ""), "journal": r.get("journal", ""),
                "text": f"{r.get('title', '')}. {r.get('abstract', '')}",
                "taxonomy": {"domain": r.get("domain", ""), "subdomain": r.get("subdomain", ""),
                             "topic": r.get("topic", "")},
                "retrieval_score": r.get("@search.score"),
            })
        return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--sample", type=int, default=3000)
    ap.add_argument("--query", help="test a query against the existing index")
    ap.add_argument("--mode", default="hybrid", choices=["hybrid", "dense", "bm25"])
    args = ap.parse_args()
    if args.build:
        build_index(sample=args.sample)
    if args.query:
        for d in AzureSearchRetriever().search(args.query, mode=args.mode):
            print(f"{d['retrieval_score']:.3f}  [{d['pmid']}] {d['title'][:80]}  "
                  f"<{d['taxonomy']['domain']}>")
