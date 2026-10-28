"""Semantic topic discovery: Sentence-BERT -> UMAP -> HDBSCAN -> TF-IDF labels,
reproducing the paper's Stage 1 and the metrics reviewers asked about.

Per field (title and abstract, clustered independently as in the paper):
  - all-MiniLM-L6-v2 embeddings (cached)
  - UMAP(n_components=5, random_state=42) -> HDBSCAN(min_cluster_size=10)
  - top-3 bigram/trigram TF-IDF descriptors per cluster
  - top-5 clusters by size, with per-year counts
  - lexical non-correspondence rate (paper 4.1: ~15% titles / ~9% abstracts)
  - cluster-stability metrics for R2#7: silhouette + bootstrap Adjusted Rand Index
  - a 2-D UMAP projection for the scatter visualization

Outputs -> experiments/results/clustering/{field}_assignments.csv, 2d coords, and
a combined summary.json.

    EMBED_BACKEND=local python3 -I experiments/clustering/cluster.py [--bootstrap 3]
"""
from __future__ import annotations
import os, sys, json, argparse
os.environ.setdefault("EMBED_BACKEND", "local")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # experiments/
import numpy as np, pandas as pd                       # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.metrics import silhouette_score, adjusted_rand_score  # noqa: E402
import umap                                             # noqa: E402
from hdbscan import HDBSCAN                             # noqa: E402
import config, corpus, embeddings                       # noqa: E402

OUTDIR = os.path.join(config.REPO, "experiments", "results", "clustering")
os.makedirs(OUTDIR, exist_ok=True)
DOMAIN_STOP = ["heart", "cardiovascular", "disease", "diseases", "cardiac", "patients", "study"]


def descriptors(texts, labels, topn=3):
    """Top bigram/trigram TF-IDF terms per cluster."""
    vec = TfidfVectorizer(ngram_range=(2, 3), min_df=5, stop_words="english")
    X = vec.fit_transform(texts)
    terms = np.array(vec.get_feature_names_out())
    out = {}
    for c in sorted(set(labels)):
        if c == -1:
            continue
        rows = np.where(labels == c)[0]
        mean = np.asarray(X[rows].mean(axis=0)).ravel()
        out[c] = terms[mean.argsort()[::-1][:topn]].tolist()
    return out


def lexical_noncorrespondence(texts, labels, desc):
    """Fraction of clustered records whose source text contains NONE of its
    cluster's descriptor phrases verbatim."""
    miss = tot = 0
    for t, c in zip(texts, labels):
        if c == -1:
            continue
        tot += 1
        low = t.lower()
        if not any(p in low for p in desc.get(c, [])):
            miss += 1
    return round(100 * miss / tot, 1) if tot else float("nan")


def cluster_field(field, docs, do_stability=False, bootstrap=3):
    texts = [d[field] for d in docs]
    years = pd.to_numeric(pd.Series([d["year"] for d in docs]), errors="coerce")
    emb = embeddings.embed_cached(texts, f"{field}s")
    red = umap.UMAP(n_components=5, random_state=42).fit_transform(emb)
    labels = HDBSCAN(min_cluster_size=10).fit_predict(red)

    desc = descriptors(texts, labels)
    sizes = pd.Series(labels)[lambda s: s != -1].value_counts()
    top5 = sizes.head(5).index.tolist()
    lex = lexical_noncorrespondence(texts, labels, desc)
    noise = float(np.mean(labels == -1))
    nz = labels != -1
    sil = float(silhouette_score(red[nz], labels[nz])) if nz.sum() > len(set(labels)) else float("nan")

    # per-year counts for the top-5 clusters
    df = pd.DataFrame({"pmid": [d["pmid"] for d in docs], "year": years, "cluster": labels})
    df.to_csv(os.path.join(OUTDIR, f"{field}_assignments.csv"), index=False)
    annual = (df[df.cluster.isin(top5)].dropna(subset=["year"])
              .assign(year=lambda x: x.year.astype(int))
              .groupby(["year", "cluster"]).size().unstack(fill_value=0))
    annual.to_csv(os.path.join(OUTDIR, f"{field}_annual.csv"))

    summary = {
        "field": field, "n": len(docs), "n_clusters": int(len(set(labels)) - (1 if -1 in labels else 0)),
        "noise_fraction": round(noise, 3), "silhouette": round(sil, 3),
        "lexical_noncorrespondence_pct": lex,
        "top5": [{"cluster": int(c), "size": int(sizes[c]), "label": " / ".join(desc.get(c, []))}
                 for c in top5],
    }

    if do_stability:
        aris = []
        rng = np.random.default_rng(42)
        for _ in range(bootstrap):
            idx = rng.choice(len(emb), size=int(0.9 * len(emb)), replace=False)
            r = umap.UMAP(n_components=5, random_state=None).fit_transform(emb[idx])
            lb = HDBSCAN(min_cluster_size=10).fit_predict(r)
            aris.append(adjusted_rand_score(labels[idx], lb))
        summary["bootstrap_ARI_mean"] = round(float(np.mean(aris)), 3)
        summary["bootstrap_ARI_runs"] = [round(a, 3) for a in aris]

    if field == "abstract":  # 2-D projection for the scatter viz
        xy = umap.UMAP(n_components=2, random_state=42).fit_transform(emb)
        pd.DataFrame({"x": xy[:, 0], "y": xy[:, 1], "cluster": labels,
                      "top5": [c if c in top5 else -1 for c in labels]}).to_csv(
            os.path.join(OUTDIR, "abstract_umap2d.csv"), index=False)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=os.path.join(config.REPO, "ETL", "pubmed_corpus_stratified.csv"))
    ap.add_argument("--bootstrap", type=int, default=3)
    args = ap.parse_args()
    docs = corpus.load_corpus(path=args.corpus)
    print(f"clustering {len(docs)} docs", file=sys.stderr, flush=True)
    summaries = {}
    for field in ["title", "abstract"]:
        print(f"  field={field} ...", file=sys.stderr, flush=True)
        summaries[field] = cluster_field(field, docs, do_stability=(field == "abstract"),
                                         bootstrap=args.bootstrap)
        print(json.dumps(summaries[field], indent=2), file=sys.stderr, flush=True)
    json.dump(summaries, open(os.path.join(OUTDIR, "summary.json"), "w"), indent=2)
    print("wrote", os.path.join(OUTDIR, "summary.json"), file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
