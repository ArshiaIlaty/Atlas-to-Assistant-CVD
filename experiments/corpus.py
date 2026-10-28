"""Load and normalize the PubMed corpus for retrieval."""
from __future__ import annotations
import pandas as pd
import config

_KEEP = ["PMID", "Title", "Authors", "Year", "Journal", "Abstract"]


def load_corpus(path=None, require_abstract=True, limit=None):
    """Return a list of doc dicts with a combined 'text' field for retrieval."""
    path = path or config.CORPUS_CSV
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    for c in _KEEP:
        if c not in df.columns:
            df[c] = ""
    if require_abstract:
        df = df[df["Abstract"].str.strip().str.len() > 0]
    df = df.drop_duplicates(subset=["PMID"]).reset_index(drop=True)
    if limit:
        df = df.head(limit)
    docs = []
    for _, r in df.iterrows():
        title = r["Title"].strip()
        abstract = r["Abstract"].strip()
        docs.append({
            "pmid": r["PMID"].strip(),
            "title": title,
            "authors": r["Authors"].strip(),
            "year": r["Year"].strip(),
            "journal": r["Journal"].strip(),
            "abstract": abstract,
            "text": f"{title}. {abstract}",
        })
    return docs
