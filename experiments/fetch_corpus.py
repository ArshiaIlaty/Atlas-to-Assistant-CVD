"""Fetch the FULL cardiovascular PubMed corpus (2002-present) via Entrez history
paging, keeping only records with abstracts (matches the paper's analytical corpus).

Robust for ~100k+ records: uses usehistory + retstart paging, retries on transient
errors, and checkpoints the CSV periodically so a mid-run failure isn't fatal.

    NCBI_EMAIL=you@uni.edu NCBI_API_KEY=xxxx \
        python3 -I experiments/fetch_corpus.py --out ETL/pubmed_full_2002_2025.csv

An NCBI_API_KEY raises the rate limit from 3 to 10 req/s (much faster + fewer 429s).
"""
from __future__ import annotations
import os, sys, time, argparse, importlib.util
import pandas as pd
from Bio import Entrez

# reuse the repo's parsing helpers
_spec = importlib.util.spec_from_file_location(
    "pubmed_script", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                  "ETL", "pubmed_script.py"))
_ps = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ps)

BASE = '(("Cardiovascular Diseases"[MeSH] OR "Heart Diseases"[MeSH]))'
QUERY = f'{BASE} AND (2002:2025[dp])'
COLS = ["PMID", "Title", "Authors", "Year", "Journal", "Abstract"]


def _parse(records):
    rows = []
    for r in records:
        med = r.get("MedlineCitation", {}); art = med.get("Article", {}) or {}
        jour = art.get("Journal", {}) or {}
        pubdate = (jour.get("JournalIssue", {}) or {}).get("PubDate", {})
        year = pubdate.get("Year") or (pubdate.get("MedlineDate") or "").split(" ")[0] or "N/A"
        abstract = _ps._flatten_abstract(art)
        if not abstract.strip():
            continue  # analytical corpus = records with abstracts
        rows.append({
            "PMID": str(med.get("PMID", "")).strip(),
            "Title": (art.get("ArticleTitle") or "").strip(),
            "Authors": _ps._authors_to_str(art),
            "Year": str(year),
            "Journal": (jour.get("Title") or "").strip(),
            "Abstract": abstract,
        })
    return rows


def _efetch_ids(ids, batch, delay):
    rows = []
    for i in range(0, len(ids), batch):
        chunk = ids[i:i + batch]
        for attempt in range(5):
            try:
                f = Entrez.efetch(db="pubmed", rettype="medline", retmode="xml",
                                  id=",".join(chunk))
                parsed = Entrez.read(f); f.close()
                rows.extend(_parse(parsed.get("PubmedArticle", [])))
                break
            except Exception as e:
                wait = 2 ** attempt
                print(f"    retry {attempt+1} ({type(e).__name__}); wait {wait}s",
                      file=sys.stderr, flush=True)
                time.sleep(wait)
        time.sleep(delay)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="ETL/pubmed_corpus_stratified.csv")
    ap.add_argument("--batch", type=int, default=200)
    ap.add_argument("--max", type=int, default=0, help="cap records in full mode (0 = all)")
    ap.add_argument("--per-year", type=int, default=0,
                    help=">0 enables year-stratified sampling of this many papers per year")
    ap.add_argument("--start-year", type=int, default=2002)
    ap.add_argument("--end-year", type=int, default=2025)
    args = ap.parse_args()

    Entrez.email = os.getenv("NCBI_EMAIL", "cvd-research@sdsu.edu")
    if os.getenv("NCBI_API_KEY"):
        Entrez.api_key = os.getenv("NCBI_API_KEY")
    delay = 0.11 if os.getenv("NCBI_API_KEY") else 0.34

    rows = []
    if args.per_year:
        print(f"year-stratified: {args.per_year}/yr, {args.start_year}-{args.end_year}",
              file=sys.stderr, flush=True)
        for year in range(args.start_year, args.end_year + 1):
            term = f"{BASE} AND ({year}:{year}[dp])"
            h = Entrez.esearch(db="pubmed", term=term, retmax=args.per_year, sort="relevance")
            ids = Entrez.read(h).get("IdList", []); h.close()
            time.sleep(delay)
            before = len(rows)
            rows.extend(_efetch_ids(ids, args.batch, delay))
            print(f"  {year}: {len(ids)} ids -> {len(rows)-before} with abstracts "
                  f"(total {len(rows)})", file=sys.stderr, flush=True)
            pd.DataFrame(rows).drop_duplicates("PMID").to_csv(args.out, index=False)  # checkpoint
    else:
        h = Entrez.esearch(db="pubmed", term=QUERY, usehistory="y", retmax=0)
        res = Entrez.read(h); h.close()
        count = int(res["Count"]); webenv = res["WebEnv"]; qk = res["QueryKey"]
        if args.max:
            count = min(count, args.max)
        print(f"query matched {res['Count']}; fetching {count}", file=sys.stderr, flush=True)
        for start in range(0, count, args.batch):
            for attempt in range(5):
                try:
                    f = Entrez.efetch(db="pubmed", rettype="medline", retmode="xml",
                                      retstart=start, retmax=args.batch, webenv=webenv, query_key=qk)
                    parsed = Entrez.read(f); f.close()
                    rows.extend(_parse(parsed.get("PubmedArticle", [])))
                    break
                except Exception as e:
                    time.sleep(2 ** attempt)
            time.sleep(delay)
            if ((start + args.batch) // args.batch) % 25 == 0:
                print(f"  {min(start+args.batch,count)}/{count}, {len(rows)} w/abstracts",
                      file=sys.stderr, flush=True)
                pd.DataFrame(rows).drop_duplicates("PMID").to_csv(args.out, index=False)

    df = pd.DataFrame(rows, columns=COLS).drop_duplicates("PMID")
    df.to_csv(args.out, index=False)
    print(f"DONE: wrote {len(df)} records with abstracts to {args.out}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
