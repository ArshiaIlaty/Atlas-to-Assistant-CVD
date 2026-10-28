"""Two more taxonomy visualizations -> experiments/results/taxonomy_extra.html

(a) Per-domain temporal trends: the 20,881-doc stratified corpus tagged to a
    Domain by nearest-neighbor to the audited taxonomy, shown as each domain's
    SHARE of the year. NOTE: the corpus is year-stratified (~constant N/year), so
    these are COMPOSITIONAL trends (share), not absolute publication growth.
(b) Where Topic mismatches land: inconsistency % by assigned Domain x level
    (GPT-o1 audit) — a heatmap showing the Topic column lighting up.

    EMBED_BACKEND=local python3 -I experiments/viz/taxonomy_extra.py
"""
from __future__ import annotations
import os, sys
os.environ["EMBED_BACKEND"] = "local"          # free/cached SBERT for tagging

VIZ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(VIZ))        # experiments/
import numpy as np, pandas as pd                # noqa: E402
import plotly.graph_objects as go               # noqa: E402
import config, corpus, embeddings               # noqa: E402

REPO = config.REPO
OUT = os.path.join(REPO, "experiments", "results", "taxonomy_extra.html")
CORPUS = os.path.join(REPO, "ETL", "pubmed_corpus_stratified.csv")
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e6e2"


def tag_corpus_domains():
    tax = pd.read_csv(os.path.join(REPO, "Taxonomy/API_Inconsistency_o1.csv"),
                      dtype=str, keep_default_na=False).drop_duplicates("Title")
    tax = tax[tax.Abstract.str.strip() != ""].reset_index(drop=True)
    tax_emb = embeddings.embed_cached((tax.Title.str.strip() + ". " + tax.Abstract.str.strip()).tolist(),
                                      "taxonomy")
    docs = corpus.load_corpus(path=CORPUS)
    corp_emb = embeddings.embed_cached([d["text"] for d in docs], "corpusall")
    nn = (corp_emb @ tax_emb.T).argmax(1)
    dom = tax["Cat_L1"].to_numpy()[nn]
    yr = pd.to_numeric(pd.Series([d["year"] for d in docs]), errors="coerce")
    df = pd.DataFrame({"year": yr, "domain": dom}).dropna()
    df = df[(df.year >= 2002) & (df.year <= 2025)]
    df["year"] = df["year"].astype(int)
    return df


def fig_trends(df):
    ct = df.groupby(["year", "domain"]).size().unstack(fill_value=0).sort_index()
    share = ct.div(ct.sum(1), axis=0) * 100
    order = df["domain"].value_counts().index.tolist()
    fig = go.Figure()
    for i, dom in enumerate(order):
        if dom not in share.columns:
            continue
        fig.add_scatter(x=share.index, y=share[dom], mode="lines", name=dom,
                        line=dict(width=2, color=CAT[i % len(CAT)]),
                        hovertemplate=f"{dom}<br>%{{x}}: %{{y:.1f}}%<extra></extra>")
    fig.update_layout(
        title="Domain share of the year-stratified CVD corpus, 2002–2025 (compositional, not absolute growth)",
        yaxis_title="% of that year's sampled papers", xaxis_title=None,
        plot_bgcolor=SURFACE, paper_bgcolor=SURFACE, font=dict(color=INK2, size=12),
        legend=dict(orientation="h", y=-0.18, x=0, font=dict(size=11)),
        margin=dict(t=60, l=60, r=20, b=80))
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    fig.update_xaxes(showgrid=False, dtick=4)
    return fig


def fig_heatmap():
    d = pd.read_csv(os.path.join(REPO, "Taxonomy/API_Inconsistency_o1.csv"),
                    dtype=str, keep_default_na=False)
    doms = d["Cat_L1"].value_counts().index.tolist()
    levels = ["Domain", "Subdomain", "Topic"]
    valcols = ["Cat_L1_Valid", "Cat_L2_Valid", "Cat_L3_Valid"]
    z, text, rows = [], [], []
    for dom in doms:
        sub = d[d.Cat_L1 == dom]
        row = [round(100 * (sub[c] == "False").mean(), 1) for c in valcols]
        z.append(row)
        text.append([f"{v:.0f}%<br>(n={len(sub)})" for v in row])
        rows.append(f"{dom}  (n={len(sub)})")
    fig = go.Figure(go.Heatmap(
        z=z, x=levels, y=rows, text=text, texttemplate="%{text}",
        colorscale=[[0, "#eef4fb"], [0.5, "#7fb0e6"], [1, "#13395f"]],
        colorbar=dict(title="mismatch %"), hovertemplate="%{y}<br>%{x}: %{z}%<extra></extra>"))
    fig.update_layout(
        title="Where mismatches concentrate: inconsistency % by assigned domain × level (GPT-o1 audit)",
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, font=dict(color=INK2, size=12),
        margin=dict(t=60, l=230, r=20, b=40))
    fig.update_yaxes(autorange="reversed")
    return fig


def main():
    df = tag_corpus_domains()
    f1, f2 = fig_trends(df), fig_heatmap()
    head = (f"<!doctype html><meta charset=utf-8><title>CVD taxonomy — trends & mismatches</title>"
            f"<style>body{{background:{SURFACE};color:{INK};font:15px/1.5 system-ui,Arial;"
            f"margin:24px;max-width:980px}}h1{{font-size:20px}}p{{color:{INK2}}}</style>"
            f"<h1>Taxonomy — temporal trends &amp; mismatch concentration</h1>"
            f"<p>(a) Domain composition of the 20,881-doc year-stratified corpus over time "
            f"(tagged by nearest-neighbor to the audited taxonomy; shares, not absolute counts). "
            f"(b) Topic-level inconsistency dominates across every domain.</p>")
    b1 = f1.to_html(full_html=False, include_plotlyjs="inline")
    b2 = f2.to_html(full_html=False, include_plotlyjs=False)
    open(OUT, "w").write(head + b1 + "<hr style='border:0;border-top:1px solid #e6e6e2;margin:28px 0'>" + b2)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
