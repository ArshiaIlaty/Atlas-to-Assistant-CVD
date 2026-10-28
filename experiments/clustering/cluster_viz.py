"""Clustering visualizations -> experiments/results/clustering/clustering_viz.html

Reads the artifacts written by cluster.py:
  - 2-D UMAP scatter of abstracts, colored by the top-5 clusters (+ "other"/noise)
  - annual counts of the top-5 abstract clusters (compositional on a stratified corpus)
  - a summary panel: #clusters, noise %, silhouette, bootstrap ARI, lexical non-corr.
"""
from __future__ import annotations
import os, sys, json
import pandas as pd
import plotly.graph_objects as go

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402

D = os.path.join(config.REPO, "experiments", "results", "clustering")
OUT = os.path.join(D, "clustering_viz.html")
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6", "#e34948"]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e6e2"


def build_figures():
    summ = json.load(open(os.path.join(D, "summary.json")))
    ab = summ["abstract"]
    labels = {c["cluster"]: c["label"] for c in ab["top5"]}
    order = [c["cluster"] for c in ab["top5"]]

    # scatter
    xy = pd.read_csv(os.path.join(D, "abstract_umap2d.csv"))
    fig1 = go.Figure()
    noise = xy[xy.top5 == -1]
    fig1.add_scatter(x=noise.x, y=noise.y, mode="markers", name="other / noise",
                       marker=dict(size=3, color="#d9d9d4"), hoverinfo="skip")
    for i, c in enumerate(order):
        pts = xy[xy.top5 == c]
        fig1.add_scatter(x=pts.x, y=pts.y, mode="markers",
                           name=labels[c][:38], marker=dict(size=4, color=CAT[i % len(CAT)]),
                           hovertemplate=f"{labels[c]}<extra></extra>")
    fig1.update_layout(title="Abstract clusters (SBERT → UMAP → HDBSCAN), 2-D projection",
                       plot_bgcolor=SURFACE, paper_bgcolor=SURFACE, font=dict(color=INK2, size=12),
                       legend=dict(font=dict(size=11)), margin=dict(t=60, l=20, r=20, b=20),
                       xaxis=dict(visible=False), yaxis=dict(visible=False))

    # annual trends
    an = pd.read_csv(os.path.join(D, "abstract_annual.csv"), index_col=0)
    an = an[(an.index >= 2002) & (an.index <= 2025)]
    fig2 = go.Figure()
    for i, c in enumerate(order):
        col = str(c)
        if col in an.columns:
            fig2.add_scatter(x=an.index, y=an[col], mode="lines", name=labels[c][:38],
                             line=dict(width=2, color=CAT[i % len(CAT)]),
                             hovertemplate=f"{labels[c]}<br>%{{x}}: %{{y}}<extra></extra>")
    fig2.update_layout(title="Top-5 abstract clusters over time (counts in the year-stratified sample)",
                       plot_bgcolor=SURFACE, paper_bgcolor=SURFACE, font=dict(color=INK2, size=12),
                       yaxis_title="papers in sample", legend=dict(orientation="h", y=-0.2, font=dict(size=10)),
                       margin=dict(t=60, l=55, r=20, b=80))
    fig2.update_yaxes(gridcolor=GRID, zeroline=False)
    fig2.update_xaxes(showgrid=False, dtick=4)
    return fig1, fig2, ab, summ


def main():
    fig1, fig2, ab, summ = build_figures()

    def stat(label, val):
        return (f"<div style='display:inline-block;min-width:150px;margin:6px 14px 6px 0'>"
                f"<div style='font-size:22px;color:{INK};font-weight:650'>{val}</div>"
                f"<div style='font-size:12px;color:{INK2}'>{label}</div></div>")
    tiles = (stat("abstract clusters", ab["n_clusters"]) +
             stat("noise", f"{ab['noise_fraction']*100:.0f}%") +
             stat("silhouette", ab["silhouette"]) +
             stat("bootstrap ARI", ab.get("bootstrap_ARI_mean", "n/a")) +
             stat("lexical non-corr. (abstract)", f"{ab['lexical_noncorrespondence_pct']}%") +
             stat("lexical non-corr. (title)", f"{summ['title']['lexical_noncorrespondence_pct']}%"))

    head = (f"<!doctype html><meta charset=utf-8><title>CVD clustering</title>"
            f"<style>body{{background:{SURFACE};color:{INK};font:15px/1.5 system-ui,Arial;"
            f"margin:24px;max-width:1000px}}h1{{font-size:20px}}p{{color:{INK2}}}</style>"
            f"<h1>Semantic topic discovery — SBERT → UMAP → HDBSCAN</h1>"
            f"<p>Stage-1 clustering of the 20,881-doc corpus. Stability metrics (silhouette, "
            f"bootstrap ARI) and the lexical non-correspondence diagnostic address reviewer R2#7.</p>"
            f"<div style='margin:14px 0 24px'>{tiles}</div>")
    b1 = fig1.to_html(full_html=False, include_plotlyjs="inline")
    b2 = fig2.to_html(full_html=False, include_plotlyjs=False)
    open(OUT, "w").write(head + b1 + "<hr style='border:0;border-top:1px solid #e6e6e2;margin:24px 0'>" + b2)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
