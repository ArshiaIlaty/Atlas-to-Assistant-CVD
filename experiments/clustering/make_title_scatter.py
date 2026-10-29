"""Title-clusters UMAP 2-D scatter (fig7), built from the cached title embeddings
and the saved HDBSCAN assignments (no need to re-run the full clustering).
Writes title_umap2d.csv and exports PNG + vector PDF.

    python3 -I experiments/clustering/make_title_scatter.py
"""
from __future__ import annotations
import os, sys, json
import numpy as np, pandas as pd, umap
import plotly.graph_objects as go

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
D = os.path.join(REPO, "experiments", "results", "clustering")
FIGDIR = os.path.join(REPO, "experiments", "results", "figures")
CACHE = os.path.join(REPO, "experiments", ".cache", "emb_local_all-MiniLM-L6-v2_20881_titles.npy")
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6", "#e34948"]
SURFACE, INK2 = "#fcfcfb", "#52514e"

emb = np.load(CACHE)
asn = pd.read_csv(os.path.join(D, "title_assignments.csv"))
labels = asn["cluster"].to_numpy()
assert len(emb) == len(labels), f"length mismatch {len(emb)} vs {len(labels)}"

summ = json.load(open(os.path.join(D, "summary.json")))["title"]
top5 = [c["cluster"] for c in summ["top5"]]
names = {c["cluster"]: c["label"] for c in summ["top5"]}

print("computing 2-D UMAP for titles ...", file=sys.stderr, flush=True)
xy = umap.UMAP(n_components=2, random_state=42).fit_transform(emb)
df = pd.DataFrame({"x": xy[:, 0], "y": xy[:, 1], "cluster": labels,
                   "top5": [c if c in top5 else -1 for c in labels]})
df.to_csv(os.path.join(D, "title_umap2d.csv"), index=False)

fig = go.Figure()
noise = df[df.top5 == -1]
fig.add_scatter(x=noise.x, y=noise.y, mode="markers", name="other / noise",
                marker=dict(size=3, color="#d9d9d4"), hoverinfo="skip")
for i, c in enumerate(top5):
    pts = df[df.top5 == c]
    fig.add_scatter(x=pts.x, y=pts.y, mode="markers", name=names[c][:38],
                    marker=dict(size=4, color=CAT[i % len(CAT)]),
                    hovertemplate=f"{names[c]}<extra></extra>")
fig.update_layout(title="Title clusters (SBERT → UMAP → HDBSCAN), 2-D projection",
                  plot_bgcolor=SURFACE, paper_bgcolor=SURFACE, font=dict(color=INK2, size=12),
                  legend=dict(font=dict(size=11)), margin=dict(t=60, l=20, r=20, b=20),
                  xaxis=dict(visible=False), yaxis=dict(visible=False), width=950, height=680)

os.makedirs(FIGDIR, exist_ok=True)
png = os.path.join(FIGDIR, "fig7_title_cluster_scatter.png")
pdf = os.path.join(FIGDIR, "fig7_title_cluster_scatter.pdf")
fig.write_image(png, scale=3)
fig.write_image(pdf)
print("wrote", png, "+", pdf)
