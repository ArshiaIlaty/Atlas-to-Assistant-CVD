"""Taxonomy visualizations -> one self-contained interactive HTML (local review).

Charts:
  1. Inconsistency rate by hierarchy level (Domain/Subdomain/Topic) for the GPT-o1
     and GPT-4.5 audited hierarchies -> the resolution-reproducibility trade-off.
  2. Sunburst of the GPT-o1 Domain > Subdomain > Topic hierarchy, sized by the
     200-record audit sample -> where the literature concentrates.

Palette = the dataviz skill's validated categorical slots (CVD-safe).
    python3 -I experiments/viz/taxonomy_viz.py
"""
from __future__ import annotations
import os, sys
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(REPO, "experiments", "results", "taxonomy_viz.html")

# validated categorical slots (light mode) from the dataviz skill palette
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e6e2"
LEVELS = ["Domain", "Subdomain", "Topic"]


def inconsistency(path):
    d = pd.read_csv(path, dtype=str, keep_default_na=False)
    n = len(d)
    return [round(100 * (d[c] == "False").sum() / n, 1)
            for c in ["Cat_L1_Valid", "Cat_L2_Valid", "Cat_L3_Valid"]]


def fig_inconsistency():
    o1, g45 = inconsistency(os.path.join(REPO, "Taxonomy/API_Inconsistency_o1.csv")), \
        inconsistency(os.path.join(REPO, "Taxonomy/API_Inconsistency_4.5.csv"))
    fig = go.Figure()
    for name, vals, color in [("GPT-o1 hierarchy", o1, CAT[0]), ("GPT-4.5 hierarchy", g45, CAT[1])]:
        fig.add_bar(x=LEVELS, y=vals, name=name, marker_color=color, marker_cornerradius=4,
                    text=[f"{v:.1f}%" for v in vals], textposition="outside",
                    textfont=dict(color=INK))
    fig.update_layout(
        title="Taxonomy inconsistency rises with depth (200-record audit)",
        barmode="group", bargap=0.35, bargroupgap=0.08,
        yaxis_title="Inconsistency (% mismatched)", xaxis_title=None,
        plot_bgcolor=SURFACE, paper_bgcolor=SURFACE, font=dict(color=INK2, size=13),
        legend=dict(orientation="h", y=1.08, x=0), margin=dict(t=70, l=60, r=20, b=40))
    fig.update_yaxes(gridcolor=GRID, zeroline=False, range=[0, max(o1 + g45) * 1.2])
    fig.update_xaxes(showgrid=False)
    return fig


def fig_sunburst():
    d = pd.read_csv(os.path.join(REPO, "Taxonomy/API_Inconsistency_o1.csv"),
                    dtype=str, keep_default_na=False)
    d = d[(d.Cat_L1 != "") & (d.Cat_L2 != "") & (d.Cat_L3 != "")]
    order = d["Cat_L1"].value_counts().index.tolist()
    cmap = {name: CAT[i % len(CAT)] for i, name in enumerate(order)}
    g = d.groupby(["Cat_L1", "Cat_L2", "Cat_L3"]).size().reset_index(name="count")
    fig = px.sunburst(g, path=["Cat_L1", "Cat_L2", "Cat_L3"], values="count",
                      color="Cat_L1", color_discrete_map=cmap)
    fig.update_traces(insidetextorientation="radial",
                      hovertemplate="<b>%{label}</b><br>%{value} papers<br>%{percentParent} of parent<extra></extra>")
    fig.update_layout(title="GPT-o1 taxonomy: Domain &rsaquo; Subdomain &rsaquo; Topic (audit sample)",
                      paper_bgcolor=SURFACE, font=dict(color=INK2, size=12),
                      margin=dict(t=60, l=10, r=10, b=10))
    return fig


def main():
    f1, f2 = fig_inconsistency(), fig_sunburst()
    head = (f"<!doctype html><meta charset=utf-8><title>CVD taxonomy visualizations</title>"
            f"<style>body{{background:{SURFACE};color:{INK};font:15px/1.5 system-ui,Arial;"
            f"margin:24px;max-width:960px}}h1{{font-size:20px}}p{{color:{INK2}}}</style>"
            f"<h1>Cardiovascular taxonomy — visualizations</h1>"
            f"<p>Computed from the 200-record audited sample (Taxonomy/API_Inconsistency_*.csv). "
            f"Topic-level assignments are far less reproducible than Domain/Subdomain — the "
            f"resolution–reproducibility trade-off the reviewers asked us to foreground.</p>")
    body1 = f1.to_html(full_html=False, include_plotlyjs="inline")
    body2 = f2.to_html(full_html=False, include_plotlyjs=False)
    with open(OUT, "w") as fh:
        fh.write(head + body1 + "<hr style='border:0;border-top:1px solid #e6e6e2;margin:28px 0'>" + body2)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
