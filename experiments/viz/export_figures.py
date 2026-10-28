"""Export all paper figures to high-resolution PNG + vector PDF.

Rebuilds the six figures from saved artifacts and writes them to
experiments/results/figures/ as both .png (scale=3, publication raster) and
.pdf (vector, for LaTeX/Word). Uses local SBERT (cached) so no API calls.

    python3 -I experiments/viz/export_figures.py
"""
from __future__ import annotations
import os
os.environ["EMBED_BACKEND"] = "local"        # must precede any config import
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in ("experiments", "experiments/viz", "experiments/clustering"):
    sys.path.insert(0, os.path.join(REPO, p))

import taxonomy_viz as tv      # noqa: E402
import taxonomy_extra as te    # noqa: E402
import cluster_viz as cv       # noqa: E402

FIGDIR = os.path.join(REPO, "experiments", "results", "figures")
os.makedirs(FIGDIR, exist_ok=True)


def build_specs():
    specs = [
        ("fig1_taxonomy_inconsistency", tv.fig_inconsistency(), 1000, 560),
        ("fig2_taxonomy_sunburst", tv.fig_sunburst(), 820, 820),
    ]
    df = te.tag_corpus_domains()
    specs.append(("fig3_domain_trends", te.fig_trends(df), 1000, 560))
    specs.append(("fig4_mismatch_heatmap", te.fig_heatmap(), 1000, 560))
    c1, c2, _ab, _s = cv.build_figures()
    specs.append(("fig5_cluster_scatter", c1, 950, 680))
    specs.append(("fig6_cluster_trends", c2, 1000, 560))
    return specs


def main():
    for name, fig, w, h in build_specs():
        fig.update_layout(width=w, height=h)
        png = os.path.join(FIGDIR, name + ".png")
        pdf = os.path.join(FIGDIR, name + ".pdf")
        fig.write_image(png, scale=3)     # ~3x = publication-quality raster
        fig.write_image(pdf)              # vector
        print("wrote", os.path.basename(png), "+", os.path.basename(pdf))
    print("\nall figures ->", FIGDIR)


if __name__ == "__main__":
    main()
