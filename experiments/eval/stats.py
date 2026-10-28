"""Inferential statistics for paired RAG-vs-baseline comparisons.

Addresses reviewer comments R1#2 / R2#2 (no inferential statistics): provides
paired significance tests, effect sizes, and bootstrap confidence intervals for
the mean difference between two conditions scored on the same questions, plus
inter-rater agreement (R1#1 / R2#1).
"""
from __future__ import annotations
import numpy as np
from scipy import stats as sps


def _clean_pairs(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = ~(np.isnan(a) | np.isnan(b))
    return a[mask], b[mask]


def bootstrap_ci_mean_diff(a, b, n_boot=10000, alpha=0.05, seed=42):
    """Percentile bootstrap CI for the paired mean difference (a - b)."""
    a, b = _clean_pairs(a, b)
    d = a - b
    rng = np.random.default_rng(seed)
    n = len(d)
    if n == 0:
        return {"mean_diff": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": 0}
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = d[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"mean_diff": float(d.mean()), "ci_low": float(lo), "ci_high": float(hi), "n": int(n)}


def wilcoxon_signed_rank(a, b):
    """Paired Wilcoxon signed-rank test (non-parametric, recommended by R2#2)."""
    a, b = _clean_pairs(a, b)
    if len(a) < 1 or np.all(a - b == 0):
        return {"statistic": float("nan"), "p_value": float("nan"), "n": int(len(a))}
    try:
        res = sps.wilcoxon(a, b, zero_method="wilcox", correction=False, mode="auto")
        return {"statistic": float(res.statistic), "p_value": float(res.pvalue), "n": int(len(a))}
    except ValueError as e:
        return {"statistic": float("nan"), "p_value": float("nan"), "n": int(len(a)), "note": str(e)}


def paired_t(a, b):
    a, b = _clean_pairs(a, b)
    if len(a) < 2:
        return {"statistic": float("nan"), "p_value": float("nan"), "n": int(len(a))}
    res = sps.ttest_rel(a, b)
    return {"statistic": float(res.statistic), "p_value": float(res.pvalue), "n": int(len(a))}


def cohens_dz(a, b):
    """Within-subject effect size for paired data (mean diff / sd of diffs)."""
    a, b = _clean_pairs(a, b)
    d = a - b
    sd = d.std(ddof=1)
    return float(d.mean() / sd) if sd > 0 else float("nan")


def compare_conditions(a, b, label_a="A", label_b="B", n_boot=10000):
    """Full paired comparison bundle for two score vectors."""
    a, b = _clean_pairs(a, b)
    out = {
        "label_a": label_a, "label_b": label_b,
        "mean_a": float(a.mean()) if len(a) else float("nan"),
        "mean_b": float(b.mean()) if len(b) else float("nan"),
        "wilcoxon": wilcoxon_signed_rank(a, b),
        "paired_t": paired_t(a, b),
        "cohens_dz": cohens_dz(a, b),
    }
    out.update(bootstrap_ci_mean_diff(a, b, n_boot=n_boot))
    return out


def cohens_kappa(r1, r2, weights=None):
    """Cohen's kappa for two raters (R1#1 / R2#1 inter-rater agreement).

    weights: None (nominal), 'linear', or 'quadratic' for ordinal 0-9 scores.
    """
    r1 = np.asarray(r1); r2 = np.asarray(r2)
    mask = ~(_isnan(r1) | _isnan(r2))
    r1, r2 = r1[mask], r2[mask]
    cats = np.unique(np.concatenate([r1, r2]))
    k = len(cats)
    cat_idx = {c: i for i, c in enumerate(cats)}
    O = np.zeros((k, k))
    for x, y in zip(r1, r2):
        O[cat_idx[x], cat_idx[y]] += 1
    n = O.sum()
    if n == 0:
        return float("nan")
    row = O.sum(1); col = O.sum(0)
    E = np.outer(row, col) / n
    if weights is None:
        W = 1 - np.eye(k)
    else:
        ci = np.arange(k)
        diff = np.abs(ci[:, None] - ci[None, :]).astype(float)
        W = diff if weights == "linear" else diff ** 2
    num = (W * O).sum(); den = (W * E).sum()
    return float(1 - num / den) if den > 0 else float("nan")


def _isnan(x):
    try:
        return np.isnan(x.astype(float))
    except (TypeError, ValueError):
        return np.array([not np.isreal(v) for v in x])
