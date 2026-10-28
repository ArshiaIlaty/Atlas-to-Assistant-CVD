"""Recompute the paper's RAG-vs-no-retrieval comparison WITH inferential
statistics, from the existing Chatbot-Benchmark-Questions/Q_A_evaluations.xlsx.

This directly answers reviewers R1#2 / R2#2, who asked for confidence intervals
and paired significance tests (Wilcoxon signed-rank) instead of bare means.

Sheet layout: Question | Answer 1 (RAG) | M1 M2 M3 | Answer 2 (no-retrieval) | M1 M2 M3
M1/M2/M3 are the three 0-9 expert dimensions (Factual Correctness, Clinical
Relevance, Safety & Risk Awareness in the paper's Table 3).

Usage:  python3 -I experiments/eval/analyze_existing.py [path/to/Q_A_evaluations.xlsx]
"""
from __future__ import annotations
import sys, os
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import stats as S  # noqa: E402

METRICS = ["Factual Correctness", "Clinical Relevance", "Safety & Risk Awareness"]


def load_scores(path):
    df = pd.read_excel(path, sheet_name=0, header=0)
    cols = list(df.columns)
    # Expect: Question, Answer 1, M1, M2, M3, Answer 2, M1, M2, M3
    rag = df.iloc[:, [2, 3, 4]].apply(pd.to_numeric, errors="coerce")
    base = df.iloc[:, [6, 7, 8]].apply(pd.to_numeric, errors="coerce")
    rag.columns = METRICS
    base.columns = METRICS
    return rag, base, len(df)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else \
        "Chatbot-Benchmark-Questions/Q_A_evaluations.xlsx"
    rag, base, n = load_scores(path)
    print(f"Loaded {n} benchmark questions from {path}\n")
    print(f"{'Metric':<26}{'RAG':>7}{'NoRet':>8}{'Δ':>7}{'95% CI':>18}"
          f"{'Wilcoxon p':>13}{'dz':>7}")
    print("-" * 86)
    for m in METRICS:
        a = rag[m].to_numpy(); b = base[m].to_numpy()
        r = S.compare_conditions(a, b, "RAG", "NoRet")
        ci = f"[{r['ci_low']:+.2f}, {r['ci_high']:+.2f}]"
        p = r["wilcoxon"]["p_value"]
        pstr = f"{p:.2e}" if p == p else "n/a"
        print(f"{m:<26}{r['mean_a']:>7.2f}{r['mean_b']:>8.2f}{r['mean_diff']:>+7.2f}"
              f"{ci:>18}{pstr:>13}{r['cohens_dz']:>7.2f}")
    print("-" * 86)
    print("Δ = paired mean difference (RAG − NoRetrieval); CI = percentile bootstrap "
          "(10k resamples);\ndz = Cohen's dz (within-subject effect size); n per test "
          "excludes rows missing a score.")
    print("\nNOTE: This recomputes significance on the ORIGINAL single-rater scores. "
          "The\nrevision still needs a 2nd blinded rater + inter-rater kappa "
          "(S.cohens_kappa) and\ncitation-faithfulness scored separately (R2#4).")


if __name__ == "__main__":
    main()
