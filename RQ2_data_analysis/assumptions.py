"""
RQ2 statistical assumption checks: Shapiro–Wilk normality test on paired differences.

Before applying the Wilcoxon signed-rank test (non-parametric), this script
verifies whether the distribution of pairwise differences (FATE − baseline)
is approximately normal.

- If Shapiro–Wilk p < 0.05: differences deviate significantly from normality;
  the non-parametric Wilcoxon test (``rq2_results.py``) is appropriate.
- If Shapiro–Wilk p >= 0.05: normality cannot be rejected; a parametric
  paired t-test would also be defensible, but Wilcoxon remains a conservative
  safe choice.

Tests are run for all combinations of:
    Metrics:   fairness_score, performance_score, elapsed_seconds
    Baselines: FairSMOTE, Reweighing, DIR

Input:  ``results/rq2/rq2_all_experiments_results.csv`` (or ``--input``)
Output: printed summaries to stdout (no file output).
"""
import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import shapiro

import paths

# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------
RESULTS_NAME = "rq2_all_experiments_results.csv"

# column names in your combined CSV:
# should contain: dataset_name, protected_attribute, model_identifier, method,
# fairness_score, performance_score, elapsed_seconds, ...
METRICS = ["fairness_score", "performance_score", "elapsed_seconds"]
BASELINES = ["FairSMOTE", "Reweighing", "DIR"]
FATE_LABEL = "FATE"

# ---------------------------------------------------------------------
# HELPER: check assumptions for one metric + one baseline
# ---------------------------------------------------------------------


def check_assumptions(df: pd.DataFrame, metric: str, baseline_label: str) -> None:
    """
    Run the Shapiro–Wilk normality test on paired FATE-vs-baseline differences for one metric.

    For the given metric, pivots the experiment results so that each
    (dataset_name, protected_attribute, model_identifier) row has one value
    for FATE and one for the baseline, computes the difference vector
    ``FATE − baseline``, then applies Shapiro–Wilk.

    Parameters
    ----------
    df : pd.DataFrame
        Combined results CSV with columns: ``dataset_name``,
        ``protected_attribute``, ``model_identifier``, ``method``, and the
        target *metric* column.
    metric : str
        Metric column to test (e.g., ``'fairness_score'``,
        ``'performance_score'``, ``'elapsed_seconds'``).
    baseline_label : str
        Method label to compare against FATE in the ``method`` column
        (e.g., ``'FairSMOTE'``, ``'Reweighing'``, ``'DIR'``).

    Notes
    -----
    Prints a formatted summary to stdout including: number of pairs,
    mean values for FATE and the baseline, mean difference (FATE − baseline),
    Shapiro–Wilk W statistic and p-value, and an interpretation note.
    Skips with a ``[WARN]`` message if fewer than 3 paired observations exist.
    """
    # pivot so each (dataset_name, protected_attr, model) row has FATE + baseline
    pivot = df.pivot_table(
        index=["dataset_name", "protected_attribute", "model_identifier"],
        columns="method",
        values=metric,
    )

    if FATE_LABEL not in pivot.columns or baseline_label not in pivot.columns:
        print(f"[WARN] Missing {FATE_LABEL} or {baseline_label} for metric '{metric}'")
        return

    sub = pivot[[FATE_LABEL, baseline_label]].dropna()
    if len(sub) < 3:
        print(f"[WARN] Not enough paired data for {baseline_label} on metric '{metric}'")
        return

    x = sub[FATE_LABEL].values
    y = sub[baseline_label].values
    diffs = x - y

    # Shapiro–Wilk test for normality of differences
    stat, p = shapiro(diffs)

    print("\n" + "=" * 70)
    print(f"Metric: {metric} | Baseline: {baseline_label}")
    print(f"Number of pairs (n): {len(diffs)}")
    print(f"Mean(FATE)   = {np.mean(x):.4f}")
    print(f"Mean({baseline_label}) = {np.mean(y):.4f}")
    print(f"Mean difference (FATE - {baseline_label}) = {np.mean(diffs):.4f}")
    print(f"Shapiro–Wilk W = {stat:.4f}, p = {p:.4f}")

    if p < 0.05:
        print(" -> Differences deviate significantly from normality "
              "(p < 0.05).")
        print("    Using a non-parametric test such as Wilcoxon is appropriate.")
    else:
        print(" -> Cannot reject normality of the differences (p ≥ 0.05).")
        print("    A parametric paired t-test would be defensible;")
        print("    Wilcoxon remains a conservative choice.")


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main(argv: Optional[list[str]] = None) -> None:
    """Run the checks for every (metric, baseline) pair."""
    parser = argparse.ArgumentParser(description="RQ2: distribution checks of paired differences.")
    parser.add_argument("--input", type=Path, default=paths.RQ2_RESULTS_DIR / RESULTS_NAME)
    args = parser.parse_args(argv)
    df = pd.read_csv(args.input)
    for metric in METRICS:
        for baseline in BASELINES:
            check_assumptions(df, metric, baseline)


if __name__ == "__main__":
    main()
