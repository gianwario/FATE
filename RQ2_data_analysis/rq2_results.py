"""
RQ2 statistical analysis: Wilcoxon signed-rank tests and Vargha–Delaney A₁₂ effect sizes.

For each combination of (metric × baseline method), this module:

1. Pivots ``rq2_all_experiments_results.csv`` to align paired (FATE, baseline)
   observations per (dataset_name, protected_attribute, model_identifier) group.
2. Runs the two-sided Wilcoxon signed-rank test (appropriate given non-normal
   differences confirmed by ``assumptions.py``).
3. Computes the Vargha–Delaney A₁₂ effect size, adjusted for metric direction
   (A₁₂_eff > 0.5 always means FATE is favoured regardless of whether higher
   or lower is better for the metric).
4. Labels the outcome: ``FATE_better``, ``Baseline_better``, ``No_diff``, or
   ``Tie``.

Results are saved to ``results/rq2/rq2_hypothesis_tests.csv`` and support
the nine hypotheses tested in the paper:

    H1a/b/c – FATE achieves lower fairness_score than FairSMOTE / Reweighing / DIR.
    H2a/b/c – FATE achieves higher performance_score.
    H3a/b/c – FATE has lower elapsed_seconds (execution time).
"""
import argparse
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import wilcoxon

import paths


RESULTS_NAME = "rq2_all_experiments_results.csv"
TESTS_NAME = "rq2_hypothesis_tests.csv"


HYPOTHESIS_MAP = {
    ("fairness_score", "FairSMOTE"): "H1a",
    ("fairness_score", "Reweighing"): "H1b",
    ("fairness_score", "DIR"): "H1c",

    ("performance_score", "FairSMOTE"): "H2a",
    ("performance_score", "Reweighing"): "H2b",
    ("performance_score", "DIR"): "H2c",

    ("elapsed_seconds", "FairSMOTE"): "H3a",
    ("elapsed_seconds", "Reweighing"): "H3b",
    ("elapsed_seconds", "DIR"): "H3c",
}

METRIC_DIRECTION = {
    "fairness_score": "lower",  # lower deviation = fairer
    "elapsed_seconds": "lower",  # faster is better
    "execution_time": "lower",  # in case you used this name
    "performance_score": "higher",  # higher PR-AUC is better
}


def vargha_delaney_a12(x: ArrayLike, y: ArrayLike) -> float:
    """
    Compute the Vargha–Delaney A₁₂ effect size.

    A₁₂ = P(X > Y) + 0.5 · P(X = Y), where X and Y are the two samples.
    A₁₂ = 0.5 indicates no stochastic difference; > 0.5 means X tends to be
    larger than Y.

    Parameters
    ----------
    x : array-like
        First sample (FATE scores).
    y : array-like
        Second sample (baseline scores).

    Returns
    -------
    float
        A₁₂ value in [0, 1].

    Notes
    -----
    For metrics where *lower* is better (fairness_score, elapsed_seconds),
    ``compare_method`` computes ``a12_effective = 1 − A₁₂`` so that
    ``a12_effective > 0.5`` consistently indicates FATE is better regardless
    of the metric direction.
    """
    x = np.array(x)
    y = np.array(y)
    nx = len(x)
    ny = len(y)
    # simpler implementation:
    # use rankdata if you want exact ties, but this simple form is often enough:
    combined = np.concatenate([x, y])
    from scipy.stats import rankdata
    r = rankdata(combined)
    rx = r[:nx].sum()
    a12 = (rx / nx - (nx + 1) / 2) / ny
    return a12


def compare_method(df: pd.DataFrame, metric: str, baseline_label: str,
                   results_list: Optional[list[dict[str, object]]] = None) -> None:
    """
    Compare FATE against one baseline on one metric using Wilcoxon and A₁₂.

    Pivots *df* to align FATE and baseline scores per (dataset_name,
    protected_attribute, model_identifier) group, runs the two-sided Wilcoxon
    signed-rank test, computes A₁₂, and appends a structured result row.

    Parameters
    ----------
    df : pd.DataFrame
        Combined results with columns: ``dataset_name``, ``protected_attribute``,
        ``model_identifier``, ``method``, and the target *metric*.
    metric : str
        Metric to compare: ``'fairness_score'``, ``'performance_score'``, or
        ``'elapsed_seconds'``.
    baseline_label : str
        Label of the comparison method in the ``method`` column
        (``'FairSMOTE'``, ``'Reweighing'``, or ``'DIR'``).
    results_list : list or None, optional
        If provided, a result dict is appended with keys: ``hypothesis``,
        ``metric``, ``direction``, ``baseline``, ``n_pairs``, ``fate_mean``,
        ``baseline_mean``, ``p_value``, ``a12_raw``, ``a12_effective``,
        ``significant_0.05``, ``who_is_better``.

    Notes
    -----
    ``who_is_better`` is one of: ``'FATE_better'``, ``'Baseline_better'``,
    ``'No_diff'`` (p >= 0.05), or ``'Tie'`` (significant but A₁₂_eff = 0.5).

    The metric direction is read from ``METRIC_DIRECTION``:
    ``'lower'`` for fairness_score and elapsed_seconds; ``'higher'`` for
    performance_score.  For lower-is-better metrics, ``a12_effective = 1 −
    a12_raw`` so that values > 0.5 still indicate FATE is favoured.
    """

    direction = METRIC_DIRECTION.get(metric, "higher")  # default: higher is better

    # pivot so that each row has columns FATE and baseline_label for the given metric
    pivot = df.pivot_table(
        index=["dataset_name", "protected_attribute", "model_identifier"],
        columns="method",
        values=metric,
    )

    if "FATE" not in pivot.columns or baseline_label not in pivot.columns:
        print(f"[WARN] Missing FATE or {baseline_label} for metric {metric}")
        return

    # IMPORTANT: use baseline_label variable, not the literal string
    sub = pivot[["FATE", baseline_label]].dropna()
    if len(sub) < 1:
        print(f"[WARN] No paired data for {baseline_label} on metric {metric}")
        return

    x = sub["FATE"].values
    y = sub[baseline_label].values

    # Wilcoxon signed-rank (paired)
    try:
        stat, p = wilcoxon(x, y, alternative="two-sided")
    except ValueError as e:
        print(f"[WARN] Wilcoxon failed for {baseline_label}, metric {metric}: {e}")
        return

    a12_raw = vargha_delaney_a12(x, y)  # P(FATE > baseline) + 0.5 P(=)

    fate_mean = float(np.nanmean(x))
    base_mean = float(np.nanmean(y))
    n = len(x)

    # Re-interpret A12 depending on direction:
    # - If higher is better: a12_eff = a12_raw
    # - If lower is better:  a12_eff = 1 - a12_raw (so > 0.5 still means FATE better)
    if direction == "higher":
        a12_eff = a12_raw
    else:  # direction == "lower"
        a12_eff = 1.0 - a12_raw

    # Decide qualitative direction
    if p < 0.05:
        if a12_eff > 0.5:
            direction_label = "FATE_better"
            msg = " -> Significant: FATE tends to be better (given metric direction)."
        elif a12_eff < 0.5:
            direction_label = "Baseline_better"
            msg = " -> Significant: baseline tends to be better (given metric direction)."
        else:
            direction_label = "Tie"
            msg = " -> Significant but A12_eff == 0.5 (tie)."
    else:
        direction_label = "No_diff"
        msg = " -> No statistically significant difference."

    # console output
    print(f"\n=== {metric} ({direction}-is-better) : FATE vs {baseline_label} ===")
    print(f"n = {n}")
    print(f"FATE         mean = {fate_mean:.4f}")
    print(f"{baseline_label:13s} mean = {base_mean:.4f}")
    print(f"Wilcoxon p-value = {p:.4f}")
    print(f"Vargha–Delaney A12 (raw, FATE vs {baseline_label}) = {a12_raw:.3f}")
    print(f"A12 (interpreted wrt direction)                    = {a12_eff:.3f}")
    print(msg)

    # hypothesis ID, if you use them
    hyp_id = HYPOTHESIS_MAP.get((metric, baseline_label), "")

    row = {
        "hypothesis": hyp_id,
        "metric": metric,
        "direction": direction,          # 'higher' or 'lower'
        "baseline": baseline_label,
        "n_pairs": n,
        "fate_mean": fate_mean,
        "baseline_mean": base_mean,
        "p_value": float(p),
        "a12_raw": float(a12_raw),
        "a12_effective": float(a12_eff),  # > 0.5 => FATE better, regardless of direction
        "significant_0.05": p < 0.05,
        "who_is_better": direction_label,  # FATE_better / Baseline_better / No_diff / Tie
    }

    if results_list is not None:
        results_list.append(row)


def main(argv: Optional[list[str]] = None) -> None:
    """
    Run all nine Wilcoxon tests and save a summary CSV of hypothesis test results.

    Iterates over all (metric × baseline) combinations, calls
    ``compare_method`` for each, and writes the collected rows to
    ``results/rq2/rq2_hypothesis_tests.csv``.
    """
    parser = argparse.ArgumentParser(description="RQ2: hypothesis tests H1a-H3c.")
    parser.add_argument("--input", type=Path, default=paths.RQ2_RESULTS_DIR / RESULTS_NAME,
                        help="per-method results (e.g. reference/rq2/%s)" % RESULTS_NAME)
    parser.add_argument("--out", type=Path, default=paths.RQ2_RESULTS_DIR / TESTS_NAME)
    args = parser.parse_args(argv)
    df = pd.read_csv(args.input)

    summary_rows = []
    for metric in ["fairness_score", "performance_score", "elapsed_seconds"]:
        for baseline in ["FairSMOTE", "Reweighing", "DIR"]:
            compare_method(df, metric, baseline, results_list=summary_rows)

    summary_df = pd.DataFrame(summary_rows)
    paths.ensure_dir(args.out.parent)
    summary_df.to_csv(args.out, index=False)
    print(f"Hypothesis tests saved to {args.out}")


if __name__ == "__main__":
    main()
