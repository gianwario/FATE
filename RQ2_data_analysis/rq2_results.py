"""
RQ2 statistical analysis: hypothesis tests H1a–H3c (Table 4 of the paper).

For each (metric × baseline) pair, the per-configuration results of FATE and of
the baseline are paired on (dataset_name, protected_attribute,
model_identifier) — 24 pairs — and the following are computed:

1. **Wilcoxon signed-rank test** (two-sided, exact) on the paired differences
   ``d = FATE − baseline``.  This is the test reported in the paper.
2. **Symmetry check** of ``d``.  The Wilcoxon signed-rank test does not assume
   normality, but it assumes that the differences are symmetric about their
   median.  Symmetry is tested with the Miao–Gel–Gastwirth (MGG) test, whose
   statistic is ``sqrt(n) · (mean(d) − median(d)) / J`` with
   ``J = sqrt(pi/2) · mean(|d − median(d)|)``; its null distribution is
   obtained by a symmetrised bootstrap (resampling from ``d`` reflected about
   its median, ``BOOTSTRAP_REPLICATES`` replicates, fixed seed).
3. **Exact sign test** on ``d`` (zero differences discarded).  The sign test
   does not assume symmetry, so it is the reference for the comparisons whose
   differences are not symmetric.
4. **Holm–Bonferroni correction** over the nine hypotheses, applied separately
   to the Wilcoxon and to the sign-test p-values.  Significance is decided on
   the Holm-adjusted p-values at ``ALPHA``.
5. **Vargha–Delaney A₁₂** effect size, oriented so that ``a12_effective > 0.5``
   always favours FATE (lower is better for fairness_score and
   elapsed_seconds, higher is better for performance_score).

Hypotheses:

    H1a/b/c – fairness_score    FATE vs FairSMOTE / Reweighing / DIR
    H2a/b/c – performance_score FATE vs FairSMOTE / Reweighing / DIR
    H3a/b/c – elapsed_seconds   FATE vs FairSMOTE / Reweighing / DIR

Input:  ``results/rq2/rq2_all_experiments_results.csv`` (or ``--input``)
Output: ``results/rq2/rq2_hypothesis_tests.csv`` (or ``--out``)
"""
import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy.stats import binomtest, rankdata, wilcoxon

import paths

RESULTS_NAME = "rq2_all_experiments_results.csv"
TESTS_NAME = "rq2_hypothesis_tests.csv"

ALPHA = 0.05
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260930
FATE_LABEL = "FATE"
METRICS = ["fairness_score", "performance_score", "elapsed_seconds"]
BASELINES = ["FairSMOTE", "Reweighing", "DIR"]
PAIR_KEYS = ["dataset_name", "protected_attribute", "model_identifier"]

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
    "performance_score": "higher",  # higher PR-AUC is better
}


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def vargha_delaney_a12(x: ArrayLike, y: ArrayLike) -> float:
    """
    Compute the Vargha–Delaney A₁₂ effect size, A₁₂ = P(X > Y) + 0.5 · P(X = Y).

    Parameters
    ----------
    x, y : array-like
        The two samples (FATE and baseline scores).

    Returns
    -------
    float
        A₁₂ in [0, 1]; 0.5 means no stochastic difference.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ranks = rankdata(np.concatenate([x, y]))
    return float((ranks[:len(x)].sum() / len(x) - (len(x) + 1) / 2) / len(y))


def mgg_statistic(d: np.ndarray) -> float:
    """
    Miao–Gel–Gastwirth symmetry statistic ``sqrt(n)·(mean − median)/J``.

    Parameters
    ----------
    d : np.ndarray
        Paired differences.

    Returns
    -------
    float
        The statistic; 0.0 when all values equal the median (J = 0), which is
        a perfectly symmetric sample.
    """
    median = np.median(d)
    spread = np.sqrt(np.pi / 2) * np.mean(np.abs(d - median))
    if spread == 0:
        return 0.0
    return float(np.sqrt(len(d)) * (np.mean(d) - median) / spread)


def symmetry_test(d: np.ndarray, replicates: int = BOOTSTRAP_REPLICATES,
                  seed: int = BOOTSTRAP_SEED) -> tuple[float, float]:
    """
    Test the symmetry of *d* about its (unknown) median with the MGG statistic.

    The null distribution is obtained by resampling from the symmetrised
    sample ``{d − median(d)} ∪ {median(d) − d}``, which is symmetric by
    construction.

    Parameters
    ----------
    d : np.ndarray
        Paired differences.
    replicates : int
        Number of bootstrap replicates.
    seed : int
        Seed of the bootstrap generator (results are reproducible).

    Returns
    -------
    tuple
        ``(statistic, p_value)``; small p-values indicate asymmetry.
    """
    observed = mgg_statistic(d)
    centred = d - np.median(d)
    pool = np.concatenate([centred, -centred])
    rng = np.random.default_rng(seed)
    samples = rng.choice(pool, size=(replicates, len(d)), replace=True)
    null = np.array([mgg_statistic(s) for s in samples])
    p_value = (np.sum(np.abs(null) >= abs(observed)) + 1) / (replicates + 1)
    return observed, float(p_value)


def sign_test(d: np.ndarray) -> float:
    """
    Exact two-sided sign test of median(d) = 0; zero differences are discarded.

    Parameters
    ----------
    d : np.ndarray
        Paired differences.

    Returns
    -------
    float
        p-value (1.0 if all differences are zero).
    """
    non_zero = d[d != 0]
    if len(non_zero) == 0:
        return 1.0
    return float(binomtest(int(np.sum(non_zero > 0)), len(non_zero), 0.5).pvalue)


def holm_adjust(p_values: ArrayLike) -> np.ndarray:
    """
    Holm–Bonferroni step-down adjustment (family-wise error rate control).

    Parameters
    ----------
    p_values : array-like
        Unadjusted p-values of the family of hypotheses.

    Returns
    -------
    np.ndarray
        Adjusted p-values, in the input order; reject H_i iff adjusted p_i < alpha.
    """
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    adjusted = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(np.argsort(p, kind="stable")):
        running_max = max(running_max, (m - rank) * p[idx])
        adjusted[idx] = min(1.0, running_max)
    return adjusted


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def paired_values(df: pd.DataFrame, metric: str, baseline: str
                  ) -> tuple[np.ndarray, np.ndarray]:
    """
    Align FATE and *baseline* values of *metric* on the configuration keys.

    Parameters
    ----------
    df : pd.DataFrame
        Per-method results (columns ``PAIR_KEYS``, ``method`` and *metric*).
    metric : str
    baseline : str

    Returns
    -------
    tuple
        ``(fate_values, baseline_values)`` for the configurations where both exist.

    Raises
    ------
    KeyError
        If FATE or *baseline* does not appear in the ``method`` column.
    """
    pivot = df.pivot_table(index=PAIR_KEYS, columns="method", values=metric)
    missing = {FATE_LABEL, baseline} - set(pivot.columns)
    if missing:
        raise KeyError(f"method(s) {sorted(missing)} missing for metric '{metric}'")
    pairs = pivot[[FATE_LABEL, baseline]].dropna()
    return pairs[FATE_LABEL].to_numpy(), pairs[baseline].to_numpy()


def compare_method(df: pd.DataFrame, metric: str, baseline: str) -> dict[str, object]:
    """
    Run the unadjusted tests for one hypothesis (FATE vs *baseline* on *metric*).

    Parameters
    ----------
    df : pd.DataFrame
    metric : str
    baseline : str

    Returns
    -------
    dict
        One row of the output table without the Holm-adjusted columns, which
        are added by ``run_tests`` once all nine p-values are known.
    """
    x, y = paired_values(df, metric, baseline)
    d = x - y
    direction = METRIC_DIRECTION[metric]
    a12_raw = vargha_delaney_a12(x, y)
    symmetry_stat, symmetry_p = symmetry_test(d)
    return {
        "hypothesis": HYPOTHESIS_MAP[(metric, baseline)],
        "metric": metric,
        "direction": direction,
        "baseline": baseline,
        "n_pairs": len(d),
        "fate_mean": float(np.mean(x)),
        "baseline_mean": float(np.mean(y)),
        "p_value": float(wilcoxon(x, y, alternative="two-sided").pvalue),
        "a12_raw": a12_raw,
        "a12_effective": a12_raw if direction == "higher" else 1.0 - a12_raw,
        "symmetry_stat": symmetry_stat,
        "symmetry_p": symmetry_p,
        "sign_p": sign_test(d),
    }


def _winner(significant: bool, a12_effective: float) -> str:
    """Label the direction of a (possibly non-significant) comparison."""
    if not significant:
        return "No_diff"
    if a12_effective > 0.5:
        return "FATE_better"
    if a12_effective < 0.5:
        return "Baseline_better"
    return "Tie"


def run_tests(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run all nine comparisons and apply the Holm–Bonferroni correction.

    Parameters
    ----------
    df : pd.DataFrame
        Per-method results.

    Returns
    -------
    pd.DataFrame
        One row per hypothesis (see README, Section 6, for the columns).
    """
    table = pd.DataFrame([compare_method(df, m, b) for m in METRICS for b in BASELINES])
    table["p_holm"] = holm_adjust(table["p_value"])
    table["sign_p_holm"] = holm_adjust(table["sign_p"])
    table["symmetric_0.05"] = table["symmetry_p"] >= ALPHA
    table["significant_holm_0.05"] = table["p_holm"] < ALPHA
    table["sign_significant_holm_0.05"] = table["sign_p_holm"] < ALPHA
    table["who_is_better"] = [_winner(s, a) for s, a in
                              zip(table["significant_holm_0.05"], table["a12_effective"])]
    table["conclusion_robust"] = (table["significant_holm_0.05"]
                                  == table["sign_significant_holm_0.05"])
    return table


def print_summary(table: pd.DataFrame) -> None:
    """Print a compact, human-readable version of *table*."""
    cols = ["hypothesis", "baseline", "n_pairs", "p_value", "p_holm", "symmetry_p",
            "sign_p_holm", "a12_effective", "who_is_better", "conclusion_robust"]
    with pd.option_context("display.width", 160, "display.float_format", "{:.4g}".format):
        print(table[cols].to_string(index=False))


def main(argv: Optional[list[str]] = None) -> None:
    """Command-line entry point: compute Table 4 and write it to CSV."""
    parser = argparse.ArgumentParser(description="RQ2: hypothesis tests H1a-H3c.")
    parser.add_argument("--input", type=Path, default=paths.RQ2_RESULTS_DIR / RESULTS_NAME,
                        help="per-method results (e.g. reference/rq2/%s)" % RESULTS_NAME)
    parser.add_argument("--out", type=Path, default=paths.RQ2_RESULTS_DIR / TESTS_NAME)
    args = parser.parse_args(argv)
    table = run_tests(pd.read_csv(args.input))
    print_summary(table)
    paths.ensure_dir(args.out.parent)
    table.to_csv(args.out, index=False)
    print(f"Hypothesis tests saved to {args.out}")


if __name__ == "__main__":
    main()
