import pandas as pd
import numpy as np
from scipy.stats import shapiro

# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------
CSV_PATH = "RQ2_data_analysis/rq2_all_experiments_results.csv"  # <-- adjust if needed

# column names in your combined CSV:
# should contain: dataset_name, protected_attribute, model_identifier, method,
# fairness_score, performance_score, elapsed_seconds, ...
METRICS = ["fairness_score", "performance_score", "elapsed_seconds"]
BASELINES = ["FairSMOTE", "Reweighing", "DIR"]
FATE_LABEL = "FATE"

# ---------------------------------------------------------------------
# HELPER: check assumptions for one metric + one baseline
# ---------------------------------------------------------------------
def check_assumptions(df: pd.DataFrame, metric: str, baseline_label: str):
    """
    For a given metric and baseline method, build paired samples
    (FATE vs baseline) and run Shapiro–Wilk normality test on the
    *differences* (FATE - baseline).

    Prints a short summary you can cite in the paper.
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
if __name__ == "__main__":
    df = pd.read_csv(CSV_PATH)

    for metric in METRICS:
        for baseline in BASELINES:
            check_assumptions(df, metric, baseline)
