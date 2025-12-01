import pandas as pd
import numpy as np
from scipy.stats import wilcoxon


RQ2_RESULTS_CSV = "rq2_all_results.csv"


def vargha_delaney_a12(x, y):
    """
    Compute Vargha–Delaney A12 effect size:
    A12 = P(X > Y) + 0.5 * P(X == Y)
    """
    x = np.array(x)
    y = np.array(y)
    nx = len(x)
    ny = len(y)
    ranks = np.argsort(np.concatenate([x, y]))
    # simpler implementation:
    # use rankdata if you want exact ties, but this simple form is often enough:
    combined = np.concatenate([x, y])
    from scipy.stats import rankdata
    r = rankdata(combined)
    rx = r[:nx].sum()
    a12 = (rx / nx - (nx + 1) / 2) / ny
    return a12


def compare_method(df, metric, baseline_label):
    """
    Compare FATE vs `baseline_label` on `metric` over all (dataset, prot, model) combos.
    """
    # pivot so that each row has columns FATE and baseline_label for the given metric
    pivot = (
        df.pivot_table(
            index=["dataset", "protected_attribute", "model_identifier"],
            columns="method",
            values=metric,
        )
    )

    if "FATE" not in pivot.columns or baseline_label not in pivot.columns:
        print(f"[WARN] Missing FATE or {baseline_label} for metric {metric}")
        return

    sub = pivot[["FATE", "baseline_label"]].dropna()
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

    a12 = vargha_delaney_a12(x, y)

    print(f"\n=== {metric} : FATE vs {baseline_label} ===")
    print(f"n = {len(x)}")
    print(f"FATE   mean = {np.nanmean(x):.4f}")
    print(f"{baseline_label:7s} mean = {np.nanmean(y):.4f}")
    print(f"Wilcoxon p-value = {p:.4f}")
    print(f"Vargha–Delaney A12 (FATE vs {baseline_label}) = {a12:.3f}")
    if p < 0.05:
        if a12 > 0.5:
            print(" -> Significant: FATE tends to perform better.")
        else:
            print(" -> Significant: baseline tends to perform better.")
    else:
        print(" -> No statistically significant difference.")


def main():
    df = pd.read_csv(RQ2_RESULTS_CSV)

    baselines = ["FairSMOTE", "Reweighing", "DIR"]
    metrics = ["fairness_score", "performance_score", "elapsed_seconds"]

    for b in baselines:
        for m in metrics:
            compare_method(df, m, b)


if __name__ == "__main__":
    main()
