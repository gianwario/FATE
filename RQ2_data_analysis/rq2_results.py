import pandas as pd
import numpy as np
from scipy.stats import wilcoxon


RQ2_RESULTS_CSV = "RQ2_data_analysis/rq2_all_experiments_results.csv"
SUMMARY_CSV = "RQ2_data_analysis/rq2_summary_results.csv"


HYPOTHESIS_MAP = {
    ("fairness_score", "FairSMOTE"):  "H1a",
    ("fairness_score", "Reweighing"): "H1b",
    ("fairness_score", "DIR"):        "H1c",

    ("performance_score", "FairSMOTE"):  "H2a",
    ("performance_score", "Reweighing"): "H2b",
    ("performance_score", "DIR"):        "H2c",

    ("elapsed_seconds", "FairSMOTE"):  "H3a",
    ("elapsed_seconds", "Reweighing"): "H3b",
    ("elapsed_seconds", "DIR"):        "H3c",
}

METRIC_DIRECTION = {
    "fairness_score":    "lower",   # lower deviation = fairer
    "elapsed_seconds":   "lower",   # faster is better
    "execution_time":    "lower",   # in case you used this name
    "performance_score": "higher",  # higher PR-AUC is better
}

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


def compare_method(df, metric, baseline_label, results_list=None):
    """
    Compare FATE vs `baseline_label` on `metric` over all (dataset_name, prot, model) combos.
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
        "who_is_better": direction_label, # FATE_better / Baseline_better / No_diff / Tie
    }

    if results_list is not None:
        results_list.append(row)

def main():
    df = pd.read_csv(RQ2_RESULTS_CSV)

    summary_rows = []
    for metric in ["fairness_score", "performance_score", "elapsed_seconds"]:
        for baseline in ["FairSMOTE", "Reweighing", "DIR"]:
            compare_method(df, metric, baseline, results_list=summary_rows)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv("RQ2_data_analysis/rq2_hypothesis_tests.csv", index=False)


if __name__ == "__main__":
    main()
