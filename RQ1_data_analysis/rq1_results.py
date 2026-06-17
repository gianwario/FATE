"""
RQ1 comparative analysis: FATE vs static baselines (all-practices, no-practices).

This script has two responsibilities:

1. **``compute_baselines()``** (called on demand by uncommenting in ``__main__``):
   Evaluates two baseline configurations — applying *all* eight fairness
   techniques simultaneously and applying *none* — using the same
   ``fitness.fitness`` function as the GA.  Saves results to
   ``RQ1_data_analysis/rq1_baseline_results.csv``.

2. **Main block** (default execution):
   Merges the best FATE results (``rq1_fate_results_best_per_group.csv``)
   with the pre-computed baseline results, computes fitness deltas
   (``FATE_fitness − baseline_fitness``), and writes the comparison table to
   ``RQ1_data_analysis/rq1_fate_vs_baselines.csv``.

The comparison answers the core question of RQ1: does FATE find better
fairness–performance trade-offs than the trivial baselines?
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  # noqa: E402
from fitness import fitness  # noqa: E402
import pandas as pd  # noqa: E402
from preprocessing import prepare_data_model as _prepare_data_model  # noqa: E402
import preprocessing as _preproc  # noqa: E402

# ---- paths (adapt if needed) ----
FATE_CSV = "RQ1_data_analysis/rq1_fate_results_best_per_group.csv"
BASELINES_CSV = "RQ1_data_analysis/rq1_baselines_results.csv"
OUT_CSV = "RQ1_data_analysis/rq1_fate_vs_baselines.csv"


def compute_baselines():
    """
    Evaluate the two static baselines (all techniques and no techniques) for all groups.

    For each combination of (dataset × protected_attribute × model), calls
    ``fitness.fitness`` twice:

    - ``baseline='all'``: applies all eight fairness techniques simultaneously.
    - ``baseline='none'``: applies no techniques (raw dataset).

    Results are saved to ``RQ1_data_analysis/rq1_baseline_results.csv``.

    Notes
    -----
    This function is commented out in the ``__main__`` block by default.  Run
    it once to generate the baseline CSV, then run the main block to produce
    the comparison table.
    """
    datasets = [
        {
            'name': 'adult',
            'path': 'datasets/adult.csv',
            'preparer_name': 'prepare_adult',
            'protected_attributes': ['race', 'sex'],
            'target': 'salary'
        },
        {
            'name': 'german',
            'path': 'datasets/german.csv',
            'preparer_name': 'prepare_german',
            'protected_attributes': ['sex', 'age'],
            'target': 'Target'
        },
        {
            'name': 'heart',
            'path': 'datasets/heart.csv',
            'preparer_name': 'prepare_heart',
            'protected_attributes': ['sex', 'age'],
            'target': 'num'
        }
    ]
    models = ["lr", "rf", "svc", "xgb"]
    techniques = ['standard', 'stratified_sampling', 'oversampling', 'undersampling',
                  'clustering', 'ipw', 'matching', 'min_max_scaling']

    results = []  # <- MUST BE A LIST, not a dict

    for ds_cfg in datasets:
        raw = pd.read_csv(ds_cfg['path'])
        preparer = getattr(_preproc, ds_cfg['preparer_name'])
        processed = preparer(raw)

        for prot in ds_cfg['protected_attributes']:
            sample_ready = _prepare_data_model(
                processed,
                ds_cfg['target'],
                protected_attribute=prot,
                binarize=False
            )

            for model_id in models:
                # --- baseline: ALL techniques ---
                baseline_all = fitness(sample_ready, techniques, model_id, prot, ds_cfg['target'])
                results.append({
                    "dataset": ds_cfg['name'],
                    "protected_attribute": prot,
                    "model": model_id,
                    "baseline": "all",
                    "techniques": techniques,
                    "fitness": baseline_all[0],
                    "fairness": baseline_all[1],
                    "performance": baseline_all[2],
                })

                # --- baseline: NO techniques ---
                baseline_none = fitness(sample_ready, [], model_id, prot, ds_cfg['target'])
                results.append({
                    "dataset": ds_cfg['name'],
                    "protected_attribute": prot,
                    "model": model_id,
                    "baseline": "none",
                    "techniques": [],
                    "fitness": baseline_none[0],
                    "fairness": baseline_none[1],
                    "performance": baseline_none[2],
                })

    # --- Save all results to CSV ---
    results_df = pd.DataFrame(results)
    out_path = "RQ1_data_analysis/rq1_baseline_results.csv"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    results_df.to_csv(out_path, index=False)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    # compute_baselines()
    # --------------------------
    # Load FATE results
    # --------------------------
    fate = pd.read_csv(FATE_CSV)

    # Drop errored rows if any
    if "error" in fate.columns:
        fate = fate[fate["error"].isna()]

    # Normalize dataset name: datasets/adult.csv -> adult
    def normalize_dataset_path(p):
        base = os.path.basename(str(p))
        name, _ = os.path.splitext(base)
        return name

    fate["dataset_name"] = fate["dataset"].apply(normalize_dataset_path)

    # --------------------------
    # Load baselines
    # --------------------------
    base = pd.read_csv(BASELINES_CSV)

    # Normalize model names: svm -> svc to match FATE output
    base["model_norm"] = base["model"].replace({"svm": "svc"})

    # --------------------------
    # Pivot baselines to wide format:
    # index: (dataset, protected_attribute, model_norm)
    # columns: fitness_all, fitness_none, ...
    # --------------------------
    base_wide = base.pivot_table(
        index=["dataset", "protected_attribute", "model_norm"],
        columns="baseline",
        values=["fitness", "fairness", "performance"],
        aggfunc="mean",
    )

    # Flatten MultiIndex columns: (fitness, all) -> fitness_all
    base_wide.columns = [
        f"{metric}_{baseline}"
        for metric, baseline in base_wide.columns.to_flat_index()
    ]
    base_wide = base_wide.reset_index()

    # --------------------------
    # Prepare FATE frame for merge
    # --------------------------
    fate_small = fate[
        [
            "dataset_name",
            "protected_attribute",
            "model_identifier",
            "population_size",
            "generations",
            "alpha",
            "beta",
            "techniques",
            "fitness",
            "fairness_score",
            "performance_score",
        ]
    ].copy()

    fate_small = fate_small.rename(
        columns={
            "dataset_name": "dataset",
            "model_identifier": "model",
            "fitness": "FATE_fitness",
            "fairness_score": "FATE_fairness",
            "performance_score": "FATE_performance",
        }
    )

    # --------------------------
    # Merge FATE with baselines
    # --------------------------
    merged = fate_small.merge(
        base_wide,
        left_on=["dataset", "protected_attribute", "model"],
        right_on=["dataset", "protected_attribute", "model_norm"],
        how="left",
    )

    # We no longer need model_norm
    merged = merged.drop(columns=["model_norm"])

    # --------------------------
    # Compute deltas and flags
    # --------------------------
    # Convenience: ensure numeric
    for col in [
        "FATE_fitness",
        "FATE_fairness",
        "FATE_performance",
        "fitness_all",
        "fitness_none",
        "fairness_all",
        "fairness_none",
        "performance_all",
        "performance_none",
    ]:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce")

    # FATE vs baseline_all
    merged["delta_fitness_all"] = merged["FATE_fitness"] - merged["fitness_all"]
    merged["delta_fitness_none"] = merged["FATE_fitness"] - merged["fitness_none"]

    merged["FATE_better_than_all"] = merged["delta_fitness_all"] > 0
    merged["FATE_better_than_none"] = merged["delta_fitness_none"] > 0

    # Optional: combined flag
    merged["FATE_better_than_both"] = (
        merged["FATE_better_than_all"] & merged["FATE_better_than_none"]
    )

    # --------------------------
    # Save to CSV
    # --------------------------
    merged.to_csv(OUT_CSV, index=False)
    print(f"Comparison saved to {OUT_CSV}")

    # --------------------------
    # Quick console summary
    # --------------------------
    total = len(merged)
    both = merged["FATE_better_than_both"].sum()
    only_all = (merged["FATE_better_than_all"] & ~merged["FATE_better_than_none"]).sum()
    only_none = (~merged["FATE_better_than_all"] & merged["FATE_better_than_none"]).sum()

    print(f"\nTotal groups: {total}")
    print(f"FATE better than BOTH baselines (fitness): {both} ({both/total:.1%})")
    print(f"FATE better than ONLY 'all' baseline:      {only_all} ({only_all/total:.1%})")
    print(f"FATE better than ONLY 'none' baseline:     {only_none} ({only_none/total:.1%})")
