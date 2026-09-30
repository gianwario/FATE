"""
RQ1 best-configuration extractor: top-performing FATE configurations per experimental group.

Reads the FATE grid results (default ``results/fate/experiments_results.csv``)
and extracts the best GA
configuration (by fitness) for three levels of grouping:

1. **(dataset × model × protected_attribute)** – best config per group.
   Saved to ``rq1_fate_results_best_per_group.csv``.
   This file is the primary input for RQ2 experiments
   (``RQ2_data_analysis/preprocessing_experiments.py``).

2. **model** (absolute best across all datasets) – saved to
   ``rq1_fate_results_best_per_model.csv``.

3. **(dataset × protected_attribute)** (best across all models) – saved to
   ``rq1_fate_results_best_per_dataset_attr.csv``.

Configuration:
    ``N_BEST``                – number of top configurations to select per group.
    ``GROUP_BY_PROTECTED_ATTR`` – include protected_attribute in the group key.
"""
import argparse
from pathlib import Path
from typing import Optional

import pandas as pd

import paths

# === Config ===
N_BEST = 1  # top-N per group
GROUP_BY_PROTECTED_ATTR = True  # set False if you want to ignore protected_attribute


def main(argv: Optional[list[str]] = None) -> None:
    """
    Extract and save the best FATE configuration rows across several groupings.

    Reads the experiment results CSV, validates required columns, sorts by
    fitness descending, then uses ``groupby(...).head(N_BEST)`` to select the
    top-N configuration(s) per group.  All results are printed to stdout and
    saved to CSV files for downstream RQ1 / RQ2 analysis.
    """
    parser = argparse.ArgumentParser(description="RQ1: best FATE configuration per group.")
    parser.add_argument("--results", type=Path, default=paths.FATE_RESULTS_CSV)
    parser.add_argument("--out-dir", type=Path, default=paths.RQ1_RESULTS_DIR)
    args = parser.parse_args(argv)
    out_dir = paths.ensure_dir(args.out_dir)

    # Load CSV
    df = pd.read_csv(args.results)

    # Sanity check: required columns
    required_cols = [
        "dataset",
        "model_identifier",
        "protected_attribute",
        "population_size",
        "generations",
        "alpha",
        "beta",
        "techniques",
        "model_used",
        "fitness",
        "fairness_score",
        "performance_score",
        "elapsed_seconds",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    # Decide grouping keys for the main "best per group" summary
    if GROUP_BY_PROTECTED_ATTR:
        group_cols = ["dataset", "model_identifier", "protected_attribute"]
    else:
        group_cols = ["dataset", "model_identifier"]

    print(f"Grouping by: {group_cols}")
    print(f"Selecting top {N_BEST} configuration(s) per group based on fitness.\n")

    # Sort by fitness descending so head(N_BEST) gives top configs
    # Stable sort: among configurations tied on fitness, the first one in file
    # order is kept, so repeated runs on the same input give the same output.
    df_sorted = df.sort_values(by="fitness", ascending=False, kind="mergesort")

    # Several GA configurations often reach the same best pipeline (and hence
    # identical fitness, FS and PS).  All tied configurations are saved, so the
    # configuration reported for each group can be checked against them.
    group_max = df.groupby(group_cols)["fitness"].transform("max")
    tied_best = df[df["fitness"] == group_max].sort_values(group_cols, kind="mergesort")
    tied_best_path = out_dir / "rq1_fate_results_tied_best.csv"
    tied_best.to_csv(tied_best_path, index=False)
    print(f"Saved all configurations tied for best fitness to: {tied_best_path}")

    # 1) Best per (dataset, model, [protected_attribute])
    best_per_group = (
        df_sorted
        .groupby(group_cols, as_index=False)
        .head(N_BEST)
        .reset_index(drop=True)
    )

    # Print a compact summary to stdout
    print("=== Best per group ===\n")
    for _, row in best_per_group.iterrows():
        print("=" * 80)
        print(f"Dataset           : {row['dataset']}")
        print(f"Model             : {row['model_identifier']}")
        if GROUP_BY_PROTECTED_ATTR:
            print(f"Protected attr    : {row['protected_attribute']}")
        print(f"Population size   : {row['population_size']}")
        print(f"Generations       : {row['generations']}")
        print(f"Crossover (alpha) : {row['alpha']}")
        print(f"Mutation (beta)   : {row['beta']}")
        print(f"Techniques        : {row['techniques']}")
        print(f"Fitness           : {row['fitness']:.6f}")
        print(f"Fairness score    : {row['fairness_score']:.6f}")
        print(f"Performance score : {row['performance_score']:.6f}")
        print(f"Elapsed seconds   : {row['elapsed_seconds']:.2f}")
        print("=" * 80)
        print()

    # Save best per group
    base = out_dir / "rq1_fate_results"
    best_per_group_path = f"{base}_best_per_group.csv"
    best_per_group.to_csv(best_per_group_path, index=False)
    print(f"Saved best configurations per group to: {best_per_group_path}")

    # 2) Absolute best per model (across all datasets, protected attributes, and params)
    print("\n=== Best per model (absolute) ===\n")
    best_per_model = (
        df_sorted
        .groupby("model_identifier", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )

    for _, row in best_per_model.iterrows():
        print("-" * 80)
        print(f"Model             : {row['model_identifier']}")
        print(f"Dataset           : {row['dataset']}")
        print(f"Protected attr    : {row['protected_attribute']}")
        print(f"Population size   : {row['population_size']}")
        print(f"Generations       : {row['generations']}")
        print(f"Crossover (alpha) : {row['alpha']}")
        print(f"Mutation (beta)   : {row['beta']}")
        print(f"Techniques        : {row['techniques']}")
        print(f"Fitness           : {row['fitness']:.6f}")
        print(f"Fairness score    : {row['fairness_score']:.6f}")
        print(f"Performance score : {row['performance_score']:.6f}")
        print(f"Elapsed seconds   : {row['elapsed_seconds']:.2f}")
        print("-" * 80)
        print()

    best_per_model_path = f"{base}_best_per_model.csv"
    best_per_model.to_csv(best_per_model_path, index=False)
    print(f"Saved absolute best per model to: {best_per_model_path}")

    # 3) Absolute best per (dataset, protected_attribute) across all models
    print("\n=== Best per (dataset, protected_attribute) (absolute across models) ===\n")
    best_per_dataset_attr = (
        df_sorted
        .groupby(["dataset", "protected_attribute"], as_index=False)
        .head(1)
        .reset_index(drop=True)
    )

    for _, row in best_per_dataset_attr.iterrows():
        print("-" * 80)
        print(f"Dataset           : {row['dataset']}")
        print(f"Protected attr    : {row['protected_attribute']}")
        print(f"Model             : {row['model_identifier']}")
        print(f"Population size   : {row['population_size']}")
        print(f"Generations       : {row['generations']}")
        print(f"Crossover (alpha) : {row['alpha']}")
        print(f"Mutation (beta)   : {row['beta']}")
        print(f"Techniques        : {row['techniques']}")
        print(f"Fitness           : {row['fitness']:.6f}")
        print(f"Fairness score    : {row['fairness_score']:.6f}")
        print(f"Performance score : {row['performance_score']:.6f}")
        print(f"Elapsed seconds   : {row['elapsed_seconds']:.2f}")
        print("-" * 80)
        print()

    best_per_dataset_attr_path = f"{base}_best_per_dataset_attr.csv"
    best_per_dataset_attr.to_csv(best_per_dataset_attr_path, index=False)
    print(
        f"Saved absolute best per (dataset, protected_attribute) to: {best_per_dataset_attr_path}"
    )


if __name__ == "__main__":
    main()
