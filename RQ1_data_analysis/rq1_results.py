"""
RQ1 comparative analysis: FATE vs static baselines (all-practices, no-practices).

This script has two steps, both run by ``main()``:

1. ``compute_baselines``: evaluates the two baseline configurations, applying
   *all* eight fairness-aware practices and applying *none*, with the same
   ``fitness.fitness`` function used by the GA.  Output:
   ``results/rq1/rq1_baselines_results.csv``.
2. ``compare_with_baselines``: merges the best FATE configurations
   (``results/rq1/rq1_fate_results_best_per_group.csv``, produced by
   ``model_dataset_results``) with the baseline results, computes fitness
   deltas (``FATE_fitness - baseline_fitness``) and writes
   ``results/rq1/rq1_fate_vs_baselines.csv``.

The comparison answers the core question of RQ1: does FATE find better
fairness–performance trade-offs than the trivial baselines?
"""
import argparse
import os
from pathlib import Path
from typing import Optional

import pandas as pd

import paths
import preprocessing as _preproc
from experiment_config import DATASETS, TECHNIQUES
from fitness import fitness
from preprocessing import prepare_data_model as _prepare_data_model

BEST_PER_GROUP_NAME = "rq1_fate_results_best_per_group.csv"
BASELINES_NAME = "rq1_baselines_results.csv"
OUT_NAME = "rq1_fate_vs_baselines.csv"


def compute_baselines(out_path: Path) -> None:
    """
    Evaluate the two static baselines (all techniques and no techniques) for all groups.

    For each combination of (dataset × protected_attribute × model), calls
    ``fitness.fitness`` twice:

    - ``baseline='all'``: applies all eight fairness techniques simultaneously.
    - ``baseline='none'``: applies no techniques (raw dataset).

    Parameters
    ----------
    out_path : Path
        Destination CSV (by default ``results/rq1/rq1_baselines_results.csv``).
    """
    datasets = DATASETS
    models = ["lr", "rf", "svc", "xgb"]
    techniques = list(TECHNIQUES)

    results = []  # <- MUST BE A LIST, not a dict

    for ds_cfg in datasets:
        raw = pd.read_csv(paths.REPO_ROOT / ds_cfg['path'])
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
    paths.ensure_dir(out_path.parent)
    results_df.to_csv(out_path, index=False)
    print(f"Results saved to {out_path}")


def normalize_dataset_path(p: object) -> str:
    """Normalise a dataset path to its name, e.g. ``datasets/adult.csv`` -> ``adult``."""
    base = os.path.basename(str(p))
    name, _ = os.path.splitext(base)
    return name


def compare_with_baselines(best_csv: Path, baselines_csv: Path, out_csv: Path) -> None:
    """
    Merge the best FATE configurations with the two baselines and compute deltas.

    Parameters
    ----------
    best_csv : Path
        Best configuration per group (output of ``model_dataset_results``).
    baselines_csv : Path
        Output of ``compute_baselines``.
    out_csv : Path
        Destination of the comparison table (``rq1_fate_vs_baselines.csv``).
    """
    # --------------------------
    # Load FATE results
    # --------------------------
    fate = pd.read_csv(best_csv)

    # Drop errored rows if any
    if "error" in fate.columns:
        fate = fate[fate["error"].isna()]

    fate["dataset_name"] = fate["dataset"].apply(normalize_dataset_path)

    # --------------------------
    # Load baselines
    # --------------------------
    base = pd.read_csv(baselines_csv)

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
    merged.to_csv(out_csv, index=False)
    print(f"Comparison saved to {out_csv}")

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


def main(argv: Optional[list[str]] = None) -> None:
    """Compute the two RQ1 baselines and compare them with the best FATE pipelines."""
    parser = argparse.ArgumentParser(description="RQ1: FATE vs no/all-practice baselines.")
    parser.add_argument("--rq1-dir", type=Path, default=paths.RQ1_RESULTS_DIR,
                        help="folder with the best-per-group CSV; outputs are written here")
    parser.add_argument("--baselines", type=Path, default=None,
                        help="use an existing baselines CSV instead of recomputing it")
    args = parser.parse_args(argv)
    baselines_csv = args.baselines
    if baselines_csv is None:
        baselines_csv = args.rq1_dir / BASELINES_NAME
        compute_baselines(baselines_csv)
    compare_with_baselines(args.rq1_dir / BEST_PER_GROUP_NAME, baselines_csv,
                           args.rq1_dir / OUT_NAME)


if __name__ == "__main__":
    main()
