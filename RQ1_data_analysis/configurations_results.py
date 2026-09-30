"""
RQ1 parameter-sensitivity analysis: aggregate fitness statistics per GA hyperparameter
configuration.

Reads the full parameter-grid experiment results from
``output/experiments_results.csv`` and produces summary CSVs aggregating
fitness, fairness, performance, and elapsed-time statistics grouped by:

1. Full (population_size × generations × alpha × beta) configuration.
2. Population size alone.
3. Number of generations alone.
4. Alpha (crossover rate) alone.
5. Beta (mutation rate) alone.

Used to answer the parameter-sensitivity sub-question of RQ1: how do
individual GA hyperparameters affect FATE's optimisation outcome?

Output files (written to the working directory):
    ``rq1_fate_full_paramgrid_summary.csv``
    ``rq1_fate_by_population_size.csv``
    ``rq1_fate_by_generations.csv``
    ``rq1_fate_by_alpha.csv``
    ``rq1_fate_by_beta.csv``
"""
import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import paths


def summarize_group(df: pd.DataFrame, group_cols: list[str], prefix: str,
                    out_dir: Path) -> pd.DataFrame:
    """
    Aggregate GA result metrics over a set of grouping columns and save to CSV.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned experiment results containing columns ``fitness``,
        ``fairness_score``, ``performance_score``, ``elapsed_seconds``, and
        all columns listed in *group_cols*.
    group_cols : list of str
        Column names to group by (e.g., ``['population_size']``).
    prefix : str
        Base name for the output CSV file, without extension; written to
        ``<out_dir>/<prefix>.csv``.
    out_dir : Path
        Output folder.

    Returns
    -------
    pd.DataFrame
        Aggregated DataFrame with mean, median, and std for each metric,
        plus ``n_runs`` (count of rows in each group).
    """
    agg = (
        df.groupby(group_cols)
        .agg(
            n_runs=("fitness", "count"),
            fitness_mean=("fitness", "mean"),
            fitness_median=("fitness", "median"),
            fitness_std=("fitness", "std"),
            fairness_mean=("fairness_score", "mean"),
            fairness_median=("fairness_score", "median"),
            fairness_std=("fairness_score", "std"),
            perf_mean=("performance_score", "mean"),
            perf_median=("performance_score", "median"),
            perf_std=("performance_score", "std"),
            time_mean=("elapsed_seconds", "mean"),
            time_median=("elapsed_seconds", "median"),
            time_std=("elapsed_seconds", "std"),
        )
        .reset_index()
    )
    output_path = paths.ensure_dir(out_dir) / f"{prefix}.csv"
    agg.to_csv(output_path, index=False)
    print(f"Saved summary: {output_path}")
    return agg


def main(argv: Optional[list[str]] = None) -> None:
    """
    Load, clean, and summarise the full parameter-grid results for RQ1 sensitivity analysis.

    Steps:

    1. Load the FATE grid results (default ``results/fate/experiments_results.csv``;
       pass ``--results reference/fate/experiments_results.csv`` for the paper's run).
    2. Coerce metric columns to numeric; replace inf / -inf with NaN; drop rows
       with any invalid metric value.
    3. Call ``summarize_group`` for the full parameter grid and for each single
       hyperparameter (population_size, generations, alpha, beta).
    4. Print the top-10 parameter configurations by mean fitness to stdout.
    """
    parser = argparse.ArgumentParser(description="RQ1: GA parameter sensitivity summaries.")
    parser.add_argument("--results", type=Path, default=paths.FATE_RESULTS_CSV)
    parser.add_argument("--out-dir", type=Path, default=paths.RQ1_RESULTS_DIR)
    args = parser.parse_args(argv)
    csv_path = args.results
    out_dir = args.out_dir
    df = pd.read_csv(csv_path)

    # Basic sanity check
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

    print(f"Loaded {len(df)} rows from {csv_path}")
    # ---- Clean invalid rows (NaN / inf / -inf in key metrics) ----
    metric_cols = ["fitness", "fairness_score", "performance_score", "elapsed_seconds"]

    # 1) Force numeric conversion; invalid parsing -> NaN
    for col in metric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 2) Replace +/-inf with NaN
    df[metric_cols] = df[metric_cols].replace([np.inf, -np.inf], np.nan)

    before = len(df)
    df_clean = df.dropna(subset=metric_cols)
    after = len(df_clean)

    dropped = before - after
    if dropped > 0:
        print(f"Dropped {dropped} row(s) with invalid metric values (NaN/inf/-inf).")
    else:
        print("No invalid metric values detected after cleaning.")

    # 3) Hard sanity check: there should be no infinities left
    if np.isinf(df_clean[metric_cols].to_numpy()).any():
        raise RuntimeError("There are still inf/-inf values in cleaned metrics after filtering!")

    # ---- 1) Full parameter configuration summary ----
    print("\n[1] Summarizing per full GA parameter configuration...")
    full_group_cols = ["population_size", "generations", "alpha", "beta"]
    full_summary = summarize_group(df, full_group_cols, "rq1_fate_full_paramgrid_summary",
                                   out_dir)

    # Show a few best configs by mean fitness
    print("\nTop 10 parameter configs by mean fitness:")
    print(
        full_summary.sort_values("fitness_mean", ascending=False)
                    .head(10)
                    .to_string(index=False)
    )

    # ---- 2) Aggregated per single hyperparameter ----
    print("\n[2] Summarizing per single GA hyperparameter...")

    # By population size
    pop_summary = summarize_group(df, ["population_size"], "rq1_fate_by_population_size",
                                  out_dir)

    # By number of generations
    gen_summary = summarize_group(df, ["generations"], "rq1_fate_by_generations", out_dir)

    # By alpha (we treat as crossover rate)
    alpha_summary = summarize_group(df, ["alpha"], "rq1_fate_by_alpha", out_dir)

    # By beta (we treat as mutation rate)
    beta_summary = summarize_group(df, ["beta"], "rq1_fate_by_beta", out_dir)

    # Show short previews
    print("\nPopulation size summary (sorted by fitness_mean):")
    print(pop_summary.sort_values("fitness_mean", ascending=False).to_string(index=False))

    print("\nGenerations summary (sorted by fitness_mean):")
    print(gen_summary.sort_values("fitness_mean", ascending=False).to_string(index=False))

    print("\nAlpha (crossover) summary (sorted by fitness_mean):")
    print(alpha_summary.sort_values("fitness_mean", ascending=False).to_string(index=False))

    print("\nBeta (mutation) summary (sorted by fitness_mean):")
    print(beta_summary.sort_values("fitness_mean", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
