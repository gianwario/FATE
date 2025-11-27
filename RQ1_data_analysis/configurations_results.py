import sys
import numpy as np
import pandas as pd
csv_path = "../output/experiments_results.csv"  # results CSV path


def summarize_group(df, group_cols, prefix):
    """
    Helper to aggregate metrics over a given grouping.
    Returns the aggregated DataFrame.
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
    output_path = f"{prefix}.csv"
    agg.to_csv(output_path, index=False)
    print(f"Saved summary: {output_path}")
    return agg


def main():
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
    full_summary = summarize_group(df, full_group_cols, "rq1_full_paramgrid_summary.csv")

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
    pop_summary = summarize_group(df, ["population_size"], "rq1_by_population_size.csv")

    # By number of generations
    gen_summary = summarize_group(df, ["generations"], "rq1_by_generations.csv")

    # By alpha (we treat as crossover rate)
    alpha_summary = summarize_group(df, ["alpha"], "rq1_by_alpha.csv")  

    # By beta (we treat as mutation rate)
    beta_summary = summarize_group(df, ["beta"], "rq1_by_beta.csv")

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
