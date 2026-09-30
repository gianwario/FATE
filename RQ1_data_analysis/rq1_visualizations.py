#!/usr/bin/env python3
"""
RQ1 visualisations: boxplots comparing FATE against baselines and sensitivity line plots.

Generates publication-ready figures for RQ1, saved as both PDF (vector) and
PNG (300 dpi raster) in ``results/rq1/figures/``.

Figures produced:
    ``rq1_fitness_boxplot``        – FATE vs baselines on combined fitness.
    ``rq1_fairness_boxplot``       – FATE vs baselines on fairness deviation (FS).
    ``rq1_performance_boxplot``    – FATE vs baselines on performance (PR-AUC).
    ``fitness_vs_population_size`` – mean fitness as a function of population size.
    ``fitness_vs_generations``     – mean fitness as a function of generation count.
    ``fitness_vs_alpha``           – mean fitness as a function of crossover rate.
    ``fitness_vs_beta``            – mean fitness as a function of mutation rate.

Input files (expected in ``results/rq1/``, produced by the other RQ1 scripts):
    ``rq1_fate_vs_baselines.csv``
    ``rq1_fate_by_population_size.csv``
    ``rq1_fate_by_generations.csv``
    ``rq1_fate_by_alpha.csv``
    ``rq1_fate_by_beta.csv``
"""
import argparse
import os
from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import paths

matplotlib.use("Agg")  # file output only; no display needed

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
sns.set(style="whitegrid", context="paper", font_scale=1.1)


# ---------------------------------------------------------------------
# 1. FATE vs baselines – overall fitness / fairness / performance
# ---------------------------------------------------------------------
def plot_fate_vs_baselines(data_dir: Path, out_dir: Path) -> None:
    """
    Produce side-by-side boxplots comparing FATE against the two static baselines.

    Reads ``<data_dir>/rq1_fate_vs_baselines.csv``, melts it to long format for each of
    the three metrics (fitness, fairness deviation, performance/PR-AUC), and
    plots a boxplot with three columns: FATE, "All practices", "No practices".

    Figures saved
    -------------
    ``<out_dir>/rq1_fitness_boxplot.{pdf,png}``
    ``<out_dir>/rq1_fairness_boxplot.{pdf,png}``
    ``<out_dir>/rq1_performance_boxplot.{pdf,png}``
    """
    path = os.path.join(data_dir, "rq1_fate_vs_baselines.csv")
    df = pd.read_csv(path)

    # Long format for fitness
    fitness_long = df.melt(
        id_vars=["dataset", "protected_attribute", "model"],
        value_vars=["FATE_fitness", "fitness_all", "fitness_none"],
        var_name="method",
        value_name="fitness",
    )
    name_map = {
        "FATE_fitness": "FATE",
        "fitness_all": "All practices",
        "fitness_none": "No practices",
    }
    fitness_long["method"] = fitness_long["method"].map(name_map)

    plt.figure(figsize=(6, 4))
    sns.boxplot(data=fitness_long, x="method", y="fitness", color="#E1D5E7")
    plt.xlabel("")
    plt.ylabel("Fitness")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "rq1_fitness_boxplot.pdf"))
    plt.savefig(os.path.join(out_dir, "rq1_fitness_boxplot.png"), dpi=300)
    plt.close()

    # Same idea for fairness deviation
    fairness_long = df.melt(
        id_vars=["dataset", "protected_attribute", "model"],
        value_vars=["FATE_fairness", "fairness_all", "fairness_none"],
        var_name="method",
        value_name="fairness",
    )
    name_map_f = {
        "FATE_fairness": "FATE",
        "fairness_all": "All practices",
        "fairness_none": "No practices",
    }
    fairness_long["method"] = fairness_long["method"].map(name_map_f)

    plt.figure(figsize=(6, 4))
    sns.boxplot(data=fairness_long, x="method", y="fairness", color="#E1D5E7")
    plt.xlabel("")
    plt.ylabel("Fairness deviation (FS)")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "rq1_fairness_boxplot.pdf"))
    plt.savefig(os.path.join(out_dir, "rq1_fairness_boxplot.png"), dpi=300)
    plt.close()

    # And performance
    perf_long = df.melt(
        id_vars=["dataset", "protected_attribute", "model"],
        value_vars=["FATE_performance", "performance_all", "performance_none"],
        var_name="method",
        value_name="performance",
    )
    name_map_p = {
        "FATE_performance": "FATE",
        "performance_all": "All practices",
        "performance_none": "No practices",
    }
    perf_long["method"] = perf_long["method"].map(name_map_p)

    plt.figure(figsize=(6, 4))
    sns.boxplot(data=perf_long, x="method", y="performance", color="#E1D5E7")
    plt.xlabel("")
    plt.ylabel("Performance (PR-AUC)")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "rq1_performance_boxplot.pdf"))
    plt.savefig(os.path.join(out_dir, "rq1_performance_boxplot.png"), dpi=300)
    plt.close()


# ---------------------------------------------------------------------
# 2. Sensitivity plots for GA parameters
#    (population size, generations, alpha, beta)
# ---------------------------------------------------------------------
def plot_param_sensitivity(data_dir: Path, out_dir: Path, filename: str, param_col: str,
                           y_col: str = "fitness_mean", hue_col: Optional[str] = None,
                           out_stub: Optional[str] = None) -> None:
    """
    Generic line plot for GA parameter sensitivity.

    Parameters
    ----------
    data_dir : Path
        Folder containing *filename*.
    out_dir : Path
        Folder where the figure is written.
    filename : str
        CSV file relative to *data_dir* containing aggregated statistics per
        parameter value (produced by ``configurations_results.summarize_group``).
    param_col : str
        Column name for the x-axis (e.g., ``'population_size'``).
    y_col : str, optional
        Metric column for the y-axis (default ``'fitness_mean'``).
    hue_col : str or None, optional
        Optional column for coloured line grouping (e.g., ``'dataset'``).
    out_stub : str or None, optional
        Output file base name without extension.  Defaults to
        ``'{param_col}_{y_col}'``.

    Raises
    ------
    FileNotFoundError
        If the input CSV does not exist (run ``configurations_results`` first).
    """
    path = os.path.join(data_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found; run configurations_results first.")

    df = pd.read_csv(path)

    plt.figure(figsize=(6, 4))
    sns.lineplot(data=df, x=param_col, y=y_col, hue=hue_col, marker="o")
    plt.xlabel(param_col.replace("_", " ").title())
    plt.ylabel(y_col.replace("_", " ").title())
    plt.tight_layout()

    stub = out_stub or f"{param_col}_{y_col}"
    plt.savefig(os.path.join(out_dir, f"{stub}.pdf"))
    plt.savefig(os.path.join(out_dir, f"{stub}.png"), dpi=300)
    plt.close()


def make_all_param_plots(data_dir: Path, out_dir: Path) -> None:
    """
    Generate sensitivity line plots for all four GA hyperparameters.

    Calls ``plot_param_sensitivity`` for population size, number of
    generations, alpha (crossover rate), and beta (mutation rate), using the
    per-hyperparameter summary CSVs produced by
    ``configurations_results.summarize_group``.
    """
    plot_param_sensitivity(
        data_dir, out_dir,
        filename="rq1_fate_by_population_size.csv",
        param_col="population_size",
        y_col="fitness_mean",
        out_stub="fitness_vs_population_size",
    )

    plot_param_sensitivity(
        data_dir, out_dir,
        filename="rq1_fate_by_generations.csv",
        param_col="generations",
        y_col="fitness_mean",
        out_stub="fitness_vs_generations",
    )

    plot_param_sensitivity(
        data_dir, out_dir,
        filename="rq1_fate_by_alpha.csv",
        param_col="alpha",
        y_col="fitness_mean",
        out_stub="fitness_vs_alpha",
    )

    plot_param_sensitivity(
        data_dir, out_dir,
        filename="rq1_fate_by_beta.csv",
        param_col="beta",
        y_col="fitness_mean",
        out_stub="fitness_vs_beta",
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main(argv: Optional[list[str]] = None) -> None:
    """Generate all RQ1 figures (Fig. 2 and Fig. 3 of the paper)."""
    parser = argparse.ArgumentParser(description="RQ1: figures.")
    parser.add_argument("--rq1-dir", type=Path, default=paths.RQ1_RESULTS_DIR,
                        help="folder with the RQ1 CSVs (e.g. reference/rq1 for the paper's)")
    parser.add_argument("--out-dir", type=Path, default=paths.RQ1_FIGURES_DIR)
    args = parser.parse_args(argv)
    out_dir = paths.ensure_dir(args.out_dir)
    print("Generating RQ1 visualizations...")
    plot_fate_vs_baselines(args.rq1_dir, out_dir)
    make_all_param_plots(args.rq1_dir, out_dir)
    print(f"Figures saved in: {out_dir}")


if __name__ == "__main__":
    main()
