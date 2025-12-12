# Replication Package — Data Preparation for Fairness–Performance Trade-Offs

This repository contains the full replication package for the paper:

> **Data Preparation for Fairness–Performance Trade-Offs: A Practitioner-Friendly Alternative?**

The package includes all datasets, scripts, and results required to reproduce the experiments and analyses for **RQ1** and **RQ2**, as well as the implementation of **FATE**, our genetic-algorithm-based optimization technique for fairness-aware data preparation.

---

## Repository Structure

```
FATE/
│
├── datasets/                  # Input datasets used in the study
│   ├── adult.csv
│   ├── german.csv
│   └── heart.csv
│
├── output/                    # Raw experiment outputs
│   └── experiments_results.csv
│
├── RQ1_data_analysis/         # Results and analysis for RQ1
│   ├── visualizations/        # Generated plots (fitness trends, boxplots)
│   ├── rq1_baselines_results.csv
│   ├── rq1_fate_vs_baselines.csv
│   ├── rq1_fate_by_alpha.csv
│   ├── rq1_fate_by_beta.csv
│   ├── rq1_fate_by_generations.csv
│   ├── rq1_fate_by_population_size.csv
│   ├── rq1_fate_full_paramgrid_summary.csv
│   ├── rq1_fate_results_best_per_dataset_attr.csv
│   ├── rq1_fate_results_best_per_group.csv
│   ├── rq1_fate_results_best_per_model.csv
│   ├── rq1_results.py
│   └── rq1_visualizations.py
│
├── RQ2_data_analysis/         # Results and analysis for RQ2
│   ├── assumptions.py         # Statistical assumption checks
│   ├── preprocessing_experiments.py
│   ├── rq2_all_experiments_results.csv
│   ├── rq2_hypothesis_tests.csv
│   └── rq2_results.py
│
├── fitness.py                 # Fitness computation (fairness + performance)
├── genetic_algorithm.py       # Core GA implementation (FATE)
├── practices.py               # Fairness-aware data preparation practices
├── preprocessing.py           # Dataset preparation utilities
├── create_and_save_model.py   # Model training utilities
├── main.py                    # Main experiment runner
│
├── experiments_cache.csv      # Cache for RQ1 experiments
├── rq2_experiments_cache.csv  # Cache for RQ2 experiments
│
├── requirements.txt           # Python dependencies
└── README.md
```

---

## Experimental Overview

### RQ1 — Optimization Behavior and Parameter Sensitivity

**Goal:**
Analyze how different genetic algorithm configurations affect FATE’s ability to discover near-optimal fairness-aware data preparation pipelines.

**What we evaluate:**

- Impact of population size, generations, crossover rate, and mutation rate
- Comparison against two baselines:
  - *No practices*
  - *All practices*

**Key outputs:**

- Fitness trends across GA parameters
- Best pipelines per dataset, model, and protected attribute
- Comparative boxplots for fairness, performance, and fitness

All RQ1 results are located in `RQ1_data_analysis/`.

---

### RQ2 — Comparison Against State-of-the-Art Pre-processing Methods

**Goal:**
Statistically compare FATE-selected pipelines against standard bias mitigation techniques:

- FairSMOTE
- Reweighing
- Disparate Impact Remover (DIR)

**Metrics:**

- Fairness Score (FS) — lower is better
- Performance Score (PR-AUC) — higher is better
- Execution time — lower is better

**Statistical analysis:**

- Assumption checks (normality)
- Wilcoxon signed-rank tests
- Vargha–Delaney A₁₂ effect sizes

All RQ2 materials are located in `RQ2_data_analysis/`.

---

## Running the Experiments

### 1. Environment Setup

```bash
pip install -r requirements.txt
```

Python ≥ 3.9 is recommended.

---

### 2. Running FATE Experiments

To run the full experimental pipeline:

```bash
python main.py
```

This will:

- Apply FATE to all datasets, models, and protected attributes
- Store results incrementally in `output/experiments_results.csv`
- Use caching to avoid recomputation

---

### 3. RQ1 Analysis and Visualizations

```bash
python RQ1_data_analysis/rq1_results.py
python RQ1_data_analysis/rq1_visualizations.py
```

Generated figures are saved in:

```
RQ1_data_analysis/visualizations/
```

---

### 4. RQ2 Analysis and Hypothesis Testing

```bash
python RQ2_data_analysis/assumptions.py
python RQ2_data_analysis/rq2_results.py
```

This produces:

- `rq2_hypothesis_tests.csv`
- Statistical summaries used in the paper

---

## Caching and Reproducibility

To reduce runtime, the framework uses CSV-based caching:

- `experiments_cache.csv` (RQ1)
- `rq2_experiments_cache.csv` (RQ2)

Caches can be safely deleted to force full recomputation.
