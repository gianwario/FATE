"""
RQ2 experiment runner: FATE vs state-of-the-art fairness pre-processing baselines.

This module re-evaluates the best FATE configurations identified in RQ1
(``RQ1_data_analysis/rq1_fate_results_best_per_group.csv``) against three
state-of-the-art pre-processing bias mitigation methods under identical 5-fold
stratified cross-validation:

Baselines:
    - **FairSMOTE** (in-house implementation): oversamples each of the four
      (y ∈ {0,1}) × (s ∈ {0,1}) quadrants to equal size before training.
    - **Reweighing** (AIF360): adjusts instance weights to equalise positive
      prediction rates between privileged and unprivileged groups.
    - **DIR** – Disparate Impact Remover (AIF360): transforms feature values
      towards the marginal distribution to repair disparate impact.

Evaluation metrics (identical to FATE's fitness function for comparability):
    - **Performance**: PR-AUC (``average_precision_score``).
    - **Fairness**: FS = (|SPD| + |EOD| + |DI|) / 3.
    - **Execution time**: total wall-clock seconds summed across 5 folds.

Output: ``RQ2_data_analysis/rq2_all_experiments_results.csv`` — long-format
table with one row per (dataset, protected_attribute, model, method) group.

Note: The FATE re-evaluation block inside ``run_rq2`` is currently commented
out; the FATE rows are expected to be pre-populated from RQ1 results.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  # noqa: E402

import time  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from sklearn.model_selection import StratifiedKFold  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402

# === Your existing imports ===
import preprocessing as preproc  # noqa: E402
from preprocessing import prepare_data_model  # noqa: E402
from fitness import fairness_metrics  # noqa: E402

# AIF360 for Reweighing / DIR / FairSMOTE
from aif360.datasets import BinaryLabelDataset  # noqa: E402
from aif360.algorithms.preprocessing import Reweighing, DisparateImpactRemover  # noqa: E402
# from aif360.algorithms.preprocessing import FairSMOTE  # if you use their implementation

# === Model factory (reuse or adapt) ===
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.svm import LinearSVC  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402


BEST_CFG_CSV = "RQ1_data_analysis/rq1_fate_results_best_per_group.csv"
OUTPUT_CSV = "RQ2_data_analysis/rq2_all_experiments_results.csv"


DATASETS = [
    {
        "name": "adult",
        "path": "datasets/adult.csv",
        "preparer_name": "prepare_adult",
        "protected_attributes": ["race", "sex"],
        "target": "salary",
    },
    {
        "name": "german",
        "path": "datasets/german.csv",
        "preparer_name": "prepare_german",
        "protected_attributes": ["sex", "age"],
        "target": "Target",
    },
    {
        "name": "heart",
        "path": "datasets/heart.csv",
        "preparer_name": "prepare_heart",
        "protected_attributes": ["sex", "age"],
        "target": "num",
    },
]

DATASETS_BY_PATH = {ds["path"]: ds for ds in DATASETS}


def build_model(model_id: str):
    """
    Instantiate a classifier by string identifier with the same hyperparameters as ``fitness.py``.

    Using identical hyperparameters ensures that any performance difference
    between FATE and a baseline is attributable to the pre-processing method,
    not to the classifier configuration.

    Parameters
    ----------
    model_id : str
        One of: ``'lr'``, ``'rf'``, ``'svc'``, ``'xgb'``.

    Returns
    -------
    sklearn estimator
        Unfitted classifier instance.
    """
    if model_id == 'lr':
        classifier = LogisticRegression(
            max_iter=1000, solver='saga', penalty='l2', random_state=42, n_jobs=1)
    elif model_id == 'rf':
        classifier = RandomForestClassifier(
            n_estimators=100, max_depth=12, n_jobs=1, random_state=42)
    elif model_id == 'svc':
        classifier = LinearSVC(dual=False, max_iter=10000, tol=1e-4, random_state=42)
    elif model_id == 'xgb':
        classifier = XGBClassifier(use_label_encoder=False, eval_metric='logloss',
                                   n_estimators=100, tree_method='hist',
                                   verbosity=0, random_state=42, n_jobs=1)

    return classifier


def prepare_sample_ready(ds_cfg, protected_attr: str):
    """
    Load and prepare a dataset using the same pipeline as the GA runner.

    Ensures the data fed to baselines is identical to the data the GA was
    evaluated on, preserving comparability between FATE and the baselines.

    Parameters
    ----------
    ds_cfg : dict
        Dataset configuration dict with keys: ``path``, ``preparer_name``,
        ``target``, ``name``.
    protected_attr : str
        Protected attribute column to preserve through ``prepare_data_model``.

    Returns
    -------
    pd.DataFrame
        Model-ready dataset (features + target + protected attribute column).

    Raises
    ------
    RuntimeError
        If *protected_attr* is not present in the dataset after the
        dataset-specific preparer is applied.
    """
    raw = pd.read_csv(ds_cfg["path"])

    preparer = getattr(preproc, ds_cfg["preparer_name"])
    processed = preparer(raw)

    if protected_attr not in processed.columns:
        raise RuntimeError(
            f"Protected attribute '{protected_attr}' not found after preparation "
            f"for dataset {ds_cfg['name']}"
        )

    sample_ready = prepare_data_model(
        processed,
        ds_cfg["target"],
        protected_attribute=protected_attr,
        binarize=False,
    )
    return sample_ready


def compute_fairness_score_from_metrics(sp, eo, di):
    """
    Aggregate three fairness metric values into a single Fairness Score (FS).

    FS = (|SPD| + |EOD| + |DI|) / 3.

    This mirrors the normalised fairness term used in ``fitness.fitness`` so
    that baseline comparisons in RQ2 use an identical FS definition.

    Parameters
    ----------
    sp : float
        Absolute statistical parity difference.
    eo : float
        Absolute equal opportunity difference.
    di : float
        Disparate impact deviation (``1 − DI_ratio``).

    Returns
    -------
    float
        Fairness score in [0, 1] (assuming each component is in [0, 1]).
        Lower is fairer.
    """
    fairness_score = (sp + eo + di) / 3.0
    return fairness_score


def binarize_protected_for_fairsmote(s, protected_attribute: str):
    """
    Convert the protected attribute to a binary 0/1 indicator for FairSMOTE.

    FairSMOTE requires binary protected group labels to partition training
    data into the four (y ∈ {0,1}) × (s ∈ {0,1}) quadrants.

    Parameters
    ----------
    s : array-like
        Protected attribute values for the training fold.
    protected_attribute : str
        Column name used to select the binarisation rule:

        - ``'age'`` (numeric): threshold at the fold mean; above → 1.
        - Other: majority (mode) value → 1, all others → 0.

    Returns
    -------
    pd.Series
        Binary integer series (0 or 1) aligned with *s*.
    """
    s_series = pd.Series(s)
    name_lower = protected_attribute.lower()

    # Numeric age-like attribute
    if 'age' in name_lower and pd.api.types.is_numeric_dtype(s_series):
        thr = s_series.mean()
        return (s_series > thr).astype(int)

    # Generic fallback: majority value is privileged
    mode_val = s_series.mode().iloc[0]
    return (s_series == mode_val).astype(int)


def run_baseline_method(sample_ready, ds_cfg, protected_attr, model_id, method_name):
    """
    Evaluate one bias mitigation baseline under 5-fold stratified cross-validation.

    Applies the specified fairness-aware pre-processing method to the training
    fold at each CV iteration, trains a fresh classifier, evaluates on the test
    fold, and aggregates metrics across folds.

    Parameters
    ----------
    sample_ready : pd.DataFrame
        Pre-processed dataset (identical to the data fed to FATE's fitness
        function).
    ds_cfg : dict
        Dataset configuration dict (used to retrieve the target column name).
    protected_attr : str
        Protected attribute column name.
    model_id : str
        Classifier identifier (``'lr'``, ``'rf'``, ``'svc'``, ``'xgb'``).
    method_name : str
        Baseline method, case-insensitive: ``'fairsmote'``, ``'reweighing'``,
        or ``'dir'``.

    Returns
    -------
    dict
        ``{'performance_score': float, 'fairness_score': float,
        'elapsed_seconds': float}``

        - ``performance_score``: mean PR-AUC across 5 folds.
        - ``fairness_score``: FS = ``(|SPD| + |EOD| + |DI|) / 3``
          averaged across folds.
        - ``elapsed_seconds``: total wall-clock time summed over folds.

    Notes
    -----
    **Reweighing**: uses AIF360 ``Reweighing``; instance weights are passed
    as ``sample_weight`` to the classifier fit call.

    **DIR**: uses AIF360 ``DisparateImpactRemover`` with ``repair_level=1.0``;
    only features are modified — labels and group membership are unchanged.

    **FairSMOTE**: in-house implementation that oversamples each
    (y, s) quadrant to the size of the largest quadrant using sampling with
    replacement.
    """
    target = ds_cfg["target"]
    df = sample_ready.copy()

    X = df.drop(columns=[target, protected_attr])
    y = df[target]
    s = df[protected_attr]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    pr_aucs = []
    spds = []
    eods = []
    dis = []
    times = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        s_train = s.iloc[train_idx]

        start = time.time()

        # Build AIF360 dataset if needed
        if method_name.lower() in ["reweighing", "dir", "fairsMote", "fairsmote"]:
            bld_train = BinaryLabelDataset(
                df=pd.concat([X_train, y_train, s_train], axis=1),
                label_names=[target],
                protected_attribute_names=[protected_attr],
                favorable_label=1,
                unfavorable_label=0,
            )

        sample_weight = None

        if method_name.lower() == "reweighing":
            rw = Reweighing(
                unprivileged_groups=[{protected_attr: 0}],
                privileged_groups=[{protected_attr: 1}],
            )
            bld_rw = rw.fit_transform(bld_train)
            df_rw = bld_rw.convert_to_dataframe()[0]
            X_tr = df_rw[X.columns]
            y_tr = df_rw[target]
            sample_weight = bld_rw.instance_weights

        elif method_name.lower() in ["dir", "disparate_impact_remover"]:
            dir_ = DisparateImpactRemover(
                sensitive_attribute=protected_attr,
                repair_level=1.0,
            )
            bld_dir = dir_.fit_transform(bld_train)
            df_dir = bld_dir.convert_to_dataframe()[0]
            X_tr = df_dir[X.columns]
            y_tr = y_train

        elif method_name.lower() in ["fairsmote", "fairsMote", "fairsmoTe"]:
            """
            Generic FairSMOTE-style oversampling implementation.
            """

            # 1) Binarize protected attribute for FairSMOTE
            s_bin = binarize_protected_for_fairsmote(s_train, protected_attribute=protected_attr)

            # 2) Build a working DataFrame with features + y + s_bin
            df_train = pd.DataFrame(X_train.copy())
            df_train['_y'] = pd.Series(y_train).values
            df_train['_s'] = pd.Series(s_bin).values

            # Keep original feature columns so we can restore them later
            orig_feature_cols = list(X_train.columns)

            # 3) Check binary assumption for safety
            y_vals = set(df_train['_y'].unique())
            s_vals = set(df_train['_s'].unique())
            if not y_vals.issubset({0, 1}) or not s_vals.issubset({0, 1}):
                raise ValueError(f"FairSMOTE expects binary y and s. Got y={y_vals}, s={s_vals}")

            # 4) Split into four (y,s) groups
            g00 = df_train[(df_train['_y'] == 0) & (df_train['_s'] == 0)]
            g01 = df_train[(df_train['_y'] == 0) & (df_train['_s'] == 1)]
            g10 = df_train[(df_train['_y'] == 1) & (df_train['_s'] == 0)]
            g11 = df_train[(df_train['_y'] == 1) & (df_train['_s'] == 1)]

            n00, n01, n10, n11 = len(g00), len(g01), len(g10), len(g11)
            max_n = max(n00, n01, n10, n11)

            def oversample_group(df_group, target_size):
                if len(df_group) == 0 or len(df_group) >= target_size:
                    return df_group
                extra = target_size - len(df_group)
                return pd.concat(
                    [df_group, df_group.sample(n=extra, replace=True, random_state=42)],
                    ignore_index=True
                )

            # 5) Oversample each group up to max_n
            g00_bal = oversample_group(g00, max_n)
            g01_bal = oversample_group(g01, max_n)
            g10_bal = oversample_group(g10, max_n)
            g11_bal = oversample_group(g11, max_n)

            # 6) Reassemble balanced training data
            df_balanced = pd.concat([g00_bal, g01_bal, g10_bal, g11_bal], ignore_index=True)

            # 7) Split back into X / y / s
            y_tr = df_balanced['_y'].reset_index(drop=True)
            X_tr = df_balanced[orig_feature_cols].reset_index(drop=True)

        else:
            # Should not happen; you can also add a "no mitigation" method if you want
            X_tr = X_train.copy()
            y_tr = y_train.copy()

        model = build_model(model_id)
        if sample_weight is not None:
            model.fit(X_tr, y_tr, sample_weight=sample_weight)
        else:
            model.fit(X_tr, y_tr)

        # Predict on test
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)[:, 1]
        else:
            df_s = model.decision_function(X_test)
            y_score = (df_s - df_s.min()) / (df_s.max() - df_s.min() + 1e-12)

        y_pred = (y_score >= 0.5).astype(int)

        elapsed = time.time() - start
        times.append(elapsed)

        # Performance: PR-AUC
        try:
            pr_auc = average_precision_score(y_test, y_score)
        except Exception:
            pr_auc = np.nan
        pr_aucs.append(pr_auc)

        # Fairness: your fairness_metrics
        fairness = fairness_metrics(
            test_data=df,                # full prepared dataset
            test_indices=test_idx,       # indices of this test fold
            protected_attribute=protected_attr,
            predictions=y_pred,
            target_column=target,
        )
        spds.append(fairness["statistical_parity"])
        eods.append(fairness["equal_opportunity"])
        dis.append(fairness["disparate_impact"])

    # Aggregate
    mean_spd = np.nanmean(spds)
    mean_eod = np.nanmean(eods)
    mean_di = np.nanmean(dis)
    fs = compute_fairness_score_from_metrics(mean_spd, mean_eod, mean_di)

    return {
        "performance_score": float(np.nanmean(pr_aucs)),
        "fairness_score": fs,
        "elapsed_seconds": float(np.sum(times)),  # total time over folds
    }


def run_rq2():
    """
    Execute the full RQ2 experiment: evaluate baselines for the best FATE configurations.

    For each row in the best-per-group FATE results from RQ1, loads the
    corresponding dataset, prepares it identically to the FATE run, then
    evaluates the configured baselines (FairSMOTE by default; Reweighing and
    DIR are commented out for deferred evaluation).

    Results are collected into a long-format DataFrame and written to
    ``RQ2_data_analysis/rq2_all_experiments_results.csv``.

    Notes
    -----
    The FATE re-evaluation block (``execute_fate`` call) is intentionally
    commented out.  The output CSV is expected to already contain FATE rows
    from the RQ1 runs (via ``main.py``), so this function only appends the
    three baseline method rows.  Run ``main.py`` first to populate the FATE
    rows before calling this function.
    """
    best_cfg = pd.read_csv(BEST_CFG_CSV)

    results = []

    for _, row in best_cfg.iterrows():
        ds_path = row["dataset"]
        model_id = row["model_identifier"]
        prot_attr = row["protected_attribute"]
        pop = int(row["population_size"])  # noqa: F841
        gens = int(row["generations"])  # noqa: F841
        alpha = float(row["alpha"])  # noqa: F841
        beta = float(row["beta"])  # noqa: F841

        if ds_path not in DATASETS_BY_PATH:
            print(f"[WARN] Dataset path {ds_path} not in DATASETS config, skipping.")
            continue
        ds_cfg = DATASETS_BY_PATH[ds_path]

        print(f"\n=== RQ2 block: ds={ds_cfg['name']} prot={prot_attr} model={model_id} ===")

        # --- common prepared dataset for this combo ---
        try:
            sample_ready = prepare_sample_ready(ds_cfg, prot_attr)
        except Exception as e:
            print(f"[ERROR] prepare_sample_ready failed: {e}")
            # record failure row for FATE + baselines if you want
            continue

        # --- 1) FATE: run GA via execute_fate with selected hyperparams ---
        '''
        print("  -> Running FATE (GA)...")
        fate_rows = execute_fate(
            sample_ready=sample_ready,
            ds_name=ds_cfg["name"],
            ds_path=ds_cfg["path"],
            protected_attribute=prot_attr,
            target=ds_cfg["target"],
            models=[model_id],        # only this model
            population_size=pop,
            generations=gens,
            alpha=alpha,
            beta=beta,
            summary_path=None
        )
        # execute_fate returns a list of rows (one per model)
        for fr in fate_rows:
            results.append({
                "dataset_name": ds_cfg["name"],
                "protected_attribute": fr["protected_attribute"],
                "model_identifier": fr["model_identifier"],
                "method": "FATE",
                "performance_score": fr["performance_score"],
                "fairness_score": fr["fairness_score"],
                "elapsed_seconds": fr["elapsed_seconds"],
                "error": fr["error"],
            })
        '''
        # --- 2) Baselines: FairSMOTE, Reweighing, DIR ---
        baselines = [
            ("FairSMOTE", "fairsmote"),
            # ("Reweighing", "reweighing"),
            # ("DIR", "dir"),
        ]
        for label, method_name in baselines:
            print(f"  -> Running baseline: {label}...")
            try:
                metrics = run_baseline_method(
                    sample_ready, ds_cfg, prot_attr, model_id, method_name)
                results.append({
                    "dataset_name": ds_cfg["name"],
                    "protected_attribute": prot_attr,
                    "model_identifier": model_id,
                    "method": label,
                    "performance_score": metrics["performance_score"],
                    "fairness_score": metrics["fairness_score"],
                    "elapsed_seconds": metrics["elapsed_seconds"],
                    "error": None,
                })
            except Exception as e:
                print(f"[ERROR] Baseline {label} failed: {e}")
                results.append({
                    "dataset_name": ds_cfg["name"],
                    "protected_attribute": prot_attr,
                    "model_identifier": model_id,
                    "method": label,
                    "performance_score": np.nan,
                    "fairness_score": np.nan,
                    "elapsed_seconds": 0.0,
                    "error": str(e),
                })

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nRQ2 experiments saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    run_rq2()
