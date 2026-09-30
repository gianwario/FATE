"""
RQ2 experiment runner: FATE vs state-of-the-art fairness pre-processing baselines.

This module re-runs the best FATE configurations identified in RQ1
(``results/rq1/rq1_fate_results_best_per_group.csv``) against three
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

Output: ``results/rq2/rq2_all_experiments_results.csv`` — long-format
table with one row per (dataset, protected_attribute, model, method) group.

FATE rows: FATE is re-run with the GA parameters of each best RQ1
configuration, so the output contains FATE and the three baselines for every
group and is the direct input of ``rq2_results``.
"""
import argparse
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

import paths
import preprocessing as preproc
from experiment_config import DATASETS_BY_PATH, DatasetConfig
from fitness import Classifier, fairness_metrics
from main import execute_fate
from preprocessing import prepare_data_model

# AIF360 for Reweighing / DIR
from aif360_setup import silence_unused_backend_notices
silence_unused_backend_notices()
from aif360.datasets import BinaryLabelDataset  # noqa: E402
from aif360.algorithms.preprocessing import Reweighing, DisparateImpactRemover  # noqa: E402

BEST_CFG_NAME = "rq1_fate_results_best_per_group.csv"
OUTPUT_NAME = "rq2_all_experiments_results.csv"


def build_model(model_id: str) -> Classifier:
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


def prepare_sample_ready(ds_cfg: DatasetConfig, protected_attr: str) -> pd.DataFrame:
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
    raw = pd.read_csv(paths.REPO_ROOT / ds_cfg["path"])

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


def compute_fairness_score_from_metrics(sp: float, eo: float, di: float) -> float:
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


def binarize_protected_for_fairsmote(s: pd.Series, protected_attribute: str) -> pd.Series:
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


def _aif360_training_dataset(x_train: pd.DataFrame, y_train: pd.Series, s_train: pd.Series,
                             target: str, protected_attr: str) -> BinaryLabelDataset:
    """
    Wrap one training fold in an AIF360 ``BinaryLabelDataset``.

    Parameters
    ----------
    x_train : pd.DataFrame
        Training features (protected attribute excluded).
    y_train : pd.Series
        Training labels.
    s_train : pd.Series
        Protected attribute values of the training rows.
    target : str
        Label column name.
    protected_attr : str
        Protected attribute column name.

    Returns
    -------
    BinaryLabelDataset
    """
    return BinaryLabelDataset(
        df=pd.concat([x_train, y_train, s_train], axis=1),
        label_names=[target],
        protected_attribute_names=[protected_attr],
        favorable_label=1,
        unfavorable_label=0,
    )


def _reweighing_training_data(bld_train: BinaryLabelDataset, feature_columns: pd.Index, target: str,
                              protected_attr: str) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
    """
    Apply AIF360 ``Reweighing`` to a training fold.

    Parameters
    ----------
    bld_train : BinaryLabelDataset
        Output of ``_aif360_training_dataset``.
    feature_columns : pd.Index
        Feature columns to keep.
    target : str
    protected_attr : str

    Returns
    -------
    tuple
        ``(x_tr, y_tr, sample_weight)``.
    """
    rw = Reweighing(
        unprivileged_groups=[{protected_attr: 0}],
        privileged_groups=[{protected_attr: 1}],
    )
    bld_rw = rw.fit_transform(bld_train)
    df_rw = bld_rw.convert_to_dataframe()[0]
    return df_rw[feature_columns], df_rw[target], bld_rw.instance_weights


def _dir_training_data(bld_train: BinaryLabelDataset, feature_columns: pd.Index, y_train: pd.Series,
                       protected_attr: str) -> tuple[pd.DataFrame, pd.Series]:
    """
    Apply AIF360 ``DisparateImpactRemover`` (repair level 1.0) to a training fold.

    Only the features are repaired; labels are left unchanged.

    Parameters
    ----------
    bld_train : BinaryLabelDataset
    feature_columns : pd.Index
    y_train : pd.Series
    protected_attr : str

    Returns
    -------
    tuple
        ``(x_tr, y_tr)``.
    """
    dir_ = DisparateImpactRemover(sensitive_attribute=protected_attr, repair_level=1.0)
    bld_dir = dir_.fit_transform(bld_train)
    df_dir = bld_dir.convert_to_dataframe()[0]
    return df_dir[feature_columns], y_train


def _oversample_group(df_group: pd.DataFrame, target_size: int) -> pd.DataFrame:
    """
    Oversample *df_group* with replacement up to *target_size* rows.

    Groups that are empty or already large enough are returned unchanged.

    Parameters
    ----------
    df_group : pd.DataFrame
    target_size : int

    Returns
    -------
    pd.DataFrame
    """
    if len(df_group) == 0 or len(df_group) >= target_size:
        return df_group
    extra = target_size - len(df_group)
    return pd.concat(
        [df_group, df_group.sample(n=extra, replace=True, random_state=42)],
        ignore_index=True
    )


def _fairsmote_training_data(x_train: pd.DataFrame, y_train: pd.Series, s_train: pd.Series,
                             protected_attr: str) -> tuple[pd.DataFrame, pd.Series]:
    """
    FairSMOTE-style rebalancing of a training fold (in-house implementation).

    Every (y, s) quadrant is oversampled with replacement to the size of the
    largest quadrant.

    Parameters
    ----------
    x_train : pd.DataFrame
    y_train : pd.Series
    s_train : pd.Series
    protected_attr : str

    Returns
    -------
    tuple
        ``(x_tr, y_tr)``.

    Raises
    ------
    ValueError
        If the labels or the binarised protected attribute are not binary.
    """
    # 1) Binarize protected attribute for FairSMOTE
    s_bin = binarize_protected_for_fairsmote(s_train, protected_attribute=protected_attr)

    # 2) Build a working DataFrame with features + y + s_bin
    df_train = pd.DataFrame(x_train.copy())
    df_train['_y'] = pd.Series(y_train).values
    df_train['_s'] = pd.Series(s_bin).values
    orig_feature_cols = list(x_train.columns)

    # 3) Check binary assumption for safety
    y_vals = set(df_train['_y'].unique())
    s_vals = set(df_train['_s'].unique())
    if not y_vals.issubset({0, 1}) or not s_vals.issubset({0, 1}):
        raise ValueError(f"FairSMOTE expects binary y and s. Got y={y_vals}, s={s_vals}")

    # 4) Split into four (y,s) groups and 5) oversample each up to the largest
    groups = [df_train[(df_train['_y'] == yv) & (df_train['_s'] == sv)]
              for yv, sv in ((0, 0), (0, 1), (1, 0), (1, 1))]
    max_n = max(len(g) for g in groups)

    # 6) Reassemble balanced training data
    df_balanced = pd.concat([_oversample_group(g, max_n) for g in groups],
                            ignore_index=True)

    # 7) Split back into features / y
    y_tr = df_balanced['_y'].reset_index(drop=True)
    x_tr = df_balanced[orig_feature_cols].reset_index(drop=True)
    return x_tr, y_tr


def _mitigated_training_data(method_name: str, x_train: pd.DataFrame, y_train: pd.Series,
                             s_train: pd.Series, target: str, protected_attr: str
                             ) -> tuple[pd.DataFrame, pd.Series, Optional[np.ndarray]]:
    """
    Apply the requested bias mitigation method to one training fold.

    Parameters
    ----------
    method_name : str
        ``'fairsmote'``, ``'reweighing'`` or ``'dir'`` (case-insensitive).
    x_train : pd.DataFrame
    y_train : pd.Series
    s_train : pd.Series
    target : str
    protected_attr : str

    Returns
    -------
    tuple
        ``(x_tr, y_tr, sample_weight)``; *sample_weight* is ``None`` except for
        Reweighing.  Unknown methods return the fold unchanged.
    """
    method = method_name.lower()
    if method in ("reweighing", "dir", "fairsmote"):
        bld_train = _aif360_training_dataset(x_train, y_train, s_train, target, protected_attr)
    if method == "reweighing":
        return _reweighing_training_data(bld_train, x_train.columns, target, protected_attr)
    if method in ("dir", "disparate_impact_remover"):
        x_tr, y_tr = _dir_training_data(bld_train, x_train.columns, y_train, protected_attr)
        return x_tr, y_tr, None
    if method == "fairsmote":
        x_tr, y_tr = _fairsmote_training_data(x_train, y_train, s_train, protected_attr)
        return x_tr, y_tr, None
    return x_train.copy(), y_train.copy(), None


def _positive_class_scores(model: Classifier, x_test: pd.DataFrame) -> np.ndarray:
    """
    Return positive-class scores in [0, 1] for *x_test*.

    Uses ``predict_proba`` when available; otherwise min-max normalises the
    ``decision_function`` output (LinearSVC).

    Parameters
    ----------
    model : Classifier
        Fitted classifier.
    x_test : pd.DataFrame

    Returns
    -------
    np.ndarray
    """
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x_test)[:, 1]
    df_s = model.decision_function(x_test)
    return (df_s - df_s.min()) / (df_s.max() - df_s.min() + 1e-12)


def run_baseline_method(sample_ready: pd.DataFrame, ds_cfg: DatasetConfig, protected_attr: str,
                        model_id: str, method_name: str) -> dict[str, float]:
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

    x = df.drop(columns=[target, protected_attr])
    y = df[target]
    s = df[protected_attr]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    pr_aucs = []
    spds = []
    eods = []
    dis = []
    times = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(x, y), 1):
        x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        s_train = s.iloc[train_idx]

        start = time.time()
        x_tr, y_tr, sample_weight = _mitigated_training_data(
            method_name, x_train, y_train, s_train, target, protected_attr)
        model = build_model(model_id)
        if sample_weight is not None:
            model.fit(x_tr, y_tr, sample_weight=sample_weight)
        else:
            model.fit(x_tr, y_tr)
        y_score = _positive_class_scores(model, x_test)
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


#: Baselines compared with FATE in RQ2: (label used in the results, method name).
BASELINES: list[tuple[str, str]] = [
    ("FairSMOTE", "fairsmote"),
    ("Reweighing", "reweighing"),
    ("DIR", "dir"),
]


def _fate_rows(sample_ready: pd.DataFrame, ds_cfg: DatasetConfig, cfg_row: pd.Series,
               reset_cache: bool) -> list[dict[str, object]]:
    """
    Re-run FATE with the GA parameters of one best RQ1 configuration.

    Parameters
    ----------
    sample_ready : pd.DataFrame
        Prepared dataset (same preparation as in the RQ1 grid).
    ds_cfg : DatasetConfig
    cfg_row : pd.Series
        Row of the best-per-group RQ1 file (model, protected attribute and
        GA parameters N, G, alpha, beta).
    reset_cache : bool
        Forwarded to ``execute_fate``.

    Returns
    -------
    list of dict
        One RQ2 result row (method ``FATE``).
    """
    fate_rows = execute_fate(
        sample_ready=sample_ready,
        ds_name=ds_cfg["name"],
        ds_path=ds_cfg["path"],
        protected_attribute=cfg_row["protected_attribute"],
        target=ds_cfg["target"],
        models=[cfg_row["model_identifier"]],
        population_size=int(cfg_row["population_size"]),
        generations=int(cfg_row["generations"]),
        alpha=float(cfg_row["alpha"]),
        beta=float(cfg_row["beta"]),
        summary_path=None,
        reset_cache=reset_cache,
    )
    return [{
        "dataset_name": ds_cfg["name"],
        "protected_attribute": fr["protected_attribute"],
        "model_identifier": fr["model_identifier"],
        "method": "FATE",
        "performance_score": fr["performance_score"],
        "fairness_score": fr["fairness_score"],
        "elapsed_seconds": fr["elapsed_seconds"],
        "error": fr["error"],
    } for fr in fate_rows]


def _baseline_row(sample_ready: pd.DataFrame, ds_cfg: DatasetConfig, prot_attr: str,
                  model_id: str, label: str, method_name: str) -> dict[str, object]:
    """
    Evaluate one baseline and return its RQ2 result row.

    Parameters
    ----------
    sample_ready : pd.DataFrame
    ds_cfg : DatasetConfig
    prot_attr : str
    model_id : str
    label : str
        Method label stored in the results (e.g. ``'DIR'``).
    method_name : str
        Method identifier passed to ``run_baseline_method``.

    Returns
    -------
    dict
        Result row; on failure the metrics are NaN and ``error`` holds the message.
    """
    row: dict[str, object] = {
        "dataset_name": ds_cfg["name"],
        "protected_attribute": prot_attr,
        "model_identifier": model_id,
        "method": label,
    }
    try:
        metrics = run_baseline_method(sample_ready, ds_cfg, prot_attr, model_id, method_name)
        row.update(performance_score=metrics["performance_score"],
                   fairness_score=metrics["fairness_score"],
                   elapsed_seconds=metrics["elapsed_seconds"], error=None)
    except Exception as e:
        print(f"[ERROR] Baseline {label} failed: {e}")
        row.update(performance_score=np.nan, fairness_score=np.nan,
                   elapsed_seconds=0.0, error=str(e))
    return row


def run_rq2(best_cfg_csv: Path, output_csv: Path, reset_cache: bool = False) -> None:
    """
    Execute the RQ2 experiment: FATE and the three baselines on the best RQ1 configurations.

    For each row of the best-per-group RQ1 file, the dataset is prepared as in
    the RQ1 grid, FATE is re-run with that row's GA parameters (N, G, alpha,
    beta), and FairSMOTE, Reweighing and DIR are evaluated.  Results are
    written in long format (one row per group and method) to *output_csv*.

    Because the genetic algorithm is stochastic, the FATE rows of a new run
    are not expected to coincide exactly with the archived ones in
    ``reference/rq2/rq2_all_experiments_results.csv`` (the run reported in
    the paper).

    Parameters
    ----------
    best_cfg_csv : Path
        ``rq1_fate_results_best_per_group.csv`` (output of RQ1).
    output_csv : Path
        Destination CSV (default ``results/rq2/rq2_all_experiments_results.csv``).
    reset_cache : bool, optional
        Forwarded to the FATE re-runs (see ``fitness.fitness``).
    """
    best_cfg = pd.read_csv(best_cfg_csv)
    results: list[dict[str, object]] = []
    for _, cfg_row in best_cfg.iterrows():
        ds_path = cfg_row["dataset"]
        if ds_path not in DATASETS_BY_PATH:
            print(f"[WARN] Dataset path {ds_path} not in DATASETS config, skipping.")
            continue
        ds_cfg = DATASETS_BY_PATH[ds_path]
        model_id = cfg_row["model_identifier"]
        prot_attr = cfg_row["protected_attribute"]
        print(f"\n=== RQ2 block: ds={ds_cfg['name']} prot={prot_attr} model={model_id} ===")
        sample_ready = prepare_sample_ready(ds_cfg, prot_attr)

        print("  -> Running FATE (GA)...")
        results.extend(_fate_rows(sample_ready, ds_cfg, cfg_row, reset_cache))
        for label, method_name in BASELINES:
            print(f"  -> Running baseline: {label}...")
            results.append(_baseline_row(sample_ready, ds_cfg, prot_attr, model_id,
                                         label, method_name))

    df = pd.DataFrame(results)
    paths.ensure_dir(output_csv.parent)
    df.to_csv(output_csv, index=False)
    print(f"\nRQ2 experiments saved to {output_csv}")


def main(argv: Optional[list[str]] = None) -> None:
    """Run the RQ2 baseline experiments on the best FATE configurations of RQ1."""
    parser = argparse.ArgumentParser(description="RQ2: bias mitigation baselines.")
    parser.add_argument("--best-configs", type=Path,
                        default=paths.RQ1_RESULTS_DIR / BEST_CFG_NAME)
    parser.add_argument("--out", type=Path, default=paths.RQ2_RESULTS_DIR / OUTPUT_NAME)
    parser.add_argument("--reset-cache", action="store_true",
                        help="FATE re-runs ignore the archived fitness cache")
    args = parser.parse_args(argv)
    run_rq2(args.best_configs, args.out, args.reset_cache)


if __name__ == "__main__":
    main()
