import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import ast
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score

# === Your existing imports ===
import preprocessing as preproc
from preprocessing import prepare_data_model
from main import execute_fate  # file where execute_fate is defined
from fitness import fairness_metrics  # <-- point this to where you defined it

# AIF360 for Reweighing / DIR / FairSMOTE
from aif360.datasets import BinaryLabelDataset
from aif360.algorithms.preprocessing import Reweighing, DisparateImpactRemover
# from aif360.algorithms.preprocessing import FairSMOTE  # if you use their implementation

# === Model factory (reuse or adapt) ===
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


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
    if model_id == 'lr':
        classifier = LogisticRegression(max_iter=1000, solver='saga', penalty='l2', random_state=42, n_jobs=1)
    elif model_id == 'rf':
        classifier = RandomForestClassifier(n_estimators=100, max_depth=12, n_jobs=1, random_state=42)
    elif model_id == 'svc':
        classifier = LinearSVC(dual=False, max_iter=10000, tol=1e-4, random_state=42)
    elif model_id == 'xgb':
            classifier = XGBClassifier(use_label_encoder=False, eval_metric='logloss',
                                       n_estimators=100, tree_method='hist',
                                       verbosity=0, random_state=42, n_jobs=1)

    return classifier


def prepare_sample_ready(ds_cfg, protected_attr: str):
    """Reuse the same dataset preparation logic as in your GA runner."""
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
    Compute a single fairness score from the given metrics = (|SP|, |EO|, |DI|)/3
    The closer to 1.0, the fairer the model.
    """
    fairness_score = (sp + eo + di) / 3.0
    return fairness_score
    


def run_baseline_method(sample_ready, ds_cfg, protected_attr, model_id, method_name):
    """
    Run one baseline method (FairSMOTE/Reweighing/DIR) with 5-fold CV.

    Returns:
        dict with aggregated metrics:
            performance_score (mean PR-AUC),
            fairness_score (FS),
            mean_statistical_parity,
            mean_equal_opportunity,
            mean_disparate_impact,
            elapsed_seconds
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
        s_train, s_test = s.iloc[train_idx], s.iloc[test_idx]

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
            s_tr = df_rw[protected_attr]
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
            s_tr = s_train

        elif method_name.lower() in ["fairsmote", "fairsMote", "fairsmoTe"]:
            """
            Generic FairSMOTE-style oversampling implementation.
            """

            # Build a dataframe that contains features + label + protected attribute
            df_train = X_train.copy()
            df_train[target] = y_train.values

            # if the protected attribute is not already in X, attach it from s_train
            if protected_attr in df_train.columns:
                s_col = df_train[protected_attr]
            else:
                df_train[protected_attr] = s_train.values
                s_col = df_train[protected_attr]

            y_col = target
            s_col_name = protected_attr

            # Sanity: make sure y and s are binary
            y_vals = df_train[y_col].unique()
            s_vals = df_train[s_col_name].unique()
            if not set(y_vals).issubset({0, 1}) or not set(s_vals).issubset({0, 1}):
                raise ValueError(
                    f"FairSMOTE expects binary y and s. Got y={y_vals}, s={s_vals}"
                )

            # ---- 1. Count the four (class, protected) combinations ----
            g00 = df_train[(df_train[y_col] == 0) & (df_train[s_col_name] == 0)]
            g01 = df_train[(df_train[y_col] == 0) & (df_train[s_col_name] == 1)]
            g10 = df_train[(df_train[y_col] == 1) & (df_train[s_col_name] == 0)]
            g11 = df_train[(df_train[y_col] == 1) & (df_train[s_col_name] == 1)]

            n00, n01, n10, n11 = len(g00), len(g01), len(g10), len(g11)
            max_n = max(n00, n01, n10, n11)

            def oversample_group(df_group, target_size):
                """Random oversampling with replacement to reach target_size."""
                cur = len(df_group)
                if cur == 0 or cur >= target_size:
                    return df_group
                extra = df_group.sample(
                    n=target_size - cur,
                    replace=True,
                    random_state=42  # or pass a seed from outside if you want
                )
                return pd.concat([df_group, extra], ignore_index=True)

            # ---- 2. Oversample each group up to max_n ----
            g00_bal = oversample_group(g00, max_n)
            g01_bal = oversample_group(g01, max_n)
            g10_bal = oversample_group(g10, max_n)
            g11_bal = oversample_group(g11, max_n)

            # ---- 3. Reassemble balanced training data ----
            df_balanced = pd.concat([g00_bal, g01_bal, g10_bal, g11_bal], ignore_index=True)

            # ---- 4. Split back into X / y / s ----
            y_tr = df_balanced[y_col].copy()
            s_tr = df_balanced[s_col_name].copy()
            X_tr = df_balanced.drop(columns=[y_col])

        else:
            # Should not happen; you can also add a "no mitigation" method if you want
            X_tr = X_train.copy()
            y_tr = y_train.copy()
            s_tr = s_train.copy()

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
    best_cfg = pd.read_csv(BEST_CFG_CSV)

    results = []

    for _, row in best_cfg.iterrows():
        ds_path   = row["dataset"]
        model_id  = row["model_identifier"]
        prot_attr = row["protected_attribute"]
        pop       = int(row["population_size"])
        gens      = int(row["generations"])
        alpha     = float(row["alpha"])
        beta      = float(row["beta"])

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

        # --- 2) Baselines: FairSMOTE, Reweighing, DIR ---
        baselines = [
            ("FairSMOTE", "fairsmote"),
            ("Reweighing", "reweighing"),
            ("DIR", "dir"),
        ]
        for label, method_name in baselines:
            print(f"  -> Running baseline: {label}...")
            try:
                metrics = run_baseline_method(sample_ready, ds_cfg, prot_attr, model_id, method_name)
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
                    "mean_statistical_parity": np.nan,
                    "mean_equal_opportunity": np.nan,
                    "mean_disparate_impact": np.nan,
                    "elapsed_seconds": 0.0,
                    "error": str(e),
                })

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nRQ2 experiments saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    run_rq2()
