import time
import ast
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# AIF360 imports (adjust if your structure is different)
from aif360.datasets import BinaryLabelDataset
from aif360.algorithms.preprocessing import Reweighing, DisparateImpactRemover
# FairSMOTE: if you use AIF360's implementation, import it here
# from aif360.algorithms.preprocessing import FairSMOTE

from fitness import fairness_metrics 
from main import execute_fate
# ------------- CONFIG ------------- #

BEST_CONFIG_CSV = "../RQ1_data_analysis/rq1_results_best_per_group.csv"
OUTPUT_CSV = "rq2_results.csv"

# Map model identifiers from CSV to sklearn/xgb objects
def build_model(model_identifier: str):
    if model_identifier == 'lr':
        classifier = LogisticRegression(max_iter=1000, solver='saga', penalty='l2', random_state=42, n_jobs=1)
    elif model_identifier == 'rf':
        classifier = RandomForestClassifier(n_estimators=100, max_depth=12, n_jobs=1, random_state=42)
    elif model_identifier == 'svc':
        classifier = LinearSVC(dual=False, max_iter=10000, tol=1e-4, random_state=42)
    elif model_identifier == 'xgb':
        classifier = XGBClassifier(use_label_encoder=False, eval_metric='logloss',
                                       n_estimators=100, tree_method='hist',
                                       verbosity=0, random_state=42, n_jobs=1)
    return classifier

# You probably already have this somewhere in your codebase
def load_dataset(dataset_path: str, protected_attr: str):
    """
    Returns:
        X (DataFrame without target),
        y (Series),
        s (Series for protected attribute),
        meta (dict with names of target col, privileged/unprivileged groups, etc.)
    """
    df = pd.read_csv(dataset_path)

    if "adult" in dataset_path:
        target_col = "salary"   # adjust!
    elif "german" in dataset_path:
        target_col = "Target"   # adjust!
    elif "heart" in dataset_path:
        target_col = "num"   # adjust!
    else:
        raise ValueError(f"Unknown dataset path: {dataset_path}")

    y = df[target_col]
    s = df[protected_attr]
    X = df.drop(columns=[target_col])

    
    meta = {
        "target_col": target_col
    }
    return X, y, s, meta




# ------------------------------------------------------------
#  APPLY PRE-PROCESSING METHODS
# ------------------------------------------------------------

def apply_method(method, X, y, s, protected_attr, meta):
    """
    method ∈ {"fairsmote", "reweighing", "dir"}
    """
    # ---- AIF360 conversions ----
    bld = BinaryLabelDataset(
        df=pd.concat([X, y, s], axis=1),
        label_names=[meta["target"]],
        protected_attribute_names=[protected_attr],
        favorable_label=1,
        unfavorable_label=0,
    )

    # ---- Reweighing ----
    if method == "reweighing":
        rw = Reweighing(
            unprivileged_groups=[{protected_attr: 0}],
            privileged_groups=[{protected_attr: 1}],
        )
        bld_rw = rw.fit_transform(bld)
        df_rw = bld_rw.convert_to_dataframe()[0]
        X_t = df_rw[X.columns]
        y_t = df_rw[meta["target"]]
        s_t = df_rw[protected_attr]
        w = bld_rw.instance_weights
        return X_t, y_t, s_t, w

    # ---- Disparate Impact Remover ----
    if method == "dir":
        dir_ = DisparateImpactRemover(sensitive_attribute=protected_attr, repair_level=1.0)
        bld_dir = dir_.fit_transform(bld)
        df_dir = bld_dir.convert_to_dataframe()[0]
        X_t = df_dir[X.columns]
        return X_t, y, s, None

    # ---- FairSMOTE (placeholder) ----
    if method == "fairsmote":
        # TODO: plug your implementation
        return X.copy(), y.copy(), s.copy(), None

    raise ValueError(f"Unknown method: {method}")


# ------------------------------------------------------------
# 6. RQ2 PIPELINE
# ------------------------------------------------------------

def run_rq2(best_cfg_csv, output_csv="rq2_results.csv"):

    best_cfg = pd.read_csv(best_cfg_csv)
    best_cfg["techniques_list"] = best_cfg["techniques"].apply(
        lambda t: ast.literal_eval(t) if isinstance(t, str) else []
    )

    records = []
    methods = ["fate", "fairsmote", "reweighing", "dir"]

    for _, row in best_cfg.iterrows():

        dataset_path = row["dataset"]
        model_id = row["model_identifier"]
        protected_attr = row["protected_attribute"]
        techniques = row["techniques_list"]

        print(f"\n=== RQ2: {dataset_path} | {protected_attr} | {model_id} ===")

        
        X, y, s, meta = load_dataset(dataset_path, protected_attr)
        target_col = meta["target"]
        
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        for method in methods:
            if method == "fate":
                x = execute_fate(sample_ready, ds_name, ds_path, protected_attribute, target, models,
                 population_size, generations, alpha=0.5, beta=0.5, summary_path=None)
            for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):

                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                s_train, s_test = s.iloc[train_idx], s.iloc[test_idx]

                start_time = time.time()

                # ---- Apply pipeline ----
                if method == "fate":
                    X_t, y_t, s_t = apply_fate_pipeline(X_train, y_train, s_train, techniques)
                    sample_weight = None
                else:
                    X_t, y_t, s_t, sample_weight = apply_method(method, X_train, y_train, s_train, protected_attr, meta)

                # ---- Train model ----
                model = build_model(model_id)
                if sample_weight is not None:
                    model.fit(X_t, y_t, sample_weight=sample_weight)
                else:
                    model.fit(X_t, y_t)

                # ---- Predict ----
                if hasattr(model, "predict_proba"):
                    y_score = model.predict_proba(X_test)[:, 1]
                else:
                    df_s = model.decision_function(X_test)
                    y_score = (df_s - df_s.min()) / (df_s.max() - df_s.min() + 1e-12)

                y_pred = (y_score >= 0.5).astype(int)

                elapsed = time.time() - start_time

                # ---- Performance ----
                perf = average_precision_score(y_test, y_score)

                # ---- Fairness (your function!) ----
                fairness = fairness_metrics(
                    test_data=pd.concat([X, y, s], axis=1),
                    test_indices=test_idx,
                    protected_attribute=protected_attr,
                    predictions=y_pred,
                    target_column=target_col
                )

                records.append({
                    "dataset": dataset_path,
                    "protected_attribute": protected_attr,
                    "model": model_id,
                    "method": method,
                    "fold": fold,
                    "performance_score": perf,
                    "statistical_parity": fairness["statistical_parity"],
                    "equal_opportunity": fairness["equal_opportunity"],
                    "disparate_impact": fairness["disparate_impact"],
                    "elapsed_seconds": elapsed,
                })

    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)
    print(f"\nSaved RQ2 results to {output_csv}!")


if __name__ == "__main__":
    run_rq2("../RQ1_data_analysis/rq1_results_best_per_group.csv")