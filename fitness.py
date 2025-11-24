# fitness.py
import pandas as pd
import numpy as np
from fairlearn.postprocessing import ThresholdOptimizer
from sklearn.metrics import accuracy_score, average_precision_score
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier
from preprocessing import prepare_data_model

from practices import apply_techniques
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric


def fairness_metrics(test_data, test_indices, protected_attribute, predictions, target_column):
    """
    Compute fairness metrics (statistical parity difference, equal opportunity difference,
    and disparate impact) for binary classification.

    This function prefers AIF360 when available; otherwise it falls back to a correct
    manual computation.

    Args:
        test_data (pd.DataFrame): The full dataset (will be indexed by test_indices).
        test_indices (pd.Index or array-like): Indices corresponding to the test split.
        protected_attribute (str): Column name of the protected attribute.
        predictions (array-like): Predicted labels (0/1) or probabilities (will be thresholded at 0.5).
        target_column (str): Column name of the true label (expects 0/1 binary).

    Returns:
        dict: { 'statistical_parity': float, 'equal_opportunity': float, 'disparate_impact': float, ... }
    """
    # make predictions a binary series aligned to test indices
    preds = pd.Series(predictions, index=test_indices)

    # Extract test subset
    test_df = test_data.loc[test_indices].copy()
    test_df = test_df.reset_index(drop=True)
    preds = preds.reset_index(drop=True)

    # Determine privileged/unprivileged groups with hard-coded rules
    if protected_attribute not in test_df.columns:
        print(f"[fairness_metrics] protected attribute '{protected_attribute}' not found in test data")
        return {'statistical_parity': np.nan, 'equal_opportunity': np.nan, 'disparate_impact': np.nan}

    prot_series = test_df[protected_attribute]

    # Build a binary protected indicator column '__prot_bin__' where 1==privileged, 0==unprivileged
    prot_bin = pd.Series(index=prot_series.index, dtype=int)

    name_lower = protected_attribute.lower()
    # sex -> privileged = male
    if 'sex' in name_lower:
        # numeric 1/0 or strings like 'Male'/'Female'
        if pd.api.types.is_numeric_dtype(prot_series):
            prot_bin = prot_series.fillna(0).astype(int).apply(lambda v: 1 if v == 1 else 0)
        else:
            prot_bin = prot_series.fillna('').astype(str).str.lower().apply(lambda s: 1 if 'male' in s or s == 'm' else 0)

    # race -> privileged = 1
    elif 'race' in name_lower:
        prot_bin = pd.to_numeric(prot_series, errors='coerce').fillna(0).astype(int).apply(lambda v: 1 if v == 1 else 0)

    # age -> privileged = age > mean
    elif 'age' in name_lower:
        age_vals = pd.to_numeric(prot_series, errors='coerce')
        mean_age = age_vals.mean()
        prot_bin = age_vals.apply(lambda v: 1 if pd.notna(v) and v > mean_age else 0).astype(int)

    else:
        # fallback: most frequent value privileged
        vc = prot_series.value_counts(dropna=False)
        if len(vc) == 0:
            print(f"[fairness_metrics] protected attribute '{protected_attribute}' has no values")
            return {'statistical_parity': np.nan, 'equal_opportunity': np.nan, 'disparate_impact': np.nan}
        priv = vc.idxmax()
        prot_bin = prot_series.fillna('').astype(str).apply(lambda s: 1 if s == str(priv) else 0)

    # Prepare true labels (coerce common formats to binary 0/1)
    y_series = pd.Series(test_df[target_column]).copy()
    uniq = set(pd.unique(y_series.dropna()))
    if uniq <= {0, 1}:
        y_bin = y_series.astype(int)
    elif uniq <= {1, 2}:
        y_bin = y_series.map(lambda v: 1 if v == 2 else 0 if v == 1 else np.nan).astype(float)
    else:
        print(f"[fairness_metrics] non-binary target values in test set: {pd.unique(y_series)}")
        return {'statistical_parity': np.nan, 'equal_opportunity': np.nan, 'disparate_impact': np.nan}

    # Build df_true/df_pred using the binary protected indicator
    df_true = pd.DataFrame({'__prot_bin__': prot_bin.reset_index(drop=True), target_column: y_bin.reset_index(drop=True)})
    df_pred = pd.DataFrame({'__prot_bin__': prot_bin.reset_index(drop=True), target_column: pd.Series(preds).reset_index(drop=True)})

    # Ensure numeric labels
    df_true[target_column] = pd.to_numeric(df_true[target_column], errors='coerce')
    df_pred[target_column] = pd.to_numeric(df_pred[target_column], errors='coerce')

    # Build BinaryLabelDataset and compute metrics using __prot_bin__ as protected attribute
    try:
        dataset_true = BinaryLabelDataset(df=df_true, label_names=[target_column], protected_attribute_names=['__prot_bin__'], favorable_label=1, unfavorable_label=0)
        dataset_pred = BinaryLabelDataset(df=df_pred, label_names=[target_column], protected_attribute_names=['__prot_bin__'], favorable_label=1, unfavorable_label=0)
    except Exception as e:
        print(f"[fairness_metrics] BinaryLabelDataset construction failed: {e}")
        print("df_true sample:\n", df_true.head())
        print("df_pred sample:\n", df_pred.head())
        return {'statistical_parity': np.nan, 'equal_opportunity': np.nan, 'disparate_impact': np.nan}

    privileged_groups = [{'__prot_bin__': 1}]
    unprivileged_groups = [{'__prot_bin__': 0}]

    try:
        metric = ClassificationMetric(dataset_true, dataset_pred, unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
        spd = metric.statistical_parity_difference()
        eod = metric.equal_opportunity_difference()
        di = metric.disparate_impact()
    except Exception as e:
        print(f"[fairness_metrics] ClassificationMetric computation failed: {e}")
        print("protected (bin) counts:", df_true['__prot_bin__'].value_counts(dropna=False).to_dict())
        print("y_true distribution:", df_true[target_column].value_counts(dropna=False).to_dict())
        print("preds distribution:", df_pred[target_column].value_counts(dropna=False).to_dict())
        return {'statistical_parity': np.nan, 'equal_opportunity': np.nan, 'disparate_impact': np.nan}

    return {
        'statistical_parity': float(spd) if spd is not None else np.nan,
        'equal_opportunity': float(eod) if eod is not None else np.nan,
        'disparate_impact': float(di) if di is not None else np.nan
    }

    


def fitness(data, technique, model, protected_attribute, target_column, perf_weight=0.5, fair_weight=0.5):
    """
    Evaluate the fitness of a model based on performance and fairness.

    Args:
        data (pd.DataFrame): The dataset.
        technique (str or list): The technique(s) used (single technique or list of techniques).
        model (str): The model to be evaluated.
        protected_attribute (str): Name of the protected attribute column.
        target_column (str): Name of the target column.
        perf_weight (float): Weight (alpha) applied to the performance score (PS).
        fair_weight (float): Weight (beta) applied to the fairness score (FS).

    Returns:
        float: The fitness value computed as (alpha * PS - beta * FS).
    """
    # If technique is a list/tuple, apply each technique in sequence; if a single technique, apply it
    if isinstance(technique, (list, tuple)):
        for t in technique:
            try:
                data = apply_techniques(data, t, protected_attribute)
            except Exception:
                # if a technique fails, skip it to allow GA to continue
                continue
    elif technique:
        try:
            data = apply_techniques(data, technique, protected_attribute)
        except Exception:
            pass

    # Prepare data for modeling (centralized preprocessing)
    data = prepare_data_model(data, target_column, protected_attribute)

    # Consolidate in case preprocessing built a fragmented DataFrame (speeds up training ops)
    data = data.copy()

    # Build X and y from processed dataframe
    y = data[target_column]
    X = data.drop(columns=[target_column])

    # Ensure target is a pandas Series for checks and potential mapping
    y_series = pd.Series(y)
    unique_vals = pd.unique(y_series.dropna())
    # If no labels or only one label exist, training will fail
    if len(unique_vals) == 0:
        print(f"[fitness] empty or missing target values in column '{target_column}'")
        return float('inf'), None, None
    if len(unique_vals) == 1:
        print(f"[fitness] single-class target detected. Unique values: {unique_vals}. Cannot train classifier.")
        return float('inf'), None, None

    # If binary but not in {0,1}, map the two values to 0/1 (enables classifiers expecting numeric labels)
    if len(unique_vals) == 2 and set(unique_vals) != {0, 1}:
        mapping = {unique_vals[0]: 0, unique_vals[1]: 1}
        try:
            y = y_series.map(mapping).astype(int)
        except Exception:
            print(f"[fitness] failed to map binary target values {unique_vals} to 0/1")
            return float('inf'), None, None
    else:
        y = y_series

    # Always use K-fold cross-validation so the whole dataset is used for evaluation.
    n_splits = 5

    # Determine per-class counts to choose a safe number of splits for stratification
    y_nonnull = y.dropna()
    class_counts = y_nonnull.value_counts()
    if len(class_counts) < 2:
        print(f"[fitness] only one class present after preprocessing; cannot perform K-fold CV. unique={pd.unique(y)}")
        return float('inf'), None, None


    # prefer stratified folds for classification
    try:
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    except Exception:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    perf_scores = []
    fair_scores = []
    successful_folds = 0

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X, y), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Confirm training set has at least two classes
        if len(pd.unique(y_train.dropna())) < 2:
            print(f"[fitness] fold {fold_idx} skipped: only one class in y_train")
            continue

        # instantiate classifier per-fold to avoid carrying state
        if model == 'lr':
            classifier = LogisticRegression(max_iter=1000, solver='saga', penalty='l2', random_state=42, n_jobs=1)
        elif model == 'rf':
            classifier = RandomForestClassifier(n_estimators=100, max_depth=12, n_jobs=1, random_state=42)
        elif model == 'svc':
            classifier = LinearSVC(dual=False, max_iter=10000, tol=1e-4, random_state=42)
        elif model == 'xgb':
            classifier = XGBClassifier(use_label_encoder=False, eval_metric='logloss',
                                       n_estimators=100, tree_method='hist',
                                       verbosity=0, random_state=42, n_jobs=1)
        else:
            raise ValueError(f"Unknown model identifier: {model}")

        try:
            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_test)
        except Exception as ex:
            print(f"[fitness] fold {fold_idx} training failed for model={model}: {ex}")
            continue

        successful_folds += 1

        # performance for this fold
        try:
            accuracy = accuracy_score(y_test, y_pred)
            if hasattr(classifier, "predict_proba"):
                y_scores = classifier.predict_proba(X_test)
                if y_scores.ndim == 2 and y_scores.shape[1] == 2:
                    y_score_pos = y_scores[:, 1]
                else:
                    y_score_pos = y_scores
            elif hasattr(classifier, "decision_function"):
                y_score_pos = classifier.decision_function(X_test)
            else:
                y_score_pos = y_pred

            if len(np.unique(y_test)) == 2:
                perf_fold = average_precision_score(y_test, y_score_pos)
            else:
                if hasattr(classifier, "predict_proba") and y_scores.ndim == 2:
                    perf_fold = average_precision_score(y_test, y_scores, average='weighted')
                else:
                    perf_fold = accuracy
        except Exception:
            perf_fold = accuracy

        perf_scores.append(perf_fold)

        # fairness for this fold: use original dataframe indices for the test fold
        test_original_idx = X.index[test_idx]
        fairness = fairness_metrics(data, test_original_idx, protected_attribute, y_pred, target_column)
        fair_fold = sum([v for v in fairness.values() if isinstance(v, (int, float, np.floating, np.integer))])
        fair_scores.append(fair_fold)

    if successful_folds == 0:
        print("[fitness] no successful folds during cross-validation; returning failure tuple")
        return float('inf'), None, None

    # aggregate across folds (ignore NaNs)
    performance_score = float(np.nanmean(perf_scores))
    fairness_score = float(np.nanmean(fair_scores)) if len(fair_scores) > 0 else 0.0

    fitness_value = (perf_weight * performance_score) - (fair_weight * fairness_score)
    return fitness_value, fairness_score, performance_score


