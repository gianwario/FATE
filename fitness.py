# fitness.py
"""
Fitness evaluation for FATE: performance scoring, fairness metrics, and result caching.

This module implements Step 2 of Algorithm 1 (fitness evaluation).  For a
candidate individual — an ordered list of fairness-aware preprocessing
techniques — it:

1. Applies each technique in sequence to the dataset via
   ``practices.apply_techniques``.
2. Prepares a model-ready feature matrix via
   ``preprocessing.prepare_data_model``.
3. Trains and evaluates the specified classifier under 5-fold cross-validation
   (KFold, ``n_splits=5``, ``random_state=42``).
4. Computes the **Performance Score (PS)** as the mean PR-AUC across folds.
5. Computes the **Fairness Score (FS)** as::

       FS = mean_fold( (|SPD| + |EOD| + |DI|) / 3 )

   where SPD = statistical parity difference, EOD = equal opportunity
   difference, DI = disparate impact deviation (``1 − DI_ratio``).
6. Returns the combined fitness::

       fitness = perf_weight × PS − fair_weight × FS

A CSV-backed, thread-safe cache (``FATE_output/fitness_cache.csv``) prevents
re-evaluation of identical (model, protected_attribute, target, techniques)
combinations across parallel GA runs in ``main.py``.

Classifiers supported (selected via string identifier):
    ``'lr'``  – LogisticRegression (saga solver, L2 penalty, max_iter=1000).
    ``'rf'``  – RandomForestClassifier (100 trees, max_depth=12).
    ``'svc'`` – LinearSVC (dual=False, max_iter=10 000).
    ``'xgb'`` – XGBClassifier (hist tree method, 100 estimators).
"""
import logging
import os
import threading
import csv
import json
import time
import warnings

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.exceptions import ConvergenceWarning
from xgboost import XGBClassifier
from preprocessing import prepare_data_model

from practices import apply_techniques
# Suppress aif360's import-time logging warnings (tensorflow / inFairness extras)
# and all RuntimeWarnings that aif360 emits during metric computation
# (e.g. divide-by-zero in disparate_impact when a group has no positive predictions).
logging.getLogger('aif360').setLevel(logging.ERROR)
warnings.filterwarnings('ignore', module=r'aif360\..*')
warnings.filterwarnings('ignore', module=r'inFairness\..*')
from aif360.datasets import BinaryLabelDataset  # noqa: E402
from aif360.metrics import ClassificationMetric  # noqa: E402

logger = logging.getLogger(__name__)

# All FATE output lives in FATE_output/ — never touches pre-existing CSV files.
_FATE_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'FATE_output')
os.makedirs(_FATE_OUTPUT_DIR, exist_ok=True)

SIMPLE_CACHE_PATH = os.path.join(_FATE_OUTPUT_DIR, 'fitness_cache.csv')
simple_cache_lock = threading.Lock()
simple_cache = {}  # key -> (fitness, fairness, performance, row_dict)

# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------


def _make_simple_key(model, protected_attribute, target_column, technique):
    """
    Construct a hashable cache key from fitness-call arguments.

    The technique argument is converted to a ``frozenset`` so the cache is
    order-insensitive: two individuals with the same techniques in different
    orders are treated as equivalent (applying the same set of techniques
    produces the same dataset regardless of order under the current sequential
    application logic).

    Parameters
    ----------
    model : str
        Classifier identifier.
    protected_attribute : str
        Protected attribute column name.
    target_column : str
        Target column name.
    technique : str, list, or tuple
        Technique token or ordered list of technique tokens.

    Returns
    -------
    tuple
        ``(model_str, protected_str, target_str, frozenset_of_techniques)``
    """
    # Convert technique to a frozenset to ignore order
    if isinstance(technique, (list, tuple)):
        tech_repr = frozenset(technique)
    else:
        tech_repr = frozenset([technique]) if technique else frozenset()

    return (str(model), str(protected_attribute), str(target_column), tech_repr)


def _parse_cache_row_metrics(r):
    """
    Parse the numeric metric fields from one CSV cache row.

    Parameters
    ----------
    r : dict
        A row dict as produced by ``csv.DictReader``.

    Returns
    -------
    tuple
        ``(fitness, fairness, performance)`` as floats, or ``None`` for any
        field that is missing, empty, or non-numeric.
    """
    try:
        fitness_val = float(r.get('fitness')) if r.get('fitness') not in (None, '') else None
        fairness_val = float(r.get('fairness')) if r.get('fairness') not in (None, '') else None
        perf_val = float(r.get('performance')) if r.get('performance') not in (None, '') else None
    except Exception:
        fitness_val = fairness_val = perf_val = None
    return fitness_val, fairness_val, perf_val


def _load_simple_cache():
    """
    Load previously computed fitness results from the on-disk CSV cache.

    Called once at module import time.  Populates the module-level
    ``simple_cache`` dict so that repeated GA runs across scripts can skip
    already-evaluated individuals.  Corrupt or missing cache files are silently
    ignored and the GA continues without cached results.
    """
    if not os.path.exists(SIMPLE_CACHE_PATH):
        return
    try:
        with open(SIMPLE_CACHE_PATH, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for r in reader:
                key = (r.get('model', ''), r.get('protected_attribute', ''),
                       r.get('target_column', ''),
                       frozenset(json.loads(r.get('techniques', ''))))
                fitness_val, fairness_val, perf_val = _parse_cache_row_metrics(r)
                simple_cache[key] = (fitness_val, fairness_val, perf_val, r)
    except Exception:
        # ignore corrupt cache file
        return


def _append_simple_cache_row(model, protected_attribute, target_column, technique,
                             fitness, fairness, performance):
    """
    Persist a new fitness result to the CSV cache and update the in-memory dict.

    Thread-safe: guarded by ``simple_cache_lock`` to allow concurrent GA
    evaluations from ``main.py``'s ``ThreadPoolExecutor``.

    Parameters
    ----------
    model : str
    protected_attribute : str
    target_column : str
    technique : frozenset or list
        Technique set as originally passed to ``_make_simple_key``.
    fitness : float or None
    fairness : float or None
    performance : float or None
    """
    header = ['timestamp', 'model', 'protected_attribute', 'target_column',
              'techniques', 'fitness', 'fairness', 'performance', 'extra']
    row = {
        'timestamp': str(time.time()),
        'model': str(model),
        'protected_attribute': str(protected_attribute),
        'target_column': str(target_column),
        'techniques': json.dumps(list(technique), default=str, separators=(',', ':')),
        'fitness': '' if fitness is None else str(fitness),
        'fairness': '' if fairness is None else str(fairness),
        'performance': '' if performance is None else str(performance),
        'extra': ''
    }
    with simple_cache_lock:
        first = not os.path.exists(SIMPLE_CACHE_PATH)
        with open(SIMPLE_CACHE_PATH, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=header)
            if first:
                writer.writeheader()
            writer.writerow(row)
        key = _make_simple_key(model, protected_attribute, target_column, technique)
        simple_cache[key] = (fitness, fairness, performance, row)


def _delete_cache_file():
    """
    Delete the on-disk cache CSV file.

    Returns
    -------
    bool
        True on success; False if the file could not be removed.
    """
    try:
        os.remove(SIMPLE_CACHE_PATH)
        return True
    except Exception as e:
        logger.error("Could not remove cache file: %s", e)
        return False


def _recreate_empty_cache_file():
    """
    Write a fresh, empty cache CSV with the correct header columns.

    Returns
    -------
    bool
        True on success; False if the file could not be created.
    """
    header = ['timestamp', 'model', 'protected_attribute',
              'target_column', 'techniques', 'fitness',
              'fairness', 'performance', 'extra']
    try:
        with open(SIMPLE_CACHE_PATH, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
        return True
    except Exception as e:
        logger.error("Could not recreate cache file: %s", e)
        return False


def _conditionally_remove_cache_file(remove_file):
    """
    Remove the cache file from disk if *remove_file* is True and the file exists.

    Parameters
    ----------
    remove_file : bool

    Returns
    -------
    bool
        True if nothing needed to be done or the deletion succeeded; False on error.
    """
    if not remove_file:
        return True
    if not os.path.exists(SIMPLE_CACHE_PATH):
        return True
    return _delete_cache_file()


def _conditionally_recreate_cache_file(recreate, remove_file):
    """
    Recreate an empty cache file if both *recreate* and *remove_file* are True.

    Parameters
    ----------
    recreate : bool
    remove_file : bool

    Returns
    -------
    bool
        True if nothing needed to be done or recreation succeeded; False on error.
    """
    if not recreate or not remove_file:
        return True
    return _recreate_empty_cache_file()


def clear_simple_cache(remove_file=True, recreate=False):
    """
    Clear the in-memory and on-disk fitness cache.

    Used at the start of each new parameter-grid run (``reset_cache=True``
    in ``fitness()``) to avoid stale cached values contaminating fresh
    experimental conditions.

    Parameters
    ----------
    remove_file : bool, optional
        If True (default), delete the CSV file from disk.
    recreate : bool, optional
        If True and *remove_file* is True, write a fresh empty CSV with the
        correct header after deletion.

    Returns
    -------
    bool
        True if the operation succeeded; False if a filesystem error occurred.
    """
    with simple_cache_lock:
        simple_cache.clear()
        if not _conditionally_remove_cache_file(remove_file):
            return False
        if not _conditionally_recreate_cache_file(recreate, remove_file):
            return False

    return True


# load simple cache at import time
_load_simple_cache()


# ---------------------------------------------------------------------------
# Fairness metrics helpers (fairness_metrics sub-steps)
# ---------------------------------------------------------------------------

def _nan_fairness_result():
    """
    Return the canonical NaN fairness result dict.

    Used as the error return value in all failure paths of ``fairness_metrics``.

    Returns
    -------
    dict
        ``{'statistical_parity': nan, 'equal_opportunity': nan,
        'disparate_impact': nan}``
    """
    return {'statistical_parity': np.nan,
            'equal_opportunity': np.nan,
            'disparate_impact': np.nan}


def _binarise_sex_column(prot_series):
    """
    Map the ``sex`` protected attribute to a binary privileged indicator.

    Privileged group = male (numeric 1, or string equal to ``'male'``/``'m'``).

    Parameters
    ----------
    prot_series : pd.Series
        Raw protected attribute values for the test fold.

    Returns
    -------
    pd.Series
        Integer series; 1 = privileged (male), 0 = unprivileged (female).
    """
    if pd.api.types.is_numeric_dtype(prot_series):
        return prot_series.fillna(0).astype(int).apply(lambda v: 1 if v == 1 else 0)
    return (prot_series.fillna('').astype(str).str.lower()
            .apply(lambda s: 1 if s in ('male', 'm') else 0))


def _binarise_race_column(prot_series):
    """
    Map the ``race`` protected attribute to a binary privileged indicator.

    Privileged group = numeric value 1.

    Parameters
    ----------
    prot_series : pd.Series

    Returns
    -------
    pd.Series
        Integer series; 1 = privileged, 0 = unprivileged.
    """
    return (pd.to_numeric(prot_series, errors='coerce')
            .fillna(0).astype(int)
            .apply(lambda v: 1 if v == 1 else 0))


def _binarise_age_column(prot_series):
    """
    Map the ``age`` protected attribute to a binary privileged indicator.

    Privileged group = age strictly above the column mean.

    Parameters
    ----------
    prot_series : pd.Series

    Returns
    -------
    pd.Series
        Integer series; 1 = privileged (above-mean age), 0 = unprivileged.
    """
    age_vals = pd.to_numeric(prot_series, errors='coerce')
    mean_age = age_vals.mean()
    return age_vals.apply(lambda v: 1 if pd.notna(v) and v > mean_age else 0).astype(int)


def _binarise_generic_column(prot_series, protected_attribute):
    """
    Map an unrecognised protected attribute to a binary indicator using the majority value.

    Privileged group = most frequent value in *prot_series*.

    Parameters
    ----------
    prot_series : pd.Series
    protected_attribute : str
        Used only for the diagnostic print message.

    Returns
    -------
    pd.Series or None
        Integer series; 1 = privileged, 0 = unprivileged.
        Returns ``None`` if the series has no values (signals a NaN result).
    """
    vc = prot_series.value_counts(dropna=False)
    if len(vc) == 0:
        logger.debug("Protected attribute '%s' has no values", protected_attribute)
        return None
    priv = vc.idxmax()
    return prot_series.fillna('').astype(str).apply(lambda s: 1 if s == str(priv) else 0)


def _binarise_protected_column(prot_series, protected_attribute):
    """
    Dispatch to the correct binarisation rule based on the attribute name.

    Rules (matched by substring of the lowercase column name):
        ``sex``  → ``_binarise_sex_column``
        ``race`` → ``_binarise_race_column``
        ``age``  → ``_binarise_age_column``
        other   → ``_binarise_generic_column`` (most-frequent-value fallback)

    Parameters
    ----------
    prot_series : pd.Series
        Raw protected attribute values.
    protected_attribute : str
        Column name used to select the rule.

    Returns
    -------
    pd.Series or None
        Binary indicator series (1=privileged, 0=unprivileged), or ``None``
        if the generic fallback detects an empty series.
    """
    name_lower = protected_attribute.lower()
    if 'sex' in name_lower:
        return _binarise_sex_column(prot_series)
    if 'race' in name_lower:
        return _binarise_race_column(prot_series)
    if 'age' in name_lower:
        return _binarise_age_column(prot_series)
    return _binarise_generic_column(prot_series, protected_attribute)


def _coerce_target_to_binary(test_df, target_column):
    """
    Coerce the true-label column to a binary {0, 1} series.

    Handles two common label encodings found in the study datasets:
        ``{0, 1}`` — returned as-is (cast to int).
        ``{1, 2}`` — mapped to ``{0, 1}`` (2→1, 1→0).

    Parameters
    ----------
    test_df : pd.DataFrame
        Test fold DataFrame (index already reset).
    target_column : str

    Returns
    -------
    pd.Series or None
        Binary label series, or ``None`` if the values are non-binary and
        cannot be handled (signals a NaN result).
    """
    y_series = pd.Series(test_df[target_column]).copy()
    uniq = set(pd.unique(y_series.dropna()))
    if uniq <= {0, 1}:
        return y_series.astype(int)
    if uniq <= {1, 2}:
        return y_series.map(lambda v: 1 if v == 2 else 0 if v == 1 else np.nan).astype(float)
    logger.debug("Non-binary target values in test set: %s", pd.unique(y_series))
    return None


def _build_aif360_datasets(df_true, df_pred, target_column):
    """
    Construct a pair of AIF360 ``BinaryLabelDataset`` objects for metric computation.

    Parameters
    ----------
    df_true : pd.DataFrame
        Columns: ``__prot_bin__`` (binary protected indicator) + *target_column* (true labels).
    df_pred : pd.DataFrame
        Same structure as *df_true* but with predicted labels.
    target_column : str

    Returns
    -------
    tuple
        ``(dataset_true, dataset_pred)`` — both are ``BinaryLabelDataset`` instances.

    Raises
    ------
    Exception
        Any error raised by the AIF360 constructor is propagated to the caller
        (``fairness_metrics``) for logging and NaN fallback.
    """
    kwargs = dict(label_names=[target_column],
                  protected_attribute_names=['__prot_bin__'],
                  favorable_label=1, unfavorable_label=0)
    dataset_true = BinaryLabelDataset(df=df_true, **kwargs)
    dataset_pred = BinaryLabelDataset(df=df_pred, **kwargs)
    return dataset_true, dataset_pred


def _run_aif360_classification_metrics(dataset_true, dataset_pred):
    """
    Compute SPD, EOD, and DI using AIF360's ``ClassificationMetric``.

    Parameters
    ----------
    dataset_true : BinaryLabelDataset
    dataset_pred : BinaryLabelDataset

    Returns
    -------
    tuple
        ``(spd, eod, di)`` — raw scalar values from AIF360.

    Raises
    ------
    Exception
        Any error raised by ``ClassificationMetric`` is propagated to the
        caller (``fairness_metrics``) for logging and NaN fallback.
    """
    privileged_groups = [{'__prot_bin__': 1}]
    unprivileged_groups = [{'__prot_bin__': 0}]
    metric = ClassificationMetric(dataset_true, dataset_pred,
                                  unprivileged_groups=unprivileged_groups,
                                  privileged_groups=privileged_groups)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        spd = metric.statistical_parity_difference()
        eod = metric.equal_opportunity_difference()
        di = 1 - metric.disparate_impact()
    return spd, eod, di


def fairness_metrics(test_data, test_indices, protected_attribute, predictions, target_column):
    """
    Compute fairness metrics for binary classification using AIF360.

    Implements the fairness component (FS) of the FATE fitness function
    (Algorithm 1, Step 2).  Three metrics are returned:

    - **Statistical Parity Difference (SPD)**: difference in positive
      prediction rates between privileged and unprivileged groups.
    - **Equal Opportunity Difference (EOD)**: difference in true positive
      rates between privileged and unprivileged groups.
    - **Disparate Impact (DI)**: ``1 − (P(Ŷ=1|unprivileged) /
      P(Ŷ=1|privileged))``, so 0 is perfectly fair.

    Privileged / unprivileged group assignment follows hard-coded rules:

    - ``sex``  : numeric 1 or string containing ``'male'``/``'m'`` → privileged.
    - ``race`` : numeric value 1 → privileged.
    - ``age``  : above the column mean → privileged.
    - other   : most frequent value → privileged (fallback).

    Parameters
    ----------
    test_data : pd.DataFrame
        Full dataset indexed by its original row indices (not reset).
    test_indices : pd.Index or array-like
        Row indices of the current test fold, used to slice *test_data*.
    protected_attribute : str
        Column name of the protected attribute in *test_data*.
    predictions : array-like
        Predicted binary labels (0/1) for the test fold, ordered to match
        ``test_data.loc[test_indices]``.
    target_column : str
        Column name of the ground-truth binary label.

    Returns
    -------
    dict
        ``{'statistical_parity': float, 'equal_opportunity': float,
        'disparate_impact': float}``
        All values are absolute (non-negative).  Returns NaN for any metric
        that cannot be computed (e.g., protected attribute missing, single
        class in test fold, AIF360 construction failure).
    """
    preds = pd.Series(predictions, index=test_indices)
    test_df = test_data.loc[test_indices].copy().reset_index(drop=True)
    preds = preds.reset_index(drop=True)

    if protected_attribute not in test_df.columns:
        logger.debug("Protected attribute '%s' not found in test fold", protected_attribute)
        return _nan_fairness_result()

    prot_series = test_df[protected_attribute]
    prot_bin = _binarise_protected_column(prot_series, protected_attribute)
    if prot_bin is None:
        return _nan_fairness_result()

    y_bin = _coerce_target_to_binary(test_df, target_column)
    if y_bin is None:
        return _nan_fairness_result()

    df_true = pd.DataFrame({
        '__prot_bin__': prot_bin.reset_index(drop=True),
        target_column: y_bin.reset_index(drop=True),
    })
    df_pred = pd.DataFrame({
        '__prot_bin__': prot_bin.reset_index(drop=True),
        target_column: pd.Series(preds).reset_index(drop=True),
    })
    df_true[target_column] = pd.to_numeric(df_true[target_column], errors='coerce')
    df_pred[target_column] = pd.to_numeric(df_pred[target_column], errors='coerce')

    try:
        dataset_true, dataset_pred = _build_aif360_datasets(df_true, df_pred, target_column)
    except Exception as e:
        logger.debug("BinaryLabelDataset construction failed: %s", e)
        return _nan_fairness_result()

    try:
        spd, eod, di = _run_aif360_classification_metrics(dataset_true, dataset_pred)
    except Exception as e:
        logger.debug("ClassificationMetric computation failed: %s", e)
        return _nan_fairness_result()

    return {
        'statistical_parity': float(abs(spd)) if spd is not None else np.nan,
        'equal_opportunity': float(abs(eod)) if eod is not None else np.nan,
        'disparate_impact': float(abs(di)) if di is not None else np.nan,
    }


# ---------------------------------------------------------------------------
# Fitness helpers (fitness sub-steps)
# ---------------------------------------------------------------------------

def _apply_technique_pipeline(data, technique, protected_attribute):
    """
    Apply an individual's chromosome (one or more techniques) to the dataset.

    Role in Algorithm 1 (Step 2 – technique application before model training):
        Iterates over the technique list and calls ``apply_techniques`` for
        each token.  Failed applications are silently skipped so the GA can
        continue if a technique raises an exception on a particular dataset.

    Parameters
    ----------
    data : pd.DataFrame
        Starting dataset (copy of the input passed to ``fitness``).
    technique : str or list of str
        Individual's chromosome.
    protected_attribute : str
        Forwarded to each technique function.

    Returns
    -------
    pd.DataFrame
        Transformed dataset after all techniques have been applied.
    """
    if isinstance(technique, (list, tuple)):
        for t in technique:
            try:
                data = apply_techniques(data, t, protected_attribute)
            except Exception:
                continue  # skip failing techniques; GA continues
    elif technique:
        try:
            data = apply_techniques(data, technique, protected_attribute)
        except Exception:
            pass
    return data


def _build_classifier(model):
    """
    Instantiate a fresh classifier by string identifier.

    A new instance is created per fold to prevent any state from leaking
    between folds.

    Parameters
    ----------
    model : str
        One of: ``'lr'``, ``'rf'``, ``'svc'``, ``'xgb'``.

    Returns
    -------
    sklearn estimator
        Unfitted classifier configured with the fixed hyperparameters used
        throughout the paper's experiments.

    Raises
    ------
    ValueError
        If *model* is not one of the four recognised identifiers.
    """
    if model == 'lr':
        return LogisticRegression(max_iter=1000, solver='saga', penalty='l2',
                                  random_state=42, n_jobs=1)
    if model == 'rf':
        return RandomForestClassifier(n_estimators=100, max_depth=12,
                                      n_jobs=1, random_state=42)
    if model == 'svc':
        return LinearSVC(dual=False, max_iter=10000, tol=1e-4, random_state=42)
    if model == 'xgb':
        return XGBClassifier(use_label_encoder=False, eval_metric='logloss',
                             n_estimators=100, tree_method='hist',
                             verbosity=0, random_state=42, n_jobs=1)
    raise ValueError(f"Unknown model identifier: {model}")


def _normalise_binary_target(y_series, target_column):
    """
    Validate the target vector and map it to a binary {0, 1} label series.

    Parameters
    ----------
    y_series : pd.Series
        Raw label column values.
    target_column : str
        Column name (used only for diagnostic messages).

    Returns
    -------
    tuple
        ``(y, error_tuple)``

        - On success: ``(mapped_series, None)``
        - On failure: ``(None, (float('inf'), None, None))`` — the error tuple
          is the sentinel value ``fitness`` returns to the GA.
    """
    unique_vals = pd.unique(y_series.dropna())
    if len(unique_vals) == 0:
        logger.warning("Empty or all-NaN target column '%s' — skipping.", target_column)
        return None, (float('inf'), None, None)
    if len(unique_vals) == 1:
        logger.warning("Single-class target '%s' (unique=%s) — cannot train classifier.",
                       target_column, unique_vals)
        return None, (float('inf'), None, None)
    if len(unique_vals) == 2 and set(unique_vals) != {0, 1}:
        mapping = {unique_vals[0]: 0, unique_vals[1]: 1}
        try:
            return y_series.map(mapping).astype(int), None
        except Exception:
            logger.warning("Failed to map binary target values %s to 0/1.", unique_vals)
            return None, (float('inf'), None, None)
    return y_series, None


def _score_fold_performance(classifier, X_test, y_test, y_pred):
    """
    Compute the performance score (PR-AUC) for one CV fold.

    Uses ``predict_proba`` when available, falls back to ``decision_function``,
    then to the hard predictions themselves.  Falls back to accuracy when the
    PR-AUC cannot be computed.

    Parameters
    ----------
    classifier : fitted sklearn estimator
    X_test : pd.DataFrame
    y_test : pd.Series
    y_pred : np.ndarray
        Hard predictions already produced by ``classifier.predict(X_test)``.

    Returns
    -------
    float
        PR-AUC (or accuracy as fallback).
    """
    try:
        accuracy = accuracy_score(y_test, y_pred)
        if hasattr(classifier, "predict_proba"):
            y_scores = classifier.predict_proba(X_test)
            y_score_pos = (
                y_scores[:, 1] if y_scores.ndim == 2 and y_scores.shape[1] == 2 else y_scores
            )
        elif hasattr(classifier, "decision_function"):
            y_score_pos = classifier.decision_function(X_test)
        else:
            y_score_pos = y_pred

        if len(np.unique(y_test)) == 2:
            return average_precision_score(y_test, y_score_pos)
        if hasattr(classifier, "predict_proba") and y_scores.ndim == 2:
            return average_precision_score(y_test, y_scores, average='weighted')
        return accuracy
    except Exception:
        # accuracy defined at top of try; propagates NameError if accuracy_score failed
        return accuracy


def _aggregate_fold_fairness(fairness_dict):
    """
    Sum the numeric fairness metric values from one CV fold.

    Parameters
    ----------
    fairness_dict : dict
        Output of ``fairness_metrics``.

    Returns
    -------
    float
        Sum of SPD, EOD, and DI for this fold (non-finite values are excluded).
    """
    return sum(v for v in fairness_dict.values()
               if isinstance(v, (int, float, np.floating, np.integer)))


def _run_kfold_evaluation(X, y, data, model, protected_attribute, target_column, n_splits=5):
    """
    Train and evaluate the classifier across K folds, collecting performance and fairness scores.

    Role in Algorithm 1 (Step 2 – cross-validated fitness evaluation):
        Instantiates a fresh classifier per fold, trains it, computes PR-AUC
        via ``_score_fold_performance``, and computes fairness metrics via
        ``fairness_metrics``.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (protected attribute excluded).
    y : pd.Series
        Binary label vector.
    data : pd.DataFrame
        Full processed dataset retained for fairness metric computation
        (needs original indices).
    model : str
        Classifier identifier.
    protected_attribute : str
    target_column : str
    n_splits : int, optional
        Number of K-fold splits (default 5).

    Returns
    -------
    tuple
        ``(perf_scores, fair_scores, successful_folds)``

        - ``perf_scores`` (list of float): per-fold PR-AUC values.
        - ``fair_scores`` (list of float): per-fold fairness sums.
        - ``successful_folds`` (int): number of folds that completed without
          a training error.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    perf_scores = []
    fair_scores = []
    successful_folds = 0

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X, y), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        if len(pd.unique(y_train.dropna())) < 2:
            logger.debug("Fold %d skipped: only one class in y_train.", fold_idx)
            continue

        try:
            classifier = _build_classifier(model)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', ConvergenceWarning)
                classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_test)
        except Exception as ex:
            logger.debug("Fold %d training failed for model=%s: %s", fold_idx, model, ex)
            continue

        successful_folds += 1

        perf_scores.append(_score_fold_performance(classifier, X_test, y_test, y_pred))

        test_original_idx = X.index[test_idx]
        fairness_dict = fairness_metrics(data, test_original_idx, protected_attribute,
                                         y_pred, target_column)
        fair_scores.append(_aggregate_fold_fairness(fairness_dict))

    return perf_scores, fair_scores, successful_folds


def _compute_combined_fitness(perf_scores, fair_scores, perf_weight, fair_weight):
    """
    Aggregate per-fold scores into the final fitness value.

    Fitness formula::

        fitness = perf_weight × PS − fair_weight × FS

    where PS = mean PR-AUC and FS = mean(sum of |SPD|+|EOD|+|DI|) / 3.

    Parameters
    ----------
    perf_scores : list of float
    fair_scores : list of float
    perf_weight : float
    fair_weight : float

    Returns
    -------
    tuple
        ``(fitness_value, fairness_score, performance_score)``
    """
    performance_score = float(np.nanmean(perf_scores))
    fairness_score = (float(np.nanmean(fair_scores)) / 3) if len(fair_scores) > 0 else 0.0
    fitness_value = (perf_weight * performance_score) - (fair_weight * fairness_score)
    return fitness_value, fairness_score, performance_score


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def fitness(data, technique, model, protected_attribute, target_column,
            perf_weight=0.5, fair_weight=0.5, reset_cache=False):
    """
    Evaluate the fitness of a candidate technique list for a given classifier.

    This is the core evaluation oracle of Algorithm 1 (Step 2 – fitness
    evaluation).  It applies the candidate preprocessing pipeline, trains
    the classifier under 5-fold cross-validation, and returns a composite
    score balancing model performance and algorithmic fairness.

    Fitness formula::

        fitness = perf_weight × PS − fair_weight × FS

    where:

    - **PS** (Performance Score) = mean PR-AUC across folds.
    - **FS** (Fairness Score)    = mean ``(|SPD| + |EOD| + |DI|) / 3``
      across folds, normalised to [0, 1].

    Parameters
    ----------
    data : pd.DataFrame
        Dataset prepared by ``preprocessing.prepare_data_model`` (features +
        target + protected attribute column).
    technique : str or list of str
        Individual's chromosome: one technique name or an ordered list.
        Applied left-to-right via ``practices.apply_techniques``.
    model : str
        Classifier identifier: ``'lr'``, ``'rf'``, ``'svc'``, or ``'xgb'``.
    protected_attribute : str
        Name of the protected attribute column in *data*.
    target_column : str
        Name of the label column in *data*.
    perf_weight : float, optional
        Weight on the performance term (default 0.5).
    fair_weight : float, optional
        Weight on the fairness term (default 0.5).
    reset_cache : bool, optional
        If True, clear and recreate the on-disk cache before evaluating.
        Set to True by ``genetic_algorithm`` on the very first fitness call
        of each GA run to prevent stale entries from a prior parameter-grid
        point (default False).

    Returns
    -------
    tuple
        ``(fitness_value, fairness_score, performance_score)``
        Returns ``(float('inf'), None, None)`` on catastrophic failure
        (empty target, single class, or zero successful CV folds).

    Notes
    -----
    Failed technique applications are silently skipped (``try/except`` in
    the technique loop) so the GA can continue if a technique crashes on a
    particular dataset.

    A cache hit (keyed on model / protected_attribute / target_column /
    frozenset(technique)) returns the stored triple immediately without
    re-evaluating the classifier.
    """
    if reset_cache:
        clear_simple_cache(remove_file=True, recreate=True)

    key = _make_simple_key(model, protected_attribute, target_column, technique)
    with simple_cache_lock:
        cached = simple_cache.get(key)
    if cached and cached[0] is not None:
        return cached[0], cached[1], cached[2]

    data = _apply_technique_pipeline(data, technique, protected_attribute)
    data = prepare_data_model(data, target_column, protected_attribute)
    data = data.copy()

    y = data[target_column]
    X = data.drop(columns=[target_column])

    y, error_result = _normalise_binary_target(pd.Series(y), target_column)
    if error_result is not None:
        return error_result

    if len(y.dropna().value_counts()) < 2:
        logger.warning("Single class after preprocessing (unique=%s) — returning failure.",
                       pd.unique(y))
        return float('inf'), None, None

    perf_scores, fair_scores, successful_folds = _run_kfold_evaluation(
        X, y, data, model, protected_attribute, target_column)

    if successful_folds == 0:
        logger.warning("No successful CV folds for model=%s — returning failure.", model)
        return float('inf'), None, None

    fitness_value, fairness_score, performance_score = _compute_combined_fitness(
        perf_scores, fair_scores, perf_weight, fair_weight)

    try:
        _append_simple_cache_row(model, protected_attribute, target_column, technique,
                                 fitness_value, fairness_score, performance_score)
    except Exception:
        pass

    return fitness_value, fairness_score, performance_score
