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

A CSV-backed, thread-safe cache (``results/fate/runtime_cache.csv``) prevents
re-evaluation of identical pipelines, i.e. the same (model, protected_attribute,
target, ordered list of techniques), across parallel GA runs in ``main.py``.

Floating-point handling (see ``numerics.py`` and README, Section 7):

- Every evaluation runs with ``np.errstate(all='raise')``: an invalid
  operation raises ``FloatingPointError`` instead of yielding ``inf``/``NaN``.
- The only relaxed step is the computation of the fairness ratios, which are
  undefined when a group has no positive predictions or no positive instances
  in a test fold.  These cases are handled explicitly:

  * a metric that is 0/0 in a fold makes that fold's fairness undefined, and
    the fold is excluded from the mean FS;
  * a disparate impact of x/0 (x > 0) is an unbounded disparity: the pipeline
    is *infeasible*;
  * a pipeline without any fold with defined fairness is *infeasible*.

  An infeasible pipeline receives fitness ``INFEASIBLE`` (``-inf``), so it
  ranks below every feasible pipeline and ``NaN`` never reaches the GA.

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
from typing import Optional, Union

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.exceptions import ConvergenceWarning
from xgboost import XGBClassifier
import paths
from preprocessing import prepare_data_model

from practices import apply_techniques
from numerics import (blas_errstate, require_finite, strict_floating_point,
                      undefined_ratio_errstate)
from aif360_setup import silence_unused_backend_notices
# Suppress only aif360's import-time notices about optional back-ends that FATE
# does not use (see aif360_setup.py).  No runtime warning is suppressed.
silence_unused_backend_notices()
from aif360.datasets import BinaryLabelDataset  # noqa: E402
from aif360.metrics import ClassificationMetric  # noqa: E402

logger = logging.getLogger(__name__)

#: An individual's chromosome as accepted by ``fitness``: one technique name or
#: a sequence of names (``None``/empty means "no technique").
TechniqueSpec = Union[str, list[str], tuple[str, ...], None]
#: Cache key: (model, protected attribute, target column, ordered techniques).
CacheKey = tuple[str, str, str, tuple[str, ...]]
#: Cache value: (fitness, fairness score, performance score, raw CSV row).
CacheEntry = tuple[Optional[float], Optional[float], Optional[float], dict[str, str]]
#: Return value of ``fitness``: (fitness, fairness score, performance score).
FitnessResult = tuple[float, Optional[float], Optional[float]]
#: The four classifiers used as optimization tasks.
Classifier = Union[LogisticRegression, RandomForestClassifier, LinearSVC, XGBClassifier]

#: Fitness of an infeasible pipeline (see module docstring); ranks below any
#: feasible pipeline when the GA maximises fitness.
INFEASIBLE = float('-inf')

# Runtime cache — written during the current run (always used for reads and writes).
RUNTIME_CACHE_PATH = str(paths.RUNTIME_CACHE_CSV)
# Root cache — pre-computed fitness values from the paper's experiment grid (read-only).
ROOT_CACHE_PATH = str(paths.REFERENCE_FITNESS_CACHE_CSV)

_runtime_cache_lock = threading.Lock()
_runtime_cache = {}  # key -> (fitness, fairness, performance, row_dict)
_root_cache = {}    # key -> (fitness, fairness, performance, row_dict) — never written

# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------


def _make_simple_key(model: str, protected_attribute: str, target_column: str,
                     technique: TechniqueSpec) -> CacheKey:
    """
    Construct a hashable cache key from fitness-call arguments.

    The techniques are kept as an ordered tuple.  Practices are applied in
    sequence and the order changes the resulting dataset (e.g. oversampling
    before or after undersampling), so a cached value is reused only for the
    identical pipeline.

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
        ``(model_str, protected_str, target_str, tuple_of_techniques)``
    """
    if isinstance(technique, (list, tuple)):
        tech_repr = tuple(str(t) for t in technique)
    else:
        tech_repr = (str(technique),) if technique else ()

    return (str(model), str(protected_attribute), str(target_column), tech_repr)


def _parse_cache_row_metrics(r: dict[str, str]
                             ) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Parse the numeric metric fields from one CSV cache row.

    Parameters
    ----------
    r : dict
        A row dict as produced by ``csv.DictReader``.

    Returns
    -------
    tuple
        ``(fitness, fairness, performance)`` as floats (``None`` for an empty
        field).  A ``NaN`` fitness, written by versions of the code that did
        not handle undefined fairness explicitly, is read as ``INFEASIBLE``,
        so that ``NaN`` never reaches the GA.

    Raises
    ------
    ValueError
        If a non-empty field is not a number (corrupt cache file).
    """
    def _field(name: str) -> Optional[float]:
        value = r.get(name)
        return None if value in (None, '') else float(value)

    fitness_val = _field('fitness')
    if fitness_val is not None and np.isnan(fitness_val):
        fitness_val = INFEASIBLE
    return fitness_val, _field('fairness'), _field('performance')


def _load_cache(path: str) -> dict[CacheKey, CacheEntry]:
    """
    Load a fitness cache CSV into a dict and return it.

    Used at module import time to populate both ``_runtime_cache`` and
    ``_root_cache``.  A missing file yields an empty cache; a corrupt file
    raises (``ValueError`` or ``json.JSONDecodeError``) instead of being
    silently ignored.

    Parameters
    ----------
    path : str
        Absolute path to a cache CSV file.

    Returns
    -------
    dict
        ``{(model, protected_attribute, target_column, tuple(techniques)):
           (fitness, fairness, performance, row_dict)}``
    """
    cache = {}
    if not os.path.exists(path):
        return cache
    with open(path, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            key = _make_simple_key(r.get('model', ''), r.get('protected_attribute', ''),
                                   r.get('target_column', ''),
                                   json.loads(r.get('techniques') or '[]'))
            fitness_val, fairness_val, perf_val = _parse_cache_row_metrics(r)
            cache[key] = (fitness_val, fairness_val, perf_val, r)
    return cache


def _append_runtime_cache_row(model: str, protected_attribute: str, target_column: str,
                              technique: TechniqueSpec, fitness: Optional[float],
                              fairness: Optional[float], performance: Optional[float]) -> None:
    """
    Persist a new fitness result to the runtime cache CSV and update the in-memory dict.

    Thread-safe: guarded by ``_runtime_cache_lock`` to allow concurrent GA
    evaluations from ``main.py``'s ``ThreadPoolExecutor``.  Never writes to
    ``reference/fate/fitness_cache.csv`` (the root read-only cache).

    Parameters
    ----------
    model : str
    protected_attribute : str
    target_column : str
    technique : list or tuple
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
    with _runtime_cache_lock:
        os.makedirs(os.path.dirname(RUNTIME_CACHE_PATH), exist_ok=True)
        first = not os.path.exists(RUNTIME_CACHE_PATH)
        with open(RUNTIME_CACHE_PATH, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=header)
            if first:
                writer.writeheader()
            writer.writerow(row)
        key = _make_simple_key(model, protected_attribute, target_column, technique)
        _runtime_cache[key] = (fitness, fairness, performance, row)


# Load both caches at import time.
# _root_cache is read-only; _runtime_cache accumulates results across runs.
_runtime_cache.update(_load_cache(RUNTIME_CACHE_PATH))
_root_cache.update(_load_cache(ROOT_CACHE_PATH))


# ---------------------------------------------------------------------------
# Fairness metrics helpers (fairness_metrics sub-steps)
# ---------------------------------------------------------------------------

def _nan_fairness_result() -> dict[str, float]:
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


def _binarise_sex_column(prot_series: pd.Series) -> pd.Series:
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


def _binarise_race_column(prot_series: pd.Series) -> pd.Series:
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


def _binarise_age_column(prot_series: pd.Series) -> pd.Series:
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


def _binarise_generic_column(prot_series: pd.Series, protected_attribute: str
                             ) -> Optional[pd.Series]:
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


def _binarise_protected_column(prot_series: pd.Series, protected_attribute: str
                               ) -> Optional[pd.Series]:
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


def _coerce_target_to_binary(test_df: pd.DataFrame, target_column: str) -> Optional[pd.Series]:
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


def _build_aif360_datasets(df_true: pd.DataFrame, df_pred: pd.DataFrame, target_column: str
                           ) -> tuple[BinaryLabelDataset, BinaryLabelDataset]:
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
    ValueError
        If the frames contain missing values (raised by AIF360).
    """
    kwargs = dict(label_names=[target_column],
                  protected_attribute_names=['__prot_bin__'],
                  favorable_label=1, unfavorable_label=0)
    dataset_true = BinaryLabelDataset(df=df_true, **kwargs)
    dataset_pred = BinaryLabelDataset(df=df_pred, **kwargs)
    return dataset_true, dataset_pred


def _run_aif360_classification_metrics(dataset_true: BinaryLabelDataset,
                                       dataset_pred: BinaryLabelDataset
                                       ) -> tuple[float, float, float]:
    """
    Compute SPD, EOD, and DI using AIF360's ``ClassificationMetric``.

    The three metrics are differences and ratios of group rates.  When a group
    has no positive predictions (or no positive instances) in the fold, a rate
    or ratio is undefined and AIF360 returns ``NaN`` (0/0) or ``inf`` (x/0).
    These are the only operations evaluated with relaxed floating-point checks
    (``undefined_ratio_errstate``); their outcome is handled explicitly by
    ``_aggregate_fold_fairness`` and ``_compute_combined_fitness``.

    Parameters
    ----------
    dataset_true : BinaryLabelDataset
    dataset_pred : BinaryLabelDataset

    Returns
    -------
    tuple
        ``(spd, eod, di)`` — raw scalar values from AIF360.

    """
    privileged_groups = [{'__prot_bin__': 1}]
    unprivileged_groups = [{'__prot_bin__': 0}]
    metric = ClassificationMetric(dataset_true, dataset_pred,
                                  unprivileged_groups=unprivileged_groups,
                                  privileged_groups=privileged_groups)
    with undefined_ratio_errstate():
        spd = metric.statistical_parity_difference()
        eod = metric.equal_opportunity_difference()
        di = 1 - metric.disparate_impact()
    return spd, eod, di


def fairness_metrics(test_data: pd.DataFrame, test_indices: Union[pd.Index, np.ndarray],
                     protected_attribute: str, predictions: np.ndarray, target_column: str
                     ) -> dict[str, float]:
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
        All values are absolute (non-negative).  A metric that is undefined
        in this fold is ``NaN`` (0/0) or ``inf`` (disparate impact x/0); all
        three are ``NaN`` when the fold cannot be scored at all (protected
        attribute missing, non-binary or missing labels).  See
        ``_aggregate_fold_fairness`` for how these cases are handled.
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
    except ValueError as e:  # missing values in labels or predictions
        logger.debug("BinaryLabelDataset construction failed: %s", e)
        return _nan_fairness_result()

    spd, eod, di = _run_aif360_classification_metrics(dataset_true, dataset_pred)

    return {
        'statistical_parity': float(abs(spd)) if spd is not None else np.nan,
        'equal_opportunity': float(abs(eod)) if eod is not None else np.nan,
        'disparate_impact': float(abs(di)) if di is not None else np.nan,
    }


# ---------------------------------------------------------------------------
# Fitness helpers (fitness sub-steps)
# ---------------------------------------------------------------------------

def _apply_technique_pipeline(data: pd.DataFrame, technique: TechniqueSpec, protected_attribute: str
                              ) -> pd.DataFrame:
    """
    Apply an individual's chromosome (one or more techniques) to the dataset.

    Role in Algorithm 1 (Step 2 – technique application before model training):
        Iterates over the technique list and calls ``apply_techniques`` for
        each token, in order.  Exceptions are not caught: a practice that
        fails aborts the evaluation instead of being skipped silently.

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
    steps = technique if isinstance(technique, (list, tuple)) else [technique] if technique else []
    with blas_errstate():
        for t in steps:
            data = apply_techniques(data, t, protected_attribute)
    return data


def _build_classifier(model: str) -> Classifier:
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
        return XGBClassifier(eval_metric='logloss',
                             n_estimators=100, tree_method='hist',
                             verbosity=0, random_state=42, n_jobs=1)
    raise ValueError(f"Unknown model identifier: {model}")


def _normalise_binary_target(y_series: pd.Series, target_column: str
                             ) -> tuple[Optional[pd.Series], Optional[FitnessResult]]:
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
        - On failure: ``(None, (INFEASIBLE, None, None))`` — the value
          ``fitness`` returns to the GA for an infeasible pipeline.
    """
    unique_vals = pd.unique(y_series.dropna())
    if len(unique_vals) == 0:
        logger.warning("Empty or all-NaN target column '%s' — infeasible.", target_column)
        return None, (INFEASIBLE, None, None)
    if len(unique_vals) == 1:
        logger.warning("Single-class target '%s' (unique=%s) — infeasible.",
                       target_column, unique_vals)
        return None, (INFEASIBLE, None, None)
    if len(unique_vals) == 2 and set(unique_vals) != {0, 1}:
        mapping = {unique_vals[0]: 0, unique_vals[1]: 1}
        try:
            return y_series.map(mapping).astype(int), None
        except ValueError:  # missing labels cannot be cast to int
            logger.warning("Failed to map binary target values %s to 0/1.", unique_vals)
            return None, (INFEASIBLE, None, None)
    return y_series, None


def _score_fold_performance(classifier: Classifier, x_test: pd.DataFrame, y_test: pd.Series,
                            y_pred: np.ndarray) -> float:
    """
    Compute the performance score (PR-AUC) for one CV fold.

    Uses ``predict_proba`` when available, falls back to ``decision_function``,
    then to the hard predictions themselves.  Uses accuracy only when the test
    fold contains a single class, where PR-AUC is undefined.  The scores are
    checked to be finite.

    Parameters
    ----------
    classifier : fitted sklearn estimator
    x_test : pd.DataFrame
    y_test : pd.Series
    y_pred : np.ndarray
        Hard predictions already produced by ``classifier.predict(x_test)``.

    Returns
    -------
    float
        PR-AUC (or accuracy as fallback).
    """
    if len(np.unique(y_test)) != 2:
        return float(accuracy_score(y_test, y_pred))
    with blas_errstate():
        if hasattr(classifier, "predict_proba"):
            y_score_pos = classifier.predict_proba(x_test)[:, 1]
        elif hasattr(classifier, "decision_function"):
            y_score_pos = classifier.decision_function(x_test)
        else:
            y_score_pos = y_pred
    require_finite(y_score_pos, "classifier scores")
    return float(average_precision_score(y_test, y_score_pos))


def _aggregate_fold_fairness(fairness_dict: dict[str, float]) -> float:
    """
    Sum SPD, EOD and DI of one CV fold, making undefined cases explicit.

    Parameters
    ----------
    fairness_dict : dict
        Output of ``fairness_metrics``.

    Returns
    -------
    float
        - ``NaN`` if any metric is undefined (0/0) in this fold: the fold has
          no defined fairness and is excluded from FS;
        - ``inf`` if disparate impact is x/0 (unbounded disparity);
        - otherwise the finite sum ``|SPD| + |EOD| + |DI|``.
    """
    values = [float(v) for v in fairness_dict.values()]
    if any(np.isnan(v) for v in values):
        return float('nan')
    if any(np.isinf(v) for v in values):
        return float('inf')
    return float(sum(values))


def _run_kfold_evaluation(x: pd.DataFrame, y: pd.Series, data: pd.DataFrame, model: str,
                          protected_attribute: str, target_column: str, n_splits: int = 5
                          ) -> tuple[list[float], list[float], int]:
    """
    Train and evaluate the classifier across K folds, collecting performance and fairness scores.

    Role in Algorithm 1 (Step 2 – cross-validated fitness evaluation):
        Instantiates a fresh classifier per fold, trains it, computes PR-AUC
        via ``_score_fold_performance``, and computes fairness metrics via
        ``fairness_metrics``.

    Parameters
    ----------
    x : pd.DataFrame
        Feature matrix: all columns except the target (the protected
        attribute is included as a feature).
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
        - ``successful_folds`` (int): number of folds that were evaluated
          (folds whose training split contains a single class are skipped).

    Raises
    ------
    NonFiniteResultError
        If the classifier produces non-finite scores.  Training errors are
        not caught.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    perf_scores = []
    fair_scores = []
    successful_folds = 0

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(x, y), start=1):
        x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        if len(pd.unique(y_train.dropna())) < 2:
            logger.debug("Fold %d skipped: only one class in y_train.", fold_idx)
            continue

        classifier = _build_classifier(model)
        with blas_errstate(), warnings.catch_warnings():
            warnings.simplefilter('ignore', ConvergenceWarning)
            classifier.fit(x_train, y_train)
            y_pred = classifier.predict(x_test)
        successful_folds += 1

        perf_scores.append(_score_fold_performance(classifier, x_test, y_test, y_pred))

        test_original_idx = x.index[test_idx]
        fairness_dict = fairness_metrics(data, test_original_idx, protected_attribute,
                                         y_pred, target_column)
        fair_scores.append(_aggregate_fold_fairness(fairness_dict))

    return perf_scores, fair_scores, successful_folds


def _compute_combined_fitness(perf_scores: list[float], fair_scores: list[float],
                              perf_weight: float, fair_weight: float) -> FitnessResult:
    """
    Aggregate per-fold scores into the final fitness value.

    Fitness formula::

        fitness = perf_weight × PS − fair_weight × FS

    where PS = mean PR-AUC and FS = mean over the folds with defined fairness
    of (|SPD| + |EOD| + |DI|) / 3.  The pipeline is infeasible (fitness
    ``INFEASIBLE``) if a fold has an unbounded disparity (``inf``) or if no
    fold has defined fairness.

    Parameters
    ----------
    perf_scores : list of float
    fair_scores : list of float
    perf_weight : float
    fair_weight : float

    Returns
    -------
    tuple
        ``(fitness_value, fairness_score, performance_score)``; for an
        infeasible pipeline ``(INFEASIBLE, inf or nan, performance_score)``.
    """
    require_finite(perf_scores, "performance scores")
    performance_score = float(np.mean(perf_scores))
    defined = [f for f in fair_scores if not np.isnan(f)]
    if len(defined) < len(fair_scores):
        logger.debug("%d of %d folds without defined fairness excluded from FS",
                     len(fair_scores) - len(defined), len(fair_scores))
    if not defined:
        return INFEASIBLE, float('nan'), performance_score
    if any(np.isinf(f) for f in defined):
        return INFEASIBLE, float('inf'), performance_score
    fairness_score = float(np.mean(defined)) / 3
    fitness_value = (perf_weight * performance_score) - (fair_weight * fairness_score)
    return float(fitness_value), fairness_score, performance_score


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def fitness(data: pd.DataFrame, technique: TechniqueSpec, model: str, protected_attribute: str,
            target_column: str, perf_weight: float = 0.5, fair_weight: float = 0.5,
            reset_cache: bool = False) -> FitnessResult:
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
        Controls which caches are consulted for lookups (default False).

        - ``False``: check ``_runtime_cache`` first, then fall back to the
          read-only ``reference/fate/fitness_cache.csv`` root cache.  New evaluations are
          written to ``results/fate/runtime_cache.csv`` only.
        - ``True``: skip the root cache entirely; use only the runtime cache.
          Useful when you want a fresh evaluation uncontaminated by the paper's
          pre-computed results.

    Returns
    -------
    tuple
        ``(fitness_value, fairness_score, performance_score)``.
        ``fitness_value`` is ``INFEASIBLE`` (``-inf``) for an infeasible
        pipeline: empty or single-class target, no evaluable fold, an
        unbounded disparity, or no fold with defined fairness.

    Raises
    ------
    FloatingPointError
        On any invalid floating-point operation (the evaluation runs under
        ``np.errstate(all='raise')``, see ``numerics.py``).
    NonFiniteResultError
        If the features, the classifier scores or the performance scores
        contain non-finite values.

    Notes
    -----
    A cache hit (keyed on model / protected_attribute / target_column /
    ordered techniques) returns the stored triple without re-evaluating.
    """
    key = _make_simple_key(model, protected_attribute, target_column, technique)

    # Runtime cache check (always — accumulates results from the current run).
    with _runtime_cache_lock:
        cached = _runtime_cache.get(key)
    if cached and cached[0] is not None:
        return cached[0], cached[1], cached[2]

    # Root cache check (only when reset_cache=False — read-only, never written).
    if not reset_cache:
        cached = _root_cache.get(key)
        if cached and cached[0] is not None:
            return cached[0], cached[1], cached[2]

    with strict_floating_point():
        result = _evaluate(data, technique, model, protected_attribute, target_column,
                           perf_weight, fair_weight)
    try:
        _append_runtime_cache_row(model, protected_attribute, target_column, technique,
                                  *result)
    except OSError as e:  # the value is still returned; only persistence failed
        logger.warning("Could not write the runtime cache: %s", e)
    return result


def _evaluate(data: pd.DataFrame, technique: TechniqueSpec, model: str,
              protected_attribute: str, target_column: str, perf_weight: float,
              fair_weight: float) -> FitnessResult:
    """Evaluate one pipeline (body of ``fitness`` without the cache)."""
    data = _apply_technique_pipeline(data, technique, protected_attribute)
    data = prepare_data_model(data, target_column, protected_attribute).copy()

    y = data[target_column]
    x = data.drop(columns=[target_column])
    require_finite(x, "features after Data Preparation")

    y, error_result = _normalise_binary_target(pd.Series(y), target_column)
    if error_result is not None:
        return error_result

    if len(y.dropna().value_counts()) < 2:
        logger.warning("Single class after preprocessing (unique=%s) — infeasible.",
                       pd.unique(y))
        return INFEASIBLE, None, None

    perf_scores, fair_scores, successful_folds = _run_kfold_evaluation(
        x, y, data, model, protected_attribute, target_column)

    if successful_folds == 0:
        logger.warning("No evaluable CV fold for model=%s — infeasible.", model)
        return INFEASIBLE, None, None

    return _compute_combined_fitness(perf_scores, fair_scores, perf_weight, fair_weight)
