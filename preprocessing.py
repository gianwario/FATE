"""
Dataset loading, cleaning, encoding, and dataset-specific preparation utilities.

This module provides two layers of data preparation:

1. **Generic pipeline functions** (``load_dataset``, ``minimal_clean``,
   ``encode_and_impute``, ``binarize_target``, ``prepare_data_model``)
   that transform a raw DataFrame into a model-ready feature matrix.
   These are invoked inside ``fitness.fitness`` (Step 2 of Algorithm 1)
   after the fairness techniques from ``practices`` have been applied.

2. **Dataset-specific preparers** (``prepare_adult``, ``prepare_german``,
   ``prepare_heart``) that normalise raw CSV layouts — encoding sex codes,
   remapping target labels, coercing types — before the generic pipeline
   runs.  These are called in ``main.worker_task`` and
   ``RQ2_data_analysis.preprocessing_experiments.prepare_sample_ready``.

Role in Algorithm 1:
    ``prepare_data_model`` is the final preparation step of the fitness
    evaluation (Step 2): after the sequence of fairness techniques has been
    applied to the dataset, it ensures a clean, fully numeric feature matrix
    is ready for classifier training and cross-validation.
"""
from typing import Iterable, Optional

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer


def load_dataset(path: str) -> pd.DataFrame:
    """
    Load a CSV file from disk into a pandas DataFrame.

    Parameters
    ----------
    path : str
        File-system path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Raw dataset as produced by ``pd.read_csv``.
    """
    return pd.read_csv(path)


def sample_dataset(df: pd.DataFrame, fraction: float = 1.0, random_state: int = 42) -> pd.DataFrame:
    """
    Draw a random fractional sample of a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Source DataFrame.
    fraction : float, optional
        Proportion of rows to retain (default 1.0, full dataset).
    random_state : int, optional
        Seed for reproducibility (default 42).

    Returns
    -------
    pd.DataFrame
        Sampled (or full) DataFrame with the index reset.
    """
    if fraction >= 1.0:
        return df.copy()
    return df.sample(frac=fraction, random_state=random_state).reset_index(drop=True)


def minimal_clean(df: pd.DataFrame, target_column: Optional[str]) -> pd.DataFrame:
    """
    Remove rows with a missing target and drop all-NA columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    target_column : str or None
        If provided and present in *df*, rows where *target_column* is NaN
        are dropped.

    Returns
    -------
    pd.DataFrame
        Cleaned copy of *df* with the index reset.
    """
    df = df.copy()
    if target_column is not None and target_column in df.columns:
        df = df.dropna(subset=[target_column])
    # drop columns that are entirely NA
    df = df.dropna(axis=1, how='all')
    return df.reset_index(drop=True)


def encode_and_impute(df: pd.DataFrame, protected_attribute: Optional[str] = None) -> pd.DataFrame:
    """
    One-hot-encode categorical columns and median-impute numeric NaNs.

    The protected attribute column (if provided) is preserved unchanged:
    it is detached before encoding, then reattached at the end to ensure it
    remains available for fairness metric computation in ``fitness.fairness_metrics``.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    protected_attribute : str or None, optional
        Column name of the protected attribute.  If *None*, no column is
        explicitly preserved (default None).

    Returns
    -------
    pd.DataFrame
        Processed DataFrame with:

        - object/category columns replaced by binary dummy columns
          (``pd.get_dummies(..., drop_first=True)``);
        - numeric NaNs filled with the column median (``SimpleImputer``);
        - protected attribute column (if any) reattached unchanged.
    """
    df = df.copy()
    # preserve protected attribute column if present
    protected = None
    if protected_attribute and protected_attribute in df.columns:
        protected = df[protected_attribute]
        df = df.drop(columns=[protected_attribute])

    # One-hot encode object / category columns
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # Impute numeric columns with median
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        imputer = SimpleImputer(strategy='median')
        df[num_cols] = imputer.fit_transform(df[num_cols])

    # defragment before reattaching protected attribute — get_dummies leaves a
    # fragmented frame and column assignment on it triggers PerformanceWarning
    new = df.copy()
    if protected is not None:
        new[protected_attribute] = protected.reset_index(drop=True)
    return new


def binarize_target(df: pd.DataFrame, target_column: str,
                    positive_values: Optional[Iterable[object]] = None,
                    threshold: Optional[float] = None) -> pd.DataFrame:
    """
    Map the target column to a binary {0, 1} label.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    target_column : str
        Column to binarise; must be present in *df*.
    positive_values : iterable, optional
        Values that map to 1 (exact membership test via ``pd.Series.isin``).
    threshold : numeric, optional
        Values >= *threshold* map to 1.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with *target_column* replaced by a {0, 1} integer column.

    Notes
    -----
    Fallback rules when neither *positive_values* nor *threshold* is given:

    - Numeric target with >2 unique values: median split (>= median → 1).
    - Exactly 2 unique values (any type): first encountered value → 0,
      second → 1.
    - Otherwise: returned as-is (caller is responsible for handling).
    """
    df = df.copy()
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not in dataframe")

    if positive_values is not None:
        df[target_column] = df[target_column].isin(positive_values).astype(int)
        return df

    if threshold is not None:
        df[target_column] = (
            pd.to_numeric(df[target_column], errors='coerce') >= threshold
        ).astype(int)
        return df

    # fallback for numeric target: median split
    if pd.api.types.is_numeric_dtype(df[target_column]) and df[target_column].nunique() > 2:
        med = df[target_column].median()
        df[target_column] = (df[target_column] >= med).astype(int)
        return df

    # if already binary numeric or categorical with two values, map to 0/1
    unique_vals = df[target_column].dropna().unique()
    if len(unique_vals) == 2:
        mapping = {unique_vals[0]: 0, unique_vals[1]: 1}
        df[target_column] = df[target_column].map(mapping).astype(int)
        return df
    new = df.copy()
    # otherwise leave as-is (calling code should handle)
    return new


def prepare_data_model(df: pd.DataFrame, target_column: str,
                       protected_attribute: Optional[str] = None, binarize: bool = True
                       ) -> pd.DataFrame:
    """
    Minimal end-to-end pipeline to produce a model-ready DataFrame.

    Role in Algorithm 1 (Step 2 – final preparation before classifier training):
        Called inside ``fitness.fitness`` after fairness techniques have been
        applied.  Ensures the dataset is clean, optionally binarised, and
        fully numeric before features and labels are separated for
        cross-validation.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset after fairness technique application (output of
        ``practices.apply_techniques`` chain).
    target_column : str
        Name of the label column.
    protected_attribute : str or None, optional
        Protected attribute column to preserve through encoding (default None).
    binarize : bool, optional
        If True (default), binarise the target column with ``binarize_target``.

    Returns
    -------
    pd.DataFrame
        Processed DataFrame containing numeric features, the target column,
        and the protected attribute column (if provided).  Index is reset.

    Notes
    -----
    The returned DataFrame still contains the protected attribute column.
    ``fitness.fitness`` drops it from X before training so that the classifier
    does not directly observe the protected attribute.
    """

    df = df.copy()
    df = minimal_clean(df, target_column=target_column)
    if binarize:
        df = binarize_target(df, target_column)
    df = encode_and_impute(df, protected_attribute=protected_attribute)

    # Consolidate fragmented blocks into a single contiguous block to avoid fragmentation warnings
    new = df.copy()

    return new


def prepare_adult(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise the raw Adult (Census Income) dataset for FATE experiments.

    Handles the following dataset-specific quirks:

    - Maps textual salary labels (``'>50K'`` / ``'<=50K'``) to ``{1, 0}``.
    - Collapses pre-encoded one-hot sex columns (``'sex_Female'`` /
      ``'sex_Male'``) into a single binary ``'sex'`` column (1 = Female,
      0 = Male).
    - Coerces the ``'race'`` column to numeric.

    Parameters
    ----------
    df : pd.DataFrame
        Raw Adult dataset loaded from ``datasets/adult.csv``.

    Returns
    -------
    pd.DataFrame
        Processed dataset ready to be passed to ``prepare_data_model``.
        Protected attributes: ``sex`` (binary), ``race`` (binary, 1=privileged).
        Target column: ``salary`` (binary 0/1, 1=>50 K income).
    """
    data = df.copy()
    # Target
    if 'salary' in data.columns:
        # If salary is textual like '>50K' / '<=50K', map to 1/0
        if data['salary'].dtype == object:
            data['salary'] = data['salary'].astype(str).str.strip()
            data['salary'] = data['salary'].map(
                {'>50K': 1, '<=50K': 0, '>50K.': 1, '<=50K.': 0}
            ).fillna(data['salary'])
        # try to coerce to numeric
        data['salary'] = pd.to_numeric(data['salary'], errors='coerce')

    # Sex columns might already be one-hot encoded as 'sex_Female' and 'sex_Male'
    if 'sex_Female' in data.columns and 'sex_Male' in data.columns:
        # create single 'sex' column: 1 if Female, 0 if Male (fallback to first non-null)
        data['sex'] = data['sex_Female'].fillna(0).astype(int)
    elif 'sex' in data.columns:
        # if sex is textual, map common labels
        if data['sex'].dtype == object:
            data['sex'] = data['sex'].str.lower().map(lambda x: 1 if 'female' in str(x) else 0)
        data['sex'] = pd.to_numeric(data['sex'], errors='coerce')

    # Race should be 0/1 already per your note; coerce to numeric
    if 'race' in data.columns:
        data['race'] = pd.to_numeric(data['race'], errors='coerce')
    new = data.copy()
    return new


def _map_german_sex_code(v: object) -> int:
    """
    Map one German Credit personal-status/sex code to the binary ``sex`` value.

    A92/A95 (female) -> 1; A91/A93/A94 (male) -> 0; other strings containing an
    ``f`` -> 1; everything else -> 0.

    Parameters
    ----------
    v : object
        Raw cell value.

    Returns
    -------
    int
        1 for female, 0 for male.
    """
    s = str(v)
    if 'A95' in s or 'A92' in s:
        return 1
    if 'A93' in s or 'A91' in s or 'A94' in s:
        return 0
    if 'f' in s.lower():
        return 1
    return 0


def prepare_german(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise the raw German Credit dataset for FATE experiments.

    Handles the following dataset-specific quirks:

    - Maps German-dataset sex codes (A92/A95 → Female=1, A91/A93/A94 →
      Male=0) into a binary ``'sex'`` column.
    - Coerces ``'Age'`` / ``'age'`` to numeric.
    - Remaps the two-class ``'Target'`` column: 2 → 1 (good credit),
      1 → 0 (bad credit).

    Parameters
    ----------
    df : pd.DataFrame
        Raw German Credit dataset loaded from ``datasets/german.csv``.

    Returns
    -------
    pd.DataFrame
        Processed dataset ready to be passed to ``prepare_data_model``.
        Protected attributes: ``sex`` (binary), ``age`` (continuous —
        binarised at the mean inside ``fitness.fairness_metrics``).
        Target column: ``Target`` (binary 0/1, 1=good credit).
    """
    data = df.copy()
    # Sex mapping (German dataset uses A91..A95 codes)
    if 'sex' in data.columns:
        data['sex'] = data['sex'].apply(_map_german_sex_code)

    # Age numeric
    if 'Age' in data.columns:
        data['Age'] = pd.to_numeric(data['Age'], errors='coerce')
    if 'age' in data.columns:
        data['age'] = pd.to_numeric(data['age'], errors='coerce')

    # Target mapping: 2 -> 1, 1 -> 0
    if 'Target' in data.columns:
        data['Target'] = pd.to_numeric(data['Target'], errors='coerce')
        data['Target'] = data['Target'].map(lambda x: 1 if x == 2 else 0 if x == 1 else x)
    new = data.copy()
    return new


def prepare_heart(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise the raw Heart Disease (Cleveland) dataset for FATE experiments.

    Handles the following dataset-specific quirks:

    - Ensures ``'sex'`` is numeric (0/1).
    - Ensures ``'age'`` is numeric (continuous — binarised at mean inside
      ``fitness.fairness_metrics``).
    - Binarises the multi-class ``'num'`` target: 0 → 0 (no disease),
      any positive value → 1 (disease present).

    Parameters
    ----------
    df : pd.DataFrame
        Raw Heart Disease dataset loaded from ``datasets/heart.csv``.

    Returns
    -------
    pd.DataFrame
        Processed dataset ready to be passed to ``prepare_data_model``.
        Protected attributes: ``sex`` (binary, 1=male), ``age`` (continuous).
        Target column: ``num`` (binary 0/1, 1=disease present).
    """
    data = df.copy()
    if 'sex' in data.columns:
        data['sex'] = pd.to_numeric(data['sex'], errors='coerce')
    if 'age' in data.columns:
        data['age'] = pd.to_numeric(data['age'], errors='coerce')

    if 'num' in data.columns:
        data['num'] = pd.to_numeric(data['num'], errors='coerce')
        data['num'] = data['num'].map(lambda x: 1 if pd.notna(x) and x > 0 else 0)
    new = data.copy()
    return new
