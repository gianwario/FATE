import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer


def load_dataset(path):
    """Load a CSV dataset into a DataFrame."""
    return pd.read_csv(path)


def sample_dataset(df, fraction=1.0, random_state=42):
    """Return a random sample (fraction) of the dataframe."""
    if fraction >= 1.0:
        return df.copy()
    return df.sample(frac=fraction, random_state=random_state).reset_index(drop=True)


def minimal_clean(df, target_column):
    """Minimal cleaning:
    - Drop rows with missing target (if provided)
    - Remove columns that are all NA
    - Return a copy
    """
    df = df.copy()
    if target_column is not None and target_column in df.columns:
        df = df.dropna(subset=[target_column])
    # drop columns that are entirely NA
    df = df.dropna(axis=1, how='all')
    return df.reset_index(drop=True)


def encode_and_impute(df, protected_attribute=None):
    """Encode categorical columns with one-hot (pd.get_dummies) and impute numeric NaNs with median.
    Keep the protected_attribute column unchanged (it will be included as-is).
    Returns processed_df.
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

    # reattach protected attribute at the end
    if protected is not None:
        df[protected_attribute] = protected.reset_index(drop=True)
    new = df.copy()
    return new


def binarize_target(df, target_column, positive_values=None, threshold=None):
    """Binarize target column to 0/1.

    Options:
    - positive_values: iterable of values to map to 1 (exact match)
    - threshold: numeric threshold; values >= threshold -> 1

    If neither provided and target is numeric with >2 uniques, median threshold is used.
    """
    df = df.copy()
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not in dataframe")

    vals = df[target_column]
    if positive_values is not None:
        df[target_column] = df[target_column].isin(positive_values).astype(int)
        return df

    if threshold is not None:
        df[target_column] = (pd.to_numeric(df[target_column], errors='coerce') >= threshold).astype(int)
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


def prepare_data_model(df, target_column, protected_attribute=None, binarize=True):
    """Minimal pipeline to prepare data for model training/evaluation.

    Steps:
    - minimal cleaning (drop rows with missing target)
    - optional binarization of target
    - encode categorical features and impute numeric NaNs
    - ensure protected attribute is present and returned in dataframe

    Returns: processed_df (features + target + protected attribute)
    """

    df = df.copy()
    df = minimal_clean(df, target_column=target_column)
    if binarize:
        df = binarize_target(df, target_column)
    df = encode_and_impute(df, protected_attribute=protected_attribute)

    # Consolidate fragmented blocks into a single contiguous block to avoid fragmentation warnings
    new = df.copy()

    return new


def prepare_adult(df):
    """Prepare the adult dataset:
    - Ensure target column 'salary' is binary 0/1
    - Collapse sex one-hot columns ('sex_Female','sex_Male') into single 'sex' column with values 0/1 (Female=1)
    - Ensure race is numeric (0/1)
    Returns processed DataFrame and canonical target/protected names.
    """
    data = df.copy()
    # Target
    if 'salary' in data.columns:
        # If salary is textual like '>50K' / '<=50K', map to 1/0
        if data['salary'].dtype == object:
            data['salary'] = data['salary'].astype(str).str.strip()
            data['salary'] = data['salary'].map({'>50K': 1, '<=50K': 0, '>50K.': 1, '<=50K.': 0}).fillna(data['salary'])
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


def prepare_german(df):
    """Prepare the german dataset:
    - Map sex codes (A95 -> female, A93 -> male) into a 'sex' column (0/1)
    - Ensure Age is numeric
    - Map Target (1/2) -> binary 0/1 (we map Target==2 to 1)
    Returns processed DataFrame.
    """
    data = df.copy()
    # Sex mapping
    if 'sex' in data.columns:
        # German dataset uses A91..A95 codes; map female and male
        def map_sex_code(v):
            try:
                s = str(v)
                if 'A95' in s or 'A92' in s:
                    return 1
                if 'A93' in s or 'A91' in s or 'A94' in s:
                    return 0
                if 'f' in s.lower():
                    return 1
                return 0
            except Exception:
                return 0

        data['sex'] = data['sex'].apply(map_sex_code)

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


def prepare_heart(df):
    """Prepare the heart dataset:
    - Ensure sex is numeric (0/1)
    - Ensure age numeric
    - Binarize 'num' target: 0 -> 0 (no disease), >0 -> 1 (disease)
    Returns processed DataFrame.
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

