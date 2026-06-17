"""
Fairness-aware data preparation practices (the FATE search space).

This module defines the complete set of fairness-aware preprocessing
techniques that form the *genes* of the FATE genetic algorithm (Algorithm 1).
Each public function corresponds to one technique that can appear in an
individual's chromosome.  During fitness evaluation (see ``fitness.fitness``),
``apply_techniques`` is called for each technique token in the individual,
applying the functions in sequence to the dataset before classifier training.

Techniques available (search space T):
    standard            – StandardScaler on numeric features.
    stratified_sampling – under-sample each protected-attribute group to the
                          size of the smallest group.
    oversampling        – over-sample minority groups to the majority group size.
    undersampling       – under-sample majority groups to the minority group size.
    clustering          – append KMeans cluster-membership as a synthetic feature.
    ipw                 – append Inverse Probability Weights as a feature column.
    matching            – random shuffle (simplified proxy for matching).
    min_max_scaling     – MinMaxScaler on numeric features.

Role in Algorithm 1:
    The technique name strings defined here constitute the candidate pool
    (``techniques`` list in ``genetic_algorithm.genetic_algorithm``) from which
    individuals are drawn during *population initialisation* (Step 1) and
    repaired after *crossover* / *mutation* (Steps 4–5).  The functions
    themselves are invoked inside ``fitness.fitness`` (Step 2).
"""
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.utils import resample
from sklearn.cluster import KMeans
import numpy as np

# Function to apply OneHot encoding and standard scaling


def apply_standard_transformation(data, protected_attribute):
    """
    Apply StandardScaler to numeric features while preserving the protected attribute.

    Role in Algorithm 1 (Step 2 – applied inside fitness evaluation):
        Normalises numeric columns to zero mean / unit variance before model
        training.  The protected attribute is explicitly excluded from scaling
        to preserve its semantic meaning for fairness metric computation.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataset containing features and the protected attribute.
    protected_attribute : str
        Name of the column designating the protected group.  This column is
        detached before scaling and reattached unchanged at the last position.

    Returns
    -------
    pd.DataFrame
        Copy of *data* with numeric columns standardised; the protected
        attribute column is unchanged.
    """
    # Work on a copy
    df = data.copy()

    # Preserve the protected attribute if present
    if protected_attribute in df.columns:
        protected = df[protected_attribute]
        df = df.drop(columns=[protected_attribute])
    else:
        protected = None

    # Select numeric columns for standard scaling
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_columns:
        df[numeric_columns] = StandardScaler().fit_transform(df[numeric_columns])

    # Defragment before reattaching — StandardScaler leaves a fragmented frame
    new = df.copy()
    if protected is not None:
        new[protected_attribute] = protected.values
    return new


# Function for stratified sampling to balance the dataset
def apply_stratified_sampling(data, protected_attribute):
    """
    Balance the dataset by under-sampling all protected-attribute groups to the smallest
    group size.

    Role in Algorithm 1 (Step 2 – applied inside fitness evaluation):
        A pre-processing fairness intervention.  For each unique value of
        *protected_attribute*, samples exactly ``min_count`` rows (the size of
        the smallest group) without replacement, producing equal representation.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataset.
    protected_attribute : str
        Column whose value distribution is to be balanced.

    Returns
    -------
    pd.DataFrame
        Balanced dataset where every protected-attribute group has the same
        number of rows (equal to the size of the original smallest group),
        shuffled to remove ordering artefacts.  Returns an empty DataFrame if
        *protected_attribute* is not present in *data*.
    """
    if protected_attribute not in data.columns:
        return pd.DataFrame()

    # Get the value counts of the protected attribute
    value_counts = data[protected_attribute].value_counts()

    # Find the minimum count among the value counts (minority class size)
    min_count = value_counts.min()

    # Create an empty DataFrame to store the balanced data
    balanced_dataset = pd.DataFrame()

    # Perform undersampling to get balanced classes
    for value in value_counts.index:
        # Subset the data for the current class
        subset = data[data[protected_attribute] == value]
        # Randomly sample `min_count` rows from this class
        balanced_subset = subset.sample(n=min_count, random_state=42)
        # Concatenate to the balanced dataset
        balanced_dataset = pd.concat([balanced_dataset, balanced_subset], ignore_index=True)

    # Shuffle the balanced dataset to mix the classes
    balanced_dataset = balanced_dataset.sample(frac=1, random_state=42).reset_index(drop=True)

    return balanced_dataset.copy()


# Function for oversampling to ensure equal representation
def apply_oversampling(data, protected_attribute):
    """
    Oversample minority groups of the protected attribute to match the majority group size.

    Role in Algorithm 1 (Step 2 – applied inside fitness evaluation):
        A pre-processing fairness intervention that increases representation of
        under-represented groups via sampling with replacement up to
        ``max_size`` rows per group.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataset.
    protected_attribute : str
        Column whose minority groups are to be oversampled.

    Returns
    -------
    pd.DataFrame
        Dataset with all protected-attribute groups having ``max_size`` rows,
        shuffled to avoid systematic ordering artefacts.
    """
    max_size = data[protected_attribute].value_counts().max()
    balanced_df = pd.DataFrame()
    for value in data[protected_attribute].unique():
        # Create a subset for each unique value of the protected attribute
        subset = data[data[protected_attribute] == value]
        # Resample the subset to the maximum size
        resampled_subset = resample(subset, replace=True, n_samples=max_size, random_state=123)
        # Concatenate the resampled subset to the balanced DataFrame
        balanced_df = pd.concat([balanced_df, resampled_subset])
    # Shuffle the balanced DataFrame
    balanced_df = balanced_df.sample(frac=1, random_state=123).reset_index(drop=True)
    return balanced_df.copy()


# Function for undersampling to ensure equal representation
def apply_undersampling(data, protected_attribute):
    """
    Undersample majority groups of the protected attribute to match the minority group size.

    Role in Algorithm 1 (Step 2 – applied inside fitness evaluation):
        Symmetrical to ``apply_oversampling``: reduces all groups to the size
        of the smallest one by sampling *without* replacement.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataset.
    protected_attribute : str
        Column whose majority groups are to be reduced.

    Returns
    -------
    pd.DataFrame
        Dataset with all protected-attribute groups having ``min_size`` rows,
        shuffled to remove ordering bias.
    """
    min_size = data[protected_attribute].value_counts().min()
    balanced_df = pd.DataFrame()
    for value in data[protected_attribute].unique():
        # Create a subset for each unique value of the protected attribute
        subset = data[data[protected_attribute] == value]
        # Resample the subset to the minimum size
        resampled_subset = resample(subset, replace=False, n_samples=min_size, random_state=123)
        # Concatenate the resampled subset to the balanced DataFrame
        balanced_df = pd.concat([balanced_df, resampled_subset])
    # Shuffle the balanced DataFrame
    balanced_df = balanced_df.sample(frac=1, random_state=123).reset_index(drop=True)
    return balanced_df.copy()


# Function for applying KMeans clustering
def apply_clustering(data, protected_attribute, n_clusters=2):
    """
    Append KMeans cluster membership as a new feature column named ``Cluster``.

    Role in Algorithm 1 (Step 2 – applied inside fitness evaluation):
        Injects unsupervised group structure into the feature space.  The
        cluster label may serve as a proxy for latent group membership
        correlated with the protected attribute, providing the downstream
        classifier with additional signal to disentangle fairness-relevant
        structure.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataset.
    protected_attribute : str
        Recorded for logging; not used in the clustering computation itself.
    n_clusters : int, optional
        Number of KMeans clusters (default 2, matching binary protected attr).

    Returns
    -------
    pd.DataFrame
        Copy of *data* with an additional integer column ``Cluster`` appended.

    """
    # Select numeric columns for clustering
    numerical_columns = data.select_dtypes(include=[np.number]).columns
    X = data[numerical_columns]
    # Initialize KMeans with the specified number of clusters
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    # Fit KMeans and get the cluster labels
    cluster_labels = kmeans.fit_predict(X)
    # Create a copy of the data and add the cluster labels
    clustered_data = data.copy()
    clustered_data['Cluster'] = cluster_labels
    return clustered_data.copy()


# Function for applying Inverse Probability Weighting (IPW)
def apply_ipw(data, protected_attribute):
    """
    Weight each instance by the inverse of its protected-attribute group probability.

    Role in Algorithm 1 (Step 2 – applied inside fitness evaluation):
        Adds a ``Weight`` column encoding inverse probability weights (IPW).
        In a perfectly fair dataset the protected attribute is independent of
        the outcome; IPW re-weights observations so that each group contributes
        proportionally to its inverse frequency, reducing the influence of
        over-represented groups.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataset.
    protected_attribute : str
        Column used to compute empirical group probabilities.

    Returns
    -------
    pd.DataFrame
        Copy of *data* with an additional float column ``Weight``.
     """
    # Calculate the probabilities for each value of the protected attribute
    probabilities = data[protected_attribute].value_counts(normalize=True)
    # Map each value to its inverse probability
    weights = data[protected_attribute].map(lambda x: 1 / probabilities[x])
    # Create a copy of the data and add the weights
    weighted_data = data.copy()
    weighted_data['Weight'] = weights
    return weighted_data


# Function for creating a matched sample
def apply_matching(data, protected_attribute):
    """
    Apply a random shuffle as a simplified matching step.

    Role in Algorithm 1 (Step 2 – applied inside fitness evaluation):
        Intended as a proxy for propensity-score matching: randomly permutes
        the dataset to remove ordering biases introduced by earlier steps
        (e.g., group-level concatenation in over/under-sampling).

    Parameters
    ----------
    data : pd.DataFrame
        Input dataset.
    protected_attribute : str
        Recorded for logging; not used in the shuffle.

    Returns
    -------
    pd.DataFrame
        Randomly shuffled copy of *data* (random_state=42 for reproducibility).
    """
    # Shuffle the data
    matched_data = data.sample(frac=1, random_state=42)
    return matched_data.copy()


# Function for applying Min-Max scaling
def apply_min_max_scaling(data, protected_attribute):
    """
    Scale all numeric columns to the [0, 1] range using Min-Max normalisation.

    Role in Algorithm 1 (Step 2 – applied inside fitness evaluation):
        An alternative to standard scaling that maps every numeric feature to
        the unit interval, which can benefit distance-sensitive models (e.g.,
        SVC) and avoids the negative values produced by StandardScaler.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataset (numeric columns are scaled in-place; a copy is returned).
    protected_attribute : str
        Recorded for logging.

    Returns
    -------
    pd.DataFrame
        Copy of *data* with all numeric columns in [0, 1].
    """
    df = data.copy()
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_columns:
        df[numeric_columns] = MinMaxScaler().fit_transform(df[numeric_columns])
    return df.copy()


# Dispatch table mapping technique identifiers to their implementing functions.
# Adding a new technique to the GA search space requires only a new entry here.
_TECHNIQUE_DISPATCH = {
    'standard': apply_standard_transformation,
    'stratified_sampling': apply_stratified_sampling,
    'oversampling': apply_oversampling,
    'undersampling': apply_undersampling,
    'clustering': apply_clustering,
    'ipw': apply_ipw,
    'matching': apply_matching,
    'min_max_scaling': apply_min_max_scaling,
}


# Function to apply a specified preprocessing technique
def apply_techniques(data, technique, protected_attribute):
    """
    Dispatch a technique identifier to the corresponding preprocessing function.

    Role in Algorithm 1 (Step 2 – fitness evaluation, technique application):
        This is the single entry point called by ``fitness.fitness`` for each
        technique token in an individual's chromosome.  It maps the string
        identifier used inside GA individuals to the concrete transformation
        function via ``_TECHNIQUE_DISPATCH``.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset passed through the technique chain; previous techniques in the
        chromosome may already have modified it.
    technique : str
        One of: ``'standard'``, ``'stratified_sampling'``, ``'oversampling'``,
        ``'undersampling'``, ``'clustering'``, ``'ipw'``, ``'matching'``,
        ``'min_max_scaling'``.
    protected_attribute : str
        Forwarded unchanged to the selected technique function.

    Returns
    -------
    pd.DataFrame
        Transformed dataset.  Returns *data* unchanged if *technique* is not
        recognised (graceful no-op, matching the error-skip logic in
        ``fitness.fitness``).
    """
    handler = _TECHNIQUE_DISPATCH.get(technique)
    if handler is None:
        return data
    return handler(data, protected_attribute)
