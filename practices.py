# preprocessing.py
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import numpy as np

# Function to apply OneHot encoding and standard scaling
def apply_standard_transformation(data, protected_attribute):
    """
    Apply standard scaling to numeric columns while preserving the protected attribute.

    Parameters:
    data (pd.DataFrame): The dataset.
    protected_attribute (str): The column to preserve (will NOT be one-hot encoded).

    Returns:
    pd.DataFrame: The transformed dataset with numeric columns scaled.
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

    # Reattach protected attribute unchanged
    if protected is not None:
        df[protected_attribute] = protected.values

    print(f"Transformation: Standard Scaling for numeric columns completed (protected '{protected_attribute}' preserved).")
    return df.copy()

# Function for stratified sampling to balance the dataset
def apply_stratified_sampling(data, protected_attribute):
    """
    Perform stratified sampling to balance the dataset with respect to the protected attribute.
    
    Parameters:
    data (pd.DataFrame): The dataset.
    protected_attribute (str): The attribute to balance by.
    
    Returns:
    pd.DataFrame: A balanced dataset.
    """
    if protected_attribute not in data.columns:
        print(f"The protected attribute '{protected_attribute}' is not present in the dataset.")
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
    
    print(f"Transformation: Stratified Sampling for '{protected_attribute}' completed.")

    
    return balanced_dataset.copy()

# Function for oversampling to ensure equal representation
def apply_oversampling(data, protected_attribute):
    """
    Apply oversampling to ensure that each category of the protected attribute is equally represented.
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
    print(f"Transformation: Oversampling for Fairness for {protected_attribute}")
    return balanced_df.copy()

# Function for undersampling to ensure equal representation
def apply_undersampling(data, protected_attribute):
    """
    Apply undersampling to ensure that each category of the protected attribute is equally represented.
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
    print(f"Transformation: Undersampling for Fairness for {protected_attribute}")
    
    return balanced_df.copy()

# Function for applying KMeans clustering
def apply_clustering(data, protected_attribute, n_clusters=2):
    """
    Apply KMeans clustering to the numeric columns and add cluster labels to the dataset.
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
    print(f"Transformation: Clustering for {protected_attribute}")
    
    return clustered_data.copy()

# Function for applying Inverse Probability Weighting (IPW)
def apply_ipw(data, protected_attribute):
    """
    Apply Inverse Probability Weighting to adjust the dataset for fairness.
    """
    # Calculate the probabilities for each value of the protected attribute
    probabilities = data[protected_attribute].value_counts(normalize=True)
    # Map each value to its inverse probability
    weights = data[protected_attribute].map(lambda x: 1 / probabilities[x])
    # Create a copy of the data and add the weights
    weighted_data = data.copy()
    weighted_data['Weight'] = weights
    print(f"Transformation: Inverse Probability Weighting for {protected_attribute}")
    
    return weighted_data

# Function for creating a matched sample
def apply_matching(data, protected_attribute):
    """
    Shuffle the dataset to create a matched sample.
    """
    # Shuffle the data
    matched_data = data.sample(frac=1, random_state=42)
    print(f"Transformation: Matching for {protected_attribute}")
    
    return matched_data.copy()

# Function for applying Min-Max scaling
def apply_min_max_scaling(data, protected_attribute):
    """
    Apply Min-Max scaling to numeric columns to bring them to a specific range.
    """
    # Select numeric columns for scaling
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    # Apply MinMaxScaler to the numeric columns
    data[numeric_columns] = MinMaxScaler().fit_transform(data[numeric_columns])
    print(f"Transformation: Min-Max Scaling for {protected_attribute}")
    
    return data.copy()

# Function to apply a specified preprocessing technique
def apply_techniques(data, technique, protected_attribute):
    """
    Apply the specified preprocessing technique to the dataset.
    """
    # Apply the appropriate preprocessing technique based on the 'technique' parameter
    if technique == 'standard':
        return apply_standard_transformation(data, protected_attribute)
    elif technique == 'stratified_sampling':
        return apply_stratified_sampling(data, protected_attribute)
    elif technique == 'oversampling':
        return apply_oversampling(data, protected_attribute)
    elif technique == 'undersampling':
        return apply_undersampling(data, protected_attribute)
    elif technique == 'clustering':
        return apply_clustering(data, protected_attribute)
    elif technique == 'ipw':
        return apply_ipw(data, protected_attribute)
    elif technique == 'matching':
        return apply_matching(data, protected_attribute)
    elif technique == 'min_max_scaling':
        return apply_min_max_scaling(data, protected_attribute)
    else:
        return data
