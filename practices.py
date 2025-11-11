# preprocessing.py
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import numpy as np

# Function to apply OneHot encoding and standard scaling
def apply_onehot_standard_transformation(data, protected_attribute):
    """
    Apply OneHot encoding to the protected attribute and standard scaling to numeric columns.
    
    Parameters:
    data (pd.DataFrame): The dataset.
    protected_attribute (str): The categorical column to apply OneHot encoding.
    
    Returns:
    pd.DataFrame: The transformed dataset with OneHot encoding and standard scaling applied.
    """
    # Initialize the OneHotEncoder
    onehot_encoder = OneHotEncoder(sparse=False)  # Set sparse=False to return a dense matrix
    # Perform OneHot encoding on the specified column
    encoded_columns = onehot_encoder.fit_transform(data[[protected_attribute]])
    
    # Get the unique categories (number of new columns after encoding)
    category_names = onehot_encoder.categories_[0]  # Categories from the protected attribute
    encoded_df = pd.DataFrame(encoded_columns, columns=[f"{protected_attribute}_{category}" for category in category_names])
    
    # Reset the index to ensure proper alignment when concatenating
    encoded_df.index = data.index
    
    # Concatenate the encoded columns to the original data and drop the original column
    data = pd.concat([data, encoded_df], axis=1).drop(columns=[protected_attribute])

    # Select numeric columns for standard scaling
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    # Apply StandardScaler to the numeric columns
    data[numeric_columns] = StandardScaler().fit_transform(data[numeric_columns])

    # Print the transformation results for verification
    print(f"Transformation: OneHot Encoding and Standard Scaling for '{protected_attribute}' completed.")
    print(data.head())  # Display the first few rows of the transformed dataset
    
    return data

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
    print(balanced_dataset.head())
    
    return balanced_dataset

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
    print(balanced_df.head())
    return balanced_df

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
    print(balanced_df.head())
    return balanced_df

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
    print(clustered_data.head())
    return clustered_data

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
    print(weighted_data.head())
    return weighted_data

# Function for creating a matched sample
def apply_matching(data, protected_attribute):
    """
    Shuffle the dataset to create a matched sample.
    """
    # Shuffle the data
    matched_data = data.sample(frac=1, random_state=42)
    print(f"Transformation: Matching for {protected_attribute}")
    print(matched_data.head())
    return matched_data

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
    print(data.head())
    return data

# Function to apply a specified preprocessing technique
def apply_techniques(data, technique, protected_attribute):
    """
    Apply the specified preprocessing technique to the dataset.
    """
    # Apply the appropriate preprocessing technique based on the 'technique' parameter
    if technique == 'onehot_standard':
        return apply_onehot_standard_transformation(data, protected_attribute)
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
