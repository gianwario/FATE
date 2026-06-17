"""
Standalone utility for training and persisting an initial baseline classifier.

This module trains a Logistic Regression model on a user-specified dataset
using only the protected attribute (``Sex_Code_Text``) as a feature and saves
the fitted model to disk via ``joblib``.

Note: This script is a **standalone utility** for early-stage model exploration
and is **not** part of the main FATE pipeline.  The GA in
``genetic_algorithm.py`` instantiates classifiers internally within
``fitness.py`` and does not load pre-saved model files.
"""
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib

# Main function


def create_and_save_model():
    """
    Interactively train and save a Logistic Regression baseline model.

    Prompts the user for a dataset path and an output directory.  Trains a
    Logistic Regression on the binary ``DecileScore`` target (threshold > 5)
    using ``Sex_Code_Text`` as the sole feature, prints test accuracy, and
    saves the fitted model to ``<output_dir>/initial_model.pkl``.

    Notes
    -----
    The column names ``Sex_Code_Text`` and ``DecileScore`` are hard-coded,
    so this function applies only to COMPAS-style recidivism datasets.

    This is a standalone prototype for the COMPAS dataset and is not part
    of the main FATE pipeline.  The paper's experiments use Adult, German
    Credit, and Heart Disease; classifiers are instantiated inside
    ``fitness._build_classifier`` and are not loaded from disk.
    """
    # Path to the dataset
    file_path = input("Enter the path to the dataset (e.g., '/path/to/dataset.csv'): ").strip()

    try:
        # Loading the dataset
        dataset = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return

    # Specify the sensitive columns
    sensitive_cols = ["Sex_Code_Text"]

    # Encode the sensitive columns
    label_encoder = LabelEncoder()
    for col in sensitive_cols:
        dataset[col] = label_encoder.fit_transform(dataset[col])

    # Define feature and target columns
    feature_cols = sensitive_cols
    target_col = "DecileScore"

    # Convert the target variable to binary labels
    dataset[target_col] = (dataset[target_col] > 5).astype(int)

    # Split the dataset into train and test sets
    X = dataset[feature_cols]
    y = dataset[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train the model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Predict and calculate accuracy
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")

    # Save the model
    output_dir = input("Enter the directory to save the model (e.g., '/path/to/save'): ").strip()
    os.makedirs(output_dir, exist_ok=True)
    initial_model_path = os.path.join(output_dir, 'initial_model.pkl')
    joblib.dump(model, initial_model_path)
    print(f"Initial model saved at: {initial_model_path}")


# Execute the main function
if __name__ == "__main__":
    create_and_save_model()
