# fitness.py
import pandas as pd
import numpy as np
from fairlearn.postprocessing import ThresholdOptimizer
from sklearn.metrics import accuracy_score, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from preprocessing import prepare_data_model

from practices import apply_techniques


def fairness_metrics(test_data, test_indices, protected_attribute, predictions, target_column):
    """
    Calculate fairness metrics based on predictions and actual data.

    Args:
        test_data (pd.DataFrame): The test dataset.
        test_indices (pd.Index): Indices of the test dataset.
        protected_attribute (str): Name of the protected attribute column.
        predictions (np.array): Predictions made by the model.
        target_column (str): Name of the target column.

    Returns:
        dict: A dictionary containing disparity, statistical parity, and equal opportunity metrics.
    """
    # Create a series of predictions with the test indices
    predictions_series = pd.Series(predictions, index=test_indices)

    # Identify indices of positive target outcomes
    positive_indices = test_indices[test_data.loc[test_indices, target_column] == 1]

    # Align predictions with positive indices
    predictions_aligned = predictions_series.loc[positive_indices]

    # Calculate actual and predicted positive rates for the protected attribute
    actual_positive = test_data.loc[positive_indices, protected_attribute].value_counts(normalize=True)
    predicted_positive = predictions_aligned.value_counts(normalize=True)

    # Compute disparity between actual and predicted positive rates
    disparity = np.abs(actual_positive - predicted_positive).sum()

    # Calculate statistical parity
    predicted_positive_rate = predictions_aligned.value_counts(normalize=True)
    statistical_parity = np.abs(predicted_positive_rate - actual_positive).sum()

    # Compute equal opportunity by comparing true positives
    true_positives = test_data[(test_data[target_column] == 1) & (predictions_series == 1)][
        protected_attribute].value_counts(normalize=True)
    opportunity_difference = np.abs(true_positives - actual_positive).sum()

    return {
        'disparity': disparity,
        'statistical_parity': statistical_parity,
        'equal_opportunity': opportunity_difference
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

    # Prepare data for modeling
    #data = prepare_data_model(data, target_column)
    y = data[target_column]
    X = data.drop(columns=[target_column])
    X = pd.get_dummies(X, drop_first=True)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Choose classifier based on model argument
    if model == 'logistic_regression':
        classifier = LogisticRegression(max_iter=1000, solver='liblinear')
    elif model == 'random_forest':
        classifier = RandomForestClassifier()
    elif model == 'svm':
        classifier = SVC()
    elif model == 'knn':
        classifier = KNeighborsClassifier()
    elif model == 'gradient_boosting':
        classifier = GradientBoostingClassifier()
    else:
        raise ValueError(f"Unknown model identifier: {model}")
    
    # Train the model and make predictions
    try:
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
    except Exception:
        # if model training fails, return a large fitness (bad)
        return float('inf')

    # Calculate performance metrics
    # Calculate fallback performance metric (accuracy) for use if PR-AUC can't be computed
    accuracy = accuracy_score(y_test, y_pred)

    # Compute PR-AUC (average precision) as performance score
    try:
        # Prefer predicted probabilities for positive class
        if hasattr(classifier, "predict_proba"):
            y_scores = classifier.predict_proba(X_test)
            # binary: take column 1
            if y_scores.ndim == 2 and y_scores.shape[1] == 2:
                y_score_pos = y_scores[:, 1]
            else:
                y_score_pos = y_scores
        elif hasattr(classifier, "decision_function"):
            y_score_pos = classifier.decision_function(X_test)
        else:
            # fallback to predicted labels
            y_score_pos = y_pred

        # If binary classification compute average precision directly
        if len(np.unique(y_test)) == 2:
            performance_score = average_precision_score(y_test, y_score_pos)
        else:
            # multiclass: try using probability matrix if available
            if hasattr(classifier, "predict_proba") and y_scores.ndim == 2:
                performance_score = average_precision_score(y_test, y_scores, average='weighted')
            else:
                # fallback to accuracy if multiclass probabilities not available
                performance_score = accuracy
    except Exception:
        # on error fallback to accuracy
        performance_score = accuracy

    # Compute fairness metrics
    fairness = fairness_metrics(data, X_test.index, protected_attribute, y_pred, target_column)

    # Sum of fairness metrics
    fairness_score = sum(fairness.values())

    # Compute fitness value as alpha * PS - beta * FS
    fitness_value = perf_weight * performance_score - fair_weight * fairness_score
    return fitness_value, fairness_score, performance_score


