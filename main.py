import pandas as pd
from genetic_algorithm import genetic_algorithm
from preprocessing import sample_dataset, prepare_data_dataset
import joblib
import os


def get_user_input():
    """
    Get user input

    Returns:
        tuple: (dataset, protected_attribute, target_column, output_dir, sample_fraction, model_identifier)
    """
    # Get dataset path and load the dataset
    dataset_path = input("Enter the dataset path (e.g., 'Dataset/dataset.csv'): ").strip()
    try:
        dataset = pd.read_csv(dataset_path)
        print("Loaded dataset with columns:", dataset.columns.tolist())
    except FileNotFoundError:
        print(f"The file {dataset_path} was not found. Please ensure the path is correct.")
        return None, None, None, None, None, None

    # Get protected attribute and target variable
    protected_attribute = input("Enter the name of the protected attribute (e.g., 'Sex_Code_Text'): ").strip()
    if protected_attribute not in dataset.columns:
        print(f"The protected attribute '{protected_attribute}' is not present in the dataset.")
        return None, None, None, None, None, None

    target_column = input("Enter the name of the target variable (e.g., 'DecileScore'): ").strip()
    if target_column not in dataset.columns:
        print(f"The target variable '{target_column}' is not present in the dataset.")
        return None, None, None, None, None, None

    # Get sample fraction and output directory for saving the optimized dataset
    sample_fraction = float(input("Enter the fraction of the dataset to use (e.g., 0.1 for 10%): ").strip())
    output_dir = input("Enter the output directory to save the optimized dataset (e.g., 'Output/'): ").strip()

    # Choose a model identifier to evaluate during GA (no model file loading)
    model_identifier = input("Enter model identifier to evaluate (default 'random_forest'): ").strip() or 'random_forest'

    return dataset, protected_attribute, target_column, output_dir, sample_fraction, model_identifier


# Main flow: only dataset optimization
dataset, protected_attribute, target_column, output_dir, sample_fraction, model_identifier = get_user_input()
if dataset is None:
    exit()

# Sample the dataset
dataset_sample = sample_dataset(dataset, fraction=sample_fraction)

# Prepare the data for optimization
dataset_sample = prepare_data_dataset(dataset_sample, target_column)

# Run the genetic algorithm (model is a model identifier string used by fitness)
best_solution = genetic_algorithm(dataset_sample, protected_attribute, target_column, model_identifier,
                                  generations=3, population_size=10)

# Print and save the best solution found by the genetic algorithm
print(f"Best solution: Techniques={best_solution[0]}, Model={best_solution[1]}, Fitness={best_solution[2]}")
os.makedirs(output_dir, exist_ok=True)
best_dataset_path = os.path.join(output_dir, 'best_optimized_dataset.csv')
dataset_sample.to_csv(best_dataset_path, index=False)
print(f"Optimized dataset saved at: {best_dataset_path}")
