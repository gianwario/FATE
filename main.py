import pandas as pd
from genetic_algorithm import genetic_algorithm
from preprocessing import sample_dataset, prepare_data_dataset
import os
import time

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


# Main flow
#dataset, protected_attribute, target_column, output_dir, sample_fraction, model_identifier = get_user_input()
dataset_path = "datasets/processed-adult.csv"
dataset = pd.read_csv(dataset_path)
protected_attribute = "race"
target_column = "salary"

if dataset is None:
    exit()

# Sample the dataset
dataset_sample = sample_dataset(dataset, fraction=0.5)

# Prepare the data for optimization
dataset_sample = prepare_data_dataset(dataset_sample, target_column)

# Run the genetic algorithm (model is a model identifier string used by fitness)
best_solution = genetic_algorithm(dataset_sample, protected_attribute, target_column, "random_forest",
                                  generations=5, population_size=10, alpha=0.5, beta=0.5)

# Print and save the best solution found by the genetic algorithm
techniques = best_solution[0] if best_solution and len(best_solution) > 0 else None
model_used = best_solution[1] if best_solution and len(best_solution) > 1 else "random_forest"
fitness_value = best_solution[2] if best_solution and len(best_solution) > 2 else None
fairness_score = best_solution[3] if best_solution and len(best_solution) > 3 else None
performance_score = best_solution[4] if best_solution and len(best_solution) > 4 else None

print(f"Best solution: Techniques={techniques}, Model={model_used}, Fitness={fitness_value}, Fairness={fairness_score}, Performance={performance_score}")
os.makedirs("output", exist_ok=True)
best_dataset_path = os.path.join("output", 'best_optimized_dataset.csv')
dataset_sample.to_csv(best_dataset_path, index=False)
print(f"Optimized dataset saved at: {best_dataset_path}")

# Append this single-run result to the experiments CSV (create header if needed)
summary_path = os.path.join("output", "experiments_results.csv")
os.makedirs(os.path.dirname(summary_path), exist_ok=True)
if not os.path.exists(summary_path):
    pd.DataFrame(columns=[
        "dataset", "model_identifier", "population_size", "generations",
        "crossover_rate", "mutation_rate", "techniques", "model_used",
        "fitness", "fairness_score", "performance_score", "elapsed_seconds", "error"
    ]).to_csv(summary_path, index=False)

try:
    row = {
        "dataset": dataset_path,
        "model_identifier": model_used,
        "population_size": 10,
        "generations": 5,
        "crossover_rate": None,
        "mutation_rate": None,
        "techniques": str(techniques),
        "model_used": str(model_used),
        "fitness": fitness_value,
        "fairness_score": fairness_score,
        "performance_score": performance_score,
        "elapsed_seconds": None,
        "error": None
    }
    pd.DataFrame([row]).to_csv(summary_path, mode='a', header=False, index=False)
except Exception as e:
    print(f"Failed to append single-run result to {summary_path}: {e}")


"""
if __name__ == "__main__":
    
    Automated main:
      - iterate over a list of datasets (path + protected attribute + target)
      - for each dataset, iterate over models
      - sweep population_size and generations over [25,50,100,250,500]
      - sweep crossover_rate and mutation_rate over [0.25,0.5,0.75,1.0]
      - log results to CSV in output_dir
    Edit the `datasets` and `models` lists below to match your project.


    # Configuration: edit these entries to match your datasets and columns
    datasets = [
        {
            "path": "datasets/adult_processed.csv",             
            "protected_attribute": "sex",   
            "target_column": "Probability",           
            "output_dir": "output/adult_sex_results"     
        },
        {
            "path": "datasets/adult_processed.csv",             
            "protected_attribute": "race",   
            "target_column": "Probability",           
            "output_dir": "output/adult_race_results"     
        },
        {
            "path": "datasets/german_processed.csv",             
            "protected_attribute": "sex",   
            "target_column": "Probability",           
            "output_dir": "output/german_sex_results"     
        },
        {
            "path": "datasets/german_processed.csv",             
            "protected_attribute": "age",   
            "target_column": "Probability",           
            "output_dir": "output/german_age_results"     
        },
        {
            "path": "datasets/heart_processed.csv",             
            "protected_attribute": "sex",   
            "target_column": "num",           
            "output_dir": "output/heart_results"     
        },
        {
            "path": "datasets/heart_processed.csv",             
            "protected_attribute": "age",   
            "target_column": "num",           
            "output_dir": "output/heart_age_results"     
        }
        # Add more dataset entries as needed:
        # {"path": "Dataset/other.csv", "protected_attribute": "protected_col", "target_column": "target", "output_dir": "Output/other_results"}
    ]

    # Models to evaluate (these are model identifier strings consumed by your genetic_algorithm)
    models = ["rf", "lr", "svc", "xgb"]  

    # Parameter grids
    sweep_sizes = [25, 50, 100, 250, 500]  # used for both population_size and generations
    rates = [0.25, 0.5, 0.75, 1.0]         # for crossover_rate and mutation_rate

    # Other defaults
    sample_fraction = 1  # sample fraction for dataset sampling
    overall_results = []
    
    # progressive results file (will be appended to after every run)
    summary_path = os.path.join("output", "experiments_results.csv")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    # write header if file does not exist
    if not os.path.exists(summary_path):
        pd.DataFrame(columns=[
            "dataset", "model_identifier", "population_size", "generations",
            "crossover_rate", "mutation_rate", "techniques", "model_used",
            "fitness", "fairness_score", "performance_score", "elapsed_seconds", "error"
        ]).to_csv(summary_path, index=False)

    for ds_cfg in datasets:
        dataset_path = ds_cfg["path"]
        protected_attribute = ds_cfg["protected_attribute"]
        target_column = ds_cfg["target_column"]
        output_dir = ds_cfg.get("output_dir", "Output")

        os.makedirs(output_dir, exist_ok=True)
        try:
            dataset = pd.read_csv(dataset_path)
        except Exception as e:
            print(f"Failed to load {dataset_path}: {e}")
            continue

        print(f"Running grid for dataset: {dataset_path}")

        
        dataset_sample = sample_dataset(dataset, fraction=sample_fraction)
        dataset_sample = prepare_data_dataset(dataset_sample, target_column)

        for model_identifier in models:
            for population_size in sweep_sizes:
                for generations in sweep_sizes:
                    for crossover_rate in rates:
                        for mutation_rate in rates:
                            run_start = time.time()
                            try:
                                # Pass optional GA parameters as kwargs. If your genetic_algorithm
                                # does not accept some of these kwargs, remove them or update the GA implementation.
                                best_solution = genetic_algorithm(
                                    dataset_sample,
                                    protected_attribute,
                                    target_column,
                                    model_identifier,
                                    generations=generations,
                                    population_size=population_size,
                                    alpha=crossover_rate,
                                    beta=mutation_rate
                                )

                                elapsed = time.time() - run_start
                                
                                techniques = best_solution[0] if best_solution and len(best_solution) > 0 else None
                                model_used = best_solution[1] if best_solution and len(best_solution) > 1 else model_identifier
                                fitness = best_solution[2] if best_solution and len(best_solution) > 2 else None

                                # capture fairness and performance if returned by GA
                                fairness_score = best_solution[3] if best_solution and len(best_solution) > 3 else None
                                performance_score = best_solution[4] if best_solution and len(best_solution) > 4 else None

                                result = {
                                    "dataset": dataset_path,
                                    "model_identifier": model_identifier,
                                    "population_size": population_size,
                                    "generations": generations,
                                    "crossover_rate": crossover_rate,
                                    "mutation_rate": mutation_rate,
                                    "techniques": str(techniques),
                                    "model_used": str(model_used),
                                    "fitness": fitness,
                                    "fairness_score": fairness_score,
                                    "performance_score": performance_score,
                                    "elapsed_seconds": elapsed
                                }
                                overall_results.append(result)

                                # append this single result immediately to CSV to avoid data loss
                                try:
                                    row = result.copy()
                                    row.setdefault("error", None)
                                    pd.DataFrame([row]).to_csv(summary_path, mode='a', header=False, index=False)
                                except Exception as e:
                                    print(f"Failed to append result to {summary_path}: {e}")

                                print(f"OK: model={model_identifier} pop={population_size} gen={generations} cx={crossover_rate} mut={mutation_rate} fitness={fitness}")

                            except Exception as e:
                                elapsed = time.time() - run_start
                                print(f"ERROR: model={model_identifier} pop={population_size} gen={generations} cx={crossover_rate} mut={mutation_rate} -> {e}")
                                err_result = {
                                    "dataset": dataset_path,
                                    "model_identifier": model_identifier,
                                    "population_size": population_size,
                                    "generations": generations,
                                    "crossover_rate": crossover_rate,
                                    "mutation_rate": mutation_rate,
                                    "techniques": None,
                                    "model_used": None,
                                    "fitness": None,
                                    "elapsed_seconds": elapsed,
                                    "error": str(e)
                                }
                                overall_results.append(err_result)
                                # append error row immediately as well
                                try:
                                    pd.DataFrame([err_result]).to_csv(summary_path, mode='a', header=False, index=False)
                                except Exception as e2:
                                    print(f"Failed to append error result to {summary_path}: {e2}")

    # Final save (overwrite with aggregated results) - optional but kept to ensure the full dataframe
    try:
        results_df = pd.DataFrame(overall_results)
        results_df.to_csv(summary_path, index=False)
        print(f"Experiments complete. Results saved to: {summary_path}")
    except Exception as e:
        print(f"Failed to write final aggregated results to {summary_path}: {e}")
    """