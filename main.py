import os

# set env vars BEFORE importing numpy/pandas/sklearn/xgboost etc.
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

import pandas as pd
from genetic_algorithm import genetic_algorithm
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

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


def execute_fate(sample_ready, ds_name, ds_path, protected_attribute, target, models,
                 population_size, generations, alpha=0.5, beta=0.5, summary_path=None):
    """
    Run GA for the given prepared dataset and protected attribute across the provided models.

    Args:
        sample_ready (pd.DataFrame): Prepared dataframe ready for modeling (features + target).
        ds_name (str): Friendly dataset name.
        ds_path (str): Path to dataset (used for logging/storage).
        protected_attribute (str): Protected attribute column name present in sample_ready.
        target (str): Target column name.
        models (list): List of model identifiers to evaluate (e.g., ['rf','lr','svc','xgb']).
        population_size (int): GA population size.
        generations (int): GA generations.
        alpha (float): GA crossover probability.
        beta (float): GA mutation probability.
        summary_path (str): Path to CSV file to append results to. If None, no file writing.

    Returns:
        list: rows written or created for each model (dicts)
    """
    results = []
    for model_id in models:
        print(f"-> Running GA on {ds_name} protected={protected_attribute} model={model_id}")
        start = time.time()
        try:
            best = genetic_algorithm(sample_ready, protected_attribute, target, model_id,
                                      generations=generations, population_size=population_size,
                                      alpha=alpha, beta=beta)

            elapsed = time.time() - start
            techniques = best[0] if best and len(best) > 0 else None
            model_used = best[1] if best and len(best) > 1 else model_id
            fitness_val = best[2] if best and len(best) > 2 else None
            fairness_score = best[3] if best and len(best) > 3 else None
            perf_score = best[4] if best and len(best) > 4 else None

            row = {
                'dataset': ds_path,
                'model_identifier': model_id,
                'protected_attribute': protected_attribute,
                'population_size': population_size,
                'generations': generations,
                'alpha': alpha,
                'beta': beta,
                'techniques': str(techniques),
                'model_used': str(model_used),
                'fitness': fitness_val,
                'fairness_score': fairness_score,
                'performance_score': perf_score,
                'elapsed_seconds': elapsed,
                'error': None
            }
            results.append(row)
            if summary_path:
                pd.DataFrame([row]).to_csv(summary_path, mode='a', header=False, index=False)
            print(f"OK: ds={ds_name} prot={protected_attribute} model={model_id} fitness={fitness_val} elapsed={elapsed:.1f}s")

        except Exception as e:
            elapsed = time.time() - start
            err = {
                'dataset': ds_path,
                'model_identifier': model_id,
                'protected_attribute': protected_attribute,
                'population_size': population_size,
                'generations': generations,
                'alpha': alpha,
                'beta': beta,
                'techniques': None,
                'model_used': None,
                'fitness': None,
                'fairness_score': None,
                'performance_score': None,
                'elapsed_seconds': elapsed,
                'error': str(e)
            }
            results.append(err)
            if summary_path:
                pd.DataFrame([err]).to_csv(summary_path, mode='a', header=False, index=False)
            print(f"ERROR: ds={ds_name} prot={protected_attribute} model={model_id} -> {e}")

    return results


if __name__ == "__main__":
    # Simplified automated runner for the 3 datasets (adult, german, heart)
    datasets = [
        #{
        #    'name': 'adult',
        #    'path': 'datasets/adult.csv',
        #    'preparer_name': 'prepare_adult',
        #    'protected_attributes': ['race', 'sex'],
        #    'target': 'salary'
        #},
        #{
        #    'name': 'german',
        #    'path': 'datasets/german.csv',
        #    'preparer_name': 'prepare_german',
        #    'protected_attributes': ['sex', 'age'],
        #    'target': 'Target'
        #},
        {
            'name': 'heart',
            'path': 'datasets/heart.csv',
            'preparer_name': 'prepare_heart',
            'protected_attributes': ['sex', 'age'],
            'target': 'num'
        }
    ]

    models = ['rf', 'lr', 'svc', 'xgb']

    # results file (progressive append)
    summary_path = os.path.join('output', 'experiments_results.csv')
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    if not os.path.exists(summary_path):
        pd.DataFrame(columns=[
            'dataset', 'model_identifier', 'protected_attribute', 'population_size', 'generations',
            'alpha', 'beta',
            'techniques', 'model_used', 'fitness', 'fairness_score', 'performance_score', 'elapsed_seconds', 'error'
        ]).to_csv(summary_path, index=False)

    sample_fraction = 1
    overall_start = time.time()

    # parameter grid
    population_sizes = [5, 10, 15, 20, 50, 100]
    generations_list = [5, 10, 15, 20, 50, 100]
    rates = [0, 0.25, 0.50, 0.75, 1]  # crossover and mutation rates

    # helper worker that runs the GA for one parameter combination and returns rows (no file IO)
    def worker_task(ds_cfg, prot, pop, gen, alpha, beta):
        # ensure worker limits BLAS/OMP threads before importing heavy libs
        import os as _os
        _os.environ.setdefault('OMP_NUM_THREADS', '1')
        _os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
        _os.environ.setdefault('MKL_NUM_THREADS', '1')
        _os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')
        _os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

        # run inside worker process: load, prepare, sample, prepare for model, call execute_fate
        import pandas as _pd
        from preprocessing import prepare_data_model as _prepare_data_model
        import preprocessing as _preproc

        # load and prepare dataset
        raw = _pd.read_csv(ds_cfg['path'])
        preparer = getattr(_preproc, ds_cfg['preparer_name'])
        processed = preparer(raw)
        sample = processed

        if prot not in sample.columns:
            # return an error row for each model so parent can append
            rows = []
            for m in models:
                rows.append({
                    'dataset': ds_cfg['path'],
                    'model_identifier': m,
                    'protected_attribute': prot,
                    'population_size': pop,
                    'generations': gen,
                    'alpha': alpha,
                    'beta': beta,
                    'techniques': None,
                    'model_used': None,
                    'fitness': None,
                    'fairness_score': None,
                    'performance_score': None,
                    'elapsed_seconds': 0.0,
                    'error': f"protected attribute '{prot}' not found after preparation"
                })
            return rows

        sample_ready = _prepare_data_model(sample, ds_cfg['target'], protected_attribute=prot, binarize=False)

        # call execute_fate but don't let worker write CSV; return the rows instead
        try:
            rows = execute_fate(sample_ready, ds_cfg['name'], ds_cfg['path'], prot, ds_cfg['target'], models,
                                population_size=pop, generations=gen, alpha=alpha, beta=beta, summary_path=None)
            return rows
        except Exception as e:
            # return error rows for each model on failure
            rows = []
            for m in models:
                rows.append({
                    'dataset': ds_cfg['path'],
                    'model_identifier': m,
                    'protected_attribute': prot,
                    'population_size': pop,
                    'generations': gen,
                    'alpha': alpha,
                    'beta': beta,
                    'techniques': None,
                    'model_used': None,
                    'fitness': None,
                    'fairness_score': None,
                    'performance_score': None,
                    'elapsed_seconds': 0.0,
                    'error': str(e)
                })
            return rows

    # set environment to avoid BLAS/OMP oversubscription
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')
    os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')
    os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

    # choose how many threads to run in parallel
    max_workers = 64

    # build list of tasks (ds_cfg, prot, pop, gen, alpha, beta)
    tasks = []
    for ds in datasets:
        for prot in ds['protected_attributes']:
            for pop in population_sizes:
                for gen in generations_list:
                    for alpha in rates:
                        for beta in rates:
                            tasks.append((ds, prot, pop, gen, alpha, beta))

    print(f"Starting parallel run with up to {max_workers} workers, total tasks: {len(tasks)}")

    # submit tasks and write results progressively in the parent process
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {executor.submit(worker_task, *t): t for t in tasks}
        for fut in as_completed(future_to_task):
            ds_cfg, prot, pop, gen, alpha, beta = future_to_task[fut]
            try:
                rows = fut.result()
                # append rows to CSV (parent process writes)
                if rows:
                    _df = pd.DataFrame(rows)
                    _df.to_csv(summary_path, mode='a', header=False, index=False)
                    for r in rows:
                        print(f"APPENDED: ds={ds_cfg['name']} prot={prot} model={r['model_identifier']} pop={pop} gen={gen} alpha={alpha} beta={beta} err={r['error']}")
            except Exception as exc:
                print(f"Task failed for ds={ds_cfg['name']} prot={prot} pop={pop} gen={gen} alpha={alpha} beta={beta} -> {exc}")

    overall_elapsed = time.time() - overall_start
    print(f"\nAll experiments finished. Results appended to {summary_path}")
    print(f"Total experiments elapsed time: {overall_elapsed:.1f} seconds")

