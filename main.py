"""
FATE experiment orchestrator: parameter-grid runner and result aggregator.

This module is the top-level entry point for FATE experiments.  Its
``__main__`` block runs the full parameter-grid search described in the paper
across three datasets (Adult, German Credit, Heart Disease), four classifiers
(LR, RF, SVC, XGB), two protected attributes per dataset, six population
sizes, six generation counts, and five crossover/mutation rate values —
totalling up to 108 000 GA evaluations across the grid.

Execution is parallelised with ``concurrent.futures.ThreadPoolExecutor``
(default ``max_workers=64``).  Results are appended incrementally to
``FATE_output/experiments_results.csv`` by the parent thread after each future
completes, so partial results are preserved on interruption.

To avoid BLAS / OpenMP oversubscription in multi-threaded operation, all
thread-count environment variables (``OMP_NUM_THREADS``, ``MKL_NUM_THREADS``,
``OPENBLAS_NUM_THREADS``, ``VECLIB_MAXIMUM_THREADS``, ``NUMEXPR_NUM_THREADS``)
are forced to ``1`` both at module import and inside each worker function.

Public API:
    ``get_user_input`` – interactive stdin prompt for single-dataset exploration.
    ``execute_fate``   – run the GA for one (dataset, protected_attribute,
                         model-list) combination and return structured result rows.
"""
import os

# set env vars BEFORE importing numpy/pandas/sklearn/xgboost etc.
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

import logging  # noqa: E402
import pandas as pd  # noqa: E402
from genetic_algorithm import genetic_algorithm  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from concurrent.futures import ThreadPoolExecutor, as_completed  # noqa: E402

logger = logging.getLogger(__name__)

import sys  # noqa: E402
if sys.version_info[:2] != (3, 10):
    raise SystemError(
        f"Python 3.10 is required for replication. "
        f"Detected: Python {sys.version_info[0]}.{sys.version_info[1]}."
    )


def get_user_input():
    """
    Interactively collect experiment parameters from the user via stdin.

    Prompts for dataset path, protected attribute, target column, sample
    fraction, output directory, and model identifier.  Validates that the
    protected attribute and target column exist in the loaded CSV.

    Returns
    -------
    tuple
        ``(dataset, protected_attribute, target_column, output_dir,
        sample_fraction, model_identifier)``
        All elements are ``None`` if any validation step fails.
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
    protected_attribute = input(
        "Enter the name of the protected attribute (e.g., 'Sex_Code_Text'): "
    ).strip()
    if protected_attribute not in dataset.columns:
        print(f"The protected attribute '{protected_attribute}' is not present in the dataset.")
        return None, None, None, None, None, None

    target_column = input(
        "Enter the name of the target variable (e.g., 'DecileScore'): "
    ).strip()
    if target_column not in dataset.columns:
        print(f"The target variable '{target_column}' is not present in the dataset.")
        return None, None, None, None, None, None

    # Get sample fraction and output directory for saving the optimized dataset
    sample_fraction = float(input(
        "Enter the fraction of the dataset to use (e.g., 0.1 for 10%): "
    ).strip())
    output_dir = input(
        "Enter the output directory to save the optimized dataset (e.g., 'Output/'): "
    ).strip()

    # Choose a model identifier to evaluate during GA (no model file loading)
    model_identifier = input(
        "Enter model identifier to evaluate (default 'random_forest'): "
    ).strip() or 'random_forest'

    return (dataset, protected_attribute, target_column, output_dir,
            sample_fraction, model_identifier)


def _write_error_log(errors_path, ds_cfg, prot, pop, gen, alpha, beta,
                     message, tb_str=None):
    """
    Append a structured error entry to the errors log file.

    Called for both per-model GA failures (returned in execute_fate rows with
    error != None) and worker-level exceptions (dataset loading, attribute
    missing, unexpected crashes).  Writing errors to a dedicated file keeps
    experiments_results.csv free of exception messages.

    Parameters
    ----------
    errors_path : str
        Path to errors.log (sibling of experiments_results.csv).
    ds_cfg : dict
        Dataset configuration dict (keys: name, path).
    prot : str
        Protected attribute for this task.
    pop : int
    gen : int
    alpha : float
    beta : float
    message : str
        Short error description.
    tb_str : str or None, optional
        Full traceback string from ``traceback.format_exc()``; omitted for
        per-model errors where a traceback is not available.
    """
    with open(errors_path, 'a', encoding='utf-8') as f:
        f.write('=' * 80 + '\n')
        f.write(f"TIMESTAMP:  {time.strftime('%Y-%m-%dT%H:%M:%S')}\n")
        f.write(f"DATASET:    {ds_cfg['name']} ({ds_cfg['path']})\n")
        f.write(f"PROTECTED:  {prot}\n")
        f.write(f"PARAMS:     pop={pop} gen={gen} alpha={alpha} beta={beta}\n")
        f.write(f"ERROR:      {message}\n")
        if tb_str:
            f.write("TRACEBACK:\n")
            f.write(tb_str)
        f.write('\n')


def _build_ga_params(population_size, generations, alpha, beta):
    """
    Bundle GA hyperparameters into a shared dict used by both result-row builders.

    Parameters
    ----------
    population_size : int
    generations : int
    alpha : float
    beta : float

    Returns
    -------
    dict
        Keys: ``population_size``, ``generations``, ``alpha``, ``beta``.
    """
    return {'population_size': population_size, 'generations': generations,
            'alpha': alpha, 'beta': beta}


def _unpack_best_individual(best, model_id):
    """
    Unpack a GA best-individual tuple into its five named fields.

    Tolerates short or None tuples that can result from an error path in
    ``genetic_algorithm``.

    Parameters
    ----------
    best : tuple or None
        5-tuple ``(techniques, model_used, fitness, fairness_score,
        performance_score)`` returned by ``genetic_algorithm``.
    model_id : str
        Fallback model name when *best* does not include a model field.

    Returns
    -------
    tuple
        ``(techniques, model_used, fitness_val, fairness, perf)``
        Absent fields are ``None`` (model field falls back to *model_id*).
    """
    b = best or ()
    return (
        b[0] if len(b) > 0 else None,
        b[1] if len(b) > 1 else model_id,
        b[2] if len(b) > 2 else None,
        b[3] if len(b) > 3 else None,
        b[4] if len(b) > 4 else None,
    )


def _build_result_row(ds_path, model_id, protected_attribute, ga_params, best, elapsed):
    """
    Build a structured result dict from a successful GA run.

    Role in Algorithm 1 output: packages the best individual returned by
    ``genetic_algorithm`` into the CSV schema expected by
    ``FATE_output/experiments_results.csv``.

    Parameters
    ----------
    ds_path : str
        Path to the raw CSV dataset (stored for traceability).
    model_id : str
        Classifier identifier used in this run.
    protected_attribute : str
    ga_params : dict
        Output of ``_build_ga_params``.
    best : tuple or None
        5-tuple from ``genetic_algorithm``; unpacked via
        ``_unpack_best_individual``.
    elapsed : float
        Wall-clock seconds for the GA run.

    Returns
    -------
    dict
        Result row matching the experiments CSV schema.
    """
    techniques, model_used, fitness_val, fairness, perf = _unpack_best_individual(best, model_id)
    return {
        'dataset': ds_path, 'model_identifier': model_id,
        'protected_attribute': protected_attribute, **ga_params,
        'techniques': str(techniques), 'model_used': str(model_used),
        'fitness': fitness_val, 'fairness_score': fairness,
        'performance_score': perf, 'elapsed_seconds': elapsed, 'error': None,
    }


def _build_error_row(ds_path, model_id, protected_attribute, ga_params, elapsed, error_msg):
    """
    Build a structured error dict when a GA run raises an exception.

    Parameters
    ----------
    ds_path : str
    model_id : str
    protected_attribute : str
    ga_params : dict
        Output of ``_build_ga_params``.
    elapsed : float
        Wall-clock seconds elapsed before the exception was raised.
    error_msg : str
        String representation of the exception.

    Returns
    -------
    dict
        Error row matching the experiments CSV schema; numeric fields are None.
    """
    return {
        'dataset': ds_path, 'model_identifier': model_id,
        'protected_attribute': protected_attribute, **ga_params,
        'techniques': None, 'model_used': None,
        'fitness': None, 'fairness_score': None,
        'performance_score': None, 'elapsed_seconds': elapsed, 'error': error_msg,
    }


def _append_row_to_csv(summary_path, row):
    """
    Append one result row to the summary CSV if a path is provided.

    Parameters
    ----------
    summary_path : str or None
        Destination CSV path.  No-op when None.
    row : dict
        Single result or error row to append.
    """
    if summary_path:
        pd.DataFrame([row]).to_csv(summary_path, mode='a', header=False, index=False)


def _run_timed_fate(sample_ready, protected_attribute, target, model_id,
                    population_size, generations, alpha, beta, reset_cache=False):
    """
    Execute one FATE GA run for a single classifier and return the result with timing.

    Wraps ``genetic_algorithm`` with a wall-clock timer.

    Parameters
    ----------
    sample_ready : pd.DataFrame
        Pre-processed dataset.
    protected_attribute : str
    target : str
    model_id : str
    population_size : int
    generations : int
    alpha : float
    beta : float

    Returns
    -------
    tuple
        ``(best, elapsed)`` where *best* is the 5-tuple from
        ``genetic_algorithm`` and *elapsed* is seconds as a float.

    Raises
    ------
    Exception
        Any exception from ``genetic_algorithm`` is propagated unchanged so
        that ``execute_fate`` can build a correct error row with timing.
    """
    start = time.time()
    best = genetic_algorithm(sample_ready, protected_attribute, target, model_id,
                             generations=generations, population_size=population_size,
                             alpha=alpha, beta=beta, reset_cache=reset_cache)
    return best, time.time() - start


def execute_fate(sample_ready, ds_name, ds_path, protected_attribute, target, models,
                 population_size, generations, alpha=0.5, beta=0.5, summary_path=None,
                 reset_cache=False):
    """
    Run the FATE genetic algorithm for one dataset / protected-attribute combination
    across multiple classifiers.

    Iterates over *models*, calls ``genetic_algorithm.genetic_algorithm`` for
    each, and packages the results into structured row dicts suitable for
    appending to the experiments summary CSV.

    Parameters
    ----------
    sample_ready : pd.DataFrame
        Pre-processed dataset (output of ``preprocessing.prepare_data_model``).
    ds_name : str
        Human-readable dataset name used in console output.
    ds_path : str
        Path to the raw CSV file; stored in result rows for traceability.
    protected_attribute : str
        Name of the protected attribute column in *sample_ready*.
    target : str
        Name of the label column in *sample_ready*.
    models : list of str
        Classifier identifiers to evaluate sequentially.  One full GA run is
        performed per model.
    population_size : int
        Passed to ``genetic_algorithm`` (Algorithm 1 population size |P|).
    generations : int
        Passed to ``genetic_algorithm`` (Algorithm 1 generation count G).
    alpha : float, optional
        Crossover probability (default 0.5).
    beta : float, optional
        Mutation probability (default 0.5).
    summary_path : str or None, optional
        If provided, each result row is immediately appended to this CSV file.
        Pass ``None`` to suppress file I/O (used in worker threads where the
        parent process handles writing).
    reset_cache : bool, optional
        Forwarded to ``genetic_algorithm``.  When False (default), fitness
        evaluations are looked up in the read-only root ``experiments_cache.csv``
        before running the classifier.  When True, only the runtime cache
        ``FATE_output/runtime_cache.csv`` is used.

    Returns
    -------
    list of dict
        One dict per model containing: ``dataset``, ``model_identifier``,
        ``protected_attribute``, ``population_size``, ``generations``,
        ``alpha``, ``beta``, ``techniques``, ``model_used``, ``fitness``,
        ``fairness_score``, ``performance_score``, ``elapsed_seconds``,
        ``error``.  On exception, numeric fields are ``None`` and ``error``
        holds the exception message string.
    """
    ga_params = _build_ga_params(population_size, generations, alpha, beta)
    results = []
    for model_id in models:
        logger.info("[%s/%s/%s] Starting GA  pop=%d  gen=%d",
                    ds_name, protected_attribute, model_id, population_size, generations)
        outer_start = time.time()
        try:
            best, elapsed = _run_timed_fate(
                sample_ready, protected_attribute, target, model_id,
                population_size, generations, alpha, beta, reset_cache=reset_cache)
            row = _build_result_row(
                ds_path, model_id, protected_attribute, ga_params, best, elapsed)
            logger.info("[%s/%s/%s] Done  fitness=%.4f  elapsed=%.1fs",
                        ds_name, protected_attribute, model_id, row['fitness'], elapsed)
        except Exception as e:
            elapsed = time.time() - outer_start
            row = _build_error_row(
                ds_path, model_id, protected_attribute, ga_params, elapsed, str(e))
            logger.warning("[%s/%s/%s] Failed: %s", ds_name, protected_attribute, model_id, e)
        results.append(row)
        _append_row_to_csv(summary_path, row)
    return results


if __name__ == "__main__":
    # Simplified automated runner for the 3 datasets (adult, german, heart)
    datasets = [
        {
            'name': 'adult',
            'path': 'datasets/adult.csv',
            'preparer_name': 'prepare_adult',
            'protected_attributes': ['race', 'sex'],
            'target': 'salary'
        },
        {
            'name': 'german',
            'path': 'datasets/german.csv',
            'preparer_name': 'prepare_german',
            'protected_attributes': ['sex', 'age'],
            'target': 'Target'
        },
        {
            'name': 'heart',
            'path': 'datasets/heart.csv',
            'preparer_name': 'prepare_heart',
            'protected_attributes': ['sex', 'age'],
            'target': 'num'
        }
    ]

    models = ['rf', 'lr', 'svc', 'xgb']

    logging.basicConfig(
        format='%(asctime)s %(levelname)-7s %(message)s',
        datefmt='%H:%M:%S',
        level=logging.INFO,
    )

    # results file (progressive append) — success rows only
    summary_path = os.path.join('FATE_output', 'experiments_results.csv')
    # errors file — per-model GA failures and worker-level exceptions land here,
    # never in experiments_results.csv (Bug 2 fix)
    errors_path = os.path.join('FATE_output', 'errors.log')
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    if not os.path.exists(summary_path):
        pd.DataFrame(columns=[
            'dataset', 'model_identifier', 'protected_attribute',
            'population_size', 'generations',
            'alpha', 'beta',
            'techniques', 'model_used', 'fitness',
            'fairness_score', 'performance_score', 'elapsed_seconds',
        ]).to_csv(summary_path, index=False)

    # False  → use experiments_cache.csv (read-only) as additional lookup source.
    # True   → skip root cache; only FATE_output/runtime_cache.csv is consulted.
    reset_cache = False

    sample_fraction = 1
    overall_start = time.time()

    # parameter grid
    population_sizes = [5, 10, 15, 20, 50, 100]
    generations_list = [5, 10, 15, 20, 50, 100]
    rates = [0, 0.25, 0.50, 0.75, 1]  # crossover and mutation rates

    # helper worker that runs the GA for one parameter combination and returns rows (no file IO)
    def worker_task(ds_cfg, prot, pop, gen, alpha, beta):
        """
        Thread-worker: load the dataset, prepare it, and delegate to ``execute_fate``.

        Called by the ``ThreadPoolExecutor``.  Loads the raw CSV, applies the
        dataset-specific preparer from ``preprocessing``, constructs the
        model-ready sample, and calls ``execute_fate`` without file I/O
        (``summary_path=None``).  The parent thread aggregates and writes CSV.

        Parameters
        ----------
        ds_cfg : dict
            Dataset configuration with keys: ``name``, ``path``,
            ``preparer_name``, ``protected_attributes``, ``target``.
        prot : str
            Protected attribute to evaluate for this task.
        pop : int
            Population size for this parameter-grid point.
        gen : int
            Generation count for this parameter-grid point.
        alpha : float
            Crossover probability.
        beta : float
            Mutation probability.

        Returns
        -------
        list of dict
            Result rows (same schema as ``execute_fate``), one per model.
            Returns error rows for all models if dataset preparation or GA
            execution raises an exception.
        """
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
            # Raise so the parent as_completed handler logs this to errors.log
            # rather than writing a synthetic error row into the results CSV.
            raise ValueError(
                f"Protected attribute '{prot}' not found in dataset "
                f"'{ds_cfg['name']}' after preparation"
            )

        sample_ready = _prepare_data_model(
            sample, ds_cfg['target'], protected_attribute=prot, binarize=False)

        rows = execute_fate(
            sample_ready, ds_cfg['name'], ds_cfg['path'], prot, ds_cfg['target'], models,
            population_size=pop, generations=gen, alpha=alpha, beta=beta, summary_path=None,
            reset_cache=reset_cache)
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

    logger.info("Starting parallel run  |  workers=%d  |  tasks=%d", max_workers, len(tasks))

    # submit tasks and write results progressively in the parent process
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {executor.submit(worker_task, *t): t for t in tasks}
        for fut in as_completed(future_to_task):
            ds_cfg, prot, pop, gen, alpha, beta = future_to_task[fut]
            try:
                rows = fut.result()
                # Bug 2 fix: separate success rows from per-model error rows.
                # Only success rows go into experiments_results.csv; failures
                # are written to errors.log so the CSV stays clean.
                success_rows = [r for r in rows if r.get('error') is None]
                error_rows = [r for r in rows if r.get('error') is not None]

                if success_rows:
                    # drop the 'error' column before writing — it is always None here
                    _df = pd.DataFrame(success_rows).drop(columns=['error'], errors='ignore')
                    _df.to_csv(summary_path, mode='a', header=False, index=False)
                    for r in success_rows:
                        logger.info("Saved  %s/%s/%s  pop=%d gen=%d  α=%.2f β=%.2f",
                                    ds_cfg['name'], prot, r['model_identifier'],
                                    pop, gen, alpha, beta)

                for r in error_rows:
                    _write_error_log(errors_path, ds_cfg, prot, pop, gen, alpha, beta,
                                     f"model={r['model_identifier']}: {r['error']}")
                    logger.warning("Model error logged  %s/%s/%s: %s",
                                   ds_cfg['name'], prot, r['model_identifier'], r['error'])

            except Exception as exc:
                # Worker-level exception (dataset load, attribute missing, unexpected crash).
                # Write full traceback to errors.log; do not touch the results CSV.
                _write_error_log(errors_path, ds_cfg, prot, pop, gen, alpha, beta,
                                 str(exc), traceback.format_exc())
                logger.error("Task failed  %s/%s  pop=%d gen=%d: %s",
                             ds_cfg['name'], prot, pop, gen, exc)

    overall_elapsed = time.time() - overall_start
    logger.info("All experiments done  |  results → %s  |  elapsed %.1fs",
                summary_path, overall_elapsed)
