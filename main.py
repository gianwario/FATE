"""
FATE experiment orchestrator (Stage 1 of the replication pipeline).

Runs the FATE genetic algorithm over the parameter grid of the paper (Table 2):
three datasets (Adult, German Credit, Heart Disease), two protected attributes
per dataset, four classifiers (LR, RF, SVC, XGB), six population sizes, six
generation counts, and five crossover and five mutation rates.

Usage (from the repository root)::

    python main.py                                   # full paper grid
    python main.py --datasets adult --models lr      # restricted grid
    python main.py --datasets adult --models lr --protected sex \
                   --pop 5 --gen 5 --alpha 0.5 --beta 0.5   # one GA run

Output (see ``paths.py``):

    results/fate/experiments_results.csv   one row per successful GA run
    results/fate/errors.log                one entry per failed run (only if any)
    results/fate/runtime_cache.csv         fitness cache written during the run

Execution is parallelised with ``concurrent.futures.ThreadPoolExecutor``.
Results are appended by the parent thread as soon as each task completes, so
partial results survive an interruption.  To avoid BLAS / OpenMP
oversubscription, the thread-count environment variables are set to ``1``
before the numerical libraries are imported.

Public API:
    ``execute_fate`` – run the GA for one (dataset, protected attribute,
                       model list, GA parameters) combination.
    ``run_grid``     – run a list of such combinations in parallel.
    ``main``         – command-line entry point.
"""
import os

# set env vars BEFORE importing numpy/pandas/sklearn/xgboost etc.
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

import argparse  # noqa: E402
import logging  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from concurrent.futures import ThreadPoolExecutor, as_completed  # noqa: E402
from typing import Optional, Union  # noqa: E402

import pandas as pd  # noqa: E402

import paths  # noqa: E402
import preprocessing  # noqa: E402
from experiment_config import (  # noqa: E402
    DATASETS, DATASETS_BY_NAME, GENERATION_COUNTS, MODELS, POPULATION_SIZES, RATES,
    RESULT_COLUMNS, DatasetConfig,
)
from genetic_algorithm import ScoredIndividual, genetic_algorithm  # noqa: E402

#: Errors that make a single GA run fail without invalidating the grid:
#: numerical errors (``FloatingPointError``, ``numerics.NonFiniteResultError``;
#: both are ``ArithmeticError``) and data errors raised by pandas/scikit-learn
#: (``ValueError``).  A failed run is written to ``errors.log`` and never to the
#: results CSV.  Any other exception is a bug and stops the grid.
RUN_ERRORS = (ArithmeticError, ValueError)
#: Additional errors that can occur while loading a task's dataset.
TASK_ERRORS = RUN_ERRORS + (KeyError, OSError)

logger = logging.getLogger(__name__)

if sys.version_info[:2] != (3, 10):
    raise SystemError(
        f"Python 3.10 is required for replication. "
        f"Detected: Python {sys.version_info[0]}.{sys.version_info[1]}."
    )

#: One grid point: (dataset config, protected attribute, population size,
#: generations, crossover rate alpha, mutation rate beta).
Task = tuple[DatasetConfig, str, int, int, float, float]
#: Fields of a (possibly incomplete) best individual, see ``_unpack_best_individual``.
UnpackedIndividual = tuple[Optional[list[str]], str, Optional[float], Optional[float],
                           Optional[float]]
#: One result or error row (schema: ``experiment_config.RESULT_COLUMNS`` + ``error``).
ResultRow = dict[str, object]


def _write_error_log(errors_path: str, ds_cfg: DatasetConfig, prot: str, pop: int, gen: int,
                     alpha: float, beta: float, message: str, tb_str: Optional[str] = None) -> None:
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


def _build_ga_params(population_size: int, generations: int, alpha: float, beta: float
                     ) -> dict[str, Union[int, float]]:
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


def _unpack_best_individual(best: Optional[ScoredIndividual], model_id: str
                            ) -> UnpackedIndividual:
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


def _build_result_row(ds_path: str, model_id: str, protected_attribute: str,
                      ga_params: dict[str, Union[int, float]], best: Optional[ScoredIndividual],
                      elapsed: float) -> ResultRow:
    """
    Build a structured result dict from a successful GA run.

    Role in Algorithm 1 output: packages the best individual returned by
    ``genetic_algorithm`` into the CSV schema expected by
    ``results/fate/experiments_results.csv``.

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


def _build_error_row(ds_path: str, model_id: str, protected_attribute: str,
                     ga_params: dict[str, Union[int, float]], elapsed: float, error_msg: str
                     ) -> ResultRow:
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


def _append_row_to_csv(summary_path: Optional[str], row: ResultRow) -> None:
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


def _run_timed_fate(sample_ready: pd.DataFrame, protected_attribute: str, target: str,
                    model_id: str, population_size: int, generations: int, alpha: float,
                    beta: float, reset_cache: bool = False) -> tuple[ScoredIndividual, float]:
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
    ArithmeticError, ValueError
        Propagated unchanged from ``genetic_algorithm`` so that
        ``execute_fate`` can build a correct error row with timing.
    """
    start = time.time()
    best = genetic_algorithm(sample_ready, protected_attribute, target, model_id,
                             generations=generations, population_size=population_size,
                             alpha=alpha, beta=beta, reset_cache=reset_cache)
    return best, time.time() - start


def execute_fate(sample_ready: pd.DataFrame, ds_name: str, ds_path: str, protected_attribute: str,
                 target: str, models: list[str], population_size: int, generations: int,
                 alpha: float = 0.5, beta: float = 0.5, summary_path: Optional[str] = None,
                 reset_cache: bool = False) -> list[ResultRow]:
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
        evaluations are looked up in the read-only ``reference/fate/fitness_cache.csv``
        before running the classifier.  When True, only the runtime cache
        ``results/fate/runtime_cache.csv`` is used.

    Returns
    -------
    list of dict
        One dict per model containing: ``dataset``, ``model_identifier``,
        ``protected_attribute``, ``population_size``, ``generations``,
        ``alpha``, ``beta``, ``techniques``, ``model_used``, ``fitness``,
        ``fairness_score``, ``performance_score``, ``elapsed_seconds``,
        ``error``.  If the run raises one of ``RUN_ERRORS``, numeric fields
        are ``None`` and ``error`` holds the exception type and message; any
        other exception propagates.
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
        except RUN_ERRORS as e:
            elapsed = time.time() - outer_start
            row = _build_error_row(
                ds_path, model_id, protected_attribute, ga_params, elapsed,
                f"{type(e).__name__}: {e}")
            logger.warning("[%s/%s/%s] Failed: %s", ds_name, protected_attribute, model_id, e)
        results.append(row)
        _append_row_to_csv(summary_path, row)
    return results


# ---------------------------------------------------------------------------
# Grid execution
# ---------------------------------------------------------------------------

def load_task_sample(ds_cfg: DatasetConfig, prot: str) -> pd.DataFrame:
    """
    Load a dataset and prepare it for FATE with respect to one protected attribute.

    Applies the dataset-specific preparer from ``preprocessing`` followed by
    ``preprocessing.prepare_data_model``.

    Parameters
    ----------
    ds_cfg : DatasetConfig
        Entry of ``experiment_config.DATASETS``.
    prot : str
        Protected attribute to preserve.

    Returns
    -------
    pd.DataFrame
        Model-ready dataset.

    Raises
    ------
    ValueError
        If *prot* is not a column of the prepared dataset.
    """
    raw = pd.read_csv(paths.REPO_ROOT / ds_cfg['path'])
    preparer = getattr(preprocessing, ds_cfg['preparer_name'])
    processed = preparer(raw)
    if prot not in processed.columns:
        raise ValueError(
            f"Protected attribute '{prot}' not found in dataset "
            f"'{ds_cfg['name']}' after preparation"
        )
    return preprocessing.prepare_data_model(
        processed, ds_cfg['target'], protected_attribute=prot, binarize=False)


def worker_task(task: Task, models: list[str], reset_cache: bool) -> list[ResultRow]:
    """
    Thread worker: prepare the dataset of *task* and delegate to ``execute_fate``.

    Called by the ``ThreadPoolExecutor`` in ``run_grid``.  No file I/O is done
    here (``summary_path=None``); the parent thread writes results and errors.

    Parameters
    ----------
    task : Task
        ``(ds_cfg, prot, pop, gen, alpha, beta)``.
    models : list of str
        Classifier identifiers; one GA run is performed per model.
    reset_cache : bool
        Forwarded to ``execute_fate``.

    Returns
    -------
    list of dict
        Result rows (same schema as ``execute_fate``), one per model.
    """
    ds_cfg, prot, pop, gen, alpha, beta = task
    sample_ready = load_task_sample(ds_cfg, prot)
    return execute_fate(
        sample_ready, ds_cfg['name'], ds_cfg['path'], prot, ds_cfg['target'], models,
        population_size=pop, generations=gen, alpha=alpha, beta=beta, summary_path=None,
        reset_cache=reset_cache)


def build_tasks(datasets: list[DatasetConfig], protected: Optional[list[str]],
                population_sizes: list[int], generation_counts: list[int],
                alphas: list[float], betas: list[float]) -> list[Task]:
    """
    Enumerate the grid points (Cartesian product of the given values).

    Parameters
    ----------
    datasets : list of DatasetConfig
    protected : list of str or None
        Protected attributes to keep; ``None`` keeps every attribute listed in
        each dataset configuration.
    population_sizes, generation_counts : list of int
    alphas, betas : list of float

    Returns
    -------
    list of Task
    """
    return [
        (ds, prot, pop, gen, alpha, beta)
        for ds in datasets
        for prot in ds['protected_attributes']
        if protected is None or prot in protected
        for pop in population_sizes
        for gen in generation_counts
        for alpha in alphas
        for beta in betas
    ]


def init_results_csv(summary_path: str) -> None:
    """Create *summary_path* with the result header if it does not exist yet."""
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    if not os.path.exists(summary_path):
        pd.DataFrame(columns=RESULT_COLUMNS).to_csv(summary_path, index=False)


def record_task_rows(rows: list[ResultRow], task: Task, summary_path: str,
                     errors_path: str) -> tuple[int, int]:
    """
    Write the rows of one completed task: successes to the CSV, failures to the log.

    Parameters
    ----------
    rows : list of dict
        Rows returned by ``worker_task``.
    task : Task
        Grid point that produced *rows* (used for the error log).
    summary_path : str
        Results CSV.
    errors_path : str
        Errors log.

    Returns
    -------
    tuple of int
        ``(n_success, n_error)``.
    """
    ds_cfg, prot, pop, gen, alpha, beta = task
    success_rows = [r for r in rows if r.get('error') is None]
    error_rows = [r for r in rows if r.get('error') is not None]
    if success_rows:
        # drop the 'error' column before writing: it is always None here
        pd.DataFrame(success_rows).drop(columns=['error'], errors='ignore').to_csv(
            summary_path, mode='a', header=False, index=False)
    for r in error_rows:
        _write_error_log(errors_path, ds_cfg, prot, pop, gen, alpha, beta,
                         f"model={r['model_identifier']}: {r['error']}")
    return len(success_rows), len(error_rows)


def print_single_result(rows: list[ResultRow], elapsed: float) -> None:
    """Print a readable summary of a single GA run (used by the fast mode)."""
    print("=" * 60)
    print("  FATE result")
    print("=" * 60)
    for r in rows:
        if r.get('error') is not None:
            print(f"  FAILED ({r['model_identifier']}): {r['error']}")
            continue
        print(f"  Best pipeline : {r['techniques']}")
        print(f"  Model         : {r['model_used']}")
        print(f"  Fitness       : {r['fitness']:.4f}")
        print(f"  Performance   : {r['performance_score']:.4f}  (mean PR-AUC across 5 CV folds)")
        print(f"  Fairness      : {r['fairness_score']:.4f}  (mean (|SPD|+|EOD|+|DI|) / 3)")
    print(f"  Elapsed       : {elapsed:.1f} s")
    print("=" * 60)


def run_grid(tasks: list[Task], models: list[str], max_workers: int, reset_cache: bool,
             summary_path: str, errors_path: str) -> tuple[int, int]:
    """
    Run all *tasks* in parallel, writing results progressively.

    Parameters
    ----------
    tasks : list of Task
    models : list of str
    max_workers : int
        Number of worker threads.
    reset_cache : bool
        Forwarded to every GA run.
    summary_path : str
    errors_path : str

    Returns
    -------
    tuple of int
        ``(n_success, n_error)`` summed over all tasks.  Failed runs and
        tasks (``RUN_ERRORS``, ``TASK_ERRORS``) are counted in ``n_error`` and
        written to *errors_path* with their traceback.

    Raises
    ------
    Exception
        Any other exception raised by a task is not caught: it stops the grid.
    """
    init_results_csv(summary_path)
    total = len(tasks)
    width = len(str(total))
    n_ok = n_err = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {executor.submit(worker_task, t, models, reset_cache): t
                          for t in tasks}
        for done, fut in enumerate(as_completed(future_to_task), start=1):
            task = future_to_task[fut]
            ds_cfg, prot, pop, gen, alpha, beta = task
            try:
                rows = fut.result()
            except TASK_ERRORS as exc:  # task-level failure: log with traceback
                _write_error_log(errors_path, ds_cfg, prot, pop, gen, alpha, beta,
                                 f"{type(exc).__name__}: {exc}", traceback.format_exc())
                n_err += 1
                print(f"[{done:{width}}/{total}] FAILED ds={ds_cfg['name']} prot={prot} "
                      f"pop={pop} gen={gen}: {exc}", flush=True)
                continue
            ok, err = record_task_rows(rows, task, summary_path, errors_path)
            n_ok += ok
            n_err += err
            print(f"[{done:{width}}/{total}] ds={ds_cfg['name']} prot={prot} pop={pop} "
                  f"gen={gen} a={alpha} b={beta}  ok={ok} err={err}", flush=True)
            if total == 1:
                print_single_result(rows, float(rows[0]['elapsed_seconds']))
    return n_ok, n_err


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse the command line; every option defaults to the full paper grid."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument('--datasets', nargs='+', choices=list(DATASETS_BY_NAME),
                        default=list(DATASETS_BY_NAME))
    parser.add_argument('--protected', nargs='+', default=None,
                        help='protected attributes to keep (default: all per dataset)')
    parser.add_argument('--models', nargs='+', choices=MODELS, default=MODELS)
    parser.add_argument('--pop', nargs='+', type=int, default=POPULATION_SIZES)
    parser.add_argument('--gen', nargs='+', type=int, default=GENERATION_COUNTS)
    parser.add_argument('--alpha', nargs='+', type=float, default=RATES)
    parser.add_argument('--beta', nargs='+', type=float, default=RATES)
    parser.add_argument('--workers', type=int, default=64)
    parser.add_argument('--reset-cache', action='store_true',
                        help='ignore the archived fitness cache in reference/fate/')
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    """Command-line entry point: run the requested (sub)grid of FATE experiments."""
    args = parse_args(argv)
    logging.basicConfig(format='%(asctime)s %(levelname)-7s %(message)s',
                        datefmt='%H:%M:%S', level=logging.WARNING)
    datasets = [ds for ds in DATASETS if ds['name'] in args.datasets]
    tasks = build_tasks(datasets, args.protected, args.pop, args.gen, args.alpha, args.beta)
    summary_path = str(paths.FATE_RESULTS_CSV)
    errors_path = str(paths.FATE_ERRORS_LOG)
    print(f"Tasks       : {len(tasks)}  (x {len(args.models)} models)")
    print(f"Results CSV : {summary_path}")
    print(f"Errors log  : {errors_path}", flush=True)
    start = time.time()
    n_ok, n_err = run_grid(tasks, args.models, args.workers, args.reset_cache,
                           summary_path, errors_path)
    print(f"Complete. GA runs ok={n_ok} failed={n_err}. "
          f"Elapsed: {time.time() - start:.0f} s")


if __name__ == "__main__":
    main()
