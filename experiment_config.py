"""
Experimental configuration shared by every stage of the replication package.

Before this module existed, the dataset catalogue and the parameter grid were
duplicated in ``main.py``, ``run_replication.sh`` and the RQ1/RQ2 scripts.
They now live here only, so all stages are guaranteed to use the same setup
(Table 2 of the paper).
"""
from typing import TypedDict


class DatasetConfig(TypedDict):
    """Description of one benchmark dataset."""

    name: str
    path: str
    preparer_name: str
    protected_attributes: list[str]
    target: str


DATASETS: list[DatasetConfig] = [
    {
        'name': 'adult',
        'path': 'datasets/adult.csv',
        'preparer_name': 'prepare_adult',
        'protected_attributes': ['race', 'sex'],
        'target': 'salary',
    },
    {
        'name': 'german',
        'path': 'datasets/german.csv',
        'preparer_name': 'prepare_german',
        'protected_attributes': ['sex', 'age'],
        'target': 'Target',
    },
    {
        'name': 'heart',
        'path': 'datasets/heart.csv',
        'preparer_name': 'prepare_heart',
        'protected_attributes': ['sex', 'age'],
        'target': 'num',
    },
]

DATASETS_BY_NAME: dict[str, DatasetConfig] = {ds['name']: ds for ds in DATASETS}
DATASETS_BY_PATH: dict[str, DatasetConfig] = {ds['path']: ds for ds in DATASETS}

#: Classifiers evaluated by FATE (optimization tasks O in the paper).
MODELS: list[str] = ['rf', 'lr', 'svc', 'xgb']

#: Search space T: the fairness-aware Data Preparation practices.
TECHNIQUES: list[str] = [
    'standard', 'stratified_sampling', 'oversampling', 'undersampling',
    'clustering', 'ipw', 'matching', 'min_max_scaling',
]

#: GA parameter grid explored in RQ1 (Table 2).
POPULATION_SIZES: list[int] = [5, 10, 15, 20, 50, 100]
GENERATION_COUNTS: list[int] = [5, 10, 15, 20, 50, 100]
RATES: list[float] = [0, 0.25, 0.50, 0.75, 1]  # crossover (alpha) and mutation (beta)

#: Columns of results/fate/experiments_results.csv, in order.
RESULT_COLUMNS: list[str] = [
    'dataset', 'model_identifier', 'protected_attribute',
    'population_size', 'generations', 'alpha', 'beta',
    'techniques', 'model_used', 'fitness',
    'fairness_score', 'performance_score', 'elapsed_seconds',
]
