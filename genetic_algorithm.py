# genetic_algorithm.py
"""
Core FATE genetic algorithm implementation (Algorithm 1 from the paper).

This module exposes a single public function, ``genetic_algorithm``, which
implements all six steps of Algorithm 1:

1. **Population initialisation** – random variable-length subsets of
   fairness-aware preprocessing techniques drawn from the eight-element
   search space defined in ``practices``.
2. **Fitness evaluation** – each individual (ordered technique list) is
   scored by ``fitness.fitness`` using 5-fold cross-validation that measures
   both performance (PR-AUC) and fairness (SPD, EOD, DI).
3. **Selection** – the top 50 % of individuals ranked by fitness score are
   retained as parents for the next generation.
4. **Crossover** – single-point crossover between two parents with
   probability ``alpha``; the child copies one parent when crossover is
   skipped.
5. **Mutation** – one randomly chosen technique in the child is replaced by
   a different, not-yet-present technique with probability ``beta``.
6. **Duplicate removal** – order-preserving deduplication (``_unique_preserve_order``)
   ensures each technique appears at most once in an individual.

The algorithm is dataset-agnostic: datasets are passed as pre-processed
DataFrames; the classifier is specified by a string identifier and
instantiated inside ``fitness.fitness``.
"""
import logging
import random
import time
from typing import Iterable, Optional, Union

import numpy as np
import pandas as pd
from experiment_config import TECHNIQUES
from fitness import FitnessResult, fitness

logger = logging.getLogger(__name__)

#: An evaluated individual: (techniques, model, fitness, fairness score, performance score).
ScoredIndividual = tuple[list[str], str, float, Optional[float], Optional[float]]


# ---------------------------------------------------------------------------
# Algorithm 1 helper functions (one per step)
# ---------------------------------------------------------------------------

def _unique_preserve_order(seq: Iterable[str]) -> list[str]:
    """
    Remove duplicate entries while preserving insertion order.

    Role in Algorithm 1 (Steps 4–5 – after crossover and mutation):
        Ensures a child's chromosome contains no repeated technique token.
        Applying the same technique twice is semantically redundant, so
        duplicates introduced by tail-swapping during crossover are removed
        before the child enters the next generation.

    Parameters
    ----------
    seq : list
        Input sequence potentially containing duplicate elements.

    Returns
    -------
    list
        Deduplicated sequence in the order of first occurrence.
    """
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _initialise_population(techniques: list[str], population_size: int) -> list[list[str]]:
    """
    Generate the initial population of GA individuals (Algorithm 1, Step 1).

    Each individual is a random non-empty subset of *techniques*, with the
    subset size drawn uniformly from [1, len(techniques)].

    Parameters
    ----------
    techniques : list of str
        The full search space of available preprocessing technique names.
    population_size : int
        Number of individuals to generate.

    Returns
    -------
    list of list of str
        A list of *population_size* individuals; each individual is an ordered
        list of unique technique strings.
    """
    population = []
    for _ in range(population_size):
        k = np.random.randint(1, len(techniques) + 1)
        tech_subset = list(np.random.choice(techniques, size=k, replace=False))
        population.append(tech_subset)
    return population


def _unpack_fitness_result(fit_result: Union[FitnessResult, float], technique_list: list[str],
                           model: str) -> ScoredIndividual:
    """
    Normalise the return value of ``fitness.fitness`` into a standard 5-tuple.

    ``fitness`` may return a bare float (legacy/error path) or a
    ``(fitness_value, fairness_score, performance_score)`` tuple.

    Parameters
    ----------
    fit_result : float or tuple
        Raw return value from ``fitness.fitness``.
    technique_list : list of str
        The individual whose fitness was evaluated.
    model : str
        Classifier identifier.

    Returns
    -------
    tuple
        ``(technique_list, model, fitness_value, fairness_score, performance_score)``
    """
    if isinstance(fit_result, tuple) and len(fit_result) >= 1:
        fitness_value = fit_result[0]
        fairness_score = fit_result[1] if len(fit_result) > 1 else None
        performance_score = fit_result[2] if len(fit_result) > 2 else None
    else:
        fitness_value = fit_result
        fairness_score = None
        performance_score = None
    return (technique_list, model, fitness_value, fairness_score, performance_score)


def _evaluate_population(population: list[list[str]], dataset: pd.DataFrame,
                         protected_attribute: str, target_column: str, model: str, reset_cache: bool
                         ) -> list[ScoredIndividual]:
    """
    Evaluate the fitness of every individual in the current population (Algorithm 1, Step 2).

    Parameters
    ----------
    population : list of list of str
        Current generation of individuals (technique lists).
    dataset : pd.DataFrame
        Pre-processed dataset forwarded to ``fitness.fitness``.
    protected_attribute : str
    target_column : str
    model : str
        Classifier identifier.
    reset_cache : bool
        Forwarded unchanged to every ``fitness.fitness`` call.  When False,
        the ``reference/fate/fitness_cache.csv`` is used as an additional read-only
        lookup; when True, only the runtime cache is consulted.

    Returns
    -------
    list of tuple
        One 5-tuple per individual:
        ``(technique_list, model, fitness_value, fairness_score, performance_score)``.
    """
    fitness_scores = []
    for technique_list in population:
        fit_result = fitness(dataset.copy(), technique_list, model,
                             protected_attribute, target_column, reset_cache=reset_cache)
        fitness_scores.append(_unpack_fitness_result(fit_result, technique_list, model))
    return fitness_scores


def _select_parents(fitness_scores: list[ScoredIndividual], population_size: int
                    ) -> list[list[str]]:
    """
    Select the top 50 % of the population as parents for breeding (Algorithm 1, Step 3).

    Parameters
    ----------
    fitness_scores : list of tuple
        Evaluated population as returned by ``_evaluate_population``.
    population_size : int
        Current population size; determines how many parents are retained.

    Returns
    -------
    list of list of str
        Technique lists of the best individuals, sorted by fitness (ascending —
        lower fitness value first, consistent with the ``sorted`` call on the
        raw scores).
    """
    sorted_population = sorted(fitness_scores, key=lambda x: x[2], reverse=True)
    best_individuals = sorted_population[:max(1, population_size // 2)]
    return [ind[0] for ind in best_individuals]


def _apply_crossover(parent_a: list[str], parent_b: list[str], alpha: float) -> list[str]:
    """
    Perform single-point crossover between two parent individuals (Algorithm 1, Step 4).

    With probability *alpha*, chooses a random cut point in each parent and
    swaps the tails: child = parent_a[:cut_a] + parent_b[cut_b:].
    Otherwise the child is a copy of *parent_a*.

    Parameters
    ----------
    parent_a : list of str
        First parent technique list.
    parent_b : list of str
        Second parent technique list.
    alpha : float
        Crossover probability in [0, 1].

    Returns
    -------
    list of str
        Child individual (may contain duplicates; caller must deduplicate).
    """
    if np.random.rand() < alpha and len(parent_a) > 0 and len(parent_b) > 0:
        cut_a = np.random.randint(1, len(parent_a)) if len(parent_a) > 1 else 1
        cut_b = np.random.randint(1, len(parent_b)) if len(parent_b) > 1 else 1
        return parent_a[:cut_a] + parent_b[cut_b:]
    return list(parent_a)


def _apply_mutation(child: list[str], beta: float, techniques: list[str]) -> list[str]:
    """
    Randomly replace one technique in a child individual (Algorithm 1, Step 5).

    With probability *beta*, selects a random position in *child* and replaces
    the technique there with one not already present in *child*.  If every
    technique is already present, the child is returned unchanged.

    Parameters
    ----------
    child : list of str
        Child individual after crossover and duplicate removal.
    beta : float
        Mutation probability in [0, 1].
    techniques : list of str
        Full search-space pool from which replacement techniques are drawn.

    Returns
    -------
    list of str
        Potentially mutated child (same list, modified in-place logically).
    """
    if np.random.rand() < beta and len(child) > 0:
        replace_idx = random.randrange(len(child))
        available = [t for t in techniques if t not in child]
        if available:
            child[replace_idx] = random.choice(available)
    return child


def _breed_next_generation(best_techniques: list[list[str]], population_size: int, alpha: float,
                           beta: float, techniques: list[str]) -> list[list[str]]:
    """
    Produce the next generation via crossover, deduplication, and mutation
    (Algorithm 1, Steps 4–5).

    Parameters
    ----------
    best_techniques : list of list of str
        Parent individuals selected by ``_select_parents``.
    population_size : int
        Target size of the new generation.
    alpha : float
        Crossover probability forwarded to ``_apply_crossover``.
    beta : float
        Mutation probability forwarded to ``_apply_mutation``.
    techniques : list of str
        Full technique search space (needed for mutation fallback).

    Returns
    -------
    list of list of str
        New population of *population_size* individuals.
    """
    next_generation = []
    while len(next_generation) < population_size:
        parents = random.sample(best_techniques, 2 if len(best_techniques) > 1 else 1)
        parent_a = parents[0]
        parent_b = parents[1] if len(parents) > 1 else parents[0]

        child = _apply_crossover(parent_a, parent_b, alpha)
        child = _unique_preserve_order(child)
        if len(child) == 0:
            child = [random.choice(techniques)]

        child = _apply_mutation(child, beta, techniques)
        next_generation.append(child)
    return next_generation


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def genetic_algorithm(dataset: pd.DataFrame, protected_attribute: str, target_column: str,
                      model: Optional[str], generations: int = 10, population_size: int = 10,
                      alpha: float = 0.5, beta: float = 0.5, reset_cache: bool = False
                      ) -> ScoredIndividual:
    """
    Execute the FATE genetic algorithm to find an optimal fairness-aware preprocessing pipeline.

    Implements all six steps of Algorithm 1 end-to-end by delegating each
    step to a dedicated helper function.

    Parameters
    ----------
    dataset : pd.DataFrame
        Pre-processed dataset (features + target + protected attribute) as
        returned by ``preprocessing.prepare_data_model``.
    protected_attribute : str
        Name of the protected attribute column present in *dataset*.
        Forwarded to ``fitness.fitness`` for fairness metric computation.
    target_column : str
        Name of the label column in *dataset*.
    model : str
        Classifier identifier forwarded to ``fitness.fitness``.  One of:
        ``'lr'`` (Logistic Regression), ``'rf'`` (Random Forest),
        ``'svc'`` (LinearSVC), ``'xgb'`` (XGBoost).
    generations : int, optional
        Number of GA generations (default 10).  Corresponds to the outer loop
        counter in Algorithm 1.
    population_size : int, optional
        Number of individuals per generation (default 10).  Determines search
        breadth at each step.
    alpha : float, optional
        Single-point crossover probability in [0, 1] (default 0.5).
        Higher values increase recombination between parents (Algorithm 1,
        Step 4).
    beta : float, optional
        Mutation probability in [0, 1] (default 0.5).  Higher values increase
        exploration of the technique space (Algorithm 1, Step 5).
    reset_cache : bool, optional
        When False (default), fitness lookups also check the read-only root
        cache ``reference/fate/fitness_cache.csv`` before evaluating.  When True, only
        the runtime cache ``results/fate/runtime_cache.csv`` is used — useful
        for runs that must not reuse pre-computed results.

    Returns
    -------
    tuple
        Best individual found in the final generation as a 5-tuple:
        ``(techniques, model, fitness_value, fairness_score, performance_score)``

        - ``techniques`` (list[str]): ordered list of techniques in the
          best individual's chromosome.
        - ``model`` (str): echoes the *model* argument.
        - ``fitness_value`` (float): combined fitness
          ``perf_weight * PS − fair_weight * FS``.
        - ``fairness_score`` (float or None): mean normalised fairness
          deviation across CV folds (lower is better).
        - ``performance_score`` (float or None): mean PR-AUC across CV
          folds (higher is better).

    Notes
    -----
    Search space *T* consists of eight technique strings:
    ``'standard'``, ``'stratified_sampling'``, ``'oversampling'``,
    ``'undersampling'``, ``'clustering'``, ``'ipw'``, ``'matching'``,
    ``'min_max_scaling'``.

    Individuals are variable-length ordered lists of unique technique names
    (subsets of *T* with at least one element).

    The best solution is selected from the **last evaluated generation's**
    population, not from a global hall-of-fame across all generations.  This
    is the intended design: later generations are expected to be fitter on
    average, so the final generation serves as a sufficient proxy for the
    overall best.
    """
    if model is None:
        raise ValueError("Provide a model identifier (string) to evaluate; "
                         "the model is not part of the GA individuals.")

    techniques = list(TECHNIQUES)

    population = _initialise_population(techniques, population_size)
    fitness_scores = []

    for generation in range(generations):
        gen_start = time.time()
        logger.info("Generation %d/%d  |  %d individuals",
                    generation + 1, generations, len(population))
        fitness_scores = _evaluate_population(population, dataset, protected_attribute,
                                              target_column, model, reset_cache)
        best_techniques = _select_parents(fitness_scores, population_size)
        population = _breed_next_generation(best_techniques, population_size,
                                            alpha, beta, techniques)
        gen_elapsed = time.time() - gen_start
        valid_fits = [s[2] for s in fitness_scores if s[2] is not None and s[2] != float('inf')]
        best_fit = max(valid_fits) if valid_fits else float('nan')
        logger.info("Generation %d/%d done  |  %.1fs  |  best fitness %.4f",
                    generation + 1, generations, gen_elapsed, best_fit)

    best_solution = max(fitness_scores, key=lambda x: x[2])
    logger.info(
        "Best pipeline: %s  |  model=%s  |  fitness=%.4f  |  perf=%.4f  |  fair=%.4f",
        best_solution[0], best_solution[1], best_solution[2],
        best_solution[4] if best_solution[4] is not None else float('nan'),
        best_solution[3] if best_solution[3] is not None else float('nan'),
    )
    return best_solution
