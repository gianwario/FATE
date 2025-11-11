# genetic_algorithm.py
import numpy as np
from fitness import fitness
import os
import random


def genetic_algorithm(dataset, protected_attribute, target_column, model, generations=10,
                      population_size=10, alpha=0.7, beta=0.1):
    """
    Run a genetic algorithm to optimize preprocessing techniques for a given ML model.

    Args:
        dataset (pd.DataFrame): The dataset to optimize.
        protected_attribute (str): The name of the protected attribute column.
        target_column (str): The name of the target column.
        model (str): The ML model identifier to evaluate (passed to fitness()).
        generations (int): Number of generations to run the algorithm.
        population_size (int): Size of the population in each generation.
        alpha (float): Probability of performing single-point crossover between two parents.
        beta (float): Probability of replacing a technique in the individual with another one.

    Returns:
        tuple: Best solution found by the genetic algorithm.
    """

    if model is None:
        raise ValueError("Provide a model identifier (string) to evaluate; the model is not part of the GA individuals.")

    # Define potential techniques for optimization
    techniques = [
        'onehot_standard', 'stratified_sampling', 'oversampling', 'undersampling',
        'clustering', 'ipw', 'matching', 'min_max_scaling'
    ]


    # Generate random population: each individual is a random subset of techniques
    population = []
    for _ in range(population_size):
        k = np.random.randint(1, len(techniques) + 1)  # random subset size at least 1
        tech_subset = list(np.random.choice(techniques, size=k, replace=False))
        population.append(tech_subset)

    best_solution = None

    # helper to make list unique preserving order
    def _unique_preserve_order(seq):
        seen = set()
        out = []
        for x in seq:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    # Iterate through each generation
    for generation in range(generations):
        print(f"Generation {generation + 1}/{generations} start")
        # Evaluate fitness of each individual (technique list) using the fixed model
        fitness_scores = [
                (technique_list, model, fitness(dataset.copy(), technique_list, model, protected_attribute, target_column))
                for technique_list in population
            ]


        # Sort the population by fitness scores
        sorted_population = sorted(fitness_scores, key=lambda x: x[2])
        # Select the top half of the population for breeding
        best_individuals = sorted_population[:max(1, population_size // 2)]

        best_techniques = [ind[0] for ind in best_individuals]

        next_generation = []
        # Create the next generation by combining techniques from the best individuals
        while len(next_generation) < population_size:
            # Select one or two parents at random from the best individuals
            parents = random.sample(best_techniques, 2 if len(best_techniques) > 1 else 1)
            if len(parents) == 1:
                parent_a = parent_b = parents[0]
            else:
                parent_a, parent_b = parents

            # Single-point crossover with probability alpha
            if np.random.rand() < alpha and len(parent_a) > 0 and len(parent_b) > 0:
                # choose cut points (if length==1, cut at 1 returns empty slice for prefix)
                cut_a = np.random.randint(1, len(parent_a)) if len(parent_a) > 1 else 1
                cut_b = np.random.randint(1, len(parent_b)) if len(parent_b) > 1 else 1

                # swap tails after the cut
                child = parent_a[:cut_a] + parent_b[cut_b:]
            else:
                # no crossover: child is a copy of one parent
                child = list(parent_a)

            # ensure uniqueness of techniques and at least one technique
            child = _unique_preserve_order(child)
            if len(child) == 0:
                child = [random.choice(techniques)]

            # Mutation: with probability beta, replace one practice randomly
            if np.random.rand() < beta and len(child) > 0:
                # choose a position to replace
                replace_idx = random.randrange(len(child))
                # choose a new technique not already in the child
                available = [t for t in techniques if t not in child]
                if available:
                    child[replace_idx] = random.choice(available)
                else:
                    # if all techniques are already present, do nothing (or optionally shuffle)
                    pass

            next_generation.append(child)
        population = next_generation


        print(f"End of generation {generation + 1}")

    # Select the best solution based on the lowest fitness score
    best_solution = min(fitness_scores, key=lambda x: x[2])
    print(f"Best solution: Techniques={best_solution[0]}, Model={best_solution[1]}, Fitness={best_solution[2]}")
    return best_solution

