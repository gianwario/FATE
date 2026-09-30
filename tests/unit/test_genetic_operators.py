# tests/unit/test_genetic_operators.py
"""
Unit tests for the FATE genetic operators (Algorithm 1 Steps 3–6).

Each test targets one of the four primitive GA operations and verifies
structural invariants that must hold regardless of the random seed:

- Step 3  Selection        ``_select_parents``
- Step 4  Crossover        ``_apply_crossover``
- Step 5  Mutation         ``_apply_mutation``
- Step 6  Duplicate removal ``_unique_preserve_order``

These invariants are algorithm-level properties that, if violated, would silently
corrupt the evolutionary search without raising any Python error.

Selection direction
-------------------
``_select_parents`` sorts individuals by fitness DESCENDING and retains the
FIRST half (the highest-fitness individuals).  Because FATE's fitness =
perf_weight × PS − fair_weight × FS, higher fitness means a better solution.
"""
import random
import numpy as np

from genetic_algorithm import (
    _unique_preserve_order,
    _select_parents,
    _apply_crossover,
    _apply_mutation,
)

VALID_TECHNIQUES = [
    'standard', 'stratified_sampling', 'oversampling', 'undersampling',
    'clustering', 'ipw', 'matching', 'min_max_scaling',
]


# ---------------------------------------------------------------------------
# Step 6 — Duplicate removal
# ---------------------------------------------------------------------------

class TestUniquePreserveOrder:
    def test_removes_duplicates(self: "TestUniquePreserveOrder") -> None:
        """
        Each technique must appear at most once in the output.

        An individual whose chromosome contains duplicates is semantically
        equivalent to one without; applying the same technique twice wastes
        computation and wastes a gene slot that could carry a different technique.
        """
        result = _unique_preserve_order(['standard', 'oversampling', 'standard', 'clustering'])
        assert result == ['standard', 'oversampling', 'clustering']

    def test_preserves_insertion_order(self: "TestUniquePreserveOrder") -> None:
        """
        The first occurrence of each element must appear in its original relative order.

        GA crossover concatenates the head of parent_a with the tail of parent_b.
        The relative ordering encodes the application sequence, so reordering
        would silently change the transformation pipeline.
        """
        seq = ['clustering', 'ipw', 'clustering', 'matching', 'ipw']
        result = _unique_preserve_order(seq)
        assert result == ['clustering', 'ipw', 'matching']

    def test_empty_input_returns_empty(self: "TestUniquePreserveOrder") -> None:
        """
        An empty sequence must return an empty list without raising an error.

        The mutation step can produce empty children in edge cases; duplicate
        removal must not crash on them.
        """
        assert _unique_preserve_order([]) == []

    def test_already_unique_unchanged(self: "TestUniquePreserveOrder") -> None:
        """
        A sequence with no duplicates must be returned unchanged (by value).

        This is the common case in a healthy population; the function must not
        accidentally reorder or truncate unique sequences.
        """
        seq = ['standard', 'oversampling', 'clustering']
        assert _unique_preserve_order(seq) == seq

    def test_all_duplicates_single_element(self: "TestUniquePreserveOrder") -> None:
        """
        A sequence of the same element repeated N times collapses to length 1.

        Guards against the edge case where crossover produces a chromosome
        consisting entirely of one repeated technique.
        """
        assert _unique_preserve_order(['ipw', 'ipw', 'ipw']) == ['ipw']


# ---------------------------------------------------------------------------
# Step 3 — Selection
# ---------------------------------------------------------------------------

class TestSelectParents:
    def _make_scores(self: 'TestSelectParents', fitness_values: list[float]
                     ) -> list[tuple[list[str], str, float, float, float]]:
        """Build a fitness_scores list from a flat list of fitness values."""
        return [
            ([f'tech_{i}'], 'lr', fv, 0.1, 0.8)
            for i, fv in enumerate(fitness_values)
        ]

    def test_output_is_subset_of_input_techniques(self: "TestSelectParents") -> None:
        """
        Every technique list in the parent pool must have come from the input population.

        If selection introduces a technique list that was not in the evaluated
        population, the subsequent crossover and mutation steps would operate on
        phantom individuals that have never been fitness-evaluated.
        """
        scores = self._make_scores([0.5, 0.3, 0.8, 0.1])
        input_techniques = {tuple(s[0]) for s in scores}
        parents = _select_parents(scores, population_size=4)
        for p in parents:
            assert tuple(p) in input_techniques, (
                f"Parent {p} was not present in the evaluated population"
            )

    def test_selects_half_of_population(self: "TestSelectParents") -> None:
        """
        Exactly population_size // 2 parents must be selected.

        FATE uses a fixed 50 % elitism rate; deviating from this changes the
        selection pressure and population diversity in untested ways.
        """
        scores = self._make_scores([0.1, 0.4, 0.7, 0.9])
        parents = _select_parents(scores, population_size=4)
        assert len(parents) == 2

    def test_selects_at_least_one_for_small_population(self: "TestSelectParents") -> None:
        """
        At least one parent must be selected even when population_size == 1.

        The ``max(1, population_size // 2)`` guard prevents an empty parent pool
        which would crash ``_breed_next_generation``'s ``random.sample`` call.
        """
        scores = self._make_scores([0.5])
        parents = _select_parents(scores, population_size=1)
        assert len(parents) >= 1

    def test_selection_picks_highest_fitness(self: "TestSelectParents") -> None:
        """
        ``_select_parents`` must retain the individuals with the HIGHEST fitness.

        Since fitness = perf_weight × PS − fair_weight × FS, a higher value
        means better performance and/or lower fairness deviation.  The GA must
        breed from the best half, not the worst half, to apply positive
        evolutionary pressure.
        """
        # fitness_values: 0.9 is best, 0.1 is worst
        scores = self._make_scores([0.9, 0.6, 0.4, 0.1])
        parents = _select_parents(scores, population_size=4)
        parent_techniques = [tuple(p) for p in parents]
        assert ('tech_0',) in parent_techniques, (
            "Best individual (fitness=0.9) must be selected as a parent"
        )
        assert ('tech_1',) in parent_techniques, (
            "Second-best individual (fitness=0.6) must be selected as a parent"
        )


# ---------------------------------------------------------------------------
# Step 4 — Crossover
# ---------------------------------------------------------------------------

class TestApplyCrossover:
    # Disjoint parents so we can detect which parent contributed each element
    PARENT_A = ['standard', 'oversampling', 'clustering']     # {A-set}
    PARENT_B = ['ipw', 'matching', 'min_max_scaling']          # {B-set}

    def test_no_crossover_returns_copy_of_parent_a(self: "TestApplyCrossover") -> None:
        """
        With alpha=0.0 (crossover never fires), the child must be an independent
        copy of parent_a.

        Any crossover with probability 0 would indicate that the random draw is
        not being compared correctly against alpha (e.g., < vs <=).
        """
        child = _apply_crossover(self.PARENT_A, self.PARENT_B, alpha=0.0)
        assert child == self.PARENT_A
        # Must be a new list object (not the same reference)
        assert child is not self.PARENT_A

    def test_full_crossover_child_elements_from_valid_parents(self: "TestApplyCrossover") -> None:
        """
        With alpha=1.0 (crossover always fires), every element of the child
        must come from parent_a OR parent_b — never from outside.

        This invariant holds regardless of where the random cut points land.
        A violation would indicate that new technique names are being introduced
        during crossover, corrupting the chromosome.
        """
        valid = set(self.PARENT_A) | set(self.PARENT_B)
        for _ in range(20):   # multiple draws to cover different random cuts
            child = _apply_crossover(self.PARENT_A, self.PARENT_B, alpha=1.0)
            for elem in child:
                assert elem in valid, (
                    f"Child element '{elem}' is not from either parent"
                )

    def test_crossover_child_is_nonempty(self: "TestApplyCrossover") -> None:
        """
        The child produced by crossover must be non-empty.

        An empty chromosome cannot be evaluated by the fitness function and
        would cause ``_apply_mutation`` and ``_breed_next_generation`` to crash.
        ``_breed_next_generation`` adds a fallback random element when the child
        is empty, but we verify the crossover output itself is not empty when
        both parents are non-empty.

        Note: for single-element parents, cut_b = 1 forces parent_b[1:] = [],
        so the child is only parent_a[:1].  This is still non-empty.
        """
        child = _apply_crossover(['standard'], ['ipw'], alpha=1.0)
        assert len(child) >= 1

    def test_crossover_with_single_element_parents(self: "TestApplyCrossover") -> None:
        """
        Crossover on single-element parents must not raise an IndexError.

        When len(parent) == 1, ``randint(1, len(parent))`` would call
        ``randint(1, 1)`` which would fail on some numpy versions.
        The implementation guards this with a conditional ``if len > 1 else 1``.
        """
        child = _apply_crossover(['standard'], ['ipw'], alpha=1.0)
        assert isinstance(child, list)


# ---------------------------------------------------------------------------
# Step 5 — Mutation
# ---------------------------------------------------------------------------

class TestApplyMutation:
    def test_no_mutation_when_beta_zero(self: "TestApplyMutation") -> None:
        """
        With beta=0.0, the child must be returned identical to its input.

        Any mutation at probability 0 indicates that the random draw is not
        being compared against beta correctly.
        """
        child = ['standard', 'oversampling']
        original = list(child)
        result = _apply_mutation(child, beta=0.0, techniques=VALID_TECHNIQUES)
        assert result == original

    def test_mutation_replaces_one_technique(self: "TestApplyMutation") -> None:
        """
        With beta=1.0 and available alternatives, exactly one position changes.

        The mutated technique must differ from the original at the replacement
        index.  Changing more than one position would indicate a loop instead of
        a single-point replacement.
        """
        np.random.seed(7)
        random.seed(7)
        child = ['standard']      # only one position; must change to something else
        result = _apply_mutation(list(child), beta=1.0, techniques=VALID_TECHNIQUES)
        assert result != child, "Mutation with beta=1.0 must change the child"
        assert len(result) == 1, "Mutation must not change the chromosome length"

    def test_mutated_technique_stays_in_valid_set(self: "TestApplyMutation") -> None:
        """
        The replacement technique must be a member of the valid search space T.

        Introducing a technique name that does not exist in T would cause
        ``apply_techniques`` to silently return the data unchanged (no-op),
        wasting a gene slot without raising an error.
        """
        for seed in range(10):
            np.random.seed(seed)
            random.seed(seed)
            child = ['standard']
            result = _apply_mutation(list(child), beta=1.0, techniques=VALID_TECHNIQUES)
            for t in result:
                assert t in VALID_TECHNIQUES, f"'{t}' is not a valid technique"

    def test_no_mutation_when_all_techniques_already_present(self: "TestApplyMutation") -> None:
        """
        When the child already contains all techniques in T, mutation is a no-op.

        No available alternatives exist, so the child is returned unchanged.
        This prevents an infinite loop or a crash when ``random.choice([])`` is
        called on an empty list.
        """
        full_child = list(VALID_TECHNIQUES)
        result = _apply_mutation(list(full_child), beta=1.0, techniques=VALID_TECHNIQUES)
        assert result == full_child

    def test_mutation_never_introduces_duplicate_in_single_technique_child(self: 'TestApplyMutation'
                                                                           ) -> None:
        """
        The replacement technique must not already be present in the child.

        ``_apply_mutation`` filters available techniques with
        ``[t for t in techniques if t not in child]``.  If 'standard' is the
        only element, the replacement must be one of the other seven techniques.
        """
        for seed in range(20):
            np.random.seed(seed)
            random.seed(seed)
            child = ['standard']
            result = _apply_mutation(list(child), beta=1.0, techniques=VALID_TECHNIQUES)
            assert len(set(result)) == len(result), (
                f"Mutation introduced a duplicate: {result}"
            )
            assert result[0] != 'standard' or len(VALID_TECHNIQUES) == 1
