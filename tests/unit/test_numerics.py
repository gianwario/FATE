# tests/unit/test_numerics.py
"""
Unit tests for the floating-point policy (``numerics.py``) and for the explicit
handling of undefined fairness values and infeasible pipelines in ``fitness.py``.

The whole suite runs with ``RuntimeWarning`` promoted to an error (``pytest.ini``),
so every test below also checks that no numerical warning is emitted.
"""
import numpy as np
import pandas as pd
import pytest

import fitness
import genetic_algorithm
import main
import numerics
from fitness import INFEASIBLE, _aggregate_fold_fairness, _make_simple_key, fairness_metrics
from numerics import NonFiniteResultError, require_finite, strict_floating_point
from RQ2_data_analysis import preprocessing_experiments as rq2


def _fold(sex: list[int], target: list[int]) -> pd.DataFrame:
    """Small test fold with a binary ``sex`` attribute."""
    return pd.DataFrame({'f0': np.zeros(len(sex)), 'sex': sex, 'target': target})


# ---------------------------------------------------------------------------
# numerics.py
# ---------------------------------------------------------------------------

class TestFloatingPointPolicy:
    def test_strict_mode_raises_on_division_by_zero(self: "TestFloatingPointPolicy") -> None:
        """Inside ``strict_floating_point`` an invalid operation raises instead of yielding inf."""
        with strict_floating_point(), pytest.raises(FloatingPointError):
            _ = np.float64(1.0) / np.float64(0.0)

    def test_blas_errstate_is_strict_on_reliable_backends(
            self: "TestFloatingPointPolicy", monkeypatch: pytest.MonkeyPatch) -> None:
        """On OpenBLAS (Linux, Windows, Intel macOS) BLAS-heavy steps stay strict."""
        monkeypatch.setattr(numerics, 'BLAS_FLAGS_UNRELIABLE', False)
        with numerics.blas_errstate(), pytest.raises(FloatingPointError):
            _ = np.float64(1.0) / np.float64(0.0)

    def test_blas_errstate_relaxed_on_accelerate(
            self: "TestFloatingPointPolicy", monkeypatch: pytest.MonkeyPatch) -> None:
        """On Accelerate the (unreliable) flags are ignored; outputs are checked instead."""
        monkeypatch.setattr(numerics, 'BLAS_FLAGS_UNRELIABLE', True)
        with numerics.blas_errstate():
            value = np.float64(1.0) / np.float64(0.0)
        with pytest.raises(NonFiniteResultError):
            require_finite(np.array([value]), 'scores')

    def test_require_finite(self: "TestFloatingPointPolicy") -> None:
        """Finite data passes; inf/NaN raise; non-numeric columns are ignored."""
        require_finite(pd.DataFrame({'a': [1.0, 2.0], 'b': ['x', 'y']}), 'frame')
        for bad in ([1.0, np.inf], [np.nan]):
            with pytest.raises(NonFiniteResultError):
                require_finite(np.array(bad), 'array')


# ---------------------------------------------------------------------------
# Undefined fairness metrics
# ---------------------------------------------------------------------------

class TestUndefinedFairness:
    def test_unbounded_disparate_impact(self: "TestUndefinedFairness") -> None:
        """
        Privileged group without positive predictions: DI = x/0 is ``inf``.

        No FloatingPointError is raised (the ratio is evaluated with relaxed
        checks) and no RuntimeWarning is emitted.
        """
        df = _fold(sex=[1, 1, 1, 0, 0, 0], target=[1, 0, 0, 1, 0, 0])
        with strict_floating_point():
            result = fairness_metrics(df, pd.Index(range(6)), 'sex',
                                      np.array([0, 0, 0, 1, 0, 0]), 'target')
        assert result['disparate_impact'] == np.inf
        assert _aggregate_fold_fairness(result) == np.inf

    def test_zero_over_zero_disparate_impact(self: "TestUndefinedFairness") -> None:
        """No positive predictions at all: DI = 0/0 is NaN and the fold's FS is undefined."""
        df = _fold(sex=[1, 1, 1, 0, 0, 0], target=[1, 0, 0, 1, 0, 0])
        with strict_floating_point():
            result = fairness_metrics(df, pd.Index(range(6)), 'sex', np.zeros(6, dtype=int),
                                      'target')
        assert np.isnan(result['disparate_impact'])
        assert np.isnan(_aggregate_fold_fairness(result))

    def test_defined_fold_sums_the_three_metrics(self: "TestUndefinedFairness") -> None:
        """A fold with defined metrics contributes |SPD| + |EOD| + |DI|."""
        value = _aggregate_fold_fairness({'statistical_parity': 0.1, 'equal_opportunity': 0.2,
                                          'disparate_impact': 0.3})
        np.testing.assert_allclose(value, 0.6)


# ---------------------------------------------------------------------------
# Infeasible pipelines, exceptions and cache
# ---------------------------------------------------------------------------

class TestInfeasibleAndErrors:
    def test_single_class_target_is_infeasible(
            self: "TestInfeasibleAndErrors", small_binary_dataset: pd.DataFrame) -> None:
        """A dataset with one class cannot be scored: fitness is INFEASIBLE (-inf), not +inf."""
        data = small_binary_dataset.assign(target=1)
        fit, fs, ps = fitness.fitness(data, ['standard'], 'lr', 'sex', 'target', reset_cache=True)
        assert fit == INFEASIBLE and fs is None and ps is None

    def test_failing_practice_is_not_skipped(
            self: "TestInfeasibleAndErrors", small_binary_dataset: pd.DataFrame,
            monkeypatch: pytest.MonkeyPatch) -> None:
        """An exception in a practice aborts the evaluation instead of being swallowed."""
        def _fail(data: pd.DataFrame, technique: str, protected_attribute: str) -> pd.DataFrame:
            raise ValueError("practice failed")
        monkeypatch.setattr(fitness, 'apply_techniques', _fail)
        with pytest.raises(ValueError, match="practice failed"):
            fitness.fitness(small_binary_dataset, ['standard'], 'lr', 'sex', 'target',
                            reset_cache=True)

    def test_infeasible_individuals_rank_last(self: "TestInfeasibleAndErrors") -> None:
        """Parent selection keeps the best half; INFEASIBLE individuals are never preferred."""
        scored = [(['a'], 'lr', INFEASIBLE, None, None), (['b'], 'lr', 0.2, 0.1, 0.5),
                  (['c'], 'lr', 0.4, 0.1, 0.9), (['d'], 'lr', INFEASIBLE, None, None)]
        assert genetic_algorithm._select_parents(scored, 4) == [['c'], ['b']]

    def test_cache_key_keeps_the_order_of_practices(self: "TestInfeasibleAndErrors") -> None:
        """Practices are applied in sequence, so two orders are two different pipelines."""
        a = _make_simple_key('lr', 'sex', 'target', ['oversampling', 'undersampling'])
        b = _make_simple_key('lr', 'sex', 'target', ['undersampling', 'oversampling'])
        assert a != b

    def test_archived_nan_fitness_is_read_as_infeasible(self: "TestInfeasibleAndErrors") -> None:
        """A NaN fitness in a cache file never reaches the GA."""
        fit, fs, ps = fitness._parse_cache_row_metrics(
            {'fitness': 'nan', 'fairness': 'nan', 'performance': '0.5'})
        assert fit == INFEASIBLE and np.isnan(fs) and ps == 0.5

    def test_corrupt_cache_row_raises(self: "TestInfeasibleAndErrors") -> None:
        """A non-numeric cache field is an error, not a silently ignored value."""
        with pytest.raises(ValueError):
            fitness._parse_cache_row_metrics({'fitness': 'abc', 'fairness': '', 'performance': ''})

    def test_numerical_error_fails_only_the_run(
            self: "TestInfeasibleAndErrors", monkeypatch: pytest.MonkeyPatch) -> None:
        """A FloatingPointError in a GA run becomes an error row (written to errors.log)."""
        def _raise(*args: object, **kwargs: object) -> None:
            raise FloatingPointError("overflow encountered")
        monkeypatch.setattr(main, '_run_timed_fate', _raise)
        rows = main.execute_fate(pd.DataFrame(), 'd', 'd.csv', 'sex', 'target', ['lr'],
                                 5, 5, 0.5, 0.5, summary_path=None)
        assert rows[0]['error'] == "FloatingPointError: overflow encountered"
        assert rows[0]['fitness'] is None

    def test_unexpected_error_is_not_caught(
            self: "TestInfeasibleAndErrors", monkeypatch: pytest.MonkeyPatch) -> None:
        """Any other exception is a bug and propagates."""
        def _raise(*args: object, **kwargs: object) -> None:
            raise RuntimeError("bug")
        monkeypatch.setattr(main, '_run_timed_fate', _raise)
        with pytest.raises(RuntimeError):
            main.execute_fate(pd.DataFrame(), 'd', 'd.csv', 'sex', 'target', ['lr'],
                              5, 5, 0.5, 0.5, summary_path=None)


# ---------------------------------------------------------------------------
# RQ2 baselines: aggregation of undefined metrics
# ---------------------------------------------------------------------------

class TestBaselineAggregation:
    def test_mean_over_defined_folds(self: "TestBaselineAggregation") -> None:
        """Undefined (NaN) folds are skipped; all-NaN is NaN; an unbounded fold gives inf."""
        np.testing.assert_allclose(rq2._mean_over_defined_folds([0.2, np.nan, 0.4]), 0.3)
        assert np.isnan(rq2._mean_over_defined_folds([np.nan, np.nan]))
        assert rq2._mean_over_defined_folds([0.2, np.inf]) == np.inf
