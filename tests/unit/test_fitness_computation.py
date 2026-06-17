# tests/unit/test_fitness_computation.py
"""
Unit tests for FATE fitness value computation.

Tests target ``fitness._compute_combined_fitness``, which implements the final
aggregation step of Algorithm 1 Step 2:

    fitness = perf_weight × PS − fair_weight × FS

where PS = mean PR-AUC across folds and FS = mean fairness sum / 3.
"""
import numpy as np

from fitness import _compute_combined_fitness


class TestComputeCombinedFitness:
    """Tests for ``_compute_combined_fitness``."""

    def test_basic_formula_correctness(self):
        """
        Verify the fitness formula with hand-computable inputs.

        With equal weights (0.5 / 0.5), perf_scores [0.8, 0.9], and
        fair_scores [0.6, 0.6]:
          PS  = mean([0.8, 0.9]) = 0.85
          FS  = mean([0.6, 0.6]) / 3 = 0.2
          fit = 0.5 × 0.85 − 0.5 × 0.2 = 0.325

        This exercises the core formula and confirms the fairness divisor of 3
        (one per metric: SPD, EOD, DI).
        """
        fit, fair, perf = _compute_combined_fitness(
            perf_scores=[0.8, 0.9],
            fair_scores=[0.6, 0.6],
            perf_weight=0.5,
            fair_weight=0.5,
        )
        np.testing.assert_allclose(perf, 0.85, rtol=1e-9)
        np.testing.assert_allclose(fair, 0.2,  rtol=1e-9)
        np.testing.assert_allclose(fit,  0.325, rtol=1e-9)

    def test_performance_only_weight(self):
        """
        With fair_weight=0, the fitness equals the performance score alone.

        This edge case verifies that the fairness term vanishes when its
        weight is zero, so a purely performance-oriented configuration still
        produces sensible output.
        """
        fit, fair, perf = _compute_combined_fitness(
            perf_scores=[0.7],
            fair_scores=[0.9],
            perf_weight=1.0,
            fair_weight=0.0,
        )
        np.testing.assert_allclose(perf, 0.7, rtol=1e-9)
        np.testing.assert_allclose(fit,  0.7, rtol=1e-9)

    def test_fairness_only_weight(self):
        """
        With perf_weight=0 the fitness is −fair_weight × FS (should be negative).

        Confirms the subtraction sign in the formula: maximum fairness deviation
        with perf_weight=0 produces a negative fitness, signalling a very poor
        individual.
        """
        fit, fair, perf = _compute_combined_fitness(
            perf_scores=[0.9],
            fair_scores=[0.9],   # FS = 0.9 / 3 = 0.3
            perf_weight=0.0,
            fair_weight=1.0,
        )
        np.testing.assert_allclose(fair, 0.3, rtol=1e-9)
        np.testing.assert_allclose(fit, -0.3, rtol=1e-9)

    def test_empty_fairness_scores_defaults_to_zero(self):
        """
        When no folds contribute a fairness measurement, FS defaults to 0.0.

        This can happen when every CV fold raises a fairness metric exception.
        The fitness function must not crash and should treat the individual as
        having zero fairness cost (conservative assumption).
        """
        fit, fair, perf = _compute_combined_fitness(
            perf_scores=[0.8],
            fair_scores=[],
            perf_weight=0.5,
            fair_weight=0.5,
        )
        assert fair == 0.0
        np.testing.assert_allclose(fit, 0.5 * 0.8, rtol=1e-9)

    def test_multi_fold_averaging(self):
        """
        Verify that performance and fairness are averaged across all folds.

        Using four folds with different scores confirms that ``nanmean`` (not
        ``sum`` or a fold-weighted scheme) is applied to each list.
        """
        fit, fair, perf = _compute_combined_fitness(
            perf_scores=[0.5, 0.7, 0.9, 0.5],
            fair_scores=[0.3, 0.3, 0.3, 0.3],
            perf_weight=0.5,
            fair_weight=0.5,
        )
        expected_perf = np.mean([0.5, 0.7, 0.9, 0.5])   # 0.65
        expected_fair = np.mean([0.3, 0.3, 0.3, 0.3]) / 3  # 0.1
        expected_fit = 0.5 * expected_perf - 0.5 * expected_fair
        np.testing.assert_allclose(perf, expected_perf, rtol=1e-9)
        np.testing.assert_allclose(fair, expected_fair, rtol=1e-9)
        np.testing.assert_allclose(fit, expected_fit, rtol=1e-9)

    def test_fitness_is_float(self):
        """
        The return values are plain Python floats, not numpy scalars.

        Downstream code (e.g., CSV writers) expects serialisable Python floats;
        returning numpy floats could cause subtle serialisation errors.
        """
        fit, fair, perf = _compute_combined_fitness([0.8], [0.2], 0.5, 0.5)
        assert isinstance(fit,  float)
        assert isinstance(fair, float)
        assert isinstance(perf, float)
