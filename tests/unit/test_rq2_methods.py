# tests/unit/test_rq2_methods.py
"""
Unit tests for the methods behind RQ2: the baseline mitigation techniques
(``preprocessing_experiments.py``) and the statistical procedures of Table 4
(``rq2_results.py``).  Plotting and file-handling code is not tested.
"""
import numpy as np
import pandas as pd
import pytest

import paths
from RQ2_data_analysis import preprocessing_experiments as rq2
from RQ2_data_analysis.rq2_results import (holm_adjust, run_tests, sign_test,
                                           symmetry_test, vargha_delaney_a12)

DS_CFG = {'name': 'synthetic', 'target': 'target'}


# ---------------------------------------------------------------------------
# Baseline techniques
# ---------------------------------------------------------------------------

class TestBaselines:
    def test_fairness_score_is_mean_of_metrics(self: "TestBaselines") -> None:
        """FS of a baseline uses the same definition as FATE: (|SPD| + |EOD| + |DI|) / 3."""
        np.testing.assert_allclose(rq2.compute_fairness_score_from_metrics(0.1, 0.2, 0.3), 0.2)

    def test_binarize_protected_for_fairsmote(self: "TestBaselines") -> None:
        """Age is split at the mean; other attributes use the majority value as privileged."""
        age = rq2.binarize_protected_for_fairsmote(pd.Series([20, 30, 40, 70]), 'age')
        sex = rq2.binarize_protected_for_fairsmote(pd.Series([0, 1, 1, 1]), 'sex')
        assert list(age) == [0, 0, 0, 1]
        assert list(sex) == [0, 1, 1, 1]

    def test_fairsmote_balances_the_four_quadrants(self: "TestBaselines") -> None:
        """Every (label, group) quadrant is oversampled to the size of the largest one."""
        x = pd.DataFrame({'f0': range(10)})
        y = pd.Series([0, 0, 0, 0, 0, 0, 1, 1, 0, 1])
        s = pd.Series([0, 0, 0, 0, 1, 1, 0, 1, 1, 1])
        x_tr, y_tr = rq2._fairsmote_training_data(x, y, s, 'sex')
        quadrants = pd.crosstab(y_tr, x_tr['f0'].map(dict(zip(x['f0'], s))))
        assert (quadrants.to_numpy() == 4).all()
        assert list(x_tr.columns) == ['f0']

    def test_fairsmote_rejects_non_binary_labels(self: "TestBaselines") -> None:
        """FairSMOTE requires binary labels."""
        with pytest.raises(ValueError):
            rq2._fairsmote_training_data(pd.DataFrame({'f0': [1, 2, 3]}), pd.Series([0, 1, 2]),
                                         pd.Series([0, 1, 0]), 'sex')

    @pytest.mark.parametrize('method', ['fairsmote', 'reweighing', 'dir'])
    def test_baseline_end_to_end(self: "TestBaselines", synthetic_fate_dataset: pd.DataFrame,
                                 method: str) -> None:
        """Each baseline runs under strict floating-point mode and returns valid scores."""
        out = rq2.run_baseline_method(synthetic_fate_dataset, DS_CFG, 'sex', 'lr', method)
        assert 0.0 <= out['performance_score'] <= 1.0
        assert 0.0 <= out['fairness_score'] <= 1.0
        assert out['elapsed_seconds'] >= 0.0
        assert out['fairness_undefined'] is False


# ---------------------------------------------------------------------------
# Statistical procedures (Table 4)
# ---------------------------------------------------------------------------

class TestStatistics:
    def test_vargha_delaney(self: "TestStatistics") -> None:
        """A12 is 1 for complete dominance, 0.5 for identical samples, 0 for the reverse."""
        assert vargha_delaney_a12([3, 4], [1, 2]) == 1.0
        assert vargha_delaney_a12([1, 2], [1, 2]) == 0.5
        assert vargha_delaney_a12([1, 2], [3, 4]) == 0.0

    def test_holm_adjust(self: "TestStatistics") -> None:
        """Holm step-down: sorted p-values multiplied by (m − rank), kept monotone, capped at 1."""
        adjusted = holm_adjust([0.01, 0.04, 0.03, 0.5])
        np.testing.assert_allclose(adjusted, [0.04, 0.09, 0.09, 0.5])
        assert holm_adjust([0.6, 0.7])[1] == 1.0

    def test_sign_test(self: "TestStatistics") -> None:
        """Exact sign test: zeros are discarded; all-zero differences give p = 1."""
        np.testing.assert_allclose(sign_test(np.array([1.0] * 10 + [0.0])), 2 / 2 ** 10)
        assert sign_test(np.zeros(5)) == 1.0

    def test_symmetry_test(self: "TestStatistics") -> None:
        """Symmetry is not rejected for a symmetric sample and is rejected for a skewed one."""
        rng = np.random.default_rng(0)
        _, p_symmetric = symmetry_test(rng.normal(size=60), replicates=2000)
        _, p_skewed = symmetry_test(rng.exponential(size=60), replicates=2000)
        assert p_symmetric > 0.05
        assert p_skewed < 0.05

    def test_table4_regenerated_from_archived_results(self: "TestStatistics") -> None:
        """Regression test: the archived Table 4 is reproduced from the archived RQ2 results."""
        results = paths.REFERENCE_DIR / 'rq2' / 'rq2_all_experiments_results.csv'
        table = run_tests(pd.read_csv(results))
        archived = pd.read_csv(paths.REFERENCE_DIR / 'rq2' / 'rq2_hypothesis_tests.csv')
        for col in ('p_value', 'a12_effective', 'fate_mean', 'baseline_mean'):
            np.testing.assert_allclose(table[col], archived[col])
        assert list(table['who_is_better']) == list(archived['who_is_better'])
        assert table['conclusion_robust'].all()
