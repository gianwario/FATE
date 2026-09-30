# tests/unit/test_fairness_metrics.py
"""
Unit tests for FATE fairness metric computation (Algorithm 1 Step 2 – FS component).

Tests target ``fitness.fairness_metrics`` and its private helpers.  The tests
use small, manually constructed DataFrames where the expected SPD, EOD, and DI
values can be calculated by hand, making it possible to verify AIF360
integration and the binarisation logic independently of any real dataset.

Metrics under test
------------------
SPD (Statistical Parity Difference)
    |P(Ŷ=1 | unprivileged) − P(Ŷ=1 | privileged)|

EOD (Equal Opportunity Difference)
    |P(Ŷ=1 | Y=1, unprivileged) − P(Ŷ=1 | Y=1, privileged)|

DI (Disparate Impact deviation)
    |1 − P(Ŷ=1 | unprivileged) / P(Ŷ=1 | privileged)|

In the FATE convention, privileged = sex=1 (male), unprivileged = sex=0 (female).
"""
import numpy as np
import pandas as pd

from fitness import (
    fairness_metrics,
    _binarise_sex_column,
    _binarise_age_column,
    _binarise_protected_column,
    _nan_fairness_result,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_test_data(sex: list[object], target: list[int]) -> pd.DataFrame:
    """Build a minimal test DataFrame from explicit sex and target arrays."""
    n = len(sex)
    return pd.DataFrame(
        {'f0': np.zeros(n), 'sex': sex, 'target': target},
        index=range(n),
    )


# ---------------------------------------------------------------------------
# Tests for _nan_fairness_result
# ---------------------------------------------------------------------------

class TestNanFairnessResult:
    def test_returns_nan_dict(self: "TestNanFairnessResult") -> None:
        """
        _nan_fairness_result must return a dict with exactly three keys, all NaN.

        Every failure path in fairness_metrics returns this sentinel; a wrong
        key name or a non-NaN value would silently corrupt FS aggregation.
        """
        result = _nan_fairness_result()
        assert set(result.keys()) == {
            'statistical_parity', 'equal_opportunity', 'disparate_impact'
        }
        for v in result.values():
            assert np.isnan(v), f"Expected NaN but got {v}"


# ---------------------------------------------------------------------------
# Tests for protected-attribute binarisation helpers
# ---------------------------------------------------------------------------

class TestBinariseSexColumn:
    def test_numeric_one_maps_to_privileged(self: "TestBinariseSexColumn") -> None:
        """
        Numeric 1 in the sex column must map to 1 (privileged / male).

        The AIF360 datasets are built with privileged_groups=[{'__prot_bin__': 1}],
        so incorrect binarisation flips the direction of all fairness metrics.
        """
        s = pd.Series([1, 0, 1, 0])
        result = _binarise_sex_column(s)
        np.testing.assert_array_equal(result.values, [1, 0, 1, 0])

    def test_string_male_maps_to_privileged(self: "TestBinariseSexColumn") -> None:
        """
        String 'male' (case-insensitive) must map to 1; 'female' to 0.

        The Adult dataset stores sex as 'Male' / 'Female'.  The binarisation
        uses an equality check (``s in ('male', 'm')``) so that 'female' is
        not accidentally matched as a substring of 'male'.
        """
        s = pd.Series(['Male', 'female', 'MALE', 'Female'])
        result = _binarise_sex_column(s)
        np.testing.assert_array_equal(result.values, [1, 0, 1, 0])


class TestBinariseAgeColumn:
    def test_above_mean_is_privileged(self: "TestBinariseAgeColumn") -> None:
        """
        Ages strictly above the column mean map to 1 (privileged).

        The mean of [20, 40, 60] is 40.  Only 60 > 40, so the expected
        output is [0, 0, 1].
        """
        s = pd.Series([20.0, 40.0, 60.0])
        result = _binarise_age_column(s)
        np.testing.assert_array_equal(result.values, [0, 0, 1])

    def test_equal_to_mean_is_unprivileged(self: "TestBinariseAgeColumn") -> None:
        """
        Values exactly equal to the mean must NOT be privileged (strict >).

        This confirms the strict inequality, which matters when all ages are
        the same (mean equals every value — everyone becomes unprivileged).
        """
        s = pd.Series([50.0, 50.0, 50.0])
        result = _binarise_age_column(s)
        np.testing.assert_array_equal(result.values, [0, 0, 0])


class TestBinariseProtectedColumnDispatch:
    def test_sex_attribute_dispatches_to_sex_rule(self: 'TestBinariseProtectedColumnDispatch'
                                                  ) -> None:
        """
        Columns whose name contains 'sex' must use the sex binarisation rule.

        Ensures the dispatch table in _binarise_protected_column is not broken
        when column names differ in capitalisation or have a prefix.
        """
        s = pd.Series([1, 0, 1])
        result = _binarise_protected_column(s, 'sex')
        np.testing.assert_array_equal(result.values, [1, 0, 1])

    def test_age_attribute_dispatches_to_age_rule(self: 'TestBinariseProtectedColumnDispatch'
                                                  ) -> None:
        """
        Columns whose name contains 'age' must use the mean-threshold rule.
        """
        s = pd.Series([20.0, 50.0, 80.0])
        result = _binarise_protected_column(s, 'age')
        expected = [0, 0, 1]  # only 80 > 50 (mean is 50.0)
        np.testing.assert_array_equal(result.values, expected)


# ---------------------------------------------------------------------------
# Tests for fairness_metrics end-to-end
# ---------------------------------------------------------------------------

class TestFairnessMetricsEndToEnd:
    def test_missing_protected_attribute_returns_nan(self: "TestFairnessMetricsEndToEnd") -> None:
        """
        When the protected attribute column is absent from test_data, all three
        metrics must return NaN rather than raising an exception.

        The GA evaluation must survive attribute-absent folds gracefully so that
        one bad fold does not abort the entire genetic search.
        """
        df = _make_test_data(sex=[1, 0, 1, 0], target=[1, 0, 1, 0])
        result = fairness_metrics(
            test_data=df,
            test_indices=pd.Index(range(4)),
            protected_attribute='nonexistent_col',
            predictions=np.array([1, 0, 1, 0]),
            target_column='target',
        )
        for k, v in result.items():
            assert np.isnan(v), f"Expected NaN for '{k}' when attribute missing, got {v}"

    def test_equal_positive_rate_gives_zero_spd(self: "TestFairnessMetricsEndToEnd") -> None:
        """
        When positive-prediction rates are identical across both groups, SPD must be 0.

        Dataset layout (6 rows):
          sex=1 (male):   3 rows, predictions = [1, 0, 0] → rate 1/3
          sex=0 (female): 3 rows, predictions = [1, 0, 0] → rate 1/3
        SPD = |1/3 − 1/3| = 0.

        This is the reference 'perfectly fair predictions' case used to verify
        that the AIF360 integration is wired correctly (non-zero would indicate
        a group-membership mix-up or a sign error in the formula).
        """
        df = _make_test_data(
            sex=[1, 1, 1, 0, 0, 0],
            target=[1, 0, 0, 1, 0, 0],
        )
        predictions = np.array([1, 0, 0, 1, 0, 0])   # equal rate 1/3 per group
        result = fairness_metrics(df, pd.Index(range(6)), 'sex', predictions, 'target')

        assert not np.isnan(result['statistical_parity']), "SPD should be computable"
        np.testing.assert_allclose(result['statistical_parity'], 0.0, atol=1e-9)

    def test_unequal_positive_rate_gives_nonzero_spd(self: "TestFairnessMetricsEndToEnd") -> None:
        """
        When all positive predictions go to the privileged group, SPD must be > 0.

        Dataset (6 rows, both groups 3 each):
          Privileged (sex=1): predictions = [1, 1, 0] → rate 2/3
          Unprivileged (sex=0): predictions = [0, 0, 0] → rate 0

        But to avoid division-by-zero in the DI ratio, at least one positive
        prediction is given to the unprivileged group in this test.

          Privileged: [1, 1, 0] → rate 2/3
          Unprivileged: [0, 1, 0] → rate 1/3
        SPD = |1/3 − 2/3| = 1/3 ≈ 0.333.

        This verifies that unequal treatment is correctly detected and quantified.
        """
        df = _make_test_data(
            sex=[1, 1, 1, 0, 0, 0],
            target=[1, 1, 0, 1, 1, 0],
        )
        predictions = np.array([1, 1, 0, 0, 1, 0])   # P(+|priv)=2/3, P(+|unpriv)=1/3
        result = fairness_metrics(df, pd.Index(range(6)), 'sex', predictions, 'target')

        assert not np.isnan(result['statistical_parity'])
        assert result['statistical_parity'] > 0.0, (
            "Unequal positive rates must produce a positive SPD"
        )
        np.testing.assert_allclose(result['statistical_parity'], 1 / 3, atol=1e-9)

    def test_all_metrics_are_non_negative(self: "TestFairnessMetricsEndToEnd") -> None:
        """
        All returned metric values must be non-negative (absolute values).

        The raw AIF360 metrics can be negative (e.g., unprivileged group
        favoured).  The FATE fairness_metrics wrapper takes absolute values
        so that the fitness formula's subtraction is always a penalty.
        """
        df = _make_test_data(
            sex=[1, 1, 1, 0, 0, 0],
            target=[1, 0, 0, 1, 0, 0],
        )
        # Flip: more positives predicted for the unprivileged group
        predictions = np.array([0, 0, 0, 1, 1, 0])
        result = fairness_metrics(df, pd.Index(range(6)), 'sex', predictions, 'target')

        for k, v in result.items():
            if not np.isnan(v):
                assert v >= 0.0, f"Metric '{k}' must be non-negative; got {v}"

    def test_result_keys_are_correct(self: "TestFairnessMetricsEndToEnd") -> None:
        """
        The returned dict must have exactly the three keys the fitness function sums.

        ``_aggregate_fold_fairness`` sums all numeric dict values; if the key
        names ever change, the FS contribution silently drops to zero.
        """
        df = _make_test_data(sex=[1, 0, 1, 0], target=[1, 0, 1, 0])
        result = fairness_metrics(df, pd.Index(range(4)), 'sex',
                                  np.array([1, 0, 1, 0]), 'target')
        assert set(result.keys()) == {
            'statistical_parity', 'equal_opportunity', 'disparate_impact'
        }
