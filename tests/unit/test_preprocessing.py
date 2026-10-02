# tests/unit/test_preprocessing.py
"""
Unit tests for ``preprocessing.py``: generic cleaning, encoding and target
binarisation, and the dataset-specific preparation of Adult, German and Heart.
"""
import numpy as np
import pandas as pd
import pytest

from preprocessing import (binarize_target, encode_and_impute, minimal_clean,
                           prepare_adult, prepare_data_model, prepare_german,
                           prepare_heart)


class TestGenericPreparation:
    def test_minimal_clean(self: "TestGenericPreparation") -> None:
        """Rows with a missing target and all-NA columns are dropped; the index is reset."""
        df = pd.DataFrame({'a': [1.0, 2.0, 3.0], 'empty': [np.nan] * 3, 'y': [1, np.nan, 0]})
        out = minimal_clean(df, 'y')
        assert list(out.columns) == ['a', 'y']
        assert list(out['a']) == [1.0, 3.0]
        assert list(out.index) == [0, 1]

    def test_encode_and_impute(self: "TestGenericPreparation") -> None:
        """Categorical columns are one-hot encoded, NaNs imputed, the protected attribute kept."""
        df = pd.DataFrame({'num': [1.0, np.nan, 3.0], 'cat': ['x', 'y', 'x'], 'sex': [1, 0, 1]})
        out = encode_and_impute(df, protected_attribute='sex')
        assert 'cat' not in out.columns and 'cat_y' in out.columns
        assert list(out['num']) == [1.0, 2.0, 3.0]
        assert list(out['sex']) == [1, 0, 1]

    @pytest.mark.parametrize('kwargs, values, expected', [
        ({'positive_values': ['yes']}, ['yes', 'no', 'yes'], [1, 0, 1]),
        ({'threshold': 2}, [1, 2, 3], [0, 1, 1]),
        ({}, [1, 2, 3, 4], [0, 0, 1, 1]),            # median split
        ({}, ['b', 'a', 'b'], [0, 1, 0]),            # two values: first seen -> 0
    ])
    def test_binarize_target(self: "TestGenericPreparation", kwargs: dict,
                             values: list, expected: list) -> None:
        """Each binarisation rule maps the target to {0, 1} as documented."""
        out = binarize_target(pd.DataFrame({'y': values}), 'y', **kwargs)
        assert list(out['y']) == expected

    def test_binarize_target_missing_column(self: "TestGenericPreparation") -> None:
        """A missing target column is an error."""
        with pytest.raises(ValueError):
            binarize_target(pd.DataFrame({'a': [1]}), 'y')

    def test_prepare_data_model_is_numeric(self: "TestGenericPreparation") -> None:
        """The model-ready frame is fully numeric and keeps target and protected attribute."""
        df = pd.DataFrame({'num': [1.0, np.nan, 3.0, 4.0], 'cat': ['x', 'y', 'x', 'y'],
                           'sex': [1, 0, 1, 0], 'y': ['p', 'n', 'p', 'n']})
        out = prepare_data_model(df, 'y', 'sex')
        assert {'y', 'sex'} <= set(out.columns)
        assert all(pd.api.types.is_numeric_dtype(out[c]) for c in out.columns)
        assert set(out['y']) == {0, 1}


class TestDatasetPreparation:
    def test_prepare_adult(self: "TestDatasetPreparation") -> None:
        """Textual salary labels map to {0, 1}; one-hot sex columns collapse to sex (1 = female)."""
        df = pd.DataFrame({'salary': ['>50K', '<=50K', '>50K.'], 'sex_Female': [1, 0, 0],
                           'sex_Male': [0, 1, 1], 'race': ['1', '0', '1']})
        out = prepare_adult(df)
        assert list(out['salary']) == [1, 0, 1]
        assert list(out['sex']) == [1, 0, 0]
        assert list(out['race']) == [1, 0, 1]

    def test_prepare_german(self: "TestDatasetPreparation") -> None:
        """Personal-status codes map to sex (A92/A95 female = 1); Target 2 -> 1, 1 -> 0."""
        df = pd.DataFrame({'sex': ['A91', 'A92', 'A93', 'A95'], 'age': ['30', '45', '22', '60'],
                           'Target': [1, 2, 2, 1]})
        out = prepare_german(df)
        assert list(out['sex']) == [0, 1, 0, 1]
        assert list(out['age']) == [30, 45, 22, 60]
        assert list(out['Target']) == [0, 1, 1, 0]

    def test_prepare_heart(self: "TestDatasetPreparation") -> None:
        """The multi-class num target is binarised: 0 -> 0, any positive value -> 1."""
        df = pd.DataFrame({'sex': ['1', '0', '1', '0'], 'age': ['50', '60', '41', '70'],
                           'num': [0, 1, 3, 0]})
        out = prepare_heart(df)
        assert list(out['num']) == [0, 1, 1, 0]
        assert list(out['sex']) == [1, 0, 1, 0]
