# conftest.py — shared pytest configuration and fixtures
"""
Shared fixtures for the FATE test suite.

Adds the project root to sys.path so that all test modules can import
``fitness``, ``genetic_algorithm``, ``practices``, and ``preprocessing``
directly without any package-install step.
"""
import sys
import os

# Ensure the project root is on the path regardless of where pytest is invoked
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from pathlib import Path  # noqa: E402

import pytest  # noqa: E402
import pandas as pd  # noqa: E402
import numpy as np  # noqa: E402


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

VALID_TECHNIQUES = [
    'standard', 'stratified_sampling', 'oversampling', 'undersampling',
    'clustering', 'ipw', 'matching', 'min_max_scaling',
]


# ---------------------------------------------------------------------------
# Test isolation: never read or write the real fitness caches
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def isolated_fitness_cache(tmp_path: Path,
                           monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Redirect the runtime cache to a temporary file and empty both in-memory caches.

    Without this fixture, tests would append synthetic entries to
    ``results/fate/runtime_cache.csv`` and could be served cached values from
    earlier runs, making them order-dependent.
    """
    import fitness
    monkeypatch.setattr(fitness, "RUNTIME_CACHE_PATH", str(tmp_path / "runtime_cache.csv"))
    monkeypatch.setattr(fitness, "_runtime_cache", {})
    monkeypatch.setattr(fitness, "_root_cache", {})


# ---------------------------------------------------------------------------
# Small deterministic DataFrames used across multiple test modules
# ---------------------------------------------------------------------------

@pytest.fixture
def small_binary_dataset() -> pd.DataFrame:
    """
    Return a 12-row DataFrame with a binary 'sex' protected attribute and a
    binary 'target' column.

    Two balanced groups (sex=1 male, sex=0 female), each with 6 rows and a
    mix of positive/negative outcomes.  All feature columns are numeric so
    ``prepare_data_model`` passes through without any encoding step.

    This fixture is used by fairness-metric tests where exact metric values
    can be predicted from the data layout.
    """
    rng = np.random.RandomState(0)
    df = pd.DataFrame({
        'f0': rng.randn(12),
        'f1': rng.randn(12),
        'sex': [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        'target': [1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0],
    })
    return df


@pytest.fixture
def synthetic_fate_dataset() -> pd.DataFrame:
    """
    Return a 200-row DataFrame suitable for end-to-end FATE integration tests.

    Generated with ``sklearn.datasets.make_classification`` and extended with:
    - A binary ``sex`` protected attribute (0/1, balanced).
    - A binary ``target`` label column.

    All columns are numeric so no categorical encoding is triggered.
    The dataset is large enough for 5-fold cross-validation but small enough
    for a fast GA run with few generations.
    """
    from sklearn.datasets import make_classification
    x_raw, y_raw = make_classification(
        n_samples=200, n_features=5, n_informative=3,
        n_redundant=1, n_classes=2, random_state=42,
    )
    df = pd.DataFrame(x_raw, columns=[f'f{i}' for i in range(5)])
    df['sex'] = np.random.RandomState(42).randint(0, 2, size=200)
    df['target'] = y_raw.astype(int)
    return df
