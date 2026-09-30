"""
Floating-point policy of the replication package.

NumPy's default is to *warn* about invalid floating-point operations (division
by zero, overflow, invalid values) and to continue with ``inf``/``NaN``.  FATE
does not rely on that default:

* **Strict mode.**  Every fitness evaluation and every RQ2 baseline evaluation
  runs inside ``strict_floating_point()``, i.e. ``np.errstate(all='raise')``.
  Any invalid operation raises ``FloatingPointError`` instead of producing a
  corrupt value.  The context manager is used per call (not ``np.seterr``)
  because the error state is thread-local and ``main.py`` evaluates in
  worker threads.

* **Undefined fairness ratios.**  The fairness metrics are ratios of group
  rates.  In a test fold where a group has no positive predictions (or no
  positive instances), a ratio is undefined: 0/0 or x/0.  This is a property
  of the fold, not a numerical error, so it is the only place where the
  floating-point checks are relaxed (``undefined_ratio_errstate``), and the
  outcome is handled explicitly by ``fitness`` (see ``fitness.py`` and the
  README, Section 7).

* **BLAS back-ends that report spurious errors.**  On Apple-silicon Macs,
  NumPy >= 2.0 uses Apple's Accelerate BLAS, which raises floating-point
  status flags on valid matrix products (NumPy issue #28687, fixed only in
  NumPy 2.3, which requires Python >= 3.11).  With the flags unreliable,
  strict mode would abort valid computations.  On such back-ends only, the
  BLAS-heavy steps (Data Preparation practices and model training/prediction)
  run under ``blas_errstate()``, which ignores the flags, and their *outputs*
  are then checked with ``require_finite``.  On every other back-end
  (OpenBLAS: Linux, Windows, Intel macOS) ``blas_errstate()`` is strict.
"""
from contextlib import AbstractContextManager
from typing import Union

import numpy as np
import pandas as pd

#: Arrays/frames that ``require_finite`` accepts.
NumericData = Union[np.ndarray, pd.DataFrame, pd.Series]


class NonFiniteResultError(ArithmeticError):
    """Raised when a computation that must be finite produced ``inf`` or ``NaN``."""


def blas_backend() -> str:
    """Return the name of the BLAS library NumPy was built against (lower case)."""
    config = np.show_config(mode="dicts")
    return str(config["Build Dependencies"]["blas"]["name"]).lower()


#: True when the BLAS back-end is known to raise spurious floating-point flags.
BLAS_FLAGS_UNRELIABLE: bool = "accelerate" in blas_backend()


def strict_floating_point() -> AbstractContextManager:
    """Context in which every invalid floating-point operation raises an error."""
    return np.errstate(all="raise")


def blas_errstate() -> AbstractContextManager:
    """
    Context for BLAS-heavy steps (practices, model fit/predict).

    Strict on reliable back-ends; ignores the (spurious) flags on Accelerate,
    where the caller must validate the outputs with ``require_finite``.
    """
    if BLAS_FLAGS_UNRELIABLE:
        return np.errstate(all="ignore")
    return np.errstate(all="raise")


def undefined_ratio_errstate() -> AbstractContextManager:
    """Context for fairness ratios, whose undefined cases are handled explicitly."""
    return np.errstate(divide="ignore", invalid="ignore")


def require_finite(values: NumericData, what: str) -> None:
    """
    Raise ``NonFiniteResultError`` if *values* contain ``inf`` or ``NaN``.

    Parameters
    ----------
    values : np.ndarray, pd.DataFrame or pd.Series
        Numeric output to validate (non-numeric columns are ignored).
    what : str
        Description used in the error message.
    """
    if isinstance(values, pd.DataFrame):
        array = values.select_dtypes(include="number").to_numpy(dtype=float)
    else:
        array = np.asarray(values, dtype=float)
    if not np.isfinite(array).all():
        n_bad = int(np.size(array) - np.isfinite(array).sum())
        raise NonFiniteResultError(f"{what}: {n_bad} non-finite value(s)")
