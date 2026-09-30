"""
Targeted handling of AIF360's import-time notices about optional algorithms.

When ``aif360`` is imported, it probes optional back-ends for in-processing
algorithms and logs a notice on the *root* logger for each one that is not
installed.  FATE does not use these algorithms (it only uses AIF360's
fairness metrics, ``Reweighing`` and ``DisparateImpactRemover``), and their
back-ends (TensorFlow, PyTorch/inFairness) are multi-gigabyte dependencies,
so they are deliberately not listed in ``requirements.txt``.

This module suppresses exactly those notices, i.e. the ones that concern the
two back-ends below.  Every other log record, including notices about any
other missing dependency, is left untouched, so an incomplete environment is
still reported.
"""
import logging

#: Optional AIF360 back-ends that FATE intentionally does not install.
UNUSED_OPTIONAL_BACKENDS = ("tensorflow", "inFairness")


class _UnusedBackendNoticeFilter(logging.Filter):
    """Drop AIF360 notices that refer only to back-ends FATE does not use."""

    def filter(self: "_UnusedBackendNoticeFilter", record: logging.LogRecord) -> bool:
        """Return False for the targeted notices, True for every other record."""
        message = record.getMessage()
        if "will be unavailable" not in message:
            return True
        return not any(f"No module named '{name}'" in message
                       for name in UNUSED_OPTIONAL_BACKENDS)


def silence_unused_backend_notices() -> None:
    """Install the notice filter on the root logger (idempotent)."""
    root = logging.getLogger()
    if not any(isinstance(f, _UnusedBackendNoticeFilter) for f in root.filters):
        root.addFilter(_UnusedBackendNoticeFilter())
