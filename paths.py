"""
Single source of truth for every file location used by the replication package.

All paths are anchored at the repository root, so every script behaves the
same regardless of the working directory it is launched from.

Two top-level folders hold data produced by FATE:

``results/``
    Everything produced by *running* the package (FATE grid, RQ1 and RQ2
    analyses, figures).  Created on demand and git-ignored: a fresh clone
    contains no results, so a replication always starts from an empty folder.

``reference/``
    Read-only archive of the outputs of the runs reported in the paper.  It is
    never written by any script.  It exists so that replicators can (i) compare
    their own ``results/`` against the numbers in the paper and (ii) re-run the
    RQ1/RQ2 analyses on the paper's raw GA results without re-running the
    multi-day FATE grid.
"""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
DATASETS_DIR = REPO_ROOT / "datasets"

# ---------------------------------------------------------------------------
# Outputs of a run (git-ignored)
# ---------------------------------------------------------------------------
RESULTS_DIR = REPO_ROOT / "results"

FATE_RESULTS_DIR = RESULTS_DIR / "fate"
FATE_RESULTS_CSV = FATE_RESULTS_DIR / "experiments_results.csv"
FATE_ERRORS_LOG = FATE_RESULTS_DIR / "errors.log"
RUNTIME_CACHE_CSV = FATE_RESULTS_DIR / "runtime_cache.csv"

RQ1_RESULTS_DIR = RESULTS_DIR / "rq1"
RQ1_FIGURES_DIR = RQ1_RESULTS_DIR / "figures"

RQ2_RESULTS_DIR = RESULTS_DIR / "rq2"

# ---------------------------------------------------------------------------
# Archived outputs of the paper's runs (read-only, versioned)
# ---------------------------------------------------------------------------
REFERENCE_DIR = REPO_ROOT / "reference"
REFERENCE_FATE_RESULTS_CSV = REFERENCE_DIR / "fate" / "experiments_results.csv"
REFERENCE_FITNESS_CACHE_CSV = REFERENCE_DIR / "fate" / "fitness_cache.csv"
REFERENCE_RQ1_DIR = REFERENCE_DIR / "rq1"
REFERENCE_RQ2_DIR = REFERENCE_DIR / "rq2"


def ensure_dir(path: Path) -> Path:
    """Create *path* (and its parents) if missing and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path
