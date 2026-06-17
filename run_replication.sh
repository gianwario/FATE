#!/usr/bin/env bash
# =============================================================================
# run_replication.sh — FATE experiment runner
#
# Two modes:
#
#   --mode fast   Run FATE once and print what it found (best technique
#                 pipeline, fitness, performance, fairness).  Defaults to the
#                 Adult dataset / Logistic Regression / pop=5 / gen=5.
#                 Completes in < 5 minutes.  Use this to verify the pipeline
#                 works and to explore FATE on any dataset you choose.
#
#   --mode full   Run the complete parameter grid used in the paper:
#                 3 datasets × 4 models × 2 protected attributes ×
#                 6 pop sizes × 6 gen counts × 5×5 rate combinations.
#                 Estimated runtime: 24–48 hours on 64 cores.
#
# Usage:
#   ./run_replication.sh --mode fast [--pop N] [--gen N] [--dataset NAME] [--model NAME]
#   ./run_replication.sh --mode full [--pop N] [--gen N] [--dataset NAME] [--model NAME]
#
# See --help for full option reference.
# =============================================================================
set -euo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# =============================================================================
# CONFIGURABLE — edit these to run FATE on your own dataset or scenario.
# CLI flags (--pop, --gen, --dataset, --model) override anything set here.
# =============================================================================

# Dataset to evaluate. Built-in options: adult | german | heart
# To use your own dataset, add a preparer function in preprocessing.py and
# add its entry to _DS_CFGS inside the Python block below.
DEFAULT_DATASET="adult"

# Classifier to use: lr (Logistic Regression) | rf (Random Forest) |
#                    svc (LinearSVC)           | xgb (XGBoost)
DEFAULT_MODEL="lr"

# GA population size — how many candidate pipelines per generation.
# Larger values explore more of the search space but take longer.
DEFAULT_POP=5

# GA generation count — how many evolutionary rounds to run.
# More generations allow the algorithm more time to converge.
DEFAULT_GEN=5

# Full-mode grid values (used when --mode full with no overrides).
# These exactly replicate the experimental setup from the paper.
FULL_POP_SIZES=(5 10 15 20 50 100)
FULL_GEN_COUNTS=(5 10 15 20 50 100)
FULL_RATES=(0 0.25 0.50 0.75 1)

# =============================================================================
# internals — do not edit below this line unless you know what you are doing
# =============================================================================
OUTPUT_DIR="FATE_output"
SUMMARY_CSV="${OUTPUT_DIR}/experiments_results.csv"
ERRORS_LOG="${OUTPUT_DIR}/errors.log"
EXPECTED_COLS=13   # number of columns in a clean result row

MODE=""
POP_OVERRIDE=""
GEN_OVERRIDE=""
DATASET_OVERRIDE=""
MODEL_OVERRIDE=""

# =============================================================================
# usage
# =============================================================================
usage() {
    cat <<'USAGE'
Usage: ./run_replication.sh --mode fast|full [OPTIONS]

Modes:
  --mode fast    Run FATE once and show what it found.
                 Default: adult / lr / pop=5 / gen=5.
                 Completes in < 5 minutes.  Edit the CONFIGURABLE block at
                 the top of the script to point at your own dataset.

  --mode full    Complete paper grid (all datasets, models, hyperparameters).
                 Estimated runtime: 24–48 hours (64 parallel worker threads).

Options:
  --pop   N       Override population size (positive integer)
  --gen   N       Override number of generations (positive integer)
  --dataset NAME  One of: adult | german | heart
  --model   NAME  One of: lr | rf | svc | xgb
  -h, --help      Show this message

Examples:
  ./run_replication.sh --mode fast
  ./run_replication.sh --mode fast --dataset german --model rf --pop 5 --gen 5
  ./run_replication.sh --mode full
  ./run_replication.sh --mode full --dataset adult --model lr
USAGE
    exit 0
}

# =============================================================================
# argument parsing and validation
# =============================================================================
while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)    MODE="$2";             shift 2 ;;
        --pop)     POP_OVERRIDE="$2";     shift 2 ;;
        --gen)     GEN_OVERRIDE="$2";     shift 2 ;;
        --dataset) DATASET_OVERRIDE="$2"; shift 2 ;;
        --model)   MODEL_OVERRIDE="$2";   shift 2 ;;
        -h|--help) usage ;;
        *)
            echo "ERROR: Unknown argument: $1" >&2
            echo "Run './run_replication.sh --help' for usage." >&2
            exit 1
            ;;
    esac
done

if [[ -z "$MODE" ]]; then
    echo "ERROR: --mode is required." >&2
    echo "Run './run_replication.sh --help' for usage." >&2
    exit 1
fi
if [[ "$MODE" != "fast" && "$MODE" != "full" ]]; then
    echo "ERROR: --mode must be 'fast' or 'full'; got '$MODE'." >&2; exit 1
fi
if [[ -n "$DATASET_OVERRIDE" ]] &&
   [[ "$DATASET_OVERRIDE" != "adult" &&
      "$DATASET_OVERRIDE" != "german" &&
      "$DATASET_OVERRIDE" != "heart" ]]; then
    echo "ERROR: --dataset must be adult | german | heart; got '$DATASET_OVERRIDE'." >&2; exit 1
fi
if [[ -n "$MODEL_OVERRIDE" ]] &&
   [[ "$MODEL_OVERRIDE" != "lr" &&
      "$MODEL_OVERRIDE" != "rf" &&
      "$MODEL_OVERRIDE" != "svc" &&
      "$MODEL_OVERRIDE" != "xgb" ]]; then
    echo "ERROR: --model must be lr | rf | svc | xgb; got '$MODEL_OVERRIDE'." >&2; exit 1
fi
if [[ -n "$POP_OVERRIDE" ]] && ! [[ "$POP_OVERRIDE" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: --pop must be a positive integer; got '$POP_OVERRIDE'." >&2; exit 1
fi
if [[ -n "$GEN_OVERRIDE" ]] && ! [[ "$GEN_OVERRIDE" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: --gen must be a positive integer; got '$GEN_OVERRIDE'." >&2; exit 1
fi

# =============================================================================
# environment check
# =============================================================================
echo ""
echo "=== Environment Check ==="

is_python310() {
    local cmd="$1"
    command -v "$cmd" &>/dev/null || return 1
    local ver
    ver=$("$cmd" --version 2>&1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
    [[ "$ver" == "3.10" ]]
}

PYTHON=""
for _cand in python3.10 python3 python; do
    if is_python310 "$_cand" 2>/dev/null; then
        PYTHON="$_cand"
        break
    fi
done

if [[ -z "$PYTHON" ]]; then
    echo "ERROR: Python 3.10 is required but not found in PATH." >&2
    echo "       Activate the project virtual environment first:" >&2
    echo "         source venv/bin/activate" >&2
    exit 1
fi
echo "Python:   $($PYTHON --version)"

MISSING_PKGS=()
for _pkg in pandas numpy sklearn scipy aif360 xgboost; do
    if ! $PYTHON -c "import $_pkg" 2>/dev/null; then
        MISSING_PKGS+=("$_pkg")
    fi
done
if [[ ${#MISSING_PKGS[@]} -gt 0 ]]; then
    echo "ERROR: Missing packages: ${MISSING_PKGS[*]}" >&2
    echo "       Run:  pip install -r requirements.txt" >&2
    exit 1
fi
echo "Packages: all present"

mkdir -p "$OUTPUT_DIR"

# =============================================================================
# fast mode — run FATE once and show what it found
# =============================================================================
if [[ "$MODE" == "fast" ]]; then

    DATASET="${DATASET_OVERRIDE:-$DEFAULT_DATASET}"
    MODEL="${MODEL_OVERRIDE:-$DEFAULT_MODEL}"
    POP="${POP_OVERRIDE:-$DEFAULT_POP}"
    GEN="${GEN_OVERRIDE:-$DEFAULT_GEN}"

    echo ""
    echo "=== Running FATE ==="
    printf "  dataset = %s\n"  "$DATASET"
    printf "  model   = %s\n"  "$MODEL"
    printf "  pop     = %s\n"  "$POP"
    printf "  gen     = %s\n"  "$GEN"
    echo ""

    $PYTHON - <<PYEOF
import os
import sys
import time
import traceback
import logging

for _k, _v in [
    ('OMP_NUM_THREADS',        '1'),
    ('OPENBLAS_NUM_THREADS',   '1'),
    ('MKL_NUM_THREADS',        '1'),
    ('VECLIB_MAXIMUM_THREADS', '1'),
    ('NUMEXPR_NUM_THREADS',    '1'),
]:
    os.environ.setdefault(_k, _v)

logging.basicConfig(
    format='%(asctime)s %(levelname)-7s %(message)s',
    datefmt='%H:%M:%S',
    level=logging.INFO,
)

import pandas as pd
from main import execute_fate, _write_error_log
from preprocessing import prepare_data_model
import preprocessing as _preproc

# ── dataset catalogue ────────────────────────────────────────────────────────
# To add your own dataset:
#   1. Write a prepare_<name>(df) function in preprocessing.py
#   2. Add an entry here: name -> {path, preparer, target, prot}
_DS_CFGS = {
    'adult':  {'path': 'datasets/adult.csv',  'preparer': 'prepare_adult',
               'target': 'salary', 'prot': 'sex'},
    'german': {'path': 'datasets/german.csv', 'preparer': 'prepare_german',
               'target': 'Target', 'prot': 'sex'},
    'heart':  {'path': 'datasets/heart.csv',  'preparer': 'prepare_heart',
               'target': 'num',    'prot': 'sex'},
}

ds_name = '${DATASET}'
model   = '${MODEL}'
pop     = ${POP}
gen     = ${GEN}
summary = '${SUMMARY_CSV}'
err_log = '${ERRORS_LOG}'

cfg     = _DS_CFGS[ds_name]
ds_info = {'name': ds_name, 'path': cfg['path']}

print(f"Loading {cfg['path']} ...", flush=True)
raw       = pd.read_csv(cfg['path'])
preparer  = getattr(_preproc, cfg['preparer'])
processed = preparer(raw)
sample    = prepare_data_model(
    processed, cfg['target'],
    protected_attribute=cfg['prot'], binarize=False,
)
print(f"Rows: {len(sample)}    Columns: {len(sample.columns)}", flush=True)
print(flush=True)

# Initialise CSV header on first run
if not os.path.exists(summary):
    pd.DataFrame(columns=[
        'dataset', 'model_identifier', 'protected_attribute',
        'population_size', 'generations', 'alpha', 'beta',
        'techniques', 'model_used', 'fitness',
        'fairness_score', 'performance_score', 'elapsed_seconds',
    ]).to_csv(summary, index=False)

t0 = time.time()
try:
    rows = execute_fate(
        sample, ds_name, cfg['path'], cfg['prot'], cfg['target'],
        [model], population_size=pop, generations=gen,
        alpha=0.5, beta=0.5, summary_path=None,
    )
except Exception as exc:
    _write_error_log(
        err_log, ds_info, cfg['prot'], pop, gen, 0.5, 0.5,
        str(exc), traceback.format_exc(),
    )
    print(f"FATAL: {exc}", file=sys.stderr)
    sys.exit(1)

elapsed      = time.time() - t0
success_rows = [r for r in rows if r.get('error') is None]
error_rows   = [r for r in rows if r.get('error') is not None]

# ── print FATE result ────────────────────────────────────────────────────────
if success_rows:
    r = success_rows[0]
    print()
    print("=" * 60)
    print("  FATE result")
    print("=" * 60)
    techniques = r.get('techniques', 'N/A')
    fitness    = r.get('fitness')
    perf       = r.get('performance_score')
    fair       = r.get('fairness_score')
    print(f"  Best pipeline : {techniques}")
    print(f"  Model         : {r.get('model_used', model)}")
    print(f"  Fitness       : {fitness:.4f}"   if fitness is not None else "  Fitness       : N/A")
    print(f"  Performance   : {perf:.4f}  (mean PR-AUC across 5 CV folds)"
          if perf is not None else "  Performance   : N/A")
    print(f"  Fairness      : {fair:.4f}  (mean |SPD|+|EOD|+|DI| / 3)"
          if fair is not None else "  Fairness      : N/A")
    print(f"  Elapsed       : {elapsed:.1f} s")
    print("=" * 60)
    print()

    df = pd.DataFrame(success_rows).drop(columns=['error'], errors='ignore')
    df.to_csv(summary, mode='a', header=False, index=False)
    print(f"Result written to: {summary}")

for r in error_rows:
    _write_error_log(
        err_log, ds_info, cfg['prot'], pop, gen, 0.5, 0.5,
        f"model={r['model_identifier']}: {r['error']}",
    )
    print(f"Error logged: {r['error']}")
PYEOF

# =============================================================================
# full mode — complete paper grid
# =============================================================================
else

    echo ""
    if [[ -n "$DATASET_OVERRIDE" || -n "$MODEL_OVERRIDE" ||
          -n "$POP_OVERRIDE"     || -n "$GEN_OVERRIDE" ]]; then
        echo "=== Full Grid Run (with overrides) ==="
    else
        echo "=== Full Grid Run (complete paper configuration) ==="
    fi
    [[ -n "$DATASET_OVERRIDE" ]] && printf "  dataset filter = %s\n"             "$DATASET_OVERRIDE"
    [[ -n "$MODEL_OVERRIDE"   ]] && printf "  model filter   = %s\n"             "$MODEL_OVERRIDE"
    [[ -n "$POP_OVERRIDE"     ]] && printf "  pop override   = %s (pins grid)\n" "$POP_OVERRIDE"
    [[ -n "$GEN_OVERRIDE"     ]] && printf "  gen override   = %s (pins grid)\n" "$GEN_OVERRIDE"
    echo ""

    # Convert bash arrays to comma-separated strings for Python
    POP_STR=$(IFS=,; echo "${FULL_POP_SIZES[*]}")
    GEN_STR=$(IFS=,; echo "${FULL_GEN_COUNTS[*]}")
    RATE_STR=$(IFS=,; echo "${FULL_RATES[*]}")

    $PYTHON - <<PYEOF
import os
import sys
import time
import traceback
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

for _k, _v in [
    ('OMP_NUM_THREADS',        '1'),
    ('OPENBLAS_NUM_THREADS',   '1'),
    ('MKL_NUM_THREADS',        '1'),
    ('VECLIB_MAXIMUM_THREADS', '1'),
    ('NUMEXPR_NUM_THREADS',    '1'),
]:
    os.environ.setdefault(_k, _v)

logging.basicConfig(
    format='%(asctime)s %(levelname)-7s %(message)s',
    datefmt='%H:%M:%S',
    level=logging.INFO,
)

import pandas as pd
from main import execute_fate, _write_error_log
from preprocessing import prepare_data_model
import preprocessing as _preproc

summary = '${SUMMARY_CSV}'
err_log = '${ERRORS_LOG}'

# Dataset catalogue — mirrors main.py __main__ block exactly
_ALL_DATASETS = [
    {'name': 'adult',  'path': 'datasets/adult.csv',
     'preparer_name': 'prepare_adult',
     'protected_attributes': ['race', 'sex'], 'target': 'salary'},
    {'name': 'german', 'path': 'datasets/german.csv',
     'preparer_name': 'prepare_german',
     'protected_attributes': ['sex', 'age'], 'target': 'Target'},
    {'name': 'heart',  'path': 'datasets/heart.csv',
     'preparer_name': 'prepare_heart',
     'protected_attributes': ['sex', 'age'], 'target': 'num'},
]

# Apply CLI overrides; empty string means "no override"
dataset_filter = '${DATASET_OVERRIDE}'
model_filter   = '${MODEL_OVERRIDE}'
pop_override   = '${POP_OVERRIDE}'
gen_override   = '${GEN_OVERRIDE}'

datasets   = [d for d in _ALL_DATASETS
              if not dataset_filter or d['name'] == dataset_filter]
models     = ['rf', 'lr', 'svc', 'xgb']
if model_filter:
    models = [model_filter]

# Use the full paper grid unless an override pins the value
pop_sizes  = ([int(x) for x in '${POP_STR}'.split(',')]
              if not pop_override else [int(pop_override)])
gen_counts = ([int(x) for x in '${GEN_STR}'.split(',')]
              if not gen_override else [int(gen_override)])
rates      = [float(x) for x in '${RATE_STR}'.split(',')]

if not os.path.exists(summary):
    pd.DataFrame(columns=[
        'dataset', 'model_identifier', 'protected_attribute',
        'population_size', 'generations', 'alpha', 'beta',
        'techniques', 'model_used', 'fitness',
        'fairness_score', 'performance_score', 'elapsed_seconds',
    ]).to_csv(summary, index=False)


def _worker(ds_cfg, prot, pop, gen, alpha, beta):
    """Thread worker: load dataset, prepare sample, delegate to execute_fate."""
    for _k, _v in [
        ('OMP_NUM_THREADS',        '1'),
        ('OPENBLAS_NUM_THREADS',   '1'),
        ('MKL_NUM_THREADS',        '1'),
        ('VECLIB_MAXIMUM_THREADS', '1'),
        ('NUMEXPR_NUM_THREADS',    '1'),
    ]:
        os.environ.setdefault(_k, _v)
    raw      = pd.read_csv(ds_cfg['path'])
    preparer = getattr(_preproc, ds_cfg['preparer_name'])
    processed = preparer(raw)
    if prot not in processed.columns:
        raise ValueError(
            f"Protected attribute '{prot}' not found in "
            f"'{ds_cfg['name']}' after preparation"
        )
    sample = prepare_data_model(
        processed, ds_cfg['target'],
        protected_attribute=prot, binarize=False,
    )
    return execute_fate(
        sample, ds_cfg['name'], ds_cfg['path'], prot, ds_cfg['target'],
        models, population_size=pop, generations=gen,
        alpha=alpha, beta=beta, summary_path=None,
    )


tasks = [
    (ds, prot, pop, gen, alpha, beta)
    for ds    in datasets
    for prot  in ds['protected_attributes']
    for pop   in pop_sizes
    for gen   in gen_counts
    for alpha in rates
    for beta  in rates
]

total = len(tasks)
width = len(str(total))
print(f"Tasks       : {total}", flush=True)
print(f"Results CSV : {summary}", flush=True)
print(f"Errors log  : {err_log}", flush=True)
print(flush=True)

overall_start = time.time()
done          = 0
errors_seen   = 0

with ThreadPoolExecutor(max_workers=64) as executor:
    fut_to_task = {executor.submit(_worker, *t): t for t in tasks}
    for fut in as_completed(fut_to_task):
        ds_cfg, prot, pop, gen, alpha, beta = fut_to_task[fut]
        done += 1
        try:
            rows         = fut.result()
            success_rows = [r for r in rows if r.get('error') is None]
            error_rows   = [r for r in rows if r.get('error') is not None]
            if success_rows:
                df = pd.DataFrame(success_rows).drop(columns=['error'], errors='ignore')
                df.to_csv(summary, mode='a', header=False, index=False)
            for r in error_rows:
                errors_seen += 1
                _write_error_log(
                    err_log, ds_cfg, prot, pop, gen, alpha, beta,
                    f"model={r['model_identifier']}: {r['error']}",
                )
            print(
                f"[{done:{width}}/{total}] "
                f"ds={ds_cfg['name']} prot={prot} "
                f"pop={pop} gen={gen} a={alpha} b={beta}  "
                f"ok={len(success_rows)} err={len(error_rows)}",
                flush=True,
            )
        except Exception as exc:
            errors_seen += 1
            _write_error_log(
                err_log, ds_cfg, prot, pop, gen, alpha, beta,
                str(exc), traceback.format_exc(),
            )
            print(
                f"[{done:{width}}/{total}] "
                f"FAILED ds={ds_cfg['name']} prot={prot} "
                f"pop={pop} gen={gen}: {exc}",
                flush=True,
            )

elapsed = time.time() - overall_start
print(flush=True)
print(f"Complete.  Elapsed: {elapsed / 3600:.2f} h ({elapsed:.0f} s)")
print(f"Errors: {errors_seen}" + (f"  — see {err_log}" if errors_seen else ""))
PYEOF

fi

# =============================================================================
# output validation
# =============================================================================
echo ""
echo "=== Output Validation ==="

if [[ ! -f "$SUMMARY_CSV" ]]; then
    echo "ERROR: Results file not found: $SUMMARY_CSV" >&2; exit 1
fi

ROW_COUNT=$(tail -n +2 "$SUMMARY_CSV" | wc -l | tr -d ' ')
if [[ "$ROW_COUNT" -lt 1 ]]; then
    echo "ERROR: Results file has no data rows." >&2; exit 1
fi

HEADER=$(head -1 "$SUMMARY_CSV")
COMMA_COUNT=$(printf '%s' "$HEADER" | tr -cd ',' | wc -c | tr -d ' ')
ACTUAL_COLS=$((COMMA_COUNT + 1))
if [[ "$ACTUAL_COLS" -eq "$EXPECTED_COLS" ]]; then
    COL_STATUS="$ACTUAL_COLS / $EXPECTED_COLS  [OK]"
else
    COL_STATUS="$ACTUAL_COLS / $EXPECTED_COLS  [WARNING: mismatch]"
fi

# ANSI escape code check via Python (grep -P not available on macOS)
ANSI_STATUS="none detected"
if ! $PYTHON - <<ANSICHECK
import sys
data = open('${SUMMARY_CSV}', 'rb').read()
sys.exit(0 if b'\x1b' not in data else 1)
ANSICHECK
then
    ANSI_STATUS="DETECTED — check output"
fi

ERROR_ENTRY_COUNT=0
if [[ -f "$ERRORS_LOG" ]]; then
    ERROR_ENTRY_COUNT=$(grep -c '^ERROR:' "$ERRORS_LOG" 2>/dev/null || echo 0)
fi

printf "\n  %-27s %s\n"  "Results file:"      "$SUMMARY_CSV"
printf   "  %-27s %s\n"  "Valid result rows:" "$ROW_COUNT"
printf   "  %-27s %s\n"  "Columns:"           "$COL_STATUS"
printf   "  %-27s %s\n"  "ANSI escape codes:" "$ANSI_STATUS"
if [[ -f "$ERRORS_LOG" ]]; then
    printf "  %-27s %s\n" "Errors log:"       "$ERRORS_LOG"
    printf "  %-27s %s\n" "Error entries:"    "$ERROR_ENTRY_COUNT"
else
    printf "  %-27s %s\n" "Errors log:"       "(not created — zero errors)"
fi

echo ""
echo "Done."
