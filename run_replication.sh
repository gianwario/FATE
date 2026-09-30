#!/usr/bin/env bash
# =============================================================================
# run_replication.sh — one entry point for the whole replication pipeline
#
# Modes (see README.md, "Replication pipeline"):
#
#   --mode fast       One FATE GA run (default: adult / lr / sex / N=5 / G=5)
#                     to check that the installation works.
#   --mode full       Stage 1: the FATE parameter grid of the paper (Table 2).
#   --mode analysis   Stages 2-3: RQ1 and RQ2 analyses, tables and figures,
#                     computed from the Stage-1 output in results/fate/.
#                     Add --from-reference to start from the paper's archived
#                     grid (reference/fate/experiments_results.csv) instead.
#   --mode all        full + analysis, i.e. the complete replication from scratch.
#
# All Python logic lives in main.py and in the RQ1/RQ2 modules; this script
# only checks the environment and calls them in the right order.
# =============================================================================
set -euo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# =============================================================================
# CONFIGURABLE defaults (CLI flags override them)
# =============================================================================
DEFAULT_DATASET="adult"   # adult | german | heart (see experiment_config.py)
DEFAULT_MODEL="lr"        # lr | rf | svc | xgb
DEFAULT_POP=5             # fast mode population size
DEFAULT_GEN=5             # fast mode number of generations
WORKERS=64                # worker threads for the FATE grid
# Cache mode (see README.md, "Fitness cache"):
#   false — fitness lookups also use the archived cache reference/fate/fitness_cache.csv
#   true  — ignore it; every fitness value is computed during this run
RESET_CACHE=false

# =============================================================================
# internals
# =============================================================================
FATE_DIR="results/fate"
SUMMARY_CSV="${FATE_DIR}/experiments_results.csv"
ERRORS_LOG="${FATE_DIR}/errors.log"
EXPECTED_COLS=13

MODE=""
POP_OVERRIDE=""
GEN_OVERRIDE=""
DATASET_OVERRIDE=""
MODEL_OVERRIDE=""
FROM_REFERENCE=false

usage() {
    cat <<'USAGE'
Usage: ./run_replication.sh --mode fast|full|analysis|all [OPTIONS]

Modes:
  fast        One GA run (installation check).
  full        FATE parameter grid of the paper  -> results/fate/
  analysis    RQ1 + RQ2 from results/fate/      -> results/rq1/, results/rq2/
  all         full, then analysis (complete replication from scratch)

Options:
  --dataset NAME     adult | german | heart   (fast: dataset; full: restrict grid)
  --model   NAME     lr | rf | svc | xgb      (fast: model;   full: restrict grid)
  --pop N            population size          (fast: value;   full: pin value)
  --gen N            number of generations    (fast: value;   full: pin value)
  --workers N        worker threads for the grid (default 64)
  --reset-cache      do not use the archived fitness cache
  --from-reference   analysis mode: start from reference/fate/experiments_results.csv
  -h, --help         show this message

Examples:
  ./run_replication.sh --mode fast
  ./run_replication.sh --mode all --reset-cache
  ./run_replication.sh --mode analysis --from-reference
USAGE
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)            MODE="$2";             shift 2 ;;
        --pop)             POP_OVERRIDE="$2";     shift 2 ;;
        --gen)             GEN_OVERRIDE="$2";     shift 2 ;;
        --dataset)         DATASET_OVERRIDE="$2"; shift 2 ;;
        --model)           MODEL_OVERRIDE="$2";   shift 2 ;;
        --workers)         WORKERS="$2";          shift 2 ;;
        --reset-cache)     RESET_CACHE=true;      shift   ;;
        --from-reference)  FROM_REFERENCE=true;   shift   ;;
        -h|--help) usage ;;
        *) echo "ERROR: Unknown argument: $1 (see --help)" >&2; exit 1 ;;
    esac
done

case "$MODE" in
    fast|full|analysis|all) ;;
    *) echo "ERROR: --mode must be fast | full | analysis | all (see --help)" >&2; exit 1 ;;
esac
for _v in "$POP_OVERRIDE" "$GEN_OVERRIDE" "$WORKERS"; do
    if [[ -n "$_v" ]] && ! [[ "$_v" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: --pop/--gen/--workers must be positive integers; got '$_v'." >&2; exit 1
    fi
done

# =============================================================================
# environment check
# =============================================================================
echo ""
echo "=== Environment Check ==="
PYTHON=""
for _cand in python3.10 python3 python; do
    if command -v "$_cand" &>/dev/null &&
       [[ "$("$_cand" -c 'import sys; print("%d.%d" % sys.version_info[:2])')" == "3.10" ]]; then
        PYTHON="$_cand"; break
    fi
done
if [[ -z "$PYTHON" ]]; then
    echo "ERROR: Python 3.10 is required. Activate the virtual environment first:" >&2
    echo "         source venv/bin/activate" >&2
    exit 1
fi
echo "Python:   $($PYTHON --version)"
MISSING_PKGS=()
for _pkg in pandas numpy sklearn scipy aif360 xgboost fairlearn BlackBoxAuditing matplotlib seaborn; do
    $PYTHON -c "import $_pkg" 2>/dev/null || MISSING_PKGS+=("$_pkg")
done
if [[ ${#MISSING_PKGS[@]} -gt 0 ]]; then
    echo "ERROR: Missing packages: ${MISSING_PKGS[*]}. Run: pip install -r requirements.txt" >&2
    exit 1
fi
echo "Packages: all present"

# =============================================================================
# stage 1 — FATE (fast / full)
# =============================================================================
run_fate() {
    local args=()
    if [[ "$MODE" == "fast" ]]; then
        args+=(--datasets "${DATASET_OVERRIDE:-$DEFAULT_DATASET}")
        args+=(--models "${MODEL_OVERRIDE:-$DEFAULT_MODEL}")
        args+=(--protected sex --alpha 0.5 --beta 0.5 --workers 1)
        args+=(--pop "${POP_OVERRIDE:-$DEFAULT_POP}" --gen "${GEN_OVERRIDE:-$DEFAULT_GEN}")
    else
        [[ -n "$DATASET_OVERRIDE" ]] && args+=(--datasets "$DATASET_OVERRIDE")
        [[ -n "$MODEL_OVERRIDE"   ]] && args+=(--models "$MODEL_OVERRIDE")
        [[ -n "$POP_OVERRIDE"     ]] && args+=(--pop "$POP_OVERRIDE")
        [[ -n "$GEN_OVERRIDE"     ]] && args+=(--gen "$GEN_OVERRIDE")
        args+=(--workers "$WORKERS")
    fi
    [[ "$RESET_CACHE" == "true" ]] && args+=(--reset-cache)
    echo ""
    echo "=== Stage 1: FATE ($MODE) ==="
    echo "  $PYTHON main.py $(printf '%s ' "${args[@]}")"
    "$PYTHON" main.py "${args[@]}"
    validate_fate_output
}

validate_fate_output() {
    echo ""
    echo "=== Stage 1 output validation ==="
    [[ -f "$SUMMARY_CSV" ]] || { echo "ERROR: $SUMMARY_CSV not found" >&2; exit 1; }
    local rows cols
    rows=$(tail -n +2 "$SUMMARY_CSV" | wc -l | tr -d ' ')
    [[ "$rows" -ge 1 ]] || { echo "ERROR: $SUMMARY_CSV has no data rows" >&2; exit 1; }
    cols=$(( $(head -1 "$SUMMARY_CSV" | tr -cd ',' | wc -c) + 1 ))
    printf "  %-20s %s\n" "Results file:" "$SUMMARY_CSV"
    printf "  %-20s %s\n" "Result rows:" "$rows"
    printf "  %-20s %s\n" "Columns:" "$cols / $EXPECTED_COLS"
    if [[ -f "$ERRORS_LOG" ]]; then
        printf "  %-20s %s (%s entries)\n" "Errors log:" "$ERRORS_LOG" \
            "$(grep -c '^ERROR:' "$ERRORS_LOG" || true)"
    else
        printf "  %-20s %s\n" "Errors log:" "none (no failed runs)"
    fi
    [[ "$cols" -eq "$EXPECTED_COLS" ]] || { echo "ERROR: unexpected column count" >&2; exit 1; }
}

# =============================================================================
# stages 2-3 — RQ1 and RQ2 analyses
# =============================================================================
run_analysis() {
    local grid="$SUMMARY_CSV"
    [[ "$FROM_REFERENCE" == "true" ]] && grid="reference/fate/experiments_results.csv"
    [[ -f "$grid" ]] || { echo "ERROR: $grid not found; run --mode full first" >&2; exit 1; }
    local rq2_args=()
    [[ "$RESET_CACHE" == "true" ]] && rq2_args+=(--reset-cache)
    echo ""
    echo "=== Stage 2: RQ1 (input: $grid) ==="
    "$PYTHON" -m RQ1_data_analysis.configurations_results --results "$grid"
    "$PYTHON" -m RQ1_data_analysis.model_dataset_results --results "$grid"
    "$PYTHON" -m RQ1_data_analysis.rq1_results
    "$PYTHON" -m RQ1_data_analysis.rq1_visualizations
    echo ""
    echo "=== Stage 3: RQ2 ==="
    "$PYTHON" -m RQ2_data_analysis.preprocessing_experiments ${rq2_args[@]+"${rq2_args[@]}"}
    "$PYTHON" -m RQ2_data_analysis.rq2_results
    echo ""
    echo "Outputs: results/rq1/ (tables), results/rq1/figures/, results/rq2/"
}

case "$MODE" in
    fast|full) run_fate ;;
    analysis)  run_analysis ;;
    all)       MODE=full; run_fate; run_analysis ;;
esac
echo ""
echo "Done."
