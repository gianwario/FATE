#!/usr/bin/env bash
# Regenerate the code-quality evidence reported in docs/code_quality.md:
# flake8 (reviewer-provided configuration in .flake8), Radon cyclomatic
# complexity for the WHOLE codebase, and pytest with branch coverage.
# Requires: pip install -r requirements-dev.txt
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

echo "== flake8 (.flake8) =="
flake8 . && echo "flake8: 0 warnings"

echo ""
echo "== Radon cyclomatic complexity (all modules, all functions) =="
radon cc -s -a --exclude "venv/*,.venv/*" .

echo ""
echo "== pytest + branch coverage =="
coverage run --branch --source=. --omit="tests/*" -m pytest -q  # all modules, also untested ones
coverage report --show-missing
