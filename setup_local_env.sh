#!/usr/bin/env bash
set -euo pipefail

ENV_DIR="${1:-.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
"$PYTHON_BIN" -m venv "$ENV_DIR"
"$ENV_DIR/bin/python" -m pip install --upgrade pip
"$ENV_DIR/bin/python" -m pip install -r experiments/requirements-experiments.txt
echo
echo "Environment ready. Activate it with: source $ENV_DIR/bin/activate"
echo "Then verify it with: python experiments/check_environment.py --architecture unet"
