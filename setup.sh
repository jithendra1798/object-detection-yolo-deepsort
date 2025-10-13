#!/usr/bin/env bash
set -euo pipefail

PY=python3
VENV=.venv

echo "Creating virtualenv in ${VENV}..."
$PY -m venv ${VENV}
echo "Activating and upgrading pip..."
${VENV}/bin/pip install -U pip
echo "Installing requirements..."
${VENV}/bin/pip install -r requirements.txt
echo "Done. To run the app:"
echo "  source ${VENV}/bin/activate"
echo "  python webapp.py"
