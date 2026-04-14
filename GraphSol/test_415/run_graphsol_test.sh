#!/usr/bin/env bash
set -euo pipefail

cd /workspace/GraphSol

mkdir -p test_415

PYTHON_BIN=/opt/conda/envs/graphsol/bin/python

cd /workspace/GraphSol/Predict
"${PYTHON_BIN}" /workspace/GraphSol/test_415/prepare_graphsol_input.py
"${PYTHON_BIN}" predict.py

cp Result/result.csv /workspace/GraphSol/test_415/result.csv
