#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${SURFSOL_ENV:-SurfSol_aarch64}"

cd "${REPO_DIR}"

conda run -n "${ENV_NAME}" python run_surfsol_prediction.py "$@"
