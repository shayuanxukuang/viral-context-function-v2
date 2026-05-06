#!/usr/bin/env bash
set -euo pipefail

# One-key server entrypoint for the remaining PLOS CB strengthening analyses.
#
# Example:
#   ROOT=<PROJECT_ROOT>
#   RUN=$ROOT/runs/context_study_v2_20260430_100225
#   PY=$HOME/software/envs/promath_torch/bin/python
#   bash scripts/run_v2_review_completion.sh "$RUN"

RUN_ROOT="${1:?Usage: bash scripts/run_v2_review_completion.sh <RUN_ROOT> [OUTPUT_ROOT] [extra run_v2_review_completion.py args...]}"
shift
if [[ $# -gt 0 && "$1" != --* ]]; then
  OUTPUT_ROOT="$1"
  shift
else
  OUTPUT_ROOT="$RUN_ROOT/review_completion"
fi
PYTHON_BIN="${PYTHON_BIN:-${PY:-python}}"
GPU_IDS="${GPU_IDS:-4,5,6}"
THREADS="${THREADS:-24}"
SEEDS="${SEEDS:-42,43,44}"
MMSEQS_BIN="${MMSEQS_BIN:-mmseqs}"

cd "$(dirname "$0")/.."

"$PYTHON_BIN" -u scripts/run_v2_review_completion.py \
  --run-root "$RUN_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --python "$PYTHON_BIN" \
  --gpu-ids "$GPU_IDS" \
  --threads "$THREADS" \
  --seeds "$SEEDS" \
  --mmseqs-bin "$MMSEQS_BIN" \
  "$@"
