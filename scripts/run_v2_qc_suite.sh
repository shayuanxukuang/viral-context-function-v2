#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ $# -lt 1 ]]; then
  echo "Usage: bash ./scripts/run_v2_qc_suite.sh <run-root> [extra args]" >&2
  exit 1
fi

run_root="$1"
shift

python -u ./scripts/run_v2_qc_suite.py \
  --run-root "$run_root" \
  "$@"
