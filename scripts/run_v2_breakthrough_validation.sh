#!/usr/bin/env bash
set -euo pipefail

# One-click server entrypoint. Pass --run-esmfold --run-foldseek --foldseek-db <DB>
# when GPU structure prediction and Foldseek search should be executed.
python scripts/run_v2_breakthrough_validation.py "$@"
