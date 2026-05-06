#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

timestamp="$(date +%Y%m%d_%H%M%S)"
split_scheme="family_holdout"
output_dir="runs/context_family_a100_${timestamp}"
context_table=""
passthrough_args=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-dir)
      output_dir="$2"
      shift 2
      ;;
    --family-heldout)
      split_scheme="family_holdout"
      shift
      ;;
    --host-heldout)
      split_scheme="host_holdout"
      shift
      ;;
    --split-scheme)
      split_scheme="$2"
      shift 2
      ;;
    --context-table)
      context_table="$2"
      shift 2
      ;;
    *)
      passthrough_args+=("$1")
      shift
      ;;
  esac
done

cmd=(
  bash ./scripts/run_overnight_a100.sh
  --with-context
  --split-scheme "$split_scheme"
  --output-dir "$output_dir"
)

if [[ -n "$context_table" ]]; then
  cmd+=(--context-table "$context_table")
fi

if [[ ${#passthrough_args[@]} -gt 0 ]]; then
  cmd+=("${passthrough_args[@]}")
fi

"${cmd[@]}"
