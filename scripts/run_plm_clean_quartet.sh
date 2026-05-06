#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

output_root="runs/plm_clean_quartet"
plm_embedding_path=""
passthrough_args=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-root)
      output_root="$2"
      shift 2
      ;;
    --plm-embedding-path)
      plm_embedding_path="$2"
      shift 2
      ;;
    *)
      passthrough_args+=("$1")
      shift
      ;;
  esac
done

if [[ -z "$plm_embedding_path" ]]; then
  echo "--plm-embedding-path is required" >&2
  exit 1
fi

bash ./scripts/run_task_mode_suite.sh \
  --output-root "$output_root" \
  --sequence-backbone precomputed_plm \
  --plm-embedding-path "$plm_embedding_path" \
  "${passthrough_args[@]}"

python ./scripts/summarize_plm_quartet.py \
  --input-root "$output_root" \
  --output "$output_root/plm_quartet_summary.tsv"
