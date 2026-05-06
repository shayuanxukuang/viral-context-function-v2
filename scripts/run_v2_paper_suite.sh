#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

conda_env_name="${CONDA_ENV_NAME:-promath_torch}"
if [[ "${CONDA_DEFAULT_ENV:-}" != "$conda_env_name" ]]; then
  if ! command -v conda >/dev/null 2>&1; then
    echo "Conda is required but was not found on PATH." >&2
    echo "Activate $conda_env_name first or set CONDA_ENV_NAME to the environment you want." >&2
    exit 1
  fi

  conda_base="$(conda info --base 2>/dev/null || true)"
  if [[ -z "$conda_base" || ! -f "$conda_base/etc/profile.d/conda.sh" ]]; then
    echo "Unable to locate conda.sh for environment activation." >&2
    exit 1
  fi

  set +u
  # shellcheck disable=SC1090
  source "$conda_base/etc/profile.d/conda.sh"
  conda activate "$conda_env_name"
  set -u
fi

if [[ "${CONDA_DEFAULT_ENV:-}" != "$conda_env_name" ]]; then
  echo "Expected conda environment '$conda_env_name' to be active." >&2
  exit 1
fi

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

python -u ./scripts/run_v2_paper_suite.py \
  --gpu-ids "${V2_GPU_IDS:-4,5,6}" \
  "$@"
