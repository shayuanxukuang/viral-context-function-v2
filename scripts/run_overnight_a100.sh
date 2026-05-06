#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

timestamp="$(date +%Y%m%d_%H%M%S)"
output_dir="runs/overnight_a100_${timestamp}"
prepare=0
force_prepare=0
compile_model=0
split_scheme="default_hash"
split_manifest=""
with_context=0
context_table=""
conda_env_name="${CONDA_ENV_NAME:-promath_torch}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-dir)
      output_dir="$2"
      shift 2
      ;;
    --prepare)
      prepare=1
      shift
      ;;
    --force-prepare)
      prepare=1
      force_prepare=1
      shift
      ;;
    --compile-model)
      compile_model=1
      shift
      ;;
    --split-scheme)
      split_scheme="$2"
      shift 2
      ;;
    --split-manifest)
      split_manifest="$2"
      shift 2
      ;;
    --with-context)
      with_context=1
      shift
      ;;
    --context-table)
      with_context=1
      context_table="$2"
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
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ "${CONDA_DEFAULT_ENV:-}" != "$conda_env_name" ]]; then
  if ! command -v conda >/dev/null 2>&1; then
    echo "Conda is required but was not found on PATH." >&2
    echo "Activate an existing environment or set CONDA_ENV_NAME before running." >&2
    exit 1
  fi

  conda_base="$(conda info --base 2>/dev/null || true)"
  if [[ -z "$conda_base" || ! -f "$conda_base/etc/profile.d/conda.sh" ]]; then
    echo "Unable to locate conda.sh for environment activation." >&2
    exit 1
  fi

  # Conda activation hooks are not consistently compatible with nounset.
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

mkdir -p "$output_dir"
log_path="$output_dir/train.log"
exec > >(tee -a "$log_path") 2>&1

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"

echo "[run] root=$ROOT_DIR"
echo "[run] output_dir=$output_dir"
echo "[run] log_path=$log_path"
echo "[run] started_at=$(date -Is)"
echo "[run] conda_env=${CONDA_DEFAULT_ENV:-unknown}"
echo "[run] python=$(command -v python)"
echo "[run] python_version=$(python -V 2>&1)"
echo "[run] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[run] compile_model=$compile_model"
echo "[run] split_scheme=$split_scheme"
echo "[run] split_manifest=${split_manifest:-auto}"
echo "[run] with_context=$with_context"
echo "[run] context_table=${context_table:-auto}"

if [[ $prepare -eq 1 || ! -f data/processed/training/viral_protein_training_index.tsv.gz ]]; then
  bootstrap_args=("--skip-download" "--build-tables")
  if [[ $force_prepare -eq 1 ]]; then
    bootstrap_args=("--skip-download" "--extract-archives" "--build-tables" "--force")
  fi
  echo "[run] preparing data via bootstrap: ${bootstrap_args[*]}"
  bash ./scripts/bootstrap.sh "${bootstrap_args[@]}"
fi

if [[ "$split_scheme" != "default_hash" ]]; then
  effective_split_manifest="${split_manifest:-data/processed/splits/viral_protein_strict_splits.tsv.gz}"
  if [[ ! -f "$effective_split_manifest" ]]; then
    if [[ -n "$split_manifest" ]]; then
      echo "[run] requested split manifest was not found: $split_manifest" >&2
      exit 1
    fi
    echo "[run] strict split manifest missing; generating $effective_split_manifest"
    python ./scripts/build_strict_splits.py
  fi
fi

effective_context_table=""
if [[ $with_context -eq 1 ]]; then
  effective_context_table="${context_table:-data/processed/training/viral_protein_context_features.tsv.gz}"
  if [[ ! -f "$effective_context_table" ]]; then
    if [[ -n "$context_table" ]]; then
      echo "[run] requested context table was not found: $context_table" >&2
      exit 1
    fi
    echo "[run] context table missing; generating $effective_context_table"
    python ./scripts/build_context_features.py
  fi
fi

train_args=(
  python ./scripts/train_overnight_baseline.py
  --input data/processed/training/viral_protein_training_index.tsv.gz
  --output-dir "$output_dir"
  --device cuda:0
  --save-test-predictions
  --epochs 12
  --batch-size 1024
  --eval-batch-size 2048
  --max-length 2048
  --embed-dim 128
  --hidden-dim 256
  --learning-rate 3e-4
  --weight-decay 1e-2
  --num-workers 8
  --prefetch-factor 4
)

if [[ $compile_model -eq 1 ]]; then
  train_args+=(--compile-model)
fi

if [[ "$split_scheme" != "default_hash" ]]; then
  train_args+=(--split-scheme "$split_scheme")
  if [[ -n "$split_manifest" ]]; then
    train_args+=(--split-manifest "$split_manifest")
  fi
fi

if [[ $with_context -eq 1 ]]; then
  train_args+=(--context-table "$effective_context_table")
fi

"${train_args[@]}"

echo "[run] finished_at=$(date -Is)"
