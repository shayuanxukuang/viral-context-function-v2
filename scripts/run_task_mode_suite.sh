#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

timestamp="$(date +%Y%m%d_%H%M%S)"
output_root="runs/task_mode_suite_${timestamp}"
prepare=0
force_prepare=0
run_family=1
run_host=1
compile_model=0
sequence_backbone="cnn"
plm_embedding_path=""
device="cuda:0"
epochs=12
batch_size=512
eval_batch_size=1024
max_length=2048
embed_dim=128
hidden_dim=256
learning_rate=3e-4
weight_decay=1e-2
num_workers=8
prefetch_factor=4
min_label_count=500
gradient_clip=1.0
warmup_fraction=0.05
max_pos_weight=50.0
neighbor_radius=2
save_test_predictions=1
with_static_artifacts=1
dry_run=0
include_annotation_refinement=0
conda_env_name="${CONDA_ENV_NAME:-promath_torch}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-root)
      output_root="$2"
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
    --family-only)
      run_family=1
      run_host=0
      shift
      ;;
    --host-only)
      run_family=0
      run_host=1
      shift
      ;;
    --compile-model)
      compile_model=1
      shift
      ;;
    --sequence-backbone)
      sequence_backbone="$2"
      shift 2
      ;;
    --plm-embedding-path)
      plm_embedding_path="$2"
      shift 2
      ;;
    --device)
      device="$2"
      shift 2
      ;;
    --epochs)
      epochs="$2"
      shift 2
      ;;
    --batch-size)
      batch_size="$2"
      shift 2
      ;;
    --eval-batch-size)
      eval_batch_size="$2"
      shift 2
      ;;
    --max-length)
      max_length="$2"
      shift 2
      ;;
    --embed-dim)
      embed_dim="$2"
      shift 2
      ;;
    --hidden-dim)
      hidden_dim="$2"
      shift 2
      ;;
    --learning-rate)
      learning_rate="$2"
      shift 2
      ;;
    --weight-decay)
      weight_decay="$2"
      shift 2
      ;;
    --num-workers)
      num_workers="$2"
      shift 2
      ;;
    --prefetch-factor)
      prefetch_factor="$2"
      shift 2
      ;;
    --min-label-count)
      min_label_count="$2"
      shift 2
      ;;
    --gradient-clip)
      gradient_clip="$2"
      shift 2
      ;;
    --warmup-fraction)
      warmup_fraction="$2"
      shift 2
      ;;
    --max-pos-weight)
      max_pos_weight="$2"
      shift 2
      ;;
    --neighbor-radius)
      neighbor_radius="$2"
      shift 2
      ;;
    --no-save-test-predictions)
      save_test_predictions=0
      shift
      ;;
    --skip-static-artifacts)
      with_static_artifacts=0
      shift
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    --include-annotation-refinement)
      include_annotation_refinement=1
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ $run_family -eq 0 && $run_host -eq 0 ]]; then
  echo "Nothing to run: both family and host benchmarks are disabled." >&2
  exit 1
fi

if [[ "$sequence_backbone" == "precomputed_plm" && -z "$plm_embedding_path" ]]; then
  echo "--plm-embedding-path is required when --sequence-backbone precomputed_plm" >&2
  exit 1
fi

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

mkdir -p "$output_root"
suite_log="$output_root/suite.log"
exec > >(tee -a "$suite_log") 2>&1

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

run_logged() {
  local run_name="$1"
  shift
  local log_path="$output_root/${run_name}.log"
  echo "[suite] >>> $run_name"
  echo "[suite] log=$log_path"
  if [[ $dry_run -eq 1 ]]; then
    printf '[dry-run] '
    printf '%q ' "$@"
    printf '\n'
    return 0
  fi
  "$@" 2>&1 | tee "$log_path"
}

ensure_context_table() {
  local split_scheme="$1"
  local task_mode="$2"
  local context_path="data/processed/context/viral_protein_context.${split_scheme}.${task_mode}.tsv.gz"
  if [[ ! -f "$context_path" ]]; then
    run_logged "build_context.${split_scheme}.${task_mode}" \
      python -u ./scripts/build_context_features_splitaware.py \
        --split-scheme "$split_scheme" \
        --task-mode "$task_mode" \
        --output-dir data/processed/context
  fi
}

run_one_training() {
  local split_scheme="$1"
  local task_mode="$2"
  local run_name="$3"
  shift 3
  local run_dir="$output_root/$run_name"
  mkdir -p "$run_dir"

  local context_table=""
  local extra_args=("$@")
  if [[ "$task_mode" != "protein_only" ]]; then
    ensure_context_table "$split_scheme" "$task_mode"
    context_table="data/processed/context/viral_protein_context.${split_scheme}.${task_mode}.tsv.gz"
    extra_args+=(--context-table "$context_table")
  fi
  if [[ "$sequence_backbone" == "precomputed_plm" ]]; then
    extra_args+=(--sequence-backbone precomputed_plm --plm-embedding-path "$plm_embedding_path")
  fi
  if [[ $compile_model -eq 1 ]]; then
    extra_args+=(--compile-model)
  fi
  if [[ $save_test_predictions -eq 1 ]]; then
    extra_args+=(--save-test-predictions)
  fi

  run_logged "train.${run_name}" \
    python -u ./scripts/train_task_modes.py \
      --input data/processed/training/viral_protein_training_index.tsv.gz \
      --output-dir "$run_dir" \
      --split-scheme "$split_scheme" \
      --task-mode "$task_mode" \
      --device "$device" \
      --epochs "$epochs" \
      --batch-size "$batch_size" \
      --eval-batch-size "$eval_batch_size" \
      --max-length "$max_length" \
      --embed-dim "$embed_dim" \
      --hidden-dim "$hidden_dim" \
      --learning-rate "$learning_rate" \
      --weight-decay "$weight_decay" \
      --num-workers "$num_workers" \
      --prefetch-factor "$prefetch_factor" \
      --min-label-count "$min_label_count" \
      --gradient-clip "$gradient_clip" \
      --warmup-fraction "$warmup_fraction" \
      --max-pos-weight "$max_pos_weight" \
      --neighbor-radius "$neighbor_radius" \
      "${extra_args[@]}"
}

echo "[suite] root=$ROOT_DIR"
echo "[suite] output_root=$output_root"
echo "[suite] suite_log=$suite_log"
echo "[suite] started_at=$(date -Is)"
echo "[suite] conda_env=${CONDA_DEFAULT_ENV:-unknown}"
echo "[suite] python=$(command -v python)"
echo "[suite] python_version=$(python -V 2>&1)"
echo "[suite] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[suite] PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"
echo "[suite] sequence_backbone=$sequence_backbone"
echo "[suite] device=$device"
echo "[suite] batch_size=$batch_size"
echo "[suite] eval_batch_size=$eval_batch_size"

if [[ $prepare -eq 1 || ! -f data/processed/training/viral_protein_training_index.tsv.gz ]]; then
  bootstrap_args=("--skip-download" "--build-tables")
  if [[ $force_prepare -eq 1 ]]; then
    bootstrap_args=("--skip-download" "--extract-archives" "--build-tables" "--force")
  fi
  run_logged "bootstrap" bash ./scripts/bootstrap.sh "${bootstrap_args[@]}"
fi

if [[ ! -f data/processed/splits/viral_protein_strict_splits.tsv.gz ]]; then
  run_logged "build_strict_splits" python -u ./scripts/build_strict_splits.py
fi

if [[ $with_static_artifacts -eq 1 ]]; then
  run_logged "audit_feature_leakage" python -u ./scripts/audit_feature_leakage.py --output-dir data/audits
  run_logged "sample_gold_eval_set" \
    python -u ./scripts/sample_gold_eval_set.py \
      --split-scheme family_holdout \
      --positive-per-label 20 \
      --negative-per-label 10 \
      --output-dir data/gold_eval
fi

if [[ $run_family -eq 1 ]]; then
  run_one_training "family_holdout" "protein_only" "protein_only.family_holdout"
  run_one_training "family_holdout" "protein_only" "protein_only_biophysics.family_holdout" --with-biophysics
  run_one_training "family_holdout" "genome_aware_denovo" "genome_aware_denovo.family_holdout"
  run_one_training "family_holdout" "genome_aware_denovo" "genome_aware_denovo_biophysics.family_holdout" --with-biophysics
  if [[ $include_annotation_refinement -eq 1 ]]; then
    run_one_training "family_holdout" "annotation_refinement" "annotation_refinement.family_holdout" --with-biophysics
  fi
fi

if [[ $run_host -eq 1 ]]; then
  run_one_training "host_holdout" "protein_only" "protein_only.host_holdout"
  run_one_training "host_holdout" "protein_only" "protein_only_biophysics.host_holdout" --with-biophysics
  run_one_training "host_holdout" "genome_aware_denovo" "genome_aware_denovo.host_holdout"
  run_one_training "host_holdout" "genome_aware_denovo" "genome_aware_denovo_biophysics.host_holdout" --with-biophysics
  if [[ $include_annotation_refinement -eq 1 ]]; then
    run_one_training "host_holdout" "annotation_refinement" "annotation_refinement.host_holdout" --with-biophysics
  fi
fi

run_logged "collect_task_mode_results" python -u ./scripts/collect_task_mode_results.py --input-root "$output_root"

echo "[suite] finished_at=$(date -Is)"
echo "[suite] results_root=$output_root"
