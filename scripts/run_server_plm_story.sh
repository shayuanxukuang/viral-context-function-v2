#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

timestamp="$(date +%Y%m%d_%H%M%S)"
study_root="runs/plm_story_${timestamp}"
plm_embedding_path="data/processed/plm/esm2_t33_650M_UR50D_embeddings.pt"
model_name="facebook/esm2_t33_650M_UR50D"
device="cuda:0"
cache_device="cpu"
batch_size=256
eval_batch_size=512
epochs=12
num_workers=8
prefetch_factor=4
neighbor_radius=2
bootstrap_iterations=200
permutation_iterations=200
discovery_run_name="genome_aware_denovo.family_holdout"
run_discovery=1
run_qc=1
run_freeze=0
run_study=1
run_summary=1
run_source_summary=1
run_atlas=1
conda_env_name="${CONDA_ENV_NAME:-promath_torch}"
log_path=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --study-root)
      study_root="$2"
      shift 2
      ;;
    --plm-embedding-path)
      plm_embedding_path="$2"
      shift 2
      ;;
    --model-name)
      model_name="$2"
      shift 2
      ;;
    --device)
      device="$2"
      shift 2
      ;;
    --cache-device)
      cache_device="$2"
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
    --epochs)
      epochs="$2"
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
    --neighbor-radius)
      neighbor_radius="$2"
      shift 2
      ;;
    --bootstrap-iterations)
      bootstrap_iterations="$2"
      shift 2
      ;;
    --permutation-iterations)
      permutation_iterations="$2"
      shift 2
      ;;
    --discovery-run-name)
      discovery_run_name="$2"
      shift 2
      ;;
    --skip-discovery)
      run_discovery=0
      shift
      ;;
    --skip-study)
      run_study=0
      shift
      ;;
    --skip-summary)
      run_summary=0
      shift
      ;;
    --skip-source-decomposition)
      run_source_summary=0
      shift
      ;;
    --skip-atlas)
      run_atlas=0
      shift
      ;;
    --skip-qc)
      run_qc=0
      shift
      ;;
    --refresh-freeze)
      run_freeze=1
      shift
      ;;
    --log-path)
      log_path="$2"
      shift 2
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

mkdir -p "$study_root"
if [[ -z "$log_path" ]]; then
  log_path="$study_root/server_story.log"
fi
exec > >(tee -a "$log_path") 2>&1

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

run_logged() {
  local step_name="$1"
  shift
  echo "[story] >>> $step_name"
  echo "[story] cwd=$ROOT_DIR"
  echo "[story] cmd=$*"
  "$@"
}

require_path() {
  local target_path="$1"
  local label="$2"
  if [[ ! -e "$target_path" ]]; then
    echo "[story] missing required ${label}: $target_path" >&2
    exit 1
  fi
}

echo "[story] root=$ROOT_DIR"
echo "[story] study_root=$study_root"
echo "[story] log_path=$log_path"
echo "[story] started_at=$(date -Is)"
echo "[story] conda_env=${CONDA_DEFAULT_ENV:-unknown}"
echo "[story] python=$(command -v python)"
echo "[story] python_version=$(python -V 2>&1)"
echo "[story] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[story] device=$device"
echo "[story] cache_device=$cache_device"
echo "[story] plm_embedding_path=$plm_embedding_path"

if [[ $run_study -eq 1 ]]; then
  if [[ ! -f "$plm_embedding_path" ]]; then
    mkdir -p "$(dirname "$plm_embedding_path")"
    run_logged "cache_plm_embeddings" \
      python -u ./scripts/cache_plm_embeddings.py \
        --output "$plm_embedding_path" \
        --model-name "$model_name" \
        --device "$cache_device"
  else
    echo "[story] reuse existing PLM embeddings: $plm_embedding_path"
  fi
else
  echo "[story] skip PLM embedding cache because --skip-study was set"
fi

if [[ $run_qc -eq 1 ]]; then
  run_logged "qc_biophysics_features" \
    python -u ./scripts/qc_biophysics_features.py \
      --output-dir "$study_root/biophysics_qc"
fi

if [[ $run_study -eq 1 ]]; then
  run_logged "run_context_study_plm" \
    python -u ./scripts/run_context_study.py \
      --output-root "$study_root" \
      --sequence-backbone precomputed_plm \
      --plm-embedding-path "$plm_embedding_path" \
      --skip-atlas \
      --device "$device" \
      --epochs "$epochs" \
      --batch-size "$batch_size" \
      --eval-batch-size "$eval_batch_size" \
      --num-workers "$num_workers" \
      --prefetch-factor "$prefetch_factor" \
      --neighbor-radius "$neighbor_radius"
fi

if [[ $run_summary -eq 1 ]]; then
  require_path "$study_root/suite_summary.json" "suite summary"
  run_logged "summarize_plm_quartet" \
    python -u ./scripts/summarize_plm_quartet.py \
      --input-root "$study_root" \
      --cnn-baseline-root runs/task_mode_suite_server \
      --output "$study_root/plm_quartet_summary.tsv"
fi

if [[ $run_source_summary -eq 1 ]]; then
  require_path "$study_root/suite_summary.json" "suite summary"
  run_logged "summarize_source_decomposition" \
    python -u ./scripts/summarize_source_decomposition.py \
      --input "$study_root" \
      --output-dir "$study_root/source_decomposition"
fi

if [[ $run_atlas -eq 1 ]]; then
  require_path "$study_root/protein_only.family_holdout" "family protein run"
  require_path "$study_root/genome_aware_denovo.family_holdout" "family context run"
  require_path "$study_root/protein_only.host_holdout" "host protein run"
  require_path "$study_root/genome_aware_denovo.host_holdout" "host context run"

  run_logged "atlas_family_plm" \
    python -u ./scripts/build_context_dependence_atlas_v2.py \
      --protein-run "$study_root/protein_only.family_holdout" \
      --context-run "$study_root/genome_aware_denovo.family_holdout" \
      --output-dir "$study_root/context_atlas_plain.family_holdout.v2" \
      --bootstrap-iterations "$bootstrap_iterations" \
      --permutation-iterations "$permutation_iterations"

  run_logged "atlas_host_plm" \
    python -u ./scripts/build_context_dependence_atlas_v2.py \
      --protein-run "$study_root/protein_only.host_holdout" \
      --context-run "$study_root/genome_aware_denovo.host_holdout" \
      --output-dir "$study_root/context_atlas_plain.host_holdout.v2" \
      --bootstrap-iterations "$bootstrap_iterations" \
      --permutation-iterations "$permutation_iterations"
fi

if [[ $run_discovery -eq 1 ]]; then
  discovery_root="$study_root/module_discovery"
  embedding_export="$discovery_root/exported_fused_test_embeddings.pt"
  best_run_dir="$study_root/$discovery_run_name"
  require_path "$best_run_dir" "discovery run directory"
  run_logged "export_task_mode_embeddings" \
    python -u ./scripts/export_task_mode_embeddings.py \
      --run-dir "$best_run_dir" \
      --output "$embedding_export" \
      --representation fused \
      --split test \
      --device "$device"

  run_logged "discover_module_candidates" \
    python -u ./scripts/discover_module_candidates.py \
      --embedding-file "$embedding_export" \
      --output-dir "$discovery_root"

  if [[ -f "$discovery_root/ranked_hypothetical_clusters.tsv" && -f "$discovery_root/module_candidates.tsv" ]]; then
    run_logged "prepare_targeted_structure_validation" \
      python -u ./scripts/prepare_targeted_structure_validation.py \
        --ranked-clusters "$discovery_root/ranked_hypothetical_clusters.tsv" \
        --module-candidates "$discovery_root/module_candidates.tsv" \
        --output-dir "$discovery_root/targeted_structure_validation"
  fi
fi

if [[ $run_freeze -eq 1 ]]; then
  run_logged "freeze_benchmark_v1" \
    python -u ./scripts/freeze_benchmark_v1.py
fi

echo "[story] finished_at=$(date -Is)"
echo "[story] study_root=$study_root"
echo "[story] quartet_summary=$study_root/plm_quartet_summary.tsv"
