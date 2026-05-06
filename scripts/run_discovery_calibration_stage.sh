#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

study_root="runs/plm_story_server_20260423"
best_run_name="genome_aware_denovo.family_holdout"
device="cuda:0"
batch_size=512
num_workers=8
prefetch_factor=4
export_split="test"
representation="fused"
fdr_target=0.1
bootstrap_iterations=500
cluster_method="auto"
min_cluster_size=5
window_radius=1
top_casebooks=20
top_structure_clusters=5
representatives_per_cluster=3
conda_env_name="${CONDA_ENV_NAME:-promath_torch}"
run_analysis=1
run_calibration=1
run_discovery=1
run_structure_prep=1
run_package=1
log_path=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --study-root)
      study_root="$2"
      shift 2
      ;;
    --best-run-name)
      best_run_name="$2"
      shift 2
      ;;
    --device)
      device="$2"
      shift 2
      ;;
    --batch-size)
      batch_size="$2"
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
    --export-split)
      export_split="$2"
      shift 2
      ;;
    --representation)
      representation="$2"
      shift 2
      ;;
    --fdr-target)
      fdr_target="$2"
      shift 2
      ;;
    --bootstrap-iterations)
      bootstrap_iterations="$2"
      shift 2
      ;;
    --cluster-method)
      cluster_method="$2"
      shift 2
      ;;
    --min-cluster-size)
      min_cluster_size="$2"
      shift 2
      ;;
    --window-radius)
      window_radius="$2"
      shift 2
      ;;
    --top-casebooks)
      top_casebooks="$2"
      shift 2
      ;;
    --top-structure-clusters)
      top_structure_clusters="$2"
      shift 2
      ;;
    --representatives-per-cluster)
      representatives_per_cluster="$2"
      shift 2
      ;;
    --skip-analysis)
      run_analysis=0
      shift
      ;;
    --skip-calibration)
      run_calibration=0
      shift
      ;;
    --skip-discovery)
      run_discovery=0
      shift
      ;;
    --skip-structure-prep)
      run_structure_prep=0
      shift
      ;;
    --skip-package)
      run_package=0
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
  log_path="$study_root/discovery_calibration_stage.log"
fi
exec > >(tee -a "$log_path") 2>&1

run_logged() {
  local step_name="$1"
  shift
  echo "[discovery-stage] >>> $step_name"
  echo "[discovery-stage] cmd=$*"
  "$@"
}

require_path() {
  local target_path="$1"
  local label="$2"
  if [[ ! -e "$target_path" ]]; then
    echo "[discovery-stage] missing required ${label}: $target_path" >&2
    exit 1
  fi
}

best_run_dir="$study_root/$best_run_name"
stage_root="$study_root/discovery_calibration_${best_run_name//./_}"
embedding_file="$stage_root/exported_${representation}_${export_split}_embeddings.pt"
discovery_root="$stage_root/module_discovery"
calibration_root="$stage_root/uncertainty"
linked_root="$stage_root/linked_uncertainty_modules"
structure_root="$stage_root/targeted_structure_validation"

echo "[discovery-stage] root=$ROOT_DIR"
echo "[discovery-stage] study_root=$study_root"
echo "[discovery-stage] best_run_dir=$best_run_dir"
echo "[discovery-stage] device=$device"
echo "[discovery-stage] export_split=$export_split"
echo "[discovery-stage] started_at=$(date -Is)"

if [[ $run_analysis -eq 1 ]]; then
  run_logged "family_heterogeneity" \
    python -u ./scripts/analyze_family_heterogeneity.py \
      --protein-run "$study_root/protein_only.family_holdout" \
      --context-run "$study_root/genome_aware_denovo.family_holdout" \
      --output-dir "$study_root/family_heterogeneity" \
      --bootstrap-iterations "$bootstrap_iterations"

  run_logged "structural_axis_atlas" \
    python -u ./scripts/build_structural_axis_atlas.py \
      --study-root "$study_root" \
      --output-dir "$study_root/structural_axis_atlas"
fi

require_path "$best_run_dir/best_model.pt" "best model"
require_path "$best_run_dir/dataset_cache.pt" "dataset cache"
require_path "$best_run_dir/run_manifest.json" "run manifest"

if [[ $run_calibration -eq 1 ]]; then
  run_logged "calibrate_task_mode_uncertainty" \
    python -u ./scripts/calibrate_task_mode_uncertainty.py \
      --run-dir "$best_run_dir" \
      --output-dir "$calibration_root" \
      --device "$device" \
      --batch-size "$batch_size" \
      --num-workers "$num_workers" \
      --prefetch-factor "$prefetch_factor" \
      --fdr-target "$fdr_target"
fi

if [[ $run_discovery -eq 1 ]]; then
  run_logged "export_task_mode_embeddings" \
    python -u ./scripts/export_task_mode_embeddings.py \
      --run-dir "$best_run_dir" \
      --output "$embedding_file" \
      --representation "$representation" \
      --split "$export_split" \
      --device "$device" \
      --batch-size "$batch_size" \
      --num-workers "$num_workers" \
      --prefetch-factor "$prefetch_factor"

  run_logged "discover_module_candidates" \
    python -u ./scripts/discover_module_candidates.py \
      --embedding-file "$embedding_file" \
      --output-dir "$discovery_root" \
      --cluster-method "$cluster_method" \
      --min-cluster-size "$min_cluster_size" \
      --window-radius "$window_radius" \
      --top-casebooks "$top_casebooks"
fi

if [[ $run_calibration -eq 1 && $run_discovery -eq 1 ]]; then
  require_path "$discovery_root/module_candidates.tsv" "module candidates"
  require_path "$discovery_root/ranked_hypothetical_clusters.tsv" "ranked clusters"
  require_path "$calibration_root/candidate_prioritization.tsv" "candidate prioritization"
  run_logged "link_uncertainty_modules" \
    python -u ./scripts/link_uncertainty_modules.py \
      --module-candidates "$discovery_root/module_candidates.tsv" \
      --ranked-clusters "$discovery_root/ranked_hypothetical_clusters.tsv" \
      --candidate-prioritization "$calibration_root/candidate_prioritization.tsv" \
      --output-dir "$linked_root"
fi

if [[ $run_structure_prep -eq 1 ]]; then
  require_path "$discovery_root/module_candidates.tsv" "module candidates"
  require_path "$discovery_root/ranked_hypothetical_clusters.tsv" "ranked clusters"
  run_logged "prepare_targeted_structure_validation" \
    python -u ./scripts/prepare_targeted_structure_validation.py \
      --ranked-clusters "$discovery_root/ranked_hypothetical_clusters.tsv" \
      --module-candidates "$discovery_root/module_candidates.tsv" \
      --output-dir "$structure_root" \
      --top-clusters "$top_structure_clusters" \
      --representatives-per-cluster "$representatives_per_cluster"
fi

if [[ $run_package -eq 1 ]]; then
  mkdir -p artifacts/return
  archive_path="artifacts/return/discovery_calibration_${best_run_name//./_}_$(date +%Y%m%d_%H%M%S).tar.gz"
  run_logged "package_discovery_calibration_outputs" \
    tar -czf "$archive_path" \
      --exclude="*.pt" \
      "$stage_root" \
      "$study_root/family_heterogeneity" \
      "$study_root/structural_axis_atlas"
  echo "[discovery-stage] archive=$archive_path"
fi

echo "[discovery-stage] finished_at=$(date -Is)"
echo "[discovery-stage] stage_root=$stage_root"
