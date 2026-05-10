#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-python}"

OUTPUT_ROOT="runs/v2_nohost_primary_multiseed"
INPUT="data/processed/training/viral_protein_training_index.tsv.gz"
SPLIT_MANIFEST="data/processed/splits/viral_protein_strict_splits.tsv.gz"
SPLITS="family_holdout,host_holdout"
SEEDS="42,43,44"
PLM_EMBEDDING_PATH=""
DEVICE="auto"
CUDA_VISIBLE_DEVICES_VALUE=""
EPOCHS="12"
BATCH_SIZE="512"
EVAL_BATCH_SIZE="1024"
NUM_WORKERS="8"
PREFETCH_FACTOR="4"
NEIGHBOR_RADIUS="2"
MAX_LENGTH="2048"
MIN_LABEL_COUNT="500"
LEARNING_RATE="3e-4"
WEIGHT_DECAY="1e-2"
DROPOUT="0.2"
EMBED_DIM="128"
HIDDEN_DIM="256"
MAX_POS_WEIGHT="50.0"
GRADIENT_CLIP="1.0"
WARMUP_FRACTION="0.05"
WITH_BIOPHYSICS=0
RUN_HOST_SECONDARY=1
RUN_ALL_CLEAN_SECONDARY=1
RERUN_COMPLETED=0
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_v2_nohost_primary_multiseed.sh [options]

This trains the reviewer-facing primary de novo comparison:
  protein-only pLM vs local+genome no-host context.

Optional secondary runs:
  host-only metadata branch and all-clean-context branch.

Key options:
  --output-root PATH
  --python PATH
  --input PATH
  --split-manifest PATH
  --splits family_holdout,host_holdout,sequence_cluster_30_holdout
  --seeds 42,43,44
  --plm-embedding-path PATH
  --device auto|cuda|cpu
  --cuda-visible-devices IDS
  --epochs INT
  --batch-size INT
  --eval-batch-size INT
  --with-biophysics
  --no-host-secondary
  --no-all-clean-secondary
  --rerun-completed
  --dry-run
EOF
}

die() {
  echo "[error] $*" >&2
  exit 2
}

to_abs() {
  local p="$1"
  if [[ -z "$p" ]]; then
    return 0
  fi
  if [[ "$p" = /* ]]; then
    printf '%s\n' "$p"
  else
    printf '%s\n' "$ROOT/$p"
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --help|-h) usage; exit 0 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --input) INPUT="$2"; shift 2 ;;
    --split-manifest) SPLIT_MANIFEST="$2"; shift 2 ;;
    --splits) SPLITS="$2"; shift 2 ;;
    --seeds) SEEDS="$2"; shift 2 ;;
    --plm-embedding-path) PLM_EMBEDDING_PATH="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --cuda-visible-devices) CUDA_VISIBLE_DEVICES_VALUE="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --eval-batch-size) EVAL_BATCH_SIZE="$2"; shift 2 ;;
    --num-workers) NUM_WORKERS="$2"; shift 2 ;;
    --prefetch-factor) PREFETCH_FACTOR="$2"; shift 2 ;;
    --neighbor-radius) NEIGHBOR_RADIUS="$2"; shift 2 ;;
    --max-length) MAX_LENGTH="$2"; shift 2 ;;
    --min-label-count) MIN_LABEL_COUNT="$2"; shift 2 ;;
    --learning-rate) LEARNING_RATE="$2"; shift 2 ;;
    --weight-decay) WEIGHT_DECAY="$2"; shift 2 ;;
    --dropout) DROPOUT="$2"; shift 2 ;;
    --embed-dim) EMBED_DIM="$2"; shift 2 ;;
    --hidden-dim) HIDDEN_DIM="$2"; shift 2 ;;
    --max-pos-weight) MAX_POS_WEIGHT="$2"; shift 2 ;;
    --gradient-clip) GRADIENT_CLIP="$2"; shift 2 ;;
    --warmup-fraction) WARMUP_FRACTION="$2"; shift 2 ;;
    --with-biophysics) WITH_BIOPHYSICS=1; shift ;;
    --no-host-secondary) RUN_HOST_SECONDARY=0; shift ;;
    --no-all-clean-secondary) RUN_ALL_CLEAN_SECONDARY=0; shift ;;
    --rerun-completed) RERUN_COMPLETED=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) die "Unknown option: $1" ;;
  esac
done

OUTPUT_ROOT="$(to_abs "$OUTPUT_ROOT")"
INPUT="$(to_abs "$INPUT")"
SPLIT_MANIFEST="$(to_abs "$SPLIT_MANIFEST")"
PLM_EMBEDDING_PATH="$(to_abs "$PLM_EMBEDDING_PATH")"

[[ -f "$INPUT" ]] || die "Input protein index not found: $INPUT"
[[ -f "$SPLIT_MANIFEST" ]] || die "Split manifest not found: $SPLIT_MANIFEST"
if [[ -n "$PLM_EMBEDDING_PATH" && ! -f "$PLM_EMBEDDING_PATH" ]]; then
  die "PLM embedding file not found: $PLM_EMBEDDING_PATH"
fi

mkdir -p "$OUTPUT_ROOT"
COMMAND_LOG="$OUTPUT_ROOT/nohost_primary_commands.tsv"
printf "seed\tsplit\trun_name\tcommand\n" > "$COMMAND_LOG"

run_state() {
  local run_dir="$1"
  if [[ -f "$run_dir/metrics_summary.json" && -f "$run_dir/run_manifest.json" && -f "$run_dir/best_model.pt" ]]; then
    echo "complete"
  elif [[ -d "$run_dir" ]]; then
    echo "partial"
  else
    echo "missing"
  fi
}

run_training() {
  local seed="$1"
  local split="$2"
  local run_name="$3"
  shift 3
  local extra_args=("$@")
  local seed_root="$OUTPUT_ROOT/seed_${seed}"
  local run_dir="$seed_root/${run_name}.${split}"
  mkdir -p "$seed_root"

  local state
  state="$(run_state "$run_dir")"
  if [[ "$state" == "complete" && "$RERUN_COMPLETED" -eq 0 ]]; then
    echo "[skip] $run_dir already complete"
    return
  fi

  local cmd=("$PYTHON_BIN")
  if [[ -n "$PLM_EMBEDDING_PATH" ]]; then
    cmd+=("$ROOT/scripts/train_task_modes_plm.py" "--plm-embedding-path" "$PLM_EMBEDDING_PATH" "--")
  else
    cmd+=("$ROOT/scripts/train_task_modes.py")
  fi
  cmd+=(
    --input "$INPUT"
    --split-manifest "$SPLIT_MANIFEST"
    --split-scheme "$split"
    --output-dir "$run_dir"
    --device "$DEVICE"
    --epochs "$EPOCHS"
    --batch-size "$BATCH_SIZE"
    --eval-batch-size "$EVAL_BATCH_SIZE"
    --embed-dim "$EMBED_DIM"
    --hidden-dim "$HIDDEN_DIM"
    --dropout "$DROPOUT"
    --learning-rate "$LEARNING_RATE"
    --weight-decay "$WEIGHT_DECAY"
    --num-workers "$NUM_WORKERS"
    --prefetch-factor "$PREFETCH_FACTOR"
    --neighbor-radius "$NEIGHBOR_RADIUS"
    --max-length "$MAX_LENGTH"
    --min-label-count "$MIN_LABEL_COUNT"
    --seed "$seed"
    --gradient-clip "$GRADIENT_CLIP"
    --warmup-fraction "$WARMUP_FRACTION"
    --max-pos-weight "$MAX_POS_WEIGHT"
    --save-test-predictions
    "${extra_args[@]}"
  )
  if [[ "$WITH_BIOPHYSICS" -eq 1 ]]; then
    cmd+=(--with-biophysics)
  fi

  printf "%s\t%s\t%s\t%s\n" "$seed" "$split" "$run_name" "${cmd[*]}" >> "$COMMAND_LOG"
  echo "[cmd] ${cmd[*]}"
  if [[ "$DRY_RUN" -eq 0 ]]; then
    if [[ -n "$CUDA_VISIBLE_DEVICES_VALUE" ]]; then
      CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_VALUE" "${cmd[@]}"
    else
      "${cmd[@]}"
    fi
  fi
}

IFS=',' read -r -a split_values <<< "$SPLITS"
IFS=',' read -r -a seed_values <<< "$SEEDS"

for seed in "${seed_values[@]}"; do
  seed="$(echo "$seed" | xargs)"
  [[ -n "$seed" ]] || continue
  for split in "${split_values[@]}"; do
    split="$(echo "$split" | xargs)"
    [[ -n "$split" ]] || continue
    run_training "$seed" "$split" "protein_only" --task-mode protein_only
    run_training "$seed" "$split" "genome_aware_nohost_local_genome" \
      --task-mode genome_aware_denovo \
      --context-blocks local_neighborhood,genome_organization
    if [[ "$RUN_HOST_SECONDARY" -eq 1 ]]; then
      run_training "$seed" "$split" "genome_aware_host_only_secondary" \
        --task-mode genome_aware_denovo \
        --context-blocks host_metadata
    fi
    if [[ "$RUN_ALL_CLEAN_SECONDARY" -eq 1 ]]; then
      run_training "$seed" "$split" "genome_aware_all_clean_secondary" \
        --task-mode genome_aware_denovo
    fi
  done
done

cat > "$OUTPUT_ROOT/nohost_primary_manifest.json" <<JSON
{
  "output_root": "$OUTPUT_ROOT",
  "splits": "$SPLITS",
  "seeds": "$SEEDS",
  "primary_comparison": "protein_only vs genome_aware_nohost_local_genome",
  "secondary_branches": {
    "host_only": $RUN_HOST_SECONDARY,
    "all_clean_context": $RUN_ALL_CLEAN_SECONDARY
  },
  "with_biophysics": $WITH_BIOPHYSICS,
  "commands": "$COMMAND_LOG"
}
JSON

echo "[done] commands logged to $COMMAND_LOG"
