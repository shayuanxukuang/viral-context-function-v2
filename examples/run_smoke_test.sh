#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

python_has_core_deps() {
  "$1" - <<'PY' >/dev/null 2>&1
import numpy  # noqa: F401
import pandas  # noqa: F401
import sklearn  # noqa: F401
import yaml  # noqa: F401
PY
}

PYTHON_CMD=""
if [[ -n "${PYTHON_BIN:-}" ]]; then
  if ! python_has_core_deps "$PYTHON_BIN"; then
    echo "smoke test failed: PYTHON_BIN lacks numpy, pandas, scikit-learn, or pyyaml: $PYTHON_BIN" >&2
    exit 127
  fi
  PYTHON_CMD="$PYTHON_BIN"
elif command -v python3 >/dev/null 2>&1 && python_has_core_deps python3; then
  PYTHON_CMD="python3"
elif command -v python >/dev/null 2>&1 && python_has_core_deps python; then
  PYTHON_CMD="python"
elif command -v py.exe >/dev/null 2>&1 && python_has_core_deps py.exe; then
  PYTHON_CMD="py.exe"
else
  echo "smoke test failed: no Python candidate has numpy, pandas, scikit-learn, and pyyaml" >&2
  echo "Set PYTHON_BIN to the ViruFunc Atlas Core environment python." >&2
  exit 127
fi

OUT_DIR="runs/smoke_test"
OUT_JSON="$OUT_DIR/minimal_evaluation.json"
mkdir -p "$OUT_DIR"

"$PYTHON_CMD" benchmark/evaluate_predictions.py \
  --pred examples/minimal_prediction_example.csv \
  --split examples/minimal_family_heldout_test.tsv \
  --labels examples/minimal_label_ontology.tsv \
  --track T2_genome_context_nohost \
  --used-features target_sequence,sequence_embedding,neighbor_sequences,gene_order,coordinates,strand,gap_overlap,genome_topology \
  --bootstrap-iterations 20 \
  --out "$OUT_JSON"

"$PYTHON_CMD" - "$OUT_JSON" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
required = {"macro_AP", "micro_AP", "macro_F1", "micro_F1"}
missing = sorted(required.difference(payload))
if missing:
    raise SystemExit(f"smoke test failed: missing metrics {missing}")
print(f"smoke test passed: {path}")
PY
