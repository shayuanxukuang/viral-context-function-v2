#!/usr/bin/env bash
set -euo pipefail

ZENODO_RECORD_URL="${ZENODO_RECORD_URL:-https://zenodo.org/records/20925484/files}"
OUT_DIR="${OUT_DIR:-downloads/virufunc_atlas_v1.0}"

FILES=(
  "ViruFunc_Atlas_v1.0_Core.tar.gz"
  "ViruFunc_Atlas_v1.0_SourceTables.tar.gz"
  "ViruFunc_Atlas_v1.0_ReproducibilityManifest.tar.gz"
)

mkdir -p "$OUT_DIR"

if command -v curl >/dev/null 2>&1; then
  DOWNLOADER=(curl -L --fail --retry 5 --retry-delay 5 -o)
elif command -v wget >/dev/null 2>&1; then
  DOWNLOADER=(wget -O)
else
  echo "download failed: install curl or wget" >&2
  exit 127
fi

for file in "${FILES[@]}"; do
  url="${ZENODO_RECORD_URL}/${file}?download=1"
  dest="${OUT_DIR}/${file}"
  if [[ -s "$dest" ]]; then
    echo "[skip] $dest already exists"
    continue
  fi
  echo "[download] $url"
  "${DOWNLOADER[@]}" "$dest" "$url"
done

echo "[done] downloaded ViruFunc Atlas core bundles to $OUT_DIR"
echo "Next: bash examples/run_smoke_test.sh"
