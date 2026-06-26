#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

OUT_DIR="${OUT_DIR:-release/zenodo_bundles}"
STAGE_DIR="${STAGE_DIR:-release/.zenodo_stage}"

rm -rf "$STAGE_DIR"
mkdir -p "$OUT_DIR" "$STAGE_DIR"

copy_path() {
  local src="$1"
  local dst="$2"
  if [[ -e "$src" ]]; then
    mkdir -p "$(dirname "$dst")"
    cp -a "$src" "$dst"
  else
    echo "[warn] missing optional path: $src" >&2
  fi
}

make_tarball() {
  local name="$1"
  local subdir="$2"
  local tarball="$OUT_DIR/$name"
  if [[ ! -d "$STAGE_DIR/$subdir" ]]; then
    echo "[skip] no staged directory for $name" >&2
    return
  fi
  tar -czf "$tarball" -C "$STAGE_DIR" "$subdir"
  echo "[tar] $tarball"
}

core="ViruFunc_Atlas_v1.0_Core"
mkdir -p "$STAGE_DIR/$core"
for path in README.md LICENSE CITATION.cff environment.yml Dockerfile data_card.md benchmark_card.md CHANGELOG.md; do
  copy_path "$path" "$STAGE_DIR/$core/$path"
done
for path in benchmark configs data_manifest predictions examples supplementary_tables/figure_source_tables release/zenodo scripts/download_atlas_core.sh scripts/build_zenodo_release_bundles.sh; do
  copy_path "$path" "$STAGE_DIR/$core/$path"
done
make_tarball "ViruFunc_Atlas_v1.0_Core.tar.gz" "$core"

sources="ViruFunc_Atlas_v1.0_SourceTables"
mkdir -p "$STAGE_DIR/$sources"
copy_path "supplementary_tables" "$STAGE_DIR/$sources/supplementary_tables"
copy_path "artifacts/revision_phold_integrated_20260625/supplementary_tables" "$STAGE_DIR/$sources/revision_phold_integrated_supplementary_tables"
copy_path "artifacts/github_release/viral-context-function-v2_20260506/supplementary_tables" "$STAGE_DIR/$sources/github_release_supplementary_tables"
make_tarball "ViruFunc_Atlas_v1.0_SourceTables.tar.gz" "$sources"

baselines="ViruFunc_Atlas_v1.0_BaselinePredictions"
mkdir -p "$STAGE_DIR/$baselines"
copy_path "predictions/baselines" "$STAGE_DIR/$baselines/baselines"
copy_path "artifacts/github_release/viral-context-function-v2_20260506/predictions/baselines" "$STAGE_DIR/$baselines/github_release_baselines"
make_tarball "ViruFunc_Atlas_v1.0_BaselinePredictions.tar.gz" "$baselines"

phold="ViruFunc_Atlas_v1.0_Phold_PHROG_panel"
mkdir -p "$STAGE_DIR/$phold"
copy_path "supplementary_tables/figure_source_tables" "$STAGE_DIR/$phold/figure_source_tables"
copy_path "artifacts/revision_phold_integrated_20260625/supplementary_tables/S33_phold_manual_gold_160_targets.faa" "$STAGE_DIR/$phold/S33_phold_manual_gold_160_targets.faa"
copy_path "artifacts/revision_phold_integrated_20260625/supplementary_tables/S33_phold_panel_manifest.json" "$STAGE_DIR/$phold/S33_phold_panel_manifest.json"
copy_path "artifacts/revision_phold_integrated_20260625/supplementary_tables/S33_phold_panel_by_label.tsv" "$STAGE_DIR/$phold/S33_phold_panel_by_label.tsv"
copy_path "artifacts/revision_phold_integrated_20260625/supplementary_tables/S33_phold_panel_leave_one_label_out.tsv" "$STAGE_DIR/$phold/S33_phold_panel_leave_one_label_out.tsv"
copy_path "artifacts/revision_phold_integrated_20260625/supplementary_tables/S33_phold_panel_paired_by_label.tsv" "$STAGE_DIR/$phold/S33_phold_panel_paired_by_label.tsv"
make_tarball "ViruFunc_Atlas_v1.0_Phold_PHROG_panel.tar.gz" "$phold"

repro="ViruFunc_Atlas_v1.0_ReproducibilityManifest"
mkdir -p "$STAGE_DIR/$repro"
for path in data_manifest/checksum_manifest.tsv data_manifest/zenodo_expected_checksum_manifest.tsv data_manifest/freeze_report.json release/zenodo/zenodo_file_manifest.tsv release/atlas_core_manifest.tsv release/LICENSE_DECISION.md; do
  copy_path "$path" "$STAGE_DIR/$repro/$path"
done
copy_path "artifacts/revision_phold_integrated_20260625/revision_package_manifest.json" "$STAGE_DIR/$repro/revision_package_manifest.json"
copy_path "artifacts/revision_phold_integrated_20260625/03_Supporting_Information/S2_File_reproducibility_manifest.json" "$STAGE_DIR/$repro/S2_File_reproducibility_manifest.json"
make_tarball "ViruFunc_Atlas_v1.0_ReproducibilityManifest.tar.gz" "$repro"

cat > "$OUT_DIR/README.md" <<'EOF'
ViruFunc Atlas v1.0 Zenodo Bundle Staging
========================================

Tarballs staged by `scripts/build_zenodo_release_bundles.sh`.

The Sequences bundle comes from the frozen server export and contains FASTA,
accession, source database, retrieval, and checksum metadata. ESM embeddings,
checkpoints, Foldseek/Phold databases, and predicted structure archives are
handled as rebuildable large artifacts.

Checksums for generated tarballs are written to checksums.tsv when sha256sum is
available.
EOF

if command -v sha256sum >/dev/null 2>&1; then
  {
    printf "filename\tsha256\tbytes\n"
    for tarball in "$OUT_DIR"/*.tar.gz; do
      [[ -e "$tarball" ]] || continue
      filename="$(basename "$tarball")"
      hash="$(sha256sum "$tarball" | awk '{print $1}')"
      bytes="$(wc -c < "$tarball" | tr -d ' ')"
      printf "%s\t%s\t%s\n" "$filename" "$hash" "$bytes"
    done
  } > "$OUT_DIR/checksums.tsv"
  echo "[checksum] $OUT_DIR/checksums.tsv"
else
  echo "[warn] sha256sum not available; checksums.tsv not generated" >&2
fi

rm -rf "$STAGE_DIR"
echo "[done] bundles staged in $OUT_DIR"
