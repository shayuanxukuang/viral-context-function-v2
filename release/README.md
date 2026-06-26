# ViruFunc Atlas Release Staging

Release layout for ViruFunc Atlas v1.0.

## Layers

- Atlas-Core: public evaluation files; no GPU required.
- Atlas-Full: optional large artifacts for regenerating pLM embeddings,
  checkpoints, ESMFold/Foldseek panels, and structure outputs.
- Zenodo bundles: tarballs listed in `release/zenodo/zenodo_file_manifest.tsv`.

## GitHub Entry Point

GitHub contains:

- code, evaluator, validator, Dockerfile, and `environment.yml`;
- `configs/labels/` ontology, groups, mapping rules, and forbidden features;
- `data_manifest/` split, checksum, label, feature, and freeze manifests;
- `predictions/baselines/` compact metrics and smoke-test predictions;
- `examples/` smoke-test files;
- small figure source tables, including the 160-protein Phold panel.

## Zenodo Entry Point

Zenodo DOI `10.5281/zenodo.20925484` contains the data bundles. File names are
listed in `release/zenodo/zenodo_file_manifest.tsv`. The Core, SourceTables,
BaselinePredictions, Phold panel, and ReproducibilityManifest tarballs are built
with `scripts/build_zenodo_release_bundles.sh`. The Sequences tarball is built
from the frozen server export.

## External Baseline Note

Full-test PHROG or Phold baseline scoring can be run with:

```bash
python scripts/score_external_annotation_baseline.py \
  --annotation-table /path/to/phrog_or_phold_annotations.tsv \
  --prediction-cache runs/review_additional_20260509_extracted/runs/v2_main_baseline_table/_prediction_cache/protein_only.family_holdout.npz \
  --baseline-name PHROG_mapped_family_holdout \
  --mapping-rules configs/external_baseline_label_mapping.tsv \
  --output-dir runs/virufunc_atlas_v3_server/external_baselines/PHROG_mapped_family_holdout
```

Outputs include `.baseline_metrics.tsv`, `.predictions.csv`, `.mapped_hits.tsv`,
`.unmapped_annotation_text.tsv`, and `.report.json`.
