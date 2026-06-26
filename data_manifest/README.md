# Data Manifest

Frozen ViruFunc Atlas v1.0 benchmark manifests.

Included here:

- `split_manifest.tsv`: split-family inventory and provenance.
- `family_holdout_split.tsv.gz`: primary family-heldout split assignment
  manifest.
- `host_holdout_split.tsv.gz`: host-heldout split assignment manifest.
- `calibration_split.tsv.gz` and `default_split.tsv.gz`: supporting split
  assignment manifests.
- `checksum_manifest.tsv`: checksums for GitHub release files.
- `zenodo_expected_checksum_manifest.tsv`: expected checksums for the larger
  Zenodo manifest files from the frozen data export.
- `label_manifest.tsv`: 17-label ontology, functional groups, synonym patterns,
  and split-level positive counts.
- `feature_manifest.tsv`: feature provenance and input-boundary annotations.
- `forbidden_feature_check.tsv`: small audit table for feature-boundary checks.
- `freeze_report.json`: frozen-data summary.

The split files are compressed TSV assignment manifests, not separate
train/validation/test tables. Filter by the `split` column.

Large source sequences, processed feature matrices, protein language-model
embeddings, trained checkpoints, external databases, and predicted structure
archives are in the Zenodo release or are rebuildable from external sources.
DOI: `10.5281/zenodo.20925484`.
