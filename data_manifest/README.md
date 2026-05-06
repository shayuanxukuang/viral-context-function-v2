# Data Manifest Notes

This directory contains lightweight manifests and checksums for the frozen ViruFunc V2 benchmark.

The large split files are compressed into:

```text
split_files.tar.gz
```

It contains:

- `family_holdout_split.tsv`
- `host_holdout_split.tsv`
- `calibration_split.tsv`
- `default_split.tsv`

To unpack:

```bash
tar xzf data_manifest/split_files.tar.gz -C data_manifest
```

Raw public database downloads, processed feature matrices, protein language model embeddings, model checkpoints, Foldseek databases, and predicted PDB archives are not tracked in this Git repository.

