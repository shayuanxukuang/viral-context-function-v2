# Zenodo Release Bundles

Zenodo DOI `10.5281/zenodo.20925484` is the data archive. GitHub carries code,
evaluator files, rules, and documentation.

Tarballs are listed in `zenodo_file_manifest.tsv`.

```bash
bash scripts/build_zenodo_release_bundles.sh
```

`ViruFunc_Atlas_v1.0_Sequences.tar.gz` is exported from the frozen server data
and includes FASTA, accession, source database, retrieval, and checksum
metadata.

Keep the following outside the Zenodo manuscript bundle unless the final release
policy requires them:

- frozen ESM embeddings;
- model checkpoints;
- Foldseek/Phold databases;
- predicted PDB archives;
- large intermediate caches.

Sequence provenance statement:

```text
No new sequence data were generated. Source viral protein and genome records
were derived from public NCBI/RefSeq/GenBank-linked resources. Original
accession identifiers, source databases, freeze date, retrieval scripts, and
checksums are provided.
```
