# ViruFunc Atlas Data Card

## Release Metadata

- Version: `v1.0.2-atlas-resource-revision`
- DOI: <https://doi.org/10.5281/zenodo.20925484>
- Repository: <https://github.com/shayuanxukuang/viral-context-function-v2>
- GitHub release tag: `v1.0.2-atlas-resource-revision`

## Dataset

ViruFunc Atlas v1.0 is a frozen viral protein function annotation benchmark.
The manuscript freeze contains 713,487 proteins, 19,149 genomes, 1,283 viral
families, 12 host supergroups, and 17 primary labels.

## Use

The Core archive supports evaluation under fixed split manifests, label
ontology files, feature-boundary rules, and baseline metrics. A rebuildable
large-artifact layer records the identifiers, checksums, FASTA targets, source
tables, and scripts needed to regenerate embeddings, model checkpoints, and
structure-panel artifacts.

## Main Splits

- default/hash split for optimistic-bias auditing only;
- family-heldout split as the primary out-of-distribution benchmark;
- strict-zero family-heldout test subset for exact-transfer sensitivity;
- host-heldout split as a secondary OOD benchmark;
- sequence-cluster 30/50/70 holdouts as sequence-relatedness controls.

## Limits

Benchmark labels are incomplete and are derived from normalized product text
and curated synonym rules. Post hoc sources such as MMseqs2, Foldseek, Phold,
PHROG, InterPro/Pfam/CDD, product text, and current NCBI concordance are
evidence for triage or annotation-refinement tracks. They are not de novo T1/T2
inputs.
