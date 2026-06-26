# ViruFunc Atlas Benchmark Card

## Release Metadata

- Version: `v1.0.2-atlas-resource-revision`
- DOI: <https://doi.org/10.5281/zenodo.20925484>
- Repository: <https://github.com/shayuanxukuang/viral-context-function-v2>
- GitHub release tag: `v1.0.2-atlas-resource-revision`

## Task

Predict one or more functional labels for each viral protein in a fixed split.
Submissions provide long-format scores:

```text
protein_id,label_id,score
```

## Tracks

- T1 sequence-only de novo: target sequence or sequence-derived embedding.
- T2 genome-context no-host de novo: T1 plus neighbor sequences, gene order,
  coordinates, strand, gaps/overlaps, and genome topology.
- T3 audited metadata/context: T2 plus declared host or taxonomy metadata.
- T4 annotation-refinement/open evidence: external databases, product text, and
  external annotation evidence are allowed but are not ranked with T1/T2.

## Primary Metric

The default primary metric for T1/T2 is strict-zero family-heldout macro AP,
with family-block bootstrap confidence intervals when family identifiers are
available.

## File Checks

The evaluator rejects unknown protein IDs, unknown label IDs, duplicate rows,
scores outside `[0, 1]`, missing scores unless explicitly allowed, and
feature declarations that conflict with the chosen track.

## Scope

Genome context is evaluated as a label- and evidence-regime-dependent signal,
not as a universally superior predictor. Post hoc evidence sources, including
current NCBI concordance and the matched Phold panel, support triage and
annotation-refinement claims only.
