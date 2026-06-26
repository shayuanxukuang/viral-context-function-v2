# ViruFunc Atlas v1.0

**ViruFunc Atlas** is a leakage-aware benchmark for viral protein function
annotation. It separates sequence-only prediction, genome-context prediction,
audited metadata use, and open-evidence annotation refinement, with fixed split
manifests and feature-boundary rules.

Release metadata:

- Version: `v1.0.2-atlas-resource-revision`
- Title: `ViruFunc Atlas v1.0: a leakage-aware benchmark and reusable evaluation resource for viral protein function annotation`
- DOI: <https://doi.org/10.5281/zenodo.20925484>
- Repository: <https://github.com/shayuanxukuang/viral-context-function-v2>
- GitHub release tag: `v1.0.2-atlas-resource-revision`

## Contents

- Frozen manifests for 713,487 viral proteins, 19,149 genomes, 1,283 viral
  families, 12 host supergroups, and 17 primary labels.
- Family-heldout, host-heldout, default/hash, calibration, and strict-zero
  exact-transfer split resources.
- Label ontology, functional-group mapping, and feature-boundary rules.
- Baseline metric tables for sequence-only, genome-context, homology, and
  external-evidence analyses.
- Evaluator and validator for long-format prediction submissions.
- Mapping utilities for PHROG-, Phold-, Foldseek-, InterPro/Pfam/CDD-, and
  related evidence tables.
- Source tables for label-level robustness, current-NCBI evidence triage,
  sequence-structure-context triage, and the matched 160-protein Phold panel.

## GitHub vs Zenodo

GitHub contains code, documentation, label rules, split/checksum manifests,
baseline summaries, small examples, and Phold panel source tables.

The archived data release is on Zenodo:

```text
https://doi.org/10.5281/zenodo.20925484
```

Zenodo contains sequence FASTA/metadata, full source tables, baseline
prediction exports, Phold panel outputs, and reproducibility manifests. ESM
embeddings, model checkpoints, Foldseek/Phold databases, and predicted
structure archives are treated as rebuildable large artifacts.

## Benchmark Tracks

- `T1 sequence-only de novo`: target sequence or sequence-derived embeddings.
- `T2 genome-context no-host de novo`: T1 plus neighbor sequences and non-text
  genome-organization features such as gene order, coordinates, strand,
  gaps/overlaps, segment identifiers, and genome topology.
- `T3 audited metadata/context`: T2 plus declared host or taxonomy metadata,
  interpreted with host-heldout, host-shuffle, and host-corruption controls.
- `T4 annotation-refinement/open evidence`: declared external evidence such as
  MMseqs2, PHROG/Phold-style annotations, Foldseek, InterPro/Pfam/CDD, or
  product text. T4 is practical annotation refinement and is not ranked against
  T1/T2 de novo models.

## Repository Layout

- `benchmark/`: evaluator, validator, family-block bootstrap helper, and
  leaderboard utilities.
- `configs/labels/`: label ontology, functional-group table, mapping rules, and
  forbidden-feature rules.
- `configs/`: external-evidence label mapping and dataset/source configuration.
- `data_manifest/`: frozen manifests, feature/label manifests, checksums, and
  compressed split files where included.
- `predictions/baselines/`: compact baseline metrics and a small prediction
  example.
- `examples/`: smoke-test prediction, minimal split, minimal ontology, and
  submission template.
- `supplementary_tables/figure_source_tables/`: small Phold-panel source tables.
- `release/zenodo/`: Zenodo tarball manifest.
- `scripts/`: analysis, QC, homology baseline, source-decomposition, candidate
  triage, external annotation scoring, and manuscript-asset scripts.
- `supplementary_tables/`: manuscript and benchmark source tables.
- `docs/`, `data_card.md`, and `benchmark_card.md`: resource documentation.

## Quick Start

Run the evaluator smoke test:

```bash
conda env create -f environment.yml
conda activate virufunc-atlas-core
bash examples/run_smoke_test.sh
```

Download the archived bundles:

```bash
bash scripts/download_atlas_core.sh
```

Stage Zenodo tarballs locally:

```bash
bash scripts/build_zenodo_release_bundles.sh
```

Evaluate a long-format prediction file:

```bash
python benchmark/evaluate_predictions.py \
  --pred predictions/baselines/small_example_predictions.csv \
  --split examples/minimal_family_heldout_test.tsv \
  --labels examples/minimal_label_ontology.tsv \
  --track T2_genome_context_nohost \
  --used-features target_sequence,sequence_embedding,neighbor_sequences,gene_order,coordinates,strand,gap_overlap,genome_topology \
  --out runs/smoke_test/minimal_evaluation.json
```

For full benchmark submissions, use the split assignment manifests in
`data_manifest/`, the full label ontology in `configs/labels/label_ontology.tsv`,
and the track rules in `configs/labels/forbidden_features.yaml`.

## Excluded Large Files

The GitHub repository does not include:

- raw RefSeq/GenBank/NCBI/UniProt downloads;
- processed protein indexes and full feature matrices;
- frozen pLM embeddings;
- trained model checkpoints;
- ESMFold predicted PDB archives;
- Foldseek, Phold, PDB100, or other large external databases;
- local `runs/` directories.

The release records identifiers, manifests, source tables, and commands needed
to rebuild these files when the external databases and compute environment are
available.

## Sequence Provenance

No new sequence data were generated. Source viral protein and genome records
were derived from public NCBI/RefSeq/GenBank-linked resources. Original
accession identifiers, source databases, freeze date, retrieval scripts, and
checksums are provided in the release manifests and Zenodo bundles.

## Scope

ViruFunc Atlas evaluates genome context as a label- and evidence-regime-specific
signal. Candidate outputs are prioritized hypotheses. Current NCBI concordance,
sequence/structure evidence, module coherence, and the matched Phold panel are
T4 annotation-refinement evidence, not experimental validation or de novo T1/T2
inputs.

## Citation

If you use this resource, cite the manuscript and archived release:

```text
ViruFunc Atlas v1.0: a leakage-aware benchmark and reusable evaluation resource
for viral protein function annotation. Zenodo.
https://doi.org/10.5281/zenodo.20925484
```
