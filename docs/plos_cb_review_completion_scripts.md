# PLOS CB review-completion scripts

This note lists the scripts added to turn the current ViruFunc V2 manuscript package into reviewer-facing supplementary artifacts and rerunnable missing analyses.

## Local supplement package

Generate concrete supplementary tables and figures from returned artifacts:

```bash
python scripts/make_v2_supplementary_package.py \
  --core-dir artifacts/return/extracted_v2_20260430_100225 \
  --qc-dir artifacts/return/extracted_v2_qc_20260430_100225/qc_review \
  --assets-dir artifacts/return/v2_manuscript_assets_20260430_100225 \
  --output-dir artifacts/return/v2_plos_cb_supplementary_package_20260503 \
  --protein-index data/processed/training/viral_protein_training_index.tsv.gz \
  --split-manifest data/processed/splits/viral_protein_strict_splits.tsv.gz \
  --make-zip
```

Outputs:

- `supplementary_tables/`: concrete S1-S20 source tables where available.
- `supplementary_figures/`: generated S1-S5 and S7-S15 figures.
- `supplement_completion_gap_report.tsv`: analyses that still require server outputs.
- Optional zip archives when `--make-zip` is used for local transfer. The PLOS upload-ready package is organized instead as `S1_Text.pdf`, `S1_Table.xlsx`, `S1_Fig.pdf`, and `S2_File_reproducibility_manifest.json`.

## Candidate case evidence

Build post hoc evidence tables and local neighborhood windows for all 27 high-context-gain candidates:

```bash
python scripts/build_candidate_case_evidence.py \
  --candidates artifacts/return/v2_plos_cb_supplementary_package_20260503/supplementary_tables/S16_high_context_gain_candidates.tsv \
  --protein-index data/processed/training/viral_protein_training_index.tsv.gz \
  --split-manifest data/processed/splits/viral_protein_strict_splits.tsv.gz \
  --module-candidates artifacts/return/extracted_v2_20260430_100225/module_discovery/module_candidates.tsv \
  --output-dir artifacts/return/v2_plos_cb_supplementary_package_20260503/candidate_case_evidence
```

If homology, domain, or structure hits are available, add:

```bash
--homology-hits <homology_top_hit_assignments.tsv> \
--domain-hits <domain_hits.tsv> \
--structure-hits <foldseek_hits.tsv>
```

## Server one-key completion

Run the missing server-side analyses, including 3-seed training on GPUs 4, 5, and 6, source add-back CIs, MMseqs2 homology baseline, nucleocapsid synonym sensitivity, candidate evidence, and supplement package assembly:

```bash
ROOT=<PROJECT_ROOT>
RUN=$ROOT/runs/context_study_v2_20260430_100225
PYTHON_BIN=<PROMATH_TORCH_PYTHON>
GPU_IDS=4,5,6
MMSEQS_BIN=mmseqs
bash scripts/run_v2_review_completion.sh "$RUN"
```

Useful switches:

```bash
bash scripts/run_v2_review_completion.sh "$RUN" "$RUN/review_completion" --skip-multiseed
bash scripts/run_v2_review_completion.sh "$RUN" "$RUN/review_completion" --skip-homology
bash scripts/run_v2_review_completion.sh "$RUN" "$RUN/review_completion" --include-annotation-refinement
MMSEQS_BIN=/path/to/mmseqs bash scripts/run_v2_review_completion.sh "$RUN" "$RUN/review_completion" --skip-multiseed --skip-source-ci
```

If the run has already completed multi-seed training and source add-back CIs but stops with
`FileNotFoundError: mmseqs`, install MMseqs2 or point the runner at the executable, then resume:

```bash
conda install -c conda-forge -c bioconda mmseqs2

python scripts/run_v2_review_completion.py \
  --run-root "$RUN" \
  --output-root "$RUN/review_completion" \
  --skip-multiseed \
  --skip-source-ci \
  --mmseqs-bin "$(command -v mmseqs)" \
  --gpu-ids 4,5,6 \
  --threads 24
```

## Individual missing-analysis scripts

Source-decomposition CIs:

```bash
python scripts/bootstrap_source_addback_ci.py \
  --run-root "$RUN" \
  --input data/processed/training/viral_protein_training_index.tsv.gz \
  --split-manifest data/processed/splits/viral_protein_strict_splits.tsv.gz \
  --output-dir "$RUN/qc_review" \
  --bootstrap-iterations 1000 \
  --device cuda:0
```

MMseqs2 top-hit homology baseline:

```bash
python scripts/run_homology_label_transfer.py \
  --protein-index data/processed/training/viral_protein_training_index.tsv.gz \
  --split-manifest data/processed/splits/viral_protein_strict_splits.tsv.gz \
  --freeze-dir data/v2_freeze \
  --output-dir "$RUN/review_completion/homology_baseline" \
  --mmseqs-bin "$(command -v mmseqs)" \
  --threads 24
```

Nucleocapsid synonym-expanded sensitivity:

```bash
python scripts/nucleocapsid_synonym_sensitivity.py \
  --run-root "$RUN" \
  --input data/processed/training/viral_protein_training_index.tsv.gz \
  --split-manifest data/processed/splits/viral_protein_strict_splits.tsv.gz \
  --output-dir "$RUN/qc_review" \
  --device cuda:0
```

## Notes

- Multi-seed training is not done locally; it is scheduled by `run_v2_review_completion.py`.
- The current local package contains S1-S5 and S7-S15 supplementary figures. S6 is intentionally tied to the multi-seed server rerun.
- Classical homology baselines require `mmseqs` in `PATH` or `--mmseqs-bin`.
