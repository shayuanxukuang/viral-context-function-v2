#!/usr/bin/env bash
set -euo pipefail

python benchmark/evaluate_predictions.py \
  --pred examples/minimal_prediction_example.csv \
  --split examples/minimal_family_heldout_test.tsv \
  --labels examples/minimal_label_ontology.tsv \
  --track T2_genome_context_nohost \
  --used-features target_sequence,sequence_embedding,neighbor_sequences,gene_order,coordinates,strand,gap_overlap,genome_topology \
  --bootstrap-iterations 100 \
  --out runs/example_minimal_evaluation.json
