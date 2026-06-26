# ViruFunc Atlas Core Benchmark

Core evaluator for ViruFunc Atlas prediction files.

Core command:

```bash
conda env create -f environment.yml
conda activate virufunc-atlas-core
python benchmark/evaluate_predictions.py \
  --pred predictions/baselines/small_example_predictions.csv \
  --split examples/minimal_family_heldout_test.tsv \
  --labels examples/minimal_label_ontology.tsv \
  --track T2_genome_context_nohost \
  --used-features target_sequence,sequence_embedding,neighbor_sequences,gene_order,coordinates,strand,gap_overlap,genome_topology \
  --out results/my_model_familyheldout.json
```

For full benchmark runs, use the compressed split assignment manifests in
`data_manifest/`, the label ontology in `configs/labels/label_ontology.tsv`,
and the feature-boundary rules in `configs/labels/forbidden_features.yaml`.

The evaluator validates:

- coverage of all proteins and labels in the chosen split;
- unknown protein or label identifiers;
- duplicate protein-label rows;
- scores outside `[0, 1]`;
- feature declarations that conflict with the chosen track.

It reports macro AP, macro F1, micro AP, micro F1, per-label AP, and optional
family-block bootstrap confidence intervals.
