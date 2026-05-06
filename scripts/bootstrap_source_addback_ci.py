#!/usr/bin/env python3
"""Block-bootstrap uncertainty for source add-back and leave-one-source comparisons."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch

from run_v2_qc_suite import (
    align_predictions,
    load_run_predictions,
    load_strict_split_rows,
    load_training_metadata,
    paired_bootstrap_delta,
)
from train_overnight_baseline import choose_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--input", type=Path, default=Path("data/processed/training/viral_protein_training_index.tsv.gz"))
    parser.add_argument("--split-manifest", type=Path, default=Path("data/processed/splits/viral_protein_strict_splits.tsv.gz"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force-predict", action="store_true")
    return parser.parse_args()


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def available(run_root: Path, run_name: str) -> bool:
    run_dir = run_root / run_name
    return (run_dir / "run_manifest.json").exists() and (run_dir / "best_model.pt").exists()


def main() -> None:
    args = parse_args()
    run_root = args.run_root.resolve()
    output_dir = (args.output_dir or run_root / "qc_review").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)

    strict_rows = load_strict_split_rows(args.split_manifest)
    metadata = load_training_metadata(args.input, strict_rows)

    pair_specs = [
        ("family_holdout", "protein_only.family_holdout", "genome_aware_denovo_addback_local_only.family_holdout", "local_only", "virus_family"),
        ("family_holdout", "protein_only.family_holdout", "genome_aware_denovo_addback_genome_only.family_holdout", "genome_only", "virus_family"),
        ("family_holdout", "protein_only.family_holdout", "genome_aware_denovo_addback_host_only.family_holdout", "host_only", "virus_family"),
        ("family_holdout", "protein_only.family_holdout", "genome_aware_denovo_addback_local_genome.family_holdout", "local_plus_genome", "virus_family"),
        ("family_holdout", "protein_only.family_holdout", "genome_aware_denovo.family_holdout", "all_clean_context", "virus_family"),
        ("host_holdout", "protein_only.host_holdout", "genome_aware_denovo_addback_local_only.host_holdout", "local_only", "host_taxid_key"),
        ("host_holdout", "protein_only.host_holdout", "genome_aware_denovo_addback_genome_only.host_holdout", "genome_only", "host_taxid_key"),
        ("host_holdout", "protein_only.host_holdout", "genome_aware_denovo_addback_host_only.host_holdout", "host_only", "host_taxid_key"),
        ("host_holdout", "protein_only.host_holdout", "genome_aware_denovo_addback_local_genome.host_holdout", "local_plus_genome", "host_taxid_key"),
        ("host_holdout", "protein_only.host_holdout", "genome_aware_denovo.host_holdout", "all_clean_context", "host_taxid_key"),
    ]

    prediction_cache: dict[str, dict[str, Any]] = {}

    def pred(run_name: str) -> dict[str, Any]:
        if run_name not in prediction_cache:
            prediction_cache[run_name] = load_run_predictions(
                run_root / run_name,
                output_dir,
                device,
                args.batch_size,
                args.num_workers,
                args.prefetch_factor,
                args.force_predict,
            )
        return prediction_cache[run_name]

    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for idx, (split, ref, variant, source, block_field) in enumerate(pair_specs):
        if not available(run_root, ref) or not available(run_root, variant):
            skipped.append({"split": split, "reference_run": ref, "variant_run": variant, "source": source, "reason": "missing run checkpoint or manifest"})
            continue
        aligned = align_predictions(pred(ref), pred(variant))
        block_values = [str(metadata.get(accession, {}).get(block_field, "")) for accession in aligned["accessions"]]
        boot = paired_bootstrap_delta(aligned, block_values, args.bootstrap_iterations, args.seed + idx)
        rows.append(
            {
                "split": split,
                "reference_run": ref,
                "variant_run": variant,
                "source": source,
                "block_unit": block_field,
                **boot,
            }
        )

    write_tsv(output_dir / "qc_source_addback_block_bootstrap_ci.tsv", rows)
    write_tsv(output_dir / "qc_source_addback_block_bootstrap_ci_skipped.tsv", skipped)
    summary = {
        "output_tsv": str(output_dir / "qc_source_addback_block_bootstrap_ci.tsv"),
        "completed_comparisons": len(rows),
        "skipped_comparisons": len(skipped),
        "bootstrap_iterations": args.bootstrap_iterations,
        "device": str(device),
    }
    (output_dir / "qc_source_addback_block_bootstrap_ci_report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
