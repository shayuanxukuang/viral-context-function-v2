#!/usr/bin/env python3
"""Summarize multi-seed paired block-bootstrap deltas for V2 model comparisons."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from run_v2_qc_suite import (
    align_predictions,
    delta_row_from_arrays,
    load_run_predictions,
    load_strict_split_rows,
    load_training_metadata,
)
from train_overnight_baseline import choose_device


DEFAULT_COMPARISONS = (
    "family_nohost_primary:protein_only.family_holdout:genome_aware_nohost_local_genome.family_holdout:virus_family",
    "host_nohost_primary:protein_only.host_holdout:genome_aware_nohost_local_genome.host_holdout:host_taxid_key",
    "family_host_only_secondary:protein_only.family_holdout:genome_aware_host_only_secondary.family_holdout:virus_family",
    "family_all_clean_secondary:protein_only.family_holdout:genome_aware_all_clean_secondary.family_holdout:virus_family",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True, help="Directory containing seed_*/ run subdirectories.")
    parser.add_argument("--input", type=Path, default=Path("data/processed/training/viral_protein_training_index.tsv.gz"))
    parser.add_argument("--split-manifest", type=Path, default=Path("data/processed/splits/viral_protein_strict_splits.tsv.gz"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed-glob", default="seed_*")
    parser.add_argument("--comparison", action="append", default=[], help="name:left_run:right_run:block_field")
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
    if not keys:
        keys = ["status"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_comparison(text: str) -> tuple[str, str, str, str]:
    parts = text.split(":")
    if len(parts) != 4:
        raise SystemExit(f"Comparison must be name:left_run:right_run:block_field, got: {text}")
    return tuple(parts)  # type: ignore[return-value]


def available(seed_root: Path, run_name: str) -> bool:
    run_dir = seed_root / run_name
    return (run_dir / "run_manifest.json").exists() and (run_dir / "best_model.pt").exists()


def block_indices(block_values: list[str]) -> dict[str, list[int]]:
    out: dict[str, list[int]] = defaultdict(list)
    for idx, block in enumerate(block_values):
        out[str(block) or "__MISSING__"].append(idx)
    return dict(out)


def bootstrap_seed_averaged(
    seed_payloads: list[dict[str, Any]],
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    metrics = ("delta_macro_ap", "delta_macro_f1", "delta_micro_ap", "delta_micro_f1")
    seed_points = [payload["point"] for payload in seed_payloads]
    out: dict[str, Any] = {}
    for metric in metrics:
        values = np.asarray([row[metric] for row in seed_points], dtype=np.float64)
        out[metric] = float(np.mean(values)) if values.size else None
        out[f"{metric}_seed_sd"] = float(np.std(values, ddof=1)) if values.size > 1 else None

    rng = np.random.default_rng(seed)
    sampled_rows: list[dict[str, float]] = []
    for _ in range(iterations):
        per_seed_rows = []
        for payload in seed_payloads:
            aligned = payload["aligned"]
            block_map = payload["block_map"]
            blocks = sorted(block_map)
            if not blocks:
                continue
            sampled_blocks = rng.choice(blocks, size=len(blocks), replace=True)
            sampled_indices = np.asarray(
                [idx for block in sampled_blocks for idx in block_map[str(block)]],
                dtype=np.int64,
            )
            if sampled_indices.size == 0:
                continue
            per_seed_rows.append(
                delta_row_from_arrays(
                    aligned["y_true"][sampled_indices],
                    aligned["left_prob"][sampled_indices],
                    aligned["right_prob"][sampled_indices],
                    aligned["left_thresholds"],
                    aligned["right_thresholds"],
                    aligned["label_names"],
                )
            )
        if per_seed_rows:
            sampled_rows.append({metric: float(np.mean([row[metric] for row in per_seed_rows])) for metric in metrics})

    for metric in metrics:
        values = np.asarray([row[metric] for row in sampled_rows], dtype=np.float64)
        out[f"{metric}_ci_low"] = float(np.percentile(values, 2.5)) if values.size else None
        out[f"{metric}_ci_high"] = float(np.percentile(values, 97.5)) if values.size else None
    out["seed_count"] = len(seed_payloads)
    out["bootstrap_iterations"] = iterations
    out["block_count_min"] = min(payload["block_count"] for payload in seed_payloads) if seed_payloads else 0
    out["block_count_max"] = max(payload["block_count"] for payload in seed_payloads) if seed_payloads else 0
    return out


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    run_root = args.run_root if args.run_root.is_absolute() else root / args.run_root
    input_path = args.input if args.input.is_absolute() else root / args.input
    split_manifest = args.split_manifest if args.split_manifest.is_absolute() else root / args.split_manifest
    output_dir = args.output_dir if args.output_dir else run_root / "multiseed_summary"
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if not run_root.exists():
        raise SystemExit(f"Run root not found: {run_root}")
    strict_rows = load_strict_split_rows(split_manifest)
    metadata = load_training_metadata(input_path, strict_rows)
    device = choose_device(args.device)

    comparison_specs = [parse_comparison(item) for item in (args.comparison or list(DEFAULT_COMPARISONS))]
    seed_roots = sorted(path for path in run_root.glob(args.seed_glob) if path.is_dir())
    if not seed_roots:
        raise SystemExit(f"No seed directories matched {args.seed_glob} under {run_root}")

    prediction_cache: dict[tuple[str, str], dict[str, Any]] = {}

    def pred(seed_root: Path, run_name: str) -> dict[str, Any]:
        key = (str(seed_root), run_name)
        if key not in prediction_cache:
            # Keep seed-specific caches separate. The generic loader names cache
            # files by run_dir.name, which is identical across seed_42/43/44.
            # Without this subdirectory, different seeds silently collide.
            seed_cache_root = output_dir / "_prediction_by_seed" / seed_root.name
            prediction_cache[key] = load_run_predictions(
                seed_root / run_name,
                seed_cache_root,
                device,
                args.batch_size,
                args.num_workers,
                args.prefetch_factor,
                args.force_predict,
            )
        return prediction_cache[key]

    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for comp_idx, (name, left_run, right_run, block_field) in enumerate(comparison_specs):
        seed_payloads: list[dict[str, Any]] = []
        for seed_root in seed_roots:
            if not available(seed_root, left_run) or not available(seed_root, right_run):
                skipped.append(
                    {
                        "comparison": name,
                        "seed_root": str(seed_root),
                        "left_run": left_run,
                        "right_run": right_run,
                        "reason": "missing checkpoint or manifest",
                    }
                )
                continue
            aligned = align_predictions(pred(seed_root, left_run), pred(seed_root, right_run))
            block_values = [
                str(
                    metadata.get(accession, {}).get(block_field, "")
                    or strict_rows.get(str(accession), {}).get(block_field, "")
                )
                for accession in aligned["accessions"]
            ]
            bmap = block_indices(block_values)
            point = delta_row_from_arrays(
                aligned["y_true"],
                aligned["left_prob"],
                aligned["right_prob"],
                aligned["left_thresholds"],
                aligned["right_thresholds"],
                aligned["label_names"],
            )
            seed_payloads.append(
                {
                    "seed_root": seed_root.name,
                    "aligned": aligned,
                    "block_map": bmap,
                    "block_count": len(bmap),
                    "point": point,
                }
            )
        if not seed_payloads:
            continue
        row = {
            "comparison": name,
            "left_run": left_run,
            "right_run": right_run,
            "block_unit": block_field,
            **bootstrap_seed_averaged(seed_payloads, args.bootstrap_iterations, args.seed + comp_idx),
        }
        rows.append(row)

    write_tsv(output_dir / "multiseed_paired_block_bootstrap.tsv", rows)
    write_tsv(output_dir / "multiseed_paired_block_bootstrap_skipped.tsv", skipped)
    report = {
        "run_root": str(run_root),
        "seed_roots": [str(path) for path in seed_roots],
        "completed_comparisons": len(rows),
        "skipped": len(skipped),
        "output_tsv": str(output_dir / "multiseed_paired_block_bootstrap.tsv"),
        "claim_frame": "Use seed-averaged paired block bootstrap as the primary model-comparison statistic.",
    }
    (output_dir / "multiseed_paired_block_bootstrap_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
