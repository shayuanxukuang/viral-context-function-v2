from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from context_features import derive_baltimore_like_class, derive_virus_family


LABEL_GROUPS = {
    "replication": ("polymerase", "helicase", "nuclease", "methyltransferase", "ligase"),
    "processing": ("protease", "polyprotein", "transcription_regulator"),
    "structural_assembly": ("capsid_head", "portal_terminase_packaging", "tail_assembly", "nucleocapsid"),
    "membrane_entry": ("envelope_glycoprotein", "membrane_matrix", "tail_fiber_receptor"),
    "lysis_integration": ("lysis", "integrase_recombinase"),
}


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    return (root / path).resolve()


def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a context-dependence atlas with block bootstrap and permutation tests.")
    parser.add_argument("--protein-run", required=True)
    parser.add_argument("--context-run", required=True)
    parser.add_argument("--input", default="data/processed/training/viral_protein_training_index.tsv.gz")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--min-stratum-size", type=int, default=100)
    parser.add_argument("--bootstrap-iterations", type=int, default=200)
    parser.add_argument("--permutation-iterations", type=int, default=200)
    parser.add_argument("--block-unit", choices=("auto", "genome", "family"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def maybe_float(value: Any) -> float | None:
    try:
        if value in {None, ""}:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def maybe_int(value: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def read_label_metrics(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            label_name = row.get("label", "").strip()
            if label_name:
                rows[label_name] = row
    return rows


def load_predictions(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with open_text(path) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            accession = row.get("protein_accession", "").strip()
            if not accession:
                continue
            rows[accession] = {
                "true_labels": set(json.loads(row.get("true_labels", "[]"))),
                "predicted_labels": set(json.loads(row.get("predicted_labels", "[]"))),
                "top_label": row.get("top_label", "").strip(),
                "top_probability": maybe_float(row.get("top_probability")),
            }
    return rows


def genome_key(row: dict[str, str]) -> str:
    return (
        row.get("genome_version", "").strip()
        or row.get("genome_accession", "").strip()
        or row.get("virus_tax_id", "").strip()
        or row.get("protein_accession", "").strip()
    )


def sort_key(row: dict[str, Any]) -> tuple[int, int, int, int]:
    start = int(row["cds_start"])
    end = int(row["cds_end"])
    return (
        0 if start > 0 else 1,
        start if start > 0 else 10**12,
        end if end > 0 else 10**12,
        int(row["row_order"]),
    )


def overlap_stats(rows: list[dict[str, Any]]) -> tuple[str, float]:
    if len(rows) <= 1:
        return "none", 0.0
    ordered = sorted(rows, key=sort_key)
    overlaps = 0
    comparable = 0
    for left, right in zip(ordered, ordered[1:]):
        left_end = int(left["cds_end"])
        right_start = int(right["cds_start"])
        if left_end <= 0 or right_start <= 0:
            continue
        comparable += 1
        if left_end >= right_start:
            overlaps += 1
    if comparable == 0:
        return "unknown", 0.0
    density = overlaps / comparable
    if density == 0:
        return "none", density
    if density < 0.2:
        return "low", density
    if density < 0.5:
        return "medium", density
    return "high", density


def compression_bucket(value: float) -> str:
    if value <= 0:
        return "unknown"
    if value < 0.6:
        return "low"
    if value < 0.85:
        return "medium"
    return "high"


def is_putative_enveloped(rows: list[dict[str, Any]]) -> bool:
    for row in rows:
        text = f"{row.get('cds_product', '')} {row.get('protein_description', '')}".lower()
        if "envelope" in text or "glycoprotein" in text or "matrix protein" in text:
            return True
    return False


def load_metadata(path: Path, accessions_of_interest: set[str]) -> dict[str, dict[str, Any]]:
    genome_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    accessions_to_genome: dict[str, str] = {}
    per_accession_row: dict[str, dict[str, str]] = {}
    genome_anchor_rows: dict[str, dict[str, str]] = {}

    with open_text(path) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row_idx, row in enumerate(reader, start=1):
            accession = row.get("protein_accession", "").strip()
            if not accession:
                continue
            key = genome_key(row)
            genome_anchor_rows.setdefault(key, row)
            genome_rows[key].append(
                {
                    "protein_accession": accession,
                    "cds_start": maybe_int(row.get("cds_start", "0")),
                    "cds_end": maybe_int(row.get("cds_end", "0")),
                    "cds_product": row.get("cds_product", "").strip(),
                    "protein_description": row.get("protein_description", "").strip(),
                    "protein_length_aa": maybe_int(row.get("protein_length_aa", "0")),
                    "row_order": row_idx,
                }
            )
            accessions_to_genome[accession] = key
            if accession in accessions_of_interest:
                per_accession_row[accession] = row

    genome_metadata: dict[str, dict[str, Any]] = {}
    for key, rows in genome_rows.items():
        sample_row = genome_anchor_rows.get(key, {"source_segment": "", "virus_lineage": "", "source_mol_type": ""})
        starts = [int(item["cds_start"]) for item in rows if int(item["cds_start"]) > 0]
        ends = [int(item["cds_end"]) for item in rows if int(item["cds_end"]) > 0]
        genome_span_nt = max(0, max(ends) - min(starts) + 1) if starts and ends else 0
        total_coding_nt = sum(max(0, int(item["cds_end"]) - int(item["cds_start"]) + 1) for item in rows if int(item["cds_start"]) > 0 and int(item["cds_end"]) > 0)
        compression = (total_coding_nt / genome_span_nt) if genome_span_nt > 0 else 0.0
        overlap_bucket, overlap_density = overlap_stats(rows)
        genome_metadata[key] = {
            "overlap_density_bucket": overlap_bucket,
            "overlap_density": overlap_density,
            "genome_compression": compression,
            "genome_compression_bucket": compression_bucket(compression),
            "putative_enveloped": "1" if is_putative_enveloped(rows) else "0",
            "segmented": "1" if sample_row.get("source_segment", "").strip() else "0",
        }

    metadata: dict[str, dict[str, Any]] = {}
    for accession in accessions_of_interest:
        row = per_accession_row.get(accession)
        if row is None:
            continue
        key = accessions_to_genome.get(accession, accession)
        genome_meta = genome_metadata.get(key, {})
        metadata[accession] = {
            "genome_block": key,
            "virus_family": derive_virus_family(row.get("virus_lineage", "")),
            "baltimore_like_class": derive_baltimore_like_class(row.get("virus_lineage", ""), row.get("source_mol_type", "")),
            "segmented": genome_meta.get("segmented", "0"),
            "putative_enveloped": genome_meta.get("putative_enveloped", "0"),
            "overlap_density_bucket": genome_meta.get("overlap_density_bucket", "unknown"),
            "overlap_density": genome_meta.get("overlap_density", 0.0),
            "genome_compression_bucket": genome_meta.get("genome_compression_bucket", "unknown"),
            "genome_compression": genome_meta.get("genome_compression", 0.0),
        }
    return metadata


def label_f1_counts(true_positive: int, false_positive: int, false_negative: int) -> float | None:
    denominator = (2 * true_positive) + false_positive + false_negative
    if denominator <= 0:
        return None
    return (2 * true_positive) / denominator


def label_delta_from_block_rows(block_rows: dict[str, list[dict[str, Any]]], label_name: str, sampled_blocks: list[str] | None = None, swapped_blocks: set[str] | None = None) -> float | None:
    tp_protein = fp_protein = fn_protein = 0
    tp_context = fp_context = fn_context = 0
    block_keys = sampled_blocks if sampled_blocks is not None else list(block_rows.keys())
    swapped = swapped_blocks or set()
    for block_key in block_keys:
        rows = block_rows.get(block_key, [])
        for row in rows:
            protein_key = "context_predicted_labels" if block_key in swapped else "protein_predicted_labels"
            context_key = "protein_predicted_labels" if block_key in swapped else "context_predicted_labels"
            true_hit = label_name in row["true_labels"]
            protein_hit = label_name in row[protein_key]
            context_hit = label_name in row[context_key]
            tp_protein += int(true_hit and protein_hit)
            fp_protein += int((not true_hit) and protein_hit)
            fn_protein += int(true_hit and (not protein_hit))
            tp_context += int(true_hit and context_hit)
            fp_context += int((not true_hit) and context_hit)
            fn_context += int(true_hit and (not context_hit))
    protein_f1 = label_f1_counts(tp_protein, fp_protein, fn_protein)
    context_f1 = label_f1_counts(tp_context, fp_context, fn_context)
    if protein_f1 is None or context_f1 is None:
        return None
    return context_f1 - protein_f1


def group_delta_from_block_rows(block_rows: dict[str, list[dict[str, Any]]], labels: tuple[str, ...], sampled_blocks: list[str] | None = None, swapped_blocks: set[str] | None = None) -> tuple[float | None, int]:
    label_set = set(labels)
    tp_protein = fp_protein = fn_protein = 0
    tp_context = fp_context = fn_context = 0
    positives = 0
    block_keys = sampled_blocks if sampled_blocks is not None else list(block_rows.keys())
    swapped = swapped_blocks or set()
    for block_key in block_keys:
        rows = block_rows.get(block_key, [])
        for row in rows:
            protein_key = "context_predicted_labels" if block_key in swapped else "protein_predicted_labels"
            context_key = "protein_predicted_labels" if block_key in swapped else "context_predicted_labels"
            true_set = set(row["true_labels"]) & label_set
            protein_set = set(row[protein_key]) & label_set
            context_set = set(row[context_key]) & label_set
            positives += len(true_set)
            tp_protein += len(true_set & protein_set)
            fp_protein += len(protein_set - true_set)
            fn_protein += len(true_set - protein_set)
            tp_context += len(true_set & context_set)
            fp_context += len(context_set - true_set)
            fn_context += len(true_set - context_set)
    protein_f1 = label_f1_counts(tp_protein, fp_protein, fn_protein)
    context_f1 = label_f1_counts(tp_context, fp_context, fn_context)
    if protein_f1 is None or context_f1 is None:
        return None, positives
    return context_f1 - protein_f1, positives


def bootstrap_distribution(
    block_rows: dict[str, list[dict[str, Any]]],
    stat_fn,
    iterations: int,
    rng: np.random.Generator,
) -> list[float]:
    block_keys = list(block_rows.keys())
    if not block_keys:
        return []
    values: list[float] = []
    for _ in range(iterations):
        sampled = [block_keys[idx] for idx in rng.integers(0, len(block_keys), size=len(block_keys))]
        value = stat_fn(sampled_blocks=sampled)
        if value is not None:
            values.append(float(value))
    return values


def permutation_distribution(
    block_rows: dict[str, list[dict[str, Any]]],
    stat_fn,
    iterations: int,
    rng: np.random.Generator,
) -> list[float]:
    block_keys = list(block_rows.keys())
    if not block_keys:
        return []
    values: list[float] = []
    for _ in range(iterations):
        swap_mask = rng.random(len(block_keys)) < 0.5
        swapped_blocks = {block_key for block_key, should_swap in zip(block_keys, swap_mask) if should_swap}
        value = stat_fn(swapped_blocks=swapped_blocks)
        if value is not None:
            values.append(float(value))
    return values


def ci_bounds(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def permutation_pvalue(observed: float | None, null_values: list[float]) -> float | None:
    if observed is None or not null_values:
        return None
    observed_abs = abs(float(observed))
    return float(sum(abs(value) >= observed_abs for value in null_values) / len(null_values))


def split_scheme_from_run(run_dir: Path) -> str:
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    split_strategy = manifest.get("split_strategy", {}) or {}
    return str(split_strategy.get("scheme", "") or "")


def main() -> int:
    args = parse_args()
    root = repo_root()
    protein_run = resolve_path(root, args.protein_run)
    context_run = resolve_path(root, args.context_run)
    input_path = resolve_path(root, args.input)
    output_dir = resolve_path(root, args.output_dir) if args.output_dir else context_run / "context_dependence_atlas_v2"
    output_dir.mkdir(parents=True, exist_ok=True)

    protein_label_metrics = read_label_metrics(protein_run / "test_label_metrics.tsv")
    context_label_metrics = read_label_metrics(context_run / "test_label_metrics.tsv")
    all_labels = sorted(set(protein_label_metrics) | set(context_label_metrics))

    protein_predictions = load_predictions(protein_run / "test_predictions.tsv.gz")
    context_predictions = load_predictions(context_run / "test_predictions.tsv.gz")
    shared_accessions = sorted(set(protein_predictions) & set(context_predictions))
    metadata = load_metadata(input_path, set(shared_accessions))
    split_scheme = split_scheme_from_run(context_run)

    paired_rows: list[dict[str, Any]] = []
    for accession in shared_accessions:
        if accession not in metadata:
            continue
        paired_rows.append(
            {
                "protein_accession": accession,
                "true_labels": protein_predictions[accession]["true_labels"],
                "protein_predicted_labels": protein_predictions[accession]["predicted_labels"],
                "context_predicted_labels": context_predictions[accession]["predicted_labels"],
                **metadata[accession],
            }
        )

    if args.block_unit == "auto":
        block_field = "virus_family" if split_scheme == "family_holdout" else "genome_block"
    elif args.block_unit == "family":
        block_field = "virus_family"
    else:
        block_field = "genome_block"

    rng = np.random.default_rng(args.seed)

    label_delta_rows: list[dict[str, Any]] = []
    block_rows_all: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in paired_rows:
        block_rows_all[str(row[block_field])].append(row)

    for label_name in all_labels:
        protein_row = protein_label_metrics.get(label_name, {})
        context_row = context_label_metrics.get(label_name, {})
        protein_ap = maybe_float(protein_row.get("average_precision"))
        context_ap = maybe_float(context_row.get("average_precision"))
        protein_f1 = maybe_float(protein_row.get("f1"))
        context_f1 = maybe_float(context_row.get("f1"))
        observed_delta_f1 = label_delta_from_block_rows(block_rows_all, label_name)
        bootstrap_values = bootstrap_distribution(
            block_rows_all,
            stat_fn=lambda sampled_blocks=None, swapped_blocks=None, label_name=label_name: label_delta_from_block_rows(
                block_rows_all,
                label_name,
                sampled_blocks=sampled_blocks,
                swapped_blocks=swapped_blocks,
            ),
            iterations=args.bootstrap_iterations,
            rng=rng,
        )
        permutation_values = permutation_distribution(
            block_rows_all,
            stat_fn=lambda sampled_blocks=None, swapped_blocks=None, label_name=label_name: label_delta_from_block_rows(
                block_rows_all,
                label_name,
                sampled_blocks=sampled_blocks,
                swapped_blocks=swapped_blocks,
            ),
            iterations=args.permutation_iterations,
            rng=rng,
        )
        ci_low, ci_high = ci_bounds(bootstrap_values)
        label_delta_rows.append(
            {
                "label": label_name,
                "label_group": next((group for group, labels in LABEL_GROUPS.items() if label_name in labels), "other"),
                "protein_average_precision": protein_ap,
                "context_average_precision": context_ap,
                "delta_average_precision": None if protein_ap is None or context_ap is None else context_ap - protein_ap,
                "protein_f1_from_label_metrics": protein_f1,
                "context_f1_from_label_metrics": context_f1,
                "delta_f1_from_label_metrics": None if protein_f1 is None or context_f1 is None else context_f1 - protein_f1,
                "delta_f1_from_predictions": observed_delta_f1,
                "delta_f1_ci_low": ci_low,
                "delta_f1_ci_high": ci_high,
                "delta_f1_permutation_pvalue": permutation_pvalue(observed_delta_f1, permutation_values),
                "block_unit": block_field,
                "protein_support": protein_row.get("support") or context_row.get("support") or "",
            }
        )
    write_tsv(output_dir / "label_deltas.tsv", label_delta_rows)

    group_rows: list[dict[str, Any]] = []
    for group_name, labels in LABEL_GROUPS.items():
        observed_delta, positives = group_delta_from_block_rows(block_rows_all, labels)
        bootstrap_values = bootstrap_distribution(
            block_rows_all,
            stat_fn=lambda sampled_blocks=None, swapped_blocks=None, labels=labels: group_delta_from_block_rows(
                block_rows_all,
                labels,
                sampled_blocks=sampled_blocks,
                swapped_blocks=swapped_blocks,
            )[0],
            iterations=args.bootstrap_iterations,
            rng=rng,
        )
        permutation_values = permutation_distribution(
            block_rows_all,
            stat_fn=lambda sampled_blocks=None, swapped_blocks=None, labels=labels: group_delta_from_block_rows(
                block_rows_all,
                labels,
                sampled_blocks=sampled_blocks,
                swapped_blocks=swapped_blocks,
            )[0],
            iterations=args.permutation_iterations,
            rng=rng,
        )
        ci_low, ci_high = ci_bounds(bootstrap_values)
        protein_f1_delta_entries = [
            float(row["delta_f1_from_predictions"])
            for row in label_delta_rows
            if row["label"] in labels and row["delta_f1_from_predictions"] is not None
        ]
        group_rows.append(
            {
                "label_group": group_name,
                "label_count": sum(1 for row in label_delta_rows if row["label"] in labels),
                "mean_delta_average_precision": (
                    float(np.mean([float(row["delta_average_precision"]) for row in label_delta_rows if row["label"] in labels and row["delta_average_precision"] is not None]))
                    if any(row["label"] in labels and row["delta_average_precision"] is not None for row in label_delta_rows)
                    else None
                ),
                "mean_delta_f1": float(np.mean(protein_f1_delta_entries)) if protein_f1_delta_entries else None,
                "delta_micro_f1": observed_delta,
                "delta_micro_f1_ci_low": ci_low,
                "delta_micro_f1_ci_high": ci_high,
                "delta_micro_f1_permutation_pvalue": permutation_pvalue(observed_delta, permutation_values),
                "positive_labels": positives,
                "block_unit": block_field,
            }
        )
    write_tsv(output_dir / "group_summary.tsv", group_rows)

    stratified_rows: list[dict[str, Any]] = []
    stratum_fields = (
        "putative_enveloped",
        "segmented",
        "baltimore_like_class",
        "overlap_density_bucket",
        "genome_compression_bucket",
    )
    for group_name, labels in LABEL_GROUPS.items():
        observed_delta, positives = group_delta_from_block_rows(block_rows_all, labels)
        bootstrap_values = bootstrap_distribution(
            block_rows_all,
            stat_fn=lambda sampled_blocks=None, swapped_blocks=None, labels=labels: group_delta_from_block_rows(
                block_rows_all,
                labels,
                sampled_blocks=sampled_blocks,
                swapped_blocks=swapped_blocks,
            )[0],
            iterations=args.bootstrap_iterations,
            rng=rng,
        )
        permutation_values = permutation_distribution(
            block_rows_all,
            stat_fn=lambda sampled_blocks=None, swapped_blocks=None, labels=labels: group_delta_from_block_rows(
                block_rows_all,
                labels,
                sampled_blocks=sampled_blocks,
                swapped_blocks=swapped_blocks,
            )[0],
            iterations=args.permutation_iterations,
            rng=rng,
        )
        ci_low, ci_high = ci_bounds(bootstrap_values)
        stratified_rows.append(
            {
                "scope": "overall",
                "stratum_field": "all",
                "stratum_value": "all",
                "label_group": group_name,
                "delta_micro_f1": observed_delta,
                "delta_micro_f1_ci_low": ci_low,
                "delta_micro_f1_ci_high": ci_high,
                "delta_micro_f1_permutation_pvalue": permutation_pvalue(observed_delta, permutation_values),
                "positive_labels": positives,
                "protein_count": len(paired_rows),
            }
        )

    for stratum_field in stratum_fields:
        buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in paired_rows:
            buckets[str(row[stratum_field])].append(row)
        for stratum_value, bucket_rows in buckets.items():
            if len(bucket_rows) < args.min_stratum_size:
                continue
            bucket_block_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in bucket_rows:
                bucket_block_rows[str(row[block_field])].append(row)
            for group_name, labels in LABEL_GROUPS.items():
                observed_delta, positives = group_delta_from_block_rows(bucket_block_rows, labels)
                bootstrap_values = bootstrap_distribution(
                    bucket_block_rows,
                    stat_fn=lambda sampled_blocks=None, swapped_blocks=None, labels=labels: group_delta_from_block_rows(
                        bucket_block_rows,
                        labels,
                        sampled_blocks=sampled_blocks,
                        swapped_blocks=swapped_blocks,
                    )[0],
                    iterations=args.bootstrap_iterations,
                    rng=rng,
                )
                permutation_values = permutation_distribution(
                    bucket_block_rows,
                    stat_fn=lambda sampled_blocks=None, swapped_blocks=None, labels=labels: group_delta_from_block_rows(
                        bucket_block_rows,
                        labels,
                        sampled_blocks=sampled_blocks,
                        swapped_blocks=swapped_blocks,
                    )[0],
                    iterations=args.permutation_iterations,
                    rng=rng,
                )
                ci_low, ci_high = ci_bounds(bootstrap_values)
                stratified_rows.append(
                    {
                        "scope": "stratified",
                        "stratum_field": stratum_field,
                        "stratum_value": stratum_value,
                        "label_group": group_name,
                        "delta_micro_f1": observed_delta,
                        "delta_micro_f1_ci_low": ci_low,
                        "delta_micro_f1_ci_high": ci_high,
                        "delta_micro_f1_permutation_pvalue": permutation_pvalue(observed_delta, permutation_values),
                        "positive_labels": positives,
                        "protein_count": len(bucket_rows),
                    }
                )
    write_tsv(output_dir / "stratified_group_summary.tsv", stratified_rows)

    atlas_report = {
        "created_at": timestamp(),
        "protein_run": str(protein_run),
        "context_run": str(context_run),
        "input": str(input_path),
        "shared_test_proteins": len(paired_rows),
        "split_scheme": split_scheme,
        "block_unit": block_field,
        "bootstrap_iterations": args.bootstrap_iterations,
        "permutation_iterations": args.permutation_iterations,
        "group_summary": group_rows,
    }
    (output_dir / "atlas_report.json").write_text(json.dumps(atlas_report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "shared_test_proteins": len(paired_rows), "block_unit": block_field}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
