from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from context_features import derive_host_supergroup, normalize_segment_bucket, parse_json_list
from label_rules import LABEL_RULES, label_hits, normalize_text
from task_mode_config import TASK_MODE_ORDER, prior_context_numeric_fields, task_mode_feature_lists
from train_overnight_baseline import SPLIT_SCHEME_TO_COLUMN, load_split_assignments


UNKNOWN_MARKERS = ("hypothetical protein", "uncharacterized", "unknown protein")
WINDOW_RADIUS = 2


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build split-aware context features with explicit task-mode gating.")
    parser.add_argument(
        "--input",
        default="data/processed/training/viral_protein_training_index.tsv.gz",
        help="Protein-level training index table",
    )
    parser.add_argument(
        "--split-manifest",
        default="data/processed/splits/viral_protein_strict_splits.tsv.gz",
        help="Strict split manifest",
    )
    parser.add_argument(
        "--split-scheme",
        default="family_holdout",
        choices=sorted(name for name, column in SPLIT_SCHEME_TO_COLUMN.items() if column is not None),
        help="Strict split scheme to use for train-only priors",
    )
    parser.add_argument(
        "--task-mode",
        default="genome_aware_denovo",
        choices=TASK_MODE_ORDER,
        help="Task mode that determines which features are materialized",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed/context",
        help="Directory for task-mode-specific context tables",
    )
    parser.add_argument("--window-radius", type=int, default=WINDOW_RADIUS, help="Neighbor radius for local train priors")
    parser.add_argument("--debug-limit", type=int, default=0, help="Optional row cap for smoke tests")
    return parser.parse_args()


def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def open_tsv_writer(path: Path, fieldnames: list[str]) -> tuple[csv.DictWriter, gzip.GzipFile]:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = gzip.open(path, "wt", encoding="utf-8", newline="")
    writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
    writer.writeheader()
    return writer, handle


def maybe_int(value: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def safe_log1p(value: int | float) -> float:
    return math.log1p(max(float(value), 0.0))


def float_text(value: float) -> str:
    return f"{value:.6f}"


def genome_key(row: dict[str, str]) -> str:
    return (
        row.get("genome_version", "").strip()
        or row.get("genome_accession", "").strip()
        or row.get("virus_tax_id", "").strip()
        or row.get("protein_accession", "").strip()
    )


def length_bin(length_aa: int) -> str:
    if length_aa <= 0:
        return "len_0"
    lower = int(math.floor(math.log2(length_aa)))
    upper = lower + 1
    return f"len2^{lower}-{upper}"


def is_unknown_text(text: str) -> bool:
    return any(marker in text for marker in UNKNOWN_MARKERS)


def sort_genome_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return sorted(
        rows,
        key=lambda row: (
            0 if int(row["cds_start"]) > 0 else 1,
            int(row["cds_start"]) if int(row["cds_start"]) > 0 else 10**12,
            int(row["cds_end"]) if int(row["cds_end"]) > 0 else 10**12,
            int(row["row_order"]),
        ),
    )


def load_rows(
    path: Path,
    split_assignments: dict[str, int],
    task_mode: str,
    debug_limit: int,
) -> tuple[dict[str, list[dict[str, object]]], dict[str, int], int]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    split_counts: Counter[str] = Counter()
    loaded = 0
    with open_text(path) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row_idx, row in enumerate(reader, start=1):
            if debug_limit and row_idx > debug_limit:
                break

            protein_accession = row.get("protein_accession", "").strip()
            sequence = row.get("protein_sequence", "").strip()
            if not protein_accession or not sequence:
                continue
            split_id = split_assignments.get(protein_accession)
            if split_id is None:
                raise RuntimeError(f"Missing split assignment for protein '{protein_accession}'")
            split_name = {0: "train", 1: "val", 2: "test"}[split_id]

            label_ids: tuple[int, ...] = ()
            unknown_flag = False
            if task_mode == "annotation_refinement":
                text = normalize_text(row)
                label_ids = tuple(sorted(set(label_hits(text))))
                unknown_flag = is_unknown_text(text)

            grouped[genome_key(row)].append(
                {
                    "protein_accession": protein_accession,
                    "split_name": split_name,
                    "feature_type": row.get("protein_feature_type", "").strip() or "__MISSING__",
                    "segment_bucket": normalize_segment_bucket(row.get("source_segment", "")),
                    "host_supergroup": derive_host_supergroup(row.get("host_lineages_json", ""), row.get("source_host", "")),
                    "host_taxid_count": len(parse_json_list(row.get("host_tax_ids_json", ""))),
                    "host_lineage_count": len(parse_json_list(row.get("host_lineages_json", ""))),
                    "label_ids": label_ids,
                    "is_hypothetical": unknown_flag,
                    "is_mat_peptide": row.get("protein_feature_type", "").strip() == "mat_peptide",
                    "cds_start": maybe_int(row.get("cds_start", "0")),
                    "cds_end": maybe_int(row.get("cds_end", "0")),
                    "cds_strand": row.get("cds_strand", "").strip(),
                    "protein_length_aa": maybe_int(row.get("protein_length_aa", "0")),
                    "row_order": row_idx,
                }
            )
            split_counts[split_name] += 1
            loaded += 1
    return grouped, dict(split_counts), loaded


def compute_gap_and_overlap(left: dict[str, object] | None, right: dict[str, object] | None) -> tuple[float, float]:
    if left is None or right is None:
        return 0.0, 0.0
    left_end = int(left["cds_end"])
    right_start = int(right["cds_start"])
    if left_end <= 0 or right_start <= 0:
        return 0.0, 0.0
    gap_nt = max(0, right_start - left_end - 1)
    overlap_nt = max(0, left_end - right_start + 1)
    return safe_log1p(gap_nt), safe_log1p(overlap_nt)


def safe_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def safe_std(values: list[float], mean_value: float) -> float:
    if not values:
        return 0.0
    variance = sum((value - mean_value) ** 2 for value in values) / len(values)
    return math.sqrt(max(variance, 0.0))


def local_train_label_counts(rows: list[dict[str, object]], center_idx: int, window_radius: int, label_count: int) -> list[int]:
    counts = [0] * label_count
    left = max(0, center_idx - window_radius)
    right = min(len(rows), center_idx + window_radius + 1)
    for idx in range(left, right):
        if idx == center_idx or rows[idx]["split_name"] != "train":
            continue
        for label_id in rows[idx]["label_ids"]:
            counts[int(label_id)] += 1
    return counts


def genome_train_label_counts(rows: list[dict[str, object]], label_count: int) -> list[int]:
    counts = [0] * label_count
    for row in rows:
        if row["split_name"] != "train":
            continue
        for label_id in row["label_ids"]:
            counts[int(label_id)] += 1
    return counts


def select_output_fields(task_mode: str) -> list[str]:
    feature_lists = task_mode_feature_lists(task_mode, with_biophysics=False)
    return ["protein_accession", *feature_lists["context_category_fields"], *feature_lists["context_numeric_fields"]]


def write_context_table(
    grouped_rows: dict[str, list[dict[str, object]]],
    output_path: Path,
    task_mode: str,
    window_radius: int,
) -> dict[str, object]:
    output_fields = select_output_fields(task_mode)
    writer, handle = open_tsv_writer(output_path, output_fields)
    feature_lists = task_mode_feature_lists(task_mode, with_biophysics=False)

    label_names = [rule.name for rule in LABEL_RULES]
    protein_rows = 0
    genome_groups = 0
    prior_nonzero_rows = 0
    feature_type_counter: Counter[str] = Counter()

    try:
        for rows in grouped_rows.values():
            ordered = sort_genome_rows(rows)
            genome_groups += 1
            total = len(ordered)
            if total == 0:
                continue

            genome_hypothetical_fraction = sum(1 for row in ordered if row["is_hypothetical"]) / total
            genome_mat_fraction = sum(1 for row in ordered if row["is_mat_peptide"]) / total
            genome_train_counts = genome_train_label_counts(ordered, len(label_names))
            genome_train_total = sum(1 for row in ordered if row["split_name"] == "train")
            segment_count = len({str(row["segment_bucket"]) for row in ordered if str(row["segment_bucket"]) != "__UNSEGMENTED__"})
            if segment_count == 0:
                segment_count = 1
            coordinate_starts = [int(row["cds_start"]) for row in ordered if int(row["cds_start"]) > 0]
            coordinate_ends = [int(row["cds_end"]) for row in ordered if int(row["cds_end"]) > 0]
            genome_span_nt = 0
            if coordinate_starts and coordinate_ends:
                genome_span_nt = max(0, max(coordinate_ends) - min(coordinate_starts) + 1)
            protein_lengths = [max(0, int(row["protein_length_aa"])) for row in ordered]
            mean_protein_length = safe_mean([float(length) for length in protein_lengths])
            length_std = safe_std([float(length) for length in protein_lengths], mean_protein_length)
            length_cv = (length_std / mean_protein_length) if mean_protein_length > 0 else 0.0
            orf_density_per_kb = (total * 1000.0 / genome_span_nt) if genome_span_nt > 0 else 0.0

            for idx, row in enumerate(ordered):
                prev_row = ordered[idx - 1] if idx > 0 else None
                next_row = ordered[idx + 1] if idx + 1 < total else None
                prev_gap_nt, prev_overlap_nt = compute_gap_and_overlap(prev_row, row)
                next_gap_nt, next_overlap_nt = compute_gap_and_overlap(row, next_row)
                relative_order_fraction = 0.0 if total == 1 else idx / (total - 1)

                values = {
                    "protein_accession": row["protein_accession"],
                    "context_host_supergroup": row["host_supergroup"],
                    "context_segment_bucket": row["segment_bucket"],
                    "context_prev_length_bin": length_bin(int(prev_row["protein_length_aa"])) if prev_row else "__START__",
                    "context_next_length_bin": length_bin(int(next_row["protein_length_aa"])) if next_row else "__END__",
                    "context_log_genome_protein_count": float_text(safe_log1p(total)),
                    "context_relative_order_fraction": float_text(relative_order_fraction),
                    "context_segment_count": float_text(float(segment_count)),
                    "context_log_genome_span_nt": float_text(safe_log1p(genome_span_nt)),
                    "context_orf_density_per_kb": float_text(orf_density_per_kb),
                    "context_genome_mean_protein_length": float_text(mean_protein_length),
                    "context_genome_protein_length_cv": float_text(length_cv),
                    "context_has_prev_neighbor": float_text(1.0 if prev_row else 0.0),
                    "context_has_next_neighbor": float_text(1.0 if next_row else 0.0),
                    "context_prev_gap_nt": float_text(prev_gap_nt),
                    "context_next_gap_nt": float_text(next_gap_nt),
                    "context_prev_overlap_nt": float_text(prev_overlap_nt),
                    "context_next_overlap_nt": float_text(next_overlap_nt),
                    "context_same_strand_prev": float_text(
                        1.0 if prev_row and prev_row["cds_strand"] and prev_row["cds_strand"] == row["cds_strand"] else 0.0
                    ),
                    "context_same_strand_next": float_text(
                        1.0 if next_row and next_row["cds_strand"] and next_row["cds_strand"] == row["cds_strand"] else 0.0
                    ),
                    "context_log_host_taxid_count": float_text(safe_log1p(int(row["host_taxid_count"]))),
                    "context_log_host_lineage_count": float_text(safe_log1p(int(row["host_lineage_count"]))),
                    "context_prev_feature_type": prev_row["feature_type"] if prev_row else "__START__",
                    "context_next_feature_type": next_row["feature_type"] if next_row else "__END__",
                    "context_genome_hypothetical_fraction": float_text(genome_hypothetical_fraction),
                    "context_genome_mat_peptide_fraction": float_text(genome_mat_fraction),
                    "context_prev_is_hypothetical": float_text(1.0 if prev_row and prev_row["is_hypothetical"] else 0.0),
                    "context_next_is_hypothetical": float_text(1.0 if next_row and next_row["is_hypothetical"] else 0.0),
                }

                if task_mode == "annotation_refinement":
                    train_local_counts = local_train_label_counts(ordered, idx, window_radius, len(label_names))
                    self_train_hits = set(int(label_id) for label_id in row["label_ids"]) if row["split_name"] == "train" else set()
                    train_denom = max(genome_train_total - (1 if self_train_hits else 0), 1)
                    if any(train_local_counts):
                        prior_nonzero_rows += 1
                    for label_idx, label_name in enumerate(label_names):
                        remaining = max(genome_train_counts[label_idx] - (1 if label_idx in self_train_hits else 0), 0)
                        values[f"context_train_genome_{label_name}_fraction"] = float_text(remaining / train_denom)
                        values[f"context_train_local_{label_name}_count"] = float_text(float(train_local_counts[label_idx]))

                row_payload = {"protein_accession": row["protein_accession"]}
                for field_name in feature_lists["context_category_fields"] + feature_lists["context_numeric_fields"]:
                    row_payload[field_name] = values[field_name]

                writer.writerow(row_payload)
                feature_type_counter[str(row["feature_type"])] += 1
                protein_rows += 1
    finally:
        handle.close()

    return {
        "protein_rows": protein_rows,
        "genome_groups": genome_groups,
        "feature_type_count": len(feature_type_counter),
        "prior_nonzero_rows": prior_nonzero_rows,
        "output_fields": output_fields,
    }


def main() -> int:
    args = parse_args()
    root = repo_root()
    input_path = (root / args.input).resolve()
    split_manifest_path = (root / args.split_manifest).resolve()
    output_dir = (root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    split_column = SPLIT_SCHEME_TO_COLUMN[args.split_scheme]
    if split_column is None:
        raise ValueError(f"Split scheme '{args.split_scheme}' is not strict and cannot drive split-aware priors.")

    split_assignments = load_split_assignments(split_manifest_path, split_column)
    grouped_rows, split_counts, loaded_rows = load_rows(input_path, split_assignments, args.task_mode, args.debug_limit)

    stem = f"viral_protein_context.{args.split_scheme}.{args.task_mode}"
    output_path = output_dir / f"{stem}.tsv.gz"
    report_path = output_dir / f"{stem}.report.json"

    summary = write_context_table(grouped_rows, output_path, args.task_mode, args.window_radius)
    report = {
        "created_at": timestamp(),
        "input": str(input_path),
        "split_manifest": str(split_manifest_path),
        "split_scheme": args.split_scheme,
        "task_mode": args.task_mode,
        "output": str(output_path),
        "rows_loaded": loaded_rows,
        "split_counts": split_counts,
        "window_radius": args.window_radius,
        **summary,
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
