from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from context_features import (
    CONTEXT_CATEGORY_FIELDS,
    context_numeric_field_names,
    derive_host_supergroup,
    normalize_segment_bucket,
    parse_json_list,
)
from train_overnight_baseline import LABEL_RULES, label_hits, normalize_text


WINDOW_RADIUS = 2
UNKNOWN_MARKERS = ("hypothetical protein", "uncharacterized", "unknown protein")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build genome and host context features for ViruFunc-FM protein training rows.")
    parser.add_argument(
        "--input",
        default="data/processed/training/viral_protein_training_index.tsv.gz",
        help="Protein-level training index table",
    )
    parser.add_argument(
        "--output",
        default="data/processed/training/viral_protein_context_features.tsv.gz",
        help="Per-protein context feature table",
    )
    parser.add_argument(
        "--report",
        default="data/processed/training/viral_protein_context_features_report.json",
        help="JSON summary report",
    )
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


def is_hypothetical(text: str) -> bool:
    return any(marker in text for marker in UNKNOWN_MARKERS)


def build_output_fields() -> list[str]:
    label_names = [rule.name for rule in LABEL_RULES]
    return ["protein_accession", *CONTEXT_CATEGORY_FIELDS, *context_numeric_field_names(label_names)]


def load_rows(path: Path, debug_limit: int) -> tuple[dict[str, list[dict[str, object]]], int]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
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

            text = normalize_text(row)
            label_ids = tuple(sorted(set(label_hits(text))))
            grouped[genome_key(row)].append(
                {
                    "protein_accession": protein_accession,
                    "feature_type": row.get("protein_feature_type", "").strip() or "__MISSING__",
                    "segment_bucket": normalize_segment_bucket(row.get("source_segment", "")),
                    "host_supergroup": derive_host_supergroup(row.get("host_lineages_json", ""), row.get("source_host", "")),
                    "host_taxid_count": len(parse_json_list(row.get("host_tax_ids_json", ""))),
                    "host_lineage_count": len(parse_json_list(row.get("host_lineages_json", ""))),
                    "label_ids": label_ids,
                    "is_hypothetical": is_hypothetical(text),
                    "is_mat_peptide": row.get("protein_feature_type", "").strip() == "mat_peptide",
                    "cds_start": maybe_int(row.get("cds_start", "0")),
                    "cds_end": maybe_int(row.get("cds_end", "0")),
                    "cds_strand": row.get("cds_strand", "").strip(),
                    "row_order": row_idx,
                }
            )
            loaded += 1
    return grouped, loaded


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


def local_label_counts(rows: list[dict[str, object]], center_idx: int) -> list[int]:
    counts = [0] * len(LABEL_RULES)
    left = max(0, center_idx - WINDOW_RADIUS)
    right = min(len(rows), center_idx + WINDOW_RADIUS + 1)
    for idx in range(left, right):
        if idx == center_idx:
            continue
        for label_id in rows[idx]["label_ids"]:
            counts[int(label_id)] += 1
    return counts


def genome_label_counts(rows: list[dict[str, object]]) -> list[int]:
    counts = [0] * len(LABEL_RULES)
    for row in rows:
        for label_id in row["label_ids"]:
            counts[int(label_id)] += 1
    return counts


def write_context_table(
    grouped_rows: dict[str, list[dict[str, object]]],
    output_path: Path,
) -> dict[str, object]:
    label_names = [rule.name for rule in LABEL_RULES]
    output_fields = build_output_fields()
    writer, handle = open_tsv_writer(output_path, output_fields)

    protein_count = 0
    genome_count = 0
    host_supergroups: set[str] = set()
    segment_buckets: set[str] = set()

    try:
        for rows in grouped_rows.values():
            ordered = sort_genome_rows(rows)
            genome_count += 1
            total = len(ordered)
            if total == 0:
                continue

            hypothetical_fraction = sum(1 for row in ordered if row["is_hypothetical"]) / total
            mat_fraction = sum(1 for row in ordered if row["is_mat_peptide"]) / total
            genome_counts = genome_label_counts(ordered)

            for idx, row in enumerate(ordered):
                prev_row = ordered[idx - 1] if idx > 0 else None
                next_row = ordered[idx + 1] if idx + 1 < total else None
                local_counts = local_label_counts(ordered, idx)
                self_hits = set(int(label_id) for label_id in row["label_ids"])

                output_row = {
                    "protein_accession": row["protein_accession"],
                    "context_prev_feature_type": prev_row["feature_type"] if prev_row else "__START__",
                    "context_next_feature_type": next_row["feature_type"] if next_row else "__END__",
                    "context_host_supergroup": row["host_supergroup"],
                    "context_segment_bucket": row["segment_bucket"],
                    "context_log_genome_protein_count": float_text(safe_log1p(total)),
                    "context_genome_hypothetical_fraction": float_text(hypothetical_fraction),
                    "context_genome_mat_peptide_fraction": float_text(mat_fraction),
                    "context_has_prev_neighbor": float_text(1.0 if prev_row else 0.0),
                    "context_has_next_neighbor": float_text(1.0 if next_row else 0.0),
                    "context_same_strand_prev": float_text(
                        1.0 if prev_row and prev_row["cds_strand"] and prev_row["cds_strand"] == row["cds_strand"] else 0.0
                    ),
                    "context_same_strand_next": float_text(
                        1.0 if next_row and next_row["cds_strand"] and next_row["cds_strand"] == row["cds_strand"] else 0.0
                    ),
                    "context_log_host_taxid_count": float_text(safe_log1p(int(row["host_taxid_count"]))),
                    "context_log_host_lineage_count": float_text(safe_log1p(int(row["host_lineage_count"]))),
                }

                denom = max(total - 1, 1)
                for label_idx, label_name in enumerate(label_names):
                    remaining = max(genome_counts[label_idx] - (1 if label_idx in self_hits else 0), 0)
                    output_row[f"context_genome_{label_name}_fraction"] = float_text(remaining / denom)
                    output_row[f"context_local_{label_name}_count"] = float_text(float(local_counts[label_idx]))

                writer.writerow(output_row)
                protein_count += 1
                host_supergroups.add(str(row["host_supergroup"]))
                segment_buckets.add(str(row["segment_bucket"]))
    finally:
        handle.close()

    return {
        "protein_rows": protein_count,
        "genome_groups": genome_count,
        "host_supergroup_count": len(host_supergroups),
        "segment_bucket_count": len(segment_buckets),
        "context_category_fields": CONTEXT_CATEGORY_FIELDS,
        "context_numeric_field_count": len(output_fields) - 1 - len(CONTEXT_CATEGORY_FIELDS),
    }


def save_report(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    args = parse_args()
    root = repo_root()
    input_path = (root / args.input).resolve()
    output_path = (root / args.output).resolve()
    report_path = (root / args.report).resolve()

    grouped_rows, loaded_rows = load_rows(input_path, args.debug_limit)
    summary = write_context_table(grouped_rows, output_path)
    report = {
        "created_at": timestamp(),
        "input": str(input_path),
        "output": str(output_path),
        "rows_loaded": loaded_rows,
        **summary,
    }
    save_report(report_path, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
