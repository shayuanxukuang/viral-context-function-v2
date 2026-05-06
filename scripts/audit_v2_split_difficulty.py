from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from build_strict_splits import derive_virus_family
from label_rules import LABEL_RULES, label_hits, normalize_text
from train_overnight_baseline import assign_split


PARTITION_NAMES = {0: "train", 1: "val", 2: "test"}


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit V2 split difficulty and leakage-warning baselines.")
    parser.add_argument("--input", default="data/processed/training/viral_protein_training_index.tsv.gz")
    parser.add_argument("--split-manifest", default="data/processed/splits/viral_protein_strict_splits.tsv.gz")
    parser.add_argument("--output-dir", default="runs/v2_split_difficulty")
    parser.add_argument("--host-split-column", default="host_taxid_holdout_split")
    parser.add_argument("--debug-limit", type=int, default=0)
    return parser.parse_args()


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    return (root / path).resolve() if not path.is_absolute() else path.resolve()


def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def load_strict_rows(path: Path, debug_limit: int) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    with open_text(path) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for idx, row in enumerate(reader, start=1):
            if debug_limit and idx > debug_limit:
                break
            accession = row.get("protein_accession", "").strip()
            if accession:
                rows[accession] = dict(row)
    return rows


def iter_input_rows(path: Path, debug_limit: int):
    with open_text(path) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for idx, row in enumerate(reader, start=1):
            if debug_limit and idx > debug_limit:
                break
            yield row


def write_tsv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    positives = int(y_true.sum())
    if positives <= 0:
        return None
    order = np.argsort(-y_score, kind="mergesort")
    sorted_true = y_true[order].astype(np.float64)
    cumulative_tp = np.cumsum(sorted_true)
    precision = cumulative_tp / (np.arange(sorted_true.shape[0], dtype=np.float64) + 1.0)
    return float((precision * sorted_true).sum() / positives)


def fmax(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    if int(y_true.sum()) <= 0:
        return None
    thresholds = np.unique(y_score)
    if thresholds.shape[0] > 256:
        thresholds = np.quantile(thresholds, np.linspace(0.0, 1.0, 256))
    best = 0.0
    for threshold in thresholds:
        pred = y_score >= threshold
        tp = float(np.sum((pred == 1) & (y_true == 1)))
        fp = float(np.sum((pred == 1) & (y_true == 0)))
        fn = float(np.sum((pred == 0) & (y_true == 1)))
        denom = (2.0 * tp) + fp + fn
        score = 0.0 if denom <= 0 else (2.0 * tp) / denom
        if score > best:
            best = score
    return best


def macro(values: list[float | None]) -> float | None:
    valid = [value for value in values if value is not None]
    return None if not valid else float(np.mean(valid))


def build_records(input_path: Path, strict_rows: dict[str, dict[str, str]], debug_limit: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in iter_input_rows(input_path, debug_limit):
        accession = row.get("protein_accession", "").strip()
        strict = strict_rows.get(accession, {})
        family = strict.get("virus_family", "")
        if not family:
            family, _ = derive_virus_family(row.get("virus_lineage", ""))
        hits = set(label_hits(normalize_text(row)))
        labels = np.asarray([1 if idx in hits else 0 for idx, _rule in enumerate(LABEL_RULES)], dtype=np.uint8)
        records.append(
            {
                "protein_accession": accession,
                "genome_version": row.get("genome_version", "").strip(),
                "virus_tax_id": row.get("virus_tax_id", "").strip(),
                "virus_family": family,
                "host_taxid_key": strict.get("host_taxid_key", ""),
                "host_supergroup": strict.get("host_supergroup", ""),
                "sequence_sketch_key": strict.get("sequence_sketch_key", ""),
                "protein_sequence_sha256": row.get("protein_sequence_sha256", "").strip(),
                "default_split": PARTITION_NAMES[assign_split(row)],
                "family_holdout_split": strict.get("family_holdout_split", ""),
                "host_taxid_holdout_split": strict.get("host_taxid_holdout_split", ""),
                "host_supergroup_holdout_split": strict.get("host_supergroup_holdout_split", ""),
                "labels": labels,
            }
        )
    return records


def split_of(record: dict[str, Any], scheme: str, host_split_column: str) -> str:
    if scheme == "default":
        return str(record["default_split"])
    if scheme == "family_holdout":
        return str(record["family_holdout_split"])
    if scheme == "host_holdout":
        return str(record.get(host_split_column, ""))
    raise ValueError(f"Unsupported scheme: {scheme}")


def evaluate_label_transfer(
    records: list[dict[str, Any]],
    scheme: str,
    host_split_column: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    train = [record for record in records if split_of(record, scheme, host_split_column) == "train"]
    test = [record for record in records if split_of(record, scheme, host_split_column) == "test"]
    label_count = len(LABEL_RULES)

    sketch_counts: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(label_count, dtype=np.float64))
    sketch_totals: Counter[str] = Counter()
    sha_counts: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(label_count, dtype=np.float64))
    sha_totals: Counter[str] = Counter()
    for record in train:
        labels = record["labels"].astype(np.float64)
        sketch = str(record["sequence_sketch_key"])
        sha = str(record["protein_sequence_sha256"])
        if sketch:
            sketch_counts[sketch] += labels
            sketch_totals[sketch] += 1
        if sha:
            sha_counts[sha] += labels
            sha_totals[sha] += 1

    y_true = np.zeros((len(test), label_count), dtype=np.uint8)
    y_score = np.zeros((len(test), label_count), dtype=np.float32)
    exact_hits = 0
    sketch_hits = 0
    for row_idx, record in enumerate(test):
        y_true[row_idx] = record["labels"]
        sha = str(record["protein_sequence_sha256"])
        sketch = str(record["sequence_sketch_key"])
        if sha in sha_totals:
            y_score[row_idx] = (sha_counts[sha] / max(sha_totals[sha], 1)).astype(np.float32)
            exact_hits += 1
        elif sketch in sketch_totals:
            y_score[row_idx] = (sketch_counts[sketch] / max(sketch_totals[sketch], 1)).astype(np.float32)
            sketch_hits += 1

    label_rows: list[dict[str, Any]] = []
    aps: list[float | None] = []
    fmaxes: list[float | None] = []
    for idx, rule in enumerate(LABEL_RULES):
        ap = average_precision(y_true[:, idx], y_score[:, idx]) if len(test) else None
        fm = fmax(y_true[:, idx], y_score[:, idx]) if len(test) else None
        aps.append(ap)
        fmaxes.append(fm)
        label_rows.append(
            {
                "scheme": scheme,
                "label": rule.name,
                "test_positives": int(y_true[:, idx].sum()) if len(test) else 0,
                "nearest_neighbor_ap": ap,
                "nearest_neighbor_fmax": fm,
            }
        )

    micro_ap = average_precision(y_true.reshape(-1), y_score.reshape(-1)) if len(test) else None
    micro_fmax = fmax(y_true.reshape(-1), y_score.reshape(-1)) if len(test) else None
    summary = {
        "scheme": scheme,
        "train_count": len(train),
        "test_count": len(test),
        "exact_sequence_transfer_rate": 0.0 if not test else exact_hits / len(test),
        "sketch_transfer_rate": 0.0 if not test else sketch_hits / len(test),
        "nearest_neighbor_macro_ap": macro(aps),
        "nearest_neighbor_macro_fmax": macro(fmaxes),
        "nearest_neighbor_micro_ap": micro_ap,
        "nearest_neighbor_micro_fmax": micro_fmax,
    }
    return summary, label_rows


def overlap_summary(records: list[dict[str, Any]], scheme: str, host_split_column: str) -> dict[str, Any]:
    partitions = defaultdict(list)
    for record in records:
        partitions[split_of(record, scheme, host_split_column)].append(record)
    train = partitions.get("train", [])
    test = partitions.get("test", [])

    def values(rows: list[dict[str, Any]], field: str) -> set[str]:
        return {str(row.get(field, "")) for row in rows if str(row.get(field, ""))}

    train_families = values(train, "virus_family")
    test_families = values(test, "virus_family")
    train_hosts = values(train, "host_taxid_key")
    test_hosts = values(test, "host_taxid_key")
    train_genomes = values(train, "genome_version")
    test_genomes = values(test, "genome_version")
    train_sha = values(train, "protein_sequence_sha256")
    test_sha = values(test, "protein_sequence_sha256")
    train_sketch = values(train, "sequence_sketch_key")
    test_sketch = values(test, "sequence_sketch_key")

    partition_counts = Counter(split_of(record, scheme, host_split_column) for record in records)
    return {
        "scheme": scheme,
        "partition_counts_json": json.dumps(dict(partition_counts), ensure_ascii=False, sort_keys=True),
        "train_family_count": len(train_families),
        "test_family_count": len(test_families),
        "train_test_family_overlap": len(train_families & test_families),
        "train_host_count": len(train_hosts),
        "test_host_count": len(test_hosts),
        "train_test_host_overlap": len(train_hosts & test_hosts),
        "train_genome_count": len(train_genomes),
        "test_genome_count": len(test_genomes),
        "train_test_genome_overlap": len(train_genomes & test_genomes),
        "test_exact_sequence_overlap_count": sum(1 for row in test if str(row.get("protein_sequence_sha256", "")) in train_sha),
        "test_exact_sequence_overlap_rate": 0.0 if not test else sum(1 for row in test if str(row.get("protein_sequence_sha256", "")) in train_sha) / len(test),
        "train_test_sequence_sketch_overlap": len(train_sketch & test_sketch),
    }


def label_distribution_rows(records: list[dict[str, Any]], scheme: str, host_split_column: str) -> list[dict[str, Any]]:
    counts: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(len(LABEL_RULES), dtype=np.int64))
    partition_sizes = Counter()
    for record in records:
        split = split_of(record, scheme, host_split_column)
        counts[split] += record["labels"].astype(np.int64)
        partition_sizes[split] += 1
    rows = []
    for split, label_counts in sorted(counts.items()):
        for idx, rule in enumerate(LABEL_RULES):
            rows.append(
                {
                    "scheme": scheme,
                    "split": split,
                    "label": rule.name,
                    "positive_count": int(label_counts[idx]),
                    "partition_size": int(partition_sizes[split]),
                    "positive_rate": 0.0 if partition_sizes[split] == 0 else float(label_counts[idx]) / partition_sizes[split],
                }
            )
    return rows


def main() -> int:
    args = parse_args()
    root = repo_root()
    input_path = resolve_path(root, args.input)
    split_path = resolve_path(root, args.split_manifest)
    output_dir = resolve_path(root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    strict_rows = load_strict_rows(split_path, args.debug_limit)
    records = build_records(input_path, strict_rows, args.debug_limit)
    schemes = ["default", "family_holdout", "host_holdout"]

    overlap_rows = []
    transfer_rows = []
    label_metric_rows = []
    distribution = []
    for scheme in schemes:
        overlap_rows.append(overlap_summary(records, scheme, args.host_split_column))
        summary, label_rows = evaluate_label_transfer(records, scheme, args.host_split_column)
        transfer_rows.append(summary)
        label_metric_rows.extend(label_rows)
        distribution.extend(label_distribution_rows(records, scheme, args.host_split_column))

    write_tsv(output_dir / "split_overlap_summary.tsv", overlap_rows)
    write_tsv(output_dir / "nearest_neighbor_label_transfer.tsv", transfer_rows)
    write_tsv(output_dir / "nearest_neighbor_label_metrics.tsv", label_metric_rows)
    write_tsv(output_dir / "label_distribution_by_split.tsv", distribution)
    report = {
        "created_at": timestamp(),
        "input": str(input_path),
        "split_manifest": str(split_path),
        "output_dir": str(output_dir),
        "host_split_column": args.host_split_column,
        "debug_limit": args.debug_limit,
        "record_count": len(records),
        "schemes": schemes,
    }
    (output_dir / "split_difficulty_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "record_count": len(records), "schemes": schemes}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
