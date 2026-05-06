from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a context-dependence atlas from paired protein-only and context-aware runs.")
    parser.add_argument("--protein-run", required=True, help="Run directory for the sequence-only baseline")
    parser.add_argument("--context-run", required=True, help="Run directory for the context-aware model")
    parser.add_argument(
        "--input",
        default="data/processed/training/viral_protein_training_index.tsv.gz",
        help="Training index used to derive metadata strata",
    )
    parser.add_argument("--output-dir", default="", help="Output directory. Defaults to <context-run>/context_dependence_atlas")
    parser.add_argument("--min-stratum-size", type=int, default=50, help="Minimum proteins per stratum to report")
    return parser.parse_args()


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


def read_label_metrics(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            label_name = row.get("label", "").strip()
            if label_name:
                rows[label_name] = row
    return rows


def maybe_float(value: Any) -> float | None:
    try:
        if value in {None, ""}:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def load_predictions(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with open_text(path) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            accession = row.get("protein_accession", "").strip()
            if not accession:
                continue
            true_labels = set(json.loads(row.get("true_labels", "[]")))
            predicted_labels = set(json.loads(row.get("predicted_labels", "[]")))
            rows[accession] = {
                "true_labels": true_labels,
                "predicted_labels": predicted_labels,
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


def maybe_int(value: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def sort_key(row: dict[str, Any]) -> tuple[int, int, int, int]:
    start = int(row["cds_start"])
    end = int(row["cds_end"])
    return (
        0 if start > 0 else 1,
        start if start > 0 else 10**12,
        end if end > 0 else 10**12,
        int(row["row_order"]),
    )


def overlap_degree(rows: list[dict[str, Any]]) -> str:
    if len(rows) <= 1:
        return "none"
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
        return "unknown"
    fraction = overlaps / comparable
    if fraction == 0:
        return "none"
    if fraction < 0.2:
        return "low"
    if fraction < 0.5:
        return "medium"
    return "high"


def is_putative_enveloped(rows: list[dict[str, Any]]) -> bool:
    for row in rows:
        text = f"{row.get('cds_product', '')} {row.get('protein_description', '')}".lower()
        if "envelope" in text or "glycoprotein" in text or "matrix protein" in text:
            return True
    return False


def load_metadata(path: Path, accessions_of_interest: set[str]) -> dict[str, dict[str, str]]:
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
                    "row_order": row_idx,
                }
            )
            accessions_to_genome[accession] = key
            if accession in accessions_of_interest:
                per_accession_row[accession] = row

    genome_metadata: dict[str, dict[str, str]] = {}
    for key, rows in genome_rows.items():
        sample_row = genome_anchor_rows.get(key, {"source_segment": "", "virus_lineage": "", "source_mol_type": ""})
        genome_metadata[key] = {
            "overlap_degree": overlap_degree(rows),
            "putative_enveloped": "1" if is_putative_enveloped(rows) else "0",
            "segmented": "1" if sample_row.get("source_segment", "").strip() else "0",
        }

    metadata: dict[str, dict[str, str]] = {}
    for accession in accessions_of_interest:
        row = per_accession_row.get(accession)
        if row is None:
            continue
        key = accessions_to_genome.get(accession, "")
        genome_meta = genome_metadata.get(key, {})
        metadata[accession] = {
            "virus_family": derive_virus_family(row.get("virus_lineage", "")),
            "baltimore_like_class": derive_baltimore_like_class(row.get("virus_lineage", ""), row.get("source_mol_type", "")),
            "segmented": genome_meta.get("segmented", "0"),
            "putative_enveloped": genome_meta.get("putative_enveloped", "0"),
            "overlap_degree": genome_meta.get("overlap_degree", "unknown"),
        }
    return metadata


def f1_from_counts(tp: int, fp: int, fn: int) -> float | None:
    denominator = (2 * tp) + fp + fn
    if denominator <= 0:
        return None
    return (2 * tp) / denominator


def group_micro_f1(
    rows: list[dict[str, Any]],
    labels: tuple[str, ...],
    prediction_key: str,
) -> tuple[float | None, int]:
    tp = 0
    fp = 0
    fn = 0
    positives = 0
    label_set = set(labels)
    for row in rows:
        true_set = set(row["true_labels"]) & label_set
        pred_set = set(row[prediction_key]) & label_set
        positives += len(true_set)
        tp += len(true_set & pred_set)
        fp += len(pred_set - true_set)
        fn += len(true_set - pred_set)
    return f1_from_counts(tp, fp, fn), positives


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    root = repo_root()
    protein_run = resolve_path(root, args.protein_run)
    context_run = resolve_path(root, args.context_run)
    input_path = resolve_path(root, args.input)
    output_dir = resolve_path(root, args.output_dir) if args.output_dir else context_run / "context_dependence_atlas"
    output_dir.mkdir(parents=True, exist_ok=True)

    protein_label_metrics = read_label_metrics(protein_run / "test_label_metrics.tsv")
    context_label_metrics = read_label_metrics(context_run / "test_label_metrics.tsv")
    all_labels = sorted(set(protein_label_metrics) | set(context_label_metrics))

    label_delta_rows: list[dict[str, Any]] = []
    for label_name in all_labels:
        protein_row = protein_label_metrics.get(label_name, {})
        context_row = context_label_metrics.get(label_name, {})
        protein_ap = maybe_float(protein_row.get("average_precision"))
        context_ap = maybe_float(context_row.get("average_precision"))
        protein_f1 = maybe_float(protein_row.get("f1"))
        context_f1 = maybe_float(context_row.get("f1"))
        label_delta_rows.append(
            {
                "label": label_name,
                "label_group": next((group for group, labels in LABEL_GROUPS.items() if label_name in labels), "other"),
                "protein_average_precision": protein_ap,
                "context_average_precision": context_ap,
                "delta_average_precision": None if protein_ap is None or context_ap is None else context_ap - protein_ap,
                "protein_f1": protein_f1,
                "context_f1": context_f1,
                "delta_f1": None if protein_f1 is None or context_f1 is None else context_f1 - protein_f1,
                "protein_support": protein_row.get("support") or context_row.get("support") or "",
            }
        )
    write_tsv(output_dir / "label_deltas.tsv", label_delta_rows)

    group_rows: list[dict[str, Any]] = []
    for group_name, labels in LABEL_GROUPS.items():
        group_entries = [row for row in label_delta_rows if row["label"] in labels]
        delta_ap_values = [float(row["delta_average_precision"]) for row in group_entries if row["delta_average_precision"] is not None]
        delta_f1_values = [float(row["delta_f1"]) for row in group_entries if row["delta_f1"] is not None]
        group_rows.append(
            {
                "label_group": group_name,
                "label_count": len(group_entries),
                "mean_delta_average_precision": sum(delta_ap_values) / len(delta_ap_values) if delta_ap_values else None,
                "mean_delta_f1": sum(delta_f1_values) / len(delta_f1_values) if delta_f1_values else None,
                "positive_delta_label_count": sum(1 for value in delta_ap_values if value > 0),
            }
        )
    write_tsv(output_dir / "group_summary.tsv", group_rows)

    protein_predictions_path = protein_run / "test_predictions.tsv.gz"
    context_predictions_path = context_run / "test_predictions.tsv.gz"
    stratified_rows: list[dict[str, Any]] = []
    prediction_summary: dict[str, Any] = {"available": False}
    if protein_predictions_path.exists() and context_predictions_path.exists():
        protein_predictions = load_predictions(protein_predictions_path)
        context_predictions = load_predictions(context_predictions_path)
        shared_accessions = sorted(set(protein_predictions) & set(context_predictions))
        metadata = load_metadata(input_path, set(shared_accessions))
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

        prediction_summary = {
            "available": True,
            "shared_test_proteins": len(paired_rows),
        }

        overall_group_rows: list[dict[str, Any]] = []
        for group_name, labels in LABEL_GROUPS.items():
            protein_f1, positives = group_micro_f1(paired_rows, labels, "protein_predicted_labels")
            context_f1, _ = group_micro_f1(paired_rows, labels, "context_predicted_labels")
            overall_group_rows.append(
                {
                    "scope": "overall",
                    "stratum_field": "all",
                    "stratum_value": "all",
                    "label_group": group_name,
                    "protein_micro_f1": protein_f1,
                    "context_micro_f1": context_f1,
                    "delta_micro_f1": None if protein_f1 is None or context_f1 is None else context_f1 - protein_f1,
                    "positive_labels": positives,
                    "protein_count": len(paired_rows),
                }
            )
        stratified_rows.extend(overall_group_rows)

        for stratum_field in ("virus_family", "baltimore_like_class", "segmented", "putative_enveloped", "overlap_degree"):
            buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in paired_rows:
                buckets[str(row[stratum_field])].append(row)
            for stratum_value, bucket_rows in buckets.items():
                if len(bucket_rows) < args.min_stratum_size:
                    continue
                for group_name, labels in LABEL_GROUPS.items():
                    protein_f1, positives = group_micro_f1(bucket_rows, labels, "protein_predicted_labels")
                    context_f1, _ = group_micro_f1(bucket_rows, labels, "context_predicted_labels")
                    if protein_f1 is None and context_f1 is None:
                        continue
                    stratified_rows.append(
                        {
                            "scope": "stratified",
                            "stratum_field": stratum_field,
                            "stratum_value": stratum_value,
                            "label_group": group_name,
                            "protein_micro_f1": protein_f1,
                            "context_micro_f1": context_f1,
                            "delta_micro_f1": None if protein_f1 is None or context_f1 is None else context_f1 - protein_f1,
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
        "label_delta_count": len(label_delta_rows),
        "group_summary": group_rows,
        "prediction_summary": prediction_summary,
        "top_context_dependent_labels": sorted(
            [row for row in label_delta_rows if row["delta_average_precision"] is not None],
            key=lambda row: float(row["delta_average_precision"]),
            reverse=True,
        )[:10],
        "top_sequence_dominant_labels": sorted(
            [row for row in label_delta_rows if row["delta_average_precision"] is not None],
            key=lambda row: float(row["delta_average_precision"]),
        )[:10],
    }
    (output_dir / "atlas_report.json").write_text(json.dumps(atlas_report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "label_delta_count": len(label_delta_rows),
                "prediction_summary": prediction_summary,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
