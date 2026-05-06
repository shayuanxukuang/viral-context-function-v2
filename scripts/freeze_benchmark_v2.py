from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from build_strict_splits import derive_virus_family
from label_rules import LABEL_RULES, label_hits, normalize_text
from task_mode_config import feature_audit_specs
from train_overnight_baseline import assign_split


PARTITION_NAMES = {0: "train", 1: "val", 2: "test"}

LABEL_GROUPS = {
    "polymerase": "replication_transcription",
    "helicase": "replication_transcription",
    "protease": "processing_enzyme",
    "capsid_head": "structural_assembly",
    "tail_fiber_receptor": "host_interface",
    "tail_assembly": "structural_assembly",
    "portal_terminase_packaging": "genome_packaging",
    "lysis": "lysis_integration",
    "envelope_glycoprotein": "membrane_entry",
    "membrane_matrix": "membrane_entry",
    "nucleocapsid": "structural_assembly",
    "integrase_recombinase": "lysis_integration",
    "nuclease": "processing_enzyme",
    "methyltransferase": "processing_enzyme",
    "ligase": "processing_enzyme",
    "transcription_regulator": "replication_transcription",
    "polyprotein": "processing_enzyme",
}


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze the V2 benchmark data package and checksums.")
    parser.add_argument("--protein-index", default="data/processed/training/viral_protein_training_index.tsv.gz")
    parser.add_argument("--genome-index", default="data/processed/training/viral_genome_training_index.tsv.gz")
    parser.add_argument("--strict-splits", default="data/processed/splits/viral_protein_strict_splits.tsv.gz")
    parser.add_argument("--taxonomy", default="data/processed/taxonomy/observed_taxonomy.tsv.gz")
    parser.add_argument("--output-dir", default="data/v2_freeze")
    parser.add_argument(
        "--host-split-column",
        default="host_taxid_holdout_split",
        choices=("host_taxid_holdout_split", "host_supergroup_holdout_split"),
        help="Column to export as splits/host_holdout_split.tsv.",
    )
    parser.add_argument("--calibration-fraction", type=float, default=0.1)
    parser.add_argument("--min-label-count", type=int, default=500)
    parser.add_argument("--debug-limit", type=int, default=0)
    return parser.parse_args()


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    return (root / path).resolve() if not path.is_absolute() else path.resolve()


def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def write_tsv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            count += 1
    return count


def stable_fraction(key: str) -> float:
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / float(0xFFFFFFFF)


def sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def load_strict_splits(path: Path, debug_limit: int) -> dict[str, dict[str, str]]:
    splits: dict[str, dict[str, str]] = {}
    with open_text(path) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for idx, row in enumerate(reader, start=1):
            if debug_limit and idx > debug_limit:
                break
            accession = row.get("protein_accession", "").strip()
            if accession:
                splits[accession] = dict(row)
    return splits


def iter_rows(path: Path, debug_limit: int):
    with open_text(path) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for idx, row in enumerate(reader, start=1):
            if debug_limit and idx > debug_limit:
                break
            yield row


def split_writer_rows(
    strict_rows: dict[str, dict[str, str]],
    column: str,
) -> Iterable[dict[str, str]]:
    for accession, row in sorted(strict_rows.items()):
        yield {
            "protein_accession": accession,
            "genome_version": row.get("genome_version", ""),
            "virus_tax_id": row.get("virus_tax_id", ""),
            "virus_name": row.get("virus_name", ""),
            "virus_family": row.get("virus_family", ""),
            "host_taxid_key": row.get("host_taxid_key", ""),
            "host_supergroup": row.get("host_supergroup", ""),
            "sequence_sketch_key": row.get("sequence_sketch_key", ""),
            "split": row.get(column, ""),
        }


def write_checksums(output_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file() or path.name == "checksums.tsv":
            continue
        digest, size = sha256_file(path)
        rows.append(
            {
                "relative_path": str(path.relative_to(output_dir)).replace("\\", "/"),
                "sha256": digest,
                "bytes": size,
            }
        )
    write_tsv(output_dir / "checksums.tsv", ["relative_path", "sha256", "bytes"], rows)
    return rows


def main() -> int:
    args = parse_args()
    if not 0.0 < args.calibration_fraction < 1.0:
        raise ValueError("--calibration-fraction must be between 0 and 1")

    root = repo_root()
    protein_index = resolve_path(root, args.protein_index)
    genome_index = resolve_path(root, args.genome_index)
    strict_splits = resolve_path(root, args.strict_splits)
    taxonomy_path = resolve_path(root, args.taxonomy)
    output_dir = resolve_path(root, args.output_dir)
    split_dir = output_dir / "splits"
    output_dir.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)

    strict_rows = load_strict_splits(strict_splits, args.debug_limit)
    family_split_by_accession = {
        accession: row.get("family_holdout_split", "") for accession, row in strict_rows.items()
    }
    host_split_by_accession = {
        accession: row.get(args.host_split_column, "") for accession, row in strict_rows.items()
    }

    protein_fields = [
        "protein_accession",
        "protein_length_aa",
        "protein_sequence_sha256",
        "protein_sequence",
        "protein_description",
        "protein_organism",
        "genome_accession",
        "genome_version",
        "virus_tax_id",
        "virus_name",
        "virus_family",
        "protein_feature_type",
    ]
    coordinate_fields = [
        "protein_accession",
        "genome_accession",
        "genome_version",
        "source_segment",
        "protein_feature_type",
        "cds_gene",
        "cds_locus_tag",
        "cds_product",
        "cds_location_raw",
        "cds_location_kind",
        "cds_start",
        "cds_end",
        "cds_strand",
        "cds_part_count",
        "cds_partial_left",
        "cds_partial_right",
    ]
    default_split_rows: list[dict[str, str]] = []
    calibration_rows: list[dict[str, str]] = []
    calibration_split_by_accession: dict[str, str] = {}
    default_split_by_accession: dict[str, str] = {}
    family_counter: Counter[str] = Counter()
    host_counter: Counter[str] = Counter()
    protein_count = 0

    with (output_dir / "proteins.tsv").open("w", encoding="utf-8", newline="") as protein_handle, (
        output_dir / "genome_coordinates.tsv"
    ).open("w", encoding="utf-8", newline="") as coord_handle:
        protein_writer = csv.DictWriter(protein_handle, fieldnames=protein_fields, delimiter="\t", extrasaction="ignore")
        coord_writer = csv.DictWriter(coord_handle, fieldnames=coordinate_fields, delimiter="\t", extrasaction="ignore")
        protein_writer.writeheader()
        coord_writer.writeheader()
        for row in iter_rows(protein_index, args.debug_limit):
            accession = row.get("protein_accession", "").strip()
            virus_family, _source = derive_virus_family(row.get("virus_lineage", ""))
            protein_row = dict(row)
            protein_row["virus_family"] = virus_family
            protein_writer.writerow(protein_row)
            coord_writer.writerow(row)

            default_split = PARTITION_NAMES[assign_split(row)]
            default_split_by_accession[accession] = default_split
            default_split_rows.append(
                {
                    "protein_accession": accession,
                    "genome_version": row.get("genome_version", ""),
                    "virus_tax_id": row.get("virus_tax_id", ""),
                    "split": default_split,
                }
            )

            family_split = family_split_by_accession.get(accession, "")
            if family_split == "train":
                calibration_split = (
                    "calibration"
                    if stable_fraction(f"calibration|{row.get('genome_version', accession)}") < args.calibration_fraction
                    else "train"
                )
            elif family_split == "val":
                calibration_split = "val"
            elif family_split == "test":
                calibration_split = "test"
            else:
                calibration_split = ""
            calibration_split_by_accession[accession] = calibration_split
            calibration_rows.append(
                {
                    "protein_accession": accession,
                    "genome_version": row.get("genome_version", ""),
                    "virus_tax_id": row.get("virus_tax_id", ""),
                    "virus_family": virus_family,
                    "split": calibration_split,
                }
            )

            family_counter[virus_family] += 1
            host_supergroup = strict_rows.get(accession, {}).get("host_supergroup", "")
            host_counter[host_supergroup or "unknown"] += 1
            protein_count += 1

    write_tsv(
        split_dir / "default_split.tsv",
        ["protein_accession", "genome_version", "virus_tax_id", "split"],
        default_split_rows,
    )
    split_fields = [
        "protein_accession",
        "genome_version",
        "virus_tax_id",
        "virus_name",
        "virus_family",
        "host_taxid_key",
        "host_supergroup",
        "sequence_sketch_key",
        "split",
    ]
    write_tsv(split_dir / "family_holdout_split.tsv", split_fields, split_writer_rows(strict_rows, "family_holdout_split"))
    write_tsv(split_dir / "host_holdout_split.tsv", split_fields, split_writer_rows(strict_rows, args.host_split_column))
    write_tsv(
        split_dir / "calibration_split.tsv",
        ["protein_accession", "genome_version", "virus_tax_id", "virus_family", "split"],
        calibration_rows,
    )

    genome_fields = [
        "genome_accession",
        "genome_version",
        "virus_tax_id",
        "virus_name",
        "virus_lineage",
        "genome_length_nt",
        "molecule_type",
        "topology",
        "division",
        "source_mol_type",
        "source_segment",
        "protein_count",
    ]
    host_fields = [
        "genome_accession",
        "genome_version",
        "virus_tax_id",
        "virus_name",
        "host_join_strategy",
        "host_record_count",
        "host_tax_ids_json",
        "host_names_json",
        "host_lineages_json",
        "host_evidence_json",
        "host_pmids_json",
        "host_sample_types_json",
        "host_source_organisms_json",
    ]
    genome_count = write_tsv(output_dir / "genomes.tsv", genome_fields, iter_rows(genome_index, args.debug_limit))
    write_tsv(output_dir / "host_metadata.tsv", host_fields, iter_rows(genome_index, args.debug_limit))

    with (output_dir / "taxonomy.tsv").open("w", encoding="utf-8", newline="") as out_handle:
        with open_text(taxonomy_path) as in_handle:
            for idx, line in enumerate(in_handle):
                if args.debug_limit and idx > args.debug_limit:
                    break
                out_handle.write(line)

    label_names = [rule.name for rule in LABEL_RULES]
    label_counts = Counter({name: 0 for name in label_names})
    split_label_counts: dict[str, dict[str, Counter[str]]] = {
        "default": defaultdict(Counter),
        "family_holdout": defaultdict(Counter),
        "host_holdout": defaultdict(Counter),
        "calibration": defaultdict(Counter),
    }
    label_fields = ["protein_accession", *label_names, "positive_label_count"]
    with (output_dir / "labels.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=label_fields, delimiter="\t")
        writer.writeheader()
        for row in iter_rows(protein_index, args.debug_limit):
            accession = row.get("protein_accession", "").strip()
            hits = set(label_hits(normalize_text(row)))
            output_row: dict[str, Any] = {"protein_accession": accession}
            for idx, label_name in enumerate(label_names):
                value = 1 if idx in hits else 0
                output_row[label_name] = value
                if value:
                    label_counts[label_name] += 1
                    split_label_counts["default"][default_split_by_accession.get(accession, "")][label_name] += 1
                    split_label_counts["family_holdout"][family_split_by_accession.get(accession, "")][label_name] += 1
                    split_label_counts["host_holdout"][host_split_by_accession.get(accession, "")][label_name] += 1
                    split_label_counts["calibration"][calibration_split_by_accession.get(accession, "")][label_name] += 1
            output_row["positive_label_count"] = len(hits)
            writer.writerow(output_row)

    label_manifest_rows = []
    for rule in LABEL_RULES:
        row: dict[str, Any] = {
            "label": rule.name,
            "functional_group": LABEL_GROUPS.get(rule.name, "other_accessory"),
            "description": rule.description,
            "positive_count": label_counts[rule.name],
            "kept_primary": int(label_counts[rule.name] >= args.min_label_count),
            "patterns_json": json.dumps(rule.patterns, ensure_ascii=False),
        }
        for scheme_name, scheme_counts in split_label_counts.items():
            for partition in ("train", "calibration", "val", "test"):
                row[f"{scheme_name}_{partition}_positives"] = scheme_counts.get(partition, Counter()).get(rule.name, 0)
        label_manifest_rows.append(row)

    label_manifest_fields = [
        "label",
        "functional_group",
        "description",
        "positive_count",
        "kept_primary",
        "patterns_json",
        *[
            f"{scheme}_{partition}_positives"
            for scheme in ("default", "family_holdout", "host_holdout", "calibration")
            for partition in ("train", "calibration", "val", "test")
        ],
    ]
    write_tsv(output_dir / "label_manifest.tsv", label_manifest_fields, label_manifest_rows)

    feature_rows = [
        {
            "name": spec.name,
            "source_table": spec.source_table,
            "provenance_group": spec.provenance_group,
            "minimum_task_mode": spec.minimum_task_mode,
            "is_model_input_candidate": int(spec.is_model_input_candidate),
            "is_text_derived": int(spec.is_text_derived),
            "is_train_only_stat": int(spec.is_train_only_stat),
            "notes": spec.notes,
        }
        for spec in feature_audit_specs()
    ]
    write_tsv(
        output_dir / "feature_manifest.tsv",
        [
            "name",
            "source_table",
            "provenance_group",
            "minimum_task_mode",
            "is_model_input_candidate",
            "is_text_derived",
            "is_train_only_stat",
            "notes",
        ],
        feature_rows,
    )

    report = {
        "created_at": timestamp(),
        "output_dir": str(output_dir),
        "inputs": {
            "protein_index": str(protein_index),
            "genome_index": str(genome_index),
            "strict_splits": str(strict_splits),
            "taxonomy": str(taxonomy_path),
        },
        "host_split_column": args.host_split_column,
        "calibration_fraction": args.calibration_fraction,
        "debug_limit": args.debug_limit,
        "protein_count": protein_count,
        "genome_count": genome_count,
        "family_count": len(family_counter),
        "host_group_count": len(host_counter),
        "label_count": len(label_names),
        "primary_label_count": sum(1 for count in label_counts.values() if count >= args.min_label_count),
        "low_frequency_label_count": sum(1 for count in label_counts.values() if count < args.min_label_count),
        "label_positive_counts": dict(label_counts),
        "top_families": family_counter.most_common(25),
        "top_host_groups": host_counter.most_common(25),
        "split_label_counts": {
            scheme: {partition: dict(counter) for partition, counter in partitions.items()}
            for scheme, partitions in split_label_counts.items()
        },
    }
    (output_dir / "freeze_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    checksum_rows = write_checksums(output_dir)
    summary = {
        "output_dir": str(output_dir),
        "protein_count": protein_count,
        "genome_count": genome_count,
        "label_count": len(label_names),
        "checksummed_files": len(checksum_rows),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
