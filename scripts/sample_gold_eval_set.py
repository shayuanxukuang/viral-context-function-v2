from __future__ import annotations

import argparse
import csv
import gzip
import json
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from label_rules import LABEL_RULES, UNKNOWN_TEXT_MARKERS, label_hits, normalize_text
from train_overnight_baseline import SPLIT_SCHEME_TO_COLUMN, open_text


REALM_MARKERS = ("Riboviria", "Duplodnaviria", "Monodnaviria", "Varidnaviria", "Adnaviria")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample a stratified manual-review gold benchmark candidate set.")
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
        help="Split scheme whose test partition seeds the gold set",
    )
    parser.add_argument("--partition", default="test", choices=("train", "val", "test"), help="Partition to sample from")
    parser.add_argument("--positive-per-label", type=int, default=20, help="Positive candidates per label")
    parser.add_argument("--negative-per-label", type=int, default=10, help="Hard-negative candidates per label")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed")
    parser.add_argument(
        "--output-dir",
        default="data/gold_eval",
        help="Directory for candidate tables and report",
    )
    parser.add_argument("--debug-limit", type=int, default=0, help="Optional row cap for smoke tests")
    return parser.parse_args()


def lineage_parts(lineage: str) -> list[str]:
    return [part.strip().rstrip(".") for part in lineage.split(";") if part.strip()]


def derive_broad_group(lineage: str, source_mol_type: str) -> str:
    parts = lineage_parts(lineage)
    part_set = set(parts)
    for marker in REALM_MARKERS:
        if marker in part_set:
            return marker
    if source_mol_type.strip():
        return f"mol::{source_mol_type.strip()}"
    if parts:
        return parts[0]
    return "unknown"


def is_unknown_text_marker(text: str) -> bool:
    return any(marker in text for marker in UNKNOWN_TEXT_MARKERS)


def load_split_rows(path: Path, split_column: str) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    with open_text(path) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames or split_column not in reader.fieldnames:
            raise ValueError(f"Split column '{split_column}' was not found in {path}")
        for row in reader:
            protein_accession = row.get("protein_accession", "").strip()
            if protein_accession:
                rows[protein_accession] = row
    return rows


def choose_positive_candidates(pool: list[dict[str, object]], count: int) -> list[dict[str, object]]:
    if not pool or count <= 0:
        return []

    by_stratum: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in pool:
        by_stratum[(str(row["virus_family"]), str(row["broad_group"]), str(row["length_bin"]))].append(row)

    for rows in by_stratum.values():
        rows.sort(
            key=lambda row: (
                -int(bool(row["is_sequence_novel_vs_train"])),
                -int(not bool(row["is_unknown_text"])),
                -int(bool(row["is_single_label"])),
                str(row["protein_accession"]),
            )
        )

    chosen: list[dict[str, object]] = []
    while len(chosen) < count:
        made_progress = False
        for stratum in sorted(by_stratum):
            rows = by_stratum[stratum]
            if not rows:
                continue
            chosen.append(rows.pop(0))
            made_progress = True
            if len(chosen) >= count:
                break
        if not made_progress:
            break
    return chosen


def candidate_score_for_negative(row: dict[str, object], target_broad_groups: set[str], target_length_bins: set[str]) -> tuple[int, int, int, str]:
    return (
        int(str(row["broad_group"]) in target_broad_groups),
        int(str(row["length_bin"]) in target_length_bins),
        int(bool(row["has_other_labels"])),
        str(row["protein_accession"]),
    )


def choose_negative_candidates(
    pool: list[dict[str, object]],
    count: int,
    positives: list[dict[str, object]],
) -> list[dict[str, object]]:
    if not pool or count <= 0:
        return []
    target_broad_groups = {str(row["broad_group"]) for row in positives}
    target_length_bins = {str(row["length_bin"]) for row in positives}
    ordered = sorted(pool, key=lambda row: candidate_score_for_negative(row, target_broad_groups, target_length_bins), reverse=True)
    return ordered[:count]


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)
    root = repo_root()
    input_path = (root / args.input).resolve()
    split_manifest_path = (root / args.split_manifest).resolve()
    output_dir = (root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    split_column = SPLIT_SCHEME_TO_COLUMN[args.split_scheme]
    if split_column is None:
        raise ValueError(f"Split scheme '{args.split_scheme}' is not a strict split.")
    split_rows = load_split_rows(split_manifest_path, split_column)

    label_names = [rule.name for rule in LABEL_RULES]
    train_label_sketch_keys: dict[str, set[str]] = {label_name: set() for label_name in label_names}
    candidate_pool_by_label: dict[str, dict[str, list[dict[str, object]]]] = {
        label_name: {"positive": [], "negative": []} for label_name in label_names
    }

    with open_text(input_path) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row_idx, row in enumerate(reader, start=1):
            if args.debug_limit and row_idx > args.debug_limit:
                break

            protein_accession = row.get("protein_accession", "").strip()
            split_row = split_rows.get(protein_accession)
            if split_row is None:
                continue

            text = normalize_text(row)
            hit_ids = tuple(sorted(set(label_hits(text))))
            hit_names = {label_names[idx] for idx in hit_ids}
            split_partition = split_row.get(split_column, "").strip()
            sequence_sketch_key = split_row.get("sequence_sketch_key", "").strip()
            candidate_record = {
                "protein_accession": protein_accession,
                "target_partition": split_partition,
                "virus_family": split_row.get("virus_family", "").strip() or "unknown",
                "broad_group": derive_broad_group(row.get("virus_lineage", ""), row.get("source_mol_type", "")),
                "length_bin": split_row.get("sequence_length_bin", "").strip() or "unknown",
                "sequence_sketch_key": sequence_sketch_key,
                "virus_name": row.get("virus_name", "").strip(),
                "virus_tax_id": row.get("virus_tax_id", "").strip(),
                "virus_lineage": row.get("virus_lineage", "").strip(),
                "source_mol_type": row.get("source_mol_type", "").strip(),
                "protein_length_aa": row.get("protein_length_aa", "").strip(),
                "protein_description": row.get("protein_description", "").strip(),
                "cds_product": row.get("cds_product", "").strip(),
                "weak_labels_json": json.dumps(sorted(hit_names), ensure_ascii=False),
                "is_unknown_text": is_unknown_text_marker(text),
                "is_single_label": len(hit_names) == 1,
                "has_other_labels": len(hit_names) > 0,
                "protein_sequence": row.get("protein_sequence", "").strip(),
            }

            if split_partition == "train":
                for label_name in hit_names:
                    train_label_sketch_keys[label_name].add(sequence_sketch_key)
                continue

            if split_partition != args.partition:
                continue

            for label_name in label_names:
                record = dict(candidate_record)
                record["target_label"] = label_name
                record["is_sequence_novel_vs_train"] = sequence_sketch_key not in train_label_sketch_keys[label_name]
                if label_name in hit_names:
                    candidate_pool_by_label[label_name]["positive"].append(record)
                else:
                    candidate_pool_by_label[label_name]["negative"].append(record)

    tsv_path = output_dir / f"gold_eval_candidates.{args.split_scheme}.{args.partition}.tsv"
    report_path = output_dir / f"gold_eval_candidates.{args.split_scheme}.{args.partition}.report.json"
    fieldnames = [
        "target_label",
        "candidate_role",
        "target_partition",
        "protein_accession",
        "virus_family",
        "broad_group",
        "length_bin",
        "sequence_sketch_key",
        "is_sequence_novel_vs_train",
        "virus_name",
        "virus_tax_id",
        "virus_lineage",
        "source_mol_type",
        "protein_length_aa",
        "protein_description",
        "cds_product",
        "protein_sequence",
        "weak_labels_json",
        "is_unknown_text",
        "curation_priority",
    ]

    sampled_counts: dict[str, dict[str, int]] = {}
    sampled_rows: list[dict[str, object]] = []

    for label_name in label_names:
        positives = candidate_pool_by_label[label_name]["positive"]
        negatives = candidate_pool_by_label[label_name]["negative"]
        rng.shuffle(positives)
        rng.shuffle(negatives)

        chosen_positives = choose_positive_candidates(positives, args.positive_per_label)
        chosen_negatives = choose_negative_candidates(negatives, args.negative_per_label, chosen_positives)

        sampled_counts[label_name] = {
            "positive": len(chosen_positives),
            "negative": len(chosen_negatives),
            "positive_pool": len(positives),
            "negative_pool": len(negatives),
        }

        for record in chosen_positives:
            sampled_rows.append(
                {
                    **{field: record.get(field, "") for field in fieldnames if field not in {"candidate_role", "curation_priority"}},
                    "candidate_role": "positive",
                    "curation_priority": "high" if record["is_sequence_novel_vs_train"] and not record["is_unknown_text"] else "medium",
                }
            )
        for record in chosen_negatives:
            sampled_rows.append(
                {
                    **{field: record.get(field, "") for field in fieldnames if field not in {"candidate_role", "curation_priority"}},
                    "candidate_role": "hard_negative",
                    "curation_priority": "high" if record["has_other_labels"] else "medium",
                }
            )

    with tsv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(sampled_rows)

    report = {
        "created_at": timestamp(),
        "input": str(input_path),
        "split_manifest": str(split_manifest_path),
        "split_scheme": args.split_scheme,
        "partition": args.partition,
        "positive_per_label": args.positive_per_label,
        "negative_per_label": args.negative_per_label,
        "sampled_row_count": len(sampled_rows),
        "sampled_counts": sampled_counts,
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Wrote candidate table to {tsv_path}")
    print(f"Wrote report to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
