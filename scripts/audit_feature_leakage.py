from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from build_training_index import PROTEIN_INDEX_FIELDS
from label_rules import LABEL_RULES
from task_mode_config import (
    ANNOTATION_CONTEXT_CATEGORY_FIELDS,
    ANNOTATION_CONTEXT_NUMERIC_FIELDS,
    BASE_REFINEMENT_CATEGORY_FIELDS,
    BASE_REFINEMENT_NUMERIC_FIELDS,
    NON_TEXT_CONTEXT_CATEGORY_FIELDS,
    NON_TEXT_CONTEXT_NUMERIC_FIELDS,
    TASK_MODE_ORDER,
    allowed_modes,
    feature_audit_specs,
    prior_context_numeric_fields,
)
from train_overnight_baseline import BASE_CATEGORY_FIELDS, BASE_NUMERIC_FIELD_NAMES


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit feature provenance and leakage risk across ViruFunc-FM task modes.")
    parser.add_argument(
        "--input",
        default="data/processed/training/viral_protein_training_index.tsv.gz",
        help="Training index used as the source-of-truth header",
    )
    parser.add_argument(
        "--output-dir",
        default="data/audits",
        help="Directory for leakage audit outputs",
    )
    return parser.parse_args()


def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def read_header(path: Path) -> list[str]:
    with open_text(path) as handle:
        reader = csv.reader(handle, delimiter="\t")
        return next(reader)


def current_context_fields() -> tuple[list[str], list[str]]:
    label_names = [rule.name for rule in LABEL_RULES]
    categories = ["context_prev_feature_type", "context_next_feature_type", "context_host_supergroup", "context_segment_bucket"]
    numerics = [
        "context_log_genome_protein_count",
        "context_genome_hypothetical_fraction",
        "context_genome_mat_peptide_fraction",
        "context_has_prev_neighbor",
        "context_has_next_neighbor",
        "context_same_strand_prev",
        "context_same_strand_next",
        "context_log_host_taxid_count",
        "context_log_host_lineage_count",
    ]
    numerics.extend(f"context_genome_{label_name}_fraction" for label_name in label_names)
    numerics.extend(f"context_local_{label_name}_count" for label_name in label_names)
    return categories, numerics


def build_known_fields(training_index_fields: list[str]) -> list[str]:
    current_context_categories, current_context_numerics = current_context_fields()
    all_fields = list(training_index_fields)
    all_fields.extend(spec.name for spec in feature_audit_specs() if spec.name not in training_index_fields)
    for field in current_context_categories + current_context_numerics:
        if field not in all_fields:
            all_fields.append(field)
    return all_fields


def main() -> int:
    args = parse_args()
    root = repo_root()
    input_path = (root / args.input).resolve()
    output_dir = (root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    header = read_header(input_path)
    known_fields = build_known_fields(header)
    specs = {spec.name: spec for spec in feature_audit_specs()}

    current_context_categories, current_context_numerics = current_context_fields()
    current_baseline_fields = set(BASE_CATEGORY_FIELDS) | set(BASE_NUMERIC_FIELD_NAMES) | set(current_context_categories) | set(current_context_numerics)

    rows: list[dict[str, object]] = []
    min_mode_counter: Counter[str] = Counter()
    reviewer_flags: list[dict[str, object]] = []

    for field_name in known_fields:
        spec = specs.get(field_name)
        if spec is None:
            source_table = "unknown"
            provenance_group = "unclassified"
            minimum_task_mode = "annotation_refinement"
            notes = "Field not yet classified in the provenance map"
            is_model_input_candidate = False
            is_text_derived = False
            is_train_only_stat = False
        else:
            source_table = spec.source_table
            provenance_group = spec.provenance_group
            minimum_task_mode = spec.minimum_task_mode
            notes = spec.notes
            is_model_input_candidate = spec.is_model_input_candidate
            is_text_derived = spec.is_text_derived
            is_train_only_stat = spec.is_train_only_stat

        allowed = allowed_modes(minimum_task_mode)
        used_by_current_baseline = field_name in current_baseline_fields
        row = {
            "field_name": field_name,
            "source_table": source_table,
            "present_in_training_index": field_name in header,
            "present_in_current_context_builder": field_name in current_context_categories or field_name in current_context_numerics,
            "provenance_group": provenance_group,
            "minimum_task_mode": minimum_task_mode,
            "allowed_modes_json": json.dumps(list(allowed), ensure_ascii=False),
            "is_model_input_candidate": is_model_input_candidate,
            "is_text_derived": is_text_derived,
            "is_train_only_stat": is_train_only_stat,
            "used_by_current_baseline": used_by_current_baseline,
            "used_by_proposed_protein_only": field_name == "protein_sequence",
            "used_by_proposed_genome_aware_denovo": field_name in NON_TEXT_CONTEXT_CATEGORY_FIELDS or field_name in NON_TEXT_CONTEXT_NUMERIC_FIELDS,
            "used_by_proposed_annotation_refinement": (
                field_name in BASE_REFINEMENT_CATEGORY_FIELDS
                or field_name in BASE_REFINEMENT_NUMERIC_FIELDS
                or field_name in NON_TEXT_CONTEXT_CATEGORY_FIELDS
                or field_name in NON_TEXT_CONTEXT_NUMERIC_FIELDS
                or field_name in ANNOTATION_CONTEXT_CATEGORY_FIELDS
                or field_name in ANNOTATION_CONTEXT_NUMERIC_FIELDS
                or field_name in prior_context_numeric_fields()
            ),
            "notes": notes,
        }
        rows.append(row)
        min_mode_counter[minimum_task_mode] += 1

        if used_by_current_baseline and (is_text_derived or provenance_group in {"annotation_feature_type", "knowledgebase_summary", "annotation_pipeline"}):
            reviewer_flags.append(
                {
                    "field_name": field_name,
                    "provenance_group": provenance_group,
                    "why_flagged": "Current baseline uses a text-derived or annotation-derived feature that should be gated out of de novo claims.",
                }
            )

    tsv_path = output_dir / "feature_leakage_audit.tsv"
    with tsv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    json_path = output_dir / "feature_leakage_audit.json"
    report = {
        "created_at": timestamp(),
        "input": str(input_path),
        "training_index_field_count": len(header),
        "audited_field_count": len(rows),
        "minimum_task_mode_counts": dict(min_mode_counter),
        "current_baseline_feature_count": len(current_baseline_fields),
        "current_baseline_reviewer_flags": reviewer_flags,
        "task_modes": list(TASK_MODE_ORDER),
    }
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Wrote TSV audit to {tsv_path}")
    print(f"Wrote JSON audit to {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
