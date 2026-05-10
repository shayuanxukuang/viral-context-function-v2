#!/usr/bin/env python3
"""Prepare temporal/blind-curation validation inputs for V2 candidates.

The script creates randomized blind IDs, FASTA queries, and a separate key that
maps blind IDs back to high-context candidates or matched controls. It does not
score candidates as validated; the intended labels are evidence grades for
downstream manual or temporal-database review.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import random
from pathlib import Path
from typing import Any, Iterable


def open_text(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("r", encoding="utf-8", newline="")


def read_tsv(path: Path) -> list[dict[str, str]]:
    with open_text(path) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames:
            raise SystemExit(f"No TSV header found: {path}")
        return [dict(row) for row in reader]


def write_tsv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def first_present(row: dict[str, Any], names: Iterable[str]) -> str:
    for name in names:
        if name in row and str(row.get(name, "")).strip():
            return str(row.get(name, "")).strip()
    return ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path, required=True, help="Validation targets/regime table containing candidate/control rows.")
    parser.add_argument("--protein-index", type=Path, default=Path("data/processed/training/viral_protein_training_index.tsv.gz"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-product-text", action="store_true", help="Include product/description in the curator sheet.")
    parser.add_argument("--max-per-group", type=int, default=0, help="Optional cap per target_group.")
    return parser.parse_args()


def load_sequences(path: Path) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    with open_text(path) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"protein_accession", "protein_sequence"}
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise SystemExit(f"Protein index missing columns: {', '.join(missing)}")
        for row in reader:
            accession = row.get("protein_accession", "").strip()
            if accession:
                out[accession] = {
                    "protein_sequence": row.get("protein_sequence", "").strip(),
                    "protein_length_aa": row.get("protein_length_aa", "").strip(),
                    "protein_description": first_present(row, ["protein_description", "cds_product", "description"]),
                    "genome_version": row.get("genome_version", "").strip(),
                }
    return out


def target_group(row: dict[str, Any]) -> str:
    value = first_present(row, ["target_group", "panel_source", "group", "evidence_group"])
    if value:
        return value
    if str(row.get("high_context_gain", "")).strip().lower() in {"1", "true", "yes"}:
        return "high_context_candidate"
    return "unspecified"


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_tsv(args.targets)
    seqs = load_sequences(args.protein_index)
    if args.max_per_group > 0:
        grouped: dict[str, list[dict[str, str]]] = {}
        for row in rows:
            grouped.setdefault(target_group(row), []).append(row)
        capped = []
        for group, group_rows in grouped.items():
            rng.shuffle(group_rows)
            capped.extend(group_rows[: args.max_per_group])
        rows = capped

    rng.shuffle(rows)
    curator_rows: list[dict[str, Any]] = []
    key_rows: list[dict[str, Any]] = []
    fasta_records: list[tuple[str, str]] = []
    missing: list[dict[str, Any]] = []

    for idx, row in enumerate(rows, start=1):
        accession = first_present(row, ["protein_accession", "query", "accession"])
        if not accession:
            continue
        meta = seqs.get(accession)
        if not meta or not meta.get("protein_sequence"):
            missing.append({"protein_accession": accession, "reason": "missing_sequence"})
            continue
        blind_id = f"VF2-BLIND-{idx:05d}"
        label = first_present(row, ["candidate_label", "predicted_label", "top_label", "query_labels"])
        curator = {
            "blind_id": blind_id,
            "sequence_length_aa": meta.get("protein_length_aa", ""),
            "candidate_label_for_review": label,
            "reviewer_evidence_grade": "",
            "reviewer_primary_evidence": "",
            "reviewer_notes": "",
            "allowed_grades": "A=strong independent support; B=moderate support; C=ambiguous; D=no support; E=contradictory",
        }
        if args.include_product_text:
            curator["product_or_description_for_review"] = meta.get("protein_description", "")
        curator_rows.append(curator)
        key_rows.append(
            {
                "blind_id": blind_id,
                "protein_accession": accession,
                "genome_version": first_present(row, ["genome_version"]) or meta.get("genome_version", ""),
                "target_group": target_group(row),
                "candidate_label": label,
                "context_gain": first_present(row, ["context_gain", "delta_p", "context_gain_or_delta_p"]),
                "top_probability": first_present(row, ["top_probability_calibrated", "top_probability", "p_context"]),
                "source_row": json.dumps(row, ensure_ascii=False),
            }
        )
        fasta_records.append((blind_id, meta["protein_sequence"]))

    write_tsv(out_dir / "blind_curation_sheet.tsv", curator_rows)
    write_tsv(out_dir / "blind_curation_key_private.tsv", key_rows)
    write_tsv(out_dir / "blind_curation_missing_sequences.tsv", missing)
    with (out_dir / "temporal_validation_queries.fasta").open("w", encoding="utf-8", newline="") as handle:
        for blind_id, sequence in fasta_records:
            handle.write(f">{blind_id}\n")
            for pos in range(0, len(sequence), 80):
                handle.write(sequence[pos : pos + 80] + "\n")

    rubric = {
        "interpretation": "Manual/temporal review grades support prioritization only; they are not experimental validation.",
        "recommended_external_sources": [
            "NCBI/RefSeq records added after the training freeze",
            "UniProt/InterPro/Pfam/HMMER annotations added after the training freeze",
            "PHROG/Phold or other viral structure-aware annotations",
            "HHpred/Phyre-style remote homology where available",
            "primary literature only when experimental evidence is explicit",
        ],
        "blind_sheet": str(out_dir / "blind_curation_sheet.tsv"),
        "private_key": str(out_dir / "blind_curation_key_private.tsv"),
        "fasta": str(out_dir / "temporal_validation_queries.fasta"),
        "missing_sequence_count": len(missing),
    }
    (out_dir / "independent_validation_panel_report.json").write_text(json.dumps(rubric, indent=2), encoding="utf-8")
    print(json.dumps(rubric, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
