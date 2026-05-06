#!/usr/bin/env python3
"""Build post hoc evidence tables for high-context-gain candidate cases."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True, help="S16 or figure5_high_context_gain_candidates.tsv")
    parser.add_argument("--protein-index", type=Path, default=Path("data/processed/training/viral_protein_training_index.tsv.gz"))
    parser.add_argument("--split-manifest", type=Path, default=Path("data/processed/splits/viral_protein_strict_splits.tsv.gz"))
    parser.add_argument("--module-candidates", type=Path, required=True)
    parser.add_argument("--homology-hits", type=Path, help="Optional homology_top_hit_assignments.tsv")
    parser.add_argument("--domain-hits", type=Path, help="Optional domain/hmmer/pfam hit TSV keyed by protein_accession")
    parser.add_argument("--structure-hits", type=Path, help="Optional Foldseek/structure hit TSV keyed by protein_accession")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--neighbor-window", type=int, default=5)
    return parser.parse_args()


def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("r", encoding="utf-8", newline="")


def read_tsv(path: Path) -> list[dict[str, str]]:
    if not path or not path.exists():
        return []
    with open_text(path) as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def candidate_accession(row: dict[str, str]) -> str:
    return row.get("protein_id") or row.get("protein_accession") or row.get("candidate_id") or ""


def candidate_label(row: dict[str, str]) -> str:
    return row.get("predicted_label") or row.get("candidate_label") or row.get("top_label") or ""


def as_float(value: Any, default: float = math.nan) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def index_by_accession(rows: list[dict[str, str]], keys: tuple[str, ...] = ("protein_accession", "protein_id", "query")) -> dict[str, dict[str, str]]:
    out = {}
    for row in rows:
        for key in keys:
            accession = row.get(key, "")
            if accession:
                out.setdefault(accession, row)
                break
    return out


def collect_candidate_genomes(index_path: Path, candidates: list[dict[str, str]]) -> tuple[dict[str, dict[str, str]], dict[str, list[dict[str, str]]]]:
    wanted = {candidate_accession(row) for row in candidates}
    candidate_meta: dict[str, dict[str, str]] = {}
    genomes: set[str] = {row.get("genome_id", "") or row.get("genome_version", "") for row in candidates}
    genome_rows: dict[str, list[dict[str, str]]] = {genome: [] for genome in genomes if genome}
    with open_text(index_path) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            accession = row.get("protein_accession", "")
            genome = row.get("genome_version", "")
            if accession in wanted:
                candidate_meta[accession] = row
                genomes.add(genome)
                genome_rows.setdefault(genome, [])
            if genome in genomes:
                genome_rows.setdefault(genome, []).append(row)
    return candidate_meta, genome_rows


def load_split_map(path: Path, wanted: set[str]) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    if not path.exists():
        return out
    with open_text(path) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            accession = row.get("protein_accession", "")
            if accession in wanted:
                out[accession] = row
                if len(out) == len(wanted):
                    break
    return out


def sorted_genome_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    def key(row: dict[str, str]):
        start = as_float(row.get("cds_start"), math.inf)
        end = as_float(row.get("cds_end"), math.inf)
        return (start, end, row.get("protein_accession", ""))

    return sorted(rows, key=key)


def neighborhood_for(accession: str, genome_rows: list[dict[str, str]], window: int) -> list[dict[str, Any]]:
    rows = sorted_genome_rows(genome_rows)
    idx_by_acc = {row.get("protein_accession", ""): idx for idx, row in enumerate(rows)}
    center = idx_by_acc.get(accession)
    if center is None:
        return []
    out = []
    lo = max(0, center - window)
    hi = min(len(rows), center + window + 1)
    for rank, idx in enumerate(range(lo, hi), start=lo - center):
        row = rows[idx]
        out.append(
            {
                "center_accession": accession,
                "neighbor_accession": row.get("protein_accession", ""),
                "relative_rank": idx - center,
                "genome_version": row.get("genome_version", ""),
                "cds_start": row.get("cds_start", ""),
                "cds_end": row.get("cds_end", ""),
                "cds_strand": row.get("cds_strand", ""),
                "product": row.get("cds_product", "") or row.get("protein_description", ""),
                "is_center": int(idx == center),
            }
        )
    return out


def summarize_neighbors(neighborhood: list[dict[str, Any]]) -> str:
    parts = []
    for row in neighborhood:
        prefix = "*" if row["is_center"] else ""
        product = str(row.get("product", ""))[:80]
        parts.append(f"{row['relative_rank']}:{prefix}{product}")
    return " | ".join(parts)


def main() -> None:
    args = parse_args()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates = read_tsv(args.candidates)
    wanted = {candidate_accession(row) for row in candidates}
    candidate_meta, genome_rows = collect_candidate_genomes(args.protein_index, candidates)
    split_map = load_split_map(args.split_manifest, wanted)
    module_map = index_by_accession(read_tsv(args.module_candidates), ("center_accession",))
    homology_map = index_by_accession(read_tsv(args.homology_hits), ("query", "protein_accession")) if args.homology_hits else {}
    domain_map = index_by_accession(read_tsv(args.domain_hits)) if args.domain_hits else {}
    structure_map = index_by_accession(read_tsv(args.structure_hits)) if args.structure_hits else {}

    case_rows: list[dict[str, Any]] = []
    neighborhood_rows: list[dict[str, Any]] = []
    for row in candidates:
        accession = candidate_accession(row)
        meta = candidate_meta.get(accession, {})
        split = split_map.get(accession, {})
        module = module_map.get(accession, {})
        genome = row.get("genome_id") or row.get("genome_version") or meta.get("genome_version", "")
        nhood = neighborhood_for(accession, genome_rows.get(genome, []), args.neighbor_window)
        neighborhood_rows.extend(nhood)
        homology = homology_map.get(accession, {})
        domain = domain_map.get(accession, {})
        structure = structure_map.get(accession, {})
        p_context = row.get("p_context") or row.get("top_probability_calibrated") or row.get("calibrated_probability", "")
        delta = row.get("delta_p") or row.get("context_gain", "")
        p_protein = row.get("p_protein_only") or row.get("p_protein_only_estimated", "")
        if not p_protein and p_context and delta:
            p_protein = as_float(p_context) - as_float(delta)
        case_rows.append(
            {
                "candidate_id": accession,
                "predicted_label": candidate_label(row),
                "p_protein_only": p_protein,
                "p_context": p_context,
                "delta_p": delta,
                "family": split.get("virus_family", row.get("family", "")),
                "host_group": split.get("host_supergroup", row.get("host_group", "")),
                "genome_id": genome,
                "description": row.get("description", meta.get("protein_description", "")),
                "hypothetical_or_uncharacterized": row.get("hypothetical_or_uncharacterized", row.get("hypothetical_or_unknown", "")),
                "exact_transfer_flag": row.get("exact_transfer_flag", ""),
                "module_cluster_id": module.get("cluster_id", row.get("module_cluster_id", "")),
                "module_neighborhood_signature": module.get("neighborhood_signature", ""),
                "module_weak_label_counts": module.get("weak_label_counts_json", ""),
                "tm_helix_count": module.get("bio_tm_helix_count", ""),
                "signal_peptide_score": module.get("bio_signal_peptide_score", ""),
                "disorder_score": module.get("bio_disorder_score", ""),
                "nearest_homolog_accession": homology.get("target", ""),
                "nearest_homolog_identity": homology.get("pident", row.get("nearest_train_identity", "")),
                "domain_or_remote_hit": json.dumps(domain, ensure_ascii=False) if domain else "",
                "structure_hit": json.dumps(structure, ensure_ascii=False) if structure else "",
                "neighbor_products_window": summarize_neighbors(nhood),
                "not_annotation_leakage_note": "de novo model excludes product text, database hits, neighbor labels, and annotation-derived priors",
                "caveat": "computational candidate; independent validation required",
            }
        )
    write_tsv(out_dir / "candidate_case_evidence.tsv", case_rows)
    write_tsv(out_dir / "candidate_case_neighborhoods.tsv", neighborhood_rows)
    report = {
        "candidate_case_evidence": str(out_dir / "candidate_case_evidence.tsv"),
        "candidate_case_neighborhoods": str(out_dir / "candidate_case_neighborhoods.tsv"),
        "candidate_count": len(case_rows),
        "neighborhood_row_count": len(neighborhood_rows),
    }
    (out_dir / "candidate_case_evidence_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
