from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path
from typing import Any


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare representative FASTA and metadata for targeted structure validation.")
    parser.add_argument("--ranked-clusters", required=True)
    parser.add_argument("--module-candidates", required=True)
    parser.add_argument("--input", default="data/processed/training/viral_protein_training_index.tsv.gz")
    parser.add_argument("--output-dir", default="runs/targeted_structure_validation")
    parser.add_argument("--top-clusters", type=int, default=5)
    parser.add_argument("--representatives-per-cluster", type=int, default=3)
    return parser.parse_args()


def read_tsv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def main() -> int:
    args = parse_args()
    root = repo_root()
    ranked_clusters_path = (root / args.ranked_clusters).resolve() if not Path(args.ranked_clusters).is_absolute() else Path(args.ranked_clusters).resolve()
    module_candidates_path = (root / args.module_candidates).resolve() if not Path(args.module_candidates).is_absolute() else Path(args.module_candidates).resolve()
    input_path = (root / args.input).resolve()
    output_dir = (root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    ranked_rows = read_tsv(ranked_clusters_path)[: args.top_clusters]
    cluster_ids = {str(row["cluster_id"]) for row in ranked_rows}
    module_rows = [row for row in read_tsv(module_candidates_path) if str(row.get("cluster_id", "")) in cluster_ids]
    accessions = {str(row["center_accession"]) for row in module_rows}

    accession_to_sequence: dict[str, str] = {}
    accession_to_description: dict[str, str] = {}
    with open_text(input_path) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            accession = str(row.get("protein_accession", "") or "").strip()
            if accession not in accessions:
                continue
            accession_to_sequence[accession] = str(row.get("protein_sequence", "") or "").strip()
            accession_to_description[accession] = str(row.get("protein_description", "") or row.get("cds_product", "") or "").strip()

    fasta_lines: list[str] = []
    metadata_rows: list[dict[str, Any]] = []
    for cluster_row in ranked_rows:
        cluster_id = str(cluster_row["cluster_id"])
        members = [row for row in module_rows if str(row["cluster_id"]) == cluster_id][: args.representatives_per_cluster]
        for member in members:
            accession = str(member["center_accession"])
            sequence = accession_to_sequence.get(accession, "")
            fasta_lines.append(f">{accession} cluster={cluster_id} family={member['virus_family']}")
            fasta_lines.append(sequence)
            metadata_rows.append(
                {
                    "cluster_id": cluster_id,
                    "protein_accession": accession,
                    "virus_family": member["virus_family"],
                    "description": accession_to_description.get(accession, ""),
                    "hypothetical_ratio": member["hypothetical_ratio"],
                    "neighborhood_signature": member["neighborhood_signature"],
                    "structural_membrane_vote_fraction": member["structural_membrane_vote_fraction"],
                }
            )

    (output_dir / "representatives.fasta").write_text("\n".join(fasta_lines), encoding="utf-8")
    with (output_dir / "representatives.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metadata_rows[0].keys()) if metadata_rows else [], delimiter="\t")
        writer.writeheader()
        writer.writerows(metadata_rows)

    report = {
        "ranked_clusters": str(ranked_clusters_path),
        "module_candidates": str(module_candidates_path),
        "representative_count": len(metadata_rows),
        "output_dir": str(output_dir),
    }
    (output_dir / "structure_validation_prep.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
