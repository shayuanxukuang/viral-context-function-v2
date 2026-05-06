#!/usr/bin/env python3
"""Run target-specific MMseqs2 homology search for V2 validation targets."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Iterable

from label_rules import LABEL_RULES, label_hits, normalize_text


DEFAULT_TARGETS = Path("runs/v2_sequence_structure_validation/targets/validation_targets.tsv")
DEFAULT_FASTA = Path("runs/v2_sequence_structure_validation/targets/all_validation_targets.fasta")
DEFAULT_INDEX = Path("data/processed/training/viral_protein_training_index.tsv.gz")
DEFAULT_SPLITS = Path("data/processed/splits/viral_protein_strict_splits.tsv.gz")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--target-fasta", type=Path, default=DEFAULT_FASTA)
    parser.add_argument("--protein-index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--reference-split-column", default="family_holdout_split")
    parser.add_argument("--reference-split", default="train")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/v2_sequence_structure_validation/homology"))
    parser.add_argument("--mmseqs-bin", default="mmseqs")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--sensitivity", default="7.5")
    parser.add_argument("--reuse-hits", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_path(path: Path, root: Path) -> Path:
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("r", encoding="utf-8", newline="")


def read_tsv(path: Path, required: bool = True) -> list[dict[str, str]]:
    if not path.exists():
        if required:
            raise SystemExit(f"Required TSV not found: {path}")
        return []
    with open_text(path) as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    rows = list(rows)
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


def resolve_executable(value: str) -> str | None:
    if any(sep in value for sep in ("/", "\\")):
        path = Path(value).expanduser()
        return str(path) if path.exists() else None
    return shutil.which(value)


def fasta_write(handle, accession: str, sequence: str) -> None:
    handle.write(f">{accession}\n")
    for i in range(0, len(sequence), 80):
        handle.write(sequence[i : i + 80] + "\n")


def labels_for_row(row: dict[str, str]) -> list[str]:
    return [LABEL_RULES[idx].name for idx in label_hits(normalize_text(row))]


def load_target_accessions(targets_path: Path) -> set[str]:
    rows = read_tsv(targets_path)
    accessions = {row.get("protein_accession", "") for row in rows if row.get("protein_accession", "")}
    if not accessions:
        raise SystemExit(f"No protein_accession values found in {targets_path}")
    return accessions


def load_split_map(path: Path, column: str) -> dict[str, str]:
    if not path.exists():
        raise SystemExit(f"Split manifest not found: {path}")
    out: dict[str, str] = {}
    with open_text(path) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fields = set(reader.fieldnames or [])
        missing = {"protein_accession", column} - fields
        if missing:
            raise SystemExit(f"Split manifest missing columns {sorted(missing)}: {path}")
        for row in reader:
            accession = row.get("protein_accession", "")
            if accession:
                out[accession] = row.get(column, "")
    return out


def build_reference_fasta(
    protein_index: Path,
    split_map: dict[str, str],
    target_accessions: set[str],
    reference_split: str,
    out_fasta: Path,
) -> dict[str, list[str]]:
    label_map: dict[str, list[str]] = {}
    out_fasta.parent.mkdir(parents=True, exist_ok=True)
    with out_fasta.open("w", encoding="utf-8") as handle, open_text(protein_index) as source:
        reader = csv.DictReader(source, delimiter="\t")
        fields = set(reader.fieldnames or [])
        missing = {"protein_accession", "protein_sequence"} - fields
        if missing:
            raise SystemExit(f"Protein index missing required columns {sorted(missing)}: {protein_index}")
        for row in reader:
            accession = row.get("protein_accession", "")
            sequence = row.get("protein_sequence", "")
            if not accession or not sequence:
                continue
            label_map[accession] = labels_for_row(row)
            if accession in target_accessions:
                continue
            if reference_split.lower() != "any" and split_map.get(accession) != reference_split:
                continue
            fasta_write(handle, accession, sequence)
    return label_map


def parse_target_fasta_labels(path: Path) -> list[str]:
    accessions: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith(">"):
                accessions.append(line[1:].strip().split("|")[0].split()[0])
    return accessions


def run_mmseqs(args: argparse.Namespace, target_fasta: Path, reference_fasta: Path, output_dir: Path) -> Path:
    hits = output_dir / "target_validation.mmseqs_hits.tsv"
    if hits.exists() and args.reuse_hits:
        return hits
    if args.dry_run:
        print("[dry-run]", args.mmseqs_bin, "easy-search", target_fasta, reference_fasta, hits)
        return hits
    mmseqs = resolve_executable(args.mmseqs_bin)
    if not mmseqs:
        raise SystemExit(
            f"MMseqs2 executable not found: {args.mmseqs_bin}. Install mmseqs2, pass --mmseqs-bin, or rerun the one-click wrapper with --skip-homology."
        )
    tmp = output_dir / "mmseqs_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    cmd = [
        mmseqs,
        "easy-search",
        str(target_fasta),
        str(reference_fasta),
        str(hits),
        str(tmp),
        "--format-output",
        "query,target,pident,evalue,bits,qcov,tcov",
        "--threads",
        str(args.threads),
        "-s",
        str(args.sensitivity),
    ]
    print("[cmd]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    return hits


def parse_best_hits(path: Path) -> dict[str, dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return best
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for parts in reader:
            if len(parts) < 5:
                continue
            query, target, pident, evalue, bits = parts[:5]
            qcov = parts[5] if len(parts) > 5 else ""
            tcov = parts[6] if len(parts) > 6 else ""
            try:
                row = {
                    "query": query,
                    "target": target,
                    "pident": float(pident),
                    "evalue": evalue,
                    "bits": float(bits),
                    "qcov": qcov,
                    "tcov": tcov,
                }
            except ValueError:
                continue
            old = best.get(query)
            if old is None or (row["bits"], row["pident"]) > (old["bits"], old["pident"]):
                best[query] = row
    return best


def main() -> int:
    args = parse_args()
    root = repo_root()
    targets_path = resolve_path(args.targets, root)
    target_fasta = resolve_path(args.target_fasta, root)
    protein_index = resolve_path(args.protein_index, root)
    split_manifest = resolve_path(args.split_manifest, root)
    output_dir = resolve_path(args.output_dir, root)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not target_fasta.exists():
        raise SystemExit(f"Target FASTA not found: {target_fasta}")

    target_accessions = load_target_accessions(targets_path)
    split_map = load_split_map(split_manifest, args.reference_split_column)
    reference_fasta = output_dir / "reference_train.faa"
    label_map = build_reference_fasta(protein_index, split_map, target_accessions, args.reference_split, reference_fasta)
    hits_path = run_mmseqs(args, target_fasta, reference_fasta, output_dir)
    best_hits = parse_best_hits(hits_path)

    target_order = parse_target_fasta_labels(target_fasta)
    rows: list[dict[str, Any]] = []
    for query in target_order:
        hit = best_hits.get(query)
        if not hit:
            rows.append(
                {
                    "scheme": "target_validation",
                    "subset": "all_targets",
                    "query": query,
                    "target": "",
                    "pident": "",
                    "evalue": "",
                    "bits": "",
                    "qcov": "",
                    "tcov": "",
                    "query_labels": "[]",
                    "target_labels": "[]",
                }
            )
            continue
        rows.append(
            {
                "scheme": "target_validation",
                "subset": "all_targets",
                "query": query,
                "target": hit["target"],
                "pident": hit["pident"],
                "evalue": hit["evalue"],
                "bits": hit["bits"],
                "qcov": hit.get("qcov", ""),
                "tcov": hit.get("tcov", ""),
                "query_labels": json.dumps(label_map.get(query, [])),
                "target_labels": json.dumps(label_map.get(hit["target"], [])),
            }
        )
    assignments = output_dir / "target_homology_top_hit_assignments.tsv"
    write_tsv(
        assignments,
        rows,
        ["scheme", "subset", "query", "target", "pident", "evalue", "bits", "qcov", "tcov", "query_labels", "target_labels"],
    )
    report = {
        "target_count": len(target_order),
        "targets_with_hits": sum(1 for row in rows if row.get("target")),
        "reference_fasta": str(reference_fasta),
        "raw_hits": str(hits_path),
        "assignments": str(assignments),
        "reference_split_column": args.reference_split_column,
        "reference_split": args.reference_split,
    }
    (output_dir / "target_homology_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
