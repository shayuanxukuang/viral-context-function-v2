#!/usr/bin/env python3
"""Prepare sequence-structure validation targets for ViruFunc V2 candidates.

This script extracts the high-context-gain candidate set, chooses matched
low-context controls when a candidate universe is available, streams amino-acid
sequences from the training index, and writes FASTA files plus command templates
for ESMFold, ColabFold, and Foldseek.

The outputs are preparation artifacts only. Product text, homology hits,
Foldseek hits, and neighbor annotations are not used as de novo model inputs.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Iterable

from label_rules import LABEL_RULES, label_hits, normalize_text


DEFAULT_CANDIDATES = Path(
    "artifacts/return/v2_plos_cb_supplementary_package_20260504/"
    "supplementary_tables/S16_high_context_gain_candidates.tsv"
)
DEFAULT_UNIVERSE = Path("artifacts/return/context_study_v2_review_completion_20260504/qc_review/qc7_candidate_assignments.tsv")
DEFAULT_INDEX = Path("data/processed/training/viral_protein_training_index.tsv.gz")
DEFAULT_SPLITS = Path("data/processed/splits/viral_protein_strict_splits.tsv.gz")

ACCESSION_COLUMNS = ("protein_id", "protein_accession", "candidate_id", "query")
LABEL_COLUMNS = ("predicted_label", "candidate_label", "top_label")
GENOME_COLUMNS = ("genome_id", "genome_version", "genome_accession")
CONTEXT_PROB_COLUMNS = ("p_context", "top_probability_calibrated", "calibrated_probability")
PROTEIN_PROB_COLUMNS = ("p_protein_only", "p_protein_only_estimated")
CONTEXT_GAIN_COLUMNS = ("delta_p", "context_gain")

SAFE_AA_RE = re.compile(r"[^ACDEFGHIKLMNPQRSTVWYBXZJUO*-]", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES, help="S16 high-context-gain candidate TSV.")
    parser.add_argument(
        "--candidate-universe",
        type=Path,
        default=DEFAULT_UNIVERSE,
        help="Optional QC candidate universe for matched controls.",
    )
    parser.add_argument("--protein-index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/v2_sequence_structure_validation/targets"))
    parser.add_argument("--controls-per-candidate", type=int, default=1)
    parser.add_argument("--control-max-context-gain", type=float, default=0.05)
    parser.add_argument("--min-candidate-context-gain", type=float, default=0.2)
    parser.add_argument("--min-candidate-context-prob", type=float, default=0.8)
    parser.add_argument("--allow-control-reuse", action="store_true")
    parser.add_argument(
        "--known-positives-per-label",
        type=int,
        default=3,
        help="Known positive calibration targets per candidate label, selected from annotation-derived label rules.",
    )
    parser.add_argument(
        "--known-positive-split",
        default="train",
        help="Preferred family_holdout split for known positives; use any to ignore split.",
    )
    parser.add_argument("--max-targets", type=int, help="Optional cap for smoke tests.")
    parser.add_argument("--foldseek-bin", default="foldseek")
    parser.add_argument("--foldseek-db", type=Path, help="Foldseek target DB path for runnable command templates.")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument(
        "--run-foldseek",
        action="store_true",
        help="Run Foldseek using --pdb-dir and --foldseek-db after writing command templates.",
    )
    parser.add_argument("--pdb-dir", type=Path, help="Predicted PDB directory for --run-foldseek.")
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_path(path: Path, root: Path) -> Path:
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("r", encoding="utf-8", newline="")


def require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise SystemExit(f"Required {label} not found: {path}")


def first_present(row: dict[str, Any], columns: Iterable[str], default: str = "") -> str:
    for column in columns:
        value = row.get(column)
        if value is not None and str(value).strip() != "":
            return str(value).strip()
    return default


def require_any_column(fieldnames: list[str] | None, columns: Iterable[str], table_name: str) -> None:
    fields = set(fieldnames or [])
    if not any(column in fields for column in columns):
        joined = ", ".join(columns)
        raise SystemExit(f"{table_name} is missing a required accession/label column. Expected one of: {joined}")


def read_tsv(path: Path, table_name: str, require_columns: Iterable[Iterable[str]] = ()) -> list[dict[str, str]]:
    require_file(path, table_name)
    with open_text(path) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for columns in require_columns:
            require_any_column(reader.fieldnames, columns, table_name)
        return list(reader)


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


def as_float(value: Any, default: float = math.nan) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def normalize_candidate(row: dict[str, str]) -> dict[str, Any]:
    p_context = first_present(row, CONTEXT_PROB_COLUMNS)
    delta = first_present(row, CONTEXT_GAIN_COLUMNS)
    p_protein = first_present(row, PROTEIN_PROB_COLUMNS)
    if not p_protein and p_context and delta:
        p_protein_float = as_float(p_context) - as_float(delta)
        if not math.isnan(p_protein_float):
            p_protein = f"{max(0.0, min(1.0, p_protein_float)):.12g}"
    module_cluster_id = first_present(row, ("module_cluster_id", "cluster_id"))
    if module_cluster_id.lower() in {"-1", "none", "nan", "null"}:
        module_cluster_id = ""
    return {
        "protein_accession": first_present(row, ACCESSION_COLUMNS),
        "predicted_label": first_present(row, LABEL_COLUMNS),
        "genome_id": first_present(row, GENOME_COLUMNS),
        "family": first_present(row, ("family", "virus_family")),
        "host_group": first_present(row, ("host_group", "host_supergroup")),
        "description": first_present(row, ("description", "protein_description", "cds_product")),
        "p_protein_only": p_protein,
        "p_context": p_context,
        "delta_p": delta,
        "calibrated_probability": first_present(row, ("calibrated_probability", "top_probability_calibrated", "p_context")),
        "validation_gate_status": first_present(row, ("validation_gate_status", "fdr_gate_status")),
        "hypothetical_or_uncharacterized": first_present(
            row, ("hypothetical_or_uncharacterized", "hypothetical_or_unknown"), "0"
        ),
        "module_cluster_id": module_cluster_id,
        "exact_transfer_flag": first_present(row, ("exact_transfer_flag",)),
        "source_row": row,
    }


def load_split_map(path: Path, wanted: set[str]) -> dict[str, dict[str, str]]:
    if not path.exists() or not wanted:
        return {}
    out: dict[str, dict[str, str]] = {}
    with open_text(path) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if "protein_accession" not in (reader.fieldnames or []):
            raise SystemExit(f"Split manifest must contain protein_accession: {path}")
        for row in reader:
            accession = row.get("protein_accession", "")
            if accession in wanted:
                out[accession] = row
                if len(out) == len(wanted):
                    break
    return out


def load_all_split_map(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    out: dict[str, dict[str, str]] = {}
    with open_text(path) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if "protein_accession" not in (reader.fieldnames or []):
            raise SystemExit(f"Split manifest must contain protein_accession: {path}")
        for row in reader:
            accession = row.get("protein_accession", "")
            if accession:
                out[accession] = row
    return out


def load_sequence_map(path: Path, wanted: set[str]) -> dict[str, dict[str, str]]:
    require_file(path, "protein index")
    out: dict[str, dict[str, str]] = {}
    with open_text(path) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fields = set(reader.fieldnames or [])
        missing = {"protein_accession", "protein_sequence"} - fields
        if missing:
            raise SystemExit(f"Protein index is missing required columns {sorted(missing)}: {path}")
        for row in reader:
            accession = row.get("protein_accession", "")
            if accession in wanted:
                out[accession] = row
                if len(out) == len(wanted):
                    break
    missing_accessions = sorted(wanted - set(out))
    if missing_accessions:
        preview = ", ".join(missing_accessions[:10])
        raise SystemExit(f"Protein index lacks sequences for {len(missing_accessions)} target(s): {preview}")
    return out


def labels_for_annotation_row(row: dict[str, str]) -> list[str]:
    return [LABEL_RULES[idx].name for idx in label_hits(normalize_text(row))]


def select_known_positives(
    index_path: Path,
    split_map: dict[str, dict[str, str]],
    candidate_labels: set[str],
    exclude: set[str],
    per_label: int,
    preferred_split: str,
) -> list[dict[str, Any]]:
    if per_label <= 0 or not candidate_labels:
        return []
    selected: dict[str, list[dict[str, Any]]] = {label: [] for label in candidate_labels}

    def split_penalty(accession: str) -> int:
        if preferred_split.lower() == "any":
            return 0
        return 0 if split_map.get(accession, {}).get("family_holdout_split") == preferred_split else 1

    candidates_by_label: dict[str, list[dict[str, Any]]] = {label: [] for label in candidate_labels}
    with open_text(index_path) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fields = set(reader.fieldnames or [])
        missing = {"protein_accession", "protein_sequence"} - fields
        if missing:
            raise SystemExit(f"Protein index is missing required columns {sorted(missing)}: {index_path}")
        for protein in reader:
            accession = protein.get("protein_accession", "")
            if not accession or accession in exclude:
                continue
            labels = set(labels_for_annotation_row(protein)) & candidate_labels
            if not labels:
                continue
            description = (protein.get("protein_description", "") or protein.get("cds_product", "")).lower()
            hypothetical_penalty = 1 if any(token in description for token in ("hypothetical", "unknown", "uncharacterized")) else 0
            length = as_float(protein.get("protein_length_aa"), len(protein.get("protein_sequence", "")))
            for label in labels:
                candidates_by_label[label].append(
                    {
                        "protein_accession": accession,
                        "predicted_label": label,
                        "genome_id": protein.get("genome_version", ""),
                        "family": split_map.get(accession, {}).get("virus_family", ""),
                        "host_group": split_map.get(accession, {}).get("host_supergroup", ""),
                        "description": protein.get("protein_description", "") or protein.get("cds_product", ""),
                        "p_protein_only": "",
                        "p_context": "",
                        "delta_p": "",
                        "calibrated_probability": "",
                        "validation_gate_status": "known-positive calibration target",
                        "hypothetical_or_uncharacterized": int(hypothetical_penalty == 1),
                        "module_cluster_id": "",
                        "exact_transfer_flag": "",
                        "target_type": "known_positive_control",
                        "matched_candidate_id": "",
                        "match_rank": "",
                        "control_match_note": (
                            "Annotation-derived known positive used to calibrate sequence/structure evidence interpretation; "
                            "not a de novo model input."
                        ),
                        "_sort_key": (split_penalty(accession), hypothetical_penalty, abs(length - 300.0), accession),
                    }
                )

    for label, rows in candidates_by_label.items():
        rows.sort(key=lambda row: row["_sort_key"])
        selected[label] = rows[:per_label]
    out: list[dict[str, Any]] = []
    for label in sorted(selected):
        for row in selected[label]:
            row.pop("_sort_key", None)
            out.append(row)
    return out


def enrich_with_metadata(row: dict[str, Any], split: dict[str, str], protein: dict[str, str]) -> dict[str, Any]:
    enriched = dict(row)
    enriched["genome_id"] = enriched.get("genome_id") or split.get("genome_version") or protein.get("genome_version", "")
    enriched["family"] = enriched.get("family") or split.get("virus_family", "")
    enriched["host_group"] = enriched.get("host_group") or split.get("host_supergroup", "")
    enriched["description"] = enriched.get("description") or protein.get("protein_description", "") or protein.get("cds_product", "")
    enriched["sequence_length_aa"] = protein.get("protein_length_aa") or len(protein.get("protein_sequence", ""))
    enriched["protein_sequence_sha256"] = protein.get("protein_sequence_sha256") or hashlib.sha256(
        protein.get("protein_sequence", "").encode("utf-8")
    ).hexdigest()
    return enriched


def row_context_gain(row: dict[str, Any]) -> float:
    return as_float(row.get("delta_p"), math.inf)


def row_context_probability(row: dict[str, Any]) -> float:
    return as_float(row.get("p_context"), 0.0)


def is_high_context_candidate(row: dict[str, Any], min_gain: float, min_prob: float) -> bool:
    flag = str(row.get("source_row", {}).get("high_context_gain", "")).strip().lower()
    if flag in {"1", "true", "yes"}:
        return True
    return row_context_gain(row) >= min_gain and row_context_probability(row) >= min_prob


def choose_controls(
    candidates: list[dict[str, Any]],
    universe: list[dict[str, Any]],
    controls_per_candidate: int,
    max_context_gain: float,
    allow_reuse: bool,
) -> list[dict[str, Any]]:
    candidate_accessions = {row["protein_accession"] for row in candidates}
    selected: set[str] = set()
    controls: list[dict[str, Any]] = []
    pool = [row for row in universe if row.get("protein_accession") and row["protein_accession"] not in candidate_accessions]

    for candidate in candidates:
        label = candidate.get("predicted_label", "")
        family = candidate.get("family", "")
        host = candidate.get("host_group", "")
        length = as_float(candidate.get("sequence_length_aa"), 0.0)

        def control_score(row: dict[str, Any]) -> tuple[float, ...]:
            control_gain = row_context_gain(row)
            control_len = as_float(row.get("sequence_length_aa"), 0.0)
            return (
                0.0 if row.get("predicted_label", "") == label else 1.0,
                0.0 if control_gain <= max_context_gain else 1.0,
                0.0 if row.get("family", "") == family else 1.0,
                0.0 if row.get("host_group", "") == host else 1.0,
                abs(control_len - length),
                control_gain if not math.isinf(control_gain) else 999.0,
                -row_context_probability(row),
            )

        available = [row for row in pool if allow_reuse or row["protein_accession"] not in selected]
        available.sort(key=control_score)
        picked = available[:controls_per_candidate]
        for rank, row in enumerate(picked, start=1):
            selected.add(row["protein_accession"])
            item = dict(row)
            item["target_type"] = "matched_control"
            item["matched_candidate_id"] = candidate["protein_accession"]
            item["match_rank"] = rank
            item["control_match_note"] = (
                "Matched by predicted label first, then low context gain, family, host group, and sequence length. "
                "Controls are for analysis contrast only."
            )
            controls.append(item)
    return controls


def clean_sequence(sequence: str) -> str:
    cleaned = SAFE_AA_RE.sub("X", sequence.strip().upper()).replace("*", "")
    return cleaned


def fasta_header(row: dict[str, Any]) -> str:
    return str(row.get("protein_accession", "")).replace(" ", "_")


def write_fasta(path: Path, rows: list[dict[str, Any]]) -> None:
    lines: list[str] = []
    for row in rows:
        sequence = clean_sequence(str(row.get("protein_sequence", "")))
        if not sequence:
            continue
        lines.append(f">{fasta_header(row)}")
        lines.extend(sequence[i : i + 80] for i in range(0, len(sequence), 80))
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def write_command_templates(output_dir: Path, args: argparse.Namespace) -> None:
    fasta = output_dir / "all_validation_targets.fasta"
    pdb_dir = output_dir / "esmfold_pdb"
    foldseek_dir = output_dir / "foldseek"
    sh_db = str(args.foldseek_db).replace("\\", "/") if args.foldseek_db else "${FOLDSEEK_DB:?Set FOLDSEEK_DB to a Foldseek target DB prefix}"
    ps_db_setup = (
        f'$FoldseekDb = "{args.foldseek_db}"'
        if args.foldseek_db
        else '$FoldseekDb = $env:FOLDSEEK_DB\nif (-not $FoldseekDb) { throw "Set FOLDSEEK_DB to a Foldseek target DB prefix." }'
    )
    sh = f"""#!/usr/bin/env bash
set -euo pipefail

# Predict structures with ESMFold. Requires an environment with the esm-fold CLI.
mkdir -p "{pdb_dir.as_posix()}"
${{ESMFOLD_BIN:-esm-fold}} -i "{fasta.as_posix()}" -o "{pdb_dir.as_posix()}" --cpu-offload --chunk-size 128

# Alternative ColabFold entrypoint. Run this instead of the ESMFold command if preferred.
# colabfold_batch "{fasta.as_posix()}" "{(output_dir / 'colabfold_models').as_posix()}"

# Search predicted structures against a Foldseek database.
mkdir -p "{foldseek_dir.as_posix()}"
foldseek createdb "{pdb_dir.as_posix()}" "{(foldseek_dir / 'query_db').as_posix()}"
foldseek search "{(foldseek_dir / 'query_db').as_posix()}" "{sh_db}" "{(foldseek_dir / 'aln').as_posix()}" "{(foldseek_dir / 'tmp').as_posix()}" --threads {args.threads}
foldseek convertalis "{(foldseek_dir / 'query_db').as_posix()}" "{sh_db}" "{(foldseek_dir / 'aln').as_posix()}" "{(foldseek_dir / 'pdb_hits.tsv').as_posix()}" --format-output "query,target,evalue,bits,prob,alnlen,pident,lddt,alntmscore,qtmscore,ttmscore,taxid,taxname"
"""
    (output_dir / "run_structure_prediction_and_foldseek.sh").write_text(sh, encoding="utf-8")

    ps1 = f"""$ErrorActionPreference = "Stop"
{ps_db_setup}

# Predict structures with ESMFold. Requires an environment with the esm-fold CLI.
New-Item -ItemType Directory -Force -Path "{pdb_dir}" | Out-Null
$EsmfoldBin = if ($env:ESMFOLD_BIN) {{ $env:ESMFOLD_BIN }} else {{ "esm-fold" }}
& $EsmfoldBin -i "{fasta}" -o "{pdb_dir}" --cpu-offload --chunk-size 128

# Alternative ColabFold entrypoint. Run this instead of the ESMFold command if preferred.
# colabfold_batch "{fasta}" "{output_dir / 'colabfold_models'}"

# Search predicted structures against a Foldseek database.
New-Item -ItemType Directory -Force -Path "{foldseek_dir}" | Out-Null
foldseek createdb "{pdb_dir}" "{foldseek_dir / 'query_db'}"
foldseek search "{foldseek_dir / 'query_db'}" "$FoldseekDb" "{foldseek_dir / 'aln'}" "{foldseek_dir / 'tmp'}" --threads {args.threads}
foldseek convertalis "{foldseek_dir / 'query_db'}" "$FoldseekDb" "{foldseek_dir / 'aln'}" "{foldseek_dir / 'pdb_hits.tsv'}" --format-output "query,target,evalue,bits,prob,alnlen,pident,lddt,alntmscore,qtmscore,ttmscore,taxid,taxname"
"""
    (output_dir / "run_structure_prediction_and_foldseek.ps1").write_text(ps1, encoding="utf-8")


def foldseek_db_exists(prefix: Path) -> bool:
    return prefix.exists() or prefix.with_suffix(prefix.suffix + ".dbtype").exists() or Path(str(prefix) + ".dbtype").exists()


def run_foldseek(args: argparse.Namespace, output_dir: Path) -> None:
    if not args.pdb_dir:
        raise SystemExit("--run-foldseek requires --pdb-dir.")
    if not args.foldseek_db:
        raise SystemExit("--run-foldseek requires --foldseek-db.")
    foldseek = shutil.which(args.foldseek_bin) or str(args.foldseek_bin)
    if not Path(foldseek).exists() and shutil.which(args.foldseek_bin) is None:
        raise SystemExit(f"Foldseek executable not found: {args.foldseek_bin}")
    pdb_dir = resolve_path(args.pdb_dir, repo_root())
    db = resolve_path(args.foldseek_db, repo_root())
    if not foldseek_db_exists(db):
        raise SystemExit(f"Foldseek target database prefix not found: {db}")
    if not pdb_dir.exists():
        raise SystemExit(f"Predicted PDB directory not found: {pdb_dir}")
    foldseek_dir = output_dir / "foldseek"
    foldseek_dir.mkdir(parents=True, exist_ok=True)
    query_db = foldseek_dir / "query_db"
    aln = foldseek_dir / "aln"
    tmp = foldseek_dir / "tmp"
    hits = foldseek_dir / "pdb_hits.tsv"
    commands = [
        [foldseek, "createdb", str(pdb_dir), str(query_db)],
        [foldseek, "search", str(query_db), str(db), str(aln), str(tmp), "--threads", str(args.threads)],
        [
            foldseek,
            "convertalis",
            str(query_db),
            str(db),
            str(aln),
            str(hits),
            "--format-output",
            "query,target,evalue,bits,prob,alnlen,pident,lddt,alntmscore,qtmscore,ttmscore,taxid,taxname",
        ],
    ]
    for command in commands:
        print("[cmd]", " ".join(command), flush=True)
        subprocess.run(command, check=True)


def main() -> int:
    args = parse_args()
    root = repo_root()
    candidates_path = resolve_path(args.candidates, root)
    universe_path = resolve_path(args.candidate_universe, root)
    protein_index = resolve_path(args.protein_index, root)
    split_manifest = resolve_path(args.split_manifest, root)
    output_dir = resolve_path(args.output_dir, root)
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates_raw = read_tsv(
        candidates_path,
        "high-context-gain candidates",
        require_columns=(ACCESSION_COLUMNS, LABEL_COLUMNS),
    )
    if args.max_targets:
        candidates_raw = candidates_raw[: args.max_targets]
    candidates = [normalize_candidate(row) for row in candidates_raw]
    candidates = [row for row in candidates if row.get("protein_accession")]
    if candidates and any("high_context_gain" in row.get("source_row", {}) for row in candidates):
        candidates = [
            row
            for row in candidates
            if is_high_context_candidate(row, args.min_candidate_context_gain, args.min_candidate_context_prob)
        ]
    if not candidates:
        raise SystemExit(f"No candidate accessions were found in {candidates_path}")

    universe: list[dict[str, Any]] = []
    if universe_path.exists():
        universe_raw = read_tsv(universe_path, "candidate universe", require_columns=(ACCESSION_COLUMNS, LABEL_COLUMNS))
        universe = [normalize_candidate(row) for row in universe_raw if first_present(row, ACCESSION_COLUMNS)]

    wanted = {row["protein_accession"] for row in candidates}
    wanted.update(row["protein_accession"] for row in universe)
    split_map = load_split_map(split_manifest, wanted)
    sequence_map = load_sequence_map(protein_index, wanted)

    candidates = [
        enrich_with_metadata(row, split_map.get(row["protein_accession"], {}), sequence_map[row["protein_accession"]])
        for row in candidates
    ]
    for row in candidates:
        row["target_type"] = "high_context_candidate"
        row["matched_candidate_id"] = ""
        row["match_rank"] = ""
        row["control_match_note"] = ""

    universe = [
        enrich_with_metadata(row, split_map.get(row["protein_accession"], {}), sequence_map[row["protein_accession"]])
        for row in universe
        if row["protein_accession"] in sequence_map
    ]

    controls = choose_controls(
        candidates,
        universe,
        controls_per_candidate=max(0, args.controls_per_candidate),
        max_context_gain=args.control_max_context_gain,
        allow_reuse=args.allow_control_reuse,
    )

    all_split_map = load_all_split_map(split_manifest) if args.known_positives_per_label > 0 else {}
    candidate_labels = {str(row.get("predicted_label", "")) for row in candidates if row.get("predicted_label", "")}
    excluded = {row["protein_accession"] for row in candidates + controls}
    known_positives = select_known_positives(
        protein_index,
        all_split_map,
        candidate_labels=candidate_labels,
        exclude=excluded,
        per_label=args.known_positives_per_label,
        preferred_split=args.known_positive_split,
    )
    known_wanted = {row["protein_accession"] for row in known_positives}
    if known_wanted:
        known_sequences = load_sequence_map(protein_index, known_wanted)
        sequence_map.update(known_sequences)
        for row in known_positives:
            accession = row["protein_accession"]
            enriched = enrich_with_metadata(row, all_split_map.get(accession, {}), known_sequences[accession])
            row.update(enriched)

    all_targets = candidates + controls + known_positives
    for row in all_targets:
        row["protein_sequence"] = sequence_map[row["protein_accession"]]["protein_sequence"]
        row["manuscript_caveat"] = "computationally prioritized or matched analysis target; independent validation required"
        row["de_novo_input_guardrail"] = (
            "do not use product text, neighbor true labels, label counts, homology hits, or structure hits as de novo inputs"
        )

    fields = [
        "target_type",
        "matched_candidate_id",
        "match_rank",
        "protein_accession",
        "predicted_label",
        "p_protein_only",
        "p_context",
        "delta_p",
        "family",
        "host_group",
        "genome_id",
        "description",
        "hypothetical_or_uncharacterized",
        "module_cluster_id",
        "exact_transfer_flag",
        "sequence_length_aa",
        "protein_sequence_sha256",
        "validation_gate_status",
        "control_match_note",
        "de_novo_input_guardrail",
        "manuscript_caveat",
    ]
    write_tsv(output_dir / "validation_candidates.tsv", candidates, fields)
    write_tsv(output_dir / "validation_controls.tsv", controls, fields)
    write_tsv(output_dir / "known_positive_controls.tsv", known_positives, fields)
    write_tsv(output_dir / "validation_targets.tsv", all_targets, fields)
    write_fasta(output_dir / "candidate_targets.fasta", candidates)
    write_fasta(output_dir / "matched_control_targets.fasta", controls)
    write_fasta(output_dir / "known_positive_targets.fasta", known_positives)
    write_fasta(output_dir / "all_validation_targets.fasta", all_targets)

    pdb_manifest = []
    for row in all_targets:
        pdb_manifest.append(
            {
                "protein_accession": row["protein_accession"],
                "target_type": row["target_type"],
                "expected_esmfold_pdb": str(output_dir / "esmfold_pdb" / f"{row['protein_accession']}.pdb"),
                "expected_colabfold_glob": str(output_dir / "colabfold_models" / f"{row['protein_accession']}*.pdb"),
            }
        )
    write_tsv(output_dir / "structure_prediction_manifest.tsv", pdb_manifest)
    write_command_templates(output_dir, args)

    report = {
        "claim_frame": (
            "Genome context complements sequence and structure by prioritizing and sometimes disambiguating "
            "candidate viral protein functions under leakage-aware OOD evaluation."
        ),
        "candidates": str(candidates_path),
        "candidate_count": len(candidates),
        "candidate_universe": str(universe_path) if universe_path.exists() else "",
        "control_count": len(controls),
        "known_positive_count": len(known_positives),
        "targets_total": len(all_targets),
        "outputs": {
            "targets_tsv": str(output_dir / "validation_targets.tsv"),
            "candidate_fasta": str(output_dir / "candidate_targets.fasta"),
            "control_fasta": str(output_dir / "matched_control_targets.fasta"),
            "all_fasta": str(output_dir / "all_validation_targets.fasta"),
            "commands_sh": str(output_dir / "run_structure_prediction_and_foldseek.sh"),
            "commands_ps1": str(output_dir / "run_structure_prediction_and_foldseek.ps1"),
        },
    }
    (output_dir / "validation_target_prep_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if args.run_foldseek:
        run_foldseek(args, output_dir)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
