#!/usr/bin/env python3
"""One-click runner for ViruFunc V2 sequence-structure-context experiments.

Stages:
1. Build Group A/B/C validation targets: high-context candidates, matched
   controls, and known-positive calibration controls.
2. Run target-specific MMseqs2 sequence evidence, unless skipped.
3. Optionally run ESMFold and Foldseek structure search.
4. Build manuscript-ready evidence, structural ambiguity, Figure 6 source, and
   sequence-context landscape outputs.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("runs/v2_sequence_structure_validation"))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--protein-index", type=Path, default=Path("data/processed/training/viral_protein_training_index.tsv.gz"))
    parser.add_argument("--split-manifest", type=Path, default=Path("data/processed/splits/viral_protein_strict_splits.tsv.gz"))
    parser.add_argument(
        "--candidates",
        type=Path,
        default=Path(
            "artifacts/return/v2_plos_cb_supplementary_package_20260504/"
            "supplementary_tables/S16_high_context_gain_candidates.tsv"
        ),
    )
    parser.add_argument(
        "--candidate-universe",
        type=Path,
        default=Path("artifacts/return/context_study_v2_review_completion_20260504/qc_review/qc7_candidate_assignments.tsv"),
    )
    parser.add_argument("--module-candidates", type=Path, default=Path("artifacts/return/extracted_v2_20260430_100225/module_discovery/module_candidates.tsv"))
    parser.add_argument(
        "--module-clusters",
        type=Path,
        default=Path("artifacts/return/v2_plos_cb_supplementary_package_20260504/supplementary_tables/S17_module_clusters.tsv"),
    )
    parser.add_argument(
        "--fallback-homology-hits",
        type=Path,
        default=Path("artifacts/return/v2_plos_cb_supplementary_package_20260504/supplementary_tables/S21_homology_top_hit_assignments.tsv"),
    )
    parser.add_argument("--controls-per-candidate", type=int, default=1)
    parser.add_argument("--known-positives-per-label", type=int, default=3)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--mmseqs-bin", default=os.environ.get("MMSEQS_BIN", "mmseqs"))
    parser.add_argument("--foldseek-bin", default=os.environ.get("FOLDSEEK_BIN", "foldseek"))
    parser.add_argument("--esmfold-bin", default=os.environ.get("ESMFOLD_BIN"), help="Optional esm-fold console script.")
    parser.add_argument(
        "--esmfold-python",
        default=os.environ.get("ESMFOLD_PYTHON"),
        help="Optional Python executable from an environment that has fair-esm/ESMFold installed.",
    )
    parser.add_argument("--foldseek-db", type=Path, default=Path(os.environ["FOLDSEEK_DB"]) if os.environ.get("FOLDSEEK_DB") else None)
    parser.add_argument("--pdb-dir", type=Path, help="Existing predicted PDB directory. Defaults to <output-root>/targets/esmfold_pdb.")
    parser.add_argument("--foldseek-hits", type=Path, help="Existing Foldseek convertalis TSV.")
    parser.add_argument("--structure-summary", type=Path, help="Existing ESMFold/Foldseek summary TSV.")
    parser.add_argument("--run-esmfold", action="store_true", help="Run python -m esm.scripts.fold for all validation targets.")
    parser.add_argument("--run-foldseek", action="store_true", help="Run Foldseek search against --foldseek-db.")
    parser.add_argument("--skip-homology", action="store_true")
    parser.add_argument("--skip-landscape", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve(path: Path | None, root: Path) -> Path | None:
    if path is None:
        return None
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def run_cmd(cmd: list[str], cwd: Path, dry_run: bool = False) -> None:
    print("[cmd]", " ".join(str(part) for part in cmd), flush=True)
    if dry_run:
        return
    subprocess.run([str(part) for part in cmd], cwd=str(cwd), check=True)


def executable_exists(value: str) -> bool:
    if not value:
        return False
    return bool(shutil.which(value) or (Path(value).exists() if any(sep in value for sep in ("/", "\\")) else False))


def resolve_executable_path(value: str | None) -> str | None:
    if not value:
        return None
    if any(sep in value for sep in ("/", "\\")):
        path = Path(value).expanduser()
        return str(path) if path.exists() else None
    return shutil.which(value)


def discover_file(root: Path, patterns: list[str]) -> Path | None:
    matches: list[Path] = []
    for pattern in patterns:
        matches.extend(path for path in root.glob(pattern) if path.is_file())
    if not matches:
        return None
    matches = sorted(set(matches), key=lambda path: (path.stat().st_mtime, len(str(path))), reverse=True)
    return matches[0]


def resolve_or_discover(path: Path | None, root: Path, label: str, patterns: list[str], required: bool) -> Path | None:
    resolved = resolve(path, root) if path is not None else None
    if resolved is not None and resolved.exists():
        return resolved
    found = discover_file(root, patterns)
    if found is not None:
        print(f"[auto-discover] {label}: {found}", flush=True)
        return found
    if required:
        hint = "\n  ".join(patterns)
        raise SystemExit(
            f"Could not find required {label}.\n"
            f"Requested path: {resolved}\n"
            f"Searched patterns under {root}:\n  {hint}\n\n"
            "Pass the file explicitly, or first unpack/copy the returned review/supplementary package."
        )
    if resolved is not None:
        print(f"[note] optional {label} not found: {resolved}", flush=True)
    return resolved


def foldseek_db_exists(path: Path | None) -> bool:
    if path is None:
        return False
    return path.exists() or Path(str(path) + ".dbtype").exists() or path.with_suffix(path.suffix + ".dbtype").exists()


def main() -> int:
    args = parse_args()
    root = repo_root()
    out_root = resolve(args.output_root, root)
    assert out_root is not None
    targets_dir = out_root / "targets"
    homology_dir = out_root / "homology"
    evidence_dir = out_root / "evidence"
    landscape_dir = out_root / "landscape"
    targets_dir.mkdir(parents=True, exist_ok=True)

    candidates_path = resolve_or_discover(
        args.candidates,
        root,
        "high-context-gain candidates",
        [
            "artifacts/return/**/supplementary_tables/S16_high_context_gain_candidates.tsv",
            "artifacts/return/**/figure5_high_context_gain_candidates.tsv",
            "runs/**/supplementary_tables/S16_high_context_gain_candidates.tsv",
            "runs/**/figure5_high_context_gain_candidates.tsv",
            "artifacts/return/**/qc_review/qc7_candidate_assignments.tsv",
            "runs/**/qc_review/qc7_candidate_assignments.tsv",
        ],
        required=True,
    )
    candidate_universe_path = resolve_or_discover(
        args.candidate_universe,
        root,
        "candidate universe/qc7 assignments",
        [
            "artifacts/return/**/qc_review/qc7_candidate_assignments.tsv",
            "runs/**/qc_review/qc7_candidate_assignments.tsv",
        ],
        required=False,
    )
    module_candidates_path = resolve_or_discover(
        args.module_candidates,
        root,
        "module candidates",
        ["artifacts/return/**/module_discovery/module_candidates.tsv", "runs/**/module_discovery/module_candidates.tsv"],
        required=False,
    )
    module_clusters_path = resolve_or_discover(
        args.module_clusters,
        root,
        "module cluster table",
        [
            "artifacts/return/**/supplementary_tables/S17_module_clusters.tsv",
            "artifacts/return/**/module_discovery/ranked_hypothetical_clusters.tsv",
            "runs/**/module_discovery/ranked_hypothetical_clusters.tsv",
        ],
        required=False,
    )
    fallback_homology_path = resolve_or_discover(
        args.fallback_homology_hits,
        root,
        "fallback S21/MMseqs2 homology table",
        [
            "artifacts/return/**/supplementary_tables/S21_homology_top_hit_assignments.tsv",
            "artifacts/return/**/homology_baseline/homology_top_hit_assignments.tsv",
            "runs/**/homology_baseline/homology_top_hit_assignments.tsv",
        ],
        required=False,
    )

    pdb_dir = resolve(args.pdb_dir, root) if args.pdb_dir else targets_dir / "esmfold_pdb"
    foldseek_hits = resolve(args.foldseek_hits, root) if args.foldseek_hits else targets_dir / "foldseek" / "pdb_hits.tsv"
    structure_summary = resolve(args.structure_summary, root)
    foldseek_db = resolve(args.foldseek_db, root)

    run_cmd(
        [
            args.python,
            "scripts/prepare_v2_sequence_structure_validation.py",
            "--candidates",
            candidates_path,
            "--candidate-universe",
            candidate_universe_path or candidates_path,
            "--protein-index",
            resolve(args.protein_index, root),
            "--split-manifest",
            resolve(args.split_manifest, root),
            "--output-dir",
            targets_dir,
            "--controls-per-candidate",
            str(args.controls_per_candidate),
            "--known-positives-per-label",
            str(args.known_positives_per_label),
            "--threads",
            str(args.threads),
            "--foldseek-bin",
            args.foldseek_bin,
            *(
                ["--foldseek-db", foldseek_db]
                if foldseek_db is not None
                else []
            ),
        ],
        root,
        dry_run=args.dry_run,
    )

    homology_hits = fallback_homology_path
    landscape_homology_hits = fallback_homology_path
    homology_scheme = "family_holdout"
    homology_subset = "all_test"
    landscape_homology_scheme = "family_holdout"
    landscape_homology_subset = "all_test"
    if not args.skip_homology:
        if not executable_exists(args.mmseqs_bin) and not args.dry_run:
            raise SystemExit(
                f"MMseqs2 executable not found: {args.mmseqs_bin}. Install MMseqs2, set MMSEQS_BIN, or pass --skip-homology."
            )
        run_cmd(
            [
                args.python,
                "scripts/run_v2_target_homology_search.py",
                "--targets",
                targets_dir / "validation_targets.tsv",
                "--target-fasta",
                targets_dir / "all_validation_targets.fasta",
                "--protein-index",
                resolve(args.protein_index, root),
                "--split-manifest",
                resolve(args.split_manifest, root),
                "--output-dir",
                homology_dir,
                "--mmseqs-bin",
                args.mmseqs_bin,
                "--threads",
                str(args.threads),
                "--reuse-hits",
            ],
            root,
            dry_run=args.dry_run,
        )
        homology_hits = homology_dir / "target_homology_top_hit_assignments.tsv"
        homology_scheme = "target_validation"
        homology_subset = "all_targets"

    if args.run_esmfold:
        esmfold_bin = resolve_executable_path(args.esmfold_bin) or resolve_executable_path("esm-fold")
        esmfold_python = resolve_executable_path(args.esmfold_python)
        if esmfold_bin:
            run_cmd(
                [
                    esmfold_bin,
                    "-i",
                    targets_dir / "all_validation_targets.fasta",
                    "-o",
                    pdb_dir,
                    "--cpu-offload",
                    "--chunk-size",
                    "128",
                ],
                root,
                dry_run=args.dry_run,
            )
        else:
            python_for_esmfold = esmfold_python or args.python
            run_cmd(
                [
                    python_for_esmfold,
                    "scripts/predict_esmfold_structures.py",
                    "-i",
                    targets_dir / "all_validation_targets.fasta",
                    "-o",
                    pdb_dir,
                    "--cpu-offload",
                    "--chunk-size",
                    "128",
                ],
                root,
                dry_run=args.dry_run,
            )

    if args.run_foldseek:
        if not foldseek_db_exists(foldseek_db):
            raise SystemExit("Foldseek DB prefix not found. Pass --foldseek-db or set FOLDSEEK_DB.")
        if not executable_exists(args.foldseek_bin) and not args.dry_run:
            raise SystemExit(f"Foldseek executable not found: {args.foldseek_bin}. Set FOLDSEEK_BIN or pass --foldseek-bin.")
        if not pdb_dir.exists() and not args.dry_run:
            raise SystemExit(f"PDB directory not found for Foldseek: {pdb_dir}. Run --run-esmfold first or pass --pdb-dir.")
        foldseek_dir = targets_dir / "foldseek"
        foldseek_dir.mkdir(parents=True, exist_ok=True)
        run_cmd([args.foldseek_bin, "createdb", pdb_dir, foldseek_dir / "query_db"], root, dry_run=args.dry_run)
        run_cmd(
            [
                args.foldseek_bin,
                "search",
                foldseek_dir / "query_db",
                foldseek_db,
                foldseek_dir / "aln",
                foldseek_dir / "tmp",
                "-a",
                "--threads",
                str(args.threads),
            ],
            root,
            dry_run=args.dry_run,
        )
        run_cmd(
            [
                args.foldseek_bin,
                "convertalis",
                foldseek_dir / "query_db",
                foldseek_db,
                foldseek_dir / "aln",
                foldseek_hits,
                "--format-output",
                "query,target,evalue,bits,prob,alnlen,pident,lddt,alntmscore,qtmscore,ttmscore,taxid,taxname",
            ],
            root,
            dry_run=args.dry_run,
        )

    evidence_cmd: list[Any] = [
        args.python,
        "scripts/build_v2_sequence_structure_context_validation.py",
        "--targets",
        targets_dir / "validation_targets.tsv",
        "--candidates",
        candidates_path,
        "--module-candidates",
        module_candidates_path or Path("__missing_module_candidates.tsv"),
        "--module-clusters",
        module_clusters_path or Path("__missing_module_clusters.tsv"),
        "--homology-scheme",
        homology_scheme,
        "--homology-subset",
        homology_subset,
        "--output-dir",
        evidence_dir,
        "--pdb-dir",
        pdb_dir,
    ]
    if homology_hits is not None:
        evidence_cmd.extend(["--homology-hits", homology_hits])
    if foldseek_hits and foldseek_hits.exists():
        evidence_cmd.extend(["--foldseek-hits", foldseek_hits])
    elif not args.dry_run:
        print(f"[note] Foldseek hits not found for current targets: {foldseek_hits}; structure evidence will be marked pending.")
    if structure_summary and structure_summary.exists():
        evidence_cmd.extend(["--structure-summary", structure_summary])
    run_cmd(evidence_cmd, root, dry_run=args.dry_run)

    if not args.skip_landscape and candidate_universe_path is None:
        print("[note] skipping landscape because qc7 candidate-universe table was not found.", flush=True)
    elif not args.skip_landscape:
        landscape_cmd: list[Any] = [
                args.python,
                "scripts/analyze_v2_sequence_context_landscape.py",
                "--candidate-assignments",
                candidate_universe_path,
                "--split-manifest",
                resolve(args.split_manifest, root),
                "--output-dir",
                landscape_dir,
            ]
        if landscape_homology_hits is not None:
            landscape_cmd.extend(
                [
                    "--homology-hits",
                    landscape_homology_hits,
                    "--homology-scheme",
                    landscape_homology_scheme,
                    "--homology-subset",
                    landscape_homology_subset,
                ]
            )
        run_cmd(landscape_cmd, root, dry_run=args.dry_run)

    report: dict[str, Any] = {
        "output_root": str(out_root),
        "target_prep": str(targets_dir / "validation_target_prep_report.json"),
        "homology": str(homology_hits) if homology_hits else "",
        "landscape_homology": str(landscape_homology_hits) if landscape_homology_hits else "",
        "foldseek_hits": str(foldseek_hits) if foldseek_hits else "",
        "evidence": str(evidence_dir / "sequence_structure_context_validation_report.json"),
        "landscape": str(landscape_dir / "sequence_context_landscape_report.json") if not args.skip_landscape else "",
        "claim_frame": (
            "Genome context complements sequence and structure by prioritizing and sometimes disambiguating "
            "candidate viral protein functions under leakage-aware OOD evaluation."
        ),
    }
    (out_root / "breakthrough_validation_manifest.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
