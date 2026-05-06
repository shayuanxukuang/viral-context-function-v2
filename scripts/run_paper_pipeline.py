from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def timestamp() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="One-click paper pipeline for the tasks that are ready in this repo.")
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--benchmark-root", default="runs/task_mode_suite_server")
    parser.add_argument("--context-study-input", default="runs/context_study_main.essential.tar.gz")
    parser.add_argument("--freeze-output-tsv", default="runs/frozen_benchmark_v1.tsv")
    parser.add_argument("--claims-ledger", default="runs/claims_ledger.md")
    parser.add_argument("--paper-numbers", default="runs/paper_numbers.json")
    parser.add_argument("--biophysics-output-dir", default="runs/biophysics_qc")
    parser.add_argument("--atlas-output-root", default="runs/context_atlas_v2")
    parser.add_argument("--uncertainty-output-root", default="runs/uncertainty")
    parser.add_argument("--bootstrap-iterations", type=int, default=100)
    parser.add_argument("--permutation-iterations", type=int, default=100)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--skip-freeze", action="store_true")
    parser.add_argument("--skip-biophysics-qc", action="store_true")
    parser.add_argument("--skip-atlas", action="store_true")
    parser.add_argument("--skip-source-decomposition", action="store_true")
    parser.add_argument("--skip-uncertainty", action="store_true")
    parser.add_argument("--skip-plm", action="store_true")
    parser.add_argument("--plm-embedding-path", default="")
    parser.add_argument("--plm-output-root", default="runs/plm_clean_quartet")
    parser.add_argument("--skip-discovery", action="store_true")
    parser.add_argument("--discovery-run-dir", default="")
    parser.add_argument("--discovery-output-root", default="runs/module_discovery")
    return parser.parse_args()


def run(command: list[str], cwd: Path) -> None:
    print("[run]", " ".join(command))
    subprocess.run(command, cwd=cwd, check=True)


def main() -> int:
    args = parse_args()
    root = repo_root()
    benchmark_root = (root / args.benchmark_root).resolve()
    context_study_input = (root / args.context_study_input).resolve()
    atlas_output_root = (root / args.atlas_output_root).resolve()
    uncertainty_output_root = (root / args.uncertainty_output_root).resolve()

    completed_steps: list[str] = []

    if not args.skip_freeze:
        run(
            [
                sys.executable,
                "scripts/freeze_benchmark_v1.py",
                "--runs-root",
                args.runs_root,
                "--output-tsv",
                args.freeze_output_tsv,
                "--claims-ledger",
                args.claims_ledger,
                "--paper-numbers",
                args.paper_numbers,
            ],
            root,
        )
        completed_steps.append("freeze_benchmark_v1")

    if not args.skip_biophysics_qc:
        run(
            [
                sys.executable,
                "scripts/qc_biophysics_features.py",
                "--output-dir",
                args.biophysics_output_dir,
            ],
            root,
        )
        completed_steps.append("biophysics_qc")

    if not args.skip_atlas:
        atlas_pairs = [
            ("family_holdout", benchmark_root / "protein_only.family_holdout", benchmark_root / "genome_aware_denovo.family_holdout"),
            ("host_holdout", benchmark_root / "protein_only.host_holdout", benchmark_root / "genome_aware_denovo.host_holdout"),
        ]
        for split_name, protein_run, context_run in atlas_pairs:
            run(
                [
                    sys.executable,
                    "scripts/build_context_dependence_atlas_v2.py",
                    "--protein-run",
                    str(protein_run),
                    "--context-run",
                    str(context_run),
                    "--output-dir",
                    str(atlas_output_root / split_name),
                    "--bootstrap-iterations",
                    str(args.bootstrap_iterations),
                    "--permutation-iterations",
                    str(args.permutation_iterations),
                ],
                root,
            )
        completed_steps.append("context_atlas_v2")

    if not args.skip_source_decomposition and context_study_input.exists():
        run(
            [
                sys.executable,
                "scripts/summarize_source_decomposition.py",
                "--input",
                str(context_study_input),
                "--output-dir",
                str((context_study_input.parent / "source_decomposition")),
            ],
            root,
        )
        completed_steps.append("source_decomposition_summary")

    if not args.skip_uncertainty:
        for run_name in ("protein_only.family_holdout", "genome_aware_denovo.family_holdout", "protein_only.host_holdout", "genome_aware_denovo.host_holdout"):
            run(
                [
                    sys.executable,
                    "scripts/calibrate_task_mode_uncertainty.py",
                    "--run-dir",
                    str(benchmark_root / run_name),
                    "--output-dir",
                    str(uncertainty_output_root / run_name.replace(".", "_")),
                    "--device",
                    args.device,
                ],
                root,
            )
        completed_steps.append("uncertainty_calibration")

    if not args.skip_plm and args.plm_embedding_path:
        run(
            [
                "bash",
                "./scripts/run_plm_clean_quartet.sh",
                "--output-root",
                args.plm_output_root,
                "--plm-embedding-path",
                args.plm_embedding_path,
            ],
            root,
        )
        completed_steps.append("plm_clean_quartet")

    if not args.skip_discovery and args.discovery_run_dir:
        discovery_output_root = (root / args.discovery_output_root).resolve()
        embedding_file = discovery_output_root / "exported_fused_test_embeddings.pt"
        run(
            [
                sys.executable,
                "scripts/export_task_mode_embeddings.py",
                "--run-dir",
                args.discovery_run_dir,
                "--output",
                str(embedding_file),
                "--representation",
                "fused",
                "--split",
                "test",
                "--device",
                args.device,
            ],
            root,
        )
        run(
            [
                sys.executable,
                "scripts/discover_module_candidates.py",
                "--embedding-file",
                str(embedding_file),
                "--output-dir",
                str(discovery_output_root),
            ],
            root,
        )
        ranked_clusters = discovery_output_root / "ranked_hypothetical_clusters.tsv"
        module_candidates = discovery_output_root / "module_candidates.tsv"
        if ranked_clusters.exists() and module_candidates.exists():
            run(
                [
                    sys.executable,
                    "scripts/prepare_targeted_structure_validation.py",
                    "--ranked-clusters",
                    str(ranked_clusters),
                    "--module-candidates",
                    str(module_candidates),
                    "--output-dir",
                    str(discovery_output_root / "targeted_structure_validation"),
                ],
                root,
            )
        completed_steps.append("module_discovery")

    manifest = {
        "created_at": timestamp(),
        "completed_steps": completed_steps,
        "benchmark_root": str(benchmark_root),
        "context_study_input": str(context_study_input),
        "atlas_output_root": str(atlas_output_root),
        "uncertainty_output_root": str(uncertainty_output_root),
    }
    manifest_path = root / "runs" / "paper_pipeline_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
