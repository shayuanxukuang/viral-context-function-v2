#!/usr/bin/env python3
"""One-command runner for remaining PLOS CB reviewer-facing V2 analyses."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True, help="Completed main V2 run root.")
    parser.add_argument("--output-root", type=Path, help="Defaults to <run-root>/review_completion.")
    parser.add_argument("--input", type=Path, default=Path("data/processed/training/viral_protein_training_index.tsv.gz"))
    parser.add_argument("--split-manifest", type=Path, default=Path("data/processed/splits/viral_protein_strict_splits.tsv.gz"))
    parser.add_argument("--freeze-dir", type=Path, default=Path("data/v2_freeze"))
    parser.add_argument("--assets-dir", type=Path, help="Manuscript assets directory.")
    parser.add_argument("--qc-dir", type=Path, help="QC directory. Defaults to <run-root>/qc_review.")
    parser.add_argument("--plm-embedding-path", type=Path, help="Defaults to value in protein_only.family_holdout/run_manifest.json.")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--gpu-ids", default="4,5,6")
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--module-null-iterations", type=int, default=500)
    parser.add_argument("--threads", type=int, default=24)
    parser.add_argument("--mmseqs-bin", default=os.environ.get("MMSEQS_BIN", "mmseqs"))
    parser.add_argument("--skip-multiseed", action="store_true")
    parser.add_argument("--skip-source-ci", action="store_true")
    parser.add_argument("--skip-homology", action="store_true")
    parser.add_argument("--skip-nucleocapsid-sensitivity", action="store_true")
    parser.add_argument("--skip-candidate-evidence", action="store_true")
    parser.add_argument("--skip-supplement-package", action="store_true")
    parser.add_argument("--include-annotation-refinement", action="store_true")
    parser.add_argument("--rerun-completed", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def infer_plm_path(args: argparse.Namespace) -> Path:
    if args.plm_embedding_path:
        return args.plm_embedding_path
    manifest = read_json(args.run_root / "protein_only.family_holdout" / "run_manifest.json")
    config = manifest.get("config", {})
    value = config.get("plm_embedding_path", "")
    if value:
        return Path(value)
    return Path("data/processed/plm/esm2_t33_650M_UR50D_embeddings.pt")


def run_cmd(cmd: list[str], cwd: Path, env: dict[str, str] | None = None, dry_run: bool = False) -> int:
    print("[cmd]", " ".join(cmd), flush=True)
    if dry_run:
        return 0
    return subprocess.run(cmd, cwd=str(cwd), env=env, check=True).returncode


def multiseed_jobs(args: argparse.Namespace, out_root: Path, plm_path: Path) -> list[tuple[str, list[str]]]:
    jobs: list[tuple[str, list[str]]] = []
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    splits = ["family_holdout", "host_holdout"]
    modes = ["protein_only", "genome_aware_denovo"]
    if args.include_annotation_refinement:
        modes.append("annotation_refinement")
    for seed in seeds:
        for split in splits:
            for mode in modes:
                run_name = f"{mode}.{split}.seed{seed}"
                run_dir = out_root / "multiseed" / run_name
                status = run_dir / "run_status.json"
                if status.exists() and not args.rerun_completed:
                    continue
                cmd = [
                    args.python,
                    "-u",
                    "scripts/train_task_modes.py",
                    "--input",
                    str(args.input),
                    "--split-manifest",
                    str(args.split_manifest),
                    "--split-scheme",
                    split,
                    "--task-mode",
                    mode,
                    "--sequence-backbone",
                    "precomputed_plm",
                    "--plm-embedding-path",
                    str(plm_path),
                    "--output-dir",
                    str(run_dir),
                    "--epochs",
                    str(args.epochs),
                    "--batch-size",
                    str(args.batch_size),
                    "--eval-batch-size",
                    str(args.eval_batch_size),
                    "--num-workers",
                    str(args.num_workers),
                    "--prefetch-factor",
                    str(args.prefetch_factor),
                    "--neighbor-radius",
                    "2",
                    "--hidden-dim",
                    "256",
                    "--dropout",
                    "0.2",
                    "--learning-rate",
                    "0.0003",
                    "--weight-decay",
                    "0.01",
                    "--min-label-count",
                    "500",
                    "--seed",
                    str(seed),
                    "--gradient-clip",
                    "1.0",
                    "--warmup-fraction",
                    "0.05",
                    "--max-pos-weight",
                    "50.0",
                    "--device",
                    "cuda:0",
                    "--save-test-predictions",
                ]
                jobs.append((run_name, cmd))
    return jobs


def run_multiseed(args: argparse.Namespace, out_root: Path, plm_path: Path, cwd: Path) -> None:
    jobs = multiseed_jobs(args, out_root, plm_path)
    if not jobs:
        print("[multiseed] no jobs to run")
        return
    gpus = [gpu.strip() for gpu in args.gpu_ids.split(",") if gpu.strip()]
    print(f"[multiseed] {len(jobs)} jobs across GPUs {gpus}")

    def worker(item: tuple[int, tuple[str, list[str]]]) -> str:
        idx, (name, cmd) = item
        gpu = gpus[idx % len(gpus)]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        run_cmd(cmd, cwd, env=env, dry_run=args.dry_run)
        return name

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures = [pool.submit(worker, item) for item in enumerate(jobs)]
        for future in concurrent.futures.as_completed(futures):
            print("[multiseed done]", future.result(), flush=True)
    run_cmd(
        [
            args.python,
            "scripts/collect_task_mode_results.py",
            "--input-root",
            str(out_root / "multiseed"),
            "--output-tsv",
            str(out_root / "multiseed" / "suite_summary.tsv"),
            "--output-json",
            str(out_root / "multiseed" / "suite_summary.json"),
        ],
        cwd,
        dry_run=args.dry_run,
    )


def main() -> None:
    args = parse_args()
    cwd = Path.cwd()
    run_root = args.run_root.resolve()
    out_root = (args.output_root or run_root / "review_completion").resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    qc_dir = (args.qc_dir or run_root / "qc_review").resolve()
    assets_dir = (args.assets_dir or out_root / "manuscript_assets").resolve()
    plm_path = infer_plm_path(args)

    if not args.skip_multiseed:
        run_multiseed(args, out_root, plm_path, cwd)

    if not args.skip_source_ci:
        run_cmd(
            [
                args.python,
                "scripts/bootstrap_source_addback_ci.py",
                "--run-root",
                str(run_root),
                "--input",
                str(args.input),
                "--split-manifest",
                str(args.split_manifest),
                "--output-dir",
                str(qc_dir),
                "--bootstrap-iterations",
                str(args.bootstrap_iterations),
                "--device",
                "cuda:0",
            ],
            cwd,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": args.gpu_ids.split(",")[0]},
            dry_run=args.dry_run,
        )

    if not args.skip_homology:
        if not args.dry_run and not shutil.which(args.mmseqs_bin) and not Path(args.mmseqs_bin).expanduser().exists():
            raise SystemExit(
                "MMseqs2 executable was not found. Install it with "
                "`conda install -c conda-forge -c bioconda mmseqs2`, set "
                "`MMSEQS_BIN=/path/to/mmseqs`, pass `--mmseqs-bin /path/to/mmseqs`, "
                "or resume without the homology baseline using `--skip-homology`."
            )
        run_cmd(
            [
                args.python,
                "scripts/run_homology_label_transfer.py",
                "--protein-index",
                str(args.input),
                "--split-manifest",
                str(args.split_manifest),
                "--freeze-dir",
                str(args.freeze_dir),
                "--output-dir",
                str(out_root / "homology_baseline"),
                "--threads",
                str(args.threads),
                "--mmseqs-bin",
                str(args.mmseqs_bin),
            ],
            cwd,
            dry_run=args.dry_run,
        )

    if not args.skip_nucleocapsid_sensitivity:
        run_cmd(
            [
                args.python,
                "scripts/nucleocapsid_synonym_sensitivity.py",
                "--run-root",
                str(run_root),
                "--input",
                str(args.input),
                "--split-manifest",
                str(args.split_manifest),
                "--output-dir",
                str(qc_dir),
                "--device",
                "cuda:0",
            ],
            cwd,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": args.gpu_ids.split(",")[0]},
            dry_run=args.dry_run,
        )

    if not args.skip_supplement_package:
        if not assets_dir.exists():
            run_cmd(
                [
                    args.python,
                    "scripts/build_v2_manuscript_assets.py",
                    "--core-dir",
                    str(run_root),
                    "--qc-dir",
                    str(qc_dir),
                    "--output-dir",
                    str(assets_dir),
                ],
                cwd,
                dry_run=args.dry_run,
            )

    if not args.skip_candidate_evidence:
        candidate_path = assets_dir / "figure5_high_context_gain_candidates.tsv"
        if not candidate_path.exists():
            candidate_path = out_root / "supplementary_package" / "supplementary_tables" / "S16_high_context_gain_candidates.tsv"
        run_cmd(
            [
                args.python,
                "scripts/build_candidate_case_evidence.py",
                "--candidates",
                str(candidate_path),
                "--protein-index",
                str(args.input),
                "--split-manifest",
                str(args.split_manifest),
                "--module-candidates",
                str(run_root / "module_discovery" / "module_candidates.tsv"),
                "--output-dir",
                str(out_root / "candidate_case_evidence"),
            ],
            cwd,
            dry_run=args.dry_run,
        )

    if not args.skip_supplement_package:
        run_cmd(
            [
                args.python,
                "scripts/make_v2_supplementary_package.py",
                "--core-dir",
                str(run_root),
                "--qc-dir",
                str(qc_dir),
                "--assets-dir",
                str(assets_dir),
                "--output-dir",
                str(out_root / "supplementary_package"),
                "--protein-index",
                str(args.input),
                "--split-manifest",
                str(args.split_manifest),
                "--candidate-evidence-dir",
                str(out_root / "candidate_case_evidence"),
                "--make-zip",
            ],
            cwd,
            dry_run=args.dry_run,
        )

    summary = {
        "run_root": str(run_root),
        "output_root": str(out_root),
        "qc_dir": str(qc_dir),
        "assets_dir": str(assets_dir),
        "plm_embedding_path": str(plm_path),
    }
    (out_root / "review_completion_manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
