from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


TRAIN_COMPLETE_FILES = (
    "run_manifest.json",
    "metrics_summary.json",
    "best_model.pt",
    "best_thresholds.json",
    "val_label_metrics.tsv",
    "test_label_metrics.tsv",
)


@dataclass
class Job:
    name: str
    command: list[str]
    log_path: Path
    done_paths: tuple[Path, ...] = field(default_factory=tuple)
    env: dict[str, str] = field(default_factory=dict)
    gpu_id: str | None = None


@dataclass
class RunningJob:
    job: Job
    process: subprocess.Popen
    log_handle: object
    started_at: float


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def timestamp_slug() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_csv(value: str) -> list[str]:
    return [token.strip() for token in value.split(",") if token.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V2 paper-grade orchestration with GPU 4/5/6 parallel scheduling.")
    parser.add_argument("--output-root", default="", help="Defaults to runs/context_study_v2_<timestamp>")
    parser.add_argument("--input", default="data/processed/training/viral_protein_training_index.tsv.gz")
    parser.add_argument("--split-manifest", default="data/processed/splits/viral_protein_strict_splits.tsv.gz")
    parser.add_argument("--freeze-dir", default="data/v2_freeze")
    parser.add_argument("--gpu-ids", default="4,5,6", help="Physical GPU ids used as one-job-per-GPU workers")
    parser.add_argument("--max-parallel", type=int, default=0, help="Defaults to the number of GPU ids")
    parser.add_argument("--splits", default="family_holdout,host_holdout")
    parser.add_argument("--include-default-split", dest="include_default_split", action="store_true")
    parser.add_argument("--skip-default-split", dest="include_default_split", action="store_false")
    parser.add_argument("--skip-data-freeze", action="store_true")
    parser.add_argument("--skip-split-difficulty-audit", action="store_true")
    parser.add_argument("--skip-plm-cache", action="store_true")
    parser.add_argument("--skip-cnn-baselines", action="store_true")
    parser.add_argument("--skip-main-grid", action="store_true")
    parser.add_argument("--skip-source-decomposition", action="store_true")
    parser.add_argument("--skip-controls", action="store_true")
    parser.add_argument("--skip-atlas", action="store_true")
    parser.add_argument("--skip-calibration", action="store_true")
    parser.add_argument("--skip-discovery", action="store_true")
    parser.add_argument("--include-annotation-refinement", action="store_true")
    parser.add_argument("--rerun-completed", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--model-name", default="facebook/esm2_t33_650M_UR50D")
    parser.add_argument("--plm-embedding-path", default="data/processed/plm/esm2_t33_650M_UR50D_embeddings.pt")
    parser.add_argument("--cache-device", default="cuda:0")
    parser.add_argument("--cache-batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--neighbor-radius", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--min-label-count", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--warmup-fraction", type=float, default=0.05)
    parser.add_argument("--max-pos-weight", type=float, default=50.0)
    parser.add_argument("--host-corruption-fractions", default="0.25,0.5,0.75,1.0")
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--permutation-iterations", type=int, default=1000)
    parser.add_argument("--calibration-runs", default="genome_aware_denovo.family_holdout,protein_only.family_holdout")
    parser.add_argument("--discovery-run-name", default="genome_aware_denovo.family_holdout")
    parser.add_argument("--debug-limit", type=int, default=0)
    parser.set_defaults(include_default_split=True)
    return parser.parse_args()


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    return (root / path).resolve() if not path.is_absolute() else path.resolve()


def command_text(command: Iterable[str]) -> str:
    return " ".join(str(part) for part in command)


def path_is_complete(path: Path) -> bool:
    return path.exists()


def train_run_is_complete(run_dir: Path) -> bool:
    if not run_dir.exists():
        return False
    if not all((run_dir / name).exists() for name in TRAIN_COMPLETE_FILES):
        return False
    status_path = run_dir / "run_status.json"
    if status_path.exists():
        try:
            payload = json.loads(status_path.read_text(encoding="utf-8"))
            return payload.get("stage") == "completed"
        except json.JSONDecodeError:
            return False
    return True


def atlas_is_complete(output_dir: Path) -> bool:
    return all((output_dir / name).exists() for name in ("atlas_report.json", "label_deltas.tsv", "group_summary.tsv"))


def uncertainty_is_complete(output_dir: Path) -> bool:
    return (output_dir / "uncertainty_report.json").exists()


def maybe_run(
    *,
    name: str,
    command: list[str],
    log_path: Path,
    cwd: Path,
    dry_run: bool,
    env: dict[str, str] | None = None,
    done_paths: tuple[Path, ...] = (),
    rerun_completed: bool = False,
) -> None:
    if done_paths and not rerun_completed and all(path_is_complete(path) for path in done_paths):
        print(f"[skip] {name} already complete")
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[run] {name}")
    print(f"[run] log={log_path}")
    print(f"[run] cmd={command_text(command)}")
    if dry_run:
        return
    with log_path.open("a", encoding="utf-8", newline="") as handle:
        handle.write(f"[v2] started_at={utc_now()}\n")
        handle.write(f"[v2] cmd={command_text(command)}\n")
        handle.flush()
        subprocess.run(command, cwd=cwd, check=True, env=env, stdout=handle, stderr=subprocess.STDOUT)
        handle.write(f"[v2] finished_at={utc_now()}\n")


def run_jobs_parallel(
    jobs: list[Job],
    *,
    root: Path,
    gpu_ids: list[str],
    max_parallel: int,
    dry_run: bool,
    rerun_completed: bool,
) -> None:
    pending = list(jobs)
    active: dict[str, RunningJob] = {}
    failed: list[tuple[Job, int]] = []

    if dry_run:
        for job in pending:
            print(f"[dry-run job] {job.name}")
            print(f"[dry-run job] log={job.log_path}")
            print(f"[dry-run job] cmd={command_text(job.command)}")
        return

    while pending or active:
        free_gpus = [gpu for gpu in gpu_ids if gpu not in active]
        while pending and free_gpus and len(active) < max_parallel:
            job = pending.pop(0)
            if job.done_paths and not rerun_completed and all(path_is_complete(path) for path in job.done_paths):
                print(f"[skip] {job.name} already complete")
                continue

            gpu_id = free_gpus.pop(0)
            job.gpu_id = gpu_id
            env = os.environ.copy()
            env.update(job.env)
            env["CUDA_VISIBLE_DEVICES"] = gpu_id
            env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
            job.log_path.parent.mkdir(parents=True, exist_ok=True)
            handle = job.log_path.open("a", encoding="utf-8", newline="")
            handle.write(f"[v2] job={job.name}\n")
            handle.write(f"[v2] started_at={utc_now()}\n")
            handle.write(f"[v2] physical_gpu={gpu_id}\n")
            handle.write(f"[v2] CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}\n")
            handle.write(f"[v2] cmd={command_text(job.command)}\n")
            handle.flush()
            print(f"[start] gpu={gpu_id} {job.name}")
            process = subprocess.Popen(job.command, cwd=root, env=env, stdout=handle, stderr=subprocess.STDOUT)
            active[gpu_id] = RunningJob(job=job, process=process, log_handle=handle, started_at=time.time())

        time.sleep(5)
        for gpu_id, running in list(active.items()):
            return_code = running.process.poll()
            if return_code is None:
                continue
            elapsed = time.time() - running.started_at
            running.log_handle.write(f"[v2] finished_at={utc_now()}\n")
            running.log_handle.write(f"[v2] return_code={return_code}\n")
            running.log_handle.write(f"[v2] elapsed_seconds={elapsed:.1f}\n")
            running.log_handle.close()
            active.pop(gpu_id)
            if return_code != 0:
                failed.append((running.job, int(return_code)))
                print(f"[fail] gpu={gpu_id} {running.job.name} rc={return_code}")
            else:
                print(f"[done] gpu={gpu_id} {running.job.name} elapsed={elapsed:.1f}s")

        if failed:
            for running in active.values():
                running.process.terminate()
                running.log_handle.close()
            details = ", ".join(f"{job.name}: {code}" for job, code in failed)
            raise RuntimeError(f"One or more jobs failed: {details}")


def train_command(args: argparse.Namespace, run_dir: Path, split: str, task_mode: str, extra: list[str], backbone: str) -> list[str]:
    command = [
        sys.executable,
        "-u",
        "scripts/train_task_modes.py",
        "--input",
        args.input,
        "--split-manifest",
        args.split_manifest,
        "--split-scheme",
        split,
        "--task-mode",
        task_mode,
        "--output-dir",
        str(run_dir),
        "--device",
        "cuda:0",
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--eval-batch-size",
        str(args.eval_batch_size),
        "--embed-dim",
        str(args.embed_dim),
        "--hidden-dim",
        str(args.hidden_dim),
        "--dropout",
        str(args.dropout),
        "--learning-rate",
        str(args.learning_rate),
        "--weight-decay",
        str(args.weight_decay),
        "--num-workers",
        str(args.num_workers),
        "--prefetch-factor",
        str(args.prefetch_factor),
        "--neighbor-radius",
        str(args.neighbor_radius),
        "--max-length",
        str(args.max_length),
        "--min-label-count",
        str(args.min_label_count),
        "--seed",
        str(args.seed),
        "--gradient-clip",
        str(args.gradient_clip),
        "--warmup-fraction",
        str(args.warmup_fraction),
        "--max-pos-weight",
        str(args.max_pos_weight),
        "--save-test-predictions",
    ]
    if args.debug_limit:
        command.extend(["--debug-limit", str(args.debug_limit)])
    if backbone == "precomputed_plm":
        command.extend(["--sequence-backbone", "precomputed_plm", "--plm-embedding-path", args.plm_embedding_path])
    command.extend(extra)
    return command


def training_job(
    args: argparse.Namespace,
    output_root: Path,
    logs_dir: Path,
    run_name: str,
    split: str,
    task_mode: str,
    extra: list[str],
    backbone: str = "precomputed_plm",
) -> Job:
    run_dir = output_root / f"{run_name}.{split}"
    return Job(
        name=f"train.{run_name}.{split}",
        command=train_command(args, run_dir, split, task_mode, extra, backbone),
        log_path=logs_dir / f"train.{run_name}.{split}.log",
        done_paths=tuple(run_dir / name for name in TRAIN_COMPLETE_FILES),
    )


def build_training_jobs(args: argparse.Namespace, output_root: Path, logs_dir: Path, splits: list[str]) -> list[Job]:
    jobs: list[Job] = []
    if args.include_default_split:
        if not args.skip_cnn_baselines:
            jobs.append(training_job(args, output_root, logs_dir, "cnn_protein_only", "default_hash", "protein_only", [], "cnn"))
        jobs.append(training_job(args, output_root, logs_dir, "protein_only", "default_hash", "protein_only", [], "precomputed_plm"))

    for split in splits:
        if not args.skip_cnn_baselines:
            jobs.append(training_job(args, output_root, logs_dir, "cnn_protein_only", split, "protein_only", [], "cnn"))
            jobs.append(training_job(args, output_root, logs_dir, "cnn_genome_aware_denovo", split, "genome_aware_denovo", [], "cnn"))

        if not args.skip_main_grid:
            main_runs = [
                ("protein_only", "protein_only", []),
                ("protein_only_biophysics", "protein_only", ["--with-biophysics"]),
                ("genome_aware_denovo", "genome_aware_denovo", []),
                ("genome_aware_denovo_biophysics", "genome_aware_denovo", ["--with-biophysics"]),
            ]
            if args.include_annotation_refinement:
                main_runs.append(("annotation_refinement", "annotation_refinement", ["--with-biophysics"]))
            for run_name, task_mode, extra in main_runs:
                jobs.append(training_job(args, output_root, logs_dir, run_name, split, task_mode, extra))

        if not args.skip_source_decomposition:
            addbacks = [
                ("genome_aware_denovo_addback_local_only", ["--context-blocks", "local_neighborhood"]),
                ("genome_aware_denovo_addback_genome_only", ["--context-blocks", "genome_organization"]),
                ("genome_aware_denovo_addback_host_only", ["--context-blocks", "host_metadata"]),
                ("genome_aware_denovo_addback_local_genome", ["--context-blocks", "local_neighborhood,genome_organization"]),
                ("genome_aware_denovo_addback_local_host", ["--context-blocks", "local_neighborhood,host_metadata"]),
                ("genome_aware_denovo_addback_genome_host", ["--context-blocks", "genome_organization,host_metadata"]),
            ]
            for run_name, extra in addbacks:
                jobs.append(training_job(args, output_root, logs_dir, run_name, split, "genome_aware_denovo", extra))

            ablations = [
                ("genome_aware_denovo_biophysics_minus_local", ["--with-biophysics", "--context-blocks", "genome_organization,host_metadata"]),
                ("genome_aware_denovo_biophysics_minus_genome_org", ["--with-biophysics", "--context-blocks", "local_neighborhood,host_metadata"]),
                ("genome_aware_denovo_biophysics_minus_host", ["--with-biophysics", "--context-blocks", "local_neighborhood,genome_organization"]),
            ]
            for run_name, extra in ablations:
                jobs.append(training_job(args, output_root, logs_dir, run_name, split, "genome_aware_denovo", extra))

        if not args.skip_controls:
            controls = [
                ("genome_aware_denovo_biophysics_control_local_shuffle", "shuffle_local_order"),
                ("genome_aware_denovo_biophysics_control_host_shuffle", "shuffle_host_within_family"),
                ("genome_aware_denovo_biophysics_control_position_shuffle", "shuffle_genome_relative_position"),
            ]
            for run_name, control in controls:
                jobs.append(
                    training_job(
                        args,
                        output_root,
                        logs_dir,
                        run_name,
                        split,
                        "genome_aware_denovo",
                        ["--with-biophysics", "--context-control", control],
                    )
                )

            for fraction in parse_csv(args.host_corruption_fractions):
                percentage = int(round(float(fraction) * 100.0))
                run_name = f"genome_aware_denovo_biophysics_host_corrupt_{percentage:02d}"
                jobs.append(
                    training_job(
                        args,
                        output_root,
                        logs_dir,
                        run_name,
                        split,
                        "genome_aware_denovo",
                        ["--with-biophysics", "--host-corruption-fraction", str(float(fraction))],
                    )
                )
    return jobs


def context_build_jobs(args: argparse.Namespace, output_root: Path, logs_dir: Path, splits: list[str]) -> list[Job]:
    jobs: list[Job] = []
    task_modes = ["genome_aware_denovo"]
    if args.include_annotation_refinement:
        task_modes.append("annotation_refinement")
    for split in splits:
        if split == "default_hash":
            continue
        for task_mode in task_modes:
            out_path = repo_root() / f"data/processed/context/viral_protein_context.{split}.{task_mode}.tsv.gz"
            command = [
                sys.executable,
                "-u",
                "scripts/build_context_features_splitaware.py",
                "--input",
                args.input,
                "--split-manifest",
                args.split_manifest,
                "--split-scheme",
                split,
                "--task-mode",
                task_mode,
                "--output-dir",
                "data/processed/context",
            ]
            if args.debug_limit:
                command.extend(["--debug-limit", str(args.debug_limit)])
            jobs.append(
                Job(
                    name=f"build_context.{split}.{task_mode}",
                    command=command,
                    log_path=logs_dir / f"build_context.{split}.{task_mode}.log",
                    done_paths=(out_path,),
                )
            )
    return jobs


def write_manifest(output_root: Path, args: argparse.Namespace, splits: list[str], gpu_ids: list[str]) -> None:
    manifest = {
        "created_at": utc_now(),
        "output_root": str(output_root),
        "splits": splits,
        "gpu_ids": gpu_ids,
        "plm_embedding_path": args.plm_embedding_path,
        "model_name": args.model_name,
        "host_corruption_fractions": parse_csv(args.host_corruption_fractions),
        "include_default_split": bool(args.include_default_split),
        "include_annotation_refinement": bool(args.include_annotation_refinement),
        "skip_cnn_baselines": bool(args.skip_cnn_baselines),
        "command_args": vars(args),
    }
    (output_root / "v2_suite_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    args = parse_args()
    root = repo_root()
    output_root = resolve_path(root, args.output_root) if args.output_root else root / "runs" / f"context_study_v2_{timestamp_slug()}"
    logs_dir = output_root / "logs"
    output_root.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    gpu_ids = parse_csv(args.gpu_ids)
    if not gpu_ids:
        raise ValueError("--gpu-ids must contain at least one GPU id")
    max_parallel = args.max_parallel or len(gpu_ids)
    max_parallel = max(1, min(max_parallel, len(gpu_ids)))
    splits = parse_csv(args.splits)
    if not splits:
        raise ValueError("--splits must contain at least one split")

    write_manifest(output_root, args, splits, gpu_ids)
    print(json.dumps({"output_root": str(output_root), "gpu_ids": gpu_ids, "max_parallel": max_parallel}, indent=2))

    setup_env = os.environ.copy()
    setup_env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    if not args.skip_data_freeze:
        freeze_command = [
            sys.executable,
            "-u",
            "scripts/freeze_benchmark_v2.py",
            "--protein-index",
            args.input,
            "--strict-splits",
            args.split_manifest,
            "--output-dir",
            args.freeze_dir,
            "--min-label-count",
            str(args.min_label_count),
        ]
        if args.debug_limit:
            freeze_command.extend(["--debug-limit", str(args.debug_limit)])
        maybe_run(
            name="freeze_benchmark_v2",
            command=freeze_command,
            log_path=logs_dir / "freeze_benchmark_v2.log",
            cwd=root,
            dry_run=args.dry_run,
            env=setup_env,
            done_paths=(resolve_path(root, args.freeze_dir) / "checksums.tsv",),
            rerun_completed=args.rerun_completed,
        )

    maybe_run(
        name="audit_feature_leakage",
        command=[sys.executable, "-u", "scripts/audit_feature_leakage.py", "--input", args.input, "--output-dir", "data/audits"],
        log_path=logs_dir / "audit_feature_leakage.log",
        cwd=root,
        dry_run=args.dry_run,
        env=setup_env,
        done_paths=(root / "data/audits/feature_leakage_audit.tsv", root / "data/audits/feature_leakage_audit.json"),
        rerun_completed=args.rerun_completed,
    )

    if not args.skip_split_difficulty_audit:
        split_audit_dir = output_root / "split_difficulty"
        split_audit_command = [
            sys.executable,
            "-u",
            "scripts/audit_v2_split_difficulty.py",
            "--input",
            args.input,
            "--split-manifest",
            args.split_manifest,
            "--output-dir",
            str(split_audit_dir),
        ]
        if args.debug_limit:
            split_audit_command.extend(["--debug-limit", str(args.debug_limit)])
        maybe_run(
            name="audit_v2_split_difficulty",
            command=split_audit_command,
            log_path=logs_dir / "audit_v2_split_difficulty.log",
            cwd=root,
            dry_run=args.dry_run,
            env=setup_env,
            done_paths=(split_audit_dir / "split_difficulty_report.json",),
            rerun_completed=args.rerun_completed,
        )

    if not resolve_path(root, args.split_manifest).exists():
        maybe_run(
            name="build_strict_splits",
            command=[sys.executable, "-u", "scripts/build_strict_splits.py", "--output-dir", str(resolve_path(root, args.split_manifest).parent)],
            log_path=logs_dir / "build_strict_splits.log",
            cwd=root,
            dry_run=args.dry_run,
            env=setup_env,
            done_paths=(resolve_path(root, args.split_manifest),),
            rerun_completed=args.rerun_completed,
        )

    context_jobs = context_build_jobs(args, output_root, logs_dir, splits)
    for job in context_jobs:
        maybe_run(
            name=job.name,
            command=job.command,
            log_path=job.log_path,
            cwd=root,
            dry_run=args.dry_run,
            env=setup_env,
            done_paths=job.done_paths,
            rerun_completed=args.rerun_completed,
        )

    plm_path = resolve_path(root, args.plm_embedding_path)
    if not args.skip_plm_cache:
        cache_env = setup_env.copy()
        cache_env["CUDA_VISIBLE_DEVICES"] = gpu_ids[0]
        cache_device = "cuda:0" if args.cache_device.startswith("cuda") else args.cache_device
        cache_command = [
            sys.executable,
            "-u",
            "scripts/cache_plm_embeddings.py",
            "--input",
            args.input,
            "--output",
            args.plm_embedding_path,
            "--model-name",
            args.model_name,
            "--device",
            cache_device,
            "--batch-size",
            str(args.cache_batch_size),
        ]
        if args.debug_limit:
            cache_command.extend(["--max-records", str(args.debug_limit)])
        maybe_run(
            name="cache_plm_embeddings",
            command=cache_command,
            log_path=logs_dir / "cache_plm_embeddings.log",
            cwd=root,
            dry_run=args.dry_run,
            env=cache_env,
            done_paths=(plm_path,),
            rerun_completed=args.rerun_completed,
        )

    training_jobs = build_training_jobs(args, output_root, logs_dir, splits)
    needs_plm_embeddings = any("--sequence-backbone" in job.command for job in training_jobs)
    if needs_plm_embeddings and not args.dry_run and not plm_path.exists():
        raise FileNotFoundError(
            f"PLM embedding file was not found: {plm_path}. "
            "Remove --skip-plm-cache or pass --plm-embedding-path to an existing embedding file."
        )
    run_jobs_parallel(
        training_jobs,
        root=root,
        gpu_ids=gpu_ids,
        max_parallel=max_parallel,
        dry_run=args.dry_run,
        rerun_completed=args.rerun_completed,
    )

    maybe_run(
        name="collect_task_mode_results",
        command=[sys.executable, "-u", "scripts/collect_task_mode_results.py", "--input-root", str(output_root)],
        log_path=logs_dir / "collect_task_mode_results.log",
        cwd=root,
        dry_run=args.dry_run,
        env=setup_env,
        done_paths=(output_root / "suite_summary.json", output_root / "suite_summary.tsv"),
        rerun_completed=True,
    )

    if not args.skip_source_decomposition:
        maybe_run(
            name="summarize_source_decomposition",
            command=[sys.executable, "-u", "scripts/summarize_source_decomposition.py", "--input", str(output_root), "--output-dir", str(output_root / "source_decomposition")],
            log_path=logs_dir / "summarize_source_decomposition.log",
            cwd=root,
            dry_run=args.dry_run,
            env=setup_env,
            done_paths=(output_root / "source_decomposition/source_decomposition_summary.tsv",),
            rerun_completed=args.rerun_completed,
        )

    if not args.skip_atlas:
        for split in splits:
            atlas_specs = [
                ("plain", output_root / f"protein_only.{split}", output_root / f"genome_aware_denovo.{split}"),
                ("biophysics", output_root / f"protein_only_biophysics.{split}", output_root / f"genome_aware_denovo_biophysics.{split}"),
            ]
            for atlas_name, protein_run, context_run in atlas_specs:
                out_dir = output_root / f"context_atlas_{atlas_name}.{split}.v2"
                if args.dry_run or (train_run_is_complete(protein_run) and train_run_is_complete(context_run)):
                    maybe_run(
                        name=f"atlas.{atlas_name}.{split}",
                        command=[
                            sys.executable,
                            "-u",
                            "scripts/build_context_dependence_atlas_v2.py",
                            "--protein-run",
                            str(protein_run),
                            "--context-run",
                            str(context_run),
                            "--input",
                            args.input,
                            "--output-dir",
                            str(out_dir),
                            "--bootstrap-iterations",
                            str(args.bootstrap_iterations),
                            "--permutation-iterations",
                            str(args.permutation_iterations),
                        ],
                        log_path=logs_dir / f"atlas.{atlas_name}.{split}.log",
                        cwd=root,
                        dry_run=args.dry_run,
                        env=setup_env,
                        done_paths=(out_dir / "atlas_report.json", out_dir / "label_deltas.tsv", out_dir / "group_summary.tsv"),
                        rerun_completed=args.rerun_completed,
                    )
                else:
                    print(f"[skip] atlas.{atlas_name}.{split} waiting for upstream training outputs")

    if not args.skip_calibration:
        calibration_jobs: list[Job] = []
        for run_name in parse_csv(args.calibration_runs):
            run_dir = output_root / run_name
            out_dir = output_root / "uncertainty" / run_name
            if args.dry_run or train_run_is_complete(run_dir):
                calibration_jobs.append(
                    Job(
                        name=f"calibration.{run_name}",
                        command=[
                            sys.executable,
                            "-u",
                            "scripts/calibrate_task_mode_uncertainty.py",
                            "--run-dir",
                            str(run_dir),
                            "--output-dir",
                            str(out_dir),
                            "--device",
                            "cuda:0",
                        ],
                        log_path=logs_dir / f"calibration.{run_name}.log",
                        done_paths=(out_dir / "uncertainty_report.json",),
                    )
                )
            else:
                print(f"[skip] calibration.{run_name} waiting for upstream training output")
        run_jobs_parallel(
            calibration_jobs,
            root=root,
            gpu_ids=gpu_ids,
            max_parallel=max_parallel,
            dry_run=args.dry_run,
            rerun_completed=args.rerun_completed,
        )

    if not args.skip_discovery:
        discovery_run_dir = output_root / args.discovery_run_name
        discovery_root = output_root / "module_discovery"
        embedding_export = discovery_root / "exported_fused_test_embeddings.pt"
        if args.dry_run or train_run_is_complete(discovery_run_dir):
            discovery_env = setup_env.copy()
            discovery_env["CUDA_VISIBLE_DEVICES"] = gpu_ids[0]
            maybe_run(
                name="export_task_mode_embeddings",
                command=[
                    sys.executable,
                    "-u",
                    "scripts/export_task_mode_embeddings.py",
                    "--run-dir",
                    str(discovery_run_dir),
                    "--output",
                    str(embedding_export),
                    "--representation",
                    "fused",
                    "--split",
                    "test",
                    "--device",
                    "cuda:0",
                ],
                log_path=logs_dir / "export_task_mode_embeddings.log",
                cwd=root,
                dry_run=args.dry_run,
                env=discovery_env,
                done_paths=(embedding_export,),
                rerun_completed=args.rerun_completed,
            )
            maybe_run(
                name="discover_module_candidates",
                command=[
                    sys.executable,
                    "-u",
                    "scripts/discover_module_candidates.py",
                    "--embedding-file",
                    str(embedding_export),
                    "--output-dir",
                    str(discovery_root),
                ],
                log_path=logs_dir / "discover_module_candidates.log",
                cwd=root,
                dry_run=args.dry_run,
                env=setup_env,
                done_paths=(discovery_root / "module_discovery_report.json",),
                rerun_completed=args.rerun_completed,
            )
            ranked = discovery_root / "ranked_hypothetical_clusters.tsv"
            candidates = discovery_root / "module_candidates.tsv"
            if args.dry_run or (ranked.exists() and candidates.exists()):
                maybe_run(
                    name="prepare_targeted_structure_validation",
                    command=[
                        sys.executable,
                        "-u",
                        "scripts/prepare_targeted_structure_validation.py",
                        "--ranked-clusters",
                        str(ranked),
                        "--module-candidates",
                        str(candidates),
                        "--output-dir",
                        str(discovery_root / "targeted_structure_validation"),
                    ],
                    log_path=logs_dir / "prepare_targeted_structure_validation.log",
                    cwd=root,
                    dry_run=args.dry_run,
                    env=setup_env,
                    done_paths=(discovery_root / "targeted_structure_validation/structure_validation_prep.json",),
                    rerun_completed=args.rerun_completed,
                )
        else:
            print(f"[skip] discovery waiting for upstream run {discovery_run_dir.name}")

    maybe_run(
        name="freeze_result_registry",
        command=[
            sys.executable,
            "-u",
            "scripts/freeze_benchmark_v1.py",
            "--runs-root",
            str(output_root),
            "--output-tsv",
            str(output_root / "frozen_benchmark_v2_runs.tsv"),
            "--claims-ledger",
            str(output_root / "claims_ledger.md"),
            "--paper-numbers",
            str(output_root / "paper_numbers.json"),
        ],
        log_path=logs_dir / "freeze_result_registry.log",
        cwd=root,
        dry_run=args.dry_run,
        env=setup_env,
        done_paths=(output_root / "frozen_benchmark_v2_runs.tsv", output_root / "paper_numbers.json"),
        rerun_completed=True,
    )

    final_summary = {
        "finished_at": utc_now(),
        "output_root": str(output_root),
        "suite_summary": str(output_root / "suite_summary.tsv"),
        "data_freeze": str(resolve_path(root, args.freeze_dir)),
        "logs": str(logs_dir),
    }
    (output_root / "v2_suite_done.json").write_text(json.dumps(final_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(final_summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
