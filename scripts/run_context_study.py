from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def timestamp_slug() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.gmtime())


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the clean biophysics/context study suite.")
    parser.add_argument("--input", default="data/processed/training/viral_protein_training_index.tsv.gz")
    parser.add_argument("--split-manifest", default="data/processed/splits/viral_protein_strict_splits.tsv.gz")
    parser.add_argument("--splits", default="family_holdout,host_holdout", help="Comma-separated split schemes")
    parser.add_argument("--output-root", default="", help="Output directory. Defaults to runs/context_study_<timestamp>")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--cuda-visible-devices",
        default="",
        help=(
            "Optional CUDA_VISIBLE_DEVICES override for every spawned training process. "
            "For single-GPU runs, pass a single physical GPU id such as '3' or '7'."
        ),
    )
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=1024)
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
    parser.add_argument("--sequence-backbone", default="cnn", choices=("cnn", "precomputed_plm"))
    parser.add_argument("--plm-embedding-path", default="")
    parser.add_argument("--compile-model", action="store_true")
    parser.add_argument("--force-rebuild-cache", action="store_true")
    parser.set_defaults(save_test_predictions=True)
    parser.add_argument("--save-test-predictions", dest="save_test_predictions", action="store_true")
    parser.add_argument("--no-save-test-predictions", dest="save_test_predictions", action="store_false")
    parser.add_argument("--host-corruption-fractions", default="0.1,0.3,0.5,0.7")
    parser.add_argument("--skip-main-grid", action="store_true")
    parser.add_argument("--skip-ablations", action="store_true")
    parser.add_argument("--skip-addbacks", action="store_true")
    parser.add_argument("--skip-controls", action="store_true")
    parser.add_argument("--skip-host-corruption", action="store_true")
    parser.add_argument("--skip-atlas", action="store_true")
    parser.add_argument("--include-annotation-refinement", action="store_true")
    parser.add_argument(
        "--rerun-completed",
        action="store_true",
        help="Rerun steps even when their completion artifacts already exist. Default behavior is resume/skip completed work.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def parse_visible_devices(raw: str) -> list[str]:
    return [token.strip() for token in raw.split(",") if token.strip()]


def build_subprocess_env(args: argparse.Namespace) -> tuple[dict[str, str], str]:
    env = os.environ.copy()
    requested_device = args.device
    override = args.cuda_visible_devices.strip()
    if override:
        env["CUDA_VISIBLE_DEVICES"] = override
        if requested_device in {"auto", "cuda"}:
            requested_device = "cuda:0"
        elif requested_device.startswith("cuda:"):
            requested_device = "cuda:0"
    else:
        inherited = env.get("CUDA_VISIBLE_DEVICES", "").strip()
        tokens = parse_visible_devices(inherited)
        if requested_device == "auto" and len(tokens) == 1:
            requested_device = "cuda:0"
    return env, requested_device


def maybe_run(command: list[str], cwd: Path, dry_run: bool, env: dict[str, str] | None = None) -> None:
    pretty = " ".join(command)
    print(f"[run] {pretty}")
    if dry_run:
        return
    subprocess.run(command, cwd=cwd, check=True, env=env)


def load_json_if_exists(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def training_run_is_complete(run_dir: Path) -> bool:
    if not run_dir.exists():
        return False

    required_files = [
        run_dir / "run_manifest.json",
        run_dir / "metrics_summary.json",
        run_dir / "best_model.pt",
        run_dir / "best_thresholds.json",
        run_dir / "val_label_metrics.tsv",
        run_dir / "test_label_metrics.tsv",
    ]
    if not all(path.exists() for path in required_files):
        return False

    status_payload = load_json_if_exists(run_dir / "run_status.json")
    if status_payload and status_payload.get("stage") == "completed":
        return True

    metrics_payload = load_json_if_exists(run_dir / "metrics_summary.json")
    if not metrics_payload:
        return False
    return isinstance(metrics_payload.get("test"), dict)


def training_run_state(run_dir: Path) -> str:
    if not run_dir.exists():
        return "missing"
    if training_run_is_complete(run_dir):
        return "completed"
    if (run_dir / "dataset_cache.pt").exists():
        return "partial_with_cache"
    if any(run_dir.iterdir()):
        return "partial"
    return "missing"


def atlas_is_complete(output_dir: Path) -> bool:
    if not output_dir.exists():
        return False
    required_files = [
        output_dir / "atlas_report.json",
        output_dir / "label_deltas.tsv",
        output_dir / "group_summary.tsv",
    ]
    return all(path.exists() for path in required_files)


def audit_is_complete(root: Path) -> bool:
    audit_dir = root / "data" / "audits"
    required_files = [
        audit_dir / "feature_leakage_audit.tsv",
        audit_dir / "feature_leakage_audit.json",
    ]
    return all(path.exists() for path in required_files)


def print_skip(message: str) -> None:
    print(f"[skip] {message}")


def print_resume(message: str) -> None:
    print(f"[resume] {message}")


def train_command(
    args: argparse.Namespace,
    resolved_device: str,
    output_dir: str,
    split_scheme: str,
    task_mode: str,
    extra_args: list[str],
) -> list[str]:
    command = [
        sys.executable,
        "scripts/train_task_modes.py",
        "--input",
        args.input,
        "--split-manifest",
        args.split_manifest,
        "--split-scheme",
        split_scheme,
        "--task-mode",
        task_mode,
        "--output-dir",
        output_dir,
        "--device",
        resolved_device,
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
    ]
    if args.sequence_backbone == "precomputed_plm":
        if not args.plm_embedding_path:
            raise ValueError("--plm-embedding-path is required when --sequence-backbone precomputed_plm")
        command.extend(["--sequence-backbone", "precomputed_plm", "--plm-embedding-path", args.plm_embedding_path])
    if args.compile_model:
        command.append("--compile-model")
    if args.force_rebuild_cache:
        command.append("--force-rebuild-cache")
    if args.save_test_predictions:
        command.append("--save-test-predictions")
    command.extend(extra_args)
    return command


def main() -> int:
    args = parse_args()
    root = repo_root()
    subprocess_env, resolved_device = build_subprocess_env(args)
    output_root = Path(args.output_root) if args.output_root else Path("runs") / f"context_study_{timestamp_slug()}"
    if not output_root.is_absolute():
        output_root = (root / output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    split_values = [value.strip() for value in args.splits.split(",") if value.strip()]
    host_corruption_fractions = [
        float(value.strip()) for value in args.host_corruption_fractions.split(",") if value.strip()
    ]

    strict_splits_path = (root / args.split_manifest).resolve()
    if not strict_splits_path.exists():
        maybe_run([sys.executable, "scripts/build_strict_splits.py"], root, args.dry_run, env=subprocess_env)

    if args.rerun_completed or not audit_is_complete(root):
        maybe_run(
            [sys.executable, "scripts/audit_feature_leakage.py", "--output-dir", "data/audits"],
            root,
            args.dry_run,
            env=subprocess_env,
        )
    else:
        print_skip("audit_feature_leakage is already complete")

    for split_scheme in split_values:
        split_prefix = split_scheme
        if not args.skip_main_grid:
            main_runs = [
                ("protein_only", "protein_only", []),
                ("protein_only_biophysics", "protein_only", ["--with-biophysics"]),
                ("genome_aware_denovo", "genome_aware_denovo", []),
                ("genome_aware_denovo_biophysics", "genome_aware_denovo", ["--with-biophysics"]),
            ]
            if args.include_annotation_refinement:
                main_runs.append(("annotation_refinement", "annotation_refinement", ["--with-biophysics"]))
            for run_name, task_mode, extra_args in main_runs:
                run_dir = output_root / f"{run_name}.{split_prefix}"
                state = training_run_state(run_dir)
                if not args.rerun_completed and state == "completed":
                    print_skip(f"training run {run_dir.name} is already complete")
                    continue
                if state in {"partial", "partial_with_cache"}:
                    detail = "reusing existing cache" if state == "partial_with_cache" else "rerunning incomplete output directory"
                    print_resume(f"training run {run_dir.name}: {detail}")
                maybe_run(
                    train_command(
                        args,
                        resolved_device,
                        str(run_dir),
                        split_scheme,
                        task_mode,
                        extra_args,
                    ),
                    root,
                    args.dry_run,
                    env=subprocess_env,
                )

        if not args.skip_ablations:
            ablations = [
                (
                    "genome_aware_denovo_biophysics_minus_local",
                    ["--with-biophysics", "--context-blocks", "genome_organization,host_metadata"],
                ),
                (
                    "genome_aware_denovo_biophysics_minus_genome_org",
                    ["--with-biophysics", "--context-blocks", "local_neighborhood,host_metadata"],
                ),
                (
                    "genome_aware_denovo_biophysics_minus_host",
                    ["--with-biophysics", "--context-blocks", "local_neighborhood,genome_organization"],
                ),
            ]
            for run_name, extra_args in ablations:
                run_dir = output_root / f"{run_name}.{split_prefix}"
                state = training_run_state(run_dir)
                if not args.rerun_completed and state == "completed":
                    print_skip(f"training run {run_dir.name} is already complete")
                    continue
                if state in {"partial", "partial_with_cache"}:
                    detail = "reusing existing cache" if state == "partial_with_cache" else "rerunning incomplete output directory"
                    print_resume(f"training run {run_dir.name}: {detail}")
                maybe_run(
                    train_command(
                        args,
                        resolved_device,
                        str(run_dir),
                        split_scheme,
                        "genome_aware_denovo",
                        extra_args,
                    ),
                    root,
                    args.dry_run,
                    env=subprocess_env,
                )

        if not args.skip_addbacks:
            addbacks = [
                ("genome_aware_denovo_addback_local_only", ["--context-blocks", "local_neighborhood"]),
                ("genome_aware_denovo_addback_genome_only", ["--context-blocks", "genome_organization"]),
                ("genome_aware_denovo_addback_host_only", ["--context-blocks", "host_metadata"]),
                ("genome_aware_denovo_addback_local_genome", ["--context-blocks", "local_neighborhood,genome_organization"]),
                ("genome_aware_denovo_addback_local_host", ["--context-blocks", "local_neighborhood,host_metadata"]),
                ("genome_aware_denovo_addback_genome_host", ["--context-blocks", "genome_organization,host_metadata"]),
            ]
            for run_name, extra_args in addbacks:
                run_dir = output_root / f"{run_name}.{split_prefix}"
                state = training_run_state(run_dir)
                if not args.rerun_completed and state == "completed":
                    print_skip(f"training run {run_dir.name} is already complete")
                    continue
                if state in {"partial", "partial_with_cache"}:
                    detail = "reusing existing cache" if state == "partial_with_cache" else "rerunning incomplete output directory"
                    print_resume(f"training run {run_dir.name}: {detail}")
                maybe_run(
                    train_command(
                        args,
                        resolved_device,
                        str(run_dir),
                        split_scheme,
                        "genome_aware_denovo",
                        extra_args,
                    ),
                    root,
                    args.dry_run,
                    env=subprocess_env,
                )

        if not args.skip_controls:
            controls = [
                ("genome_aware_denovo_biophysics_control_local_shuffle", "shuffle_local_order"),
                ("genome_aware_denovo_biophysics_control_host_shuffle", "shuffle_host_within_family"),
                ("genome_aware_denovo_biophysics_control_position_shuffle", "shuffle_genome_relative_position"),
            ]
            for run_name, control_name in controls:
                run_dir = output_root / f"{run_name}.{split_prefix}"
                state = training_run_state(run_dir)
                if not args.rerun_completed and state == "completed":
                    print_skip(f"training run {run_dir.name} is already complete")
                    continue
                if state in {"partial", "partial_with_cache"}:
                    detail = "reusing existing cache" if state == "partial_with_cache" else "rerunning incomplete output directory"
                    print_resume(f"training run {run_dir.name}: {detail}")
                maybe_run(
                    train_command(
                        args,
                        resolved_device,
                        str(run_dir),
                        split_scheme,
                        "genome_aware_denovo",
                        ["--with-biophysics", "--context-control", control_name],
                    ),
                    root,
                    args.dry_run,
                    env=subprocess_env,
                )

        if not args.skip_host_corruption:
            for fraction in host_corruption_fractions:
                percentage = int(round(fraction * 100))
                run_name = f"genome_aware_denovo_biophysics_host_corrupt_{percentage:02d}"
                run_dir = output_root / f"{run_name}.{split_prefix}"
                state = training_run_state(run_dir)
                if not args.rerun_completed and state == "completed":
                    print_skip(f"training run {run_dir.name} is already complete")
                    continue
                if state in {"partial", "partial_with_cache"}:
                    detail = "reusing existing cache" if state == "partial_with_cache" else "rerunning incomplete output directory"
                    print_resume(f"training run {run_dir.name}: {detail}")
                maybe_run(
                    train_command(
                        args,
                        resolved_device,
                        str(run_dir),
                        split_scheme,
                        "genome_aware_denovo",
                        ["--with-biophysics", "--host-corruption-fraction", str(fraction)],
                    ),
                    root,
                    args.dry_run,
                    env=subprocess_env,
                )

        if not args.skip_atlas:
            atlas_pairs = [
                (
                    output_root / f"protein_only.{split_prefix}",
                    output_root / f"genome_aware_denovo.{split_prefix}",
                    output_root / f"context_atlas_plain.{split_prefix}",
                ),
                (
                    output_root / f"protein_only_biophysics.{split_prefix}",
                    output_root / f"genome_aware_denovo_biophysics.{split_prefix}",
                    output_root / f"context_atlas_biophysics.{split_prefix}",
                ),
            ]
            for protein_run, context_run, atlas_output in atlas_pairs:
                if not args.dry_run and not (training_run_is_complete(protein_run) and training_run_is_complete(context_run)):
                    print_skip(
                        f"atlas {atlas_output.name} is waiting for completed upstream runs "
                        f"({protein_run.name}, {context_run.name})"
                    )
                    continue
                if not args.rerun_completed and atlas_is_complete(atlas_output):
                    print_skip(f"atlas {atlas_output.name} is already complete")
                    continue
                maybe_run(
                    [
                        sys.executable,
                        "scripts/build_context_dependence_atlas_v2.py",
                        "--protein-run",
                        str(protein_run),
                        "--context-run",
                        str(context_run),
                        "--input",
                        args.input,
                        "--output-dir",
                        str(atlas_output),
                    ],
                    root,
                    args.dry_run,
                    env=subprocess_env,
                )

    maybe_run(
        [
            sys.executable,
            "scripts/collect_task_mode_results.py",
            "--input-root",
            str(output_root),
        ],
        root,
        args.dry_run,
        env=subprocess_env,
    )

    summary = {
        "created_at": timestamp_slug(),
        "output_root": str(output_root),
        "splits": split_values,
        "host_corruption_fractions": host_corruption_fractions,
        "sequence_backbone": args.sequence_backbone,
        "requested_device": args.device,
        "resolved_device": resolved_device,
        "cuda_visible_devices": subprocess_env.get("CUDA_VISIBLE_DEVICES", ""),
    }
    (output_root / "context_study_manifest.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
