from __future__ import annotations

import argparse
import io
import json
import tarfile
from datetime import datetime, timezone
from pathlib import Path


TOP_LEVEL_RESULT_FILES = (
    "suite_summary.json",
    "suite_summary.tsv",
    "context_study_manifest.json",
)

RUN_RESULT_FILES = (
    "run_manifest.json",
    "metrics_summary.json",
    "history.jsonl",
    "val_label_metrics.tsv",
    "test_label_metrics.tsv",
    "best_thresholds.json",
    "run_status.json",
)

ATLAS_RESULT_FILES = (
    "atlas_report.json",
    "label_deltas.tsv",
    "group_summary.tsv",
    "stratified_group_summary.tsv",
)


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Package the key outputs from a context-study run without including large caches or checkpoints."
    )
    parser.add_argument(
        "--input-root",
        required=True,
        help="Study output directory such as runs/context_study_main",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Archive path. Defaults to <input-root>.essential.tar.gz",
    )
    parser.add_argument(
        "--include-test-predictions",
        action="store_true",
        help="Also include per-run test_predictions.tsv.gz files. This can substantially increase archive size.",
    )
    parser.add_argument(
        "--include-audits",
        action="store_true",
        default=True,
        help="Include data/audits leakage audit outputs. Enabled by default.",
    )
    parser.add_argument(
        "--no-include-audits",
        dest="include_audits",
        action="store_false",
        help="Skip data/audits outputs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the files that would be packaged without writing an archive.",
    )
    return parser.parse_args()


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    return (root / path).resolve()


def maybe_add_file(file_path: Path, root: Path, collected: list[Path], seen: set[Path]) -> None:
    if not file_path.exists() or not file_path.is_file():
        return
    resolved = file_path.resolve()
    if resolved in seen:
        return
    try:
        resolved.relative_to(root)
    except ValueError:
        return
    collected.append(resolved)
    seen.add(resolved)


def load_suite_rows(input_root: Path) -> list[dict]:
    summary_path = input_root / "suite_summary.json"
    if not summary_path.exists():
        return []
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = payload.get("rows", [])
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, dict)]


def discover_run_dirs(input_root: Path) -> list[Path]:
    rows = load_suite_rows(input_root)
    run_dirs: list[Path] = []
    seen: set[Path] = set()
    for row in rows:
        run_dir_raw = str(row.get("run_dir", "")).strip()
        if not run_dir_raw:
            continue
        run_dir = Path(run_dir_raw)
        if not run_dir.is_absolute():
            run_dir = (input_root / run_dir).resolve()
        else:
            run_dir = run_dir.resolve()
        if run_dir.exists() and run_dir not in seen:
            run_dirs.append(run_dir)
            seen.add(run_dir)

    if run_dirs:
        return run_dirs

    fallback: list[Path] = []
    for candidate in sorted(input_root.iterdir()):
        if not candidate.is_dir():
            continue
        if (candidate / "metrics_summary.json").exists():
            fallback.append(candidate.resolve())
    return fallback


def discover_atlas_dirs(input_root: Path) -> list[Path]:
    atlas_dirs: list[Path] = []
    for candidate in sorted(input_root.iterdir()):
        if not candidate.is_dir():
            continue
        if candidate.name.startswith("context_atlas_"):
            atlas_dirs.append(candidate.resolve())
    return atlas_dirs


def collect_files(root: Path, input_root: Path, include_test_predictions: bool, include_audits: bool) -> list[Path]:
    collected: list[Path] = []
    seen: set[Path] = set()

    for name in TOP_LEVEL_RESULT_FILES:
        maybe_add_file(input_root / name, root, collected, seen)

    for run_dir in discover_run_dirs(input_root):
        for name in RUN_RESULT_FILES:
            maybe_add_file(run_dir / name, root, collected, seen)
        if include_test_predictions:
            maybe_add_file(run_dir / "test_predictions.tsv.gz", root, collected, seen)

    for atlas_dir in discover_atlas_dirs(input_root):
        for name in ATLAS_RESULT_FILES:
            maybe_add_file(atlas_dir / name, root, collected, seen)

    if include_audits:
        audit_dir = root / "data" / "audits"
        maybe_add_file(audit_dir / "feature_leakage_audit.tsv", root, collected, seen)
        maybe_add_file(audit_dir / "feature_leakage_audit.json", root, collected, seen)

    return sorted(collected)


def archive_member_name(root: Path, file_path: Path) -> str:
    return file_path.relative_to(root).as_posix()


def write_manifest(
    archive: tarfile.TarFile,
    input_root: Path,
    output_path: Path,
    include_test_predictions: bool,
    include_audits: bool,
    included_files: list[Path],
    total_bytes: int,
    root: Path,
) -> None:
    manifest = {
        "created_at": timestamp(),
        "input_root": str(input_root),
        "output_archive": str(output_path),
        "include_test_predictions": include_test_predictions,
        "include_audits": include_audits,
        "file_count": len(included_files),
        "total_bytes": total_bytes,
        "files": [archive_member_name(root, path) for path in included_files],
    }
    payload = json.dumps(manifest, indent=2, ensure_ascii=False).encode("utf-8")
    info = tarfile.TarInfo(name="package_manifest.json")
    info.size = len(payload)
    info.mtime = int(datetime.now(timezone.utc).timestamp())
    archive.addfile(info, io.BytesIO(payload))


def main() -> int:
    args = parse_args()
    root = repo_root()
    input_root = resolve_path(root, args.input_root)
    if not input_root.exists():
        raise FileNotFoundError(f"Input root was not found: {input_root}")

    output_path = resolve_path(root, args.output) if args.output else input_root.with_suffix(".essential.tar.gz")
    files = collect_files(root, input_root, args.include_test_predictions, args.include_audits)
    total_bytes = sum(path.stat().st_size for path in files)

    if args.dry_run:
        print(
            json.dumps(
                {
                    "input_root": str(input_root),
                    "output_archive": str(output_path),
                    "file_count": len(files),
                    "total_bytes": total_bytes,
                    "files": [archive_member_name(root, path) for path in files],
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output_path, "w:gz") as archive:
        for file_path in files:
            archive.add(file_path, arcname=archive_member_name(root, file_path), recursive=False)
        write_manifest(
            archive,
            input_root=input_root,
            output_path=output_path,
            include_test_predictions=args.include_test_predictions,
            include_audits=args.include_audits,
            included_files=files,
            total_bytes=total_bytes,
            root=root,
        )

    print(
        json.dumps(
            {
                "input_root": str(input_root),
                "output_archive": str(output_path),
                "file_count": len(files),
                "total_bytes": total_bytes,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
