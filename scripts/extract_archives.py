from __future__ import annotations

import argparse
import gzip
import json
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract archives downloaded for ViruFunc-FM")
    parser.add_argument("--manifest", default="configs/datasets.json", help="Path to dataset manifest")
    parser.add_argument("--dataset", action="append", default=[], help="Specific dataset id to extract")
    parser.add_argument("--group", action="append", default=[], help="Dataset group to extract")
    parser.add_argument("--include-disabled", action="store_true", help="Include datasets disabled in the manifest")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing extracted directory/file")
    return parser.parse_args()


def load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def select_datasets(
    datasets: list[dict[str, Any]],
    dataset_ids: list[str],
    groups: list[str],
    include_disabled: bool,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    wanted_ids = set(dataset_ids)
    wanted_groups = set(groups)

    for dataset in datasets:
        if not include_disabled and not dataset.get("enabled", False):
            continue
        if not wanted_ids and not wanted_groups:
            selected.append(dataset)
            continue
        if dataset["id"] in wanted_ids or dataset.get("group") in wanted_groups:
            selected.append(dataset)

    return selected


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_extract_tar(archive_path: Path, destination_dir: Path, force: bool) -> list[str]:
    if destination_dir.exists() and force:
        for child in destination_dir.rglob("*"):
            if child.is_file():
                child.unlink()
        for child in sorted(destination_dir.rglob("*"), reverse=True):
            if child.is_dir():
                child.rmdir()
    destination_dir.mkdir(parents=True, exist_ok=True)

    extracted: list[str] = []
    with tarfile.open(archive_path, "r:*") as archive:
        for member in archive.getmembers():
            target = (destination_dir / member.name).resolve()
            if destination_dir.resolve() not in target.parents and target != destination_dir.resolve():
                raise RuntimeError(f"Unsafe tar member path: {member.name}")
        try:
            archive.extractall(destination_dir, filter="data")
        except TypeError:
            archive.extractall(destination_dir)
        extracted = [member.name for member in archive.getmembers() if member.isfile()]
    return extracted


def extract_gzip_file(archive_path: Path, destination_path: Path, force: bool) -> str:
    if destination_path.exists() and not force:
        return str(destination_path)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(archive_path, "rb") as source, destination_path.open("wb") as target:
        while True:
            chunk = source.read(4 * 1024 * 1024)
            if not chunk:
                break
            target.write(chunk)
    return str(destination_path)


def extract_dataset(root: Path, dataset: dict[str, Any], force: bool) -> dict[str, Any]:
    source_path = root / dataset["destination"]
    if not source_path.exists():
        raise FileNotFoundError(f"Missing source file: {source_path}")

    dataset_dir = root / "data" / "interim" / dataset["id"]
    suffixes = source_path.suffixes

    if suffixes[-2:] == [".tar", ".gz"] or source_path.name.endswith(".tar.gz"):
        extracted = safe_extract_tar(source_path, dataset_dir, force=force)
        return {
            "dataset_id": dataset["id"],
            "status": "extracted",
            "archive": str(source_path.relative_to(root)),
            "target": str(dataset_dir.relative_to(root)),
            "members": extracted,
            "timestamp": timestamp(),
        }

    if suffixes and suffixes[-1] == ".gz":
        output_name = source_path.name[:-3]
        output_path = dataset_dir / output_name
        extracted_path = extract_gzip_file(source_path, output_path, force=force)
        return {
            "dataset_id": dataset["id"],
            "status": "extracted",
            "archive": str(source_path.relative_to(root)),
            "target": str(Path(extracted_path).relative_to(root)),
            "timestamp": timestamp(),
        }

    return {
        "dataset_id": dataset["id"],
        "status": "skipped",
        "archive": str(source_path.relative_to(root)),
        "reason": "not an archive",
        "timestamp": timestamp(),
    }


def write_report(root: Path, records: list[dict[str, Any]]) -> None:
    report_path = root / "data" / "provenance" / "extract_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    args = parse_args()
    root = repo_root()
    manifest = load_manifest((root / args.manifest).resolve())
    datasets = select_datasets(
        manifest["datasets"],
        dataset_ids=args.dataset,
        groups=args.group,
        include_disabled=args.include_disabled,
    )

    records: list[dict[str, Any]] = []
    failures = 0

    for dataset in datasets:
        print(f"==> {dataset['id']}")
        try:
            record = extract_dataset(root, dataset, force=args.force)
            records.append(record)
            print(f"    {record['status']} -> {record.get('target', record.get('reason'))}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            record = {
                "dataset_id": dataset["id"],
                "status": "failed",
                "error": str(exc),
                "timestamp": timestamp(),
            }
            records.append(record)
            print(f"    failed: {exc}")

    write_report(root, records)
    print("Wrote extraction report to data/provenance/extract_report.json")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
