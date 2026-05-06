from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


USER_AGENT = "ViruFunc-FM/0.1 (+https://local.workspace)"
CHUNK_SIZE = 4 * 1024 * 1024


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download datasets listed in configs/datasets.json")
    parser.add_argument("--manifest", default="configs/datasets.json", help="Path to dataset manifest")
    parser.add_argument("--dataset", action="append", default=[], help="Specific dataset id to download")
    parser.add_argument("--group", action="append", default=[], help="Dataset group to download")
    parser.add_argument("--include-disabled", action="store_true", help="Include datasets disabled in the manifest")
    parser.add_argument("--force", action="store_true", help="Redownload even if the local file already exists")
    parser.add_argument("--dry-run", action="store_true", help="Show the selected datasets without downloading")
    parser.add_argument("--list", action="store_true", help="List selected datasets and exit")
    parser.add_argument("--timeout", type=int, default=90, help="Per-request timeout in seconds")
    return parser.parse_args()


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

        dataset_id = dataset["id"]
        group = dataset.get("group")

        if not wanted_ids and not wanted_groups:
            selected.append(dataset)
            continue

        if dataset_id in wanted_ids or group in wanted_groups:
            selected.append(dataset)

    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for dataset in selected:
        if dataset["id"] not in seen:
            deduped.append(dataset)
            seen.add(dataset["id"])
    return deduped


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def request_headers() -> dict[str, str]:
    return {
        "User-Agent": USER_AGENT,
        "Accept": "*/*",
    }


def fetch_remote_metadata(url: str, timeout: int) -> dict[str, Any]:
    request = Request(url, method="HEAD", headers=request_headers())
    try:
        with urlopen(request, timeout=timeout) as response:
            headers = response.headers
            return {
                "status": getattr(response, "status", None),
                "content_length": parse_int(headers.get("Content-Length")),
                "content_type": headers.get("Content-Type"),
                "last_modified": headers.get("Last-Modified"),
                "etag": headers.get("ETag"),
            }
    except HTTPError as exc:
        if exc.code in {403, 405, 501}:
            return {}
        raise
    except URLError:
        return {}


def parse_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def download_dataset(dataset: dict[str, Any], root: Path, timeout: int, force: bool) -> dict[str, Any]:
    dataset_id = dataset["id"]
    url = dataset["url"]
    destination = root / dataset["destination"]
    destination.parent.mkdir(parents=True, exist_ok=True)

    remote = fetch_remote_metadata(url, timeout)
    remote_size = remote.get("content_length")

    if destination.exists() and not force:
        local_size = destination.stat().st_size
        if remote_size is None or local_size == remote_size:
            status = "skipped"
            sha256 = sha256_of_file(destination)
            return {
                "dataset_id": dataset_id,
                "status": status,
                "path": str(destination.relative_to(root)),
                "size_bytes": local_size,
                "sha256": sha256,
                "source_url": url,
                "timestamp": timestamp(),
                "remote": remote,
            }

    temp_path = Path(f"{destination}.part")
    if temp_path.exists():
        temp_path.unlink()

    digest = hashlib.sha256()
    request = Request(url, headers=request_headers())
    started_at = time.time()

    with urlopen(request, timeout=timeout) as response, temp_path.open("wb") as handle:
        while True:
            chunk = response.read(CHUNK_SIZE)
            if not chunk:
                break
            handle.write(chunk)
            digest.update(chunk)

    downloaded_size = temp_path.stat().st_size
    if remote_size is not None and downloaded_size != remote_size:
        temp_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"{dataset_id}: downloaded size {downloaded_size} does not match remote size {remote_size}"
        )

    temp_path.replace(destination)
    elapsed = round(time.time() - started_at, 2)

    return {
        "dataset_id": dataset_id,
        "status": "downloaded",
        "path": str(destination.relative_to(root)),
        "size_bytes": downloaded_size,
        "sha256": digest.hexdigest(),
        "elapsed_seconds": elapsed,
        "source_url": url,
        "timestamp": timestamp(),
        "remote": remote,
    }


def write_reports(root: Path, records: list[dict[str, Any]]) -> None:
    provenance_dir = root / "data" / "provenance"
    provenance_dir.mkdir(parents=True, exist_ok=True)

    latest_report = provenance_dir / "download_report.json"
    latest_report.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")

    history_path = provenance_dir / "download_history.jsonl"
    with history_path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def print_dataset_table(datasets: list[dict[str, Any]]) -> None:
    for dataset in datasets:
        print(f"[{dataset['group']}] {dataset['id']}")
        print(f"  -> {dataset['destination']}")
        print(f"  -> {dataset['url']}")


def main() -> int:
    args = parse_args()
    root = repo_root()
    manifest_path = (root / args.manifest).resolve()
    manifest = load_manifest(manifest_path)
    datasets = select_datasets(
        manifest["datasets"],
        dataset_ids=args.dataset,
        groups=args.group,
        include_disabled=args.include_disabled,
    )

    if not datasets:
        print("No datasets matched the requested filters.", file=sys.stderr)
        return 1

    if args.list or args.dry_run:
        print_dataset_table(datasets)
        return 0

    records: list[dict[str, Any]] = []
    failures = 0

    for dataset in datasets:
        dataset_id = dataset["id"]
        print(f"==> {dataset_id}")
        try:
            record = download_dataset(dataset, root=root, timeout=args.timeout, force=args.force)
            records.append(record)
            print(
                f"    {record['status']} {record['path']} "
                f"({record['size_bytes']} bytes, sha256={record['sha256'][:12]}...)"
            )
        except Exception as exc:  # noqa: BLE001
            failures += 1
            error_record = {
                "dataset_id": dataset_id,
                "status": "failed",
                "source_url": dataset["url"],
                "timestamp": timestamp(),
                "error": str(exc),
            }
            records.append(error_record)
            print(f"    failed: {exc}", file=sys.stderr)

    write_reports(root, records)
    print(f"Wrote download report to data/provenance/download_report.json")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
