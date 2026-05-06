from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def scan_files(base_dir: Path) -> list[dict[str, object]]:
    if not base_dir.exists():
        return []

    items: list[dict[str, object]] = []
    for path in sorted(base_dir.rglob("*")):
        if not path.is_file():
            continue
        stat = path.stat()
        items.append(
            {
                "path": str(path),
                "relative_path": str(path.relative_to(repo_root())),
                "size_bytes": stat.st_size,
                "modified_at": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
            }
        )
    return items


def main() -> int:
    root = repo_root()
    inventory = {
        "generated_at": timestamp(),
        "raw": scan_files(root / "data" / "raw"),
        "interim": scan_files(root / "data" / "interim"),
        "processed": scan_files(root / "data" / "processed"),
    }

    inventory["summary"] = {
        "raw_files": len(inventory["raw"]),
        "interim_files": len(inventory["interim"]),
        "processed_files": len(inventory["processed"]),
        "raw_bytes": sum(item["size_bytes"] for item in inventory["raw"]),
        "interim_bytes": sum(item["size_bytes"] for item in inventory["interim"]),
        "processed_bytes": sum(item["size_bytes"] for item in inventory["processed"]),
    }

    output_path = root / "data" / "provenance" / "inventory.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(inventory, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote inventory to {output_path}")
    print(json.dumps(inventory["summary"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
