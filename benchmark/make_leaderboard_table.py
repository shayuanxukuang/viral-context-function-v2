#!/usr/bin/env python
"""Build a compact leaderboard TSV from ViruFunc Atlas result JSON files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


FIELDS = [
    "rank",
    "method",
    "track",
    "split_version",
    "primary_metric",
    "macro_AP",
    "macro_F1",
    "micro_AP",
    "micro_F1",
    "submission_valid",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, nargs="+", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    rows = []
    for path in args.results:
        data = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "method": data.get("method", path.stem),
                "track": data.get("track", ""),
                "split_version": data.get("split_version", ""),
                "primary_metric": data.get("primary_metric", ""),
                "macro_AP": data.get("macro_AP"),
                "macro_F1": data.get("macro_F1"),
                "micro_AP": data.get("micro_AP"),
                "micro_F1": data.get("micro_F1"),
                "submission_valid": data.get("submission_valid", False),
            }
        )
    rows.sort(key=lambda row: (-1.0 if row["macro_AP"] is None else -float(row["macro_AP"])))
    for idx, row in enumerate(rows, start=1):
        row["rank"] = idx

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
