#!/usr/bin/env python
"""Compute a family-block bootstrap CI for a scored long-format table.

Input table columns:
protein_id,label_id,y_true,score,family
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score


def read_rows(path: Path) -> list[dict[str, str]]:
    delimiter = "\t" if path.name.endswith(".tsv") else ","
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def macro_ap(rows: list[dict[str, str]]) -> float | None:
    by_label: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for row in rows:
        by_label[row["label_id"]].append((int(float(row["y_true"])), float(row["score"])))
    values: list[float] = []
    for pairs in by_label.values():
        y_true = np.asarray([pair[0] for pair in pairs])
        if int(y_true.sum()) == 0:
            continue
        y_score = np.asarray([pair[1] for pair in pairs], dtype=np.float32)
        values.append(float(average_precision_score(y_true, y_score)))
    if not values:
        return None
    return float(np.mean(values))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scored-table", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = read_rows(args.scored_table)
    by_family: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_family[row.get("family", "unknown")].append(row)
    families = sorted(by_family)
    rng = random.Random(args.seed)
    values: list[float] = []
    for _ in range(args.iterations):
        sampled: list[dict[str, str]] = []
        for family in (rng.choice(families) for _ in families):
            sampled.extend(by_family[family])
        value = macro_ap(sampled)
        if value is not None:
            values.append(value)
    observed = macro_ap(rows)
    low, high = np.percentile(np.asarray(values), [2.5, 97.5])
    result = {
        "metric": "macro_AP",
        "observed": observed,
        "block_count": len(families),
        "iterations": args.iterations,
        "ci_95": [float(low), float(high)],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
