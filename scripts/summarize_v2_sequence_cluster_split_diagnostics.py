#!/usr/bin/env python3
"""Diagnostics for sequence-cluster holdout splits.

These diagnostics clarify that sequence-cluster holdouts control sequence
relatedness and are not replacements for family- or host-disjoint OOD splits.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path
from typing import Any

import numpy as np


def open_text(path: Path):
    return gzip.open(path, "rt", encoding="utf-8", newline="") if path.suffix == ".gz" else path.open("r", encoding="utf-8", newline="")


def read_tsv(path: Path) -> list[dict[str, str]]:
    with open_text(path) as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    if not fields:
        fields = ["status"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_hashes(protein_index: Path) -> dict[str, str]:
    hashes = {}
    for row in read_tsv(protein_index):
        acc = row.get("protein_accession", "").strip()
        seq_hash = row.get("protein_sequence_sha256", "").strip()
        if acc and seq_hash:
            hashes[acc] = seq_hash
    return hashes


def load_support(cache_path: Path) -> tuple[list[str], dict[str, int]]:
    if not cache_path.exists():
        return [], {}
    payload = np.load(cache_path, allow_pickle=False)
    labels = payload["label_names"].astype(str).tolist()
    y_true = payload["y_true"].astype(np.uint8)
    return labels, {label: int(y_true[:, idx].sum()) for idx, label in enumerate(labels)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cluster-manifest", type=Path, required=True)
    parser.add_argument("--protein-index", type=Path, default=Path("data/processed/training/viral_protein_training_index.tsv.gz"))
    parser.add_argument("--prediction-cache-root", type=Path, help="Optional cluster prediction cache root with seed_*/_prediction_cache/*.npz.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", default="seed_42")
    args = parser.parse_args()

    rows = read_tsv(args.cluster_manifest)
    seq_hashes = load_hashes(args.protein_index)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    support_rows = []
    for threshold in (30, 50, 70):
        split_col = f"sequence_cluster_{threshold}_holdout_split"
        cluster_col = f"sequence_cluster_{threshold}_id"
        train = [row for row in rows if row.get(split_col, "") == "train"]
        test = [row for row in rows if row.get(split_col, "") == "test"]
        train_families = {row.get("virus_family", "") for row in train if row.get("virus_family", "")}
        test_families = {row.get("virus_family", "") for row in test if row.get("virus_family", "")}
        train_hosts = {row.get("host_taxid_key", "") for row in train if row.get("host_taxid_key", "")}
        test_hosts = {row.get("host_taxid_key", "") for row in test if row.get("host_taxid_key", "")}
        train_clusters = {row.get(cluster_col, "") for row in train if row.get(cluster_col, "")}
        test_clusters = {row.get(cluster_col, "") for row in test if row.get(cluster_col, "")}
        train_hashes = {seq_hashes.get(row.get("protein_accession", ""), "") for row in train}
        train_hashes.discard("")
        exact_transfer = sum(1 for row in test if seq_hashes.get(row.get("protein_accession", ""), "") in train_hashes)

        labels, supports = [], {}
        if args.prediction_cache_root:
            cache_path = args.prediction_cache_root / args.seed / "_prediction_cache" / f"protein_only.sequence_cluster_{threshold}_holdout.npz"
            labels, supports = load_support(cache_path)
        for label in labels:
            support_rows.append(
                {
                    "threshold": threshold,
                    "split": f"sequence_cluster_{threshold}_holdout",
                    "label": label,
                    "test_positive_count_seed_cache": supports.get(label, 0),
                }
            )

        summary_rows.append(
            {
                "threshold": threshold,
                "split": f"sequence_cluster_{threshold}_holdout",
                "train_proteins": len(train),
                "test_proteins": len(test),
                "train_clusters": len(train_clusters),
                "test_clusters": len(test_clusters),
                "train_test_cluster_overlap": len(train_clusters & test_clusters),
                "train_families": len(train_families),
                "test_families": len(test_families),
                "test_families_seen_in_train": len(test_families & train_families),
                "test_family_overlap_fraction": (len(test_families & train_families) / len(test_families)) if test_families else 0.0,
                "train_hosts": len(train_hosts),
                "test_hosts": len(test_hosts),
                "test_hosts_seen_in_train": len(test_hosts & train_hosts),
                "test_host_overlap_fraction": (len(test_hosts & train_hosts) / len(test_hosts)) if test_hosts else 0.0,
                "exact_sequence_transfer_test_count": exact_transfer,
                "exact_sequence_transfer_rate": (exact_transfer / len(test)) if test else 0.0,
                "interpretation": "sequence-relatedness control; family and host overlap are expected and this split is not a replacement for family-heldout OOD evaluation",
            }
        )

    write_tsv(output_dir / "sequence_cluster_split_diagnostics.tsv", summary_rows)
    write_tsv(output_dir / "sequence_cluster_split_label_support.tsv", support_rows)
    report = {
        "cluster_manifest": str(args.cluster_manifest),
        "protein_index": str(args.protein_index),
        "prediction_cache_root": str(args.prediction_cache_root) if args.prediction_cache_root else None,
        "output_summary": str(output_dir / "sequence_cluster_split_diagnostics.tsv"),
        "output_label_support": str(output_dir / "sequence_cluster_split_label_support.tsv"),
        "claim_frame": "Sequence-cluster holdouts control sequence relatedness but retain family/host overlap; they are supportive sensitivity analyses, not replacements for family-heldout OOD evaluation.",
    }
    (output_dir / "sequence_cluster_split_diagnostics_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
