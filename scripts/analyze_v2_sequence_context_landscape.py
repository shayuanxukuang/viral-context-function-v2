#!/usr/bin/env python3
"""Analyze the sequence-context landscape for ViruFunc V2 candidate outputs.

This optional secondary script uses model prediction scores, context gain, split
metadata, and the MMseqs2 top-hit baseline. It intentionally omits product text
and post hoc annotation descriptions from its outputs.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


DEFAULT_ASSIGNMENTS = Path("artifacts/return/context_study_v2_review_completion_20260504/qc_review/qc7_candidate_assignments.tsv")
DEFAULT_HOMOLOGY = Path(
    "artifacts/return/v2_plos_cb_supplementary_package_20260504/"
    "supplementary_tables/S21_homology_top_hit_assignments.tsv"
)
DEFAULT_SPLITS = Path("data/processed/splits/viral_protein_strict_splits.tsv.gz")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-assignments", type=Path, default=DEFAULT_ASSIGNMENTS)
    parser.add_argument("--homology-hits", type=Path, default=DEFAULT_HOMOLOGY)
    parser.add_argument("--homology-scheme", default="family_holdout")
    parser.add_argument("--homology-subset", default="all_test")
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/v2_sequence_structure_validation/landscape"))
    parser.add_argument("--high-context-gain", type=float, default=0.2)
    parser.add_argument("--low-sequence-identity", type=float, default=30.0)
    parser.add_argument("--figure-max-points", type=int, default=30000)
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_path(path: Path, root: Path) -> Path:
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("r", encoding="utf-8", newline="")


def read_tsv(path: Path, required: bool = False, table_name: str = "table") -> list[dict[str, str]]:
    if not path.exists():
        if required:
            raise SystemExit(f"Required {table_name} not found: {path}")
        return []
    with open_text(path) as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: clean_cell(row.get(key, "")) for key in fieldnames})


def clean_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.12g}"
    return str(value)


def as_float(value: Any, default: float = math.nan) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def load_homology(path: Path, scheme: str, subset: str) -> dict[str, dict[str, str]]:
    rows = read_tsv(path, required=False, table_name="MMseqs2 top-hit table")
    preferred: dict[str, dict[str, str]] = {}
    fallback: dict[str, dict[str, str]] = {}
    for row in rows:
        query = row.get("query", "")
        if not query:
            continue
        fallback.setdefault(query, row)
        if (not scheme or row.get("scheme") == scheme) and (not subset or row.get("subset") == subset):
            preferred.setdefault(query, row)
    merged = dict(fallback)
    merged.update(preferred)
    return merged


def load_split_map(path: Path, wanted: set[str]) -> dict[str, dict[str, str]]:
    if not path.exists() or not wanted:
        return {}
    out: dict[str, dict[str, str]] = {}
    with open_text(path) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if "protein_accession" not in (reader.fieldnames or []):
            raise SystemExit(f"Split manifest must contain protein_accession: {path}")
        for row in reader:
            accession = row.get("protein_accession", "")
            if accession in wanted:
                out[accession] = row
                if len(out) == len(wanted):
                    break
    return out


def identity_bin(identity: float) -> str:
    if math.isnan(identity):
        return "no_hit"
    if identity < 20:
        return "0-20"
    if identity < 30:
        return "20-30"
    if identity < 50:
        return "30-50"
    if identity < 70:
        return "50-70"
    if identity < 90:
        return "70-90"
    return "90-100"


def gain_bin(gain: float) -> str:
    if math.isnan(gain):
        return "missing"
    if gain < 0:
        return "<0"
    if gain < 0.05:
        return "0-0.05"
    if gain < 0.1:
        return "0.05-0.10"
    if gain < 0.2:
        return "0.10-0.20"
    if gain < 0.5:
        return "0.20-0.50"
    return ">=0.50"


def build_landscape(assignments: list[dict[str, str]], homology: dict[str, dict[str, str]], splits: dict[str, dict[str, str]], args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in assignments:
        accession = row.get("protein_accession", "")
        if not accession:
            continue
        hit = homology.get(accession, {})
        split = splits.get(accession, {})
        identity = as_float(hit.get("pident"))
        gain = as_float(row.get("context_gain"))
        prob = as_float(row.get("top_probability_calibrated"))
        low_seq = math.isnan(identity) or identity < args.low_sequence_identity
        high_context = not math.isnan(gain) and gain >= args.high_context_gain
        rows.append(
            {
                "protein_accession": accession,
                "predicted_label": row.get("candidate_label") or row.get("top_label", ""),
                "top_probability_calibrated": prob,
                "context_gain": gain,
                "high_context_gain": int(high_context),
                "nearest_homolog_identity": "" if math.isnan(identity) else identity,
                "nearest_homolog_bits": hit.get("bits", ""),
                "identity_bin": identity_bin(identity),
                "context_gain_bin": gain_bin(gain),
                "low_sequence_identity": int(low_seq),
                "low_sequence_high_context": int(low_seq and high_context),
                "module_supported": row.get("module_supported", ""),
                "hypothetical_or_unknown": row.get("hypothetical_or_unknown", ""),
                "family": split.get("virus_family", ""),
                "host_group": split.get("host_supergroup", ""),
                "family_holdout_split": split.get("family_holdout_split", ""),
            }
        )
    return rows


def summarize_bins(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row.get("identity_bin", ""), row.get("context_gain_bin", ""))].append(row)
    out = []
    for (ident_bin, ctx_bin), group in sorted(grouped.items()):
        gains = [as_float(row.get("context_gain")) for row in group if not math.isnan(as_float(row.get("context_gain")))]
        probs = [
            as_float(row.get("top_probability_calibrated"))
            for row in group
            if not math.isnan(as_float(row.get("top_probability_calibrated")))
        ]
        out.append(
            {
                "identity_bin": ident_bin,
                "context_gain_bin": ctx_bin,
                "protein_count": len(group),
                "high_context_count": sum(int(row.get("high_context_gain", 0)) for row in group),
                "low_sequence_high_context_count": sum(int(row.get("low_sequence_high_context", 0)) for row in group),
                "mean_context_gain": sum(gains) / len(gains) if gains else "",
                "mean_top_probability_calibrated": sum(probs) / len(probs) if probs else "",
            }
        )
    return out


def summarize_labels(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row.get("predicted_label", "")].append(row)
    out = []
    for label, group in sorted(grouped.items()):
        gains = sorted(
            as_float(row.get("context_gain"))
            for row in group
            if not math.isnan(as_float(row.get("context_gain")))
        )
        high = sum(int(row.get("high_context_gain", 0)) for row in group)
        low_high = sum(int(row.get("low_sequence_high_context", 0)) for row in group)
        out.append(
            {
                "predicted_label": label,
                "protein_count": len(group),
                "high_context_count": high,
                "high_context_fraction": high / len(group) if group else "",
                "low_sequence_high_context_count": low_high,
                "mean_context_gain": sum(gains) / len(gains) if gains else "",
                "median_context_gain": gains[len(gains) // 2] if gains else "",
            }
        )
    return sorted(out, key=lambda row: (as_float(row.get("high_context_fraction"), 0.0), as_float(row.get("mean_context_gain"), 0.0)), reverse=True)


def make_figures(rows: list[dict[str, Any]], label_rows: list[dict[str, Any]], fig_dir: Path, max_points: int) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc:
        raise SystemExit(f"matplotlib and numpy are required to render landscape figures: {exc}") from exc

    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_rows = rows[:max_points]
    x = [0.0 if row.get("nearest_homolog_identity", "") == "" else as_float(row.get("nearest_homolog_identity"), 0.0) for row in plot_rows]
    y = [as_float(row.get("context_gain"), 0.0) for row in plot_rows]
    c = [as_float(row.get("top_probability_calibrated"), 0.0) for row in plot_rows]
    sizes = [10 + 25 * int(row.get("high_context_gain", 0)) for row in plot_rows]
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    scatter = ax.scatter(x, y, c=c, s=sizes, cmap="viridis", alpha=0.5, edgecolors="none")
    ax.set_xlabel("MMseqs2 nearest-hit identity (%)")
    ax.set_ylabel("context gain")
    ax.set_title("Sequence-context landscape", loc="left", fontsize=10, fontweight="bold")
    ax.grid(True, color="#D8DEE9", linewidth=0.6, alpha=0.8)
    cb = fig.colorbar(scatter, ax=ax)
    cb.set_label("calibrated context probability")
    fig.tight_layout()
    fig.savefig(fig_dir / "sequence_context_landscape_scatter.png", dpi=240, bbox_inches="tight")
    fig.savefig(fig_dir / "sequence_context_landscape_scatter.pdf", bbox_inches="tight")
    plt.close(fig)

    labels = [row["predicted_label"] for row in label_rows[:12]]
    values = [as_float(row.get("high_context_fraction"), 0.0) for row in label_rows[:12]]
    counts = [int(row.get("protein_count", 0)) for row in label_rows[:12]]
    fig, ax = plt.subplots(figsize=(7.0, max(3.8, 0.28 * len(labels) + 1.4)))
    ypos = np.arange(len(labels))
    ax.barh(ypos, list(reversed(values)), color="#F58518")
    ax.set_yticks(ypos, labels=list(reversed([f"{label} (n={count})" for label, count in zip(labels, counts)])))
    ax.set_xlabel("high-context-gain fraction")
    ax.set_title("Labels enriched for context-sensitive predictions", loc="left", fontsize=10, fontweight="bold")
    ax.grid(True, axis="x", color="#D8DEE9", linewidth=0.6, alpha=0.8)
    fig.tight_layout()
    fig.savefig(fig_dir / "sequence_context_label_summary.png", dpi=240, bbox_inches="tight")
    fig.savefig(fig_dir / "sequence_context_label_summary.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    root = repo_root()
    assignments_path = resolve_path(args.candidate_assignments, root)
    homology_path = resolve_path(args.homology_hits, root)
    split_path = resolve_path(args.split_manifest, root)
    output_dir = resolve_path(args.output_dir, root)
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    assignments = read_tsv(assignments_path, required=True, table_name="candidate assignments")
    accessions = {row.get("protein_accession", "") for row in assignments if row.get("protein_accession", "")}
    homology = load_homology(homology_path, args.homology_scheme, args.homology_subset)
    splits = load_split_map(split_path, accessions)
    rows = build_landscape(assignments, homology, splits, args)
    bin_rows = summarize_bins(rows)
    label_rows = summarize_labels(rows)
    context_only_rows = [
        row for row in rows if int(row.get("low_sequence_high_context", 0)) == 1 and as_float(row.get("top_probability_calibrated"), 0.0) >= 0.8
    ]
    context_only_rows.sort(key=lambda row: as_float(row.get("context_gain"), 0.0), reverse=True)

    write_tsv(tables_dir / "sequence_context_landscape.tsv", rows)
    write_tsv(tables_dir / "sequence_context_landscape_bins.tsv", bin_rows)
    write_tsv(tables_dir / "context_gain_by_label.tsv", label_rows)
    write_tsv(tables_dir / "context_only_low_sequence_candidates.tsv", context_only_rows)
    make_figures(rows, label_rows, figures_dir, args.figure_max_points)

    report = {
        "claim_frame": "Landscape analysis of when genome context complements the MMseqs2 sequence baseline.",
        "assignment_count": len(rows),
        "low_sequence_high_context_count": len(context_only_rows),
        "outputs": {
            "landscape": str(tables_dir / "sequence_context_landscape.tsv"),
            "bins": str(tables_dir / "sequence_context_landscape_bins.tsv"),
            "label_summary": str(tables_dir / "context_gain_by_label.tsv"),
            "figures": str(figures_dir),
        },
    }
    (output_dir / "sequence_context_landscape_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
