#!/usr/bin/env python3
"""Build a main-text baseline table and optional pLM+MMseqs2 score ensemble."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from run_v2_qc_suite import load_run_predictions
from train_overnight_baseline import choose_device, compute_metrics


def open_text(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("r", encoding="utf-8", newline="")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--protein-run", type=Path, help="Protein-only model run directory.")
    parser.add_argument("--context-run", type=Path, help="Genome-aware no-host model run directory.")
    parser.add_argument("--homology-metrics", type=Path, help="S21_homology_top_hit_metrics.tsv")
    parser.add_argument("--homology-assignments", type=Path, help="S21_homology_top_hit_assignments.tsv")
    parser.add_argument("--structure-metrics", type=Path, help="Optional mapped Phold/Foldseek baseline metrics TSV.")
    parser.add_argument("--scheme", default="family_holdout")
    parser.add_argument("--subset", default="all_test")
    parser.add_argument("--min-seq-identity", type=float, default=30.0)
    parser.add_argument("--min-seq-bits", type=float, default=50.0)
    parser.add_argument("--max-seq-evalue", type=float)
    parser.add_argument("--ensemble-weight", type=float, default=0.5, help="Weight on MMseqs binary-label score.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--force-predict", action="store_true")
    return parser.parse_args()


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    if not keys:
        keys = ["baseline"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_labels(text: str) -> list[str]:
    text = str(text or "").strip()
    if not text or text == "[]":
        return []
    try:
        value = json.loads(text)
        if isinstance(value, list):
            return [str(item) for item in value]
    except Exception:
        pass
    return [item.strip().strip("'\"") for item in text.strip("[]").split(",") if item.strip().strip("'\"")]


def as_float(value: Any, default: float = math.nan) -> float:
    try:
        text = str(value or "").strip()
        if not text:
            return default
        return float(text)
    except ValueError:
        return default


def pass_homology(row: dict[str, Any], min_identity: float, min_bits: float, max_evalue: float | None) -> bool:
    if as_float(row.get("pident")) < min_identity:
        return False
    if as_float(row.get("bits")) < min_bits:
        return False
    if max_evalue is not None:
        evalue = as_float(row.get("evalue"), math.nan)
        if math.isnan(evalue) or evalue > max_evalue:
            return False
    return True


def load_official_homology_metrics(path: Path, scheme: str, subset: str) -> dict[str, Any] | None:
    if not path or not path.exists():
        return None
    with open_text(path) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            if row.get("scheme") == scheme and row.get("subset") == subset:
                return dict(row)
    return None


def load_homology_scores(
    path: Path,
    accessions: np.ndarray,
    label_names: list[str],
    scheme: str,
    subset: str,
    min_identity: float,
    min_bits: float,
    max_evalue: float | None,
) -> np.ndarray:
    label_index = {label: idx for idx, label in enumerate(label_names)}
    accession_index = {str(acc): idx for idx, acc in enumerate(accessions)}
    scores = np.zeros((len(accessions), len(label_names)), dtype=np.float32)
    if not path or not path.exists():
        return scores
    with open_text(path) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"scheme", "subset", "query", "target_labels"}
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise SystemExit(f"Missing columns from homology assignment table {path}: {', '.join(missing)}")
        for row in reader:
            if row.get("scheme") != scheme or row.get("subset") != subset:
                continue
            query = str(row.get("query", "")).strip()
            idx = accession_index.get(query)
            if idx is None or not pass_homology(row, min_identity, min_bits, max_evalue):
                continue
            for label in parse_labels(row.get("target_labels", "")):
                j = label_index.get(label)
                if j is not None:
                    scores[idx, j] = 1.0
    return scores


def metrics_row(
    baseline: str,
    y_true: np.ndarray,
    y_score: np.ndarray,
    thresholds: np.ndarray,
    label_names: list[str],
    notes: str = "",
) -> dict[str, Any]:
    metrics = compute_metrics(y_true, y_score, thresholds, label_names)
    return {
        "baseline": baseline,
        "coverage_subset": "all_loaded_test",
        "test_proteins": int(y_true.shape[0]),
        "macro_ap": float(metrics["macro_average_precision"]),
        "micro_ap": float(metrics["micro_average_precision"]),
        "macro_f1": float(metrics["macro_f1"]),
        "micro_f1": float(metrics["micro_f1"]),
        "notes": notes,
    }


def model_predictions(args: argparse.Namespace, run_dir: Path, output_dir: Path) -> dict[str, Any]:
    return load_run_predictions(
        run_dir.resolve(),
        output_dir,
        choose_device(args.device),
        args.batch_size,
        args.num_workers,
        args.prefetch_factor,
        args.force_predict,
    )


def append_structure_metrics(path: Path, rows: list[dict[str, Any]]) -> None:
    if not path or not path.exists():
        return
    with open_text(path) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            rows.append(
                {
                    "baseline": row.get("baseline", row.get("method", "structure_or_phold_mapped")),
                    "coverage_subset": row.get("coverage_subset", row.get("subset", "")),
                    "test_proteins": row.get("test_proteins", row.get("n", "")),
                    "macro_ap": row.get("macro_ap", ""),
                    "micro_ap": row.get("micro_ap", ""),
                    "macro_f1": row.get("macro_f1", row.get("macro_fmax", "")),
                    "micro_f1": row.get("micro_f1", row.get("micro_fmax", "")),
                    "notes": row.get("notes", "external mapped structure-aware baseline"),
                }
            )


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    out_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    official = load_official_homology_metrics(args.homology_metrics, args.scheme, args.subset) if args.homology_metrics else None
    if official:
        rows.append(
            {
                "baseline": "MMseqs2_top_hit",
                "coverage_subset": f"{args.scheme}:{args.subset}",
                "test_proteins": official.get("test_proteins", ""),
                "covered_by_top_hit": official.get("covered_by_top_hit", ""),
                "coverage": official.get("coverage", ""),
                "macro_ap": official.get("macro_ap", ""),
                "micro_ap": official.get("micro_ap", ""),
                "macro_f1": official.get("macro_fmax", ""),
                "micro_f1": official.get("micro_fmax", ""),
                "notes": "sequence baseline from S21 top-hit label transfer",
            }
        )

    pred_cache: dict[str, dict[str, Any]] = {}
    if args.protein_run and args.protein_run.exists():
        pred_cache["protein"] = model_predictions(args, args.protein_run, out_dir)
        rows.append(
            metrics_row(
                "protein_only_pLM",
                pred_cache["protein"]["y_true"],
                pred_cache["protein"]["y_prob"],
                pred_cache["protein"]["thresholds"],
                pred_cache["protein"]["label_names"],
                "model metrics recomputed from checkpoint/test predictions",
            )
        )
    if args.context_run and args.context_run.exists():
        pred_cache["context"] = model_predictions(args, args.context_run, out_dir)
        rows.append(
            metrics_row(
                "genome_aware_nohost_local_genome",
                pred_cache["context"]["y_true"],
                pred_cache["context"]["y_prob"],
                pred_cache["context"]["thresholds"],
                pred_cache["context"]["label_names"],
                "clean de novo genome-organization model without host metadata",
            )
        )

    if "protein" in pred_cache and args.homology_assignments and args.homology_assignments.exists():
        protein = pred_cache["protein"]
        homology_scores = load_homology_scores(
            args.homology_assignments,
            protein["accessions"],
            protein["label_names"],
            args.scheme,
            args.subset,
            args.min_seq_identity,
            args.min_seq_bits,
            args.max_seq_evalue,
        )
        rows.append(
            metrics_row(
                "MMseqs2_binary_mapped_on_model_subset",
                protein["y_true"],
                homology_scores,
                np.full(len(protein["label_names"]), 0.5, dtype=np.float32),
                protein["label_names"],
                f"binary target-label score; pident>={args.min_seq_identity}, bits>={args.min_seq_bits}",
            )
        )
        ensemble = ((1.0 - args.ensemble_weight) * protein["y_prob"]) + (args.ensemble_weight * homology_scores)
        rows.append(
            metrics_row(
                "protein_pLM_plus_MMseqs2_linear_ensemble",
                protein["y_true"],
                ensemble.astype(np.float32),
                protein["thresholds"],
                protein["label_names"],
                f"post hoc score ensemble; MMseqs2 weight={args.ensemble_weight}; not a de novo model input",
            )
        )

    append_structure_metrics(args.structure_metrics, rows) if args.structure_metrics else None
    output_tsv = out_dir / "v2_main_baseline_comparison_table.tsv"
    write_tsv(output_tsv, rows)
    report = {
        "output_tsv": str(output_tsv),
        "row_count": len(rows),
        "claim_frame": "Baselines are reported to position genome context as complementary; homology/structure evidence is post hoc unless explicitly trained as an input branch.",
    }
    (out_dir / "v2_main_baseline_comparison_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
