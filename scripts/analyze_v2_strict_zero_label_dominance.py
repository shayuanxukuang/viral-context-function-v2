#!/usr/bin/env python3
"""No-host strict-zero and label-dominance sensitivity for ViruFunc V2.

This script intentionally works from cached family-heldout predictions. It does
not retrain models and it does not use annotation text, product text, neighbor
labels, or post hoc evidence as model inputs.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, f1_score


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


def load_prediction(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"Prediction cache not found: {path}")
    payload = np.load(path, allow_pickle=False)
    return {
        "path": str(path),
        "accessions": payload["accessions"].astype(str),
        "label_names": payload["label_names"].astype(str).tolist(),
        "y_prob": payload["y_prob"].astype(np.float32),
        "y_true": payload["y_true"].astype(np.uint8),
        "thresholds": payload["thresholds"].astype(np.float32),
    }


def align(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    if left["label_names"] != right["label_names"]:
        raise SystemExit(f"Label names differ between {left['path']} and {right['path']}")
    right_index = {acc: idx for idx, acc in enumerate(right["accessions"])}
    common = [acc for acc in left["accessions"] if acc in right_index]
    left_rows = np.asarray([idx for idx, acc in enumerate(left["accessions"]) if acc in right_index], dtype=np.int64)
    right_rows = np.asarray([right_index[acc] for acc in common], dtype=np.int64)
    return {
        "accessions": np.asarray(common),
        "label_names": left["label_names"],
        "y_true": left["y_true"][left_rows],
        "left_prob": left["y_prob"][left_rows],
        "right_prob": right["y_prob"][right_rows],
        "left_thresholds": left["thresholds"],
        "right_thresholds": right["thresholds"],
    }


def metadata_from_inputs(split_manifest: Path, protein_index: Path) -> dict[str, dict[str, str]]:
    rows = read_tsv(split_manifest)
    out: dict[str, dict[str, str]] = {}
    for row in rows:
        acc = row.get("protein_accession", "").strip()
        if acc:
            out[acc] = dict(row)
    for row in read_tsv(protein_index):
        acc = row.get("protein_accession", "").strip()
        if not acc:
            continue
        merged = out.setdefault(acc, {"protein_accession": acc})
        for key in (
            "protein_sequence_sha256",
            "virus_lineage",
            "virus_tax_id",
            "virus_name",
            "genome_version",
            "protein_description",
            "cds_product",
        ):
            if row.get(key, "").strip() and not merged.get(key, "").strip():
                merged[key] = row.get(key, "").strip()
    return out


def strict_zero_accessions(metadata: dict[str, dict[str, str]]) -> set[str]:
    train_hashes = {
        row.get("protein_sequence_sha256", "").strip()
        for row in metadata.values()
        if row.get("family_holdout_split", "").strip() == "train" and row.get("protein_sequence_sha256", "").strip()
    }
    strict = set()
    for acc, row in metadata.items():
        if row.get("family_holdout_split", "").strip() != "test":
            continue
        seq_hash = row.get("protein_sequence_sha256", "").strip()
        if seq_hash and seq_hash not in train_hashes:
            strict.add(acc)
    return strict


def block_indices(accessions: np.ndarray, metadata: dict[str, dict[str, str]], block_field: str, subset_mask: np.ndarray) -> dict[str, np.ndarray]:
    blocks: dict[str, list[int]] = defaultdict(list)
    for idx, acc in enumerate(accessions):
        if not bool(subset_mask[idx]):
            continue
        row = metadata.get(str(acc), {})
        block = row.get(block_field, "") or "__MISSING__"
        blocks[str(block)].append(idx)
    return {key: np.asarray(value, dtype=np.int64) for key, value in blocks.items()}


def metrics(y_true: np.ndarray, prob: np.ndarray, thresholds: np.ndarray, labels: list[str], label_indices: list[int]) -> dict[str, Any]:
    y_true = y_true[:, label_indices]
    prob = prob[:, label_indices]
    thresholds = thresholds[label_indices]
    y_pred = (prob >= thresholds.reshape(1, -1)).astype(np.uint8)
    per_label = []
    aps = []
    f1s = []
    for out_idx, label_idx in enumerate(label_indices):
        support = int(y_true[:, out_idx].sum())
        ap = None
        if support > 0:
            ap = float(average_precision_score(y_true[:, out_idx], prob[:, out_idx]))
            aps.append(ap)
        f1 = float(f1_score(y_true[:, out_idx], y_pred[:, out_idx], zero_division=0))
        f1s.append(f1)
        per_label.append({"label": labels[label_idx], "support": support, "average_precision": ap, "f1": f1})
    return {
        "macro_ap": float(np.mean(aps)) if aps else 0.0,
        "macro_f1": float(np.mean(f1s)) if f1s else 0.0,
        "micro_ap": float(average_precision_score(y_true.reshape(-1), prob.reshape(-1))) if y_true.size else 0.0,
        "micro_f1": float(f1_score(y_true.reshape(-1), y_pred.reshape(-1), zero_division=0)) if y_true.size else 0.0,
        "per_label": per_label,
    }


def delta_metrics(payload: dict[str, Any], rows: np.ndarray, label_indices: list[int]) -> dict[str, float]:
    p = metrics(payload["y_true"][rows], payload["left_prob"][rows], payload["left_thresholds"], payload["label_names"], label_indices)
    c = metrics(payload["y_true"][rows], payload["right_prob"][rows], payload["right_thresholds"], payload["label_names"], label_indices)
    return {
        "protein_macro_ap": p["macro_ap"],
        "context_macro_ap": c["macro_ap"],
        "delta_macro_ap": c["macro_ap"] - p["macro_ap"],
        "protein_macro_f1": p["macro_f1"],
        "context_macro_f1": c["macro_f1"],
        "delta_macro_f1": c["macro_f1"] - p["macro_f1"],
        "protein_micro_ap": p["micro_ap"],
        "context_micro_ap": c["micro_ap"],
        "delta_micro_ap": c["micro_ap"] - p["micro_ap"],
        "protein_micro_f1": p["micro_f1"],
        "context_micro_f1": c["micro_f1"],
        "delta_micro_f1": c["micro_f1"] - p["micro_f1"],
    }


def weighted_ap_from_sorted(y_sorted: np.ndarray, w_sorted: np.ndarray) -> float:
    pos_weight = w_sorted * y_sorted
    total_pos = float(pos_weight.sum())
    if total_pos <= 0:
        return 0.0
    cum_pos = np.cumsum(pos_weight)
    cum_weight = np.cumsum(w_sorted)
    precision = np.divide(cum_pos, cum_weight, out=np.zeros_like(cum_pos, dtype=np.float64), where=cum_weight > 0)
    return float(np.sum(precision * pos_weight) / total_pos)


def weighted_f1(y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray) -> float:
    weights = weights.astype(np.float64)
    tp = float(np.sum(weights * y_true * y_pred))
    fp = float(np.sum(weights * (1 - y_true) * y_pred))
    fn = float(np.sum(weights * y_true * (1 - y_pred)))
    denom = 2.0 * tp + fp + fn
    return 0.0 if denom <= 0 else float(2.0 * tp / denom)


def prepare_weighted_cache(payload: dict[str, Any], label_indices: list[int], side: str) -> dict[str, Any]:
    prob_key = "left_prob" if side == "left" else "right_prob"
    threshold_key = "left_thresholds" if side == "left" else "right_thresholds"
    prob = payload[prob_key]
    y_true = payload["y_true"]
    thresholds = payload[threshold_key]
    label_orders = [np.argsort(-prob[:, label_idx], kind="mergesort") for label_idx in label_indices]
    flat_prob = prob[:, label_indices].reshape(-1)
    flat_order = np.argsort(-flat_prob, kind="mergesort")
    flat_y = y_true[:, label_indices].reshape(-1).astype(np.float64)
    flat_rows = np.repeat(np.arange(prob.shape[0], dtype=np.int64), len(label_indices))
    pred = (prob[:, label_indices] >= thresholds[label_indices].reshape(1, -1)).astype(np.uint8)
    return {
        "prob": prob,
        "y_true": y_true,
        "thresholds": thresholds,
        "label_indices": label_indices,
        "label_orders": label_orders,
        "flat_order": flat_order,
        "flat_y": flat_y,
        "flat_rows": flat_rows,
        "pred": pred,
    }


def weighted_metrics_cached(cache: dict[str, Any], row_weights: np.ndarray) -> dict[str, float]:
    y_true = cache["y_true"]
    label_indices = cache["label_indices"]
    label_orders = cache["label_orders"]
    pred = cache["pred"]
    aps: list[float] = []
    f1s: list[float] = []
    for out_idx, label_idx in enumerate(label_indices):
        order = label_orders[out_idx]
        y = y_true[:, label_idx].astype(np.float64)
        y_sorted = y[order]
        w_sorted = row_weights[order].astype(np.float64)
        if float(np.sum(w_sorted * y_sorted)) > 0:
            aps.append(weighted_ap_from_sorted(y_sorted, w_sorted))
        f1s.append(weighted_f1(y, pred[:, out_idx], row_weights))

    flat_order = cache["flat_order"]
    flat_y = cache["flat_y"][flat_order]
    flat_w = row_weights[cache["flat_rows"][flat_order]].astype(np.float64)
    pred_flat = pred.reshape(-1)
    pred_w = np.repeat(row_weights, len(label_indices))
    return {
        "macro_ap": float(np.mean(aps)) if aps else 0.0,
        "macro_f1": float(np.mean(f1s)) if f1s else 0.0,
        "micro_ap": weighted_ap_from_sorted(flat_y, flat_w),
        "micro_f1": weighted_f1(cache["flat_y"], pred_flat, pred_w),
    }


def weighted_delta_metrics(
    row_weights: np.ndarray,
    left_cache: dict[str, Any],
    right_cache: dict[str, Any],
) -> dict[str, float]:
    p = weighted_metrics_cached(left_cache, row_weights)
    c = weighted_metrics_cached(right_cache, row_weights)
    return {
        "delta_macro_ap": c["macro_ap"] - p["macro_ap"],
        "delta_macro_f1": c["macro_f1"] - p["macro_f1"],
        "delta_micro_ap": c["micro_ap"] - p["micro_ap"],
        "delta_micro_f1": c["micro_f1"] - p["micro_f1"],
    }


def seed_average_point(seed_payloads: list[dict[str, Any]], subset_mask: np.ndarray, label_indices: list[int]) -> dict[str, float]:
    rows_by_seed = []
    for payload in seed_payloads:
        rows = np.where(subset_mask)[0]
        rows_by_seed.append(delta_metrics(payload, rows, label_indices))
    keys = rows_by_seed[0].keys()
    return {key: float(np.mean([row[key] for row in rows_by_seed])) for key in keys}


def bootstrap_ci(
    seed_payloads: list[dict[str, Any]],
    block_maps: list[dict[str, np.ndarray]],
    label_indices: list[int],
    iterations: int,
    seed: int,
) -> dict[str, float]:
    metrics_to_ci = ("delta_macro_ap", "delta_macro_f1", "delta_micro_ap", "delta_micro_f1")
    rng = np.random.default_rng(seed)
    weighted_caches = [
        (
            prepare_weighted_cache(payload, label_indices, "left"),
            prepare_weighted_cache(payload, label_indices, "right"),
        )
        for payload in seed_payloads
    ]
    sampled: dict[str, list[float]] = {metric: [] for metric in metrics_to_ci}
    for _ in range(iterations):
        per_seed = []
        for payload, bmap, (left_cache, right_cache) in zip(seed_payloads, block_maps, weighted_caches):
            blocks = np.asarray(sorted(bmap), dtype=object)
            if blocks.size == 0:
                continue
            chosen = rng.choice(blocks, size=blocks.size, replace=True)
            row_weights = np.zeros(payload["accessions"].shape[0], dtype=np.float64)
            sampled_rows = np.concatenate([bmap[str(block)] for block in chosen]).astype(np.int64)
            np.add.at(row_weights, sampled_rows, 1.0)
            per_seed.append(weighted_delta_metrics(row_weights, left_cache, right_cache))
        if not per_seed:
            continue
        for metric in metrics_to_ci:
            sampled[metric].append(float(np.mean([row[metric] for row in per_seed])))
    out: dict[str, float] = {}
    for metric, values in sampled.items():
        arr = np.asarray(values, dtype=np.float64)
        out[f"{metric}_ci_low"] = float(np.percentile(arr, 2.5)) if arr.size else float("nan")
        out[f"{metric}_ci_high"] = float(np.percentile(arr, 97.5)) if arr.size else float("nan")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-cache-root", type=Path, required=True, help="Directory containing seed_*/_prediction_cache/*.npz.")
    parser.add_argument("--split-manifest", type=Path, default=Path("data/processed/splits/viral_protein_strict_splits.tsv.gz"))
    parser.add_argument("--protein-index", type=Path, default=Path("data/processed/training/viral_protein_training_index.tsv.gz"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--left-run", default="protein_only.family_holdout")
    parser.add_argument("--right-run", default="genome_aware_nohost_local_genome.family_holdout")
    parser.add_argument("--block-field", default="virus_family")
    parser.add_argument("--bootstrap-iterations", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path.cwd()
    prediction_root = args.prediction_cache_root if args.prediction_cache_root.is_absolute() else root / args.prediction_cache_root
    split_manifest = args.split_manifest if args.split_manifest.is_absolute() else root / args.split_manifest
    protein_index = args.protein_index if args.protein_index.is_absolute() else root / args.protein_index
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = metadata_from_inputs(split_manifest, protein_index)
    strict_accessions = strict_zero_accessions(metadata)
    seed_dirs = sorted(path for path in prediction_root.glob("seed_*") if path.is_dir())
    if not seed_dirs:
        raise SystemExit(f"No seed_* directories found under {prediction_root}")

    payloads = []
    for seed_dir in seed_dirs:
        left = load_prediction(seed_dir / "_prediction_cache" / f"{args.left_run}.npz")
        right = load_prediction(seed_dir / "_prediction_cache" / f"{args.right_run}.npz")
        payloads.append(align(left, right))
    first = payloads[0]
    labels = first["label_names"]
    all_label_indices = list(range(len(labels)))
    if "nucleocapsid" not in labels:
        raise SystemExit(f"nucleocapsid label not found in labels: {labels}")
    nuc_idx = labels.index("nucleocapsid")
    non_nuc_indices = [idx for idx in all_label_indices if idx != nuc_idx]

    # Accessions are aligned by construction across seeds; verify this to avoid
    # mixing masks between different test-set orders.
    for payload in payloads[1:]:
        if not np.array_equal(first["accessions"], payload["accessions"]):
            raise SystemExit("Aligned accession order differs between seeds.")

    accessions = first["accessions"]
    full_mask = np.ones(accessions.shape[0], dtype=bool)
    strict_mask = np.asarray([str(acc) in strict_accessions for acc in accessions], dtype=bool)

    comparisons = [
        ("family_full_all_labels", full_mask, all_label_indices),
        ("family_strict_zero_all_labels", strict_mask, all_label_indices),
        ("family_full_excluding_nucleocapsid", full_mask, non_nuc_indices),
        ("family_strict_zero_excluding_nucleocapsid", strict_mask, non_nuc_indices),
    ]
    sensitivity_rows = []
    for idx, (name, mask, label_indices) in enumerate(comparisons):
        point = seed_average_point(payloads, mask, label_indices)
        block_maps = [block_indices(payload["accessions"], metadata, args.block_field, mask) for payload in payloads]
        ci = bootstrap_ci(payloads, block_maps, label_indices, args.bootstrap_iterations, args.seed + idx)
        sensitivity_rows.append(
            {
                "comparison": name,
                "model": "no_host_local_genome_vs_protein_only",
                "split": "family_holdout",
                "subset": "strict_zero_exact_transfer" if "strict_zero" in name else "full_test",
                "excluded_label": "nucleocapsid" if "excluding_nucleocapsid" in name else "",
                "test_proteins": int(mask.sum()),
                "label_count": len(label_indices),
                **point,
                **ci,
                "seed_count": len(payloads),
                "bootstrap_iterations": args.bootstrap_iterations,
                "block_unit": args.block_field,
                "block_count": len(block_indices(accessions, metadata, args.block_field, mask)),
            }
        )

    # Per-label deltas for the strict-zero subset and leave-one-label-out macro
    # deltas quantify whether the primary macro point estimate is dominated by
    # any single label.
    per_label_rows = []
    strict_rows = np.where(strict_mask)[0]
    for label_idx, label in enumerate(labels):
        seed_rows = [delta_metrics(payload, strict_rows, [label_idx]) for payload in payloads]
        row = {key: float(np.mean([seed_row[key] for seed_row in seed_rows])) for key in seed_rows[0]}
        support = int(first["y_true"][strict_rows, label_idx].sum())
        per_label_rows.append(
            {
                "label": label,
                "subset": "strict_zero_exact_transfer",
                "support": support,
                **row,
                "seed_count": len(payloads),
            }
        )

    leave_one_rows = []
    for label_idx, label in enumerate(labels):
        keep = [idx for idx in all_label_indices if idx != label_idx]
        point = seed_average_point(payloads, strict_mask, keep)
        leave_one_rows.append(
            {
                "excluded_label": label,
                "subset": "strict_zero_exact_transfer",
                "label_count": len(keep),
                **point,
                "seed_count": len(payloads),
            }
        )

    median_delta_macro_ap = float(np.median([row["delta_macro_ap"] for row in per_label_rows]))
    positive_label_count = int(sum(1 for row in per_label_rows if float(row["delta_macro_ap"]) > 0))
    report = {
        "prediction_cache_root": str(prediction_root),
        "split_manifest": str(split_manifest),
        "protein_index": str(protein_index),
        "left_run": args.left_run,
        "right_run": args.right_run,
        "seed_count": len(payloads),
        "test_count": int(full_mask.sum()),
        "strict_zero_test_count": int(strict_mask.sum()),
        "residual_exact_transfer_removed": int(full_mask.sum() - strict_mask.sum()),
        "bootstrap_iterations": args.bootstrap_iterations,
        "median_strict_zero_label_delta_macro_ap": median_delta_macro_ap,
        "strict_zero_positive_label_count": positive_label_count,
        "label_count": len(labels),
        "claim_frame": "No-host strict-zero sensitivity and leave-one-label analyses quantify whether context gains are label-specific rather than uniform.",
    }

    write_tsv(output_dir / "nohost_strict_zero_and_label_sensitivity.tsv", sensitivity_rows)
    write_tsv(output_dir / "nohost_strict_zero_per_label_deltas.tsv", per_label_rows)
    write_tsv(output_dir / "nohost_strict_zero_leave_one_label_out.tsv", leave_one_rows)
    (output_dir / "nohost_strict_zero_label_sensitivity_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
