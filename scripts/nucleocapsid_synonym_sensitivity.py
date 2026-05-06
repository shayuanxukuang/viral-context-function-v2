#!/usr/bin/env python3
"""Post hoc nucleocapsid label-sensitivity analysis without retraining."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve

from run_v2_qc_suite import (
    align_predictions,
    load_run_predictions,
    load_strict_split_rows,
    load_training_metadata,
)
from train_overnight_baseline import choose_device


DEFAULT_SYNONYM_RE = r"(?i)(\bnucleocapsid\b|\bnucleoprotein\b|\bN protein\b|^N$|^N\s|\sN\s|\bprotein N\b)"
DEFAULT_AMBIGUOUS_RE = r"(?i)(hypothetical|uncharacterized|unknown|putative)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--protein-run", default="protein_only.family_holdout")
    parser.add_argument("--context-run", default="genome_aware_denovo.family_holdout")
    parser.add_argument("--input", type=Path, default=Path("data/processed/training/viral_protein_training_index.tsv.gz"))
    parser.add_argument("--split-manifest", type=Path, default=Path("data/processed/splits/viral_protein_strict_splits.tsv.gz"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--label", default="nucleocapsid")
    parser.add_argument("--synonym-regex", default=DEFAULT_SYNONYM_RE)
    parser.add_argument("--ambiguous-regex", default=DEFAULT_AMBIGUOUS_RE)
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
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def fmax(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    thresholds = np.unique(np.quantile(y_score, np.linspace(0.0, 1.0, 401)))
    best_f1 = 0.0
    best_thr = 0.5
    for thr in thresholds:
        pred = y_score >= thr
        tp = float(np.logical_and(pred, y_true == 1).sum())
        fp = float(np.logical_and(pred, y_true == 0).sum())
        fn = float(np.logical_and(~pred, y_true == 1).sum())
        precision = 0.0 if tp + fp == 0 else tp / (tp + fp)
        recall = 0.0 if tp + fn == 0 else tp / (tp + fn)
        f1 = 0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)
    return best_f1, best_thr


def metrics_for(name: str, y_true: np.ndarray, protein_score: np.ndarray, context_score: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    y = y_true[mask].astype(int)
    p = protein_score[mask]
    c = context_score[mask]
    p_ap = float(average_precision_score(y, p)) if len(np.unique(y)) > 1 else math.nan
    c_ap = float(average_precision_score(y, c)) if len(np.unique(y)) > 1 else math.nan
    p_fmax, p_thr = fmax(y, p)
    c_fmax, c_thr = fmax(y, c)
    return {
        "condition": name,
        "evaluated_examples": int(mask.sum()),
        "positives": int(y.sum()),
        "protein_only_ap": p_ap,
        "context_ap": c_ap,
        "delta_ap": c_ap - p_ap if not math.isnan(p_ap) and not math.isnan(c_ap) else "",
        "protein_only_fmax": p_fmax,
        "context_fmax": c_fmax,
        "delta_fmax": c_fmax - p_fmax,
        "protein_only_best_threshold": p_thr,
        "context_best_threshold": c_thr,
    }


def pr_rows(condition: str, y_true: np.ndarray, score: np.ndarray, model: str) -> list[dict[str, Any]]:
    precision, recall, thresholds = precision_recall_curve(y_true.astype(int), score)
    rows = []
    for idx, (p, r) in enumerate(zip(precision, recall)):
        rows.append(
            {
                "condition": condition,
                "model": model,
                "rank": idx,
                "precision": float(p),
                "recall": float(r),
                "threshold": float(thresholds[idx]) if idx < len(thresholds) else "",
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    run_root = args.run_root.resolve()
    output_dir = (args.output_dir or run_root / "qc_review").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)

    strict_rows = load_strict_split_rows(args.split_manifest)
    metadata = load_training_metadata(args.input, strict_rows)
    protein = load_run_predictions(
        run_root / args.protein_run,
        output_dir,
        device,
        args.batch_size,
        args.num_workers,
        args.prefetch_factor,
        args.force_predict,
    )
    context = load_run_predictions(
        run_root / args.context_run,
        output_dir,
        device,
        args.batch_size,
        args.num_workers,
        args.prefetch_factor,
        args.force_predict,
    )
    aligned = align_predictions(protein, context)
    label_to_idx = {label: idx for idx, label in enumerate(aligned["label_names"])}
    if args.label not in label_to_idx:
        raise ValueError(f"Label {args.label!r} not present in labels: {aligned['label_names']}")
    label_idx = label_to_idx[args.label]
    original = aligned["y_true"][:, label_idx].astype(np.uint8)
    protein_score = aligned["left_prob"][:, label_idx]
    context_score = aligned["right_prob"][:, label_idx]
    synonym_re = re.compile(args.synonym_regex)
    ambiguous_re = re.compile(args.ambiguous_regex)

    synonym_hits = []
    ambiguous_hits = []
    audit_rows = []
    for idx, accession in enumerate(aligned["accessions"]):
        meta = metadata.get(str(accession), {})
        text = " ".join(
            [
                str(meta.get("description", "")),
                str(meta.get("cds_product", "")),
                str(meta.get("text", "")),
            ]
        )
        syn = bool(synonym_re.search(text))
        amb = bool(ambiguous_re.search(text))
        synonym_hits.append(syn)
        ambiguous_hits.append(amb)
        if syn or amb:
            audit_rows.append(
                {
                    "protein_accession": accession,
                    "original_true": int(original[idx]),
                    "synonym_match": int(syn),
                    "ambiguous_match": int(amb),
                    "protein_score": float(protein_score[idx]),
                    "context_score": float(context_score[idx]),
                    "description": meta.get("description", ""),
                    "cds_product": meta.get("cds_product", ""),
                    "family": meta.get("virus_family", ""),
                    "host_group": meta.get("host_supergroup", ""),
                }
            )
    synonym_hits = np.asarray(synonym_hits, dtype=bool)
    ambiguous_hits = np.asarray(ambiguous_hits, dtype=bool)
    expanded = np.logical_or(original == 1, synonym_hits).astype(np.uint8)
    all_mask = np.ones(original.shape[0], dtype=bool)
    synonym_excluded_mask = ~(np.logical_and(original == 0, synonym_hits))
    ambiguous_excluded_mask = ~(np.logical_and(original == 0, ambiguous_hits))

    metric_rows = [
        metrics_for("original_label", original, protein_score, context_score, all_mask),
        metrics_for("synonym_expanded_label", expanded, protein_score, context_score, all_mask),
        metrics_for("synonym_false_positives_excluded", original, protein_score, context_score, synonym_excluded_mask),
        metrics_for("ambiguous_negatives_excluded", original, protein_score, context_score, ambiguous_excluded_mask),
    ]
    write_tsv(output_dir / f"qc6_{args.label}_synonym_sensitivity.tsv", metric_rows)
    write_tsv(output_dir / f"qc6_{args.label}_synonym_audit_matches.tsv", audit_rows)

    pr_out: list[dict[str, Any]] = []
    for condition, y, mask in [
        ("original_label", original, all_mask),
        ("synonym_expanded_label", expanded, all_mask),
        ("synonym_false_positives_excluded", original, synonym_excluded_mask),
        ("ambiguous_negatives_excluded", original, ambiguous_excluded_mask),
    ]:
        pr_out.extend(pr_rows(condition, y[mask], protein_score[mask], "protein_only"))
        pr_out.extend(pr_rows(condition, y[mask], context_score[mask], "genome_aware"))
    write_tsv(output_dir / f"qc6_{args.label}_synonym_sensitivity_pr_curves.tsv", pr_out)
    report = {
        "label": args.label,
        "output_metrics": str(output_dir / f"qc6_{args.label}_synonym_sensitivity.tsv"),
        "synonym_regex": args.synonym_regex,
        "ambiguous_regex": args.ambiguous_regex,
        "synonym_hits": int(synonym_hits.sum()),
        "ambiguous_hits": int(ambiguous_hits.sum()),
    }
    (output_dir / f"qc6_{args.label}_synonym_sensitivity_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
