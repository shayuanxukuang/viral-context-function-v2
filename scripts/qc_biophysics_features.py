from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neural_network import MLPClassifier

from biophysics_features import BIOPHYSICS_FIELD_NAMES, compute_biophysics
from label_rules import LABEL_RULES, label_hits, normalize_text
from train_overnight_baseline import SPLIT_SCHEME_TO_COLUMN, load_split_assignments


TARGET_LABELS = ("envelope_glycoprotein", "membrane_matrix", "nucleocapsid")
EXPECTED_RANGE_FIELDS = {
    "bio_tm_helix_count": (0.0, None),
    "bio_tm_longest_hydrophobic_run": (0.0, None),
    "bio_signal_peptide_score": (0.0, 1.0),
    "bio_coiled_coil_score": (0.0, 1.0),
    "bio_disorder_score": (0.0, 1.0),
    "bio_low_complexity_fraction": (0.0, 1.0),
}


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QC cheap biophysics features and run a biophysics-only probe.")
    parser.add_argument("--input", default="data/processed/training/viral_protein_training_index.tsv.gz")
    parser.add_argument("--split-manifest", default="data/processed/splits/viral_protein_strict_splits.tsv.gz")
    parser.add_argument("--splits", default="family_holdout,host_holdout")
    parser.add_argument("--output-dir", default="runs/biophysics_qc")
    parser.add_argument("--min-label-support", type=int, default=50)
    parser.add_argument("--probe-model", choices=("logistic", "mlp"), default="logistic")
    parser.add_argument("--max-rows", type=int, default=0)
    return parser.parse_args()


def load_rows(
    input_path: Path,
    split_assignments: dict[str, dict[str, int]],
    split_values: list[str],
    max_rows: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    features: list[list[float]] = []
    labels: list[list[int]] = []
    split_arrays: dict[str, list[int]] = {split: [] for split in split_values}

    with open_text(input_path) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row_idx, row in enumerate(reader, start=1):
            if max_rows and row_idx > max_rows:
                break
            accession = str(row.get("protein_accession", "") or "").strip()
            sequence = str(row.get("protein_sequence", "") or "").strip()
            if not accession or not sequence:
                continue
            bio = compute_biophysics(sequence)
            features.append([float(bio[name]) for name in BIOPHYSICS_FIELD_NAMES])

            label_ids = set(label_hits(normalize_text(row)))
            labels.append([1 if label_idx in label_ids else 0 for label_idx in range(len(LABEL_RULES))])

            for split in split_values:
                split_arrays[split].append(int(split_assignments[split].get(accession, -1)))

    return (
        np.asarray(features, dtype=np.float32),
        np.asarray(labels, dtype=np.int8),
        {split: np.asarray(values, dtype=np.int8) for split, values in split_arrays.items()},
    )


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def feature_summary_rows(x: np.ndarray) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    flags: list[str] = []
    for idx, name in enumerate(BIOPHYSICS_FIELD_NAMES):
        values = x[:, idx]
        finite_mask = np.isfinite(values)
        min_allowed, max_allowed = EXPECTED_RANGE_FIELDS[name]
        upper_mask = np.zeros(values.shape, dtype=bool) if max_allowed is None else (values > max_allowed)
        out_of_range = int(
            np.sum(
                (~finite_mask)
                | (values < min_allowed)
                | upper_mask
            )
        )
        if out_of_range:
            flags.append(f"{name}: {out_of_range} value(s) out of expected range")
        rows.append(
            {
                "feature": name,
                "count": int(values.shape[0]),
                "missing_count": int(np.sum(~finite_mask)),
                "mean": float(np.nanmean(values)),
                "std": float(np.nanstd(values)),
                "min": float(np.nanmin(values)),
                "p25": float(np.nanpercentile(values, 25)),
                "p50": float(np.nanpercentile(values, 50)),
                "p75": float(np.nanpercentile(values, 75)),
                "max": float(np.nanmax(values)),
                "out_of_range_count": out_of_range,
            }
        )
    return rows, flags


def enrichment_rows(x: np.ndarray, y: np.ndarray) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    label_name_to_idx = {rule.name: idx for idx, rule in enumerate(LABEL_RULES)}
    for label_name in TARGET_LABELS:
        label_idx = label_name_to_idx[label_name]
        positives = y[:, label_idx] == 1
        negatives = ~positives
        if int(positives.sum()) == 0 or int(negatives.sum()) == 0:
            continue
        for feature_idx, feature_name in enumerate(BIOPHYSICS_FIELD_NAMES):
            pos_values = x[positives, feature_idx]
            neg_values = x[negatives, feature_idx]
            delta = float(pos_values.mean() - neg_values.mean())
            pooled_std = float(np.sqrt(((pos_values.var() + neg_values.var()) / 2.0) + 1e-8))
            rows.append(
                {
                    "label": label_name,
                    "feature": feature_name,
                    "positive_count": int(positives.sum()),
                    "negative_count": int(negatives.sum()),
                    "positive_mean": float(pos_values.mean()),
                    "negative_mean": float(neg_values.mean()),
                    "delta_mean": delta,
                    "effect_size_d": delta / pooled_std if pooled_std > 0 else 0.0,
                }
            )
    return rows


def make_probe(model_name: str):
    if model_name == "logistic":
        return OneVsRestClassifier(LogisticRegression(max_iter=500, class_weight="balanced", solver="liblinear"))
    return OneVsRestClassifier(MLPClassifier(hidden_layer_sizes=(32,), max_iter=300, random_state=42))


def probe_rows(
    x: np.ndarray,
    y: np.ndarray,
    split_name: str,
    split_ids: np.ndarray,
    min_label_support: int,
    probe_model: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    train_mask = split_ids == 0
    test_mask = split_ids == 2
    label_support = y[train_mask].sum(axis=0)
    keep = np.asarray([support >= min_label_support for support in label_support], dtype=bool)
    kept_labels = [rule.name for idx, rule in enumerate(LABEL_RULES) if keep[idx]]
    if not kept_labels:
        return [], {"split_scheme": split_name, "kept_label_count": 0}

    estimator = make_probe(probe_model)
    estimator.fit(x[train_mask], y[train_mask][:, keep])
    probabilities = estimator.predict_proba(x[test_mask])
    predictions = (probabilities >= 0.5).astype(np.int8)
    y_test = y[test_mask][:, keep]

    rows: list[dict[str, Any]] = []
    macro_ap_values: list[float] = []
    macro_f1_values: list[float] = []
    for label_idx, label_name in enumerate(kept_labels):
        positives = int(y_test[:, label_idx].sum())
        if positives == 0:
            continue
        ap = float(average_precision_score(y_test[:, label_idx], probabilities[:, label_idx]))
        f1 = float(f1_score(y_test[:, label_idx], predictions[:, label_idx], zero_division=0))
        macro_ap_values.append(ap)
        macro_f1_values.append(f1)
        rows.append(
            {
                "split_scheme": split_name,
                "label": label_name,
                "support_test": positives,
                "average_precision": ap,
                "f1_at_0.5": f1,
            }
        )

    summary = {
        "split_scheme": split_name,
        "probe_model": probe_model,
        "kept_label_count": len(kept_labels),
        "macro_average_precision": float(np.mean(macro_ap_values)) if macro_ap_values else None,
        "macro_f1": float(np.mean(macro_f1_values)) if macro_f1_values else None,
        "train_rows": int(train_mask.sum()),
        "test_rows": int(test_mask.sum()),
    }
    return rows, summary


def main() -> int:
    args = parse_args()
    root = repo_root()
    input_path = (root / args.input).resolve()
    split_manifest_path = (root / args.split_manifest).resolve()
    output_dir = (root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    split_values = [token.strip() for token in args.splits.split(",") if token.strip()]
    split_assignments = {
        split: load_split_assignments(split_manifest_path, SPLIT_SCHEME_TO_COLUMN[split])
        for split in split_values
    }

    x, y, split_ids = load_rows(
        input_path=input_path,
        split_assignments=split_assignments,
        split_values=split_values,
        max_rows=args.max_rows,
    )

    summary_rows, range_flags = feature_summary_rows(x)
    enrichment = enrichment_rows(x, y)
    write_tsv(output_dir / "biophysics_feature_summary.tsv", summary_rows)
    write_tsv(output_dir / "biophysics_label_enrichment.tsv", enrichment)

    probe_summaries: list[dict[str, Any]] = []
    for split_name in split_values:
        rows, summary = probe_rows(
            x=x,
            y=y,
            split_name=split_name,
            split_ids=split_ids[split_name],
            min_label_support=args.min_label_support,
            probe_model=args.probe_model,
        )
        probe_summaries.append(summary)
        if rows:
            write_tsv(output_dir / f"biophysics_probe.{split_name}.tsv", rows)

    report = {
        "created_at": timestamp(),
        "input": str(input_path),
        "row_count": int(x.shape[0]),
        "feature_count": int(x.shape[1]),
        "label_count": int(y.shape[1]),
        "target_labels": list(TARGET_LABELS),
        "probe_model": args.probe_model,
        "range_flags": range_flags,
        "probe_summaries": probe_summaries,
    }
    (output_dir / "biophysics_qc_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
