#!/usr/bin/env python
"""Evaluate ViruFunc Atlas long-format prediction submissions.

Expected prediction format:

protein_id,label_id,score
YP_000001.1,nuclease,0.82
YP_000001.1,lysis,0.04

The split or truth table must provide the target proteins and ground-truth
labels, either as a semicolon/comma-separated label list column
(`true_labels`, `label_ids`, `labels`, or `positive_labels`) or as per-label
binary columns named by label id, `label:<label>`, or `y_<label>`.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    import numpy as np
    from sklearn.metrics import average_precision_score, f1_score
except Exception as exc:  # pragma: no cover - dependency failure path
    raise SystemExit(
        "evaluate_predictions.py requires numpy and scikit-learn. "
        "Install the ViruFunc Atlas Core environment before evaluating submissions."
    ) from exc


TRACKS: dict[str, dict[str, Any]] = {
    "T1_sequence_only_de_novo": {
        "primary_metric": "family_heldout_strict_zero_macro_AP",
        "allowed_features": {
            "target_sequence",
            "sequence_embedding",
            "training_labels",
            "sequence_length",
        },
        "forbidden_features": {
            "host",
            "taxonomy",
            "gene_order",
            "coordinates",
            "neighbor_sequences",
            "neighbor_labels",
            "product_text",
            "external_annotation_labels",
            "database_hit_counts",
            "mmseqs2",
            "foldseek",
            "phold",
            "phrog",
            "interpro",
            "pfam",
            "cdd",
            "test_annotation_priors",
        },
    },
    "T2_genome_context_nohost": {
        "primary_metric": "family_heldout_strict_zero_macro_AP",
        "allowed_features": {
            "target_sequence",
            "sequence_embedding",
            "neighbor_sequences",
            "gene_order",
            "coordinates",
            "strand",
            "gap_overlap",
            "segment_topology",
            "genome_topology",
            "sequence_length",
            "training_labels",
        },
        "forbidden_features": {
            "host",
            "taxonomy",
            "neighbor_labels",
            "product_text",
            "external_annotation_labels",
            "database_hit_counts",
            "mmseqs2",
            "foldseek",
            "phold",
            "phrog",
            "interpro",
            "pfam",
            "cdd",
            "test_annotation_priors",
        },
    },
    "T3_audited_metadata_context": {
        "primary_metric": "family_heldout_macro_AP_with_host_heldout_secondary",
        "allowed_features": {
            "target_sequence",
            "sequence_embedding",
            "neighbor_sequences",
            "gene_order",
            "coordinates",
            "strand",
            "gap_overlap",
            "segment_topology",
            "genome_topology",
            "host",
            "taxonomy",
            "sequence_length",
            "training_labels",
        },
        "forbidden_features": {
            "neighbor_labels",
            "product_text",
            "external_annotation_labels",
            "database_hit_counts",
            "mmseqs2",
            "foldseek",
            "phold",
            "phrog",
            "interpro",
            "pfam",
            "cdd",
            "test_annotation_priors",
        },
    },
    "T4_annotation_refinement_open_evidence": {
        "primary_metric": "practical_annotation_family_heldout_macro_AP",
        "allowed_features": {
            "target_sequence",
            "sequence_embedding",
            "neighbor_sequences",
            "gene_order",
            "coordinates",
            "strand",
            "gap_overlap",
            "segment_topology",
            "genome_topology",
            "host",
            "taxonomy",
            "product_text",
            "external_annotation_labels",
            "database_hit_counts",
            "mmseqs2",
            "foldseek",
            "phold",
            "phrog",
            "interpro",
            "pfam",
            "cdd",
            "test_annotation_priors",
            "training_labels",
            "sequence_length",
        },
        "forbidden_features": set(),
    },
}


TRACK_ALIASES = {
    key.lower(): key for key in TRACKS
} | {
    "t1": "T1_sequence_only_de_novo",
    "sequence_only": "T1_sequence_only_de_novo",
    "sequence-only": "T1_sequence_only_de_novo",
    "t2": "T2_genome_context_nohost",
    "genome_context_nohost": "T2_genome_context_nohost",
    "genome-context-nohost": "T2_genome_context_nohost",
    "t3": "T3_audited_metadata_context",
    "audited_metadata_context": "T3_audited_metadata_context",
    "t4": "T4_annotation_refinement_open_evidence",
    "annotation_refinement": "T4_annotation_refinement_open_evidence",
    "open_evidence": "T4_annotation_refinement_open_evidence",
}


PROTEIN_COLUMNS = ("protein_id", "protein_accession", "accession")
LABEL_COLUMNS = ("label_id", "label", "primary_label", "function_label")
SCORE_COLUMNS = ("score", "probability", "prediction", "y_score")
TRUTH_LIST_COLUMNS = ("true_labels", "label_ids", "labels", "positive_labels")
TRUTH_VALUE_COLUMNS = ("truth", "target", "y_true", "value")
FAMILY_COLUMNS = ("virus_family", "family", "family_id", "block_id")
SPLIT_NAME_COLUMNS = ("split_version", "manifest_version", "version")


def open_text(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("r", encoding="utf-8", newline="")


def detect_delimiter(path: Path) -> str:
    name = path.name.lower()
    if name.endswith(".tsv") or name.endswith(".tsv.gz"):
        return "\t"
    if name.endswith(".csv") or name.endswith(".csv.gz"):
        return ","
    with open_text(path) as handle:
        sample = handle.read(4096)
    try:
        return csv.Sniffer().sniff(sample, delimiters="\t,").delimiter
    except csv.Error:
        return "\t" if sample.count("\t") >= sample.count(",") else ","


def read_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    delimiter = detect_delimiter(path)
    with open_text(path) as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        rows = [{str(k): ("" if v is None else str(v)) for k, v in row.items()} for row in reader]
        fieldnames = [str(name) for name in (reader.fieldnames or [])]
    return rows, fieldnames


def first_present(row: dict[str, str], columns: tuple[str, ...]) -> str:
    for column in columns:
        if column in row and row[column].strip():
            return row[column].strip()
    return ""


def normalize_track(raw: str) -> str:
    key = raw.strip()
    if key in TRACKS:
        return key
    normalized = key.lower().replace(" ", "_")
    if normalized in TRACK_ALIASES:
        return TRACK_ALIASES[normalized]
    raise ValueError(f"Unknown track '{raw}'. Valid tracks: {', '.join(TRACKS)}")


def load_labels(path: Path) -> list[str]:
    rows, fieldnames = read_rows(path)
    label_column = next((column for column in LABEL_COLUMNS if column in fieldnames), "")
    if not label_column:
        raise ValueError(
            f"Could not find a label column in {path}. Expected one of {LABEL_COLUMNS}."
        )
    labels: list[str] = []
    seen: set[str] = set()
    for row in rows:
        label = row.get(label_column, "").strip()
        if label and label not in seen:
            labels.append(label)
            seen.add(label)
    if not labels:
        raise ValueError(f"No labels found in {path}.")
    return labels


def split_tokens(raw: str) -> list[str]:
    text = raw.strip()
    if not text:
        return []
    if text.startswith("["):
        try:
            values = json.loads(text)
            if isinstance(values, list):
                return [str(value).strip() for value in values if str(value).strip()]
        except json.JSONDecodeError:
            pass
    for sep in (";", "|", ","):
        text = text.replace(sep, " ")
    return [token.strip() for token in text.split() if token.strip()]


def truth_from_wide_row(row: dict[str, str], labels: list[str]) -> set[str] | None:
    for column in TRUTH_LIST_COLUMNS:
        if column in row and row[column].strip():
            tokens = split_tokens(row[column])
            return {token for token in tokens if token in set(labels)}
    truth: set[str] = set()
    found_any = False
    for label in labels:
        raw = ""
        for column in (label, f"label:{label}", f"y_{label}"):
            if column in row and row[column].strip():
                raw = row[column].strip()
                found_any = True
                break
        if raw.lower() in {"1", "1.0", "true", "t", "yes", "y", "positive", "pos"}:
            truth.add(label)
    return truth if found_any else None


def load_split_rows(
    split_path: Path,
    split_column: str | None,
    split_value: str | None,
) -> tuple[list[dict[str, str]], list[str]]:
    rows, fieldnames = read_rows(split_path)
    if split_column:
        if split_column not in fieldnames:
            raise ValueError(f"Split column '{split_column}' not found in {split_path}.")
        if split_value:
            rows = [row for row in rows if row.get(split_column, "").strip() == split_value]
        else:
            rows = [
                row
                for row in rows
                if row.get(split_column, "").strip().lower() in {"test", "2", "heldout"}
            ]
    return rows, fieldnames


def extract_proteins(split_rows: list[dict[str, str]]) -> tuple[list[str], dict[str, dict[str, str]]]:
    proteins: list[str] = []
    row_by_protein: dict[str, dict[str, str]] = {}
    seen: set[str] = set()
    for row in split_rows:
        protein_id = first_present(row, PROTEIN_COLUMNS)
        if not protein_id:
            continue
        if protein_id in seen:
            raise ValueError(f"Duplicate protein id in split/truth table: {protein_id}")
        proteins.append(protein_id)
        row_by_protein[protein_id] = row
        seen.add(protein_id)
    if not proteins:
        raise ValueError("No protein ids were found in the split/truth table.")
    return proteins, row_by_protein


def infer_split_version(
    split_path: Path,
    split_rows: list[dict[str, str]],
    explicit: str | None,
) -> str:
    if explicit:
        return explicit
    for row in split_rows:
        value = first_present(row, SPLIT_NAME_COLUMNS)
        if value:
            return value
    return split_path.stem.replace(".tsv", "").replace(".csv", "")


def load_truth_matrix(
    labels: list[str],
    proteins: list[str],
    split_rows_by_protein: dict[str, dict[str, str]],
    truth_path: Path | None,
    errors: list[str],
) -> np.ndarray | None:
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    protein_to_idx = {protein: idx for idx, protein in enumerate(proteins)}
    truth = np.zeros((len(proteins), len(labels)), dtype=np.int8)
    found_truth = False
    unknown_truth_labels: set[str] = set()
    unknown_truth_proteins: set[str] = set()

    if truth_path is not None:
        rows, fieldnames = read_rows(truth_path)
        has_long_columns = any(column in fieldnames for column in LABEL_COLUMNS) and any(
            column in fieldnames for column in TRUTH_VALUE_COLUMNS
        )
        if has_long_columns:
            for row in rows:
                protein_id = first_present(row, PROTEIN_COLUMNS)
                label = first_present(row, LABEL_COLUMNS)
                value_raw = first_present(row, TRUTH_VALUE_COLUMNS)
                if protein_id not in protein_to_idx:
                    unknown_truth_proteins.add(protein_id)
                    continue
                if label not in label_to_idx:
                    unknown_truth_labels.add(label)
                    continue
                found_truth = True
                if value_raw.lower() in {"1", "1.0", "true", "t", "yes", "y", "positive", "pos"}:
                    truth[protein_to_idx[protein_id], label_to_idx[label]] = 1
                else:
                    try:
                        truth[protein_to_idx[protein_id], label_to_idx[label]] = int(
                            float(value_raw) > 0.0
                        )
                    except ValueError:
                        errors.append(
                            f"Non-numeric truth value for {protein_id}/{label}: {value_raw}"
                        )
        else:
            for row in rows:
                protein_id = first_present(row, PROTEIN_COLUMNS)
                if protein_id not in protein_to_idx:
                    unknown_truth_proteins.add(protein_id)
                    continue
                row_truth = truth_from_wide_row(row, labels)
                if row_truth is None:
                    continue
                found_truth = True
                for label in row_truth:
                    if label in label_to_idx:
                        truth[protein_to_idx[protein_id], label_to_idx[label]] = 1
                    else:
                        unknown_truth_labels.add(label)
    else:
        for protein_id, row in split_rows_by_protein.items():
            row_truth = truth_from_wide_row(row, labels)
            if row_truth is None:
                continue
            found_truth = True
            for label in row_truth:
                if label in label_to_idx:
                    truth[protein_to_idx[protein_id], label_to_idx[label]] = 1
                else:
                    unknown_truth_labels.add(label)

    if unknown_truth_proteins:
        errors.append(
            f"Truth table contains {len(unknown_truth_proteins)} unknown proteins; "
            f"first examples: {sorted(unknown_truth_proteins)[:5]}"
        )
    if unknown_truth_labels:
        errors.append(
            f"Truth table contains {len(unknown_truth_labels)} unknown labels; "
            f"first examples: {sorted(unknown_truth_labels)[:5]}"
        )
    if not found_truth:
        return None
    return truth


def load_predictions(
    pred_path: Path,
    labels: list[str],
    proteins: list[str],
    allow_missing: bool,
    errors: list[str],
) -> tuple[np.ndarray, dict[str, Any]]:
    rows, fieldnames = read_rows(pred_path)
    missing_columns = []
    if not any(column in fieldnames for column in PROTEIN_COLUMNS):
        missing_columns.append("protein_id")
    if not any(column in fieldnames for column in LABEL_COLUMNS):
        missing_columns.append("label_id")
    if not any(column in fieldnames for column in SCORE_COLUMNS):
        missing_columns.append("score")
    if missing_columns:
        raise ValueError(
            f"Prediction file is missing required columns: {', '.join(missing_columns)}"
        )

    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    protein_to_idx = {protein: idx for idx, protein in enumerate(proteins)}
    scores = np.full((len(proteins), len(labels)), np.nan, dtype=np.float32)
    seen_pairs: set[tuple[int, int]] = set()
    duplicate_pairs: list[str] = []
    unknown_proteins: set[str] = set()
    unknown_labels: set[str] = set()
    invalid_scores: list[str] = []

    for row in rows:
        protein_id = first_present(row, PROTEIN_COLUMNS)
        label = first_present(row, LABEL_COLUMNS)
        score_raw = first_present(row, SCORE_COLUMNS)
        if protein_id not in protein_to_idx:
            unknown_proteins.add(protein_id)
            continue
        if label not in label_to_idx:
            unknown_labels.add(label)
            continue
        try:
            score = float(score_raw)
        except ValueError:
            invalid_scores.append(f"{protein_id}/{label}={score_raw}")
            continue
        if not math.isfinite(score) or score < 0.0 or score > 1.0:
            invalid_scores.append(f"{protein_id}/{label}={score_raw}")
            continue
        protein_idx = protein_to_idx[protein_id]
        label_idx = label_to_idx[label]
        pair = (protein_idx, label_idx)
        if pair in seen_pairs:
            duplicate_pairs.append(f"{protein_id}/{label}")
            continue
        scores[protein_idx, label_idx] = score
        seen_pairs.add(pair)

    missing_mask = np.isnan(scores)
    missing_count = int(missing_mask.sum())
    if missing_count and allow_missing:
        scores[missing_mask] = 0.0
    elif missing_count:
        errors.append(
            f"Prediction file is missing {missing_count} protein-label scores "
            f"out of {scores.size}; rerun with --allow-missing to treat them as zero."
        )
    if unknown_proteins:
        errors.append(
            f"Prediction file contains {len(unknown_proteins)} unknown proteins; "
            f"first examples: {sorted(unknown_proteins)[:5]}"
        )
    if unknown_labels:
        errors.append(
            f"Prediction file contains {len(unknown_labels)} unknown labels; "
            f"first examples: {sorted(unknown_labels)[:5]}"
        )
    if invalid_scores:
        errors.append(
            f"Prediction file contains {len(invalid_scores)} invalid scores; "
            f"first examples: {invalid_scores[:5]}"
        )
    if duplicate_pairs:
        errors.append(
            f"Prediction file contains {len(duplicate_pairs)} duplicate protein-label rows; "
            f"first examples: {duplicate_pairs[:5]}"
        )

    stats = {
        "row_count": len(rows),
        "expected_score_count": int(scores.size),
        "observed_unique_score_count": int(len(seen_pairs)),
        "missing_score_count": missing_count,
        "allow_missing": allow_missing,
    }
    return scores, stats


def load_used_features(args: argparse.Namespace) -> set[str]:
    features: set[str] = set()
    if args.used_features:
        features.update(token.strip() for token in args.used_features.split(",") if token.strip())
    if args.feature_declaration:
        path = Path(args.feature_declaration)
        text = path.read_text(encoding="utf-8").strip()
        if text:
            try:
                data = json.loads(text)
                raw_features = data.get("used_features", data.get("features", []))
                if isinstance(raw_features, str):
                    features.update(split_tokens(raw_features))
                elif isinstance(raw_features, list):
                    features.update(str(feature).strip() for feature in raw_features if str(feature).strip())
            except json.JSONDecodeError:
                for line in text.splitlines():
                    stripped = line.strip()
                    if not stripped or stripped.startswith("#"):
                        continue
                    if ":" in stripped:
                        key, value = stripped.split(":", 1)
                        if key.strip() in {"used_features", "features"}:
                            features.update(split_tokens(value))
                    elif stripped.startswith("-"):
                        features.add(stripped[1:].strip())
    return features


def validate_track_features(track: str, used_features: set[str]) -> dict[str, Any]:
    spec = TRACKS[track]
    forbidden = set(spec["forbidden_features"])
    allowed = set(spec["allowed_features"])
    conflicts = sorted(feature for feature in used_features if feature in forbidden)
    undeclared_to_track = sorted(
        feature for feature in used_features if feature not in allowed and feature not in forbidden
    )
    return {
        "track": track,
        "used_features": sorted(used_features),
        "forbidden_feature_declaration": "FAIL" if conflicts else "PASS",
        "forbidden_conflicts": conflicts,
        "unrecognized_features": undeclared_to_track,
    }


def per_label_ap(y_true: np.ndarray, y_score: np.ndarray, labels: list[str]) -> dict[str, float | None]:
    values: dict[str, float | None] = {}
    for idx, label in enumerate(labels):
        positives = int(y_true[:, idx].sum())
        if positives == 0:
            values[label] = None
            continue
        values[label] = float(average_precision_score(y_true[:, idx], y_score[:, idx]))
    return values


def macro_from_per_label(values: dict[str, float | None]) -> float | None:
    finite = [value for value in values.values() if value is not None and math.isfinite(value)]
    if not finite:
        return None
    return float(np.mean(finite))


def compute_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    labels: list[str],
    threshold: float,
) -> dict[str, Any]:
    label_ap = per_label_ap(y_true, y_score, labels)
    y_pred = (y_score >= threshold).astype(np.int8)
    result: dict[str, Any] = {
        "macro_AP": macro_from_per_label(label_ap),
        "per_label_AP": label_ap,
        "threshold": threshold,
    }
    if int(y_true.sum()) > 0:
        result["micro_AP"] = float(average_precision_score(y_true.ravel(), y_score.ravel()))
    else:
        result["micro_AP"] = None
    result["macro_F1"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    result["micro_F1"] = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
    return result


def family_block_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    labels: list[str],
    proteins: list[str],
    split_rows_by_protein: dict[str, dict[str, str]],
    iterations: int,
    seed: int,
) -> dict[str, Any] | None:
    if iterations <= 0:
        return None
    family_by_protein: dict[str, str] = {}
    for protein in proteins:
        row = split_rows_by_protein[protein]
        family = first_present(row, FAMILY_COLUMNS) or "unknown"
        family_by_protein[protein] = family
    families = sorted(set(family_by_protein.values()))
    if len(families) < 2:
        return None

    indices_by_family: dict[str, list[int]] = defaultdict(list)
    for idx, protein in enumerate(proteins):
        indices_by_family[family_by_protein[protein]].append(idx)

    rng = random.Random(seed)
    boot_values: list[float] = []
    for _ in range(iterations):
        sampled_indices: list[int] = []
        for family in (rng.choice(families) for _ in families):
            sampled_indices.extend(indices_by_family[family])
        sampled = np.asarray(sampled_indices, dtype=np.int64)
        label_ap = per_label_ap(y_true[sampled], y_score[sampled], labels)
        macro_ap = macro_from_per_label(label_ap)
        if macro_ap is not None:
            boot_values.append(macro_ap)
    if not boot_values:
        return None
    observed = compute_metrics(y_true, y_score, labels, threshold=0.5)["macro_AP"]
    low, high = np.percentile(np.asarray(boot_values), [2.5, 97.5])
    return {
        "metric": "macro_AP",
        "block_column": "virus_family_or_family",
        "block_count": len(families),
        "iterations": iterations,
        "observed": observed,
        "ci_95": [float(low), float(high)],
    }


def json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, set):
        return sorted(value)
    return value


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred", required=True, type=Path, help="Long-format prediction CSV/TSV.")
    parser.add_argument("--split", required=True, type=Path, help="Test split/truth manifest.")
    parser.add_argument("--labels", required=True, type=Path, help="Label ontology TSV/CSV.")
    parser.add_argument("--truth", type=Path, help="Optional separate truth table.")
    parser.add_argument("--track", required=True, help="ViruFunc Atlas track id or alias.")
    parser.add_argument("--out", required=True, type=Path, help="Output JSON path.")
    parser.add_argument("--split-version", help="Explicit split version string for output JSON.")
    parser.add_argument("--split-column", help="Optional column to filter split table.")
    parser.add_argument("--split-value", help="Value to retain when --split-column is used.")
    parser.add_argument("--allow-missing", action="store_true", help="Treat missing scores as zero.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for F1 metrics.")
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--used-features", help="Comma-separated feature groups used by the method.")
    parser.add_argument("--feature-declaration", help="JSON/YAML-like file declaring used_features.")
    parser.add_argument("--validate-only", action="store_true", help="Validate submission without scoring.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    errors: list[str] = []

    try:
        track = normalize_track(args.track)
        labels = load_labels(args.labels)
        split_rows, _ = load_split_rows(args.split, args.split_column, args.split_value)
        proteins, split_rows_by_protein = extract_proteins(split_rows)
        split_version = infer_split_version(args.split, split_rows, args.split_version)
        scores, prediction_stats = load_predictions(
            args.pred, labels, proteins, args.allow_missing, errors
        )
        used_features = load_used_features(args)
        feature_validation = validate_track_features(track, used_features)
        if feature_validation["forbidden_conflicts"]:
            errors.append(
                "Feature declaration conflicts with track "
                f"{track}: {feature_validation['forbidden_conflicts']}"
            )

        truth = None
        if not args.validate_only:
            truth = load_truth_matrix(
                labels,
                proteins,
                split_rows_by_protein,
                args.truth,
                errors,
            )
            if truth is None:
                errors.append(
                    "No ground-truth labels were found. Provide truth in --split or --truth, "
                    "or use --validate-only for schema validation."
                )

        result: dict[str, Any] = {
            "submission_valid": not errors,
            "errors": errors,
            "track": track,
            "primary_metric": TRACKS[track]["primary_metric"],
            "split_version": split_version,
            "protein_count": len(proteins),
            "label_count": len(labels),
            "labels": labels,
            "prediction_stats": prediction_stats,
            **feature_validation,
        }
        if truth is not None and not errors:
            result.update(compute_metrics(truth, scores, labels, args.threshold))
            result["family_block_CI"] = family_block_ci(
                truth,
                scores,
                labels,
                proteins,
                split_rows_by_protein,
                args.bootstrap_iterations,
                args.bootstrap_seed,
            )
        else:
            result["macro_AP"] = None
            result["macro_F1"] = None
            result["micro_AP"] = None
            result["micro_F1"] = None
            result["per_label_AP"] = {}
            result["family_block_CI"] = None

        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, sort_keys=True, default=json_default), encoding="utf-8")
    except Exception as exc:
        failure = {
            "submission_valid": False,
            "errors": [str(exc)],
            "macro_AP": None,
            "macro_F1": None,
            "micro_AP": None,
            "micro_F1": None,
            "per_label_AP": {},
            "family_block_CI": None,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(failure, indent=2, sort_keys=True), encoding="utf-8")
        print(str(exc), file=sys.stderr)
        return 2
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
