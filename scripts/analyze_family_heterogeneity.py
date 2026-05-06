from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from build_context_dependence_atlas_v2 import LABEL_GROUPS, load_metadata, load_predictions, resolve_path, split_scheme_from_run


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze held-out family heterogeneity for pLM context deltas.")
    parser.add_argument("--protein-run", required=True)
    parser.add_argument("--context-run", required=True)
    parser.add_argument("--input", default="data/processed/training/viral_protein_training_index.tsv.gz")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-family-positives", type=int, default=20)
    parser.add_argument("--min-family-proteins", type=int, default=50)
    parser.add_argument("--bootstrap-iterations", type=int, default=200)
    parser.add_argument("--max-forest-families", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in value)


def f1_from_counts(tp: int, fp: int, fn: int) -> float | None:
    denominator = (2 * tp) + fp + fn
    if denominator <= 0:
        return None
    return (2 * tp) / denominator


def metric_from_counts(counts: np.ndarray, label_indices: list[int]) -> tuple[float | None, float | None, int]:
    selected = counts[label_indices]
    positives = int(selected[:, 0].sum() + selected[:, 2].sum())
    micro_f1 = f1_from_counts(int(selected[:, 0].sum()), int(selected[:, 1].sum()), int(selected[:, 2].sum()))
    label_f1_values: list[float] = []
    for label_counts in selected:
        value = f1_from_counts(int(label_counts[0]), int(label_counts[1]), int(label_counts[2]))
        if value is not None:
            label_f1_values.append(value)
    macro_f1 = float(np.mean(label_f1_values)) if label_f1_values else None
    return micro_f1, macro_f1, positives


def scope_label_indices(label_names: list[str]) -> dict[str, list[int]]:
    label_to_idx = {label: idx for idx, label in enumerate(label_names)}
    scopes = {"all": list(range(len(label_names)))}
    for group_name, group_labels in LABEL_GROUPS.items():
        scopes[group_name] = [label_to_idx[label] for label in group_labels if label in label_to_idx]
    return scopes


def protein_counts(row: dict[str, Any], label_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    protein_counts_array = np.zeros((len(label_names), 3), dtype=np.int64)
    context_counts_array = np.zeros((len(label_names), 3), dtype=np.int64)
    true_labels = set(row["true_labels"])
    protein_labels = set(row["protein_predicted_labels"])
    context_labels = set(row["context_predicted_labels"])
    for idx, label_name in enumerate(label_names):
        true_hit = label_name in true_labels
        protein_hit = label_name in protein_labels
        context_hit = label_name in context_labels
        protein_counts_array[idx, 0] = int(true_hit and protein_hit)
        protein_counts_array[idx, 1] = int((not true_hit) and protein_hit)
        protein_counts_array[idx, 2] = int(true_hit and (not protein_hit))
        context_counts_array[idx, 0] = int(true_hit and context_hit)
        context_counts_array[idx, 1] = int((not true_hit) and context_hit)
        context_counts_array[idx, 2] = int(true_hit and (not context_hit))
    return protein_counts_array, context_counts_array


def bootstrap_family_delta(
    protein_per_row: np.ndarray,
    context_per_row: np.ndarray,
    label_indices: list[int],
    iterations: int,
    rng: np.random.Generator,
) -> tuple[float | None, float | None]:
    if iterations <= 0 or protein_per_row.shape[0] == 0:
        return None, None
    values: list[float] = []
    row_count = protein_per_row.shape[0]
    for _ in range(iterations):
        sampled = rng.integers(0, row_count, size=row_count)
        protein_counts_sum = protein_per_row[sampled].sum(axis=0)
        context_counts_sum = context_per_row[sampled].sum(axis=0)
        protein_micro, _, _ = metric_from_counts(protein_counts_sum, label_indices)
        context_micro, _, _ = metric_from_counts(context_counts_sum, label_indices)
        if protein_micro is not None and context_micro is not None:
            values.append(context_micro - protein_micro)
    if not values:
        return None, None
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def binomial_two_sided_pvalue(successes: int, trials: int) -> float | None:
    if trials <= 0:
        return None
    tail = min(successes, trials - successes)
    probability = sum(math.comb(trials, k) for k in range(tail + 1)) / (2**trials)
    return float(min(1.0, 2.0 * probability))


def wilcoxon_pvalue(values: list[float]) -> tuple[float | None, str]:
    nonzero = [float(value) for value in values if abs(float(value)) > 1e-12]
    if len(nonzero) < 2:
        return None, "insufficient_nonzero_pairs"
    try:
        from scipy.stats import wilcoxon

        result = wilcoxon(nonzero, alternative="two-sided", zero_method="wilcox")
        return float(result.pvalue), "scipy_exact_or_approx"
    except Exception:
        ranks = np.argsort(np.argsort(np.abs(nonzero))) + 1
        signed_rank_sum = float(np.sum(np.asarray(ranks)[np.asarray(nonzero) > 0]))
        n = len(nonzero)
        mean = n * (n + 1) / 4.0
        variance = n * (n + 1) * (2 * n + 1) / 24.0
        if variance <= 0:
            return None, "normal_approximation_failed"
        z = abs((signed_rank_sum - mean) / math.sqrt(variance))
        pvalue = math.erfc(z / math.sqrt(2.0))
        return float(pvalue), "normal_approximation"


def add_family_features(rows: list[dict[str, Any]]) -> dict[str, Any]:
    genome_classes = Counter(str(row["baltimore_like_class"]) for row in rows)
    overlap_buckets = Counter(str(row["overlap_density_bucket"]) for row in rows)
    compression_buckets = Counter(str(row["genome_compression_bucket"]) for row in rows)
    return {
        "protein_count": len(rows),
        "enveloped_fraction": float(np.mean([int(row["putative_enveloped"]) for row in rows])),
        "segmented_fraction": float(np.mean([int(row["segmented"]) for row in rows])),
        "overlap_density_mean": float(np.mean([float(row["overlap_density"]) for row in rows])),
        "genome_compression_mean": float(np.mean([float(row["genome_compression"]) for row in rows])),
        "high_overlap_fraction": float(np.mean([str(row["overlap_density_bucket"]) == "high" for row in rows])),
        "high_compression_fraction": float(np.mean([str(row["genome_compression_bucket"]) == "high" for row in rows])),
        "dominant_genome_class": genome_classes.most_common(1)[0][0] if genome_classes else "unknown",
        "dominant_overlap_bucket": overlap_buckets.most_common(1)[0][0] if overlap_buckets else "unknown",
        "dominant_compression_bucket": compression_buckets.most_common(1)[0][0] if compression_buckets else "unknown",
    }


def build_regression_matrix(rows: list[dict[str, Any]]) -> tuple[np.ndarray, list[str]]:
    numeric_columns = [
        "enveloped_fraction",
        "segmented_fraction",
        "overlap_density_mean",
        "genome_compression_mean",
        "high_overlap_fraction",
        "high_compression_fraction",
    ]
    feature_rows: list[list[float]] = []
    feature_names = ["intercept"]
    classes = sorted({str(row["dominant_genome_class"]) for row in rows})
    class_levels = classes[1:] if len(classes) > 1 else []
    feature_names.extend(numeric_columns)
    feature_names.extend([f"genome_class={level}" for level in class_levels])
    for row in rows:
        values = [1.0]
        for column in numeric_columns:
            values.append(float(row[column]))
        for level in class_levels:
            values.append(1.0 if str(row["dominant_genome_class"]) == level else 0.0)
        feature_rows.append(values)
    matrix = np.asarray(feature_rows, dtype=np.float64)
    keep = [0]
    for idx in range(1, matrix.shape[1]):
        if float(np.nanstd(matrix[:, idx])) > 1e-12:
            keep.append(idx)
    return matrix[:, keep], [feature_names[idx] for idx in keep]


def ols_rows(scope: str, rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(rows) < 5:
        return [], {"scope": scope, "row_count": len(rows), "status": "too_few_rows"}
    x_raw, feature_names = build_regression_matrix(rows)
    y = np.asarray([float(row["delta_micro_f1"]) for row in rows], dtype=np.float64)
    x = x_raw.copy()
    for column_idx in range(1, x.shape[1]):
        mean = float(np.mean(x[:, column_idx]))
        std = float(np.std(x[:, column_idx]))
        if std > 1e-12:
            x[:, column_idx] = (x[:, column_idx] - mean) / std
    coefficients, residuals, rank, _ = np.linalg.lstsq(x, y, rcond=None)
    predictions = x @ coefficients
    residual_values = y - predictions
    dof = max(0, x.shape[0] - rank)
    sse = float(np.sum(residual_values**2))
    sst = float(np.sum((y - float(np.mean(y))) ** 2))
    r_squared = 0.0 if sst <= 0 else 1.0 - (sse / sst)
    sigma2 = sse / dof if dof > 0 else float("nan")
    try:
        covariance = sigma2 * np.linalg.pinv(x.T @ x)
        standard_errors = np.sqrt(np.diag(covariance))
    except Exception:
        standard_errors = np.full(coefficients.shape, np.nan)
    try:
        from scipy.stats import t

        pvalues = [float(2.0 * t.sf(abs(coefficients[idx] / standard_errors[idx]), dof)) if standard_errors[idx] > 0 and dof > 0 else None for idx in range(len(coefficients))]
    except Exception:
        pvalues = [None for _ in coefficients]
    coefficient_rows = []
    for idx, name in enumerate(feature_names):
        coefficient_rows.append(
            {
                "scope": scope,
                "term": name,
                "coefficient_on_delta_micro_f1": float(coefficients[idx]),
                "standard_error": None if not np.isfinite(standard_errors[idx]) else float(standard_errors[idx]),
                "pvalue": pvalues[idx],
                "row_count": len(rows),
                "r_squared": r_squared,
            }
        )
    return coefficient_rows, {"scope": scope, "row_count": len(rows), "rank": int(rank), "dof": int(dof), "r_squared": r_squared, "sse": sse}


def maybe_plot_forest(output_dir: Path, scope: str, rows: list[dict[str, Any]], max_families: int) -> str:
    eligible = [row for row in rows if row["scope"] == scope and row["passes_filter"]]
    eligible = sorted(eligible, key=lambda row: (abs(float(row["delta_micro_f1"])), int(row["positive_labels"])), reverse=True)[:max_families]
    eligible = sorted(eligible, key=lambda row: float(row["delta_micro_f1"]))
    if not eligible:
        return ""
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return ""
    values = np.asarray([float(row["delta_micro_f1"]) for row in eligible])
    ci_low = np.asarray([float(row["delta_micro_f1_ci_low"]) if row["delta_micro_f1_ci_low"] not in {"", None} else float(row["delta_micro_f1"]) for row in eligible])
    ci_high = np.asarray([float(row["delta_micro_f1_ci_high"]) if row["delta_micro_f1_ci_high"] not in {"", None} else float(row["delta_micro_f1"]) for row in eligible])
    y = np.arange(len(eligible))
    height = max(6.0, min(24.0, 0.22 * len(eligible) + 2.0))
    fig, ax = plt.subplots(figsize=(9.0, height))
    ax.errorbar(values, y, xerr=[values - ci_low, ci_high - values], fmt="o", color="#24566b", ecolor="#8aa7b2", markersize=3.0, linewidth=0.8)
    ax.axvline(0.0, color="#444444", linewidth=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels([str(row["virus_family"]) for row in eligible], fontsize=6)
    ax.set_xlabel("Context minus protein-only micro-F1")
    ax.set_title(f"Per-family context delta: {scope}")
    fig.tight_layout()
    path = output_dir / f"family_delta_forest.{safe_name(scope)}.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return str(path)


def main() -> int:
    args = parse_args()
    root = repo_root()
    protein_run = resolve_path(root, args.protein_run)
    context_run = resolve_path(root, args.context_run)
    input_path = resolve_path(root, args.input)
    output_dir = resolve_path(root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((context_run / "run_manifest.json").read_text(encoding="utf-8"))
    label_names = [str(label) for label in manifest["label_names"]]
    scopes = scope_label_indices(label_names)

    protein_predictions = load_predictions(protein_run / "test_predictions.tsv.gz")
    context_predictions = load_predictions(context_run / "test_predictions.tsv.gz")
    shared_accessions = sorted(set(protein_predictions) & set(context_predictions))
    metadata = load_metadata(input_path, set(shared_accessions))
    split_scheme = split_scheme_from_run(context_run)

    family_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for accession in shared_accessions:
        if accession not in metadata:
            continue
        row = {
            "protein_accession": accession,
            "true_labels": protein_predictions[accession]["true_labels"],
            "protein_predicted_labels": protein_predictions[accession]["predicted_labels"],
            "context_predicted_labels": context_predictions[accession]["predicted_labels"],
            **metadata[accession],
        }
        family_rows[str(row["virus_family"])].append(row)

    rng = np.random.default_rng(args.seed)
    family_effect_rows: list[dict[str, Any]] = []
    family_feature_rows: dict[str, dict[str, Any]] = {}
    per_family_counts: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    per_family_per_row_counts: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for family, rows in sorted(family_rows.items()):
        family_feature_rows[family] = add_family_features(rows)
        protein_arrays: list[np.ndarray] = []
        context_arrays: list[np.ndarray] = []
        for row in rows:
            protein_array, context_array = protein_counts(row, label_names)
            protein_arrays.append(protein_array)
            context_arrays.append(context_array)
        protein_per_row = np.stack(protein_arrays)
        context_per_row = np.stack(context_arrays)
        protein_sum = protein_per_row.sum(axis=0)
        context_sum = context_per_row.sum(axis=0)
        per_family_counts[family] = (protein_sum, context_sum)
        per_family_per_row_counts[family] = (protein_per_row, context_per_row)

        for scope, label_indices in scopes.items():
            protein_micro, protein_macro, positives = metric_from_counts(protein_sum, label_indices)
            context_micro, context_macro, _ = metric_from_counts(context_sum, label_indices)
            ci_low, ci_high = bootstrap_family_delta(protein_per_row, context_per_row, label_indices, args.bootstrap_iterations, rng)
            passes_filter = len(rows) >= args.min_family_proteins and positives >= args.min_family_positives
            family_effect_rows.append(
                {
                    "scope": scope,
                    "virus_family": family,
                    "protein_count": len(rows),
                    "positive_labels": positives,
                    "protein_micro_f1": protein_micro,
                    "context_micro_f1": context_micro,
                    "delta_micro_f1": None if protein_micro is None or context_micro is None else context_micro - protein_micro,
                    "delta_micro_f1_ci_low": ci_low,
                    "delta_micro_f1_ci_high": ci_high,
                    "protein_macro_f1": protein_macro,
                    "context_macro_f1": context_macro,
                    "delta_macro_f1": None if protein_macro is None or context_macro is None else context_macro - protein_macro,
                    "passes_filter": passes_filter,
                    **family_feature_rows[family],
                }
            )

    test_rows: list[dict[str, Any]] = []
    regression_rows: list[dict[str, Any]] = []
    regression_reports: list[dict[str, Any]] = []
    influence_rows: list[dict[str, Any]] = []

    families = sorted(per_family_counts)
    for scope, label_indices in scopes.items():
        scope_rows = [
            row
            for row in family_effect_rows
            if row["scope"] == scope and row["passes_filter"] and row["delta_micro_f1"] is not None
        ]
        deltas = [float(row["delta_micro_f1"]) for row in scope_rows]
        positive = sum(delta > 0 for delta in deltas)
        negative = sum(delta < 0 for delta in deltas)
        zero = len(deltas) - positive - negative
        sign_pvalue = binomial_two_sided_pvalue(positive, positive + negative)
        wilcoxon_p, wilcoxon_method = wilcoxon_pvalue(deltas)
        test_rows.append(
            {
                "scope": scope,
                "family_count": len(scope_rows),
                "positive_family_count": positive,
                "negative_family_count": negative,
                "zero_family_count": zero,
                "median_delta_micro_f1": None if not deltas else float(np.median(deltas)),
                "mean_delta_micro_f1": None if not deltas else float(np.mean(deltas)),
                "paired_sign_test_pvalue": sign_pvalue,
                "wilcoxon_pvalue": wilcoxon_p,
                "wilcoxon_method": wilcoxon_method,
            }
        )

        rows_for_regression = [
            {
                **family_feature_rows[str(row["virus_family"])],
                "virus_family": row["virus_family"],
                "delta_micro_f1": float(row["delta_micro_f1"]),
            }
            for row in scope_rows
        ]
        coeff_rows, report = ols_rows(scope, rows_for_regression)
        regression_rows.extend(coeff_rows)
        regression_reports.append(report)

        all_protein = np.stack([per_family_counts[family][0] for family in families]).sum(axis=0)
        all_context = np.stack([per_family_counts[family][1] for family in families]).sum(axis=0)
        full_protein_micro, full_protein_macro, full_positives = metric_from_counts(all_protein, label_indices)
        full_context_micro, full_context_macro, _ = metric_from_counts(all_context, label_indices)
        full_delta_micro = None if full_protein_micro is None or full_context_micro is None else full_context_micro - full_protein_micro
        full_delta_macro = None if full_protein_macro is None or full_context_macro is None else full_context_macro - full_protein_macro
        for family in families:
            keep_families = [candidate for candidate in families if candidate != family]
            leave_protein = np.stack([per_family_counts[candidate][0] for candidate in keep_families]).sum(axis=0)
            leave_context = np.stack([per_family_counts[candidate][1] for candidate in keep_families]).sum(axis=0)
            leave_protein_micro, leave_protein_macro, leave_positives = metric_from_counts(leave_protein, label_indices)
            leave_context_micro, leave_context_macro, _ = metric_from_counts(leave_context, label_indices)
            leave_delta_micro = None if leave_protein_micro is None or leave_context_micro is None else leave_context_micro - leave_protein_micro
            leave_delta_macro = None if leave_protein_macro is None or leave_context_macro is None else leave_context_macro - leave_protein_macro
            influence_rows.append(
                {
                    "scope": scope,
                    "left_out_family": family,
                    "left_out_protein_count": family_feature_rows[family]["protein_count"],
                    "full_positive_labels": full_positives,
                    "leave_one_out_positive_labels": leave_positives,
                    "full_delta_micro_f1": full_delta_micro,
                    "leave_one_out_delta_micro_f1": leave_delta_micro,
                    "influence_on_delta_micro_f1": None if full_delta_micro is None or leave_delta_micro is None else leave_delta_micro - full_delta_micro,
                    "full_delta_macro_f1": full_delta_macro,
                    "leave_one_out_delta_macro_f1": leave_delta_macro,
                    "influence_on_delta_macro_f1": None if full_delta_macro is None or leave_delta_macro is None else leave_delta_macro - full_delta_macro,
                }
            )

    plot_paths = {scope: maybe_plot_forest(output_dir, scope, family_effect_rows, args.max_forest_families) for scope in scopes}

    write_tsv(output_dir / "family_effects.tsv", family_effect_rows)
    write_tsv(output_dir / "family_paired_tests.tsv", test_rows)
    write_tsv(output_dir / "family_meta_regression.tsv", regression_rows)
    write_tsv(output_dir / "family_leave_one_out_influence.tsv", influence_rows)
    report = {
        "created_at": timestamp(),
        "protein_run": str(protein_run),
        "context_run": str(context_run),
        "input": str(input_path),
        "split_scheme": split_scheme,
        "shared_proteins": len(shared_accessions),
        "families": len(family_rows),
        "min_family_positives": args.min_family_positives,
        "min_family_proteins": args.min_family_proteins,
        "bootstrap_iterations": args.bootstrap_iterations,
        "paired_tests": test_rows,
        "meta_regression_reports": regression_reports,
        "forest_plots": plot_paths,
    }
    (output_dir / "family_heterogeneity_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "families": len(family_rows), "shared_proteins": len(shared_accessions)}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
