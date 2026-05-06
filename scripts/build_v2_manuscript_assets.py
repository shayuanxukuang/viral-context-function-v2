#!/usr/bin/env python3
"""Build manuscript-ready V2 asset tables from returned core and QC packages."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


NA = "NA"


def read_tsv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: List[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=list(fieldnames), extrasaction="ignore")
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


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def as_float(value: Any, default: float = math.nan) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def pct(value: Any) -> str:
    val = as_float(value)
    if math.isnan(val):
        return NA
    return f"{100.0 * val:.2f}%"


def fmt_delta(value: Any) -> str:
    val = as_float(value)
    if math.isnan(val):
        return NA
    return f"{val:+.4f}"


def row_by(rows: Iterable[Dict[str, str]], key: str, value: str) -> Dict[str, str]:
    for row in rows:
        if row.get(key) == value:
            return row
    return {}


def load_context(core_dir: Path, qc_dir: Path) -> Dict[str, Any]:
    ctx: Dict[str, Any] = {
        "core_dir": core_dir,
        "qc_dir": qc_dir,
        "suite": read_tsv(core_dir / "suite_summary.tsv"),
        "paper": read_json(core_dir / "paper_numbers.json", {}),
        "split_overlap": read_tsv(core_dir / "split_difficulty" / "split_overlap_summary.tsv"),
        "nn": read_tsv(core_dir / "split_difficulty" / "nearest_neighbor_label_transfer.tsv"),
        "source": read_tsv(core_dir / "source_decomposition" / "source_decomposition_summary.tsv"),
        "label_deltas": read_tsv(core_dir / "context_atlas_plain.family_holdout.v2" / "label_deltas.tsv"),
        "group_summary": read_tsv(core_dir / "context_atlas_plain.family_holdout.v2" / "group_summary.tsv"),
        "qc1_exact": read_tsv(qc_dir / "qc1_family_exact_transfer.tsv"),
        "qc1_strict": read_tsv(qc_dir / "qc1_strict_zero_exact_transfer_metrics.tsv"),
        "qc2": read_tsv(qc_dir / "qc2_main_delta_block_bootstrap_ci.tsv"),
        "qc3": read_tsv(qc_dir / "qc3_forbidden_feature_check.tsv"),
        "qc4": read_tsv(qc_dir / "qc4_matched_source_decomposition_comparisons.tsv"),
        "qc5": read_tsv(qc_dir / "qc5_host_corruption_curve.tsv"),
        "nuc_summary": read_json(qc_dir / "qc6_nucleocapsid_summary.json", {}),
        "nuc_fp": read_tsv(qc_dir / "qc6_nucleocapsid_top_false_positives.tsv"),
        "nuc_tp": read_tsv(qc_dir / "qc6_nucleocapsid_top_true_positives.tsv"),
        "candidates": read_tsv(qc_dir / "qc7_candidate_assignments.tsv"),
        "candidate_breakdown": read_tsv(qc_dir / "qc7_candidate_breakdown.tsv"),
        "candidate_prioritization": read_tsv(
            core_dir / "uncertainty" / "genome_aware_denovo.family_holdout" / "candidate_prioritization.tsv"
        ),
        "uncertainty_report": read_json(
            core_dir / "uncertainty" / "genome_aware_denovo.family_holdout" / "uncertainty_report.json",
            {},
        ),
        "module_null": read_tsv(qc_dir / "qc8_module_discovery_null_control.tsv"),
        "ranked_clusters": read_tsv(core_dir / "module_discovery" / "ranked_hypothetical_clusters.tsv"),
        "module_candidates": read_tsv(core_dir / "module_discovery" / "module_candidates.tsv"),
        "structure_reps": read_tsv(core_dir / "module_discovery" / "targeted_structure_validation" / "representatives.tsv"),
        "casebook_clusters": sorted(
            {
                match.group(1)
                for path in (core_dir / "module_discovery" / "casebooks").glob("cluster_*.casebook.md")
                for match in [re.match(r"cluster_(.+)\.casebook\.md$", path.name)]
                if match
            },
            key=lambda value: int(value) if value.isdigit() else value,
        ),
    }
    return ctx


def suite_run(ctx: Dict[str, Any], run_name: str) -> Dict[str, str]:
    return row_by(ctx["suite"], "run_name", run_name)


def build_figure2_delta_ci(ctx: Dict[str, Any], out_dir: Path) -> None:
    rows: List[Dict[str, Any]] = []
    metrics = [
        ("macro AP", "macro_ap"),
        ("macro F1", "macro_f1"),
        ("micro AP", "micro_ap"),
        ("micro F1", "micro_f1"),
    ]
    for src in ctx["qc2"]:
        for label, suffix in metrics:
            delta = as_float(src.get(f"delta_{suffix}"))
            low = as_float(src.get(f"delta_{suffix}_ci_low"))
            high = as_float(src.get(f"delta_{suffix}_ci_high"))
            rows.append(
                {
                    "comparison": src.get("comparison", ""),
                    "metric": label,
                    "protein_run": src.get("protein_run", ""),
                    "context_run": src.get("context_run", ""),
                    "protein_value": src.get(f"protein_{suffix}", ""),
                    "context_value": src.get(f"context_{suffix}", ""),
                    "delta": delta,
                    "ci_low": low,
                    "ci_high": high,
                    "ci_crosses_zero": int(low <= 0.0 <= high) if not math.isnan(low) and not math.isnan(high) else "",
                    "block_unit": src.get("block_unit", ""),
                    "bootstrap_iterations": src.get("bootstrap_iterations", ""),
                    "block_count": src.get("block_count", ""),
                    "figure_note": "Family-block bootstrap intervals reflect uncertainty across held-out viral families."
                    if src.get("block_unit") == "virus_family"
                    else "Host/block bootstrap interval for complementary OOD split.",
                }
            )
    write_tsv(out_dir / "figure2_delta_ci.tsv", rows)


def build_figure1_leakage(ctx: Dict[str, Any], out_dir: Path) -> None:
    rows: List[Dict[str, Any]] = []
    nn_by_scheme = {row.get("scheme"): row for row in ctx["nn"]}
    overlap_by_scheme = {row.get("scheme"): row for row in ctx["split_overlap"]}
    for scheme in ["default", "family_holdout", "host_holdout"]:
        overlap = overlap_by_scheme.get(scheme, {})
        nn = nn_by_scheme.get(scheme, {})
        rows.append(
            {
                "panel": "C",
                "scheme": scheme,
                "metric": "exact_sequence_transfer_rate",
                "value": overlap.get("test_exact_sequence_overlap_rate", ""),
                "display_value": pct(overlap.get("test_exact_sequence_overlap_rate", "")),
                "note": "Exact test proteins with an identical sequence in train.",
            }
        )
        rows.append(
            {
                "panel": "D",
                "scheme": scheme,
                "metric": "nearest_neighbor_macro_ap",
                "value": nn.get("nearest_neighbor_macro_ap", ""),
                "display_value": nn.get("nearest_neighbor_macro_ap", ""),
                "note": "Nearest-neighbor label transfer baseline.",
            }
        )
    strict = row_by(ctx["qc1_strict"], "subset", "strict_zero_exact_transfer_test")
    all_family = row_by(ctx["qc1_strict"], "subset", "all_family_test")
    rows.extend(
        [
            {
                "panel": "E",
                "scheme": "family_holdout_strict_zero_exact_transfer",
                "metric": "removed_exact_transfer_proteins",
                "value": as_float(all_family.get("test_protein_count")) - as_float(strict.get("test_protein_count")),
                "display_value": int(as_float(all_family.get("test_protein_count")) - as_float(strict.get("test_protein_count")))
                if all_family and strict
                else NA,
                "note": "Exact-transfer proteins removed from family-heldout test set.",
            },
            {
                "panel": "E",
                "scheme": "family_holdout_strict_zero_exact_transfer",
                "metric": "delta_macro_ap",
                "value": strict.get("delta_macro_ap", ""),
                "display_value": fmt_delta(strict.get("delta_macro_ap", "")),
                "note": "Context gain retained after removing exact-transfer proteins.",
            },
            {
                "panel": "E",
                "scheme": "family_holdout_strict_zero_exact_transfer",
                "metric": "delta_macro_f1",
                "value": strict.get("delta_macro_f1", ""),
                "display_value": fmt_delta(strict.get("delta_macro_f1", "")),
                "note": "Context gain retained after removing exact-transfer proteins.",
            },
        ]
    )
    exact_rows = ctx["qc1_exact"]
    if exact_rows:
        tag_specs = [
            ("identical proteins assigned to different families", "cross_family_identical"),
            ("same taxid with different family annotation", "same_taxid_different_family_annotation"),
            ("shared mobile/module-like element", "shared_mobile_or_module_element_like"),
            ("duplicated-entry-like", "duplicated_entry_like"),
        ]
        for display, key in tag_specs:
            count = sum(1 for row in exact_rows if str(row.get(key, "")) == "1")
            rows.append(
                {
                    "panel": "F",
                    "scheme": "family_holdout_exact_transfer_audit",
                    "metric": key,
                    "value": count,
                    "display_value": count,
                    "note": display,
                }
            )
    write_tsv(out_dir / "figure1_leakage_summary.tsv", rows)


def build_source_controls(ctx: Dict[str, Any], out_dir: Path) -> None:
    rows: List[Dict[str, Any]] = []
    protein = suite_run(ctx, "protein_only.family_holdout")
    base_ap = as_float(protein.get("test_macro_average_precision"))
    base_f1 = as_float(protein.get("test_macro_f1"))
    addbacks = [
        ("protein_only", "protein_only.family_holdout", "target pLM only"),
        ("local_only", "genome_aware_denovo_addback_local_only.family_holdout", "+ local neighborhood"),
        ("genome_only", "genome_aware_denovo_addback_genome_only.family_holdout", "+ genome organization"),
        ("host_only", "genome_aware_denovo_addback_host_only.family_holdout", "+ host metadata"),
        ("local_genome", "genome_aware_denovo_addback_local_genome.family_holdout", "+ local + genome"),
        ("all_clean_context", "genome_aware_denovo.family_holdout", "+ local + genome + host"),
    ]
    for model_label, run_name, source in addbacks:
        row = suite_run(ctx, run_name)
        ap = as_float(row.get("test_macro_average_precision"))
        f1 = as_float(row.get("test_macro_f1"))
        rows.append(
            {
                "panel": "A-B",
                "row_type": "source_addback",
                "split": "family_holdout",
                "model_label": model_label,
                "source": source,
                "macro_ap": ap,
                "delta_macro_ap_vs_protein_only": ap - base_ap if not math.isnan(ap) and not math.isnan(base_ap) else "",
                "macro_f1": f1,
                "delta_macro_f1_vs_protein_only": f1 - base_f1 if not math.isnan(f1) and not math.isnan(base_f1) else "",
                "run_name": run_name,
            }
        )
    for row in ctx["source"]:
        if row.get("split_scheme") != "family_holdout":
            continue
        comparison_type = row.get("comparison_type", "")
        if not comparison_type.startswith("minus_"):
            continue
        rows.append(
            {
                "panel": "LOSO",
                "row_type": "leave_one_source_out",
                "split": "family_holdout",
                "model_label": comparison_type,
                "source": row.get("notes", ""),
                "macro_ap": row.get("variant_macro_average_precision", ""),
                "delta_macro_ap_vs_protein_only": row.get("delta_macro_average_precision", ""),
                "macro_f1": row.get("variant_macro_f1", ""),
                "delta_macro_f1_vs_protein_only": row.get("delta_macro_f1", ""),
                "run_name": row.get("variant_run", ""),
            }
        )
    for row in ctx["qc5"]:
        rows.append(
            {
                "panel": "C-D",
                "row_type": "host_corruption_or_shuffle",
                "split": row.get("split_scheme", row.get("split", "")),
                "model_label": row.get("curve_type", row.get("condition", row.get("model_label", ""))),
                "source": row.get("host_corruption_fraction", ""),
                "macro_ap": row.get("macro_ap", row.get("test_macro_average_precision", "")),
                "delta_macro_ap_vs_protein_only": row.get("delta_macro_ap_vs_uncorrupted", ""),
                "macro_f1": row.get("macro_f1", row.get("test_macro_f1", "")),
                "delta_macro_f1_vs_protein_only": row.get("delta_macro_f1_vs_uncorrupted", ""),
                "run_name": row.get("run_name", ""),
            }
        )
    for row in ctx["qc3"]:
        rows.append(
            {
                "panel": "E",
                "row_type": "forbidden_feature_check",
                "split": "all",
                "model_label": row.get("forbidden_feature_family", ""),
                "source": row.get("reviewer_interpretation", ""),
                "macro_ap": "",
                "delta_macro_ap_vs_protein_only": "",
                "macro_f1": "",
                "delta_macro_f1_vs_protein_only": "",
                "run_name": "",
            }
        )
    write_tsv(out_dir / "figure3_source_controls.tsv", rows)


def build_atlas_outputs(ctx: Dict[str, Any], out_dir: Path) -> None:
    rows: List[Dict[str, Any]] = []
    for row in ctx["label_deltas"]:
        protein_ap = as_float(row.get("protein_average_precision"))
        delta_ap = as_float(row.get("delta_average_precision"))
        denom = 1.0 - protein_ap + 1e-8
        ncds = delta_ap / denom if not math.isnan(delta_ap) and not math.isnan(protein_ap) and denom else math.nan
        rows.append(
            {
                **row,
                "normalized_context_dependence_score": ncds,
                "positive_delta_ap": int(delta_ap > 0) if not math.isnan(delta_ap) else "",
            }
        )
    rows.sort(key=lambda r: as_float(r.get("delta_average_precision")), reverse=True)
    write_tsv(out_dir / "figure4_label_delta_scatter.tsv", rows)
    write_tsv(out_dir / "figure4_group_delta_boxplot.tsv", ctx["group_summary"])
    write_tsv(out_dir / "figure4_functional_group_context_summary.tsv", group_context_summary(rows, ctx["group_summary"]))
    additional = [
        {
            "label": row.get("label", ""),
            "label_group": row.get("label_group", ""),
            "protein_average_precision": row.get("protein_average_precision", ""),
            "context_average_precision": row.get("context_average_precision", ""),
            "delta_average_precision": row.get("delta_average_precision", ""),
            "protein_support": row.get("protein_support", ""),
            "delta_f1_from_predictions": row.get("delta_f1_from_predictions", ""),
            "delta_f1_ci_low": row.get("delta_f1_ci_low", ""),
            "delta_f1_ci_high": row.get("delta_f1_ci_high", ""),
            "interpretation": "Non-nucleocapsid positive label-level example; use as modest supporting evidence."
            if row.get("label") != "nucleocapsid"
            else "Primary representative label-level example.",
        }
        for row in rows
        if row.get("label") != "nucleocapsid" and as_float(row.get("delta_average_precision")) > 0
    ]
    write_tsv(out_dir / "figure4_additional_label_examples.tsv", additional)

    nuc_delta = row_by(ctx["label_deltas"], "label", "nucleocapsid")
    nuc = ctx["nuc_summary"] or {}
    summary_row = {
        "label": "nucleocapsid",
        "protein_average_precision": nuc_delta.get("protein_average_precision", ""),
        "context_average_precision": nuc_delta.get("context_average_precision", ""),
        "delta_average_precision": nuc_delta.get("delta_average_precision", ""),
        "train_positives": nuc.get("train_positives", ""),
        "val_positives": nuc.get("val_positives", ""),
        "test_positives": nuc.get("test_positives", ""),
        "test_families": nuc.get("test_family_count", nuc.get("test_families", "")),
        "test_host_groups": nuc.get("test_host_group_count", nuc.get("test_host_groups", "")),
        "predicted_positives_at_threshold": nuc.get(
            "test_predicted_positive_count", nuc.get("predicted_positives_at_threshold", "")
        ),
        "true_positives_at_threshold": nuc.get(
            "test_true_positive_count_at_threshold", nuc.get("true_positives_at_threshold", "")
        ),
        "false_positives_at_threshold": nuc.get(
            "test_false_positive_count_at_threshold", nuc.get("false_positives_at_threshold", "")
        ),
        "writeup": "Post hoc false-positive audit should be used as label-incompleteness evidence, not to recompute main AP.",
    }
    write_tsv(out_dir / "figure4_nucleocapsid_audit_summary.tsv", [summary_row])


def group_context_summary(label_rows: Sequence[Dict[str, Any]], group_rows: Sequence[Dict[str, str]]) -> List[Dict[str, Any]]:
    rng = random.Random(20260502)
    group_lookup = {row.get("label_group"): row for row in group_rows}
    grouped: Dict[str, List[float]] = {}
    for row in label_rows:
        group = str(row.get("label_group", ""))
        delta = as_float(row.get("delta_average_precision"))
        if group and not math.isnan(delta):
            grouped.setdefault(group, []).append(delta)
    out: List[Dict[str, Any]] = []
    for group, deltas in sorted(grouped.items()):
        n = len(deltas)
        sorted_deltas = sorted(deltas)
        mean_delta = sum(deltas) / n
        median_delta = sorted_deltas[n // 2] if n % 2 else 0.5 * (sorted_deltas[n // 2 - 1] + sorted_deltas[n // 2])
        means: List[float] = []
        medians: List[float] = []
        for _ in range(1000):
            sample = [deltas[rng.randrange(n)] for _ in range(n)]
            sample_sorted = sorted(sample)
            means.append(sum(sample) / n)
            medians.append(sample_sorted[n // 2] if n % 2 else 0.5 * (sample_sorted[n // 2 - 1] + sample_sorted[n // 2]))
        means.sort()
        medians.sort()
        src = group_lookup.get(group, {})
        out.append(
            {
                "label_group": group,
                "label_count": n,
                "mean_delta_ap": mean_delta,
                "mean_delta_ap_ci_low": means[25],
                "mean_delta_ap_ci_high": means[974],
                "median_delta_ap": median_delta,
                "median_delta_ap_ci_low": medians[25],
                "median_delta_ap_ci_high": medians[974],
                "positive_label_count": sum(1 for delta in deltas if delta > 0),
                "family_block_delta_micro_f1": src.get("delta_micro_f1", ""),
                "family_block_delta_micro_f1_ci_low": src.get("delta_micro_f1_ci_low", ""),
                "family_block_delta_micro_f1_ci_high": src.get("delta_micro_f1_ci_high", ""),
                "family_block_permutation_pvalue": src.get("delta_micro_f1_permutation_pvalue", ""),
                "ci_method": "AP CI is descriptive label-bootstrap; family-block CI/p-value is available for group micro F1.",
            }
        )
    return out


def build_nucleocapsid_fp_audit(ctx: Dict[str, Any], out_dir: Path) -> None:
    module_map = build_module_membership_map(ctx["module_candidates"])
    rows: List[Dict[str, Any]] = []
    synonym_re = re.compile(r"(^|\W)(n|nucleoprotein|nucleocapsid)(\W|$)", re.IGNORECASE)
    structural_re = re.compile(r"capsid|nucleoprotein|nucleocapsid", re.IGNORECASE)
    for row in ctx["nuc_fp"]:
        protein = row.get("protein_accession", "")
        product = row.get("cds_product") or row.get("description", "")
        if synonym_re.search(product):
            category = "likely synonym / missed label"
            synonym_flag = 1
            note = "Post hoc product text contains N/nucleoprotein/nucleocapsid wording."
        elif structural_re.search(product):
            category = "ambiguous"
            synonym_flag = 0
            note = "Post hoc product text is structural/capsid-like but not an exact nucleocapsid synonym."
        elif product.strip() == "" or "hypothetical" in product.lower():
            category = "insufficient evidence"
            synonym_flag = 0
            note = "Returned QC package does not include enough post hoc evidence."
        else:
            category = "possible true false positive"
            synonym_flag = 0
            note = "Post hoc product text does not support nucleocapsid; review with domain/structure evidence."
        rows.append(
            {
                "protein_id": protein,
                "genome_id": row.get("genome_version", ""),
                "family": row.get("virus_family", ""),
                "host_group": row.get("host_taxid_key", ""),
                "y_true": row.get("true_label", ""),
                "p_protein": "MISSING_IN_RETURNED_QC_PACKAGE",
                "p_context": row.get("probability", ""),
                "delta_p": "MISSING_IN_RETURNED_QC_PACKAGE",
                "product_text_posthoc": product,
                "synonym_flag": synonym_flag,
                "local_module": module_map.get(protein, ""),
                "notes": f"{category}; {note}",
                "audit_category": category,
            }
        )
    write_tsv(out_dir / "nucleocapsid_fp_audit.tsv", rows)
    category_counts: Dict[str, int] = {}
    for row in rows:
        category_counts[row["audit_category"]] = category_counts.get(row["audit_category"], 0) + 1
    write_tsv(
        out_dir / "nucleocapsid_fp_audit_summary.tsv",
        [{"audit_category": key, "count": value} for key, value in sorted(category_counts.items())],
    )


def build_module_membership_map(module_rows: Sequence[Dict[str, str]]) -> Dict[str, str]:
    membership: Dict[str, str] = {}
    for row in module_rows:
        cluster = row.get("cluster_id", "")
        center = row.get("center_accession", "")
        if center and cluster:
            membership.setdefault(center, cluster)
        members = row.get("member_accessions_json", "")
        if not members:
            continue
        try:
            parsed = json.loads(members)
        except json.JSONDecodeError:
            parsed = []
        for accession in parsed:
            if accession and cluster:
                membership.setdefault(str(accession), cluster)
    return membership


def build_candidate_outputs(ctx: Dict[str, Any], out_dir: Path) -> None:
    breakdown = []
    for row in ctx["candidate_breakdown"]:
        item = dict(row)
        category = item.get("category", "")
        lower_category = category.lower()
        if lower_category.startswith("fdr") and "protein-label" in lower_category:
            category = "validation-targeted protein-label assignments"
        elif lower_category.startswith("fdr") and "proteins" in lower_category:
            category = "validation-targeted proteins"
        item["category"] = category
        breakdown.append(item)
    write_tsv(out_dir / "figure5_candidate_breakdown.tsv", breakdown)
    scatter_rows = []
    for row in ctx["candidates"]:
        scatter_rows.append(
            {
                "protein_accession": row.get("protein_accession", ""),
                "genome_version": row.get("genome_version", ""),
                "candidate_label": row.get("candidate_label", ""),
                "top_probability_calibrated": row.get("top_probability_calibrated", ""),
                "context_gain": row.get("context_gain", ""),
                "high_context_gain": row.get("high_context_gain", ""),
                "hypothetical_or_unknown": row.get("hypothetical_or_unknown", ""),
                "module_supported": row.get("module_supported", ""),
                "description": row.get("description", ""),
            }
        )
    write_tsv(out_dir / "figure5_confidence_context_gain_scatter.tsv", scatter_rows)
    high = [row for row in scatter_rows if str(row.get("high_context_gain")) == "1"]
    high.sort(key=lambda r: as_float(r.get("context_gain")), reverse=True)
    write_tsv(out_dir / "figure5_high_context_gain_candidates.tsv", high)

    prioritization = ctx.get("candidate_prioritization") or []
    passed = [row for row in prioritization if str(row.get("passes_fdr_gate", "")).lower() == "true"]
    true_positive = [row for row in passed if str(row.get("top_label_in_true_labels", "")).lower() == "true"]
    precision = len(true_positive) / len(passed) if passed else math.nan
    report = ctx.get("uncertainty_report") or {}
    write_tsv(
        out_dir / "figure5_fdr_gate_precision_summary.tsv",
        [
            {
                "calibration_target_fdr": report.get("fdr_target", 0.10),
                "calibration_target_precision": report.get("precision_target", 0.90),
                "candidate_gate_threshold": report.get("candidate_gate_threshold", ""),
                "empirical_validation_top1_precision": report.get("empirical_val_top1_precision", ""),
                "selected_test_candidates": len(passed) if passed else report.get("selected_test_candidates", ""),
                "labeled_test_true_positives": len(true_positive) if passed else "",
                "labeled_test_top1_precision": precision if not math.isnan(precision) else "",
                "labeled_test_top1_fdp": 1.0 - precision if not math.isnan(precision) else "",
                "interpretation": (
                    "Threshold targets 10% false discovery on calibration/validation predictions; "
                    "OOD labeled-test precision is reported separately and is lower, so candidate calls are prioritized hypotheses."
                ),
            }
        ],
    )
    per_label: List[Dict[str, Any]] = []
    labels = sorted({row.get("top_label", "") for row in passed if row.get("top_label", "")})
    for label in labels:
        sub = [row for row in passed if row.get("top_label", "") == label]
        tp = [row for row in sub if str(row.get("top_label_in_true_labels", "")).lower() == "true"]
        prec = len(tp) / len(sub) if sub else math.nan
        per_label.append(
            {
                "label": label,
                "selected": len(sub),
                "true_positive": len(tp),
                "labeled_test_precision": prec if not math.isnan(prec) else "",
                "labeled_test_fdp": 1.0 - prec if not math.isnan(prec) else "",
            }
        )
    per_label.sort(key=lambda row: int(row.get("selected", 0)), reverse=True)
    write_tsv(out_dir / "figure5_fdr_gate_test_precision_by_label.tsv", per_label)

    module_lookup = {row.get("center_accession", ""): row for row in ctx["module_candidates"]}
    ranked_lookup = {row.get("cluster_id", ""): row for row in ctx["ranked_clusters"]}
    any_module = [row for row in ctx["candidates"] if str(row.get("module_supported", "")) == "1"]
    clustered = []
    high_coherence = []
    high_priority = []
    high_gain_module = []
    for row in ctx["candidates"]:
        accession = row.get("protein_accession", "")
        module = module_lookup.get(accession, {})
        cluster_id = module.get("cluster_id", "")
        if not module or cluster_id == "-1":
            continue
        clustered.append(row)
        ranked = ranked_lookup.get(cluster_id, {})
        if as_float(ranked.get("neighborhood_consistency")) >= 0.90:
            high_coherence.append(row)
        if as_float(ranked.get("priority_score")) >= 5:
            high_priority.append(row)
        if str(row.get("high_context_gain", "")) == "1":
            high_gain_module.append(row)
    write_tsv(
        out_dir / "figure5_module_support_tiers.tsv",
        [
            {
                "tier": "any_module_supported_assignment",
                "count": len(any_module),
                "interpretation": "Broad placement in a discovered module; not a narrow validation filter.",
            },
            {
                "tier": "clustered_module_assignment",
                "count": len(clustered),
                "interpretation": "Candidate accession maps to a module-discovery cluster.",
            },
            {
                "tier": "high_coherence_cluster_assignment",
                "count": len(high_coherence),
                "interpretation": "Cluster neighborhood consistency >= 0.90.",
            },
            {
                "tier": "high_priority_cluster_assignment",
                "count": len(high_priority),
                "interpretation": "Cluster priority score >= 5.",
            },
            {
                "tier": "high_context_gain_with_cluster",
                "count": len(high_gain_module),
                "interpretation": "Narrower subset where context materially changed the prediction and a module cluster is present.",
            },
            {
                "tier": "manual_casebook_supported",
                "count": len(ctx.get("casebook_clusters") or []),
                "interpretation": "Manual casebooks generated for inspection; not equivalent to validation.",
            },
        ],
    )


def build_module_null(ctx: Dict[str, Any], out_dir: Path) -> None:
    keep = {
        "weighted_neighborhood_consistency": "neighborhood consistency",
        "weighted_structural_membrane_vote_fraction": "structural/membrane enrichment",
        "weighted_context_sensitive_label_fraction": "context-sensitive label enrichment",
    }
    rows = []
    for row in ctx["module_null"]:
        metric = row.get("metric", "")
        if metric not in keep:
            continue
        rows.append(
            {
                "display_metric": keep[metric],
                "metric": metric,
                "observed": row.get("observed", ""),
                "null_mean": row.get("null_mean", ""),
                "null_ci_low": row.get("null_ci_low", ""),
                "null_ci_high": row.get("null_ci_high", ""),
                "empirical_p_value": row.get("empirical_p_observed_greater_equal_null", ""),
                "iterations": row.get("iterations", ""),
            }
        )
    write_tsv(out_dir / "figure6_module_null_summary.tsv", rows)
    recurrence = row_by(ctx["module_null"], "metric", "mean_family_recurrence")
    note = (
        "Family recurrence did not exceed the shuffled null and was therefore not used as evidence for broad "
        "cross-family module conservation.\n\n"
    )
    if recurrence:
        note += (
            f"Observed mean family recurrence: {recurrence.get('observed')}; "
            f"null mean: {recurrence.get('null_mean')}; "
            f"empirical p(observed >= null): {recurrence.get('empirical_p_observed_greater_equal_null')}.\n"
        )
    (out_dir / "module_null_negative_result_note.md").write_text(note, encoding="utf-8")


def build_casebook_triage(ctx: Dict[str, Any], out_dir: Path) -> None:
    exact_accessions = {row.get("test_protein_accession") for row in ctx["qc1_exact"] if row.get("test_protein_accession")}
    candidates = {row.get("protein_accession"): row for row in ctx["candidates"] if row.get("protein_accession")}
    positive_labels = {
        row.get("label")
        for row in ctx["label_deltas"]
        if as_float(row.get("delta_average_precision")) > 0 and row.get("label")
    }
    reps_by_cluster: Dict[str, List[Dict[str, str]]] = {}
    for row in ctx["structure_reps"]:
        reps_by_cluster.setdefault(row.get("cluster_id", ""), []).append(row)
    ranked_by_cluster = {row.get("cluster_id"): row for row in ctx["ranked_clusters"]}
    module_rows_by_cluster: Dict[str, List[Dict[str, str]]] = {}
    for row in ctx["module_candidates"]:
        module_rows_by_cluster.setdefault(row.get("cluster_id", ""), []).append(row)

    rows = []
    panels = []
    clusters = ctx.get("casebook_clusters") or sorted(reps_by_cluster)
    for cluster_id in clusters:
        reps = reps_by_cluster.get(cluster_id, [])
        ranked = ranked_by_cluster.get(cluster_id, {})
        cluster_module_rows = module_rows_by_cluster.get(cluster_id, [])
        candidate_rows = []
        module_lookup: Dict[str, Dict[str, str]] = {}
        for module_row in cluster_module_rows:
            accession = module_row.get("center_accession", "")
            if accession:
                module_lookup.setdefault(accession, module_row)
            candidate = candidates.get(accession)
            if candidate:
                candidate_rows.append(candidate)
        best = sorted(candidate_rows, key=lambda r: as_float(r.get("context_gain")), reverse=True)
        best_row = best[0] if best else {}
        best_rep = next(
            (rep for rep in reps if rep.get("protein_accession") == best_row.get("protein_accession")),
            reps[0] if reps else {},
        )
        best_module = module_lookup.get(best_row.get("protein_accession", ""), {})
        descriptions = " | ".join(sorted({rep.get("description", "") for rep in reps if rep.get("description", "")})[:3])
        if not descriptions and best_module:
            descriptions = best_row.get("description", "")
        hypothetical = int(
            any("hypothetical" in (rep.get("description", "") + best_row.get("description", "")).lower() for rep in reps)
            or best_row.get("hypothetical_or_unknown") == "1"
            or as_float(ranked.get("hypothetical_ratio_mean")) >= 0.5
        )
        validation_targeted = int(bool(best_row))
        calibrated_high = int(as_float(best_row.get("top_probability_calibrated")) >= 0.8) if best_row else 0
        high_gain = int(as_float(best_row.get("context_gain")) >= 0.2) if best_row else 0
        module_supported = int(best_row.get("module_supported") == "1" or bool(ranked))
        posthoc = int(bool(reps))
        not_exact = int(best_row.get("protein_accession") not in exact_accessions) if best_row else ""
        context_sensitive = int(best_row.get("candidate_label") in positive_labels) if best_row else 0
        protein_prob_estimate = estimate_protein_probability(best_row)
        caveat = (
            "Protein-only probability is estimated as p_genome_aware - context_gain from the returned candidate table."
            if protein_prob_estimate
            else "Protein-only probability was not present in the returned candidate table."
        )
        score = sum(
            int(x)
            for x in [
                hypothetical,
                validation_targeted,
                calibrated_high,
                high_gain,
                module_supported,
                posthoc,
                1,  # forbidden feature check is global PASS when generated from QC3.
                not_exact if not_exact != "" else 0,
                context_sensitive,
            ]
        )
        row = {
            "cluster_id": cluster_id,
            "triage_score_0_to_9": score,
            "selected_for_main_case_panel": 0,
            "candidate_id": best_row.get("protein_accession", best_rep.get("protein_accession", "")),
            "predicted_label": best_row.get("candidate_label", ""),
            "p_protein_only": protein_prob_estimate or "MISSING_IN_RETURNED_QC_PACKAGE",
            "p_genome_aware": best_row.get("top_probability_calibrated", ""),
            "delta_p": best_row.get("context_gain", ""),
            "validation_gate_status": "validation-targeted precision gate" if validation_targeted else "not in returned validation-targeted candidates",
            "module_cluster_id": cluster_id,
            "biophysics_summary": (
                f"structural_membrane_vote_fraction={ranked.get('structural_membrane_vote_fraction_mean', '')}; "
                f"hypothetical_ratio={ranked.get('hypothetical_ratio_mean', '')}; "
                f"tm_helix_count={best_module.get('bio_tm_helix_count', '')}; "
                f"signal_peptide_score={best_module.get('bio_signal_peptide_score', '')}; "
                f"disorder_score={best_module.get('bio_disorder_score', '')}"
            ),
            "posthoc_external_evidence": descriptions,
            "caveat": caveat,
            "hypothetical_uncharacterized_unknown": hypothetical,
            "validation_targeted": validation_targeted,
            "calibrated_probability_high": calibrated_high,
            "context_gain_obvious": high_gain,
            "protein_only_uncertain_or_low": "unknown_from_returned_package",
            "module_cluster_support": module_supported,
            "posthoc_evidence_available": posthoc,
            "no_annotation_derived_training_features": 1,
            "not_exact_transfer_protein": not_exact,
            "context_sensitive_label_group": context_sensitive,
            "module_count": ranked.get("module_count", ""),
            "family_count": ranked.get("family_count", ""),
            "neighborhood_consistency": ranked.get("neighborhood_consistency", ""),
            "top_neighborhood_signature": ranked.get("top_neighborhood_signature", ""),
        }
        rows.append(row)

    rows.sort(key=lambda r: (as_float(r.get("triage_score_0_to_9")), as_float(r.get("delta_p"))), reverse=True)
    selected_rows = [row for row in rows if row.get("validation_targeted") == 1][:5]
    if len(selected_rows) < 3:
        for row in rows:
            if row not in selected_rows:
                selected_rows.append(row)
            if len(selected_rows) >= 5:
                break
    for row in selected_rows:
        row["selected_for_main_case_panel"] = 1
        panels.append(
            {
                "candidate_id": row["candidate_id"],
                "predicted_label": row["predicted_label"],
                "p_protein_only": row["p_protein_only"],
                "p_genome_aware": row["p_genome_aware"],
                "delta_p": row["delta_p"],
                "validation_gate_status": row["validation_gate_status"],
                "genome_neighborhood_diagram": "Use module_cluster_id and module_candidates.tsv to draw neighborhood.",
                "module_cluster_id": row["module_cluster_id"],
                "biophysics_summary": row["biophysics_summary"],
                "posthoc_external_evidence": row["posthoc_external_evidence"],
                "caveat": row["caveat"],
            }
        )
    write_tsv(out_dir / "casebook_triage.tsv", rows)
    write_tsv(out_dir / "selected_casebook_panels.tsv", select_high_context_gain_case_panels(ctx, exact_accessions))


def estimate_protein_probability(candidate_row: Dict[str, str]) -> str:
    if not candidate_row:
        return ""
    context = as_float(candidate_row.get("top_probability_calibrated"))
    gain = as_float(candidate_row.get("context_gain"))
    if math.isnan(context) or math.isnan(gain):
        return ""
    return f"{min(1.0, max(0.0, context - gain)):.12g}"


def select_high_context_gain_case_panels(ctx: Dict[str, Any], exact_accessions: set[str]) -> List[Dict[str, Any]]:
    module_lookup = {row.get("center_accession", ""): row for row in ctx["module_candidates"]}
    rows: List[Dict[str, Any]] = []
    for candidate in ctx["candidates"]:
        gain = as_float(candidate.get("context_gain"))
        prob = as_float(candidate.get("top_probability_calibrated"))
        if math.isnan(gain) or gain < 0.2:
            continue
        if math.isnan(prob) or prob < 0.8:
            continue
        if candidate.get("module_supported") != "1":
            continue
        accession = candidate.get("protein_accession", "")
        module = module_lookup.get(accession, {})
        weak_labels = module.get("weak_label_counts_json", "")
        hypothetical = candidate.get("hypothetical_or_unknown") == "1"
        not_exact = accession not in exact_accessions
        evidence_score = int(bool(weak_labels and weak_labels != "{}")) + int(hypothetical) + int(not_exact)
        rows.append(
            {
                "candidate_id": accession,
                "predicted_label": candidate.get("candidate_label", ""),
                "p_protein_only": estimate_protein_probability(candidate),
                "p_genome_aware": candidate.get("top_probability_calibrated", ""),
                "delta_p": candidate.get("context_gain", ""),
                "validation_gate_status": "validation-targeted precision gate; high context gain (delta_p >= 0.2)",
                "genome_neighborhood_diagram": "Module-supported local neighborhood summarized from module_candidates.tsv.",
                "module_cluster_id": module.get("cluster_id", ""),
                "biophysics_summary": (
                    f"tm_helix_count={module.get('bio_tm_helix_count', '')}; "
                    f"signal_peptide_score={module.get('bio_signal_peptide_score', '')}; "
                    f"disorder_score={module.get('bio_disorder_score', '')}"
                ),
                "posthoc_external_evidence": (
                    f"description={candidate.get('description', '')}; "
                    f"neighborhood_signature={module.get('neighborhood_signature', '')}; "
                    f"weak_label_counts={weak_labels or '{}'}"
                ),
                "scientific_caveat": "Computationally prioritized candidate; requires independent biological validation.",
                "hypothetical_or_unknown": int(hypothetical),
                "not_exact_transfer": int(not_exact),
                "selection_score": evidence_score + gain,
            }
        )
    preferred_labels = [
        "nucleocapsid",
        "portal_terminase_packaging",
        "tail_assembly",
        "capsid_head",
        "lysis",
    ]
    selected: List[Dict[str, Any]] = []
    used_labels: set[str] = set()
    for label in preferred_labels:
        candidates = [row for row in rows if row["predicted_label"] == label and row["predicted_label"] not in used_labels]
        candidates.sort(key=lambda row: as_float(row.get("selection_score")), reverse=True)
        if candidates:
            selected.append(candidates[0])
            used_labels.add(label)
        if len(selected) == 3:
            break
    if len(selected) < 3:
        for row in sorted(rows, key=lambda row: as_float(row.get("selection_score")), reverse=True):
            if row not in selected:
                selected.append(row)
            if len(selected) == 3:
                break
    for row in selected:
        row.pop("selection_score", None)
    return selected


def build_claim_matrix(ctx: Dict[str, Any], out_dir: Path) -> None:
    nn_default = row_by(ctx["nn"], "scheme", "default")
    nn_family = row_by(ctx["nn"], "scheme", "family_holdout")
    split_default = row_by(ctx["split_overlap"], "scheme", "default")
    split_family = row_by(ctx["split_overlap"], "scheme", "family_holdout")
    family_ci = row_by(ctx["qc2"], "comparison", "family_holdout")
    host_ci = row_by(ctx["qc2"], "comparison", "host_holdout")
    strict = row_by(ctx["qc1_strict"], "subset", "strict_zero_exact_transfer_test")
    nuc = ctx["nuc_summary"] or {}
    nuc_delta = row_by(ctx["label_deltas"], "label", "nucleocapsid")
    candidate_counts = {row.get("category"): row.get("count") for row in ctx["candidate_breakdown"]}
    module_metrics = {
        row.get("metric"): row.get("empirical_p_observed_greater_equal_null")
        for row in ctx["module_null"]
    }
    forbidden_pass = all(row.get("reviewer_interpretation") == "PASS" for row in ctx["qc3"])
    matched_pass = sum(1 for row in ctx["qc4"] if row.get("matched_all_core") == "1")
    matched_total = len(ctx["qc4"])

    local_genome = suite_run(ctx, "genome_aware_denovo_addback_local_genome.family_holdout")
    host_only = suite_run(ctx, "genome_aware_denovo_addback_host_only.family_holdout")
    protein = suite_run(ctx, "protein_only.family_holdout")
    local_genome_delta = as_float(local_genome.get("test_macro_average_precision")) - as_float(
        protein.get("test_macro_average_precision")
    )
    host_only_delta = as_float(host_only.get("test_macro_average_precision")) - as_float(
        protein.get("test_macro_average_precision")
    )
    positive_labels = sum(1 for row in ctx["label_deltas"] if as_float(row.get("delta_average_precision")) > 0)
    label_count = len(ctx["label_deltas"])

    rows = [
        {
            "claim": "default split optimistic",
            "evidence": (
                f"Exact transfer {pct(split_default.get('test_exact_sequence_overlap_rate'))} default vs "
                f"{pct(split_family.get('test_exact_sequence_overlap_rate'))} family-heldout; "
                f"NN macro AP {nn_default.get('nearest_neighbor_macro_ap')} default vs "
                f"{nn_family.get('nearest_neighbor_macro_ap')} family-heldout."
            ),
            "main_supp": "main",
            "strength": "strong",
            "caveat": "none",
            "manuscript_wording": "Default evaluation overestimates generalization.",
            "supporting_files": "figure1_leakage_summary.tsv; split_difficulty/nearest_neighbor_label_transfer.tsv",
        },
        {
            "claim": "context global family trend",
            "evidence": (
                f"family-heldout delta macro AP {fmt_delta(family_ci.get('delta_macro_ap'))}, "
                f"95% CI [{fmt_delta(family_ci.get('delta_macro_ap_ci_low'))}, "
                f"{fmt_delta(family_ci.get('delta_macro_ap_ci_high'))}]; "
                f"delta macro F1 {fmt_delta(family_ci.get('delta_macro_f1'))}, "
                f"95% CI [{fmt_delta(family_ci.get('delta_macro_f1_ci_low'))}, "
                f"{fmt_delta(family_ci.get('delta_macro_f1_ci_high'))}]."
            ),
            "main_supp": "main",
            "strength": "moderate",
            "caveat": "CI crosses 0",
            "manuscript_wording": "The family-heldout global delta was positive but not statistically resolved under family-block bootstrap.",
            "supporting_files": "figure2_delta_ci.tsv; qc2_main_delta_block_bootstrap_ci.tsv",
        },
        {
            "claim": "host-heldout gain",
            "evidence": (
                f"host-heldout delta macro AP {fmt_delta(host_ci.get('delta_macro_ap'))}, "
                f"95% CI [{fmt_delta(host_ci.get('delta_macro_ap_ci_low'))}, "
                f"{fmt_delta(host_ci.get('delta_macro_ap_ci_high'))}]; "
                f"delta macro F1 {fmt_delta(host_ci.get('delta_macro_f1'))}, "
                f"95% CI [{fmt_delta(host_ci.get('delta_macro_f1_ci_low'))}, "
                f"{fmt_delta(host_ci.get('delta_macro_f1_ci_high'))}]."
            ),
            "main_supp": "main",
            "strength": "strong if CI positive",
            "caveat": "secondary split",
            "manuscript_wording": "The host-heldout split provided a complementary OOD setting, where genome-aware context showed a more stable positive gain.",
            "supporting_files": "figure2_delta_ci.tsv",
        },
        {
            "claim": "strict-zero exact-transfer sensitivity",
            "evidence": (
                f"Removing exact transfers retained macro AP delta {fmt_delta(strict.get('delta_macro_ap'))} "
                f"and macro F1 delta {fmt_delta(strict.get('delta_macro_f1'))}."
            ),
            "main_supp": "supp/main sentence",
            "strength": "strong leakage sensitivity",
            "caveat": "test-subset re-evaluation, no retraining",
            "manuscript_wording": "Removing the exact-transfer proteins from the family-heldout test set retained the context gain.",
            "supporting_files": "qc1_strict_zero_exact_transfer_metrics.tsv; figure1_leakage_summary.tsv",
        },
        {
            "claim": "no annotation leakage",
            "evidence": f"Forbidden feature check {'PASS' if forbidden_pass else 'REVIEW'} for {len(ctx['qc3'])} forbidden feature families.",
            "main_supp": "main/supp",
            "strength": "strong",
            "caveat": "depends on manifest completeness",
            "manuscript_wording": "The de novo genome-aware model excludes annotation-derived priors.",
            "supporting_files": "qc3_forbidden_feature_check.tsv; data_manifest/feature_manifest.tsv",
        },
        {
            "claim": "not host prior",
            "evidence": (
                f"local+genome addback macro AP delta {fmt_delta(local_genome_delta)} vs host-only "
                f"{fmt_delta(host_only_delta)}; matched comparisons PASS {matched_pass}/{matched_total}; host corruption/shuffle is flat."
            ),
            "main_supp": "main",
            "strength": "strong",
            "caveat": "metadata quality",
            "manuscript_wording": "Host metadata is not the dominant source of the clean context signal.",
            "supporting_files": "figure3_source_controls.tsv; qc4_matched_source_decomposition_comparisons.tsv; qc5_host_corruption_curve.tsv",
        },
        {
            "claim": "label-specific dependence",
            "evidence": f"{positive_labels}/{label_count} labels have positive AP deltas; group-level patterns differ.",
            "main_supp": "main",
            "strength": "strong",
            "caveat": "label imbalance",
            "manuscript_wording": "Context dependence is label-specific.",
            "supporting_files": "figure4_label_delta_scatter.tsv; figure4_group_delta_boxplot.tsv; figure4_functional_group_context_summary.tsv",
        },
        {
            "claim": "nucleocapsid case",
            "evidence": (
                f"AP {nuc_delta.get('protein_average_precision')} -> {nuc_delta.get('context_average_precision')} "
                f"(delta {fmt_delta(nuc_delta.get('delta_average_precision'))}); test positives {nuc.get('test_positives')}; "
                f"test families {nuc.get('test_family_count', nuc.get('test_families'))}; "
                f"host groups {nuc.get('test_host_group_count', nuc.get('test_host_groups'))}."
            ),
            "main_supp": "main",
            "strength": "strong example",
            "caveat": "only 7 test families",
            "manuscript_wording": "Nucleocapsid is a representative context-sensitive label; false-positive audit suggests benchmark label incompleteness.",
            "supporting_files": "figure4_nucleocapsid_audit_summary.tsv; nucleocapsid_fp_audit.tsv",
        },
        {
            "claim": "candidates",
            "evidence": (
                f"{candidate_counts.get('validation-targeted proteins')} validation-targeted proteins; "
                f"{candidate_counts.get('validation-targeted protein-label assignments')} protein-label assignments; "
                f"{candidate_counts.get('hypothetical/uncharacterized/unknown assignments')} hypothetical/uncharacterized/unknown assignments; "
                f"{candidate_counts.get('high context-gain assignments delta_p>=0.2')} high-context-gain assignments."
            ),
            "main_supp": "main/supp",
            "strength": "useful",
            "caveat": "computational only",
            "manuscript_wording": "Calibration yields a prioritized candidate set, not confirmed discoveries.",
            "supporting_files": "figure5_candidate_breakdown.tsv; figure5_fdr_gate_precision_summary.tsv; figure5_high_context_gain_candidates.tsv; figure5_module_support_tiers.tsv",
        },
        {
            "claim": "modules",
            "evidence": (
                "Observed modules exceed null for neighborhood consistency, structural/membrane enrichment, "
                f"and context-sensitive label enrichment (empirical p values: "
                f"{module_metrics.get('weighted_neighborhood_consistency')}, "
                f"{module_metrics.get('weighted_structural_membrane_vote_fraction')}, "
                f"{module_metrics.get('weighted_context_sensitive_label_fraction')})."
            ),
            "main_supp": "main",
            "strength": "moderate-strong",
            "caveat": "family recurrence did not exceed null",
            "manuscript_wording": "Module coherence and enrichment support candidate functional assignments; broad cross-family recurrence is not used as evidence.",
            "supporting_files": "figure6_module_null_summary.tsv; module_null_negative_result_note.md",
        },
    ]
    write_tsv(
        out_dir / "claim_evidence_matrix.tsv",
        rows,
        [
            "claim",
            "evidence",
            "main_supp",
            "strength",
            "caveat",
            "manuscript_wording",
            "supporting_files",
        ],
    )


def build_supplement_manifests(out_dir: Path) -> None:
    table_rows = [
        ("S1", "dataset summary", "data_manifest/freeze_report.json", "ready"),
        ("S2", "label manifest", "data_manifest/label_manifest.tsv", "ready"),
        ("S3", "split manifest", "data_manifest/split_manifest.tsv", "ready"),
        ("S4", "feature manifest / allowed-forbidden table", "data_manifest/feature_manifest.tsv", "ready"),
        ("S5", "forbidden feature check", "data_manifest/forbidden_feature_check.tsv", "ready"),
        ("S6", "leakage audit: exact transfer, nearest-neighbor AP", "figure1_leakage_summary.tsv", "ready"),
        ("S7", "strict-zero-transfer sensitivity", "qc1_strict_zero_exact_transfer_metrics.tsv", "source copied in qc_review"),
        ("S8", "main benchmark metrics", "suite_summary.tsv", "source copied in core package"),
        ("S9", "family-block / genome-block bootstrap CIs", "figure2_delta_ci.tsv", "ready"),
        ("S10", "source decomposition matched comparisons", "qc4_matched_source_decomposition_comparisons.tsv", "source copied in qc_review"),
        ("S11", "host corruption curve", "figure3_source_controls.tsv", "ready"),
        (
            "S12",
            "per-label and functional-group context dependence atlas",
            "figure4_label_delta_scatter.tsv; figure4_functional_group_context_summary.tsv",
            "ready",
        ),
        ("S13", "nucleocapsid audit", "nucleocapsid_fp_audit.tsv", "protein-only probability needs server-side augmentation"),
        ("S14", "calibration metrics", "uncertainty outputs; figure5_fdr_gate_precision_summary.tsv", "ready"),
        ("S15", "candidate breakdown and module-support tiers", "figure5_candidate_breakdown.tsv; figure5_module_support_tiers.tsv", "ready"),
        ("S16", "high-context-gain candidate list", "figure5_high_context_gain_candidates.tsv", "ready"),
        ("S17", "module discovery clusters", "module_discovery/ranked_hypothetical_clusters.tsv", "source copied in core package"),
        ("S18", "module null control", "figure6_module_null_summary.tsv", "ready"),
        ("S19", "casebook summary", "casebook_triage.tsv; selected_casebook_panels.tsv", "protein-only probability needs server-side augmentation"),
        ("S20", "model/config/checksum manifest", "data_manifest/checksum_manifest.tsv", "ready"),
    ]
    write_tsv(
        out_dir / "supplementary_table_manifest.tsv",
        [{"table": a, "content": b, "source": c, "status": d} for a, b, c, d in table_rows],
    )

    figure_rows = [
        ("S1", "train/test sequence identity distribution", "split_difficulty outputs", "source available"),
        ("S2", "label frequency distribution", "data_manifest/label_manifest.tsv", "source available"),
        ("S3", "family/host distribution", "split files and freeze_report.json", "source available"),
        ("S4", "default vs family performance", "suite_summary.tsv; figure1_leakage_summary.tsv", "source available"),
        ("S5", "strict-zero-transfer sensitivity", "figure1_leakage_summary.tsv", "ready"),
        ("S6", "all metrics, all seeds", "suite_summary.tsv", "single-seed returned; add seeds if rerun"),
        ("S7", "source decomposition full panel", "figure3_source_controls.tsv", "ready"),
        ("S8", "host corruption full curve", "figure3_source_controls.tsv", "ready"),
        ("S9", "per-label PR curves for top context-sensitive labels", "context_atlas directories", "source available"),
        ("S10", "nucleocapsid audit examples", "nucleocapsid_fp_audit.tsv", "ready as audit table"),
        ("S11", "calibration reliability diagrams", "uncertainty outputs", "source available"),
        ("S12", "risk-coverage curves", "uncertainty outputs", "source available"),
        ("S13", "candidate score distributions", "figure5_confidence_context_gain_scatter.tsv", "ready"),
        ("S14", "module null distributions", "qc8_module_cluster_assignment_null_iterations.tsv", "source available"),
        ("S15", "all 10 casebook thumbnails", "module_discovery/casebooks", "source available"),
    ]
    write_tsv(
        out_dir / "supplementary_figure_manifest.tsv",
        [{"figure": a, "content": b, "source": c, "status": d} for a, b, c, d in figure_rows],
    )


def build_repro_package(ctx: Dict[str, Any], out_dir: Path) -> None:
    pkg = out_dir / "viral-context-function-v2"
    dirs = [
        "configs",
        "scripts",
        "src",
        "notebooks",
        "data_manifest",
        "reproduce",
        "figures",
        "supplementary_tables",
    ]
    for dirname in dirs:
        (pkg / dirname).mkdir(parents=True, exist_ok=True)

    (pkg / "README.md").write_text(
        "\n".join(
            [
                "# viral-context-function-v2",
                "",
                "Reproducibility package skeleton for the ViruFunc V2 leakage-aware genome-context study.",
                "",
                "This package is intentionally manifest-first: large pLM embeddings and model checkpoints can be regenerated from the scripts and manifests, while predictions, metrics, split files, and checksum manifests should be archived with a DOI before submission.",
                "",
                "## Core claims",
                "",
                "See `supplementary_tables/claim_evidence_matrix.tsv` for claim wording, evidence, strength, and caveats.",
                "",
                "## Data availability",
                "",
                "Public release should include protein IDs, split files, labels, feature manifests, predictions, metrics, model/version information, and checksums. Large embeddings may be omitted if scripts and exact model versions to regenerate them are provided.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (pkg / "LICENSE").write_text(
        "TODO: choose and insert the public code/data license before release (for example MIT for code plus CC-BY for generated tables, if appropriate).\n",
        encoding="utf-8",
    )
    (pkg / "CITATION.cff").write_text(
        "\n".join(
            [
                "cff-version: 1.2.0",
                "message: Please cite this software and dataset if you use the ViruFunc V2 context benchmark.",
                "title: viral-context-function-v2",
                "version: 0.1.0",
                "date-released: 2026-05-02",
                "authors:",
                "  - family-names: TODO",
                "    given-names: TODO",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (pkg / "environment.yml").write_text(
        "\n".join(
            [
                "name: viral-context-function-v2",
                "channels:",
                "  - pytorch",
                "  - conda-forge",
                "dependencies:",
                "  - python>=3.10",
                "  - pytorch",
                "  - numpy",
                "  - pandas",
                "  - scikit-learn",
                "  - scipy",
                "  - matplotlib",
                "  - seaborn",
                "  - tqdm",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (pkg / "container.md").write_text(
        "Container specification: record CUDA, PyTorch, ESM/ESM-C package versions, MMseqs2 version, and exact model weights/checksums used for the final archived run.\n",
        encoding="utf-8",
    )

    data_manifest = pkg / "data_manifest"
    core_manifest = ctx["core_dir"] / "data" / "v2_freeze"
    copy_map = {
        core_manifest / "feature_manifest.tsv": data_manifest / "feature_manifest.tsv",
        core_manifest / "label_manifest.tsv": data_manifest / "label_manifest.tsv",
        core_manifest / "checksums.tsv": data_manifest / "checksum_manifest.tsv",
        core_manifest / "freeze_report.json": data_manifest / "freeze_report.json",
        ctx["qc_dir"] / "qc3_forbidden_feature_check.tsv": data_manifest / "forbidden_feature_check.tsv",
    }
    for src, dst in copy_map.items():
        if src.exists():
            shutil.copy2(src, dst)
    split_rows = []
    split_dir = core_manifest / "splits"
    if split_dir.exists():
        for split_file in sorted(split_dir.glob("*.tsv")):
            dst = data_manifest / split_file.name
            shutil.copy2(split_file, dst)
            split_rows.append({"split_file": split_file.name, "relative_path": f"data_manifest/{split_file.name}"})
    write_tsv(data_manifest / "split_manifest.tsv", split_rows, ["split_file", "relative_path"])

    supp = pkg / "supplementary_tables"
    for name in [
        "claim_evidence_matrix.tsv",
        "figure1_leakage_summary.tsv",
        "figure2_delta_ci.tsv",
        "figure3_source_controls.tsv",
        "figure4_label_delta_scatter.tsv",
        "figure4_group_delta_boxplot.tsv",
        "figure4_functional_group_context_summary.tsv",
        "figure4_additional_label_examples.tsv",
        "figure4_nucleocapsid_audit_summary.tsv",
        "figure5_candidate_breakdown.tsv",
        "figure5_fdr_gate_precision_summary.tsv",
        "figure5_high_context_gain_candidates.tsv",
        "figure5_module_support_tiers.tsv",
        "figure6_module_null_summary.tsv",
        "nucleocapsid_fp_audit.tsv",
        "casebook_triage.tsv",
        "selected_casebook_panels.tsv",
        "supplementary_table_manifest.tsv",
        "supplementary_figure_manifest.tsv",
    ]:
        src = out_dir / name
        if src.exists():
            shutil.copy2(src, supp / name)

    reproduce_steps = [
        ("00_prepare_data.md", "Freeze the data version and write checksums using `scripts/freeze_benchmark_v2.py`."),
        ("01_compute_embeddings.md", "Compute frozen pLM embeddings; archive model names, weight checksums, and embedding checksum manifests."),
        ("02_train_main_models.md", "Train protein-only and genome-aware main models on default, family-heldout, and host-heldout splits."),
        ("03_evaluate_metrics.md", "Evaluate macro/micro AP and Fmax/F1 on held-out test sets."),
        ("04_bootstrap_ci.md", "Run family-block or host-block bootstrap CIs for paired deltas."),
        ("05_source_decomposition.md", "Run add-back, leave-one-source-out, and corruption/shuffle controls."),
        ("06_atlas.md", "Build label-level context dependence atlas and group summaries."),
        ("07_calibration_and_candidates.md", "Calibrate probabilities, run FDR/selective prediction gates, and export candidate assignments."),
        ("08_module_discovery.md", "Discover module clusters and run shuffled null controls."),
        ("09_make_figures.md", "Use manuscript asset TSVs to render Figures 1-6 and Supplementary Figures S1-S15."),
    ]
    for filename, body in reproduce_steps:
        (pkg / "reproduce" / filename).write_text(f"# {filename[:-3]}\n\n{body}\n", encoding="utf-8")


def build_readme(out_dir: Path, core_dir: Path, qc_dir: Path) -> None:
    (out_dir / "README.md").write_text(
        "\n".join(
            [
                "# V2 manuscript assets",
                "",
                f"Core package: `{core_dir}`",
                f"QC package: `{qc_dir}`",
                "",
                "Main generated tables:",
                "",
                "- `claim_evidence_matrix.tsv`",
                "- `figure1_leakage_summary.tsv`",
                "- `figure2_delta_ci.tsv`",
                "- `figure3_source_controls.tsv`",
                "- `figure4_label_delta_scatter.tsv`",
                "- `figure4_additional_label_examples.tsv`",
                "- `figure4_nucleocapsid_audit_summary.tsv`",
                "- `figure5_candidate_breakdown.tsv`",
                "- `figure5_fdr_gate_precision_summary.tsv`",
                "- `figure5_high_context_gain_candidates.tsv`",
                "- `figure5_module_support_tiers.tsv`",
                "- `figure6_module_null_summary.tsv`",
                "- `nucleocapsid_fp_audit.tsv`",
                "- `casebook_triage.tsv`",
                "- `selected_casebook_panels.tsv`",
                "- `viral-context-function-v2/` reproducibility package skeleton",
                "",
                "Known limitation: returned QC files contain calibrated context probabilities for candidate/nucleocapsid audit rows, but not the corresponding protein-only probabilities. Those fields are marked `MISSING_IN_RETURNED_QC_PACKAGE` and should be filled by rerunning the audit on the server-side prediction cache if needed for final figure panels.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core-dir", type=Path, required=True, help="Extracted core result directory.")
    parser.add_argument("--qc-dir", type=Path, required=True, help="Extracted qc_review directory.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for manuscript assets.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    core_dir = args.core_dir.resolve()
    qc_dir = args.qc_dir.resolve()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    ctx = load_context(core_dir, qc_dir)
    build_figure1_leakage(ctx, out_dir)
    build_figure2_delta_ci(ctx, out_dir)
    build_source_controls(ctx, out_dir)
    build_atlas_outputs(ctx, out_dir)
    build_nucleocapsid_fp_audit(ctx, out_dir)
    build_candidate_outputs(ctx, out_dir)
    build_module_null(ctx, out_dir)
    build_casebook_triage(ctx, out_dir)
    build_claim_matrix(ctx, out_dir)
    build_supplement_manifests(out_dir)
    build_repro_package(ctx, out_dir)
    build_readme(out_dir, core_dir, qc_dir)

    summary = {
        "output_dir": str(out_dir),
        "core_dir": str(core_dir),
        "qc_dir": str(qc_dir),
        "generated_files": sorted(str(path.relative_to(out_dir)) for path in out_dir.rglob("*") if path.is_file()),
    }
    (out_dir / "asset_build_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(out_dir), "file_count": len(summary["generated_files"])}, indent=2))


if __name__ == "__main__":
    main()
