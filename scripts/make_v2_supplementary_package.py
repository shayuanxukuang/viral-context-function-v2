#!/usr/bin/env python3
"""Build submission-ready supplementary tables and figures for ViruFunc V2.

This script turns the current manuscript/QC asset directories into concrete
supplementary attachments instead of an index-only supplement. It only uses
returned files by default, but it can enrich several tables when the full local
protein index and split manifest are available.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import re
import shutil
import textwrap
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LABELS = [
    "polymerase",
    "helicase",
    "protease",
    "capsid_head",
    "tail_fiber_receptor",
    "tail_assembly",
    "portal_terminase_packaging",
    "lysis",
    "envelope_glycoprotein",
    "membrane_matrix",
    "nucleocapsid",
    "integrase_recombinase",
    "nuclease",
    "methyltransferase",
    "ligase",
    "transcription_regulator",
    "polyprotein",
]

PALETTE = {
    "protein": "#4C78A8",
    "context": "#F58518",
    "support": "#54A24B",
    "risk": "#E45756",
    "neutral": "#8A8F98",
    "accent": "#B279A2",
    "light": "#F4F6F8",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core-dir", type=Path, required=True, help="Returned V2 core result directory.")
    parser.add_argument("--qc-dir", type=Path, required=True, help="Returned qc_review directory.")
    parser.add_argument("--assets-dir", type=Path, required=True, help="Directory from build_v2_manuscript_assets.py.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output supplementary package directory.")
    parser.add_argument("--protein-index", type=Path, default=Path("data/processed/training/viral_protein_training_index.tsv.gz"))
    parser.add_argument("--split-manifest", type=Path, default=Path("data/processed/splits/viral_protein_strict_splits.tsv.gz"))
    parser.add_argument("--candidate-evidence-dir", type=Path, help="Optional directory from build_candidate_case_evidence.py.")
    parser.add_argument("--make-zip", action="store_true", help="Write supplementary_tables.zip and supplementary_figures.zip.")
    return parser.parse_args()


def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("r", encoding="utf-8", newline="")


def read_tsv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
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


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def freeze_file(core_dir: Path, name: str) -> Path:
    """Resolve freeze artifacts in either run-output or public-release layout."""
    data_manifest_name = "checksum_manifest.tsv" if name == "checksums.tsv" else name
    candidates = [
        core_dir / "data" / "v2_freeze" / name,
        core_dir / "data_manifest" / data_manifest_name,
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def read_freeze_report(core_dir: Path) -> dict[str, Any]:
    return read_json(freeze_file(core_dir, "freeze_report.json"), {}) or {}


def copy_or_empty(src: Path, dst: Path, note: str = "") -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.exists():
        shutil.copy2(src, dst)
    else:
        write_tsv(dst, [{"missing_source": str(src), "note": note or "source file not available"}])


def dataframe(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, sep="\t")


def savefig(fig: plt.Figure, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=1.0)
    fig.savefig(out_dir / f"{name}.png", dpi=220, bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def panel(ax: plt.Axes, title: str) -> None:
    ax.set_title(title, loc="left", fontweight="bold", fontsize=10)


def write_dataset_summary(core_dir: Path, tables_dir: Path) -> None:
    freeze = read_freeze_report(core_dir)
    rows = [
        {"quantity": "created_at", "value": freeze.get("created_at", "")},
        {"quantity": "protein_count", "value": freeze.get("protein_count", "")},
        {"quantity": "genome_count", "value": freeze.get("genome_count", "")},
        {"quantity": "family_count", "value": freeze.get("family_count", "")},
        {"quantity": "host_group_count", "value": freeze.get("host_group_count", "")},
        {"quantity": "label_count", "value": freeze.get("label_count", "")},
        {"quantity": "low_frequency_label_count", "value": freeze.get("low_frequency_label_count", "")},
        {"quantity": "protein_index_source", "value": freeze.get("inputs", {}).get("protein_index", "")},
        {"quantity": "genome_index_source", "value": freeze.get("inputs", {}).get("genome_index", "")},
        {"quantity": "strict_splits_source", "value": freeze.get("inputs", {}).get("strict_splits", "")},
        {"quantity": "taxonomy_source", "value": freeze.get("inputs", {}).get("taxonomy", "")},
    ]
    write_tsv(tables_dir / "S1_dataset_summary.tsv", rows)

    label_counts = freeze.get("label_positive_counts", {})
    write_tsv(
        tables_dir / "S1_label_positive_counts.tsv",
        [{"label": label, "positive_count": count} for label, count in sorted(label_counts.items())],
    )

    split_rows = []
    for scheme, parts in (freeze.get("split_label_counts", {}) or {}).items():
        for split, labels in parts.items():
            for label, count in labels.items():
                split_rows.append({"scheme": scheme, "split": split, "label": label, "positive_count": count})
    write_tsv(tables_dir / "S1_split_label_counts.tsv", split_rows)


def write_split_manifest(split_manifest: Path, tables_dir: Path) -> None:
    if not split_manifest.exists():
        write_tsv(tables_dir / "S3_split_manifest.tsv", [{"missing_source": str(split_manifest)}])
        return
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    families: dict[str, set[str]] = defaultdict(set)
    hosts: dict[str, set[str]] = defaultdict(set)
    with open_text(split_manifest) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            for col, scheme in [
                ("family_holdout_split", "family_holdout"),
                ("host_taxid_holdout_split", "host_holdout"),
                ("host_supergroup_holdout_split", "host_supergroup_holdout"),
                ("sequence_sketch_holdout_split", "sequence_sketch_holdout"),
                ("species_holdout_split", "species_holdout"),
            ]:
                split = row.get(col, "")
                if not split:
                    continue
                counts[scheme][split] += 1
                families[f"{scheme}:{split}"].add(row.get("virus_family", ""))
                hosts[f"{scheme}:{split}"].add(row.get("host_supergroup", ""))
    rows = []
    for scheme, counter in sorted(counts.items()):
        for split, count in sorted(counter.items()):
            key = f"{scheme}:{split}"
            rows.append(
                {
                    "scheme": scheme,
                    "split": split,
                    "protein_count": count,
                    "family_count": len({x for x in families[key] if x}),
                    "host_supergroup_count": len({x for x in hosts[key] if x}),
                }
            )
    write_tsv(tables_dir / "S3_split_manifest.tsv", rows)


def stream_metadata_for_accessions(index_path: Path, accessions: set[str]) -> dict[str, dict[str, str]]:
    if not index_path.exists() or not accessions:
        return {}
    out: dict[str, dict[str, str]] = {}
    with open_text(index_path) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            accession = row.get("protein_accession", "")
            if accession in accessions:
                out[accession] = row
                if len(out) == len(accessions):
                    break
    return out


def stream_split_for_accessions(split_path: Path, accessions: set[str]) -> dict[str, dict[str, str]]:
    if not split_path.exists() or not accessions:
        return {}
    out: dict[str, dict[str, str]] = {}
    with open_text(split_path) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            accession = row.get("protein_accession", "")
            if accession in accessions:
                out[accession] = row
                if len(out) == len(accessions):
                    break
    return out


def write_candidate_tables(
    assets_dir: Path,
    core_dir: Path,
    qc_dir: Path,
    protein_index: Path,
    split_manifest: Path,
    tables_dir: Path,
) -> None:
    copy_or_empty(assets_dir / "figure5_candidate_breakdown.tsv", tables_dir / "S15_candidate_breakdown.tsv")
    copy_or_empty(assets_dir / "figure5_module_support_tiers.tsv", tables_dir / "S15_module_support_tiers.tsv")

    high = read_tsv(assets_dir / "figure5_high_context_gain_candidates.tsv")
    module_rows = {row.get("center_accession", ""): row for row in read_tsv(core_dir / "module_discovery" / "module_candidates.tsv")}
    exact = {row.get("test_protein_accession", "") for row in read_tsv(qc_dir / "qc1_family_exact_transfer.tsv")}
    thresholds = {
        row.get("label", ""): row
        for row in read_tsv(core_dir / "uncertainty" / "genome_aware_denovo.family_holdout" / "per_label_thresholds.tsv")
    }
    accessions = {row.get("protein_accession", "") for row in high}
    metadata = stream_metadata_for_accessions(protein_index, accessions)
    splits = stream_split_for_accessions(split_manifest, accessions)
    enriched = []
    for row in high:
        accession = row.get("protein_accession", "")
        meta = metadata.get(accession, {})
        split = splits.get(accession, {})
        module = module_rows.get(accession, {})
        label = row.get("candidate_label", "")
        p_context = as_float(row.get("top_probability_calibrated"))
        gain = as_float(row.get("context_gain"))
        p_protein = p_context - gain if not math.isnan(p_context) and not math.isnan(gain) else ""
        thr = thresholds.get(label, {})
        enriched.append(
            {
                "protein_id": accession,
                "genome_id": row.get("genome_version", meta.get("genome_version", "")),
                "family": split.get("virus_family", module.get("virus_family", "")),
                "host_group": split.get("host_supergroup", ""),
                "predicted_label": label,
                "p_protein_only_estimated": p_protein,
                "p_context": row.get("top_probability_calibrated", ""),
                "delta_p": row.get("context_gain", ""),
                "calibrated_probability": row.get("top_probability_calibrated", ""),
                "precision_target_threshold": thr.get("temperature_scaled_precision_target_threshold", ""),
                "fdr_gate_status": "validation-targeted gate; high context gain",
                "description": row.get("description", meta.get("protein_description", "")),
                "hypothetical_or_uncharacterized": row.get("hypothetical_or_unknown", ""),
                "module_cluster_id": module.get("cluster_id", ""),
                "exact_transfer_flag": int(accession in exact),
                "nearest_train_identity": "",
                "notes": "nearest_train_identity filled by homology baseline script when available",
            }
        )
    write_tsv(tables_dir / "S16_high_context_gain_candidates.tsv", enriched)


def write_module_and_casebook_tables(core_dir: Path, assets_dir: Path, tables_dir: Path, out_dir: Path) -> None:
    clusters = read_tsv(core_dir / "module_discovery" / "ranked_hypothetical_clusters.tsv")
    module_candidates = read_tsv(core_dir / "module_discovery" / "module_candidates.tsv")
    counts_by_cluster: dict[str, Counter[str]] = defaultdict(Counter)
    examples: dict[str, list[str]] = defaultdict(list)
    for row in module_candidates:
        cid = row.get("cluster_id", "")
        counts_by_cluster[cid]["candidate_count"] += 1
        if as_float(row.get("hypothetical_ratio")) > 0:
            counts_by_cluster[cid]["hypothetical_count"] += 1
        if len(examples[cid]) < 5:
            examples[cid].append(row.get("center_accession", ""))
    rows = []
    for row in clusters:
        cid = row.get("cluster_id", "")
        rows.append(
            {
                "cluster_id": cid,
                "cluster_size": row.get("module_count", ""),
                "number_of_families": row.get("family_count", ""),
                "hypothetical_ratio_mean": row.get("hypothetical_ratio_mean", ""),
                "structural_membrane_enrichment": row.get("structural_membrane_vote_fraction_mean", ""),
                "neighborhood_consistency": row.get("neighborhood_consistency", ""),
                "top_neighborhood_signature": row.get("top_neighborhood_signature", ""),
                "priority_score": row.get("priority_score", ""),
                "candidate_count": counts_by_cluster[cid].get("candidate_count", ""),
                "hypothetical_count": counts_by_cluster[cid].get("hypothetical_count", ""),
                "example_proteins": ";".join(examples[cid]),
            }
        )
    write_tsv(tables_dir / "S17_module_clusters.tsv", rows)

    copy_or_empty(assets_dir / "casebook_triage.tsv", tables_dir / "S19_casebook_summary.tsv")
    copy_or_empty(assets_dir / "selected_casebook_panels.tsv", tables_dir / "S19_selected_casebook_panels.tsv")
    casebook_dir = core_dir / "module_discovery" / "casebooks"
    dst_casebook_dir = out_dir / "casebooks"
    dst_casebook_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(casebook_dir.glob("*.casebook.md")):
        shutil.copy2(path, dst_casebook_dir / path.name)


def write_table_package(args: argparse.Namespace, tables_dir: Path) -> None:
    core = args.core_dir
    qc = args.qc_dir
    assets = args.assets_dir
    write_dataset_summary(core, tables_dir)
    copy_or_empty(freeze_file(core, "label_manifest.tsv"), tables_dir / "S2_label_manifest.tsv")
    write_split_manifest(args.split_manifest, tables_dir)
    copy_or_empty(freeze_file(core, "feature_manifest.tsv"), tables_dir / "S4_feature_manifest.tsv")
    copy_or_empty(qc / "qc3_forbidden_feature_check.tsv", tables_dir / "S5_forbidden_feature_check.tsv")
    copy_or_empty(assets / "figure1_leakage_summary.tsv", tables_dir / "S6_leakage_audit.tsv")
    copy_or_empty(qc / "qc1_strict_zero_exact_transfer_metrics.tsv", tables_dir / "S7_strict_zero_exact_transfer_sensitivity.tsv")
    copy_or_empty(core / "suite_summary.tsv", tables_dir / "S8_main_benchmark_metrics.tsv")
    copy_or_empty(assets / "figure2_delta_ci.tsv", tables_dir / "S9_block_bootstrap_ci.tsv")
    copy_or_empty(qc / "qc4_matched_source_decomposition_comparisons.tsv", tables_dir / "S10_matched_source_decomposition_comparisons.tsv")
    copy_or_empty(assets / "figure3_source_controls.tsv", tables_dir / "S11_host_corruption_and_source_controls.tsv")
    copy_or_empty(assets / "figure4_label_delta_scatter.tsv", tables_dir / "S12_full_per_label_atlas.tsv")
    copy_or_empty(assets / "figure4_functional_group_context_summary.tsv", tables_dir / "S12_functional_group_context_summary.tsv")
    copy_or_empty(assets / "nucleocapsid_fp_audit.tsv", tables_dir / "S13_nucleocapsid_fp_audit.tsv")
    copy_or_empty(qc / "qc6_nucleocapsid_top_true_positives.tsv", tables_dir / "S13_nucleocapsid_top_true_positives.tsv")
    copy_or_empty(qc / "qc6_nucleocapsid_top_false_positives.tsv", tables_dir / "S13_nucleocapsid_top_false_positives.tsv")
    copy_or_empty(core / "uncertainty" / "genome_aware_denovo.family_holdout" / "coverage_curves.tsv", tables_dir / "S14_coverage_curves.tsv")
    copy_or_empty(core / "uncertainty" / "genome_aware_denovo.family_holdout" / "per_label_thresholds.tsv", tables_dir / "S14_per_label_thresholds.tsv")
    copy_or_empty(assets / "figure5_fdr_gate_precision_summary.tsv", tables_dir / "S14_fdr_gate_precision_summary.tsv")
    write_candidate_tables(assets, core, qc, args.protein_index, args.split_manifest, tables_dir)
    write_module_and_casebook_tables(core, assets, tables_dir, args.output_dir)
    copy_or_empty(assets / "figure6_module_null_summary.tsv", tables_dir / "S18_module_null_control.tsv")
    copy_or_empty(qc / "qc8_module_cluster_assignment_null_iterations.tsv", tables_dir / "S18_module_null_iterations.tsv")
    evidence_dir = args.candidate_evidence_dir
    if evidence_dir is None:
        evidence_dir = args.output_dir.parent / "candidate_case_evidence"
    if evidence_dir.exists():
        copy_or_empty(evidence_dir / "candidate_case_evidence.tsv", tables_dir / "S19_candidate_case_evidence.tsv")
        copy_or_empty(evidence_dir / "candidate_case_neighborhoods.tsv", tables_dir / "S19_candidate_case_neighborhoods.tsv")
    copy_or_empty(freeze_file(core, "checksums.tsv"), tables_dir / "S20_checksum_manifest.tsv")
    copy_or_empty(core / "v2_suite_manifest.json", tables_dir / "S20_v2_suite_manifest.json")
    copy_or_empty(core / "frozen_benchmark_v2_runs.tsv", tables_dir / "S20_frozen_benchmark_run_registry.tsv")


def as_float(value: Any, default: float = math.nan) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def figure_s1(core_dir: Path, assets_dir: Path, fig_dir: Path) -> None:
    overlap = dataframe(core_dir / "split_difficulty" / "split_overlap_summary.tsv")
    nn = dataframe(core_dir / "split_difficulty" / "nearest_neighbor_label_transfer.tsv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    if not overlap.empty:
        sub = overlap[overlap["scheme"].isin(["default", "family_holdout", "host_holdout"])]
        axes[0].bar(sub["scheme"], sub["test_exact_sequence_overlap_rate"].astype(float) * 100, color=PALETTE["support"])
        axes[0].set_ylabel("exact sequence transfer (%)")
        axes[0].tick_params(axis="x", rotation=20)
    panel(axes[0], "S1A. Exact train-test transfer")
    if not nn.empty:
        axes[1].bar(nn["scheme"], nn["nearest_neighbor_macro_ap"].astype(float), color=PALETTE["neutral"])
        axes[1].set_ylabel("nearest-neighbor macro AP")
        axes[1].tick_params(axis="x", rotation=20)
    panel(axes[1], "S1B. NN label-transfer difficulty")
    savefig(fig, fig_dir, "S1_sequence_transfer_and_nn_difficulty")


def figure_s2(core_dir: Path, fig_dir: Path) -> None:
    freeze = read_freeze_report(core_dir)
    counts = freeze.get("label_positive_counts", {})
    rows = sorted(counts.items(), key=lambda kv: kv[1])
    fig, ax = plt.subplots(figsize=(8, 5))
    if rows:
        ax.barh([r[0] for r in rows], [r[1] for r in rows], color=PALETTE["context"])
        ax.set_xlabel("positive proteins")
        ax.set_xscale("log")
    else:
        ax.axis("off")
        ax.text(0.02, 0.5, "Label counts unavailable; check freeze_report.json.", transform=ax.transAxes)
    panel(ax, "S2. Label frequency distribution")
    savefig(fig, fig_dir, "S2_label_frequency_distribution")


def figure_s3(core_dir: Path, fig_dir: Path) -> None:
    label_dist = dataframe(core_dir / "split_difficulty" / "label_distribution_by_split.tsv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    if not label_dist.empty:
        sizes = label_dist.drop_duplicates(["scheme", "split"])[["scheme", "split", "partition_size"]]
        pivot = sizes.pivot(index="scheme", columns="split", values="partition_size").fillna(0)
        pivot.plot(kind="bar", stacked=True, ax=axes[0], color=[PALETTE["support"], PALETTE["context"], PALETTE["neutral"]])
        axes[0].set_ylabel("proteins")
        axes[0].tick_params(axis="x", rotation=20)
        test = label_dist[label_dist["split"] == "test"]
        by_scheme = test.groupby("scheme")["positive_rate"].mean().sort_values()
        axes[1].bar(by_scheme.index, by_scheme.values, color=PALETTE["neutral"])
        axes[1].set_ylabel("mean label positive rate in test")
        axes[1].tick_params(axis="x", rotation=20)
    panel(axes[0], "S3A. Partition sizes")
    panel(axes[1], "S3B. Test label prevalence")
    savefig(fig, fig_dir, "S3_split_distribution")


def figure_s4(core_dir: Path, fig_dir: Path) -> None:
    suite = dataframe(core_dir / "suite_summary.tsv")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    if not suite.empty:
        keep = suite[suite["run_name"].astype(str).str.contains("protein_only|genome_aware_denovo", regex=True)].copy()
        keep = keep[keep["split_scheme"].isin(["default_hash", "family_holdout", "host_holdout"])]
        keep["model"] = np.where(keep["run_name"].astype(str).str.contains("genome_aware_denovo"), "genome-aware", "protein-only")
        pivot = keep.pivot_table(index="split_scheme", columns="model", values="test_macro_average_precision", aggfunc="first")
        pivot.plot(kind="bar", ax=ax, color=[PALETTE["context"], PALETTE["protein"]])
        ax.set_ylabel("macro AP")
        ax.tick_params(axis="x", rotation=20)
    panel(ax, "S4. Default versus heldout benchmark performance")
    savefig(fig, fig_dir, "S4_default_vs_family_performance")


def figure_s5(assets_dir: Path, fig_dir: Path) -> None:
    leakage = dataframe(assets_dir / "figure1_leakage_summary.tsv")
    strict = leakage[(leakage["panel"] == "E") & (leakage["metric"].isin(["delta_macro_ap", "delta_macro_f1"]))]
    fig, ax = plt.subplots(figsize=(5, 4))
    if not strict.empty:
        labels = strict["metric"].str.replace("delta_", "", regex=False).str.replace("_", " ")
        vals = strict["value"].astype(float)
        ax.bar(labels, vals, color=PALETTE["context"])
        ax.axhline(0, color="#444", lw=0.8)
        ax.set_ylabel("context - protein-only")
    panel(ax, "S5. Strict-zero exact-transfer sensitivity")
    savefig(fig, fig_dir, "S5_strict_zero_transfer_sensitivity")


def figure_s7(assets_dir: Path, fig_dir: Path) -> None:
    source = dataframe(assets_dir / "figure3_source_controls.tsv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    add = source[source["row_type"] == "source_addback"].copy()
    if not add.empty:
        axes[0].barh(add["model_label"], add["delta_macro_ap_vs_protein_only"].astype(float), color=PALETTE["context"])
        axes[0].axvline(0, color="#444", lw=0.8)
        axes[0].set_xlabel("delta macro AP")
    panel(axes[0], "S7A. Full add-back source decomposition")
    loso = source[source["row_type"] == "leave_one_source_out"].copy()
    if not loso.empty:
        axes[1].barh(loso["model_label"], loso["delta_macro_ap_vs_protein_only"].astype(float), color=PALETTE["neutral"])
        axes[1].axvline(0, color="#444", lw=0.8)
        axes[1].set_xlabel("delta macro AP")
    panel(axes[1], "S7B. Leave-one-source-out")
    savefig(fig, fig_dir, "S7_full_source_decomposition")


def figure_s8(assets_dir: Path, fig_dir: Path) -> None:
    source = dataframe(assets_dir / "figure3_source_controls.tsv")
    curve = source[(source["row_type"] == "host_corruption_or_shuffle") & (source["model_label"] == "host_corruption")].copy()
    fig, ax = plt.subplots(figsize=(6, 4))
    if not curve.empty:
        curve = curve[curve["source"].astype(str).str.match(r"^[0-9.]+$")]
        for split, sub in curve.groupby("split"):
            sub = sub.assign(frac=sub["source"].astype(float)).sort_values("frac")
            ax.plot(sub["frac"], sub["macro_ap"].astype(float), marker="o", label=str(split).replace("_", "-"))
        ax.legend(frameon=False)
        ax.set_xlabel("host corruption fraction")
        ax.set_ylabel("macro AP")
    panel(ax, "S8. Host corruption full curve")
    savefig(fig, fig_dir, "S8_host_corruption_curve")


def figure_s9(qc_dir: Path, fig_dir: Path) -> None:
    pr = dataframe(qc_dir / "qc6_nucleocapsid_pr_curve.tsv")
    fig, ax = plt.subplots(figsize=(5, 4))
    if not pr.empty:
        ax.plot(pr["recall"], pr["precision"], color=PALETTE["context"], lw=2)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("recall")
        ax.set_ylabel("precision")
    panel(ax, "S9. Nucleocapsid precision-recall curve")
    savefig(fig, fig_dir, "S9_per_label_pr_curves")


def figure_s10(qc_dir: Path, fig_dir: Path) -> None:
    fp = dataframe(qc_dir / "qc6_nucleocapsid_top_false_positives.tsv")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if not fp.empty:
        examples = fp.head(8).copy()
        text = "\n".join(
            f"{r.protein_accession}: {str(r.description)[:70]}"
            for r in examples.itertuples(index=False)
        )
        ax.text(0.02, 0.95, text, va="top", ha="left", family="monospace", fontsize=8, transform=ax.transAxes)
    ax.axis("off")
    panel(ax, "S10. Nucleocapsid audit examples")
    savefig(fig, fig_dir, "S10_nucleocapsid_audit_examples")


def figure_s11(core_dir: Path, fig_dir: Path) -> None:
    cand = dataframe(core_dir / "uncertainty" / "genome_aware_denovo.family_holdout" / "candidate_prioritization.tsv")
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    if not cand.empty:
        cand["prob_bin"] = pd.cut(cand["top_probability_calibrated"].astype(float), bins=np.linspace(0, 1, 11), include_lowest=True)
        rows = cand.groupby("prob_bin", observed=True).agg(
            mean_prob=("top_probability_calibrated", "mean"),
            precision=("top_label_in_true_labels", lambda x: np.mean([str(v).lower() == "true" for v in x])),
            n=("top_label_in_true_labels", "size"),
        )
        rows = rows[rows["n"] > 0]
        ax.plot([0, 1], [0, 1], color="#444", ls="--", lw=1)
        ax.scatter(rows["mean_prob"], rows["precision"], s=np.clip(rows["n"], 10, 400), alpha=0.7, color=PALETTE["context"])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("mean calibrated probability")
        ax.set_ylabel("empirical top-label precision")
    panel(ax, "S11. Top-label reliability diagram")
    savefig(fig, fig_dir, "S11_reliability_diagram")


def figure_s12(core_dir: Path, fig_dir: Path) -> None:
    cov = dataframe(core_dir / "uncertainty" / "genome_aware_denovo.family_holdout" / "coverage_curves.tsv")
    fig, ax = plt.subplots(figsize=(6, 4))
    if not cov.empty:
        ax.plot(cov["coverage"], cov["top1_precision"], marker="o", label="top-label precision")
        ax.plot(cov["coverage"], cov["micro_f1"], marker="o", label="micro F1")
        ax.set_xlabel("coverage")
        ax.set_ylim(0, 1)
        ax.legend(frameon=False)
    panel(ax, "S12. Risk-coverage curve")
    savefig(fig, fig_dir, "S12_risk_coverage_curve")


def figure_s13(assets_dir: Path, fig_dir: Path) -> None:
    scatter = dataframe(assets_dir / "figure5_confidence_context_gain_scatter.tsv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    if not scatter.empty:
        gains = scatter["context_gain"].astype(float)
        axes[0].hist(gains, bins=60, color=PALETTE["context"], alpha=0.8)
        axes[0].axvline(0.2, color=PALETTE["risk"], ls="--")
        axes[0].set_xlabel("context gain")
        axes[0].set_ylabel("assignments")
        probs = scatter["top_probability_calibrated"].astype(float)
        axes[1].hist(probs, bins=60, color=PALETTE["neutral"], alpha=0.8)
        axes[1].set_xlabel("calibrated probability")
    panel(axes[0], "S13A. Context gain distribution")
    panel(axes[1], "S13B. Candidate probability distribution")
    savefig(fig, fig_dir, "S13_candidate_score_distributions")


def figure_s14(qc_dir: Path, fig_dir: Path) -> None:
    null = dataframe(qc_dir / "qc8_module_cluster_assignment_null_iterations.tsv")
    obs = dataframe(qc_dir / "qc8_module_discovery_null_control.tsv")
    metrics = [
        "weighted_neighborhood_consistency",
        "weighted_structural_membrane_vote_fraction",
        "weighted_context_sensitive_label_fraction",
        "mean_family_recurrence",
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes = axes.ravel()
    for ax, metric in zip(axes, metrics):
        if not null.empty and metric in null:
            ax.hist(null[metric].astype(float), bins=30, color=PALETTE["neutral"], alpha=0.75)
            row = obs[obs["metric"] == metric]
            if not row.empty:
                ax.axvline(float(row.iloc[0]["observed"]), color=PALETTE["context"], lw=2, label="observed")
                ax.legend(frameon=False, fontsize=8)
        panel(ax, metric.replace("_", " "))
    savefig(fig, fig_dir, "S14_module_null_distributions")


def figure_s15(core_dir: Path, fig_dir: Path) -> None:
    casebooks = sorted((core_dir / "module_discovery" / "casebooks").glob("*.casebook.md"))
    fig, axes = plt.subplots(5, 2, figsize=(11, 14))
    axes = axes.ravel()
    for idx, ax in enumerate(axes):
        ax.axis("off")
        if idx >= len(casebooks):
            continue
        path = casebooks[idx]
        text = path.read_text(encoding="utf-8", errors="replace")
        lines = [line.strip("# ").strip() for line in text.splitlines() if line.strip()]
        body = "\n".join(textwrap.wrap(" | ".join(lines[:8]), width=62))
        ax.text(0.02, 0.96, body, va="top", ha="left", fontsize=7.5, transform=ax.transAxes)
        panel(ax, f"S15.{idx + 1}. {path.stem}")
    savefig(fig, fig_dir, "S15_casebook_thumbnails")


def write_gap_report(out_dir: Path) -> None:
    rows = [
        {
            "item": "S6 all metrics and all seeds",
            "status": "completed when review-completion multi-seed outputs are provided",
            "script": "run_v2_review_completion.py multiseed; see S8b/S8c and Supplementary Fig S6",
        },
        {
            "item": "S9 PR curves for several non-nucleocapsid labels",
            "status": "generated from available per-label PR sources when present; otherwise represented by returned atlas tables",
            "script": "run_v2_qc_suite.py --force-predict or build_context_dependence_atlas_v2.py",
        },
        {
            "item": "classical homology baseline",
            "status": "completed for MMseqs2 top-hit label transfer when returned homology outputs are provided",
            "script": "run_homology_label_transfer.py; see S21 tables",
        },
        {
            "item": "candidate external validation evidence",
            "status": "completed for returned metadata and neighborhoods; optional domain/structure files can further enrich case evidence",
            "script": "build_candidate_case_evidence.py",
        },
    ]
    write_tsv(out_dir / "supplement_completion_gap_report.tsv", rows)


def write_figure_package(args: argparse.Namespace, fig_dir: Path) -> None:
    figure_s1(args.core_dir, args.assets_dir, fig_dir)
    figure_s2(args.core_dir, fig_dir)
    figure_s3(args.core_dir, fig_dir)
    figure_s4(args.core_dir, fig_dir)
    figure_s5(args.assets_dir, fig_dir)
    figure_s7(args.assets_dir, fig_dir)
    figure_s8(args.assets_dir, fig_dir)
    figure_s9(args.qc_dir, fig_dir)
    figure_s10(args.qc_dir, fig_dir)
    figure_s11(args.core_dir, fig_dir)
    figure_s12(args.core_dir, fig_dir)
    figure_s13(args.assets_dir, fig_dir)
    figure_s14(args.qc_dir, fig_dir)
    figure_s15(args.core_dir, fig_dir)


def zip_dir(src_dir: Path, zip_path: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(src_dir.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(src_dir))


def main() -> None:
    args = parse_args()
    out_dir = args.output_dir.resolve()
    tables_dir = out_dir / "supplementary_tables"
    fig_dir = out_dir / "supplementary_figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_table_package(args, tables_dir)
    write_figure_package(args, fig_dir)
    write_gap_report(out_dir)
    if args.make_zip:
        zip_dir(tables_dir, out_dir / "supplementary_tables.zip")
        zip_dir(fig_dir, out_dir / "supplementary_figures.zip")
    summary = {
        "output_dir": str(out_dir),
        "supplementary_table_count": len(list(tables_dir.glob("*.tsv"))) + len(list(tables_dir.glob("*.json"))),
        "supplementary_figure_count": len(list(fig_dir.glob("*.pdf"))),
        "tables_zip": str(out_dir / "supplementary_tables.zip") if (out_dir / "supplementary_tables.zip").exists() else "",
        "figures_zip": str(out_dir / "supplementary_figures.zip") if (out_dir / "supplementary_figures.zip").exists() else "",
    }
    (out_dir / "supplementary_package_manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
