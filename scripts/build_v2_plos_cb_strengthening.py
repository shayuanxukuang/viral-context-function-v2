#!/usr/bin/env python3
"""Create PLOS CB strengthening drafts and figure sketches from V2 manuscript assets."""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PALETTE = {
    "protein": "#4C78A8",
    "context": "#F58518",
    "support": "#54A24B",
    "risk": "#E45756",
    "neutral": "#8A8F98",
    "accent": "#B279A2",
    "light": "#F4F6F8",
}


def read_tsv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, sep="\t")


def wrap(text: str, width: int = 54) -> str:
    return "\n".join(textwrap.wrap(str(text), width=width, break_long_words=False))


def metric_row(df: pd.DataFrame, comparison: str, metric: str) -> pd.Series:
    row = df[(df["comparison"] == comparison) & (df["metric"] == metric)]
    if row.empty:
        return pd.Series(dtype=object)
    return row.iloc[0]


def save_figure(fig: plt.Figure, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=1.1)
    fig.savefig(out_dir / f"{name}.png", dpi=220, bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def panel_label(ax: plt.Axes, label: str, title: str) -> None:
    ax.set_title(f"{label}. {title}", loc="left", fontweight="bold", fontsize=10)


def text_panel(ax: plt.Axes, label: str, title: str, body: str, fontsize: int = 9) -> None:
    ax.axis("off")
    panel_label(ax, label, title)
    wrapped_body = "\n".join(wrap(line, width=48) if line.strip() else "" for line in str(body).splitlines())
    ax.text(
        0.02,
        0.92,
        wrapped_body,
        va="top",
        ha="left",
        fontsize=fontsize,
        linespacing=1.25,
        transform=ax.transAxes,
        bbox=dict(boxstyle="round,pad=0.35", facecolor=PALETTE["light"], edgecolor="#D0D5DD"),
    )


def build_figure1(assets: Path, out_dir: Path) -> None:
    leakage = read_tsv(assets / "figure1_leakage_summary.tsv")
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7))
    text_panel(
        axes[0, 0],
        "A",
        "Task definitions",
        "protein-only de novo\n"
        "  target sequence -> pLM -> labels\n\n"
        "genome-aware de novo\n"
        "  target + neighbor pLM + non-text topology\n\n"
        "annotation refinement\n"
        "  separate task; annotation priors allowed only here",
        fontsize=8.6,
    )
    text_panel(
        axes[0, 1],
        "B",
        "Allowed and forbidden features",
        "Allowed in de novo:\n"
        "- target and neighbor sequence embeddings\n"
        "- gene order, coordinates, strand, gaps\n"
        "- segment and genome organization\n\n"
        "Forbidden in de novo:\n"
        "- product text, hypothetical flag\n"
        "- database hit counts\n"
        "- neighbor labels or label counts",
        fontsize=8.3,
    )
    exact = leakage[(leakage["panel"] == "C") & (leakage["scheme"].isin(["default", "family_holdout"]))]
    axes[0, 2].bar(["default", "family-heldout"], exact["value"].astype(float) * 100, color=[PALETTE["risk"], PALETTE["support"]])
    panel_label(axes[0, 2], "C", "Exact sequence transfer")
    axes[0, 2].set_ylabel("test proteins with exact train match (%)")
    axes[0, 2].set_ylim(0, max(30, exact["value"].astype(float).max() * 115))
    for idx, val in enumerate(exact["value"].astype(float) * 100):
        axes[0, 2].text(idx, val + 0.8, f"{val:.1f}%", ha="center", fontsize=9)

    nn = leakage[(leakage["panel"] == "D") & (leakage["scheme"].isin(["default", "family_holdout"]))]
    axes[1, 0].bar(["default", "family-heldout"], nn["value"].astype(float), color=[PALETTE["risk"], PALETTE["support"]])
    panel_label(axes[1, 0], "D", "Nearest-neighbor label transfer")
    axes[1, 0].set_ylabel("macro AP")
    axes[1, 0].set_ylim(0, 0.42)
    for idx, val in enumerate(nn["value"].astype(float)):
        axes[1, 0].text(idx, val + 0.012, f"{val:.3f}", ha="center", fontsize=9)

    strict = leakage[(leakage["panel"] == "E") & (leakage["metric"].isin(["delta_macro_ap", "delta_macro_f1"]))]
    labels = ["macro AP", "macro F1"]
    vals = [float(strict[strict["metric"] == "delta_macro_ap"]["value"].iloc[0]), float(strict[strict["metric"] == "delta_macro_f1"]["value"].iloc[0])]
    axes[1, 1].bar(labels, vals, color=PALETTE["context"])
    axes[1, 1].axhline(0, color="#444444", linewidth=0.8)
    panel_label(axes[1, 1], "E", "Strict-zero exact-transfer sensitivity")
    axes[1, 1].set_ylabel("context - protein-only")
    axes[1, 1].set_ylim(0, max(vals) * 1.45)
    for idx, val in enumerate(vals):
        axes[1, 1].text(idx, val + 0.001, f"+{val:.4f}", ha="center", fontsize=9)
    removed = leakage[leakage["metric"] == "removed_exact_transfer_proteins"]["display_value"].iloc[0]
    axes[1, 1].text(0.5, 0.92, f"removed {removed} exact-transfer proteins", transform=axes[1, 1].transAxes, ha="center", fontsize=8)

    residual = leakage[(leakage["panel"] == "F") & (leakage["scheme"] == "family_holdout_exact_transfer_audit")]
    if not residual.empty:
        residual = residual.copy()
        residual["short"] = residual["note"].astype(str).str.replace("identical proteins assigned to different families", "cross-family identical", regex=False)
        axes[1, 2].barh(residual["short"], residual["value"].astype(float), color=PALETTE["neutral"])
        axes[1, 2].invert_yaxis()
        axes[1, 2].set_xlabel("proteins")
        panel_label(axes[1, 2], "F", "Residual exact-transfer audit")
    else:
        axes[1, 2].axis("off")
    save_figure(fig, out_dir, "figure1_leakage_aware_benchmark_design")


def build_figure2(assets: Path, out_dir: Path) -> None:
    ci = read_tsv(assets / "figure2_delta_ci.tsv")
    strict = read_tsv(assets / "figure1_leakage_summary.tsv")
    labels = read_tsv(assets / "figure4_label_delta_scatter.tsv")
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.2))

    fam_ap = metric_row(ci, "family_holdout", "macro AP")
    fam_f1 = metric_row(ci, "family_holdout", "macro F1")
    raw_labels = ["macro AP", "macro F1"]
    x = np.arange(len(raw_labels))
    width = 0.36
    axes[0, 0].bar(x - width / 2, [fam_ap["protein_value"], fam_f1["protein_value"]], width, label="protein-only pLM", color=PALETTE["protein"])
    axes[0, 0].bar(x + width / 2, [fam_ap["context_value"], fam_f1["context_value"]], width, label="genome-aware pLM", color=PALETTE["context"])
    axes[0, 0].set_xticks(x, raw_labels)
    axes[0, 0].set_ylim(0.58, 0.75)
    axes[0, 0].set_ylabel("score")
    axes[0, 0].legend(frameon=False, fontsize=8)
    panel_label(axes[0, 0], "A", "Family-heldout pLM performance")

    for ax, comparison, panel, title in [
        (axes[0, 1], "family_holdout", "B", "Family-block delta CI"),
        (axes[0, 2], "host_holdout", "C", "Host-heldout supportive CI"),
    ]:
        sub = ci[(ci["comparison"] == comparison) & (ci["metric"].isin(["macro AP", "macro F1", "micro AP", "micro F1"]))]
        y = np.arange(len(sub))
        deltas = sub["delta"].astype(float).to_numpy()
        lows = sub["ci_low"].astype(float).to_numpy()
        highs = sub["ci_high"].astype(float).to_numpy()
        ax.errorbar(deltas, y, xerr=[deltas - lows, highs - deltas], fmt="o", color=PALETTE["context"], ecolor="#333333", capsize=3)
        ax.axvline(0, color="#555555", linestyle="--", linewidth=0.9)
        ax.set_yticks(y, sub["metric"].tolist())
        ax.invert_yaxis()
        ax.set_xlabel("context - protein-only")
        panel_label(ax, panel, title)

    strict_rows = strict[strict["metric"].isin(["delta_macro_ap", "delta_macro_f1"])]
    axes[1, 0].bar(["macro AP", "macro F1"], strict_rows["value"].astype(float), color=PALETTE["support"])
    axes[1, 0].axhline(0, color="#555555", linewidth=0.8)
    panel_label(axes[1, 0], "D", "Strict-zero-transfer subset")
    axes[1, 0].set_ylabel("context - protein-only")

    water = labels.sort_values("delta_average_precision", ascending=False)
    top = pd.concat([water.head(6), water.tail(4)])
    colors = [PALETTE["support"] if v > 0 else PALETTE["risk"] for v in top["delta_average_precision"]]
    axes[1, 1].barh(top["label"], top["delta_average_precision"], color=colors)
    axes[1, 1].axvline(0, color="#555555", linewidth=0.8)
    axes[1, 1].invert_yaxis()
    axes[1, 1].set_xlabel("delta AP")
    panel_label(axes[1, 1], "E", "Per-label delta preview")

    summary = ci[ci["metric"].isin(["macro AP", "macro F1", "micro AP", "micro F1"])].copy()
    summary["direction"] = np.where(summary["delta"].astype(float) > 0, "positive", "negative")
    piv = summary.groupby(["comparison", "direction"]).size().unstack(fill_value=0)
    for direction in ["positive", "negative"]:
        if direction not in piv.columns:
            piv[direction] = 0
    y = np.arange(len(piv))
    axes[1, 2].barh(y, piv["positive"], color=PALETTE["support"], label="positive")
    axes[1, 2].barh(y, -piv["negative"], color=PALETTE["risk"], label="negative")
    axes[1, 2].set_yticks(y, [str(x).replace("_", "-") for x in piv.index])
    axes[1, 2].axvline(0, color="#555555", linewidth=0.8)
    axes[1, 2].set_xlabel("number of metric deltas")
    axes[1, 2].legend(frameon=False, fontsize=7)
    panel_label(axes[1, 2], "F", "Metric-direction summary")
    save_figure(fig, out_dir, "figure2_main_ood_performance_uncertainty")


def build_figure3(assets: Path, qc_dir: Path, out_dir: Path) -> None:
    source = read_tsv(assets / "figure3_source_controls.tsv")
    source_ci = read_tsv(qc_dir / "qc_source_addback_block_bootstrap_ci.tsv")
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.2))
    add = source[source["row_type"] == "source_addback"].copy()
    add_order = ["protein_only", "local_only", "genome_only", "host_only", "local_genome", "all_clean_context"]
    add["model_label"] = pd.Categorical(add["model_label"], categories=add_order, ordered=True)
    add = add.sort_values("model_label")
    axes[0, 0].barh(add["model_label"].astype(str), add["macro_ap"].astype(float), color=PALETTE["context"])
    axes[0, 0].set_xlabel("macro AP")
    axes[0, 0].set_xlim(0.64, 0.695)
    panel_label(axes[0, 0], "A", "Add-back source decomposition")

    key = add[add["model_label"].astype(str).isin(["local_genome", "host_only"])].copy()
    key["source_ci_name"] = key["model_label"].astype(str).replace({"local_genome": "local_plus_genome"})
    vals = key["delta_macro_ap_vs_protein_only"].astype(float).to_numpy()
    yerr = None
    if not source_ci.empty:
        ci_lookup = source_ci[source_ci["split"].astype(str) == "family_holdout"].set_index("source")
        lows = []
        highs = []
        for _, row in key.iterrows():
            ci = ci_lookup.loc[row["source_ci_name"]] if row["source_ci_name"] in ci_lookup.index else pd.Series(dtype=object)
            low = float(ci.get("delta_macro_ap_ci_low", row["delta_macro_ap_vs_protein_only"]))
            high = float(ci.get("delta_macro_ap_ci_high", row["delta_macro_ap_vs_protein_only"]))
            value = float(row["delta_macro_ap_vs_protein_only"])
            lows.append(max(0.0, value - low))
            highs.append(max(0.0, high - value))
        yerr = [lows, highs]
    axes[0, 1].bar(key["model_label"].astype(str), vals, yerr=yerr, capsize=3, color=[PALETTE["support"], PALETTE["neutral"]])
    axes[0, 1].axhline(0, color="#555555", linewidth=0.8)
    panel_label(axes[0, 1], "B", "Matched estimates with block CI")
    axes[0, 1].set_ylabel("delta macro AP")
    for idx, val in enumerate(vals):
        axes[0, 1].text(idx, val + 0.0007, f"+{val:.4f}", ha="center", fontsize=8)

    curve = source[(source["row_type"] == "host_corruption_or_shuffle") & (source["model_label"] == "host_corruption")].copy()
    curve = curve[curve["source"].astype(str).str.match(r"^[0-9.]+$")]
    for split, sub in curve.groupby("split"):
        sub = sub.assign(frac=sub["source"].astype(float)).sort_values("frac")
        axes[0, 2].plot(sub["frac"], sub["macro_ap"].astype(float), marker="o", label=split)
    axes[0, 2].set_xlabel("host corruption fraction")
    axes[0, 2].set_ylabel("macro AP")
    axes[0, 2].legend(frameon=False, fontsize=8)
    panel_label(axes[0, 2], "C", "Host corruption curve")

    loso = source[source["row_type"] == "leave_one_source_out"].copy()
    if not loso.empty:
        loso["display"] = loso["model_label"].astype(str).str.replace("minus_", "-", regex=False)
        vals = loso["delta_macro_ap_vs_protein_only"].astype(float)
        axes[1, 0].barh(loso["display"], vals, color=[PALETTE["risk"] if v < 0 else PALETTE["neutral"] for v in vals])
        axes[1, 0].axvline(0, color="#555555", linewidth=0.8)
        axes[1, 0].set_xlabel("minus-source delta macro AP")
    else:
        axes[1, 0].axis("off")
        axes[1, 0].text(0.02, 0.8, "leave-one-source-out rows not available", transform=axes[1, 0].transAxes)
    panel_label(axes[1, 0], "D", "Leave-one-source-out")

    forbidden = source[source["row_type"] == "forbidden_feature_check"]
    status_counts = forbidden["source"].value_counts()
    axes[1, 1].bar(status_counts.index, status_counts.values, color=PALETTE["support"])
    axes[1, 1].set_ylabel("feature families")
    axes[1, 1].set_ylim(0, max(9, status_counts.max() + 1))
    panel_label(axes[1, 1], "E", "Forbidden feature check")
    axes[1, 1].text(0, status_counts.iloc[0] + 0.25, f"{int(status_counts.iloc[0])}/8 PASS", ha="center", fontsize=9)

    summary_rows = []
    for split, sub in curve.groupby("split"):
        sub = sub.assign(frac=sub["source"].astype(float)).sort_values("frac")
        if sub.empty:
            continue
        base = float(sub.iloc[0]["macro_ap"])
        full = float(sub.iloc[-1]["macro_ap"])
        max_drop = base - float(sub["macro_ap"].astype(float).min())
        summary_rows.append((str(split), full - base, max_drop))
    if summary_rows:
        labels = [row[0].replace("_", "-") for row in summary_rows]
        full_delta = [row[1] for row in summary_rows]
        max_drop = [row[2] for row in summary_rows]
        yy = np.arange(len(labels))
        axes[1, 2].barh(yy - 0.18, full_delta, height=0.32, color=PALETTE["neutral"], label="delta at 100%")
        axes[1, 2].barh(yy + 0.18, [-v for v in max_drop], height=0.32, color=PALETTE["risk"], label="max drop")
        axes[1, 2].axvline(0, color="#555555", linewidth=0.8)
        axes[1, 2].set_yticks(yy, labels)
        axes[1, 2].set_xlabel("macro AP change")
        axes[1, 2].legend(frameon=False, fontsize=7)
    else:
        axes[1, 2].axis("off")
    panel_label(axes[1, 2], "F", "Host-corruption magnitude")
    save_figure(fig, out_dir, "figure3_source_decomposition_leakage_controls")


def build_figure4(assets: Path, qc_dir: Path, out_dir: Path) -> None:
    labels = read_tsv(assets / "figure4_label_delta_scatter.tsv")
    group = read_tsv(assets / "figure4_functional_group_context_summary.tsv")
    additional = read_tsv(assets / "figure4_additional_label_examples.tsv")
    nuc = read_tsv(assets / "figure4_nucleocapsid_audit_summary.tsv")
    fp = read_tsv(assets / "nucleocapsid_fp_audit_summary.tsv")
    pr = read_tsv(qc_dir / "qc6_nucleocapsid_pr_curve.tsv")
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.4))

    axes[0, 0].scatter(labels["protein_average_precision"], labels["delta_average_precision"], c=[PALETTE["support"] if v > 0 else PALETTE["risk"] for v in labels["delta_average_precision"]], s=48, alpha=0.85)
    for _, row in labels.sort_values("delta_average_precision", ascending=False).head(3).iterrows():
        axes[0, 0].annotate(row["label"], (row["protein_average_precision"], row["delta_average_precision"]), fontsize=7, xytext=(5, 5), textcoords="offset points")
    axes[0, 0].axhline(0, color="#555555", linewidth=0.8)
    axes[0, 0].set_xlabel("protein-only AP")
    axes[0, 0].set_ylabel("context delta AP")
    panel_label(axes[0, 0], "A", "Label-level context dependence")

    if not group.empty:
        group = group.sort_values("median_delta_ap")
        vals = group["median_delta_ap"].astype(float)
        lows = group["median_delta_ap_ci_low"].astype(float)
        highs = group["median_delta_ap_ci_high"].astype(float)
        y = np.arange(len(group))
        colors = [PALETTE["support"] if v > 0 else PALETTE["risk"] for v in vals]
        axes[0, 1].barh(group["label_group"], vals, color=colors, alpha=0.85)
        axes[0, 1].errorbar(vals, y, xerr=[vals - lows, highs - vals], fmt="none", ecolor="#333333", capsize=2)
    else:
        label_groups = labels.groupby("label_group")["delta_average_precision"].median().sort_values()
        colors = [PALETTE["support"] if v > 0 else PALETTE["risk"] for v in label_groups.values]
        axes[0, 1].barh(label_groups.index, label_groups.values, color=colors)
    axes[0, 1].axvline(0, color="#555555", linewidth=0.8)
    axes[0, 1].set_xlabel("median delta AP")
    panel_label(axes[0, 1], "B", "Functional group atlas")

    top = labels.sort_values("delta_average_precision", ascending=False).head(8)
    axes[0, 2].barh(top["label"], top["delta_average_precision"], color=PALETTE["support"])
    axes[0, 2].invert_yaxis()
    axes[0, 2].set_xlabel("delta AP")
    panel_label(axes[0, 2], "C", "Top context-sensitive labels")

    axes[1, 0].plot(pr["recall"], pr["precision"], color=PALETTE["context"], linewidth=2)
    axes[1, 0].set_xlim(0, 1)
    axes[1, 0].set_ylim(0, 1.03)
    axes[1, 0].set_xlabel("recall")
    axes[1, 0].set_ylabel("precision")
    row = nuc.iloc[0]
    axes[1, 0].text(0.04, 0.13, f"AP 0.240 -> 0.571\nDelta +0.331\npositives {row['test_positives']}; families {row['test_families']}; hosts {row['test_host_groups']}", fontsize=8, transform=axes[1, 0].transAxes)
    panel_label(axes[1, 0], "D", "Nucleocapsid PR curve")

    axes[1, 1].barh(fp["audit_category"], fp["count"].astype(int), color=[PALETTE["support"], PALETTE["neutral"], PALETTE["risk"]][: len(fp)])
    axes[1, 1].set_xlabel("top false positives")
    panel_label(axes[1, 1], "E", "Nucleocapsid FP synonym audit")

    if not additional.empty:
        additional = additional.sort_values("delta_average_precision", ascending=True)
        axes[1, 2].barh(additional["label"], additional["delta_average_precision"].astype(float), color=PALETTE["support"])
        axes[1, 2].axvline(0, color="#555555", linewidth=0.8)
        axes[1, 2].set_xlabel("delta AP")
        panel_label(axes[1, 2], "F", "Additional positive labels")
    else:
        axes[1, 2].axis("off")
    save_figure(fig, out_dir, "figure4_context_dependence_atlas")


def normalize_candidate_category(value: str) -> str:
    category = str(value)
    lower_category = category.lower()
    if lower_category.startswith("fdr") and "protein-label" in lower_category:
        return "validation-targeted protein-label assignments"
    if lower_category.startswith("fdr") and "proteins" in lower_category:
        return "validation-targeted proteins"
    return category


def build_figure5(assets: Path, out_dir: Path) -> None:
    breakdown = read_tsv(assets / "figure5_candidate_breakdown.tsv")
    if not breakdown.empty:
        breakdown["category"] = breakdown["category"].map(normalize_candidate_category)
    scatter = read_tsv(assets / "figure5_confidence_context_gain_scatter.tsv")
    top = read_tsv(assets / "figure5_high_context_gain_candidates.tsv").head(5)
    precision = read_tsv(assets / "figure5_fdr_gate_precision_summary.tsv")
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.4))

    text_panel(
        axes[0, 0],
        "A",
        "Calibration and validation-targeted gate",
        "validation: model selection + post hoc calibration/threshold\nheld-out test: final evaluation\n\nOutput unit: protein-label assignment.\nInterpretation: prioritized candidates, not confirmed discoveries.",
        fontsize=8.8,
    )
    display = breakdown[breakdown["category"].isin([
        "validation-targeted proteins",
        "validation-targeted protein-label assignments",
        "hypothetical/uncharacterized/unknown assignments",
        "hypothetical/uncharacterized/unknown unique proteins",
        "high context-gain assignments delta_p>=0.2",
    ])].copy()
    display["short"] = ["val-targeted\nproteins", "assignments", "poor\nannotation", "poor\nproteins", "high\ngain"]
    axes[0, 1].bar(display["short"], display["count"].astype(int), color=[PALETTE["neutral"], PALETTE["neutral"], PALETTE["support"], PALETTE["support"], PALETTE["context"]])
    axes[0, 1].set_yscale("log")
    axes[0, 1].tick_params(axis="x", rotation=20)
    axes[0, 1].set_ylabel("count (log scale)")
    panel_label(axes[0, 1], "B", "Candidate breakdown")

    if len(scatter) > 2500:
        scatter_plot = scatter.sample(2500, random_state=7)
    else:
        scatter_plot = scatter
    axes[0, 2].scatter(scatter_plot["top_probability_calibrated"], scatter_plot["context_gain"], s=8, alpha=0.25, color=PALETTE["context"])
    axes[0, 2].axhline(0.2, color=PALETTE["risk"], linestyle="--", linewidth=1)
    high = scatter[scatter["context_gain"].astype(float) >= 0.2]
    if not high.empty:
        axes[0, 2].scatter(high["top_probability_calibrated"], high["context_gain"], s=18, alpha=0.85, color=PALETTE["risk"], label="high gain")
        axes[0, 2].legend(frameon=False, fontsize=7, loc="upper left")
    axes[0, 2].set_xlabel("calibrated probability")
    axes[0, 2].set_ylabel("context gain")
    axes[0, 2].set_ylim(-0.02, 1.02)
    axes[0, 2].set_xlim(0.985, 1.001)
    panel_label(axes[0, 2], "C", "Confidence vs context gain")

    axes[1, 0].axis("off")
    panel_label(axes[1, 0], "D", "Top high-context-gain candidates")
    lines = []
    for _, row in top.iterrows():
        lines.append(f"{row['protein_accession']} | {row['candidate_label']} | p={float(row['top_probability_calibrated']):.3f} | gain={float(row['context_gain']):.3f}")
    axes[1, 0].text(0.02, 0.92, "\n".join(lines[:5]), va="top", ha="left", fontsize=8.2, family="monospace", transform=axes[1, 0].transAxes)

    funnel_labels = ["validation-targeted\nassignments", "poor\nannotation", "high context\ngain"]
    counts = [
        int(breakdown[breakdown["category"] == "validation-targeted protein-label assignments"]["count"].iloc[0]),
        int(breakdown[breakdown["category"] == "hypothetical/uncharacterized/unknown assignments"]["count"].iloc[0]),
        int(breakdown[breakdown["category"] == "high context-gain assignments delta_p>=0.2"]["count"].iloc[0]),
    ]
    axes[1, 1].bar(funnel_labels, counts, color=[PALETTE["neutral"], PALETTE["support"], PALETTE["context"]])
    axes[1, 1].set_yscale("log")
    axes[1, 1].tick_params(axis="x", rotation=20)
    panel_label(axes[1, 1], "E", "Case selection funnel")

    if not precision.empty:
        row = precision.iloc[0]
        vals = [
            float(row["empirical_validation_top1_precision"]),
            float(row["labeled_test_top1_precision"]),
        ]
        axes[1, 2].bar(["calibration", "labeled test"], vals, color=[PALETTE["support"], PALETTE["neutral"]])
        axes[1, 2].axhline(float(row["calibration_target_precision"]), color=PALETTE["risk"], linestyle="--", linewidth=1)
        axes[1, 2].set_ylim(0, 1.0)
        axes[1, 2].set_ylabel("top-label precision")
        for idx, val in enumerate(vals):
            axes[1, 2].text(idx, val + 0.025, f"{val:.3f}", ha="center", fontsize=8)
        axes[1, 2].text(0.5, 0.08, f"test FDP {float(row['labeled_test_top1_fdp']):.3f}", ha="center", transform=axes[1, 2].transAxes, fontsize=8)
        panel_label(axes[1, 2], "F", "Calibration target vs OOD test")
    else:
        axes[1, 2].axis("off")
    save_figure(fig, out_dir, "figure5_calibrated_candidate_prioritization")


def build_figure6(assets: Path, out_dir: Path) -> None:
    null = read_tsv(assets / "figure6_module_null_summary.tsv")
    cases = read_tsv(assets / "selected_casebook_panels.tsv").head(3)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.4))
    text_panel(
        axes[0, 0],
        "A",
        "Module-support workflow",
        "high-confidence candidates\n+ genome neighborhoods\n+ cluster by module signatures\n+ compare observed metrics to shuffled nulls\n+ inspect casebooks",
        fontsize=8.8,
    )
    text_panel(
        axes[0, 1],
        "B",
        "Cluster summary",
        "339 module clusters\n10 casebooks prepared\n3 main-panel examples selected\n\nFamily recurrence evaluated but not used as evidence because it did not exceed null.",
        fontsize=9.0,
    )
    y = np.arange(len(null))
    axes[0, 2].errorbar(null["observed"], y - 0.08, fmt="o", label="observed", color=PALETTE["context"])
    axes[0, 2].errorbar(null["null_mean"], y + 0.08, xerr=[null["null_mean"] - null["null_ci_low"], null["null_ci_high"] - null["null_mean"]], fmt="o", label="null", color=PALETTE["neutral"], capsize=3)
    axes[0, 2].set_yticks(y, null["display_metric"])
    axes[0, 2].invert_yaxis()
    axes[0, 2].set_xlim(0, 1.0)
    for yi, (_, row) in enumerate(null.iterrows()):
        effect = float(row["observed"]) - float(row["null_mean"])
        axes[0, 2].text(float(row["observed"]) + 0.01, yi - 0.08, f"+{effect:.3f}", va="center", fontsize=7)
    axes[0, 2].legend(frameon=False, fontsize=8)
    panel_label(axes[0, 2], "C", "Small enrichments vs null")

    for idx, ax in enumerate(axes[1, :]):
        if idx < len(cases):
            row = cases.iloc[idx]
            text_panel(
                ax,
                chr(ord("D") + idx),
                f"Example {idx + 1}: {row['candidate_id']}",
                f"Label: {row['predicted_label']}\n"
                f"p_context={float(row['p_genome_aware']):.3f}; delta={float(row['delta_p']):.3f}\n"
                f"Module cluster: {row['module_cluster_id']}\n"
                f"Annotation: {wrap(row['posthoc_external_evidence'], 36)}\n\n"
                f"Module-supported computational example; not experimental validation.",
                fontsize=8.2,
            )
        else:
            ax.axis("off")
    save_figure(fig, out_dir, "figure6_module_discovery_casebooks")


def build_text_outputs(assets: Path, out_dir: Path) -> None:
    claim = read_tsv(assets / "claim_evidence_matrix.tsv")
    ci = read_tsv(assets / "figure2_delta_ci.tsv")
    leakage = read_tsv(assets / "figure1_leakage_summary.tsv")
    candidate = read_tsv(assets / "figure5_candidate_breakdown.tsv")
    null = read_tsv(assets / "figure6_module_null_summary.tsv")
    nuc = read_tsv(assets / "figure4_nucleocapsid_audit_summary.tsv")
    text_dir = out_dir / "manuscript_text"
    text_dir.mkdir(parents=True, exist_ok=True)

    def claim_wording(name: str) -> str:
        row = claim[claim["claim"] == name]
        return "" if row.empty else str(row.iloc[0]["manuscript_wording"])

    results = f"""# Strengthened Results Draft

## 1. Leakage-aware benchmark design

Default split evaluation was optimistic. The default split had a 27.4% exact sequence transfer rate and a nearest-neighbor label-transfer macro AP of 0.369, whereas the family-heldout split reduced exact transfer to 1.37% and nearest-neighbor macro AP to 0.048. Removing the 916 residual exact-transfer proteins from the family-heldout test set retained the genome-context gain (macro AP +0.0133; macro F1 +0.0223). Together, these analyses support using family-heldout evaluation as the primary out-of-distribution benchmark.

Suggested main sentence: {claim_wording('default split optimistic')}

## 2. Main OOD performance with transparent uncertainty

Genome-aware pLM models showed positive family-heldout macro-level point estimates, but the family-block bootstrap interval overlapped zero and micro AP was negative. The paired family-heldout delta was macro AP +0.0136 (95% CI -0.0114 to +0.0276) and macro F1 +0.0225 (95% CI -0.0056 to +0.0542). The host-heldout split provided a complementary OOD setting with a more stable positive gain: macro AP +0.0216 (95% CI +0.0107 to +0.0310) and macro F1 +0.0179 (95% CI +0.0034 to +0.0306).

Suggested main sentence: {claim_wording('context global family trend')}

## 3. Source decomposition and leakage controls

The clean genome-aware de novo model used target and neighbor sequence embeddings plus non-text genome organization features. It did not use product text, hypothetical flags, database hit counts, neighbor true labels, genome/local label counts, or other annotation-derived priors. All 8 forbidden feature families passed the manifest check, and all 38 source-decomposition comparisons were matched for the core experimental factors. Local+genome add-back produced a larger family-heldout macro AP gain (+0.0170) than host-only add-back (+0.0047), and host shuffle/corruption did not show a large monotonic performance collapse.

Suggested main sentence: {claim_wording('not host prior')}

## 4. Context dependence atlas and nucleocapsid audit

Context dependence was not uniform across labels. Ten of 17 labels had positive AP deltas, with the strongest example in structural/assembly-associated labels. Nucleocapsid was the clearest representative example (AP 0.240 -> 0.571, delta +0.331; 133 test positives across 7 held-out families and 91 host groups). Several non-nucleocapsid labels had smaller positive AP deltas, including ligase, protease, transcription regulator, polyprotein, membrane/matrix, lysis, tail assembly, helicase, and nuclease. A post hoc audit of high-scoring nucleocapsid false positives found many N/nucleoprotein/nucleocapsid product names, consistent with incomplete benchmark labels. This audit should be presented as post hoc evidence only and should not be used to revise the primary metric.

Safe wording: Nucleocapsid provides a representative example of strong context dependence, although the number of held-out families is limited.

## 5. Calibrated candidate prioritization

The calibrated candidate analysis should be presented in three layers. First, 12,144 protein-label assignments across 11,839 proteins passed the validation-targeted precision gate. Second, 1,378 assignments involving 1,375 unique proteins corresponded to hypothetical, uncharacterized, or unknown proteins. Third, 27 candidates had context gain >= 0.2, representing cases where genome context materially changed the prediction.

Suggested main sentence: {claim_wording('candidates')}

## 6. Module discovery as auxiliary biological support

Module discovery yielded 339 clusters and 10 casebooks. Compared with shuffled nulls, observed modules showed higher neighborhood consistency, structural/membrane enrichment, and context-sensitive label enrichment (empirical p approximately 0.002 for all three metrics). Family recurrence did not exceed the shuffled null and should not be used as evidence for broad cross-family conservation.

Suggested main sentence: {claim_wording('modules')}
"""
    (text_dir / "results_strengthened_claims.md").write_text(results, encoding="utf-8")

    methods = """# Strengthened Methods Draft

## Task definitions

We defined three prediction settings. In the protein-only de novo setting, the model used only the target protein sequence representation. In the genome-aware de novo setting, the model used the target sequence representation, neighbor sequence embeddings, and non-text genomic organization features such as relative gene order, coordinate-derived topology, strand, gaps, overlaps, segment identifiers, and genome-scale organization. The genome-aware de novo setting explicitly excluded product text, hypothetical-protein flags, database hit counts, Pfam/InterPro/CDD hits, neighbor true labels, genome/local label counts, protein feature type annotations, and annotation text embeddings. Annotation refinement was treated as a separate task in which annotation-derived priors could be used, and it was not used to support de novo discovery claims.

## Split construction

Family-heldout evaluation was the primary OOD benchmark. Test families were absent from training, validation families were disjoint from both training and test families, and proteins from the same genome did not cross split boundaries. Host-heldout evaluation was used as a secondary OOD benchmark, while the default split was used only to quantify optimistic bias. Exact sequence transfer and nearest-neighbor label transfer audits were performed for each split. A strict-zero exact-transfer sensitivity analysis removed the residual exact-transfer proteins from the family-heldout test set and recomputed metrics without retraining.

## Statistical evaluation

The primary metrics were macro average precision, validation-selected macro F1, micro average precision, validation-selected micro F1, and per-label AP. Because the family-heldout test set contains correlated proteins within viral families, family-block bootstrap was used as the primary uncertainty estimator for family-heldout paired deltas. Host-heldout uncertainty was estimated using host-group/block bootstrap. Bootstrap intervals were computed for paired model differences, with the primary comparison defined as genome-aware de novo pLM minus protein-only pLM.

## Source decomposition

Source decomposition included add-back models, leave-one-source-out controls, host-only controls, host shuffle, and host corruption curves. Matched comparison criteria required the same split, seed set, backbone, label set, train/validation/test partition, training budget, evaluation thresholding, and calibration status. The final QC table verified that all 38 source-decomposition comparisons satisfied the matched comparison criteria.

## Calibration and validation-targeted candidate prioritization

Validation data were used for model selection and post hoc probability calibration/threshold selection. The test set was used only for final evaluation. Candidate discovery was interpreted as calibrated prioritization rather than held-out FDR control or experimental validation. The candidate unit was a protein-label assignment, which was reported separately from the number of unique proteins and unique genomes.

## Module discovery null

Observed module metrics were compared with shuffled null distributions generated by preserving cluster-size structure while randomizing assignments. Metrics included neighborhood consistency, structural/membrane enrichment, context-sensitive label enrichment, and family recurrence. Empirical p-values were computed as the fraction of null iterations with metric values at least as large as observed. Family recurrence was evaluated but did not exceed the null distribution; therefore, it was not used as evidence for broad cross-family conservation.
"""
    (text_dir / "methods_strengthened_sections.md").write_text(methods, encoding="utf-8")

    legends = """# Main Figure Legends

## Figure 1. Leakage-aware benchmark design

Default split evaluation is optimistic. Panels define the de novo and annotation-refinement tasks, identify allowed and forbidden features, compare exact sequence transfer and nearest-neighbor label-transfer performance across splits, and show that removing residual exact-transfer proteins from the family-heldout test set retains the context gain.

## Figure 2. Main OOD performance with honest uncertainty

Genome-aware pLM models show positive family-heldout macro-level point estimates with unresolved family-block uncertainty and a stable host-heldout gain. Error bars show block-bootstrap 95% confidence intervals for paired deltas.

## Figure 3. Source decomposition and leakage controls

Local/genome organization has larger matched point estimates than host-only add-back, while forbidden-feature and host-corruption controls argue against annotation leakage or host metadata as sufficient explanations. Panels show add-back comparisons, local/genome versus host-only gains, host corruption and shuffle controls, and forbidden-feature PASS status.

## Figure 4. Context dependence atlas

Context dependence is label-specific, with the strongest example in a structural/assembly-associated label and smaller positive deltas in several additional labels. Nucleocapsid is shown as a representative context-sensitive label, with post hoc false-positive synonym audit used only as label-incompleteness evidence.

## Figure 5. Calibrated candidate prioritization

Calibration yields a prioritized, not automatically validated, candidate set. Candidate counts are reported as validation-targeted protein-label assignments, unique proteins, poorly annotated proteins, and high-context-gain candidates.

## Figure 6. Module discovery and representative module-supported examples

Module clusters provide interpretable genomic support for candidate functional assignments. Observed modules show small enrichments over shuffled nulls for neighborhood consistency, structural/membrane enrichment, and context-sensitive label enrichment; family recurrence is reported as a negative/neutral supplementary result.
"""
    (text_dir / "figure_legends.md").write_text(legends, encoding="utf-8")

    interpretation_limits = """# Interpretation Limits

- Family-heldout context results are positive at the macro-level point-estimate scale, with family-block uncertainty overlapping zero.
- Host-heldout results are complementary OOD support rather than the primary benchmark.
- Validation-targeted candidate gates prioritize protein-label assignments; they are not experimental validation.
- Annotation-derived priors support only the separate annotation-refinement setting.
- Broad cross-family module recurrence is not claimed because family recurrence did not exceed the shuffled null.
- Nucleocapsid is a representative context-sensitive label with limited held-out family count, not proof of universal context generalization.
"""
    (text_dir / "interpretation_limits.md").write_text(interpretation_limits, encoding="utf-8")

    panel_specs = [
        ("Figure 1", "A", "three task definitions", "schematic/text", "main"),
        ("Figure 1", "B", "allowed/forbidden features", "feature table", "main"),
        ("Figure 1", "C", "exact transfer default vs family", "bar", "main"),
        ("Figure 1", "D", "NN macro AP default vs family", "bar", "main"),
        ("Figure 1", "E", "strict-zero sensitivity", "bar", "main"),
        ("Figure 2", "A", "family pLM raw scores", "grouped bar", "main"),
        ("Figure 2", "B", "family deltas with family-block CI", "point interval", "main"),
        ("Figure 2", "C", "host deltas with host-block CI", "point interval", "main"),
        ("Figure 2", "D", "strict-zero subset delta", "bar", "main"),
        ("Figure 2", "E", "per-label delta preview", "waterfall", "main"),
        ("Figure 3", "A", "addback", "bar", "main"),
        ("Figure 3", "B", "local+genome vs host-only", "bar", "main"),
        ("Figure 3", "C", "host corruption curve", "line", "main"),
        ("Figure 3", "D", "leave-one-source-out", "bar", "main"),
        ("Figure 3", "S", "host shuffle", "bar", "supp/main optional"),
        ("Figure 3", "E", "forbidden feature PASS", "count/table", "main/supp"),
        ("Figure 4", "A", "AP_protein vs delta_AP", "scatter", "main"),
        ("Figure 4", "B", "functional group delta", "box/bar", "main"),
        ("Figure 4", "C", "top context-sensitive labels", "bar/heatmap", "main"),
        ("Figure 4", "D", "nucleocapsid PR curve", "line", "main"),
        ("Figure 4", "E", "nucleocapsid FP audit", "bar", "main/supp"),
        ("Figure 5", "A", "validation-targeted precision-gate workflow", "schematic/text", "main"),
        ("Figure 5", "B", "candidate breakdown", "bar", "main"),
        ("Figure 5", "C", "confidence vs context gain", "scatter", "main"),
        ("Figure 5", "D", "top high-gain candidates", "table", "main/supp"),
        ("Figure 5", "E", "case selection funnel", "bar/funnel", "main"),
        ("Figure 6", "A", "module workflow", "schematic/text", "main"),
        ("Figure 6", "B", "339 clusters summary", "text/count", "main"),
        ("Figure 6", "C", "observed vs null", "point interval", "main"),
        ("Figure 6", "D-F", "3 case studies", "case panels", "main"),
    ]
    pd.DataFrame(panel_specs, columns=["figure", "panel", "content", "plot_type", "placement"]).to_csv(
        text_dir / "figure_panel_specs.tsv", sep="\t", index=False
    )

    claim_map = claim[["claim", "evidence", "strength", "caveat", "manuscript_wording", "supporting_files"]].copy()
    claim_map.to_csv(text_dir / "claim_evidence_map.tsv", sep="\t", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-dir", type=Path, required=True, help="Directory produced by build_v2_manuscript_assets.py.")
    parser.add_argument("--qc-dir", type=Path, required=True, help="Extracted qc_review directory for PR curve source.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for strengthened package.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    assets = args.assets_dir.resolve()
    qc_dir = args.qc_dir.resolve()
    out = args.output_dir.resolve()
    fig_dir = out / "figures"
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    build_figure1(assets, fig_dir)
    build_figure2(assets, fig_dir)
    build_figure3(assets, qc_dir, fig_dir)
    build_figure4(assets, qc_dir, fig_dir)
    build_figure5(assets, fig_dir)
    build_figure6(assets, fig_dir)
    build_text_outputs(assets, out)
    summary = {
        "assets_dir": str(assets),
        "qc_dir": str(qc_dir),
        "output_dir": str(out),
        "generated_files": sorted(str(path.relative_to(out)) for path in out.rglob("*") if path.is_file()),
    }
    (out / "plos_cb_strengthening_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(out), "file_count": len(summary["generated_files"])}, indent=2))


if __name__ == "__main__":
    main()
