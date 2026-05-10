#!/usr/bin/env python3
"""Render Figure 1: compact framework overview plus leakage diagnostics."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import textwrap
from typing import Iterable


PALETTE = {
    "protein": "#4C78A8",
    "context": "#54A24B",
    "risk": "#E45756",
    "support": "#54A24B",
    "neutral": "#8A8F98",
    "light": "#F4F7FA",
    "line": "#586672",
    "dark": "#253047",
    "yellow": "#F2CF5B",
}


def wrap(text: str, width: int) -> str:
    return "\n".join(textwrap.wrap(text, width=width, break_long_words=False))


def bullet_lines(items: Iterable[str], width: int = 28) -> str:
    lines: list[str] = []
    for item in items:
        wrapped = textwrap.wrap(str(item), width=width, break_long_words=False)
        if not wrapped:
            continue
        lines.append(f"- {wrapped[0]}")
        lines.extend(f"  {part}" for part in wrapped[1:])
    return "\n".join(lines)


def read_leakage_table(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Required Figure 1 leakage summary not found: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def first_value(rows: list[dict[str, str]], *, panel: str | None = None, scheme: str | None = None, metric: str | None = None) -> float:
    for row in rows:
        if panel is not None and row.get("panel") != panel:
            continue
        if scheme is not None and row.get("scheme") != scheme:
            continue
        if metric is not None and row.get("metric") != metric:
            continue
        return float(row["value"])
    filters = {"panel": panel, "scheme": scheme, "metric": metric}
    raise KeyError(f"Metric not found in Figure 1 leakage summary: {filters}")


def matching_rows(rows: list[dict[str, str]], *, panel: str, scheme: str | None = None) -> list[dict[str, str]]:
    out = [row for row in rows if row.get("panel") == panel and (scheme is None or row.get("scheme") == scheme)]
    if not out:
        raise KeyError(f"No rows found for panel={panel!r}, scheme={scheme!r}")
    return out


def panel_label(ax, label: str, title: str) -> None:
    ax.set_title(f"{label}. {title}", loc="left", fontweight="bold", fontsize=9.6, color=PALETTE["dark"])


def rounded_box(ax, xy, width, height, title, body, fc="white", ec=None, title_color=None, title_size=8.4, body_size=6.9):
    from matplotlib.patches import FancyBboxPatch

    x, y = xy
    ec = ec or PALETTE["line"]
    title_color = title_color or PALETTE["dark"]
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.025",
        transform=ax.transAxes,
        facecolor=fc,
        edgecolor=ec,
        linewidth=1.4,
    )
    ax.add_patch(patch)
    ax.text(x + 0.018, y + height - 0.08, title, transform=ax.transAxes, fontsize=title_size, fontweight="bold", color=title_color, va="top")
    ax.text(x + 0.018, y + height - 0.18, body, transform=ax.transAxes, fontsize=body_size, color=PALETTE["dark"], va="top", linespacing=1.12)
    return patch


def draw_panel_a(ax) -> None:
    ax.axis("off")
    panel_label(ax, "A", "Framework overview")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    boxes = [
        (
            0.01,
            "Frozen viral data",
            "713,487 proteins\n19,149 genomes\n1,283 families; 17 labels",
            "#FBFDFF",
            PALETTE["line"],
        ),
        (
            0.205,
            "Tasks and splits",
            "protein-only vs genome-aware\nfamily-heldout primary\nrefinement separated",
            "#FBFDFF",
            PALETTE["line"],
        ),
        (
            0.40,
            "Model comparison",
            "frozen pLM baseline\ngenome-aware pLM\nmacro delta > 0; CI includes 0",
            "#EEF8EE",
            PALETTE["context"],
        ),
        (
            0.595,
            "Interpretation",
            "feature audit PASS\nlocal+genome > host-only\nlabel-specific signal",
            "#FBFDFF",
            PALETTE["line"],
        ),
        (
            0.79,
            "Candidate triage",
            "validation-targeted gate\n72-target panel\nmodule-coherent unresolved",
            "#FFF8D9",
            PALETTE["yellow"],
        ),
    ]
    for x, title, body, fc, ec in boxes:
        rounded_box(ax, (x, 0.28), 0.18, 0.56, title, body, fc=fc, ec=ec)

    for start in [0.19, 0.385, 0.58, 0.775]:
        ax.annotate(
            "",
            xy=(start + 0.015, 0.56),
            xytext=(start - 0.01, 0.56),
            xycoords=ax.transAxes,
            arrowprops=dict(arrowstyle="-|>", color=PALETTE["line"], lw=1.5, mutation_scale=13),
        )

    ax.text(
        0.5,
        0.095,
        "Genome context provides label-specific candidate-prioritization signal in weak or ambiguous evidence regimes.",
        transform=ax.transAxes,
        ha="center",
        fontsize=8.3,
        color="#1F6F35",
        fontweight="bold",
    )


def draw_panel_b(ax) -> None:
    ax.axis("off")
    panel_label(ax, "B", "Allowed/forbidden feature boundary")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    rounded_box(
        ax,
        (0.04, 0.16),
        0.42,
        0.68,
        "Allowed in de novo",
        bullet_lines(["target pLM embedding", "neighbor pLM embeddings", "non-text genome topology", "strand / gaps / overlaps / segments"], 25),
        fc="#EEF5FC",
        ec=PALETTE["protein"],
        title_size=7.3,
        body_size=6.2,
    )
    rounded_box(
        ax,
        (0.54, 0.16),
        0.42,
        0.68,
        "Forbidden in de novo",
        bullet_lines(["product text", "database hit counts", "neighbor labels/counts", "annotation text embeddings"], 22),
        fc="#FFF5F5",
        ec=PALETTE["risk"],
        title_color=PALETTE["risk"],
        title_size=7.3,
        body_size=6.2,
    )


def draw_exact_transfer(ax, rows: list[dict[str, str]]) -> None:
    schemes = ["default", "family_holdout"]
    labels = ["Default", "Family-heldout"]
    vals = [first_value(rows, panel="C", scheme=scheme, metric="exact_sequence_transfer_rate") * 100 for scheme in schemes]
    ax.bar(labels, vals, color=[PALETTE["risk"], PALETTE["support"]], width=0.65)
    panel_label(ax, "C", "Exact sequence transfer")
    ax.set_ylabel("test proteins with exact train match (%)", fontsize=9)
    ax.set_ylim(0, max(30, max(vals) * 1.18))
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=8.5)
    for idx, val in enumerate(vals):
        ax.text(idx, val + 0.8, f"{val:.1f}%", ha="center", fontsize=9, fontweight="bold")


def draw_nearest_neighbor(ax, rows: list[dict[str, str]]) -> None:
    schemes = ["default", "family_holdout"]
    labels = ["Default", "Family-heldout"]
    vals = [first_value(rows, panel="D", scheme=scheme, metric="nearest_neighbor_macro_ap") for scheme in schemes]
    ax.bar(labels, vals, color=[PALETTE["risk"], PALETTE["support"]], width=0.65)
    panel_label(ax, "D", "Nearest-neighbor label transfer")
    ax.set_ylabel("macro AP", fontsize=9)
    ax.set_ylim(0, 0.42)
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=8.5)
    for idx, val in enumerate(vals):
        ax.text(idx, val + 0.012, f"{val:.3f}", ha="center", fontsize=9, fontweight="bold")


def draw_strict_zero(ax, rows: list[dict[str, str]]) -> None:
    metrics = [("delta_macro_ap", "macro AP"), ("delta_macro_f1", "macro F1")]
    vals = [first_value(rows, panel="E", scheme="family_holdout_strict_zero_exact_transfer", metric=metric) for metric, _ in metrics]
    ax.bar([label for _, label in metrics], vals, color=PALETTE["context"], width=0.62)
    ax.axhline(0, color="#404040", linewidth=0.8)
    panel_label(ax, "E", "Strict-zero exact-transfer sensitivity")
    ax.set_ylabel("Delta metric\n(genome-aware - protein-only)", fontsize=8.5)
    ax.set_ylim(0, max(vals) * 1.45)
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=8.5)
    for idx, val in enumerate(vals):
        ax.text(idx, val + 0.001, f"+{val:.4f}", ha="center", fontsize=9, fontweight="bold")
    removed = first_value(rows, panel="E", scheme="family_holdout_strict_zero_exact_transfer", metric="removed_exact_transfer_proteins")
    ax.text(0.5, 0.92, f"removed {int(removed)} exact-transfer proteins", transform=ax.transAxes, ha="center", fontsize=8)


def draw_residual_audit(ax, rows: list[dict[str, str]]) -> None:
    residual = matching_rows(rows, panel="F", scheme="family_holdout_exact_transfer_audit")
    label_map = {
        "identical proteins assigned to different families": "cross-family identical",
        "same taxid with different family annotation": "same taxid, different family",
        "shared mobile/module-like element": "mobile/module-like",
        "duplicated-entry-like": "duplicated-entry-like",
    }
    labels = [label_map.get(row.get("note", ""), row.get("note", row.get("metric", ""))) for row in residual]
    vals = [float(row["value"]) for row in residual]
    colors = [PALETTE["neutral"], "#C0C4CC", PALETTE["context"], PALETTE["protein"]]
    ax.barh(labels, vals, color=colors[: len(vals)])
    panel_label(ax, "F", "Residual exact-transfer audit")
    ax.set_xlabel("proteins", fontsize=9)
    ax.invert_yaxis()
    ax.grid(axis="x", color="#E5E7EB", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=8.5)
    for idx, val in enumerate(vals):
        ax.text(val + max(vals) * 0.015 + 3, idx, f"{int(val)}", va="center", fontsize=8)


def save_figure(fig, out_prefix: Path) -> None:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.25)
    fig.savefig(out_prefix.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.25)
    fig.savefig(out_prefix.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.25)


def build_cover(out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none"})
    fig, ax = plt.subplots(figsize=(13.0, 3.1))
    draw_panel_a(ax)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--leakage-summary",
        type=Path,
        default=Path("artifacts/return/v2_manuscript_assets_20260504_rebuilt/figure1_leakage_summary.tsv"),
        help="TSV containing Figure 1 leakage summary metrics.",
    )
    parser.add_argument(
        "--out-prefix",
        type=Path,
        default=Path("manuscript/v2_plos_cb/figures/figure1_leakage_aware_benchmark_design"),
        help="Output prefix for .png/.pdf/.svg figure files.",
    )
    parser.add_argument(
        "--cover-output",
        type=Path,
        default=Path("manuscript/v2_plos_cb/figures/figure1A_framework_overview_schematic.png"),
        help="Optional PNG output for the standalone compact overview panel.",
    )
    return parser.parse_args()


def main() -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    args = parse_args()
    rows = read_leakage_table(args.leakage_summary)

    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none"})
    fig = plt.figure(figsize=(13.2, 8.2))
    fig.subplots_adjust(left=0.085, right=0.985, top=0.965, bottom=0.075)
    gs = GridSpec(3, 3, figure=fig, height_ratios=[0.88, 1.0, 1.02], hspace=0.36, wspace=0.36)

    draw_panel_a(fig.add_subplot(gs[0, :]))
    draw_panel_b(fig.add_subplot(gs[1, 0]))
    draw_exact_transfer(fig.add_subplot(gs[1, 1]), rows)
    draw_nearest_neighbor(fig.add_subplot(gs[1, 2]), rows)
    draw_strict_zero(fig.add_subplot(gs[2, 0]), rows)
    draw_residual_audit(fig.add_subplot(gs[2, 1:]), rows)

    save_figure(fig, args.out_prefix)
    plt.close(fig)

    if args.cover_output:
        build_cover(args.cover_output)

    print(
        {
            "figure_png": str(args.out_prefix.with_suffix(".png")),
            "figure_pdf": str(args.out_prefix.with_suffix(".pdf")),
            "figure_svg": str(args.out_prefix.with_suffix(".svg")),
            "cover_png": str(args.cover_output) if args.cover_output else None,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
