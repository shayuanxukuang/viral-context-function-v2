"""Render manuscript Figure 6 from sequence-structure-context source tables."""

from __future__ import annotations

import argparse
import csv
import math
import textwrap
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case-rankings",
        default="artifacts/return/v2_plos_cb_supplementary_package_20260506/supplementary_tables/S23_figure6_candidate_case_rankings.tsv",
        help="S23 case-ranking table.",
    )
    parser.add_argument(
        "--enrichment",
        default="artifacts/return/v2_plos_cb_supplementary_package_20260506/supplementary_tables/S25_independent_evidence_enrichment.tsv",
        help="S25 independent-evidence enrichment summary.",
    )
    parser.add_argument(
        "--output-prefix",
        default="manuscript/v2_plos_cb/figures/figure6_sequence_structure_context_triangulation",
        help="Output prefix without extension.",
    )
    return parser.parse_args()


def read_tsv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise SystemExit(f"Required table not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def as_float(value: Any, default: float = math.nan) -> float:
    try:
        if value is None or str(value).strip() == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or str(value).strip() == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def wrap(text: str, width: int) -> str:
    return "\n".join(textwrap.wrap(text, width=width, break_long_words=False))


def add_panel_label(ax: Any, label: str, title: str) -> None:
    ax.text(0.0, 1.045, label, transform=ax.transAxes, fontsize=12.5, fontweight="bold", va="bottom")
    ax.text(0.062, 1.045, title, transform=ax.transAxes, fontsize=12.5, fontweight="bold", va="bottom")


def draw_box(
    ax: Any,
    xy: tuple[float, float],
    wh: tuple[float, float],
    text: str,
    fc: str,
    ec: str = "#53616c",
    fontsize: float = 9.2,
) -> None:
    from matplotlib.patches import FancyBboxPatch

    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.1,
        edgecolor=ec,
        facecolor=fc,
        transform=ax.transAxes,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, transform=ax.transAxes)


def draw_arrow(ax: Any, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        xycoords=ax.transAxes,
        textcoords=ax.transAxes,
        arrowprops=dict(arrowstyle="-|>", lw=1.2, color="#53616c", shrinkA=0, shrinkB=0),
    )


def panel_a(ax: Any) -> None:
    add_panel_label(ax, "A", "Evidence workflow")
    ax.set_axis_off()
    draw_box(ax, (0.02, 0.66), (0.22, 0.14), "27 high-context\ncandidates", "#eaf2fb")
    draw_box(ax, (0.02, 0.43), (0.22, 0.14), "27 matched\ncontrols", "#f2f5f8")
    draw_box(ax, (0.02, 0.20), (0.22, 0.14), "18 known-positive\ncontrols", "#f2f5f8", fontsize=8.5)
    draw_box(ax, (0.32, 0.56), (0.20, 0.14), "MMseqs2\nsequence hits", "#fff0d9")
    draw_box(ax, (0.32, 0.31), (0.20, 0.14), "ESMFold\nmonomer models", "#eaf7ef")
    draw_box(ax, (0.57, 0.56), (0.20, 0.14), "Foldseek\nPDB100 search", "#f0e9f8")
    draw_box(ax, (0.57, 0.31), (0.20, 0.14), "Ambiguity +\nmodule evidence", "#fff6cc")
    draw_box(ax, (0.83, 0.435), (0.15, 0.14), "Case roles\nfor Figure 6", "#eaf2fb", fontsize=8.4)
    draw_arrow(ax, (0.24, 0.73), (0.32, 0.63))
    draw_arrow(ax, (0.24, 0.50), (0.32, 0.38))
    draw_arrow(ax, (0.24, 0.27), (0.32, 0.38))
    draw_arrow(ax, (0.52, 0.63), (0.57, 0.63))
    draw_arrow(ax, (0.52, 0.38), (0.57, 0.38))
    draw_arrow(ax, (0.77, 0.63), (0.83, 0.505))
    draw_arrow(ax, (0.77, 0.38), (0.83, 0.505))
    ax.text(0.04, 0.04, "Post hoc evidence only; not de novo model input.", fontsize=8.8, color="#4b5563", transform=ax.transAxes)


def panel_b(ax: Any) -> None:
    add_panel_label(ax, "B", "72-target evidence overview")
    labels = ["High-context\ncands.", "Matched\ncontrols", "Known-pos.\ncontrols", "MMseqs2\ntop hits", "ESMFold\nmodels", "Foldseek\nPDB100 hits"]
    values = [27, 27, 18, 53, 72, 70]
    colors = ["#3b82f6", "#94a3b8", "#94a3b8", "#f59e0b", "#2aa06f", "#8b5cf6"]
    bars = ax.bar(range(len(labels)), values, color=colors, width=0.70)
    ax.set_xticks(range(len(labels)), labels=labels, fontsize=8.2)
    ax.set_ylabel("Targets")
    ax.set_ylim(0, 78)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, axis="y", color="#d7dee8", linewidth=0.7, alpha=0.8)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 1.2, str(value), ha="center", va="bottom", fontsize=11, fontweight="bold")


def panel_c(ax: Any, rows: list[dict[str, str]]) -> None:
    add_panel_label(ax, "C", "Structure-evidence classes")
    candidates = [row for row in rows if row.get("figure6_recommendation") != "matched_control_reference"]
    order = [
        ("low_model_confidence", "Low-confidence\nmodel", "#b45309"),
        ("structure_consistent_but_ambiguous", "Structure-consistent\nbut ambiguous", "#2563eb"),
        ("ambiguous_or_weak_structure_signal", "Ambiguous or weak\nstructure signal", "#64748b"),
        ("structure_hit_on_low_confidence_model", "Hit on low-confidence\nmodel", "#d97706"),
        ("local_structure_consistent", "Local structure\nconsistent", "#16a34a"),
        ("no_foldseek_hit_available", "No PDB100\nFoldseek hit", "#7c3aed"),
    ]
    counts = {key: 0 for key, _, _ in order}
    for row in candidates:
        status = row.get("structure_evidence_status", "")
        if status in counts:
            counts[status] += 1
    labels = [label for key, label, _ in order if counts[key] > 0]
    values = [counts[key] for key, _, _ in order if counts[key] > 0]
    colors = [color for key, _, color in order if counts[key] > 0]
    y = list(range(len(labels)))[::-1]
    ax.barh(y, values, color=colors)
    ax.set_yticks(y, labels=labels, fontsize=9)
    ax.set_xlabel("Candidate assignments")
    ax.set_xlim(0, max(values) + 1.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, axis="x", color="#d7dee8", linewidth=0.7, alpha=0.8)
    for yi, value in zip(y, values):
        ax.text(value + 0.12, yi, str(value), va="center", fontsize=10.5, fontweight="bold")
    ax.text(
        0.0,
        -0.18,
        "Weak, ambiguous, low-confidence, and PDB100-orphan regimes dominate the high-context panel.",
        fontsize=9.3,
        color="#4b5563",
        transform=ax.transAxes,
    )


def panel_d(ax: Any, rows: list[dict[str, str]]) -> None:
    from matplotlib.patches import FancyBboxPatch

    add_panel_label(ax, "D", "Case-study roles")
    ax.set_axis_off()
    roles = [
        ("YP_010680760.1", "Structure-ambiguous context prioritization", "#eef4ff", "High pLDDT; Foldseek hits are functionally diverse."),
        ("YP_009881537.1", "Sequence/structure/context convergence", "#ecfdf3", "Sequence, local structure, context, and module evidence converge."),
        ("YP_009337740.1", "PDB100-orphan prioritized hypothesis", "#f4f1ff", "No PDB100 Foldseek hit; prioritized for downstream validation."),
        ("YP_009666897.1", "High-context-gain exemplar", "#fff7cc", "Largest context gain; low pLDDT keeps structure evidence inconclusive."),
    ]
    by_acc = {row.get("protein_accession"): row for row in rows}
    y_positions = [0.74, 0.50, 0.26, 0.02]
    for (acc, role, color, interpretation), y in zip(roles, y_positions):
        row = by_acc.get(acc, {})
        label = row.get("predicted_label", "").replace("_", " ")
        delta = as_float(row.get("delta_p"))
        plddt = as_float(row.get("mean_plddt"))
        qtm = as_float(row.get("foldseek_top_qtmscore"))
        amb = as_float(row.get("foldseek_ambiguity_index"))
        metrics = f"delta p={delta:.2f}; pLDDT={plddt:.1f}; qTM={qtm:.2f}; ambiguity={amb:.2f}"
        if math.isnan(qtm):
            metrics = f"delta p={delta:.2f}; pLDDT={plddt:.1f}; qTM=NA; ambiguity=NA"
        patch = FancyBboxPatch(
            (0.02, y),
            0.96,
            0.20,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            linewidth=1.0,
            edgecolor="#64748b",
            facecolor=color,
            transform=ax.transAxes,
        )
        ax.add_patch(patch)
        ax.text(0.05, y + 0.145, f"{acc} | {label}", transform=ax.transAxes, fontsize=8.8, fontweight="bold", va="center")
        ax.text(0.05, y + 0.100, role, transform=ax.transAxes, fontsize=8.5, fontweight="bold", va="center")
        ax.text(0.05, y + 0.060, metrics, transform=ax.transAxes, fontsize=7.4, color="#253047", va="center")
        ax.text(0.05, y + 0.025, wrap(interpretation, 70), transform=ax.transAxes, fontsize=7.0, color="#334155", va="bottom")
    ax.text(0.04, -0.05, "Cases are prioritized hypotheses for downstream validation.", fontsize=9.3, color="#4b5563", transform=ax.transAxes)


def panel_e(ax: Any, enrichment_rows: list[dict[str, str]]) -> None:
    add_panel_label(ax, "E", "Independent evidence regime enrichment")
    by_feature = {row.get("feature"): row for row in enrichment_rows}
    features = [
        ("mmseqs_label_agreement", "MMseqs2 label\nagreement"),
        ("foldseek_confident_structural_hit", "Confident\nFoldseek hit"),
        ("context_complement_regime", "Module-supported\nweak-evidence"),
    ]
    x = list(range(len(features)))
    width = 0.34
    candidate_counts = [as_int(by_feature[f].get("candidate_supported")) for f, _ in features]
    control_counts = [as_int(by_feature[f].get("control_supported")) for f, _ in features]
    candidate_totals = [as_int(by_feature[f].get("candidate_total"), 27) for f, _ in features]
    control_totals = [as_int(by_feature[f].get("control_total"), 27) for f, _ in features]
    ax.bar([i - width / 2 for i in x], candidate_counts, width=width, color="#3b82f6", label="High-context")
    ax.bar([i + width / 2 for i in x], control_counts, width=width, color="#94a3b8", label="Matched controls")
    ax.set_xticks(x, labels=[label for _, label in features], fontsize=9)
    ax.set_ylabel("Supported targets")
    ax.set_ylim(0, 28.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, axis="y", color="#d7dee8", linewidth=0.7, alpha=0.8)
    for i, (c, ct, m, mt) in enumerate(zip(candidate_counts, candidate_totals, control_counts, control_totals)):
        ax.text(i - width / 2, c + 0.7, f"{c}/{ct}", ha="center", fontsize=9, fontweight="bold")
        ax.text(i + width / 2, m + 0.7, f"{m}/{mt}", ha="center", fontsize=9, fontweight="bold")
    ax.legend(loc="upper left", frameon=False, ncols=2, fontsize=8.8)
    comp = by_feature["context_complement_regime"]
    raw_or = as_float(comp.get("raw_odds_ratio"))
    ci_low = as_float(comp.get("raw_odds_ratio_woolf_95ci_low"))
    ci_high = as_float(comp.get("raw_odds_ratio_woolf_95ci_high"))
    fisher = as_float(comp.get("fisher_exact_p"))
    stats = "Module-supported weak-evidence: raw OR={:.2f} (95% CI {:.2f}-{:.2f}); McNemar exact p=0.0117; Fisher p={:.3f}".format(
        raw_or, ci_low, ci_high, fisher
    )
    ax.text(
        0.02,
        -0.23,
        wrap(
            "High-context-gain candidates are less often sequence/structure-resolved, but are enriched in the complementarity regime. "
            + stats,
            150,
        ),
        transform=ax.transAxes,
        fontsize=9.0,
        color="#334155",
        va="top",
    )


def main() -> int:
    args = parse_args()
    case_rows = read_tsv(Path(args.case_rankings))
    enrichment_rows = read_tsv(Path(args.enrichment))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.dpi": 140,
        }
    )

    fig = plt.figure(figsize=(13.8, 14.8))
    gs = fig.add_gridspec(
        nrows=3,
        ncols=2,
        height_ratios=[1.05, 1.55, 0.92],
        width_ratios=[1, 1],
        hspace=0.82,
        wspace=0.40,
    )
    panel_a(fig.add_subplot(gs[0, 0]))
    panel_b(fig.add_subplot(gs[0, 1]))
    panel_c(fig.add_subplot(gs[1, 0]), case_rows)
    panel_d(fig.add_subplot(gs[1, 1]), case_rows)
    panel_e(fig.add_subplot(gs[2, :]), enrichment_rows)

    out_prefix = Path(args.output_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_prefix.with_suffix(".png"), dpi=240, bbox_inches="tight")
    fig.savefig(out_prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(
        {
            "figure_png": str(out_prefix.with_suffix(".png")),
            "figure_pdf": str(out_prefix.with_suffix(".pdf")),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
