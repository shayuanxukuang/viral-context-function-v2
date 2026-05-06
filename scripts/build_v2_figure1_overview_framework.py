"""Render Figure 1: overview of the leakage-aware ViruFunc V2 framework."""

from __future__ import annotations

from pathlib import Path
import textwrap


WIDTH = 1600
HEIGHT = 950

BLUE = "#4C78A8"
GREEN = "#54A24B"
RED = "#E45756"
GRAY = "#8D99A6"
YELLOW = "#F2CF5B"
SOFT = "#F7F9FB"
DARK = "#253047"
LINE = "#586672"


def wrap(text: str, width: int) -> str:
    return "\n".join(textwrap.wrap(text, width=width, break_long_words=False))


def wrap_bullets(items: list[str], width: int) -> str:
    lines: list[str] = []
    for item in items:
        wrapped = textwrap.wrap(item, width=width, break_long_words=False)
        if not wrapped:
            continue
        lines.append(f"- {wrapped[0]}")
        lines.extend(f"  {line}" for line in wrapped[1:])
    return "\n".join(lines)


def add_round_rect(ax, x, y, w, h, fc="white", ec=LINE, lw=1.8, radius=12):
    from matplotlib.patches import FancyBboxPatch

    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    return patch


def add_card(ax, x, y, w, h, title, body=None, fc="#FBFDFF", ec=LINE):
    add_round_rect(ax, x, y, w, h, fc=fc, ec=ec, lw=1.8, radius=12)
    ax.text(x + 16, y + h - 28, title, fontsize=9.2, fontweight="bold", color=DARK, va="top")
    if body:
        ax.text(x + 16, y + h - 62, body, fontsize=6.7, color=DARK, va="top", linespacing=1.16)


def add_section(ax, x, y, w, h, title, body, accent=LINE, fc="white", body_size=7.2):
    add_round_rect(ax, x, y, w, h, fc=fc, ec=accent, lw=1.3, radius=8)
    ax.text(x + 12, y + h - 18, title, fontsize=7.6, fontweight="bold", color=DARK, va="top")
    ax.text(x + 12, y + h - 42, body, fontsize=body_size, color=DARK, va="top", linespacing=1.12)


def add_center_box(ax, x, y, w, h, text, fc, ec, fontsize=7.0):
    add_round_rect(ax, x, y, w, h, fc=fc, ec=ec, lw=1.3, radius=8)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        fontsize=fontsize,
        color=DARK,
        ha="center",
        va="center",
        fontweight="bold",
        linespacing=1.08,
    )


def add_chip(ax, x, y, text, fc, ec=None, color="white", w=None):
    if w is None:
        w = max(86, 7.2 * len(text) + 20)
    add_round_rect(ax, x, y, w, 24, fc=fc, ec=ec or fc, lw=1.2, radius=7)
    ax.text(x + w / 2, y + 12, text, fontsize=6.9, color=color, ha="center", va="center", fontweight="bold")
    return w


def add_arrow(ax, x1, y1, x2, y2, color=LINE, lw=1.7, alpha=1.0):
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", lw=lw, color=color, mutation_scale=14, alpha=alpha, shrinkA=5, shrinkB=5),
    )


def add_warning(ax, x, y, scale=1.0):
    from matplotlib.patches import Polygon

    pts = [(x, y + 18 * scale), (x + 18 * scale, y - 14 * scale), (x - 18 * scale, y - 14 * scale)]
    ax.add_patch(Polygon(pts, closed=True, facecolor=RED, edgecolor="#9B2C2C", linewidth=1.0))
    ax.text(x, y - 3 * scale, "!", ha="center", va="center", fontsize=15 * scale, color="white", fontweight="bold")


def add_database_icon(ax, x, y, color=BLUE):
    from matplotlib.patches import Ellipse, Rectangle

    ax.add_patch(Rectangle((x, y - 24), 30, 34, facecolor="#EAF2FB", edgecolor=color, linewidth=1.1))
    ax.add_patch(Ellipse((x + 15, y + 10), 30, 11, facecolor="#D9E8F7", edgecolor=color, linewidth=1.1))
    ax.add_patch(Ellipse((x + 15, y - 24), 30, 11, facecolor="#EAF2FB", edgecolor=color, linewidth=1.1))
    ax.plot([x, x], [y - 24, y + 10], color=color, linewidth=0.9)
    ax.plot([x + 30, x + 30], [y - 24, y + 10], color=color, linewidth=0.9)


def add_genome_icon(ax, x, y, color=GREEN):
    import numpy as np

    t = np.linspace(0, 1, 60)
    ax.plot(x + 48 * t, y + 7 * np.sin(2 * np.pi * t), color=color, linewidth=1.5)
    ax.plot(x + 48 * t, y - 7 * np.sin(2 * np.pi * t), color=color, linewidth=1.5)
    for i in range(6):
        xx = x + 7 + i * 7
        ax.plot([xx, xx], [y - 6, y + 6], color=color, linewidth=0.8, alpha=0.75)


def add_heatmap_icon(ax, x, y):
    from matplotlib.patches import Rectangle

    colors = ["#DCEBFA", "#84B7DE", "#54A24B", "#F2CF5B", "#EDF2F7"]
    for i in range(4):
        for j in range(3):
            ax.add_patch(Rectangle((x + i * 11, y + j * 11), 9, 9, facecolor=colors[(i + j) % len(colors)], edgecolor="white", linewidth=0.45))


def main() -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none"})
    fig, ax = plt.subplots(figsize=(16, 9.5))
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(0, HEIGHT)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(WIDTH / 2, 912, "Overview of the leakage-aware ViruFunc V2 framework", ha="center", fontsize=18.5, fontweight="bold", color="#111827")
    ax.text(
        WIDTH / 2,
        878,
        "Benchmark design, context-aware modeling, and sequence-structure-context candidate triage",
        ha="center",
        fontsize=9.2,
        color="#606A76",
    )

    xs = [60, 360, 660, 960, 1260]
    w = 280
    y = 330
    h = 500

    # 1. Frozen viral dataset.
    add_card(
        ax,
        xs[0],
        y,
        w,
        h,
        "1. Frozen viral dataset",
        wrap_bullets(
            [
                "713,487 proteins",
                "19,149 genomes",
                "1,283 viral families",
                "17 primary labels",
                "frozen manifests for proteins, genomes, host, taxonomy, coordinates, and splits",
            ],
            34,
        ),
    )
    add_database_icon(ax, xs[0] + 222, 748)
    add_genome_icon(ax, xs[0] + 210, 697)
    add_round_rect(ax, xs[0] + 18, y + 25, w - 36, 88, fc="#F7FAFC", ec="#C4CCD5", lw=1.2, radius=8)
    ax.text(xs[0] + 32, y + 92, "Goal", fontsize=7.8, fontweight="bold", color=DARK, va="top")
    ax.text(xs[0] + 32, y + 66, wrap("Credible benchmark for viral protein function annotation.", 38), fontsize=6.9, color=DARK, va="top")
    ax.text(xs[0] + 18, y + 142, "V2 data freeze", fontsize=8.0, color="#657381", fontweight="bold")

    # 2-3. Tasks and evaluation.
    add_card(ax, xs[1], y, w, h, "2-3. Tasks and splits")
    ax.text(xs[1] + 18, y + h - 72, "Task settings", fontsize=7.8, fontweight="bold", color=DARK, va="top")
    add_center_box(ax, xs[1] + 18, y + h - 128, w - 36, 42, "Protein-only de novo\nTarget only", "#EEF5FC", BLUE, fontsize=6.5)
    add_center_box(ax, xs[1] + 18, y + h - 184, w - 36, 42, "Genome-aware de novo\nTarget + neighbors + gene order", "#EEF8EE", GREEN, fontsize=6.1)
    add_center_box(ax, xs[1] + 18, y + h - 240, w - 36, 42, "Annotation refinement\nSeparate setting", "#F8F8F8", GRAY, fontsize=6.2)
    ax.text(xs[1] + 18, y + 230, "Leakage-aware evaluation", fontsize=7.8, fontweight="bold", color=DARK, va="top")
    add_chip(ax, xs[1] + 18, y + 190, "Default", RED, color="white", w=72)
    add_chip(ax, xs[1] + 98, y + 190, "Family-heldout", "#EAF7EF", ec=GREEN, color="#1F6F35", w=120)
    add_chip(ax, xs[1] + 18, y + 158, "Host-heldout", BLUE, color="white", w=98)
    add_warning(ax, xs[1] + 30, y + 108, 0.70)
    ax.text(xs[1] + 56, y + 124, "Default split is optimistic", fontsize=7.3, fontweight="bold", color=RED, va="center")
    ax.text(xs[1] + 18, y + 89, "Exact transfer: 27.4% -> 1.37%", fontsize=6.9, color=DARK)
    ax.text(xs[1] + 18, y + 66, "NN macro AP: 0.369 -> 0.048", fontsize=6.9, color=DARK)
    ax.text(xs[1] + 18, y + 43, "Strict-zero sensitivity retained context gain", fontsize=6.3, color="#4B5563")

    # 4. Model comparison.
    add_card(ax, xs[2], y, w, h, "4. Model comparison")
    add_center_box(ax, xs[2] + 18, y + h - 178, 112, 88, "Protein-only\nESM-2 target", "#EEF5FC", BLUE, fontsize=6.1)
    add_center_box(
        ax,
        xs[2] + 150,
        y + h - 178,
        112,
        88,
        "Genome-aware\nneighbors + order\naudited host",
        "#EEF8EE",
        GREEN,
        fontsize=5.8,
    )
    add_arrow(ax, xs[2] + 74, y + h - 185, xs[2] + 140, y + h - 232, BLUE, lw=1.2)
    add_arrow(ax, xs[2] + 206, y + h - 185, xs[2] + 140, y + h - 232, GREEN, lw=1.2)
    add_section(
        ax,
        xs[2] + 18,
        y + 136,
        w - 36,
        145,
        "OOD readout",
        wrap_bullets(
            [
                "family-heldout macro point estimates positive",
                "CI overlaps zero; micro AP decreased",
                "host-heldout supportive",
            ],
            35,
        ),
        accent="#C7D0D9",
        body_size=6.1,
    )
    add_section(
        ax,
        xs[2] + 18,
        y + 26,
        w - 36,
        78,
        "Forbidden in de novo",
        "No product text,\nhit counts, neighbor labels,\nor annotation-text embeddings.",
        accent=RED,
        fc="#FFF5F5",
        body_size=5.4,
    )

    # 5-6. Interpretation.
    add_card(ax, xs[3], y, w, h, "5-6. Interpretation")
    add_section(
        ax,
        xs[3] + 18,
        y + 277,
        w - 36,
        156,
        "Source interpretation",
        wrap_bullets(
            [
                "source decomposition",
                "local + genome > host-only",
                "forbidden-feature audit: PASS",
                "host corruption/shuffle argues against host-only explanation",
            ],
            34,
        ),
        accent="#C7D0D9",
        body_size=6.0,
    )
    add_heatmap_icon(ax, xs[3] + 218, y + 392)
    add_section(
        ax,
        xs[3] + 18,
        y + 93,
        w - 36,
        156,
        "Biological interpretation",
        wrap_bullets(
            [
                "context dependence is label-specific",
                "10 / 17 labels positive",
                "strongest example: nucleocapsid",
                "not a uniform gain across all functions",
            ],
            34,
        ),
        accent="#C7D0D9",
        body_size=6.0,
    )
    add_chip(ax, xs[3] + 34, y + 112, "nucleocapsid", "#EAF7EF", ec=GREEN, color="#1F6F35", w=114)

    # 7. Candidate prioritization.
    add_card(ax, xs[4], y, w, h, "7. Candidate prioritization")
    add_section(
        ax,
        xs[4] + 18,
        y + 352,
        w - 36,
        78,
        "Calibration",
        "Validation-targeted gate;\nprioritized hypotheses.",
        accent="#C7D0D9",
        body_size=6.0,
    )
    add_section(
        ax,
        xs[4] + 18,
        y + 242,
        w - 36,
        88,
        "72-target panel",
        wrap_bullets(["27 high-context-gain", "27 matched controls", "18 known-positive controls"], 32),
        accent="#C7D0D9",
        body_size=6.0,
    )
    add_section(
        ax,
        xs[4] + 18,
        y + 134,
        w - 36,
        86,
        "Sequence-structure-context",
        "MMseqs2, ESMFold,\nFoldseek PDB100,\nand module evidence.",
        accent="#C7D0D9",
        body_size=5.9,
    )
    add_section(
        ax,
        xs[4] + 18,
        y + 18,
        w - 36,
        100,
        "Key Figure 6 takeaway",
        wrap_bullets(
            [
                "less sequence/structure-resolved",
                "module-supported weak evidence",
                "context complements; not replaces",
            ],
            30,
        ),
        accent=YELLOW,
        fc="#FFF8D9",
        body_size=5.25,
    )

    # Cross-column arrows.
    for i in range(4):
        add_arrow(ax, xs[i] + w + 8, y + h / 2, xs[i + 1] - 8, y + h / 2)
    add_arrow(ax, xs[4] + w / 2, y - 8, xs[4] + w / 2, 270, GREEN, lw=1.8)
    # Summary band.
    ax.add_patch(Rectangle((60, 145), 1480, 110, facecolor="#EEF8EE", edgecolor=GREEN, linewidth=1.6))
    ax.text(84, 222, "Main conclusion", fontsize=11.2, fontweight="bold", color="#1F6F35", va="center")
    ax.text(
        84,
        185,
        wrap(
            "Leakage-aware evaluation shows that genome context is not a uniformly superior signal; it complements sequence and structure by prioritizing candidates when conventional evidence is weak, ambiguous, or incomplete.",
            145,
        ),
        fontsize=8.5,
        color=DARK,
        va="center",
    )

    ax.text(
        1538,
        72,
        "De novo claims exclude annotation-derived priors; sequence/structure/module evidence are used post hoc for candidate triage.",
        fontsize=6.5,
        color="#6B7280",
        ha="right",
    )

    out_prefix = Path("manuscript/v2_plos_cb/figures/figure1_leakage_aware_benchmark_design")
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out_prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_prefix.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    print(
        {
            "figure_png": str(out_prefix.with_suffix(".png")),
            "figure_pdf": str(out_prefix.with_suffix(".pdf")),
            "figure_svg": str(out_prefix.with_suffix(".svg")),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
