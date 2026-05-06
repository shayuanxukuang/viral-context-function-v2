from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


FOCUS_GROUPS = {
    "structural_assembly": "main",
    "membrane_entry": "conditional",
    "replication": "mixed_control",
}


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    return (root / path).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collapse atlas v2 outputs into a structural-assembly centered result set.")
    parser.add_argument("--study-root", required=True)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--splits", default="family_holdout,host_holdout")
    return parser.parse_args()


def read_tsv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


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


def as_float(row: dict[str, Any], key: str) -> float | None:
    try:
        value = row.get(key, "")
        if value in {"", None}:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def direction(delta: float | None, ci_low: float | None, ci_high: float | None) -> str:
    if delta is None:
        return "missing"
    if ci_low is not None and ci_high is not None:
        if ci_low > 0:
            return "positive_ci_excludes_zero"
        if ci_high < 0:
            return "negative_ci_excludes_zero"
    if delta > 0:
        return "positive_direction"
    if delta < 0:
        return "negative_direction"
    return "zero"


def conclusion(group: str, split: str, delta: float | None, ci_low: float | None, ci_high: float | None) -> str:
    state = direction(delta, ci_low, ci_high)
    if group == "structural_assembly":
        if state == "positive_ci_excludes_zero":
            return "structural_axis_supported"
        if state == "positive_direction":
            return "structural_axis_positive_but_uncertain"
        return "structural_axis_not_supported_in_split"
    if group == "membrane_entry":
        if state == "positive_ci_excludes_zero":
            return "conditional_membrane_context_signal"
        if state == "positive_direction":
            return "membrane_positive_direction_only"
        return "membrane_not_global_positive"
    if group == "replication":
        if state == "positive_ci_excludes_zero":
            return "replication_positive_in_this_split"
        if state == "negative_ci_excludes_zero":
            return "replication_negative_in_this_split"
        return "replication_mixed_or_neutral"
    return state


def maybe_plot_group_summary(output_dir: Path, rows: list[dict[str, Any]]) -> str:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:
        return ""
    if not rows:
        return ""
    splits = list(dict.fromkeys(str(row["split_scheme"]) for row in rows))
    groups = list(FOCUS_GROUPS)
    x = np.arange(len(groups))
    width = 0.36 if len(splits) > 1 else 0.6
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    for split_idx, split in enumerate(splits):
        split_rows = {row["label_group"]: row for row in rows if row["split_scheme"] == split}
        offsets = x + (split_idx - (len(splits) - 1) / 2.0) * width
        values = [float(split_rows[group]["delta_micro_f1"]) if group in split_rows and split_rows[group].get("delta_micro_f1") not in {"", None} else 0.0 for group in groups]
        lows = [float(split_rows[group]["delta_micro_f1_ci_low"]) if group in split_rows and split_rows[group].get("delta_micro_f1_ci_low") not in {"", None} else values[idx] for idx, group in enumerate(groups)]
        highs = [float(split_rows[group]["delta_micro_f1_ci_high"]) if group in split_rows and split_rows[group].get("delta_micro_f1_ci_high") not in {"", None} else values[idx] for idx, group in enumerate(groups)]
        yerr = [np.asarray(values) - np.asarray(lows), np.asarray(highs) - np.asarray(values)]
        ax.bar(offsets, values, width=width, label=split, alpha=0.85)
        ax.errorbar(offsets, values, yerr=yerr, fmt="none", color="#333333", linewidth=0.8, capsize=2)
    ax.axhline(0.0, color="#444444", linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=20, ha="right")
    ax.set_ylabel("Context minus protein-only micro-F1")
    ax.set_title("Structural-axis atlas summary")
    ax.legend(frameon=False)
    fig.tight_layout()
    path = output_dir / "structural_axis_group_summary.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return str(path)


def main() -> int:
    args = parse_args()
    root = repo_root()
    study_root = resolve_path(root, args.study_root)
    output_dir = resolve_path(root, args.output_dir) if args.output_dir else study_root / "structural_axis_atlas"
    output_dir.mkdir(parents=True, exist_ok=True)
    splits = [token.strip() for token in args.splits.split(",") if token.strip()]

    group_rows: list[dict[str, Any]] = []
    strata_rows: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for split in splits:
        atlas_dir = study_root / f"context_atlas_plain.{split}.v2"
        group_path = atlas_dir / "group_summary.tsv"
        strata_path = atlas_dir / "stratified_group_summary.tsv"
        label_path = atlas_dir / "label_deltas.tsv"
        if not group_path.exists():
            missing.append(str(group_path))
            continue
        for row in read_tsv(group_path):
            group = str(row.get("label_group", ""))
            if group not in FOCUS_GROUPS:
                continue
            delta = as_float(row, "delta_micro_f1")
            ci_low = as_float(row, "delta_micro_f1_ci_low")
            ci_high = as_float(row, "delta_micro_f1_ci_high")
            group_rows.append(
                {
                    "split_scheme": split,
                    "axis_role": FOCUS_GROUPS[group],
                    "label_group": group,
                    "mean_delta_average_precision": row.get("mean_delta_average_precision", ""),
                    "mean_delta_f1": row.get("mean_delta_f1", ""),
                    "delta_micro_f1": row.get("delta_micro_f1", ""),
                    "delta_micro_f1_ci_low": row.get("delta_micro_f1_ci_low", ""),
                    "delta_micro_f1_ci_high": row.get("delta_micro_f1_ci_high", ""),
                    "delta_micro_f1_permutation_pvalue": row.get("delta_micro_f1_permutation_pvalue", ""),
                    "positive_labels": row.get("positive_labels", ""),
                    "direction": direction(delta, ci_low, ci_high),
                    "conclusion": conclusion(group, split, delta, ci_low, ci_high),
                }
            )
        if strata_path.exists():
            for row in read_tsv(strata_path):
                group = str(row.get("label_group", ""))
                if group not in FOCUS_GROUPS:
                    continue
                delta = as_float(row, "delta_micro_f1")
                ci_low = as_float(row, "delta_micro_f1_ci_low")
                ci_high = as_float(row, "delta_micro_f1_ci_high")
                strata_rows.append(
                    {
                        "split_scheme": split,
                        "axis_role": FOCUS_GROUPS[group],
                        "label_group": group,
                        "stratum_field": row.get("stratum_field", ""),
                        "stratum_value": row.get("stratum_value", ""),
                        "delta_micro_f1": row.get("delta_micro_f1", ""),
                        "delta_micro_f1_ci_low": row.get("delta_micro_f1_ci_low", ""),
                        "delta_micro_f1_ci_high": row.get("delta_micro_f1_ci_high", ""),
                        "delta_micro_f1_permutation_pvalue": row.get("delta_micro_f1_permutation_pvalue", ""),
                        "positive_labels": row.get("positive_labels", ""),
                        "protein_count": row.get("protein_count", ""),
                        "direction": direction(delta, ci_low, ci_high),
                    }
                )
        if label_path.exists():
            for row in read_tsv(label_path):
                group = str(row.get("label_group", ""))
                if group not in FOCUS_GROUPS:
                    continue
                label_rows.append({"split_scheme": split, "axis_role": FOCUS_GROUPS[group], **row})

    structural_ok_splits = [
        row["split_scheme"]
        for row in group_rows
        if row["label_group"] == "structural_assembly" and str(row["direction"]).startswith("positive")
    ]
    membrane_positive_strata = [
        row
        for row in strata_rows
        if row["label_group"] == "membrane_entry" and row["direction"] == "positive_ci_excludes_zero"
    ]
    replication_states = [row["direction"] for row in group_rows if row["label_group"] == "replication"]
    plot_path = maybe_plot_group_summary(output_dir, group_rows)

    write_tsv(output_dir / "structural_axis_group_summary.tsv", group_rows)
    write_tsv(output_dir / "structural_axis_stratified_summary.tsv", strata_rows)
    write_tsv(output_dir / "structural_axis_label_deltas.tsv", label_rows)
    report = {
        "created_at": timestamp(),
        "study_root": str(study_root),
        "splits": splits,
        "missing_inputs": missing,
        "structural_positive_direction_splits": structural_ok_splits,
        "membrane_positive_ci_strata_count": len(membrane_positive_strata),
        "replication_global_directions": replication_states,
        "plot_path": plot_path,
        "interpretation": {
            "structural_assembly": "main axis if positive in both splits; CI-excluding-zero is stronger than direction-only.",
            "membrane_entry": "conditional axis; use only strata with biological rationale and stable CI.",
            "replication": "control/mixed group; do not force a binary interpretation.",
        },
    }
    (output_dir / "structural_axis_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "group_rows": len(group_rows), "strata_rows": len(strata_rows)}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
