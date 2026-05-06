from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize the clean pLM quartet into a compact TSV.")
    parser.add_argument("--input-root", required=True, help="Run root produced by run_task_mode_suite.sh")
    parser.add_argument(
        "--cnn-baseline-root",
        default="runs/task_mode_suite_server",
        help="Optional CNN clean benchmark root for same-config deltas",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Output TSV path. Defaults to <input-root>/plm_quartet_summary.tsv",
    )
    return parser.parse_args()


def load_rows(input_root: Path) -> list[dict[str, Any]]:
    summary_path = input_root / "suite_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing suite summary: {summary_path}")
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = payload.get("rows", [])
    if not isinstance(rows, list):
        raise ValueError(f"Malformed suite summary: {summary_path}")
    return [row for row in rows if isinstance(row, dict)]


def maybe_float(value: Any) -> float | None:
    try:
        if value in {"", None}:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def canonical_run_name(split: str, task_mode: str, with_biophysics: bool) -> str:
    suffix = f".{split}"
    if task_mode == "protein_only":
        return f"protein_only_biophysics{suffix}" if with_biophysics else f"protein_only{suffix}"
    if task_mode == "genome_aware_denovo":
        return f"genome_aware_denovo_biophysics{suffix}" if with_biophysics else f"genome_aware_denovo{suffix}"
    raise ValueError(f"Unsupported task mode for clean quartet: {task_mode}")


def main() -> int:
    args = parse_args()
    input_root = Path(args.input_root).resolve()
    cnn_baseline_root = Path(args.cnn_baseline_root).resolve()
    output_path = Path(args.output).resolve() if args.output else input_root / "plm_quartet_summary.tsv"

    rows = load_rows(input_root)
    cnn_rows = load_rows(cnn_baseline_root) if cnn_baseline_root.exists() else []
    cnn_index = {
        (str(row.get("split_scheme", "")), bool(row.get("with_biophysics", False)), str(row.get("task_mode", ""))): row
        for row in cnn_rows
        if str(row.get("sequence_backbone", "")) == "cnn"
        and str(row.get("context_control", "none")) == "none"
        and float(maybe_float(row.get("host_corruption_fraction")) or 0.0) == 0.0
        and str(row.get("task_mode", "")) in {"protein_only", "genome_aware_denovo"}
    }
    quartet_rows = [
        row
        for row in rows
        if str(row.get("sequence_backbone", "")) == "precomputed_plm"
        and str(row.get("task_mode", "")) in {"protein_only", "genome_aware_denovo"}
        and str(row.get("context_control", "none")) == "none"
        and float(maybe_float(row.get("host_corruption_fraction")) or 0.0) == 0.0
        and str(row.get("run_name", ""))
        == canonical_run_name(
            str(row.get("split_scheme", "")),
            str(row.get("task_mode", "")),
            bool(row.get("with_biophysics", False)),
        )
    ]

    split_index: dict[tuple[str, bool, str], dict[str, Any]] = {}
    for row in quartet_rows:
        split = str(row.get("split_scheme", ""))
        with_bio = bool(row.get("with_biophysics", False))
        task_mode = str(row.get("task_mode", ""))
        split_index[(split, with_bio, task_mode)] = row

    out_rows: list[dict[str, Any]] = []
    for split in sorted({str(row.get("split_scheme", "")) for row in quartet_rows}):
        for with_bio in (False, True):
            protein = split_index.get((split, with_bio, "protein_only"))
            context = split_index.get((split, with_bio, "genome_aware_denovo"))
            if protein:
                cnn_match = cnn_index.get((split, with_bio, "protein_only"))
                protein_ap = maybe_float(protein.get("test_macro_average_precision"))
                protein_f1 = maybe_float(protein.get("test_macro_f1"))
                cnn_ap = maybe_float(cnn_match.get("test_macro_average_precision")) if cnn_match else None
                cnn_f1 = maybe_float(cnn_match.get("test_macro_f1")) if cnn_match else None
                out_rows.append(
                    {
                        "split_scheme": split,
                        "task_mode": "protein_only",
                        "with_biophysics": with_bio,
                        "run_name": protein.get("run_name", ""),
                        "run_dir": protein.get("run_dir", ""),
                        "test_macro_average_precision": protein.get("test_macro_average_precision"),
                        "test_macro_f1": protein.get("test_macro_f1"),
                        "delta_vs_protein_only_macro_average_precision": 0.0,
                        "delta_vs_protein_only_macro_f1": 0.0,
                        "delta_vs_no_bio_macro_average_precision": 0.0,
                        "delta_vs_no_bio_macro_f1": 0.0,
                        "cnn_reference_run_name": cnn_match.get("run_name", "") if cnn_match else "",
                        "delta_vs_cnn_same_config_macro_average_precision": None if cnn_ap is None or protein_ap is None else protein_ap - cnn_ap,
                        "delta_vs_cnn_same_config_macro_f1": None if cnn_f1 is None or protein_f1 is None else protein_f1 - cnn_f1,
                    }
                )
            if context:
                protein_ap = maybe_float(protein.get("test_macro_average_precision")) if protein else None
                context_ap = maybe_float(context.get("test_macro_average_precision"))
                protein_f1 = maybe_float(protein.get("test_macro_f1")) if protein else None
                context_f1 = maybe_float(context.get("test_macro_f1"))
                cnn_match = cnn_index.get((split, with_bio, "genome_aware_denovo"))
                cnn_ap = maybe_float(cnn_match.get("test_macro_average_precision")) if cnn_match else None
                cnn_f1 = maybe_float(cnn_match.get("test_macro_f1")) if cnn_match else None
                no_bio_context = split_index.get((split, False, "genome_aware_denovo"))
                no_bio_context_ap = maybe_float(no_bio_context.get("test_macro_average_precision")) if no_bio_context else None
                no_bio_context_f1 = maybe_float(no_bio_context.get("test_macro_f1")) if no_bio_context else None
                out_rows.append(
                    {
                        "split_scheme": split,
                        "task_mode": "genome_aware_denovo",
                        "with_biophysics": with_bio,
                        "run_name": context.get("run_name", ""),
                        "run_dir": context.get("run_dir", ""),
                        "test_macro_average_precision": context.get("test_macro_average_precision"),
                        "test_macro_f1": context.get("test_macro_f1"),
                        "delta_vs_protein_only_macro_average_precision": None if protein_ap is None or context_ap is None else context_ap - protein_ap,
                        "delta_vs_protein_only_macro_f1": None if protein_f1 is None or context_f1 is None else context_f1 - protein_f1,
                        "delta_vs_no_bio_macro_average_precision": (
                            None if not with_bio or no_bio_context_ap is None or context_ap is None else context_ap - no_bio_context_ap
                        ),
                        "delta_vs_no_bio_macro_f1": None if not with_bio or no_bio_context_f1 is None or context_f1 is None else context_f1 - no_bio_context_f1,
                        "cnn_reference_run_name": cnn_match.get("run_name", "") if cnn_match else "",
                        "delta_vs_cnn_same_config_macro_average_precision": None if cnn_ap is None or context_ap is None else context_ap - cnn_ap,
                        "delta_vs_cnn_same_config_macro_f1": None if cnn_f1 is None or context_f1 is None else context_f1 - cnn_f1,
                    }
                )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(out_rows[0].keys()) if out_rows else [
        "split_scheme",
        "task_mode",
        "with_biophysics",
        "run_name",
        "run_dir",
        "test_macro_average_precision",
        "test_macro_f1",
        "delta_vs_protein_only_macro_average_precision",
        "delta_vs_protein_only_macro_f1",
        "delta_vs_no_bio_macro_average_precision",
        "delta_vs_no_bio_macro_f1",
        "cnn_reference_run_name",
        "delta_vs_cnn_same_config_macro_average_precision",
        "delta_vs_cnn_same_config_macro_f1",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(out_rows)

    print(json.dumps({"input_root": str(input_root), "row_count": len(out_rows), "output": str(output_path)}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
