from __future__ import annotations

import argparse
import csv
import json
import re
import tarfile
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize block ablations, controls, and host corruption from a context-study bundle.")
    parser.add_argument("--input", required=True, help="Study root directory or *.essential.tar.gz bundle")
    parser.add_argument(
        "--output-dir",
        default="",
        help="Output directory. Defaults to <input>.source_decomposition",
    )
    return parser.parse_args()


def maybe_float(value: Any) -> float | None:
    try:
        if value in {"", None}:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def load_suite_rows(path: Path) -> tuple[str, list[dict[str, Any]]]:
    if path.is_dir():
        summary_path = path / "suite_summary.json"
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        return str(summary_path), [row for row in payload.get("rows", []) if isinstance(row, dict)]

    with tarfile.open(path, "r:gz") as archive:
        suite_summaries = [
            name
            for name in archive.getnames()
            if name.endswith("/suite_summary.json") and name.startswith("runs/")
        ]
        if not suite_summaries:
            raise FileNotFoundError(f"No suite_summary.json found in archive: {path}")
        suite_summary_name = sorted(suite_summaries)[0]
        payload = json.load(archive.extractfile(suite_summary_name))
        return f"{path}::{suite_summary_name}", [row for row in payload.get("rows", []) if isinstance(row, dict)]


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else [
        "split_scheme",
        "comparison_type",
        "reference_run",
        "variant_run",
        "reference_macro_average_precision",
        "variant_macro_average_precision",
        "delta_macro_average_precision",
        "reference_macro_f1",
        "variant_macro_f1",
        "delta_macro_f1",
        "notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def comparison_row(
    split_scheme: str,
    comparison_type: str,
    reference_row: dict[str, Any] | None,
    variant_row: dict[str, Any] | None,
    notes: str,
) -> dict[str, Any]:
    reference_ap = maybe_float(reference_row.get("test_macro_average_precision")) if reference_row else None
    variant_ap = maybe_float(variant_row.get("test_macro_average_precision")) if variant_row else None
    reference_f1 = maybe_float(reference_row.get("test_macro_f1")) if reference_row else None
    variant_f1 = maybe_float(variant_row.get("test_macro_f1")) if variant_row else None
    return {
        "split_scheme": split_scheme,
        "comparison_type": comparison_type,
        "reference_run": reference_row.get("run_name", "") if reference_row else "",
        "variant_run": variant_row.get("run_name", "") if variant_row else "",
        "reference_macro_average_precision": reference_ap,
        "variant_macro_average_precision": variant_ap,
        "delta_macro_average_precision": None if reference_ap is None or variant_ap is None else variant_ap - reference_ap,
        "reference_macro_f1": reference_f1,
        "variant_macro_f1": variant_f1,
        "delta_macro_f1": None if reference_f1 is None or variant_f1 is None else variant_f1 - reference_f1,
        "notes": notes,
    }


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else input_path.with_name(f"{input_path.stem}.source_decomposition")
    summary_source, rows = load_suite_rows(input_path)

    row_index = {str(row.get("run_name", "")): row for row in rows}
    out_rows: list[dict[str, Any]] = []

    split_values = sorted({str(row.get("split_scheme", "")) for row in rows if str(row.get("split_scheme", ""))})
    if not split_values:
        split_values = ["family_holdout", "host_holdout"]

    for split in split_values:
        protein = row_index.get(f"protein_only.{split}")
        protein_bio = row_index.get(f"protein_only_biophysics.{split}")
        genome = row_index.get(f"genome_aware_denovo.{split}")
        genome_bio = row_index.get(f"genome_aware_denovo_biophysics.{split}") or genome

        out_rows.append(comparison_row(split, "context_vs_protein_only", protein, genome, "Main clean context delta"))
        out_rows.append(comparison_row(split, "protein_only_plus_bio", protein, protein_bio, "Cheap biophysics delta on sequence-only backbone"))
        out_rows.append(comparison_row(split, "genome_aware_denovo_plus_bio", genome, genome_bio, "Cheap biophysics delta on context-aware backbone"))

        for comparison_type, run_name, note in [
            ("minus_local", f"genome_aware_denovo_biophysics_minus_local.{split}", "Remove local neighborhood block"),
            ("minus_genome_org", f"genome_aware_denovo_biophysics_minus_genome_org.{split}", "Remove genome organization block"),
            ("minus_host", f"genome_aware_denovo_biophysics_minus_host.{split}", "Remove host metadata block"),
            ("control_local_shuffle", f"genome_aware_denovo_biophysics_control_local_shuffle.{split}", "Shuffle local neighborhood order"),
            ("control_position_shuffle", f"genome_aware_denovo_biophysics_control_position_shuffle.{split}", "Shuffle relative genome position"),
            ("control_host_shuffle", f"genome_aware_denovo_biophysics_control_host_shuffle.{split}", "Shuffle host metadata within family"),
        ]:
            out_rows.append(comparison_row(split, comparison_type, genome_bio, row_index.get(run_name), note))

        host_corruption_pattern = re.compile(
            rf"^genome_aware_denovo_biophysics_host_corrupt_(\d+)\.{re.escape(split)}$"
        )
        host_corruption_runs = []
        for run_name in sorted(row_index):
            match = host_corruption_pattern.match(run_name)
            if match:
                host_corruption_runs.append((int(match.group(1)), run_name))
        for fraction, run_name in sorted(host_corruption_runs):
            out_rows.append(
                comparison_row(
                    split,
                    f"host_corrupt_{fraction}",
                    genome_bio,
                    row_index.get(run_name),
                    f"Host corruption sensitivity at {fraction}%",
                )
            )

        add_back_runs = [
            ("addback_local_only", f"genome_aware_denovo_addback_local_only.{split}", "Add back only local neighborhood"),
            ("addback_genome_only", f"genome_aware_denovo_addback_genome_only.{split}", "Add back only genome organization"),
            ("addback_host_only", f"genome_aware_denovo_addback_host_only.{split}", "Add back only host metadata"),
            ("addback_local_genome", f"genome_aware_denovo_addback_local_genome.{split}", "Add back local neighborhood + genome organization"),
            ("addback_local_host", f"genome_aware_denovo_addback_local_host.{split}", "Add back local neighborhood + host metadata"),
            ("addback_genome_host", f"genome_aware_denovo_addback_genome_host.{split}", "Add back genome organization + host metadata"),
        ]
        for comparison_type, run_name, note in add_back_runs:
            out_rows.append(comparison_row(split, comparison_type, protein, row_index.get(run_name), note))

    write_tsv(output_dir / "source_decomposition_summary.tsv", out_rows)
    report = {
        "input": str(input_path),
        "summary_source": summary_source,
        "row_count": len(out_rows),
        "available_run_names": sorted(row_index),
    }
    (output_dir / "source_decomposition_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "row_count": len(out_rows)}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
