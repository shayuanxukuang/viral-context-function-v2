from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect task-mode benchmark results into a compact suite summary.")
    parser.add_argument("--input-root", required=True, help="Suite output directory containing per-run subdirectories")
    parser.add_argument("--output-json", default="", help="Optional JSON summary path. Defaults to <input-root>/suite_summary.json")
    parser.add_argument("--output-tsv", default="", help="Optional TSV summary path. Defaults to <input-root>/suite_summary.tsv")
    return parser.parse_args()


def maybe_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def main() -> int:
    args = parse_args()
    input_root = Path(args.input_root).resolve()
    if not input_root.exists():
        raise FileNotFoundError(f"Suite directory was not found: {input_root}")

    output_json = Path(args.output_json).resolve() if args.output_json else input_root / "suite_summary.json"
    output_tsv = Path(args.output_tsv).resolve() if args.output_tsv else input_root / "suite_summary.tsv"

    rows: list[dict[str, object]] = []
    for metrics_path in sorted(input_root.glob("*/metrics_summary.json")):
        run_dir = metrics_path.parent
        manifest_path = run_dir / "run_manifest.json"
        if not manifest_path.exists():
            continue

        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        validation = metrics.get("validation", {})
        test = metrics.get("test", {})
        split_strategy = manifest.get("split_strategy", {})

        rows.append(
            {
                "run_name": run_dir.name,
                "task_mode": manifest.get("task_mode", ""),
                "split_scheme": split_strategy.get("scheme", ""),
                "sequence_backbone": manifest.get("sequence_backbone", ""),
                "with_biophysics": bool(manifest.get("biophysics_fields")),
                "selected_context_blocks": ",".join(manifest.get("selected_context_blocks", [])),
                "context_control": manifest.get("context_control", "none"),
                "host_corruption_fraction": maybe_float(manifest.get("host_corruption_fraction")),
                "host_corrupted_count": manifest.get("host_corrupted_count", 0),
                "best_epoch": metrics.get("best_epoch"),
                "validation_macro_average_precision": maybe_float(validation.get("macro_average_precision")),
                "validation_micro_average_precision": maybe_float(validation.get("micro_average_precision")),
                "validation_macro_f1": maybe_float(validation.get("macro_f1")),
                "validation_micro_f1": maybe_float(validation.get("micro_f1")),
                "test_macro_average_precision": maybe_float(test.get("macro_average_precision")),
                "test_micro_average_precision": maybe_float(test.get("micro_average_precision")),
                "test_macro_f1": maybe_float(test.get("macro_f1")),
                "test_micro_f1": maybe_float(test.get("micro_f1")),
                "run_dir": str(run_dir),
                "metrics_path": str(metrics_path),
            }
        )

    report = {
        "created_at": timestamp(),
        "input_root": str(input_root),
        "run_count": len(rows),
        "rows": rows,
    }
    output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    fieldnames = list(rows[0].keys()) if rows else [
        "run_name",
        "task_mode",
        "split_scheme",
        "sequence_backbone",
        "with_biophysics",
        "selected_context_blocks",
        "context_control",
        "host_corruption_fraction",
        "host_corrupted_count",
        "best_epoch",
        "validation_macro_average_precision",
        "validation_micro_average_precision",
        "validation_macro_f1",
        "validation_micro_f1",
        "test_macro_average_precision",
        "test_micro_average_precision",
        "test_macro_f1",
        "test_micro_f1",
        "run_dir",
        "metrics_path",
    ]
    with output_tsv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps({"created_at": report["created_at"], "input_root": report["input_root"], "run_count": report["run_count"]}, indent=2, ensure_ascii=False))
    print(f"Wrote suite JSON summary to {output_json}")
    print(f"Wrote suite TSV summary to {output_tsv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
