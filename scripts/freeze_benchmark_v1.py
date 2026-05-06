from __future__ import annotations

import argparse
import csv
import json
import tarfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from task_mode_config import TASK_MODE_ORDER, resolve_context_blocks


SMOKE_MARKERS = ("smoke", "dryrun", "_fast", "cuda_fix_dryrun")
SUITE_PRIORITY = {
    "clean_benchmark": 0,
    "clean_study": 1,
    "clean_misc": 2,
    "legacy_benchmark": 3,
    "smoke_excluded": 9,
}


@dataclass
class RunRecord:
    row: dict[str, Any]


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze a single benchmark registry across clean benchmark runs, "
            "clean study runs, and legacy historical baselines."
        )
    )
    parser.add_argument("--runs-root", default="runs", help="Directory containing run folders and packaged archives")
    parser.add_argument(
        "--output-tsv",
        default="runs/frozen_benchmark_v1.tsv",
        help="Frozen benchmark registry TSV output",
    )
    parser.add_argument(
        "--claims-ledger",
        default="runs/claims_ledger.md",
        help="Markdown ledger explaining canonical vs excluded numbers",
    )
    parser.add_argument(
        "--paper-numbers",
        default="runs/paper_numbers.json",
        help="JSON payload of paper-facing numbers with run traceability",
    )
    return parser.parse_args()


def maybe_float(value: Any) -> float | None:
    try:
        if value in {"", None}:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_created_at(value: str) -> float:
    if not value:
        return 0.0
    raw = value.strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(raw).timestamp()
    except ValueError:
        return 0.0


def is_smoke_name(name: str) -> bool:
    lowered = name.lower()
    return any(marker in lowered for marker in SMOKE_MARKERS)


def normalize_task_mode(manifest: dict[str, Any]) -> str:
    task_mode = str(manifest.get("task_mode", "") or "").strip()
    if task_mode:
        return task_mode
    context_table = str(manifest.get("context_table_path", "") or "").strip()
    if context_table:
        return "legacy_annotation_plus_context"
    return "legacy_annotation_baseline"


def normalize_sequence_backbone(manifest: dict[str, Any]) -> str:
    value = str(manifest.get("sequence_backbone", "") or "").strip()
    if value:
        return value
    return "cnn"


def normalize_split_scheme(manifest: dict[str, Any]) -> str:
    split_strategy = manifest.get("split_strategy", {}) or {}
    scheme = str(split_strategy.get("scheme", "") or "").strip()
    if scheme:
        return scheme
    config = manifest.get("config", {}) or {}
    return str(config.get("split_scheme", "") or "").strip()


def normalize_seed(manifest: dict[str, Any]) -> int | None:
    top_level = manifest.get("seed")
    if top_level not in {None, ""}:
        try:
            return int(top_level)
        except (TypeError, ValueError):
            pass
    config = manifest.get("config", {}) or {}
    value = config.get("seed")
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def normalize_commit(manifest: dict[str, Any]) -> str:
    for key in ("git_commit", "commit"):
        value = str(manifest.get(key, "") or "").strip()
        if value:
            return value
    return "unrecorded"


def normalize_selected_blocks(manifest: dict[str, Any], task_mode: str) -> list[str]:
    selected = manifest.get("selected_context_blocks")
    if isinstance(selected, list):
        values = [str(item).strip() for item in selected if str(item).strip()]
        if values:
            return values
    config = manifest.get("config", {}) or {}
    requested = config.get("context_blocks")
    if task_mode in TASK_MODE_ORDER:
        return list(resolve_context_blocks(task_mode, requested))
    if str(manifest.get("context_table_path", "") or "").strip():
        return ["legacy_context_table"]
    return ["legacy_annotation_metadata"]


def normalize_context_control(manifest: dict[str, Any]) -> str:
    value = str(manifest.get("context_control", "") or "").strip()
    if value:
        return value
    return "none"


def normalize_host_corruption(manifest: dict[str, Any]) -> float:
    value = manifest.get("host_corruption_fraction")
    if value in {None, ""}:
        config = manifest.get("config", {}) or {}
        value = config.get("host_corruption_fraction", 0.0)
    return float(maybe_float(value) or 0.0)


def normalize_with_biophysics(manifest: dict[str, Any]) -> bool:
    fields = manifest.get("biophysics_fields")
    if isinstance(fields, list):
        return bool(fields)
    config = manifest.get("config", {}) or {}
    return bool(config.get("with_biophysics", False))


def classify_suite(suite_name: str, run_name: str, task_mode: str) -> str:
    if is_smoke_name(suite_name) or is_smoke_name(run_name):
        return "smoke_excluded"
    suite_lower = suite_name.lower()
    if task_mode in TASK_MODE_ORDER:
        if suite_lower.startswith("task_mode_suite"):
            return "clean_benchmark"
        if suite_lower.startswith("context_study"):
            return "clean_study"
        return "clean_misc"
    return "legacy_benchmark"


def claim_scope(
    suite_class: str,
    task_mode: str,
    with_biophysics: bool,
    selected_blocks: list[str],
    context_control: str,
    host_corruption_fraction: float,
) -> str:
    if suite_class == "legacy_benchmark":
        return "historical_excluded"
    if suite_class == "smoke_excluded":
        return "smoke_excluded"
    default_blocks = list(resolve_context_blocks(task_mode, None)) if task_mode in TASK_MODE_ORDER else []
    if task_mode in TASK_MODE_ORDER and context_control == "none" and host_corruption_fraction == 0.0:
        if with_biophysics:
            return "study_auxiliary"
        if selected_blocks == default_blocks:
            return "paper_main"
    if context_control != "none" or host_corruption_fraction > 0.0:
        return "sensitivity_control"
    if with_biophysics:
        return "study_auxiliary"
    return "other_clean"


def config_signature(row: dict[str, Any]) -> str:
    key = {
        "split_scheme": row["split_scheme"],
        "task_mode": row["task_mode"],
        "sequence_backbone": row["sequence_backbone"],
        "with_biophysics": bool(row["with_biophysics"]),
        "feature_blocks": row["feature_blocks"],
        "context_control": row["context_control"],
        "host_corruption_fraction": round(float(row["host_corruption_fraction"] or 0.0), 4),
    }
    return json.dumps(key, sort_keys=True, ensure_ascii=False)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def to_run_record(
    source_kind: str,
    source_path: str,
    suite_name: str,
    run_name: str,
    manifest: dict[str, Any],
    metrics: dict[str, Any],
    has_predictions: bool,
    has_checkpoint: bool,
) -> RunRecord:
    task_mode = normalize_task_mode(manifest)
    sequence_backbone = normalize_sequence_backbone(manifest)
    split_scheme = normalize_split_scheme(manifest)
    selected_blocks = normalize_selected_blocks(manifest, task_mode)
    feature_blocks = ",".join(selected_blocks)
    with_biophysics = normalize_with_biophysics(manifest)
    context_control = normalize_context_control(manifest)
    host_corruption_fraction = normalize_host_corruption(manifest)
    suite_class = classify_suite(suite_name, run_name, task_mode)
    validation = metrics.get("validation", {}) or {}
    test = metrics.get("test", {}) or {}
    split_strategy = manifest.get("split_strategy", {}) or {}
    row = {
        "run_id": f"{suite_name}::{run_name}" if source_kind == "directory" else f"{Path(source_path).name}::{suite_name}::{run_name}",
        "source_kind": source_kind,
        "source_path": source_path,
        "suite_name": suite_name,
        "run_name": run_name,
        "created_at": str(manifest.get("created_at", "") or ""),
        "task_mode": task_mode,
        "split_scheme": split_scheme,
        "sequence_backbone": sequence_backbone,
        "with_biophysics": with_biophysics,
        "feature_blocks": feature_blocks,
        "context_control": context_control,
        "host_corruption_fraction": host_corruption_fraction,
        "host_corrupted_count": int(manifest.get("host_corrupted_count", 0) or 0),
        "seed": normalize_seed(manifest),
        "commit": normalize_commit(manifest),
        "has_predictions": has_predictions,
        "has_checkpoint": has_checkpoint,
        "suite_class": suite_class,
        "claim_scope": claim_scope(
            suite_class=suite_class,
            task_mode=task_mode,
            with_biophysics=with_biophysics,
            selected_blocks=selected_blocks,
            context_control=context_control,
            host_corruption_fraction=host_corruption_fraction,
        ),
        "best_epoch": metrics.get("best_epoch"),
        "validation_macro_average_precision": maybe_float(validation.get("macro_average_precision")),
        "validation_micro_average_precision": maybe_float(validation.get("micro_average_precision")),
        "validation_macro_f1": maybe_float(validation.get("macro_f1")),
        "validation_micro_f1": maybe_float(validation.get("micro_f1")),
        "test_macro_average_precision": maybe_float(test.get("macro_average_precision")),
        "test_micro_average_precision": maybe_float(test.get("micro_average_precision")),
        "test_macro_f1": maybe_float(test.get("macro_f1")),
        "test_micro_f1": maybe_float(test.get("micro_f1")),
        "split_counts_json": json.dumps(split_strategy.get("counts", {}), ensure_ascii=False, sort_keys=True),
        "context_table_path": str(manifest.get("context_table_path", "") or ""),
        "artifact_locator": source_path,
    }
    row["config_signature"] = config_signature(row)
    return RunRecord(row=row)


def scan_directory_runs(runs_root: Path) -> list[RunRecord]:
    records: list[RunRecord] = []
    for manifest_path in sorted(runs_root.glob("**/run_manifest.json")):
        run_dir = manifest_path.parent
        metrics_path = run_dir / "metrics_summary.json"
        if not metrics_path.exists():
            continue
        relative_parts = run_dir.relative_to(runs_root).parts
        if len(relative_parts) == 1:
            suite_name = relative_parts[0]
            run_name = relative_parts[0]
        else:
            suite_name = relative_parts[0]
            run_name = relative_parts[-1]
        manifest = load_json(manifest_path)
        metrics = load_json(metrics_path)
        records.append(
            to_run_record(
                source_kind="directory",
                source_path=str(run_dir.resolve()),
                suite_name=suite_name,
                run_name=run_name,
                manifest=manifest,
                metrics=metrics,
                has_predictions=(run_dir / "test_predictions.tsv.gz").exists(),
                has_checkpoint=(run_dir / "best_model.pt").exists(),
            )
        )
    return records


def scan_archive_runs(runs_root: Path) -> list[RunRecord]:
    records: list[RunRecord] = []
    for archive_path in sorted(runs_root.glob("*.tar.gz")):
        with tarfile.open(archive_path, "r:gz") as archive:
            names = {member.name for member in archive.getmembers() if member.isfile()}
            for manifest_name in sorted(name for name in names if name.endswith("/run_manifest.json")):
                metrics_name = manifest_name.rsplit("/", 1)[0] + "/metrics_summary.json"
                if metrics_name not in names:
                    continue
                parts = Path(manifest_name).parts
                if len(parts) < 4 or parts[0] != "runs":
                    continue
                suite_name = parts[1]
                run_name = parts[2]
                manifest = json.load(archive.extractfile(manifest_name))
                metrics = json.load(archive.extractfile(metrics_name))
                records.append(
                    to_run_record(
                        source_kind="archive",
                        source_path=str(archive_path.resolve()),
                        suite_name=suite_name,
                        run_name=run_name,
                        manifest=manifest,
                        metrics=metrics,
                        has_predictions=(metrics_name.rsplit("/", 1)[0] + "/test_predictions.tsv.gz") in names,
                        has_checkpoint=(metrics_name.rsplit("/", 1)[0] + "/best_model.pt") in names,
                    )
                )
    return records


def choose_canonical(records: list[RunRecord]) -> tuple[dict[str, str], dict[str, list[str]]]:
    groups: dict[str, list[RunRecord]] = defaultdict(list)
    for record in records:
        row = record.row
        if row["suite_class"] in {"clean_benchmark", "clean_study", "clean_misc"}:
            groups[str(row["config_signature"])].append(record)

    canonical_by_run_id: dict[str, str] = {}
    duplicates: dict[str, list[str]] = {}
    for signature, group in groups.items():
        ordered = sorted(
            group,
            key=lambda record: (
                SUITE_PRIORITY[str(record.row["suite_class"])],
                0 if bool(record.row["has_predictions"]) else 1,
                0 if record.row["source_kind"] == "directory" else 1,
                -parse_created_at(str(record.row["created_at"])),
            ),
        )
        canonical_id = str(ordered[0].row["run_id"])
        duplicates[canonical_id] = [str(item.row["run_id"]) for item in ordered]
        for item in ordered:
            canonical_by_run_id[str(item.row["run_id"])] = canonical_id
    return canonical_by_run_id, duplicates


def enrich_status(records: list[RunRecord]) -> list[dict[str, Any]]:
    canonical_by_run_id, duplicates = choose_canonical(records)
    rows: list[dict[str, Any]] = []
    for record in records:
        row = dict(record.row)
        run_id = str(row["run_id"])
        canonical_id = canonical_by_run_id.get(run_id, "")
        row["canonical_run_id"] = canonical_id
        if row["suite_class"] == "smoke_excluded":
            row["freeze_status"] = "excluded_smoke"
            row["freeze_reason"] = "Smoke or dry-run artifact"
        elif row["suite_class"] == "legacy_benchmark":
            row["freeze_status"] = "excluded_legacy"
            row["freeze_reason"] = "Historical baseline uses annotation-derived legacy feature stack"
        elif canonical_id and canonical_id != run_id:
            row["freeze_status"] = "superseded_duplicate"
            row["freeze_reason"] = f"Superseded by canonical run {canonical_id}"
        else:
            row["freeze_status"] = "canonical"
            if row["claim_scope"] == "paper_main":
                row["freeze_reason"] = "Canonical clean benchmark run"
            elif row["claim_scope"] == "study_auxiliary":
                row["freeze_reason"] = "Canonical study auxiliary run"
            elif row["claim_scope"] == "sensitivity_control":
                row["freeze_reason"] = "Canonical sensitivity-control run"
            else:
                row["freeze_reason"] = "Canonical clean run"
        duplicate_group = duplicates.get(canonical_id or run_id, [run_id])
        row["duplicate_group_size"] = len(duplicate_group)
        row["duplicate_group_json"] = json.dumps(duplicate_group, ensure_ascii=False)
        rows.append(row)
    rows.sort(
        key=lambda row: (
            {"canonical": 0, "superseded_duplicate": 1, "excluded_legacy": 2, "excluded_smoke": 3}.get(str(row["freeze_status"]), 9),
            SUITE_PRIORITY.get(str(row["suite_class"]), 9),
            str(row["split_scheme"]),
            str(row["task_mode"]),
            str(row["run_name"]),
        )
    )
    return rows


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_id",
        "freeze_status",
        "freeze_reason",
        "claim_scope",
        "suite_class",
        "suite_name",
        "run_name",
        "source_kind",
        "source_path",
        "created_at",
        "task_mode",
        "split_scheme",
        "sequence_backbone",
        "with_biophysics",
        "feature_blocks",
        "context_control",
        "host_corruption_fraction",
        "host_corrupted_count",
        "seed",
        "commit",
        "has_predictions",
        "has_checkpoint",
        "best_epoch",
        "validation_macro_average_precision",
        "validation_micro_average_precision",
        "validation_macro_f1",
        "validation_micro_f1",
        "test_macro_average_precision",
        "test_micro_average_precision",
        "test_macro_f1",
        "test_micro_f1",
        "canonical_run_id",
        "duplicate_group_size",
        "duplicate_group_json",
        "config_signature",
        "split_counts_json",
        "context_table_path",
        "artifact_locator",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def build_main_numbers(
    rows: list[dict[str, Any]],
    *,
    predicate,
) -> dict[str, dict[str, dict[str, Any]]]:
    index: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        if row["freeze_status"] != "canonical" or not predicate(row):
            continue
        split = str(row["split_scheme"])
        task_mode = str(row["task_mode"])
        index[split][task_mode] = {
            "run_id": row["run_id"],
            "macro_average_precision": row["test_macro_average_precision"],
            "macro_f1": row["test_macro_f1"],
            "with_biophysics": bool(row["with_biophysics"]),
            "feature_blocks": row["feature_blocks"],
        }
    return {split: value for split, value in index.items()}


def is_default_blocks(task_mode: str, feature_blocks: str) -> bool:
    if task_mode not in TASK_MODE_ORDER:
        return False
    expected = ",".join(resolve_context_blocks(task_mode, None))
    return feature_blocks == expected


def build_delta_section(index: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    deltas: dict[str, Any] = {}
    for split, split_rows in index.items():
        baseline = split_rows.get("protein_only")
        context = split_rows.get("genome_aware_denovo")
        if not baseline or not context:
            continue
        baseline_ap = maybe_float(baseline.get("macro_average_precision"))
        context_ap = maybe_float(context.get("macro_average_precision"))
        baseline_f1 = maybe_float(baseline.get("macro_f1"))
        context_f1 = maybe_float(context.get("macro_f1"))
        deltas[split] = {
            "protein_only_run_id": baseline["run_id"],
            "genome_aware_denovo_run_id": context["run_id"],
            "delta_macro_average_precision": None if baseline_ap is None or context_ap is None else context_ap - baseline_ap,
            "delta_macro_f1": None if baseline_f1 is None or context_f1 is None else context_f1 - baseline_f1,
        }
    return deltas


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    return "\n".join([header, divider, *body]) if body else "_none_"


def write_claims_ledger(path: Path, rows: list[dict[str, Any]], paper_numbers: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    canonical_rows = [row for row in rows if row["freeze_status"] == "canonical"]
    main_rows = [row for row in canonical_rows if row["claim_scope"] == "paper_main"]
    study_rows = [row for row in canonical_rows if row["claim_scope"] in {"study_auxiliary", "sensitivity_control"}]
    duplicate_rows = [row for row in rows if row["freeze_status"] == "superseded_duplicate"]
    legacy_rows = [row for row in rows if row["freeze_status"] == "excluded_legacy"]

    suite_counts = Counter(str(row["suite_class"]) for row in rows)
    lines = [
        "# claims_ledger",
        "",
        f"- generated_at: `{paper_numbers['generated_at']}`",
        f"- frozen_registry: `{paper_numbers['registry_path']}`",
        "",
        "## Freeze Rule",
        "",
        "1. 主 benchmark 只认 clean task-mode benchmark suite 的 canonical runs。",
        "2. clean study reruns 作为 study auxiliary / sensitivity controls 保留，但不覆盖 benchmark 主数字。",
        "3. legacy benchmark 保留在 registry 里，只作历史参考，不进入主文 claim。",
        "",
        "## Inventory",
        "",
        f"- total_rows: `{len(rows)}`",
        f"- canonical_rows: `{len(canonical_rows)}`",
        f"- superseded_duplicates: `{len(duplicate_rows)}`",
        f"- excluded_legacy: `{len(legacy_rows)}`",
        f"- suite_class_counts: `{json.dumps(dict(suite_counts), ensure_ascii=False, sort_keys=True)}`",
        "",
        "## Canonical Main Benchmark",
        "",
        markdown_table(
            [
                {
                    "run_id": row["run_id"],
                    "split": row["split_scheme"],
                    "task_mode": row["task_mode"],
                    "macro_AP": row["test_macro_average_precision"],
                    "macro_F1": row["test_macro_f1"],
                }
                for row in main_rows
            ],
            ["run_id", "split", "task_mode", "macro_AP", "macro_F1"],
        ),
        "",
        "## Canonical Study Auxiliary",
        "",
        markdown_table(
            [
                {
                    "run_id": row["run_id"],
                    "split": row["split_scheme"],
                    "task_mode": row["task_mode"],
                    "with_bio": row["with_biophysics"],
                    "feature_blocks": row["feature_blocks"],
                    "scope": row["claim_scope"],
                    "macro_AP": row["test_macro_average_precision"],
                    "macro_F1": row["test_macro_f1"],
                }
                for row in study_rows
            ],
            ["run_id", "split", "task_mode", "with_bio", "feature_blocks", "scope", "macro_AP", "macro_F1"],
        ),
        "",
        "## Superseded Duplicate Clean Runs",
        "",
        markdown_table(
            [
                {
                    "run_id": row["run_id"],
                    "split": row["split_scheme"],
                    "task_mode": row["task_mode"],
                    "canonical_run_id": row["canonical_run_id"],
                    "reason": row["freeze_reason"],
                    "macro_AP": row["test_macro_average_precision"],
                }
                for row in duplicate_rows
            ],
            ["run_id", "split", "task_mode", "canonical_run_id", "reason", "macro_AP"],
        ),
        "",
        "## Excluded Legacy Main Results",
        "",
        markdown_table(
            [
                {
                    "run_id": row["run_id"],
                    "split": row["split_scheme"],
                    "task_mode": row["task_mode"],
                    "macro_AP": row["test_macro_average_precision"],
                    "macro_F1": row["test_macro_f1"],
                    "reason": row["freeze_reason"],
                }
                for row in legacy_rows
            ],
            ["run_id", "split", "task_mode", "macro_AP", "macro_F1", "reason"],
        ),
        "",
        "## Paper Numbers",
        "",
        "```json",
        json.dumps(paper_numbers["paper_numbers"], indent=2, ensure_ascii=False),
        "```",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def build_paper_numbers(rows: list[dict[str, Any]], registry_path: Path) -> dict[str, Any]:
    main_index = build_main_numbers(
        rows,
        predicate=lambda row: row["claim_scope"] == "paper_main",
    )
    biophysics_quartet = build_main_numbers(
        rows,
        predicate=lambda row: (
            row["claim_scope"] == "study_auxiliary"
            and bool(row["with_biophysics"])
            and str(row["context_control"]) == "none"
            and float(row["host_corruption_fraction"] or 0.0) == 0.0
            and (
                (row["task_mode"] == "protein_only" and str(row["feature_blocks"]) == "")
                or (row["task_mode"] == "genome_aware_denovo" and is_default_blocks("genome_aware_denovo", str(row["feature_blocks"])))
            )
        ),
    )
    annotation_refinement = build_main_numbers(
        rows,
        predicate=lambda row: (
            row["claim_scope"] == "study_auxiliary"
            and row["task_mode"] == "annotation_refinement"
        ),
    )
    source_decomposition = [
        {
            "run_id": row["run_id"],
            "split_scheme": row["split_scheme"],
            "task_mode": row["task_mode"],
            "feature_blocks": row["feature_blocks"],
            "context_control": row["context_control"],
            "host_corruption_fraction": row["host_corruption_fraction"],
            "macro_average_precision": row["test_macro_average_precision"],
            "macro_f1": row["test_macro_f1"],
        }
        for row in rows
        if row["freeze_status"] == "canonical"
        and row["suite_class"] == "clean_study"
        and row["task_mode"] == "genome_aware_denovo"
        and (
            row["claim_scope"] == "sensitivity_control"
            or (
                row["claim_scope"] == "study_auxiliary"
                and (
                    str(row["feature_blocks"]) not in {"", ",".join(resolve_context_blocks("genome_aware_denovo", None))}
                    or str(row["context_control"]) != "none"
                    or float(row["host_corruption_fraction"] or 0.0) > 0.0
                )
            )
        )
    ]
    paper_numbers = {
        "benchmark_main": main_index,
        "benchmark_main_context_delta": build_delta_section(main_index),
        "biophysics_quartet": biophysics_quartet,
        "biophysics_quartet_context_delta": build_delta_section(biophysics_quartet),
        "annotation_refinement": annotation_refinement,
        "source_decomposition": source_decomposition,
    }
    return {
        "generated_at": timestamp(),
        "registry_path": str(registry_path),
        "paper_numbers": paper_numbers,
    }


def main() -> int:
    args = parse_args()
    root = repo_root()
    runs_root = (root / args.runs_root).resolve()
    output_tsv = (root / args.output_tsv).resolve()
    claims_ledger_path = (root / args.claims_ledger).resolve()
    paper_numbers_path = (root / args.paper_numbers).resolve()

    if not runs_root.exists():
        raise FileNotFoundError(f"Runs root was not found: {runs_root}")

    records = scan_directory_runs(runs_root)
    records.extend(scan_archive_runs(runs_root))
    rows = enrich_status(records)
    write_tsv(output_tsv, rows)

    paper_numbers = build_paper_numbers(rows, registry_path=output_tsv)
    paper_numbers_path.parent.mkdir(parents=True, exist_ok=True)
    paper_numbers_path.write_text(json.dumps(paper_numbers, indent=2, ensure_ascii=False), encoding="utf-8")
    write_claims_ledger(claims_ledger_path, rows, paper_numbers)

    print(
        json.dumps(
            {
                "generated_at": paper_numbers["generated_at"],
                "run_count": len(rows),
                "canonical_count": sum(1 for row in rows if row["freeze_status"] == "canonical"),
                "output_tsv": str(output_tsv),
                "claims_ledger": str(claims_ledger_path),
                "paper_numbers": str(paper_numbers_path),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
