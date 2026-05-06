from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from build_strict_splits import derive_virus_family
from calibrate_task_mode_uncertainty import build_model, dataloader_for_indices
from label_rules import LABEL_RULES, label_hits, normalize_text
from train_overnight_baseline import choose_device, compute_metrics
from train_task_modes import predict


MOBILE_ELEMENT_RE = re.compile(
    r"integrase|recombinase|transpos|resolvase|insertion sequence|homing endonuclease|"
    r"endonuclease|exonuclease|nuclease|ligase|terminase|portal|packaging",
    re.IGNORECASE,
)

FORBIDDEN_FEATURE_CHECKS = [
    ("product_name", ("cds_product", "protein_description", "product", "description")),
    ("hypothetical_flag", ("is_hypothetical", "neighbor_is_hypothetical")),
    ("uniprot_counts", ("uniprot",)),
    ("pfam_interpro_cdd_hits", ("pfam", "interpro", "cdd")),
    ("neighbor_true_labels", ("neighbor_label", "true_label")),
    ("genome_local_label_counts", ("context_train_genome_", "context_train_local_", "label_count")),
    ("protein_feature_type", ("protein_feature_type", "neighbor_feature_type")),
    ("annotation_text_embedding", ("text_embedding", "annotation_embedding", "product_embedding")),
]


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run reviewer-facing V2 QC tables without retraining.")
    parser.add_argument("--run-root", required=True, help="Completed V2 run root, e.g. runs/context_study_v2_...")
    parser.add_argument("--input", default="data/processed/training/viral_protein_training_index.tsv.gz")
    parser.add_argument("--split-manifest", default="data/processed/splits/viral_protein_strict_splits.tsv.gz")
    parser.add_argument("--freeze-dir", default="data/v2_freeze")
    parser.add_argument("--output-dir", default="", help="Defaults to <run-root>/qc_review")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--module-null-iterations", type=int, default=500)
    parser.add_argument("--label-of-interest", default="nucleocapsid")
    parser.add_argument("--candidate-context-gain-threshold", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force-predict", action="store_true")
    return parser.parse_args()


def resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def write_tsv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_tsv(path: Path) -> list[dict[str, str]]:
    with open_text(path) as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_json_list(raw: str) -> list[str]:
    try:
        value = json.loads(raw or "[]")
    except json.JSONDecodeError:
        return []
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def maybe_float(value: Any) -> float | None:
    try:
        if value in {"", None}:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def safe_divide(numerator: float, denominator: float) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


def load_strict_split_rows(path: Path) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    with open_text(path) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            accession = row.get("protein_accession", "").strip()
            if accession:
                rows[accession] = dict(row)
    return rows


def load_training_metadata(input_path: Path, strict_rows: dict[str, dict[str, str]]) -> dict[str, dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {}
    with open_text(input_path) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            accession = row.get("protein_accession", "").strip()
            if not accession:
                continue
            strict = strict_rows.get(accession, {})
            family = strict.get("virus_family", "")
            if not family:
                family, _source = derive_virus_family(row.get("virus_lineage", ""))
            text = normalize_text(row)
            hits = [LABEL_RULES[idx].name for idx in label_hits(text)]
            metadata[accession] = {
                "protein_accession": accession,
                "genome_version": row.get("genome_version", "").strip(),
                "virus_tax_id": row.get("virus_tax_id", "").strip(),
                "virus_name": row.get("virus_name", "").strip(),
                "virus_family": family,
                "host_taxid_key": strict.get("host_taxid_key", ""),
                "host_supergroup": strict.get("host_supergroup", ""),
                "sequence_sketch_key": strict.get("sequence_sketch_key", ""),
                "protein_sequence_sha256": row.get("protein_sequence_sha256", "").strip(),
                "description": row.get("protein_description", "").strip(),
                "cds_product": row.get("cds_product", "").strip(),
                "text": text,
                "labels": set(hits),
                "family_holdout_split": strict.get("family_holdout_split", ""),
                "host_holdout_split": strict.get("host_taxid_holdout_split", ""),
            }
    return metadata


def load_run_predictions(
    run_dir: Path,
    output_dir: Path,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    force: bool,
) -> dict[str, Any]:
    cache_dir = output_dir / "_prediction_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{run_dir.name}.npz"
    if cache_path.exists() and not force:
        payload = np.load(cache_path, allow_pickle=False)
        return {
            "run_dir": run_dir,
            "accessions": payload["accessions"].astype(str),
            "genome_versions": payload["genome_versions"].astype(str),
            "virus_tax_ids": payload["virus_tax_ids"].astype(str),
            "descriptions": payload["descriptions"].astype(str),
            "label_names": payload["label_names"].astype(str).tolist(),
            "y_prob": payload["y_prob"].astype(np.float32),
            "y_true": payload["y_true"].astype(np.uint8),
            "thresholds": payload["thresholds"].astype(np.float32),
            "indices": payload["indices"].astype(np.int64),
        }

    manifest = load_json(run_dir / "run_manifest.json")
    cache = torch.load(run_dir / "dataset_cache.pt", map_location="cpu", weights_only=False)
    checkpoint = torch.load(run_dir / "best_model.pt", map_location="cpu", weights_only=False)
    threshold_payload = load_json(run_dir / "best_thresholds.json")
    label_names = [str(item) for item in cache["label_names"]]
    thresholds = np.asarray([float(threshold_payload["thresholds"][label]) for label in label_names], dtype=np.float32)

    model = build_model(cache, manifest)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)

    splits = np.asarray(cache["splits"])
    test_idx = np.where(splits == 2)[0]
    dataloader_args = argparse.Namespace(
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
    )
    loader = dataloader_for_indices(cache, test_idx, dataloader_args, device)
    y_prob, y_true, indices = predict(model, loader, device, autocast_dtype=None)

    accessions = np.asarray([str(cache["protein_accessions"][int(idx)]) for idx in indices])
    genome_versions = np.asarray([str(cache["genome_versions"][int(idx)]) for idx in indices])
    virus_tax_ids = np.asarray([str(cache["virus_tax_ids"][int(idx)]) for idx in indices])
    descriptions = np.asarray([str(cache["descriptions"][int(idx)]) for idx in indices])
    np.savez_compressed(
        cache_path,
        accessions=accessions,
        genome_versions=genome_versions,
        virus_tax_ids=virus_tax_ids,
        descriptions=descriptions,
        label_names=np.asarray(label_names),
        y_prob=y_prob.astype(np.float32),
        y_true=y_true.astype(np.uint8),
        thresholds=thresholds,
        indices=indices.astype(np.int64),
    )
    return {
        "run_dir": run_dir,
        "accessions": accessions,
        "genome_versions": genome_versions,
        "virus_tax_ids": virus_tax_ids,
        "descriptions": descriptions,
        "label_names": label_names,
        "y_prob": y_prob.astype(np.float32),
        "y_true": y_true.astype(np.uint8),
        "thresholds": thresholds,
        "indices": indices.astype(np.int64),
    }


def align_predictions(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    left_index = {accession: idx for idx, accession in enumerate(left["accessions"])}
    right_index = {accession: idx for idx, accession in enumerate(right["accessions"])}
    common = [accession for accession in left["accessions"] if accession in right_index]
    left_rows = np.asarray([left_index[accession] for accession in common], dtype=np.int64)
    right_rows = np.asarray([right_index[accession] for accession in common], dtype=np.int64)
    if left["label_names"] != right["label_names"]:
        raise ValueError(f"Label names differ between {left['run_dir']} and {right['run_dir']}")
    return {
        "accessions": np.asarray(common),
        "left_prob": left["y_prob"][left_rows],
        "right_prob": right["y_prob"][right_rows],
        "y_true": left["y_true"][left_rows],
        "left_thresholds": left["thresholds"],
        "right_thresholds": right["thresholds"],
        "label_names": left["label_names"],
    }


def metrics_subset(y_true: np.ndarray, y_prob: np.ndarray, thresholds: np.ndarray, label_names: list[str]) -> dict[str, Any]:
    return compute_metrics(y_true, y_prob, thresholds, label_names)


def delta_row_from_arrays(
    y_true: np.ndarray,
    protein_prob: np.ndarray,
    context_prob: np.ndarray,
    protein_thresholds: np.ndarray,
    context_thresholds: np.ndarray,
    label_names: list[str],
) -> dict[str, float]:
    protein_metrics = metrics_subset(y_true, protein_prob, protein_thresholds, label_names)
    context_metrics = metrics_subset(y_true, context_prob, context_thresholds, label_names)
    return {
        "protein_macro_ap": float(protein_metrics["macro_average_precision"]),
        "context_macro_ap": float(context_metrics["macro_average_precision"]),
        "delta_macro_ap": float(context_metrics["macro_average_precision"] - protein_metrics["macro_average_precision"]),
        "protein_macro_f1": float(protein_metrics["macro_f1"]),
        "context_macro_f1": float(context_metrics["macro_f1"]),
        "delta_macro_f1": float(context_metrics["macro_f1"] - protein_metrics["macro_f1"]),
        "protein_micro_ap": float(protein_metrics["micro_average_precision"]),
        "context_micro_ap": float(context_metrics["micro_average_precision"]),
        "delta_micro_ap": float(context_metrics["micro_average_precision"] - protein_metrics["micro_average_precision"]),
        "protein_micro_f1": float(protein_metrics["micro_f1"]),
        "context_micro_f1": float(context_metrics["micro_f1"]),
        "delta_micro_f1": float(context_metrics["micro_f1"] - protein_metrics["micro_f1"]),
    }


def exact_transfer_qc(
    metadata: dict[str, dict[str, Any]],
    output_dir: Path,
) -> tuple[set[str], dict[str, Any]]:
    train_by_sha: dict[str, list[dict[str, Any]]] = defaultdict(list)
    test_by_sha: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in metadata.values():
        split = row.get("family_holdout_split", "")
        sha = row.get("protein_sequence_sha256", "")
        if not sha:
            continue
        if split == "train":
            train_by_sha[sha].append(row)
        elif split == "test":
            test_by_sha[sha].append(row)

    exact_accessions: set[str] = set()
    detail_rows: list[dict[str, Any]] = []
    reason_counter: Counter[str] = Counter()
    for sha, test_rows in test_by_sha.items():
        train_rows = train_by_sha.get(sha, [])
        if not train_rows:
            continue
        train_families = sorted({row["virus_family"] for row in train_rows})
        train_taxa = sorted({row["virus_tax_id"] for row in train_rows})
        train_genomes = sorted({row["genome_version"] for row in train_rows})
        train_text = " ".join(str(row.get("text", "")) for row in train_rows)
        for test_row in test_rows:
            exact_accessions.add(test_row["protein_accession"])
            cross_family = test_row["virus_family"] not in train_families
            same_taxid = test_row["virus_tax_id"] in train_taxa
            mobile_like = bool(MOBILE_ELEMENT_RE.search(f"{test_row.get('text', '')} {train_text}"))
            duplicated_like = (len(train_rows) + len(test_rows)) > 2
            reasons = []
            if cross_family:
                reasons.append("identical_sequence_assigned_to_different_families")
            if same_taxid and cross_family:
                reasons.append("same_taxid_different_family_annotation")
            if mobile_like:
                reasons.append("shared_mobile_or_module_element_like")
            if duplicated_like:
                reasons.append("duplicated_entry_like")
            if not reasons:
                reasons.append("unclassified_exact_transfer")
            for reason in reasons:
                reason_counter[reason] += 1
            detail_rows.append(
                {
                    "test_protein_accession": test_row["protein_accession"],
                    "sequence_sha256": sha,
                    "test_family": test_row["virus_family"],
                    "test_taxid": test_row["virus_tax_id"],
                    "test_genome": test_row["genome_version"],
                    "test_description": test_row["description"],
                    "train_identical_count": len(train_rows),
                    "test_identical_count": len(test_rows),
                    "train_families_json": json.dumps(train_families, ensure_ascii=False),
                    "train_taxids_json": json.dumps(train_taxa, ensure_ascii=False),
                    "train_genomes_json": json.dumps(train_genomes[:25], ensure_ascii=False),
                    "cross_family_identical": int(cross_family),
                    "same_taxid_different_family_annotation": int(same_taxid and cross_family),
                    "shared_mobile_or_module_element_like": int(mobile_like),
                    "duplicated_entry_like": int(duplicated_like),
                    "reason_tags_json": json.dumps(reasons, ensure_ascii=False),
                }
            )
    write_tsv(output_dir / "qc1_family_exact_transfer.tsv", detail_rows)
    summary = {
        "family_test_count": sum(1 for row in metadata.values() if row.get("family_holdout_split") == "test"),
        "exact_transfer_test_count": len(exact_accessions),
        "exact_transfer_rate": safe_divide(len(exact_accessions), sum(1 for row in metadata.values() if row.get("family_holdout_split") == "test")),
        "reason_counts": dict(reason_counter),
    }
    return exact_accessions, summary


def strict_zero_metrics_qc(
    protein_pred: dict[str, Any],
    context_pred: dict[str, Any],
    exact_accessions: set[str],
    output_dir: Path,
) -> dict[str, Any]:
    aligned = align_predictions(protein_pred, context_pred)
    exact_mask = np.asarray([accession in exact_accessions for accession in aligned["accessions"]], dtype=bool)
    rows = []
    for subset_name, mask in [
        ("all_family_test", np.ones(aligned["accessions"].shape[0], dtype=bool)),
        ("strict_zero_exact_transfer_test", ~exact_mask),
        ("exact_transfer_only_test", exact_mask),
    ]:
        if int(mask.sum()) == 0:
            continue
        row = {
            "subset": subset_name,
            "test_protein_count": int(mask.sum()),
            "exact_transfer_count": int(exact_mask[mask].sum()),
            **delta_row_from_arrays(
                aligned["y_true"][mask],
                aligned["left_prob"][mask],
                aligned["right_prob"][mask],
                aligned["left_thresholds"],
                aligned["right_thresholds"],
                aligned["label_names"],
            ),
        }
        rows.append(row)
    write_tsv(output_dir / "qc1_strict_zero_exact_transfer_metrics.tsv", rows)
    return {"rows": rows}


def paired_bootstrap_delta(
    aligned: dict[str, Any],
    block_values: list[str],
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    block_to_indices: dict[str, list[int]] = defaultdict(list)
    for idx, block in enumerate(block_values):
        block_to_indices[str(block) or "__MISSING__"].append(idx)
    blocks = sorted(block_to_indices)
    point = delta_row_from_arrays(
        aligned["y_true"],
        aligned["left_prob"],
        aligned["right_prob"],
        aligned["left_thresholds"],
        aligned["right_thresholds"],
        aligned["label_names"],
    )
    sampled = []
    for _ in range(iterations):
        sampled_blocks = rng.choice(blocks, size=len(blocks), replace=True)
        sampled_indices = np.asarray([idx for block in sampled_blocks for idx in block_to_indices[str(block)]], dtype=np.int64)
        if sampled_indices.size == 0:
            continue
        sampled.append(
            delta_row_from_arrays(
                aligned["y_true"][sampled_indices],
                aligned["left_prob"][sampled_indices],
                aligned["right_prob"][sampled_indices],
                aligned["left_thresholds"],
                aligned["right_thresholds"],
                aligned["label_names"],
            )
        )
    out = dict(point)
    for metric in ("delta_macro_ap", "delta_macro_f1", "delta_micro_ap", "delta_micro_f1"):
        values = np.asarray([row[metric] for row in sampled], dtype=np.float64)
        out[f"{metric}_ci_low"] = float(np.percentile(values, 2.5)) if values.size else None
        out[f"{metric}_ci_high"] = float(np.percentile(values, 97.5)) if values.size else None
    out["bootstrap_iterations"] = iterations
    out["block_count"] = len(blocks)
    return out


def ci_qc(
    run_root: Path,
    predictions: dict[str, dict[str, Any]],
    metadata: dict[str, dict[str, Any]],
    output_dir: Path,
    iterations: int,
    seed: int,
) -> list[dict[str, Any]]:
    pair_specs = [
        ("family_holdout", "protein_only.family_holdout", "genome_aware_denovo.family_holdout", "virus_family"),
        ("host_holdout", "protein_only.host_holdout", "genome_aware_denovo.host_holdout", "host_taxid_key"),
        ("family_holdout_biophysics", "protein_only_biophysics.family_holdout", "genome_aware_denovo_biophysics.family_holdout", "virus_family"),
        ("host_holdout_biophysics", "protein_only_biophysics.host_holdout", "genome_aware_denovo_biophysics.host_holdout", "host_taxid_key"),
    ]
    rows = []
    for comparison, protein_name, context_name, block_field in pair_specs:
        if protein_name not in predictions or context_name not in predictions:
            continue
        aligned = align_predictions(predictions[protein_name], predictions[context_name])
        block_values = [str(metadata.get(accession, {}).get(block_field, "")) for accession in aligned["accessions"]]
        row = {
            "comparison": comparison,
            "protein_run": protein_name,
            "context_run": context_name,
            "block_unit": block_field,
            **paired_bootstrap_delta(aligned, block_values, iterations, seed + len(rows)),
        }
        rows.append(row)
    write_tsv(output_dir / "qc2_main_delta_block_bootstrap_ci.tsv", rows)
    return rows


def feature_leakage_qc(run_root: Path, freeze_dir: Path, output_dir: Path) -> list[dict[str, Any]]:
    manifest = load_json(run_root / "genome_aware_denovo.family_holdout" / "run_manifest.json")
    used_fields = set()
    for key in (
        "global_category_fields",
        "global_numeric_fields",
        "host_category_fields",
        "host_numeric_fields",
        "biophysics_fields",
        "neighbor_category_fields",
        "neighbor_numeric_fields",
    ):
        used_fields.update(str(item) for item in manifest.get(key, []))
    used_fields.update(str(item) for item in manifest.get("selected_context_blocks", []))

    manifest_rows = []
    feature_manifest_path = freeze_dir / "feature_manifest.tsv"
    if feature_manifest_path.exists():
        for row in read_tsv(feature_manifest_path):
            name = row.get("name", "")
            row["used_in_genome_aware_denovo"] = int(name in used_fields)
            manifest_rows.append(row)
    write_tsv(output_dir / "qc3_feature_manifest_used_by_genome_aware.tsv", manifest_rows)

    forbidden_rows = []
    lowered_fields = {field.lower(): field for field in used_fields}
    for check_name, patterns in FORBIDDEN_FEATURE_CHECKS:
        matched = sorted(
            original
            for lowered, original in lowered_fields.items()
            if any(pattern.lower() in lowered for pattern in patterns)
        )
        forbidden_rows.append(
            {
                "forbidden_feature_family": check_name,
                "present_in_genome_aware_denovo": int(bool(matched)),
                "matched_fields_json": json.dumps(matched, ensure_ascii=False),
                "reviewer_interpretation": "PASS" if not matched else "CHECK_OR_FAIL",
            }
        )
    write_tsv(output_dir / "qc3_forbidden_feature_check.tsv", forbidden_rows)
    return forbidden_rows


def matched_comparison_qc(run_root: Path, output_dir: Path) -> list[dict[str, Any]]:
    source_path = run_root / "source_decomposition" / "source_decomposition_summary.tsv"
    if not source_path.exists():
        return []
    source_rows = read_tsv(source_path)
    rows = []
    compare_config_keys = ("epochs", "batch_size", "eval_batch_size", "min_label_count", "learning_rate", "weight_decay")
    for source_row in source_rows:
        reference_run = source_row.get("reference_run", "")
        variant_run = source_row.get("variant_run", "")
        if not reference_run or not variant_run:
            continue
        ref_manifest_path = run_root / reference_run / "run_manifest.json"
        var_manifest_path = run_root / variant_run / "run_manifest.json"
        if not ref_manifest_path.exists() or not var_manifest_path.exists():
            continue
        ref = load_json(ref_manifest_path)
        var = load_json(var_manifest_path)
        ref_config = ref.get("config", {}) or {}
        var_config = var.get("config", {}) or {}
        checks = {
            "same_split": ref.get("split_strategy", {}).get("scheme") == var.get("split_strategy", {}).get("scheme"),
            "same_seed": ref.get("seed") == var.get("seed"),
            "same_sequence_backbone": ref.get("sequence_backbone") == var.get("sequence_backbone"),
            "same_labels": ref.get("label_names") == var.get("label_names"),
            "same_training_budget": all(ref_config.get(key) == var_config.get(key) for key in compare_config_keys),
            "same_thresholding_procedure": (run_root / reference_run / "best_thresholds.json").exists()
            and (run_root / variant_run / "best_thresholds.json").exists(),
        }
        rows.append(
            {
                "split_scheme": source_row.get("split_scheme", ""),
                "comparison_type": source_row.get("comparison_type", ""),
                "reference_run": reference_run,
                "variant_run": variant_run,
                **{key: int(value) for key, value in checks.items()},
                "matched_all_core": int(all(checks.values())),
                "reference_context_blocks": json.dumps(ref.get("selected_context_blocks", []), ensure_ascii=False),
                "variant_context_blocks": json.dumps(var.get("selected_context_blocks", []), ensure_ascii=False),
            }
        )
    write_tsv(output_dir / "qc4_matched_source_decomposition_comparisons.tsv", rows)
    return rows


def host_corruption_curve_qc(run_root: Path, output_dir: Path) -> list[dict[str, Any]]:
    suite_path = run_root / "suite_summary.tsv"
    if not suite_path.exists():
        return []
    suite_rows = read_tsv(suite_path)
    row_by_name = {row["run_name"]: row for row in suite_rows}
    out_rows = []
    for split in ("family_holdout", "host_holdout"):
        baseline_name = f"genome_aware_denovo_biophysics.{split}"
        if baseline_name in row_by_name:
            row = row_by_name[baseline_name]
            out_rows.append(
                {
                    "split_scheme": split,
                    "host_corruption_fraction": 0.0,
                    "run_name": baseline_name,
                    "test_macro_average_precision": row.get("test_macro_average_precision"),
                    "test_macro_f1": row.get("test_macro_f1"),
                    "curve_type": "host_corruption",
                }
            )
        pattern = re.compile(rf"^genome_aware_denovo_biophysics_host_corrupt_(\d+)\.{re.escape(split)}$")
        for run_name, row in sorted(row_by_name.items()):
            match = pattern.match(run_name)
            if not match:
                continue
            out_rows.append(
                {
                    "split_scheme": split,
                    "host_corruption_fraction": int(match.group(1)) / 100.0,
                    "run_name": run_name,
                    "test_macro_average_precision": row.get("test_macro_average_precision"),
                    "test_macro_f1": row.get("test_macro_f1"),
                    "curve_type": "host_corruption",
                }
            )
        shuffle_name = f"genome_aware_denovo_biophysics_control_host_shuffle.{split}"
        if shuffle_name in row_by_name:
            row = row_by_name[shuffle_name]
            out_rows.append(
                {
                    "split_scheme": split,
                    "host_corruption_fraction": "shuffle_within_family",
                    "run_name": shuffle_name,
                    "test_macro_average_precision": row.get("test_macro_average_precision"),
                    "test_macro_f1": row.get("test_macro_f1"),
                    "curve_type": "host_shuffle_control",
                }
            )
    write_tsv(output_dir / "qc5_host_corruption_curve.tsv", out_rows)
    return out_rows


def label_deep_dive_qc(
    label_name: str,
    context_pred: dict[str, Any],
    metadata: dict[str, dict[str, Any]],
    output_dir: Path,
    top_k: int = 50,
) -> dict[str, Any]:
    label_names = context_pred["label_names"]
    if label_name not in label_names:
        raise ValueError(f"Label '{label_name}' was not found in run labels: {label_names}")
    label_idx = label_names.index(label_name)
    prob = context_pred["y_prob"][:, label_idx]
    true = context_pred["y_true"][:, label_idx].astype(np.uint8)
    threshold = float(context_pred["thresholds"][label_idx])
    pred = prob >= threshold
    accessions = context_pred["accessions"]

    split_counts = Counter()
    train_families = set()
    test_families = set()
    test_hosts = set()
    for row in metadata.values():
        if label_name not in row["labels"]:
            continue
        split = row.get("family_holdout_split", "")
        split_counts[split] += 1
        if split == "train":
            train_families.add(row.get("virus_family", ""))
        if split == "test":
            test_families.add(row.get("virus_family", ""))
            test_hosts.add(row.get("host_taxid_key", ""))

    order = np.argsort(-prob, kind="mergesort")
    tp_rows = []
    fp_rows = []
    for idx in order:
        accession = str(accessions[idx])
        meta = metadata.get(accession, {})
        base = {
            "protein_accession": accession,
            "probability": float(prob[idx]),
            "true_label": int(true[idx]),
            "predicted_at_threshold": int(pred[idx]),
            "genome_version": meta.get("genome_version", ""),
            "virus_family": meta.get("virus_family", ""),
            "host_taxid_key": meta.get("host_taxid_key", ""),
            "description": meta.get("description", ""),
            "cds_product": meta.get("cds_product", ""),
        }
        if true[idx] == 1 and len(tp_rows) < top_k:
            tp_rows.append(base)
        elif true[idx] == 0 and len(fp_rows) < top_k:
            fp_rows.append(base)
        if len(tp_rows) >= top_k and len(fp_rows) >= top_k:
            break
    write_tsv(output_dir / f"qc6_{label_name}_top_true_positives.tsv", tp_rows)
    write_tsv(output_dir / f"qc6_{label_name}_top_false_positives.tsv", fp_rows)

    sorted_true = true[order].astype(np.float64)
    sorted_prob = prob[order]
    cumulative_tp = np.cumsum(sorted_true)
    ranks = np.arange(1, sorted_true.shape[0] + 1, dtype=np.float64)
    precision = cumulative_tp / ranks
    recall = cumulative_tp / max(float(sorted_true.sum()), 1.0)
    pr_rows = [
        {
            "rank": int(rank),
            "threshold": float(sorted_prob[idx]),
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
            "true_positive_cumulative": int(cumulative_tp[idx]),
        }
        for idx, rank in enumerate(ranks.astype(int))
        if idx < 1000 or idx % 100 == 0
    ]
    write_tsv(output_dir / f"qc6_{label_name}_pr_curve.tsv", pr_rows)

    summary = {
        "label": label_name,
        "train_positives": int(split_counts.get("train", 0)),
        "val_positives": int(split_counts.get("val", 0)),
        "test_positives": int(split_counts.get("test", 0)),
        "test_family_count": len({value for value in test_families if value}),
        "train_family_count": len({value for value in train_families if value}),
        "test_host_group_count": len({value for value in test_hosts if value}),
        "threshold": threshold,
        "test_predicted_positive_count": int(pred.sum()),
        "test_true_positive_count_at_threshold": int(((pred == 1) & (true == 1)).sum()),
        "test_false_positive_count_at_threshold": int(((pred == 1) & (true == 0)).sum()),
    }
    (output_dir / f"qc6_{label_name}_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def load_module_supported_accessions(module_path: Path) -> set[str]:
    if not module_path.exists():
        return set()
    supported = set()
    for row in read_tsv(module_path):
        accession = row.get("center_accession", "").strip()
        cluster_id = row.get("cluster_id", "").strip()
        if accession and cluster_id and cluster_id != "-1":
            supported.add(accession)
    return supported


def candidate_breakdown_qc(
    run_root: Path,
    protein_pred: dict[str, Any],
    context_pred: dict[str, Any],
    output_dir: Path,
    context_gain_threshold: float,
) -> dict[str, Any]:
    candidate_path = run_root / "uncertainty" / "genome_aware_denovo.family_holdout" / "candidate_prioritization.tsv"
    if not candidate_path.exists():
        return {}
    module_supported = load_module_supported_accessions(run_root / "module_discovery" / "module_candidates.tsv")
    aligned = align_predictions(protein_pred, context_pred)
    accession_to_idx = {accession: idx for idx, accession in enumerate(aligned["accessions"])}
    label_to_idx = {label: idx for idx, label in enumerate(aligned["label_names"])}
    assignment_rows = []
    gated_rows = []
    for row in read_tsv(candidate_path):
        if str(row.get("passes_fdr_gate", "")).lower() != "true":
            continue
        gated_rows.append(row)
        accession = row.get("protein_accession", "")
        predicted_labels = parse_json_list(row.get("predicted_labels_at_precision_threshold", "[]"))
        if not predicted_labels:
            predicted_labels = [row.get("top_label", "")]
        for label in predicted_labels:
            idx = accession_to_idx.get(accession)
            label_idx = label_to_idx.get(label)
            context_gain = None
            if idx is not None and label_idx is not None:
                context_gain = float(aligned["right_prob"][idx, label_idx] - aligned["left_prob"][idx, label_idx])
            assignment_rows.append(
                {
                    "protein_accession": accession,
                    "genome_version": row.get("genome_version", ""),
                    "description": row.get("description", ""),
                    "candidate_label": label,
                    "top_label": row.get("top_label", ""),
                    "top_probability_calibrated": row.get("top_probability_calibrated", ""),
                    "context_gain": context_gain,
                    "high_context_gain": int(context_gain is not None and context_gain >= context_gain_threshold),
                    "hypothetical_or_unknown": int(bool(re.search(r"hypothetical|uncharacterized|unknown", row.get("description", ""), re.IGNORECASE))),
                    "module_supported": int(accession in module_supported),
                }
            )
    write_tsv(output_dir / "qc7_candidate_assignments.tsv", assignment_rows)
    unique_proteins = {row["protein_accession"] for row in assignment_rows}
    unique_genomes = {row["genome_version"] for row in assignment_rows}
    hypothetical_assignments = [row for row in assignment_rows if int(row["hypothetical_or_unknown"]) == 1]
    high_context = [row for row in assignment_rows if int(row["high_context_gain"]) == 1]
    module_rows = [row for row in assignment_rows if int(row["module_supported"]) == 1]
    casebook_count = len(list((run_root / "module_discovery" / "casebooks").glob("*.casebook.md"))) if (run_root / "module_discovery" / "casebooks").exists() else 0
    breakdown_rows = [
        {"category": "FDR-gated proteins", "count": len(gated_rows)},
        {"category": "FDR-gated protein-label assignments", "count": len(assignment_rows)},
        {"category": "unique proteins involved", "count": len(unique_proteins)},
        {"category": "unique genomes involved", "count": len(unique_genomes)},
        {"category": "hypothetical/uncharacterized/unknown assignments", "count": len(hypothetical_assignments)},
        {"category": "hypothetical/uncharacterized/unknown unique proteins", "count": len({row["protein_accession"] for row in hypothetical_assignments})},
        {"category": f"high context-gain assignments delta_p>={context_gain_threshold}", "count": len(high_context)},
        {"category": "module-supported assignments", "count": len(module_rows)},
        {"category": "selected casebook candidates", "count": casebook_count},
    ]
    write_tsv(output_dir / "qc7_candidate_breakdown.tsv", breakdown_rows)
    return {row["category"]: row["count"] for row in breakdown_rows}


def weak_label_context_sensitive_fraction(row: dict[str, str], context_sensitive_labels: set[str]) -> float:
    try:
        counts = json.loads(row.get("weak_label_counts_json", "{}") or "{}")
    except json.JSONDecodeError:
        counts = {}
    if not isinstance(counts, dict) or not counts:
        return 0.0
    total = sum(float(value) for value in counts.values())
    if total <= 0:
        return 0.0
    sensitive = sum(float(value) for key, value in counts.items() if key in context_sensitive_labels)
    return sensitive / total


def module_cluster_metrics(rows: list[dict[str, Any]], context_sensitive_labels: set[str]) -> dict[str, Any]:
    clusters: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        cluster_id = str(row.get("cluster_id", ""))
        if not cluster_id or cluster_id == "-1":
            continue
        clusters[cluster_id].append(row)
    per_cluster = []
    for cluster_id, cluster_rows in clusters.items():
        signatures = Counter(str(row.get("neighborhood_signature", "")) for row in cluster_rows)
        size = len(cluster_rows)
        per_cluster.append(
            {
                "cluster_id": cluster_id,
                "module_count": size,
                "family_count": len({str(row.get("virus_family", "")) for row in cluster_rows if str(row.get("virus_family", ""))}),
                "neighborhood_consistency": safe_divide(signatures.most_common(1)[0][1] if signatures else 0, size),
                "hypothetical_ratio_mean": float(np.mean([float(row.get("hypothetical_ratio", 0.0) or 0.0) for row in cluster_rows])),
                "structural_membrane_vote_fraction_mean": float(np.mean([float(row.get("structural_membrane_vote_fraction", 0.0) or 0.0) for row in cluster_rows])),
                "context_sensitive_label_fraction_mean": float(np.mean([weak_label_context_sensitive_fraction(row, context_sensitive_labels) for row in cluster_rows])),
            }
        )
    total = sum(row["module_count"] for row in per_cluster)

    def weighted_mean(field: str) -> float:
        return safe_divide(sum(float(row[field]) * int(row["module_count"]) for row in per_cluster), total)

    return {
        "cluster_count": len(per_cluster),
        "module_count": total,
        "weighted_neighborhood_consistency": weighted_mean("neighborhood_consistency") if per_cluster else 0.0,
        "weighted_hypothetical_ratio": weighted_mean("hypothetical_ratio_mean") if per_cluster else 0.0,
        "weighted_structural_membrane_vote_fraction": weighted_mean("structural_membrane_vote_fraction_mean") if per_cluster else 0.0,
        "weighted_context_sensitive_label_fraction": weighted_mean("context_sensitive_label_fraction_mean") if per_cluster else 0.0,
        "mean_family_recurrence": float(np.mean([row["family_count"] for row in per_cluster])) if per_cluster else 0.0,
    }


def module_null_qc(run_root: Path, output_dir: Path, iterations: int, seed: int) -> dict[str, Any]:
    module_path = run_root / "module_discovery" / "module_candidates.tsv"
    atlas_path = run_root / "context_atlas_plain.family_holdout.v2" / "label_deltas.tsv"
    if not module_path.exists():
        return {}
    context_sensitive_labels = set()
    if atlas_path.exists():
        for row in read_tsv(atlas_path):
            if float(row.get("delta_average_precision", 0.0) or 0.0) > 0:
                context_sensitive_labels.add(row.get("label", ""))
    rows = read_tsv(module_path)
    observed = module_cluster_metrics(rows, context_sensitive_labels)
    cluster_ids = [row.get("cluster_id", "") for row in rows]
    rng = np.random.default_rng(seed)
    null_rows = []
    for iteration in range(iterations):
        shuffled = list(cluster_ids)
        rng.shuffle(shuffled)
        shuffled_rows = [dict(row, cluster_id=cluster_id) for row, cluster_id in zip(rows, shuffled)]
        metrics = module_cluster_metrics(shuffled_rows, context_sensitive_labels)
        metrics["iteration"] = iteration
        null_rows.append(metrics)
    write_tsv(output_dir / "qc8_module_cluster_assignment_null_iterations.tsv", null_rows)

    summary_rows = []
    for metric in (
        "weighted_neighborhood_consistency",
        "weighted_hypothetical_ratio",
        "weighted_structural_membrane_vote_fraction",
        "weighted_context_sensitive_label_fraction",
        "mean_family_recurrence",
    ):
        values = np.asarray([float(row[metric]) for row in null_rows], dtype=np.float64)
        observed_value = float(observed.get(metric, 0.0))
        summary_rows.append(
            {
                "null_type": "cluster_assignment_permutation_preserving_cluster_sizes",
                "metric": metric,
                "observed": observed_value,
                "null_mean": float(values.mean()) if values.size else None,
                "null_ci_low": float(np.percentile(values, 2.5)) if values.size else None,
                "null_ci_high": float(np.percentile(values, 97.5)) if values.size else None,
                "empirical_p_observed_greater_equal_null": safe_divide(float(np.sum(values >= observed_value)) + 1.0, values.size + 1.0) if values.size else None,
                "iterations": iterations,
            }
        )
    write_tsv(output_dir / "qc8_module_discovery_null_control.tsv", summary_rows)
    return {"observed": observed, "summary_rows": summary_rows}


def load_predictions_for_needed_runs(
    run_root: Path,
    output_dir: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, dict[str, Any]]:
    needed = [
        "protein_only.family_holdout",
        "genome_aware_denovo.family_holdout",
        "protein_only.host_holdout",
        "genome_aware_denovo.host_holdout",
        "protein_only_biophysics.family_holdout",
        "genome_aware_denovo_biophysics.family_holdout",
        "protein_only_biophysics.host_holdout",
        "genome_aware_denovo_biophysics.host_holdout",
    ]
    predictions = {}
    for run_name in needed:
        run_dir = run_root / run_name
        if not (run_dir / "best_model.pt").exists():
            continue
        predictions[run_name] = load_run_predictions(
            run_dir,
            output_dir,
            device,
            args.batch_size,
            args.num_workers,
            args.prefetch_factor,
            args.force_predict,
        )
    return predictions


def main() -> int:
    args = parse_args()
    root = repo_root()
    run_root = resolve_path(root, args.run_root)
    input_path = resolve_path(root, args.input)
    split_manifest = resolve_path(root, args.split_manifest)
    freeze_dir = resolve_path(root, args.freeze_dir)
    output_dir = resolve_path(root, args.output_dir) if args.output_dir else run_root / "qc_review"
    output_dir.mkdir(parents=True, exist_ok=True)

    strict_rows = load_strict_split_rows(split_manifest)
    metadata = load_training_metadata(input_path, strict_rows)
    device = choose_device(args.device)
    predictions = load_predictions_for_needed_runs(run_root, output_dir, args, device)

    exact_accessions, qc1_summary = exact_transfer_qc(metadata, output_dir)
    strict_zero_summary = {}
    if "protein_only.family_holdout" in predictions and "genome_aware_denovo.family_holdout" in predictions:
        strict_zero_summary = strict_zero_metrics_qc(
            predictions["protein_only.family_holdout"],
            predictions["genome_aware_denovo.family_holdout"],
            exact_accessions,
            output_dir,
        )

    ci_rows = ci_qc(run_root, predictions, metadata, output_dir, args.bootstrap_iterations, args.seed)
    forbidden_rows = feature_leakage_qc(run_root, freeze_dir, output_dir)
    matched_rows = matched_comparison_qc(run_root, output_dir)
    host_curve_rows = host_corruption_curve_qc(run_root, output_dir)
    label_summary = {}
    if "genome_aware_denovo.family_holdout" in predictions:
        label_summary = label_deep_dive_qc(args.label_of_interest, predictions["genome_aware_denovo.family_holdout"], metadata, output_dir)
    candidate_summary = {}
    if "protein_only.family_holdout" in predictions and "genome_aware_denovo.family_holdout" in predictions:
        candidate_summary = candidate_breakdown_qc(
            run_root,
            predictions["protein_only.family_holdout"],
            predictions["genome_aware_denovo.family_holdout"],
            output_dir,
            args.candidate_context_gain_threshold,
        )
    module_summary = module_null_qc(run_root, output_dir, args.module_null_iterations, args.seed)

    report = {
        "created_at": timestamp(),
        "run_root": str(run_root),
        "output_dir": str(output_dir),
        "qc1_exact_transfer": qc1_summary,
        "qc1_strict_zero_metrics": strict_zero_summary,
        "qc2_ci_row_count": len(ci_rows),
        "qc3_forbidden_feature_rows": forbidden_rows,
        "qc4_matched_comparison_row_count": len(matched_rows),
        "qc5_host_curve_row_count": len(host_curve_rows),
        "qc6_label_summary": label_summary,
        "qc7_candidate_summary": candidate_summary,
        "qc8_module_null_summary": module_summary,
    }
    (output_dir / "qc_review_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "qc_report": str(output_dir / "qc_review_report.json")}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
