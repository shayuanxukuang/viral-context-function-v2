from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from train_overnight_baseline import (
    AA_TO_ID,
    LABEL_RULES,
    ProteinSequenceDataset,
    ViralSequenceBaseline,
    choose_device,
    label_hits,
    make_dataloader,
    normalize_text,
    open_text,
    predict,
)


FAMILY_SUFFIXES = ("viridae", "virinae", "viriformidae")
UNKNOWN_MARKERS = ("hypothetical protein", "uncharacterized", "unknown protein")
HIGH_IMPACT_LABELS = {
    "polymerase": 1.0,
    "helicase": 1.0,
    "protease": 1.0,
    "nuclease": 1.0,
    "ligase": 1.0,
    "methyltransferase": 0.95,
    "integrase_recombinase": 0.95,
    "portal_terminase_packaging": 0.9,
    "lysis": 0.9,
    "tail_fiber_receptor": 0.85,
    "tail_assembly": 0.85,
    "capsid_head": 0.8,
    "nucleocapsid": 0.8,
    "envelope_glycoprotein": 0.75,
    "membrane_matrix": 0.7,
    "transcription_regulator": 0.85,
    "polyprotein": 0.75,
}
OUTPUT_FIELDS = [
    "protein_accession",
    "virus_tax_id",
    "genome_version",
    "virus_name",
    "virus_family",
    "host_supergroup",
    "host_record_count",
    "host_join_strategy",
    "protein_feature_type",
    "protein_length_aa",
    "protein_description",
    "cds_product",
    "weak_labels_json",
    "predicted_labels_json",
    "predicted_label_count",
    "top_label",
    "top_probability",
    "second_label",
    "second_probability",
    "probability_margin",
    "threshold_for_top_label",
    "is_unknown_text",
    "is_unlabeled_by_rules",
    "reviewed_uniprot_entries_for_taxon",
    "candidate_bucket",
    "candidate_score",
]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full-dataset inference and rank candidate proteins.")
    parser.add_argument("--run-dir", required=True, help="Training run directory containing checkpoint, cache, and manifest")
    parser.add_argument("--input", default="", help="Protein index table. Defaults to run_manifest input")
    parser.add_argument("--checkpoint", default="", help="Checkpoint path. Defaults to <run-dir>/best_model.pt")
    parser.add_argument("--cache-path", default="", help="Cache path. Defaults to <run-dir>/dataset_cache.pt")
    parser.add_argument("--thresholds", default="", help="Threshold JSON. Defaults to <run-dir>/best_thresholds.json")
    parser.add_argument("--output-dir", default="", help="Output directory. Defaults to <run-dir>/inference")
    parser.add_argument("--device", default="auto", help="Device override, e.g. cuda, cuda:0, cpu")
    parser.add_argument("--batch-size", type=int, default=2048, help="Inference batch size")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--prefetch-factor", type=int, default=4, help="DataLoader prefetch factor when workers > 0")
    parser.add_argument("--min-top-prob", type=float, default=0.85, help="Minimum top probability for candidate ranking")
    parser.add_argument("--min-margin", type=float, default=0.10, help="Minimum top-vs-second margin for confident candidates")
    parser.add_argument("--top-k-candidates", type=int, default=5000, help="How many ranked candidates to keep; 0 means all")
    parser.add_argument("--debug-limit", type=int, default=0, help="Optional row cap for smoke tests")
    return parser.parse_args()


def parse_json_list(raw: str) -> list[str]:
    raw = raw.strip()
    if not raw:
        return []
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def lineage_parts(lineage: str) -> list[str]:
    return [part.strip().rstrip(".") for part in lineage.split(";") if part.strip()]


def derive_virus_family(lineage: str) -> str:
    parts = lineage_parts(lineage)
    for part in reversed(parts):
        lower = part.lower()
        if any(lower.endswith(suffix) for suffix in FAMILY_SUFFIXES):
            return part
    for part in reversed(parts):
        lower = part.lower()
        if lower not in {"virus", "viruses"} and lower.endswith("virus"):
            return part
    if parts:
        return parts[-1]
    return "unknown"


def derive_host_supergroup(host_lineages_json: str, source_host: str) -> str:
    host_lineages = parse_json_list(host_lineages_json)
    if host_lineages:
        tokens = {token.strip().lower() for token in host_lineages[0].split(";") if token.strip()}
        if "bacteria" in tokens:
            return "Bacteria"
        if "archaea" in tokens:
            return "Archaea"
        if "viridiplantae" in tokens:
            return "Viridiplantae"
        if "metazoa" in tokens:
            return "Metazoa"
        if "fungi" in tokens:
            return "Fungi"
        if {"sar", "stramenopiles", "alveolata", "rhizaria"} & tokens:
            return "SAR"
        if "amoebozoa" in tokens:
            return "Amoebozoa"
        if {"discoba", "excavata", "metamonada"} & tokens:
            return "Excavata"
        if "eukaryota" in tokens:
            return "OtherEukaryota"
        if "root" in tokens:
            return "root"
    if source_host.strip():
        return "source_host_only"
    return "unknown"


def is_unknown_text(text: str) -> bool:
    return any(marker in text for marker in UNKNOWN_MARKERS)


def understudied_taxon_score(uniprot_entries: int) -> float:
    capped = min(uniprot_entries, 500)
    return 1.0 - (math.log1p(capped) / math.log1p(500))


def load_run_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_path(root: Path, value: str, fallback: Path) -> Path:
    if not value:
        return fallback.resolve()
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    return (root / path).resolve()


def resolve_input_path(root: Path, cli_value: str, manifest_value: str) -> Path:
    if cli_value:
        cli_path = Path(cli_value)
        if cli_path.is_absolute():
            return cli_path.resolve()
        return (root / cli_path).resolve()

    manifest_path = Path(manifest_value)
    if manifest_path.is_absolute() and manifest_path.exists():
        return manifest_path.resolve()
    if not manifest_path.is_absolute():
        candidate = (root / manifest_path).resolve()
        if candidate.exists():
            return candidate
    return (root / "data/processed/training/viral_protein_training_index.tsv.gz").resolve()


def load_thresholds(path: Path, label_names: list[str]) -> np.ndarray:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "thresholds" in data:
        threshold_map = data["thresholds"]
    else:
        threshold_map = data
    return np.asarray([float(threshold_map.get(label_name, 0.5)) for label_name in label_names], dtype=np.float32)


def open_tsv_writer(path: Path, fieldnames: list[str]) -> tuple[csv.DictWriter, gzip.GzipFile]:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = gzip.open(path, "wt", encoding="utf-8", newline="")
    writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
    writer.writeheader()
    return writer, handle


def load_model(
    cache: dict[str, Any],
    checkpoint_path: Path,
    run_manifest: dict[str, Any],
    device: torch.device,
) -> nn.Module:
    category_sizes = []
    for field_idx, _field in enumerate(cache["category_fields"]):
        values = cache["categories"][:, field_idx]
        category_sizes.append(int(values.max()) + 1)

    config = run_manifest["config"]
    model = ViralSequenceBaseline(
        vocab_size=len(AA_TO_ID) + 1,
        num_labels=len(cache["label_names"]),
        category_sizes=category_sizes,
        numeric_dim=cache["numeric"].shape[1],
        embed_dim=int(config["embed_dim"]),
        hidden_dim=int(config["hidden_dim"]),
        dropout=float(config["dropout"]),
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def build_active_rule_names(label_names: list[str]) -> set[str]:
    label_set = set(label_names)
    return {rule.name for rule in LABEL_RULES if rule.name in label_set}


def top_two(probabilities: np.ndarray, label_names: list[str]) -> tuple[str, float, str, float]:
    if probabilities.size == 1:
        return label_names[0], float(probabilities[0]), label_names[0], float(probabilities[0])
    order = np.argsort(probabilities)[::-1]
    top_idx = int(order[0])
    second_idx = int(order[1])
    return (
        label_names[top_idx],
        float(probabilities[top_idx]),
        label_names[second_idx],
        float(probabilities[second_idx]),
    )


def choose_candidate_bucket(
    unlabeled_by_rules: bool,
    unknown_text: bool,
    predicted_label_count: int,
    top_probability: float,
    threshold_for_top_label: float,
    probability_margin: float,
    taxon_novelty: float,
    min_top_prob: float,
    min_margin: float,
) -> str:
    confident = top_probability >= max(min_top_prob, threshold_for_top_label)
    if unlabeled_by_rules and predicted_label_count >= 2 and top_probability >= min_top_prob - 0.05 and probability_margin < min_margin:
        return "ambiguous_unknown"
    if unlabeled_by_rules and confident and probability_margin >= min_margin:
        return "high_confidence_unknown"
    if unknown_text and top_probability >= max(threshold_for_top_label, min_top_prob - 0.05):
        return "hypothetical_watchlist"
    if taxon_novelty >= 0.8 and confident:
        return "underannotated_taxon"
    return ""


def candidate_score(
    top_probability: float,
    probability_margin: float,
    unlabeled_by_rules: bool,
    unknown_text: bool,
    taxon_novelty: float,
    label_name: str,
) -> float:
    novelty_text = 0.0
    if unlabeled_by_rules:
        novelty_text = 1.0
    elif unknown_text:
        novelty_text = 0.65
    label_weight = HIGH_IMPACT_LABELS.get(label_name, 0.75)
    return (
        0.45 * top_probability
        + 0.15 * max(probability_margin, 0.0)
        + 0.20 * novelty_text
        + 0.15 * taxon_novelty
        + 0.05 * label_weight
    )


def main() -> int:
    args = parse_args()
    root = repo_root()
    run_dir = resolve_path(root, args.run_dir, root / args.run_dir)
    run_manifest_path = run_dir / "run_manifest.json"
    run_manifest = load_run_manifest(run_manifest_path)

    input_path = resolve_input_path(root, args.input, str(run_manifest["input"]))
    checkpoint_path = resolve_path(root, args.checkpoint, run_dir / "best_model.pt")
    cache_path = resolve_path(root, args.cache_path, run_dir / "dataset_cache.pt")
    thresholds_path = resolve_path(root, args.thresholds, run_dir / "best_thresholds.json")
    output_dir = resolve_path(root, args.output_dir, run_dir / "inference")
    output_dir.mkdir(parents=True, exist_ok=True)

    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    label_names: list[str] = list(cache["label_names"])
    thresholds = load_thresholds(thresholds_path, label_names)

    all_indices = np.arange(len(cache["lengths"]), dtype=np.int64)
    if args.debug_limit:
        all_indices = all_indices[: args.debug_limit]

    device = choose_device(args.device)
    model = load_model(cache, checkpoint_path, run_manifest, device)
    pin_memory = device.type == "cuda"
    dataset = ProteinSequenceDataset(cache, all_indices)
    loader = make_dataloader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        pin_memory=pin_memory,
    )

    autocast_dtype = None
    if device.type == "cuda":
        autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    probabilities, _weak_targets, observed_indices = predict(model, loader, device, autocast_dtype)
    if not np.array_equal(observed_indices, all_indices):
        raise RuntimeError("Inference order mismatch between cache indices and loader output.")

    active_rule_names = build_active_rule_names(label_names)
    label_name_to_idx = {label_name: idx for idx, label_name in enumerate(label_names)}

    all_predictions_path = output_dir / "all_predictions.tsv.gz"
    candidate_ranked_path = output_dir / "candidate_ranked.tsv.gz"
    report_path = output_dir / "candidate_report.json"

    candidates: list[dict[str, str]] = []

    all_writer, all_handle = open_tsv_writer(all_predictions_path, OUTPUT_FIELDS)
    try:
        with open_text(input_path) as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row_idx, row in enumerate(reader):
                if row_idx >= len(all_indices):
                    break

                probs = probabilities[row_idx]
                top_label, top_probability, second_label, second_probability = top_two(probs, label_names)
                probability_margin = top_probability - second_probability
                predicted_indices = np.flatnonzero(probs >= thresholds)
                predicted_labels = [label_names[int(idx)] for idx in predicted_indices]
                top_threshold = float(thresholds[label_name_to_idx[top_label]])

                text = normalize_text(row)
                unknown_text = is_unknown_text(text)
                weak_label_names = [
                    LABEL_RULES[idx].name
                    for idx in label_hits(text)
                    if LABEL_RULES[idx].name in active_rule_names
                ]
                weak_label_names = sorted(set(weak_label_names))
                unlabeled_by_rules = len(weak_label_names) == 0

                taxon_uniprot_entries = int(row.get("reviewed_uniprot_entries_for_taxon", "0") or "0")
                taxon_novelty = understudied_taxon_score(taxon_uniprot_entries)
                bucket = choose_candidate_bucket(
                    unlabeled_by_rules=unlabeled_by_rules,
                    unknown_text=unknown_text,
                    predicted_label_count=len(predicted_labels),
                    top_probability=top_probability,
                    threshold_for_top_label=top_threshold,
                    probability_margin=probability_margin,
                    taxon_novelty=taxon_novelty,
                    min_top_prob=args.min_top_prob,
                    min_margin=args.min_margin,
                )
                score = candidate_score(
                    top_probability=top_probability,
                    probability_margin=probability_margin,
                    unlabeled_by_rules=unlabeled_by_rules,
                    unknown_text=unknown_text,
                    taxon_novelty=taxon_novelty,
                    label_name=top_label,
                )

                record = {
                    "protein_accession": row.get("protein_accession", "").strip(),
                    "virus_tax_id": row.get("virus_tax_id", "").strip(),
                    "genome_version": row.get("genome_version", "").strip(),
                    "virus_name": row.get("virus_name", "").strip(),
                    "virus_family": derive_virus_family(row.get("virus_lineage", "")),
                    "host_supergroup": derive_host_supergroup(
                        row.get("host_lineages_json", ""),
                        row.get("source_host", ""),
                    ),
                    "host_record_count": row.get("host_record_count", "").strip() or "0",
                    "host_join_strategy": row.get("host_join_strategy", "").strip(),
                    "protein_feature_type": row.get("protein_feature_type", "").strip(),
                    "protein_length_aa": row.get("protein_length_aa", "").strip(),
                    "protein_description": row.get("protein_description", "").strip(),
                    "cds_product": row.get("cds_product", "").strip(),
                    "weak_labels_json": json.dumps(weak_label_names, ensure_ascii=False),
                    "predicted_labels_json": json.dumps(predicted_labels, ensure_ascii=False),
                    "predicted_label_count": str(len(predicted_labels)),
                    "top_label": top_label,
                    "top_probability": f"{top_probability:.6f}",
                    "second_label": second_label,
                    "second_probability": f"{second_probability:.6f}",
                    "probability_margin": f"{probability_margin:.6f}",
                    "threshold_for_top_label": f"{top_threshold:.6f}",
                    "is_unknown_text": "1" if unknown_text else "0",
                    "is_unlabeled_by_rules": "1" if unlabeled_by_rules else "0",
                    "reviewed_uniprot_entries_for_taxon": str(taxon_uniprot_entries),
                    "candidate_bucket": bucket,
                    "candidate_score": f"{score:.6f}",
                }
                all_writer.writerow(record)

                if bucket:
                    candidates.append(record)
    finally:
        all_handle.close()

    candidates.sort(
        key=lambda row: (
            float(row["candidate_score"]),
            float(row["top_probability"]),
            float(row["probability_margin"]),
        ),
        reverse=True,
    )
    if args.top_k_candidates > 0:
        candidates = candidates[: args.top_k_candidates]

    candidate_writer, candidate_handle = open_tsv_writer(candidate_ranked_path, OUTPUT_FIELDS)
    try:
        for record in candidates:
            candidate_writer.writerow(record)
    finally:
        candidate_handle.close()

    final_bucket_counts = Counter(record["candidate_bucket"] for record in candidates if record["candidate_bucket"])
    final_label_counts = Counter(record["top_label"] for record in candidates)

    report = {
        "created_at": timestamp(),
        "run_dir": str(run_dir),
        "input": str(input_path),
        "checkpoint": str(checkpoint_path),
        "cache_path": str(cache_path),
        "thresholds": str(thresholds_path),
        "output_dir": str(output_dir),
        "device": str(device),
        "rows_scored": int(len(all_indices)),
        "debug_limit": args.debug_limit,
        "candidate_count": len(candidates),
        "candidate_bucket_counts": dict(final_bucket_counts),
        "candidate_top_label_counts": dict(final_label_counts.most_common(20)),
        "config": {
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "prefetch_factor": args.prefetch_factor,
            "min_top_prob": args.min_top_prob,
            "min_margin": args.min_margin,
            "top_k_candidates": args.top_k_candidates,
        },
        "candidate_preview": candidates[:20],
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(
        {
            "rows_scored": report["rows_scored"],
            "candidate_count": report["candidate_count"],
            "candidate_bucket_counts": report["candidate_bucket_counts"],
        },
        ensure_ascii=False,
        indent=2,
    ))
    print(f"Wrote all predictions to {all_predictions_path}")
    print(f"Wrote ranked candidates to {candidate_ranked_path}")
    print(f"Wrote report to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
