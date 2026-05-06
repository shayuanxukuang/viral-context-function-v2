from __future__ import annotations

import argparse
import csv
import gzip
import json
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from biophysics_features import BIOPHYSICS_FIELD_NAMES, compute_biophysics, safe_log1p
from context_features import derive_virus_family
from label_rules import LABEL_RULES, label_hits, normalize_text
from task_mode_config import (
    GENOME_ORGANIZATION_NUMERIC_FIELDS,
    HOST_METADATA_CATEGORY_FIELDS,
    HOST_METADATA_NUMERIC_FIELDS,
    LOCAL_NEIGHBORHOOD_CATEGORY_FIELDS,
    LOCAL_NEIGHBORHOOD_NUMERIC_FIELDS,
    TASK_MODE_ORDER,
    resolve_context_blocks,
    task_mode_feature_lists,
)
from train_overnight_baseline import (
    AA_TO_ID,
    PAD_ID,
    choose_device,
    compute_metrics,
    compute_pos_weight,
    encode_sequence,
    linear_warmup_cosine_decay,
    load_split_assignments,
    open_text,
    optimize_thresholds,
    save_history,
    save_json,
    save_label_metrics,
    save_test_predictions,
    set_seed,
    SPLIT_SCHEME_TO_COLUMN,
    assign_split,
    collect_git_metadata,
)


HOST_CONTEXT_CATEGORY_FIELDS = set(HOST_METADATA_CATEGORY_FIELDS)
HOST_CONTEXT_NUMERIC_FIELDS = set(HOST_METADATA_NUMERIC_FIELDS)
LOCAL_CONTEXT_CATEGORY_FIELDS = set(LOCAL_NEIGHBORHOOD_CATEGORY_FIELDS)
LOCAL_CONTEXT_NUMERIC_FIELDS = set(LOCAL_NEIGHBORHOOD_NUMERIC_FIELDS)
GENOME_ORGANIZATION_NUMERIC_FIELD_SET = set(GENOME_ORGANIZATION_NUMERIC_FIELDS)
NEIGHBOR_LENGTH_FIELD = "neighbor_length_bin"
NEIGHBOR_FEATURE_FIELD = "neighbor_feature_type"
CACHE_PROGRESS_EVERY_ROWS = 25_000
CACHE_PROGRESS_EVERY_SECONDS = 30.0
CONTEXT_CONTROL_CHOICES = (
    "none",
    "shuffle_local_order",
    "shuffle_host_within_family",
    "shuffle_genome_relative_position",
)
LOCAL_CATEGORY_SWAP_PAIRS = (
    ("context_prev_length_bin", "context_next_length_bin"),
)
LOCAL_NUMERIC_SWAP_PAIRS = (
    ("context_has_prev_neighbor", "context_has_next_neighbor"),
    ("context_prev_gap_nt", "context_next_gap_nt"),
    ("context_prev_overlap_nt", "context_next_overlap_nt"),
    ("context_same_strand_prev", "context_same_strand_next"),
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write_status(path: Path, stage: str, **extra: Any) -> None:
    payload = {
        "updated_at": timestamp(),
        "stage": stage,
        **extra,
    }
    save_json(path, payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train task-mode-aware ViruFunc-FM models with leakage-safe feature gates.")
    parser.add_argument("--input", default="data/processed/training/viral_protein_training_index.tsv.gz", help="Protein-level training index")
    parser.add_argument("--output-dir", default="runs/task_mode_model", help="Directory for checkpoints, reports, and predictions")
    parser.add_argument("--cache-path", default="", help="Optional preprocessing cache path. Defaults to <output-dir>/dataset_cache.pt")
    parser.add_argument(
        "--split-manifest",
        default="data/processed/splits/viral_protein_strict_splits.tsv.gz",
        help="Strict split manifest path when split-scheme is strict",
    )
    parser.add_argument(
        "--split-scheme",
        default="family_holdout",
        choices=sorted(SPLIT_SCHEME_TO_COLUMN),
        help="Split strategy: default_hash or one of the strict holdout schemes",
    )
    parser.add_argument(
        "--task-mode",
        default="genome_aware_denovo",
        choices=TASK_MODE_ORDER,
        help="Task definition to train: protein_only, genome_aware_denovo, or annotation_refinement",
    )
    parser.add_argument(
        "--context-blocks",
        default="",
        help=(
            "Comma-separated subset of context blocks to enable. "
            "Defaults to the full leakage-safe set for the selected task mode."
        ),
    )
    parser.add_argument(
        "--context-table",
        default="",
        help="Optional split-aware context feature table. Auto-generated when omitted for non-protein modes.",
    )
    parser.add_argument("--with-biophysics", action="store_true", help="Append cheap sequence-derived biophysics features")
    parser.add_argument(
        "--context-control",
        default="none",
        choices=CONTEXT_CONTROL_CHOICES,
        help="Optional randomized-control corruption applied to context features before training",
    )
    parser.add_argument(
        "--host-corruption-fraction",
        type=float,
        default=0.0,
        help="Fraction of examples whose host metadata block is replaced with another host profile for robustness tests",
    )
    parser.add_argument(
        "--sequence-backbone",
        default="cnn",
        choices=("cnn", "precomputed_plm"),
        help="Sequence branch: current CNN or frozen precomputed protein-LM embeddings",
    )
    parser.add_argument("--plm-embedding-path", default="", help="Torch file containing precomputed embeddings aligned by accession")
    parser.add_argument("--neighbor-radius", type=int, default=2, help="Neighbor radius for the local context transformer")
    parser.add_argument("--max-length", type=int, default=2048, help="Maximum sequence length after head-tail truncation")
    parser.add_argument("--epochs", type=int, default=12, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=512, help="Training batch size")
    parser.add_argument("--eval-batch-size", type=int, default=1024, help="Evaluation batch size")
    parser.add_argument("--embed-dim", type=int, default=128, help="Sequence embedding size for the CNN fallback")
    parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden channel size")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="AdamW learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-2, help="AdamW weight decay")
    parser.add_argument("--num-workers", type=int, default=8, help="DataLoader workers")
    parser.add_argument("--prefetch-factor", type=int, default=4, help="DataLoader prefetch factor when workers > 0")
    parser.add_argument("--min-label-count", type=int, default=500, help="Minimum support for a weak label to be kept")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--gradient-clip", type=float, default=1.0, help="Gradient clipping norm")
    parser.add_argument("--warmup-fraction", type=float, default=0.05, help="Warmup fraction for cosine schedule")
    parser.add_argument("--max-pos-weight", type=float, default=50.0, help="Upper cap for BCE positive class weights")
    parser.add_argument("--device", default="auto", help="Device override, e.g. cuda, cuda:0, cpu")
    parser.add_argument("--compile-model", action="store_true", help="Use torch.compile when available")
    parser.add_argument("--force-rebuild-cache", action="store_true", help="Ignore an existing preprocessing cache")
    parser.add_argument("--debug-limit", type=int, default=0, help="Limit loaded examples for smoke tests")
    parser.add_argument("--save-test-predictions", action="store_true", help="Write gzipped test predictions table")
    return parser.parse_args()


def maybe_int(value: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def register_category(vocab: dict[str, int], value: str) -> int:
    normalized = value.strip() if value else "__MISSING__"
    if normalized not in vocab:
        vocab[normalized] = len(vocab)
    return vocab[normalized]


def resolve_split_config(root: Path, args: argparse.Namespace) -> tuple[str, str | None, Path | None]:
    split_scheme = args.split_scheme
    split_column = SPLIT_SCHEME_TO_COLUMN[split_scheme]
    if split_column is None:
        return split_scheme, None, None
    split_manifest_path = Path(args.split_manifest)
    if not split_manifest_path.is_absolute():
        split_manifest_path = (root / split_manifest_path).resolve()
    else:
        split_manifest_path = split_manifest_path.resolve()
    return split_scheme, split_column, split_manifest_path


def default_context_table_path(root: Path, split_scheme: str, task_mode: str) -> Path:
    return (root / f"data/processed/context/viral_protein_context.{split_scheme}.{task_mode}.tsv.gz").resolve()


def read_context_table_header(context_table_path: Path) -> list[str]:
    with open_text(context_table_path) as handle:
        reader = csv.reader(handle, delimiter="\t")
        try:
            return next(reader)
        except StopIteration:
            return []


def required_context_fields_for_task_mode(task_mode: str) -> list[str]:
    feature_lists = task_mode_feature_lists(task_mode, with_biophysics=False)
    return [*feature_lists["context_category_fields"], *feature_lists["context_numeric_fields"]]


def missing_context_fields(context_table_path: Path, required_fields: list[str]) -> list[str]:
    header = read_context_table_header(context_table_path)
    if not header:
        return ["protein_accession", *required_fields]
    header_set = set(header)
    return [field for field in ["protein_accession", *required_fields] if field not in header_set]


def ensure_context_table(root: Path, args: argparse.Namespace, context_table_path: Path) -> None:
    required_fields = required_context_fields_for_task_mode(args.task_mode)
    if context_table_path.exists():
        missing = missing_context_fields(context_table_path, required_fields)
        if not missing:
            return
        preview = ", ".join(missing[:10])
        print(
            f"[context] Rebuilding stale context table {context_table_path} "
            f"because it is missing fields: {preview}"
        )
    script_path = root / "scripts" / "build_context_features_splitaware.py"
    command = [
        sys.executable,
        str(script_path),
        "--input",
        args.input,
        "--split-manifest",
        args.split_manifest,
        "--split-scheme",
        args.split_scheme,
        "--task-mode",
        args.task_mode,
        "--output-dir",
        "data/processed/context",
    ]
    if args.debug_limit:
        command.extend(["--debug-limit", str(args.debug_limit)])
    subprocess.run(command, cwd=root, check=True)


def resolve_context_table_path(root: Path, args: argparse.Namespace) -> Path | None:
    if args.task_mode == "protein_only":
        return None
    if args.context_table:
        path = Path(args.context_table)
        resolved = (root / path).resolve() if not path.is_absolute() else path.resolve()
        required_fields = [
            *task_mode_feature_lists(args.task_mode, with_biophysics=False, context_blocks=args.context_blocks)["context_category_fields"],
            *task_mode_feature_lists(args.task_mode, with_biophysics=False, context_blocks=args.context_blocks)["context_numeric_fields"],
        ]
        if resolved.exists():
            missing = missing_context_fields(resolved, required_fields)
            if missing:
                preview = ", ".join(missing[:10])
                raise ValueError(
                    f"Explicit context table {resolved} is missing fields required by task mode "
                    f"'{args.task_mode}': {preview}"
                )
        return resolved
    path = default_context_table_path(root, args.split_scheme, args.task_mode)
    ensure_context_table(root, args, path)
    return path


def resolve_plm_embedding_path(root: Path, args: argparse.Namespace) -> Path | None:
    if args.sequence_backbone != "precomputed_plm":
        return None
    if not args.plm_embedding_path:
        raise ValueError("--plm-embedding-path is required when --sequence-backbone precomputed_plm")
    path = Path(args.plm_embedding_path)
    return (root / path).resolve() if not path.is_absolute() else path.resolve()


def refinement_numeric_features(row: dict[str, str], text: str) -> dict[str, float]:
    feature_type = row.get("protein_feature_type", "").strip()
    return {
        "log_host_record_count": safe_log1p(maybe_int(row.get("host_record_count", "0"))),
        "log_uniprot_entries": safe_log1p(maybe_int(row.get("reviewed_uniprot_entries_for_taxon", "0"))),
        "log_uniprot_go_entries": safe_log1p(maybe_int(row.get("reviewed_uniprot_entries_with_go_for_taxon", "0"))),
        "log_uniprot_interpro_entries": safe_log1p(maybe_int(row.get("reviewed_uniprot_entries_with_interpro_for_taxon", "0"))),
        "log_uniprot_ec_entries": safe_log1p(maybe_int(row.get("reviewed_uniprot_entries_with_ec_for_taxon", "0"))),
        "is_hypothetical": 1.0 if ("hypothetical protein" in text or "uncharacterized" in text or "unknown protein" in text) else 0.0,
        "is_mat_peptide": 1.0 if feature_type == "mat_peptide" else 0.0,
    }


def load_context_rows(context_table_path: Path, required_fields: list[str]) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    with open_text(context_table_path) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames:
            raise ValueError(f"Context table is missing a header: {context_table_path}")
        missing = [field for field in ["protein_accession", *required_fields] if field not in reader.fieldnames]
        if missing:
            preview = ", ".join(missing[:10])
            raise ValueError(f"Context table {context_table_path} is missing fields: {preview}")
        for row in reader:
            accession = row.get("protein_accession", "").strip()
            if accession:
                rows[accession] = row
    return rows


def load_precomputed_embeddings(path: Path) -> tuple[dict[str, np.ndarray], int]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    mapping: dict[str, np.ndarray] = {}
    if isinstance(payload, dict) and "accessions" in payload and "embeddings" in payload:
        accessions = [str(item) for item in payload["accessions"]]
        embeddings = payload["embeddings"]
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.detach().cpu().numpy()
        for accession, vector in zip(accessions, embeddings):
            mapping[accession] = np.asarray(vector, dtype=np.float32)
    elif isinstance(payload, dict):
        for accession, vector in payload.items():
            if isinstance(vector, torch.Tensor):
                vector = vector.detach().cpu().numpy()
            mapping[str(accession)] = np.asarray(vector, dtype=np.float32)
    else:
        raise ValueError(f"Unsupported embedding payload format: {type(payload).__name__}")

    if not mapping:
        raise ValueError(f"No embeddings were loaded from {path}")
    first_dim = int(next(iter(mapping.values())).shape[0])
    return mapping, first_dim


def cache_matches_request(cache: dict[str, Any], signature: dict[str, Any]) -> bool:
    return all(cache.get(key) == value for key, value in signature.items())


def cache_signature(
    args: argparse.Namespace,
    input_path: Path,
    split_scheme: str,
    split_column: str | None,
    split_manifest_path: Path | None,
    context_table_path: Path | None,
    plm_embedding_path: Path | None,
) -> dict[str, Any]:
    return {
        "cache_format_version": 2,
        "input_path": str(input_path),
        "split_scheme": split_scheme,
        "split_column": split_column or "",
        "split_manifest_path": str(split_manifest_path) if split_manifest_path else "",
        "task_mode": args.task_mode,
        "context_blocks": list(resolve_context_blocks(args.task_mode, args.context_blocks)),
        "context_table_path": str(context_table_path) if context_table_path else "",
        "with_biophysics": args.with_biophysics,
        "context_control": args.context_control,
        "host_corruption_fraction": float(args.host_corruption_fraction),
        "sequence_backbone": args.sequence_backbone,
        "plm_embedding_path": str(plm_embedding_path) if plm_embedding_path else "",
        "neighbor_radius": args.neighbor_radius,
        "max_length": args.max_length,
        "min_label_count": args.min_label_count,
        "debug_limit": args.debug_limit,
    }


def host_feature_payload(
    host_category_columns: dict[str, list[int]],
    host_numeric_rows: list[list[float]],
    record_idx: int,
) -> tuple[dict[str, int], list[float]]:
    return (
        {field: int(values[record_idx]) for field, values in host_category_columns.items()},
        [float(value) for value in host_numeric_rows[record_idx]],
    )


def assign_host_feature_payload(
    host_category_columns: dict[str, list[int]],
    host_numeric_rows: list[list[float]],
    record_idx: int,
    payload: tuple[dict[str, int], list[float]],
) -> None:
    category_values, numeric_values = payload
    for field, value in category_values.items():
        host_category_columns[field][record_idx] = int(value)
    host_numeric_rows[record_idx] = [float(value) for value in numeric_values]


def build_family_groups(records: list[dict[str, Any]]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for record_idx, record in enumerate(records):
        groups[str(record.get("virus_family", "unknown"))].append(record_idx)
    return groups


def apply_host_shuffle_within_family(
    records: list[dict[str, Any]],
    host_category_columns: dict[str, list[int]],
    host_numeric_rows: list[list[float]],
    rng: np.random.Generator,
) -> None:
    has_host_features = bool(host_category_columns) or (bool(host_numeric_rows) and len(host_numeric_rows[0]) > 0)
    if not has_host_features:
        return
    family_groups = build_family_groups(records)
    for indices in family_groups.values():
        if len(indices) <= 1:
            continue
        donor_indices = list(rng.permutation(np.asarray(indices, dtype=np.int64)).tolist())
        if donor_indices == indices:
            donor_indices = donor_indices[1:] + donor_indices[:1]
        payloads = [host_feature_payload(host_category_columns, host_numeric_rows, donor_idx) for donor_idx in donor_indices]
        for target_idx, payload in zip(indices, payloads):
            assign_host_feature_payload(host_category_columns, host_numeric_rows, target_idx, payload)


def apply_host_corruption(
    records: list[dict[str, Any]],
    host_category_columns: dict[str, list[int]],
    host_numeric_rows: list[list[float]],
    fraction: float,
    rng: np.random.Generator,
) -> int:
    has_host_features = bool(host_category_columns) or (bool(host_numeric_rows) and len(host_numeric_rows[0]) > 0)
    if fraction <= 0.0 or not has_host_features:
        return 0
    eligible = [idx for idx, record in enumerate(records) if str(record.get("virus_family", ""))]
    if not eligible:
        return 0
    corrupt_count = int(round(len(eligible) * fraction))
    if corrupt_count <= 0:
        return 0

    family_groups = build_family_groups(records)
    sampled_targets = rng.choice(np.asarray(eligible, dtype=np.int64), size=min(corrupt_count, len(eligible)), replace=False)
    all_indices = list(range(len(records)))
    for target_idx in sampled_targets.tolist():
        family = str(records[target_idx].get("virus_family", "unknown"))
        donor_pool = [idx for idx in family_groups.get(family, []) if idx != target_idx]
        if not donor_pool:
            donor_pool = [idx for idx in all_indices if idx != target_idx]
        if not donor_pool:
            continue
        donor_idx = int(rng.choice(np.asarray(donor_pool, dtype=np.int64)))
        payload = host_feature_payload(host_category_columns, host_numeric_rows, donor_idx)
        assign_host_feature_payload(host_category_columns, host_numeric_rows, target_idx, payload)
    return int(len(sampled_targets))


def apply_genome_relative_position_shuffle(
    grouped_indices: dict[str, list[int]],
    global_numeric_rows: list[list[float]],
    global_numeric_fields: list[str],
    rng: np.random.Generator,
) -> None:
    if "context_relative_order_fraction" not in global_numeric_fields:
        return
    relative_idx = global_numeric_fields.index("context_relative_order_fraction")
    for indices in grouped_indices.values():
        if len(indices) <= 1:
            continue
        values = [global_numeric_rows[record_idx][relative_idx] for record_idx in indices]
        shuffled = list(values)
        rng.shuffle(shuffled)
        for record_idx, value in zip(indices, shuffled):
            global_numeric_rows[record_idx][relative_idx] = float(value)


def apply_local_order_shuffle(
    global_category_columns: dict[str, list[int]],
    global_numeric_rows: list[list[float]],
    global_category_fields: list[str],
    global_numeric_fields: list[str],
    neighbor_category_columns: dict[str, list[np.ndarray]],
    neighbor_numeric_rows: list[np.ndarray],
    neighbor_mask_rows: list[np.ndarray],
    rng: np.random.Generator,
) -> None:
    category_fields = set(global_category_fields)
    numeric_index = {field: idx for idx, field in enumerate(global_numeric_fields)}

    for record_idx in range(len(neighbor_numeric_rows)):
        if rng.random() < 0.5:
            for left_field, right_field in LOCAL_CATEGORY_SWAP_PAIRS:
                if left_field in category_fields and right_field in category_fields:
                    left_value = global_category_columns[left_field][record_idx]
                    right_value = global_category_columns[right_field][record_idx]
                    global_category_columns[left_field][record_idx] = right_value
                    global_category_columns[right_field][record_idx] = left_value
            for left_field, right_field in LOCAL_NUMERIC_SWAP_PAIRS:
                if left_field in numeric_index and right_field in numeric_index:
                    left_idx = numeric_index[left_field]
                    right_idx = numeric_index[right_field]
                    global_numeric_rows[record_idx][left_idx], global_numeric_rows[record_idx][right_idx] = (
                        global_numeric_rows[record_idx][right_idx],
                        global_numeric_rows[record_idx][left_idx],
                    )

        occupied = np.flatnonzero(neighbor_mask_rows[record_idx])
        if occupied.size <= 1:
            continue
        shuffled = occupied.copy()
        rng.shuffle(shuffled)
        if np.array_equal(occupied, shuffled):
            shuffled = np.roll(shuffled, 1)
        numeric_payload = neighbor_numeric_rows[record_idx][shuffled].copy()
        neighbor_numeric_rows[record_idx][occupied] = numeric_payload
        for field, rows in neighbor_category_columns.items():
            payload = rows[record_idx][shuffled].copy()
            rows[record_idx][occupied] = payload


class TaskModeDataset(Dataset):
    def __init__(self, cache: dict[str, Any], indices: np.ndarray):
        self.cache = cache
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, item: int):
        idx = int(self.indices[item])
        start = int(self.cache["offsets"][idx])
        end = int(self.cache["offsets"][idx + 1])
        sequence = self.cache["flat_tokens"][start:end]
        seq_embedding = self.cache["sequence_embeddings"][idx] if self.cache["sequence_embeddings"] is not None else np.zeros((0,), dtype=np.float32)
        return {
            "tokens": sequence,
            "sequence_embedding": seq_embedding,
            "global_categories": self.cache["global_categories"][idx],
            "global_numeric": self.cache["global_numeric"][idx],
            "host_categories": self.cache["host_categories"][idx],
            "host_numeric": self.cache["host_numeric"][idx],
            "biophysics": self.cache["biophysics"][idx],
            "neighbor_categories": self.cache["neighbor_categories"][idx],
            "neighbor_numeric": self.cache["neighbor_numeric"][idx],
            "neighbor_mask": self.cache["neighbor_mask"][idx],
            "labels": self.cache["labels"][idx],
            "index": idx,
        }


def make_collate_fn():
    def collate(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        token_arrays = [item["tokens"] for item in batch]
        max_len = max(len(tokens) for tokens in token_arrays)
        token_tensor = torch.zeros((len(batch), max_len), dtype=torch.long)
        for row_idx, tokens in enumerate(token_arrays):
            token_tensor[row_idx, : len(tokens)] = torch.from_numpy(tokens.astype(np.int64, copy=False))

        sequence_embeddings = np.stack([item["sequence_embedding"] for item in batch])
        return {
            "tokens": token_tensor,
            "sequence_embedding": torch.as_tensor(sequence_embeddings, dtype=torch.float32),
            "global_categories": torch.as_tensor(np.stack([item["global_categories"] for item in batch]), dtype=torch.long),
            "global_numeric": torch.as_tensor(np.stack([item["global_numeric"] for item in batch]), dtype=torch.float32),
            "host_categories": torch.as_tensor(np.stack([item["host_categories"] for item in batch]), dtype=torch.long),
            "host_numeric": torch.as_tensor(np.stack([item["host_numeric"] for item in batch]), dtype=torch.float32),
            "biophysics": torch.as_tensor(np.stack([item["biophysics"] for item in batch]), dtype=torch.float32),
            "neighbor_categories": torch.as_tensor(np.stack([item["neighbor_categories"] for item in batch]), dtype=torch.long),
            "neighbor_numeric": torch.as_tensor(np.stack([item["neighbor_numeric"] for item in batch]), dtype=torch.float32),
            "neighbor_mask": torch.as_tensor(np.stack([item["neighbor_mask"] for item in batch]), dtype=torch.bool),
            "labels": torch.as_tensor(np.stack([item["labels"] for item in batch]), dtype=torch.float32),
            "indices": torch.as_tensor([item["index"] for item in batch], dtype=torch.long),
        }

    return collate


def make_dataloader(dataset: Dataset, batch_size: int, shuffle: bool, num_workers: int, prefetch_factor: int, pin_memory: bool) -> DataLoader:
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "collate_fn": make_collate_fn(),
        "drop_last": False,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(**kwargs)


class ConvBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dropout: float):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.bn = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv(x)
        x = self.bn(x)
        x = F.gelu(x)
        x = self.dropout(x)
        return x + residual


class CNNSequenceEncoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.sequence_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_ID)
        self.input_proj = nn.Conv1d(embed_dim, hidden_dim, kernel_size=1)
        self.blocks = nn.ModuleList(
            [
                ConvBlock(hidden_dim, kernel_size=5, dropout=dropout),
                ConvBlock(hidden_dim, kernel_size=9, dropout=dropout),
                ConvBlock(hidden_dim, kernel_size=17, dropout=dropout),
                ConvBlock(hidden_dim, kernel_size=5, dropout=dropout),
            ]
        )
        self.attention = nn.Linear(hidden_dim, 1)
        self.output_dim = hidden_dim * 3

    def forward(self, tokens: torch.Tensor, _sequence_embedding: torch.Tensor) -> torch.Tensor:
        mask = tokens.ne(PAD_ID)
        x = self.sequence_embedding(tokens).transpose(1, 2)
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        x = x.transpose(1, 2)
        mask_expanded = mask.unsqueeze(-1)
        denom = mask_expanded.sum(dim=1).clamp_min(1)
        mean_pool = (x * mask_expanded).sum(dim=1) / denom
        max_pool = x.masked_fill(~mask_expanded, -1e4).amax(dim=1)
        attention_logits = self.attention(x).squeeze(-1).masked_fill(~mask, -1e4)
        attention_weights = torch.softmax(attention_logits, dim=-1)
        attention_pool = torch.sum(x * attention_weights.unsqueeze(-1), dim=1)
        return torch.cat([mean_pool, max_pool, attention_pool], dim=-1)


class FrozenEmbeddingSequenceEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.output_dim = hidden_dim * 2

    def forward(self, _tokens: torch.Tensor, sequence_embedding: torch.Tensor) -> torch.Tensor:
        return self.adapter(sequence_embedding)


class TabularEncoder(nn.Module):
    def __init__(self, category_sizes: list[int], numeric_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.category_embeddings = nn.ModuleList([nn.Embedding(size, 16) for size in category_sizes])
        self.numeric_dim = numeric_dim
        if numeric_dim > 0:
            self.numeric_encoder = nn.Sequential(
                nn.Linear(numeric_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        else:
            self.numeric_encoder = None

        combined_dim = hidden_dim if numeric_dim > 0 else 0
        combined_dim += 16 * len(category_sizes)
        self.has_features = combined_dim > 0
        if self.has_features:
            self.output = nn.Sequential(
                nn.LayerNorm(combined_dim),
                nn.Linear(combined_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        else:
            self.output = None

    def forward(self, categories: torch.Tensor, numeric: torch.Tensor) -> torch.Tensor:
        batch_size = categories.shape[0]
        device = categories.device
        if not self.has_features:
            return torch.zeros((batch_size, self.hidden_dim), device=device)

        vectors: list[torch.Tensor] = []
        if self.numeric_encoder is not None:
            vectors.append(self.numeric_encoder(numeric))
        for idx, embedding in enumerate(self.category_embeddings):
            vectors.append(embedding(categories[:, idx]))
        combined = torch.cat(vectors, dim=-1)
        return self.output(combined)


class NeighborContextEncoder(nn.Module):
    def __init__(self, category_sizes: list[int], numeric_dim: int, hidden_dim: int, dropout: float, slot_count: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.slot_count = slot_count
        self.category_embeddings = nn.ModuleList([nn.Embedding(size, 12) for size in category_sizes])
        input_dim = numeric_dim + 12 * len(category_sizes)
        self.has_features = input_dim > 0 and slot_count > 0
        if self.has_features:
            self.input_proj = nn.Linear(input_dim, hidden_dim)
            self.position_embedding = nn.Embedding(slot_count, hidden_dim)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=4,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        else:
            self.input_proj = None
            self.position_embedding = None
            self.transformer = None

    def forward(self, neighbor_categories: torch.Tensor, neighbor_numeric: torch.Tensor, neighbor_mask: torch.Tensor) -> torch.Tensor:
        batch_size = neighbor_categories.shape[0]
        device = neighbor_categories.device
        if not self.has_features:
            return torch.zeros((batch_size, self.hidden_dim), device=device)

        vectors = [neighbor_numeric]
        for idx, embedding in enumerate(self.category_embeddings):
            vectors.append(embedding(neighbor_categories[:, :, idx]))
        x = torch.cat(vectors, dim=-1)
        x = self.input_proj(x)
        positions = torch.arange(self.slot_count, device=device)
        x = x + self.position_embedding(positions).unsqueeze(0)
        safe_padding_mask = ~neighbor_mask
        empty_rows = ~neighbor_mask.any(dim=1)
        if torch.any(empty_rows):
            safe_padding_mask = safe_padding_mask.clone()
            safe_padding_mask[empty_rows, 0] = False
        x = self.transformer(x, src_key_padding_mask=safe_padding_mask)
        x = torch.nan_to_num(x)
        mask = neighbor_mask.unsqueeze(-1).to(dtype=x.dtype)
        denom = mask.sum(dim=1).clamp_min(1)
        pooled = (x * mask).sum(dim=1) / denom
        return pooled


class TaskModeV2Model(nn.Module):
    def __init__(
        self,
        sequence_backbone: str,
        plm_embedding_dim: int,
        global_category_sizes: list[int],
        global_numeric_dim: int,
        host_category_sizes: list[int],
        host_numeric_dim: int,
        neighbor_category_sizes: list[int],
        neighbor_numeric_dim: int,
        neighbor_slot_count: int,
        biophysics_dim: int,
        num_labels: int,
        embed_dim: int,
        hidden_dim: int,
        dropout: float,
    ):
        super().__init__()
        if sequence_backbone == "precomputed_plm":
            self.sequence_encoder = FrozenEmbeddingSequenceEncoder(plm_embedding_dim, hidden_dim, dropout)
        else:
            self.sequence_encoder = CNNSequenceEncoder(len(AA_TO_ID) + 1, embed_dim, hidden_dim, dropout)

        self.global_encoder = TabularEncoder(global_category_sizes, global_numeric_dim, hidden_dim, dropout)
        self.host_encoder = TabularEncoder(host_category_sizes, host_numeric_dim, hidden_dim, dropout)
        self.neighbor_encoder = NeighborContextEncoder(neighbor_category_sizes, neighbor_numeric_dim, hidden_dim, dropout, neighbor_slot_count)

        self.biophysics_encoder = None
        seq_input_dim = self.sequence_encoder.output_dim
        if biophysics_dim > 0:
            self.biophysics_encoder = nn.Sequential(
                nn.Linear(biophysics_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            seq_input_dim += hidden_dim

        self.sequence_project = nn.Sequential(
            nn.LayerNorm(seq_input_dim),
            nn.Linear(seq_input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.context_project = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.gate = nn.Sequential(
            nn.LayerNorm(hidden_dim * 3),
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4),
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, num_labels),
        )

    def forward(
        self,
        tokens: torch.Tensor,
        sequence_embedding: torch.Tensor,
        global_categories: torch.Tensor,
        global_numeric: torch.Tensor,
        host_categories: torch.Tensor,
        host_numeric: torch.Tensor,
        biophysics: torch.Tensor,
        neighbor_categories: torch.Tensor,
        neighbor_numeric: torch.Tensor,
        neighbor_mask: torch.Tensor,
    ) -> torch.Tensor:
        seq_vec = self.sequence_encoder(tokens, sequence_embedding)
        if self.biophysics_encoder is not None:
            bio_vec = self.biophysics_encoder(biophysics)
            seq_vec = torch.cat([seq_vec, bio_vec], dim=-1)
        seq_vec = self.sequence_project(seq_vec)

        global_vec = self.global_encoder(global_categories, global_numeric)
        neighbor_vec = self.neighbor_encoder(neighbor_categories, neighbor_numeric, neighbor_mask)
        context_vec = self.context_project(torch.cat([global_vec, neighbor_vec], dim=-1))
        host_vec = self.host_encoder(host_categories, host_numeric)

        branches = torch.cat([seq_vec, context_vec, host_vec], dim=-1)
        gate_weights = torch.softmax(self.gate(branches), dim=-1)
        stacked = torch.stack([seq_vec, context_vec, host_vec], dim=1)
        fused = torch.sum(stacked * gate_weights.unsqueeze(-1), dim=1)
        classifier_input = torch.cat([seq_vec, context_vec, host_vec, fused], dim=-1)
        return self.classifier(classifier_input)


def build_cache(
    args: argparse.Namespace,
    input_path: Path,
    cache_path: Path,
    split_scheme: str,
    split_column: str | None,
    split_manifest_path: Path | None,
    context_table_path: Path | None,
    plm_embedding_path: Path | None,
    status_path: Path | None = None,
) -> dict[str, Any]:
    signature = cache_signature(
        args,
        input_path,
        split_scheme,
        split_column,
        split_manifest_path,
        context_table_path,
        plm_embedding_path,
    )
    build_started = time.time()
    if status_path is not None:
        write_status(
            status_path,
            "building_cache",
            input=str(input_path),
            cache_path=str(cache_path),
            task_mode=args.task_mode,
            split_scheme=split_scheme,
            sequence_backbone=args.sequence_backbone,
        )
    print(
        f"[cache] Building cache task_mode={args.task_mode} split_scheme={split_scheme} "
        f"input={input_path} cache={cache_path}"
    )
    selected_context_blocks = resolve_context_blocks(args.task_mode, args.context_blocks)
    feature_lists = task_mode_feature_lists(args.task_mode, args.with_biophysics, selected_context_blocks)
    global_category_fields = list(feature_lists["base_category_fields"]) + [
        field for field in feature_lists["context_category_fields"] if field not in HOST_CONTEXT_CATEGORY_FIELDS
    ]
    host_category_fields = [field for field in feature_lists["context_category_fields"] if field in HOST_CONTEXT_CATEGORY_FIELDS]
    global_numeric_fields = list(feature_lists["base_numeric_fields"]) + [
        field for field in feature_lists["context_numeric_fields"] if field not in HOST_CONTEXT_NUMERIC_FIELDS
    ]
    host_numeric_fields = [field for field in feature_lists["context_numeric_fields"] if field in HOST_CONTEXT_NUMERIC_FIELDS]
    enable_neighbor_branch = "local_neighborhood" in selected_context_blocks
    neighbor_category_fields: list[str] = []
    neighbor_numeric_fields: list[str] = []
    if enable_neighbor_branch:
        neighbor_category_fields = [NEIGHBOR_LENGTH_FIELD]
        if args.task_mode == "annotation_refinement" and "annotation_context" in selected_context_blocks:
            neighbor_category_fields.append(NEIGHBOR_FEATURE_FIELD)
        neighbor_numeric_fields = ["neighbor_offset", "neighbor_log_length", "neighbor_gap_nt", "neighbor_overlap_nt", "neighbor_same_strand", "neighbor_exists"]
        if args.task_mode == "annotation_refinement" and "annotation_context" in selected_context_blocks:
            neighbor_numeric_fields.append("neighbor_is_hypothetical")

    split_assignments = None
    if split_column and split_manifest_path:
        split_assignments = load_split_assignments(split_manifest_path, split_column)

    required_context_fields = [
        *feature_lists["context_category_fields"],
        *feature_lists["context_numeric_fields"],
    ]
    context_rows = load_context_rows(context_table_path, required_context_fields) if context_table_path else {}
    plm_mapping: dict[str, np.ndarray] | None = None
    plm_dim = 0
    if plm_embedding_path is not None:
        plm_mapping, plm_dim = load_precomputed_embeddings(plm_embedding_path)
        print(f"[cache] Loaded precomputed embeddings: {len(plm_mapping)} proteins dim={plm_dim}")

    last_progress_time = time.time()
    accepted_rows = 0

    category_vocabs = {
        "global": {field: {"__MISSING__": 0} for field in global_category_fields},
        "host": {field: {"__MISSING__": 0} for field in host_category_fields},
        "neighbor": {field: {"__PAD__": 0, "__MISSING__": 1} for field in neighbor_category_fields},
    }
    global_category_columns = {field: [] for field in global_category_fields}
    host_category_columns = {field: [] for field in host_category_fields}
    offsets = [0]
    lengths: list[int] = []
    splits: list[int] = []
    global_numeric_rows: list[list[float]] = []
    host_numeric_rows: list[list[float]] = []
    biophysics_rows: list[list[float]] = []
    label_masks: list[int] = []
    protein_accessions: list[str] = []
    virus_tax_ids: list[str] = []
    genome_versions: list[str] = []
    descriptions: list[str] = []
    label_counter: Counter[int] = Counter()
    flat_tokens = bytearray()
    split_counter: Counter[int] = Counter()
    records: list[dict[str, Any]] = []
    grouped_indices: dict[str, list[int]] = defaultdict(list)
    sequence_embedding_rows: list[np.ndarray] | None = [] if plm_mapping is not None else None

    with open_text(input_path) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row_idx, row in enumerate(reader, start=1):
            if args.debug_limit and row_idx > args.debug_limit:
                break

            sequence = row.get("protein_sequence", "").strip()
            protein_accession = row.get("protein_accession", "").strip()
            if not sequence or not protein_accession:
                continue

            if split_assignments is None:
                split_id = assign_split(row)
            else:
                split_id = split_assignments.get(protein_accession)
                if split_id is None:
                    continue
            accepted_rows += 1

            text = normalize_text(row)
            hits = label_hits(text)
            mask = 0
            for hit in hits:
                mask |= 1 << hit
                label_counter[hit] += 1

            encoded = encode_sequence(sequence, args.max_length)
            flat_tokens.extend(encoded)
            offsets.append(len(flat_tokens))
            lengths.append(len(encoded))
            splits.append(split_id)
            split_counter[split_id] += 1
            label_masks.append(mask)

            descriptions.append(row.get("cds_product", "").strip() or row.get("protein_description", "").strip())
            protein_accessions.append(protein_accession)
            virus_tax_ids.append(row.get("virus_tax_id", "").strip())
            genome_versions.append(row.get("genome_version", "").strip())

            context_row = context_rows.get(protein_accession, {})
            refinement_numeric = refinement_numeric_features(row, text)
            global_numeric_rows.append(
                [float(refinement_numeric[field]) for field in feature_lists["base_numeric_fields"]]
                + [float(context_row.get(field, "0") or 0.0) for field in global_numeric_fields[len(feature_lists["base_numeric_fields"]) :]]
            )
            host_numeric_rows.append([float(context_row.get(field, "0") or 0.0) for field in host_numeric_fields])
            biophysics_rows.append([compute_biophysics(sequence)[field] for field in feature_lists["biophysics_fields"]])

            for field in global_category_fields:
                if field in row:
                    global_category_columns[field].append(register_category(category_vocabs["global"][field], row.get(field, "")))
                else:
                    global_category_columns[field].append(register_category(category_vocabs["global"][field], context_row.get(field, "")))
            for field in host_category_fields:
                host_category_columns[field].append(register_category(category_vocabs["host"][field], context_row.get(field, "")))

            if sequence_embedding_rows is not None:
                vector = plm_mapping.get(protein_accession)
                if vector is None:
                    raise RuntimeError(f"Missing precomputed PLM embedding for protein '{protein_accession}'")
                sequence_embedding_rows.append(vector.astype(np.float16, copy=False))

            record = {
                "protein_accession": protein_accession,
                "genome_key": row.get("genome_version", "").strip()
                or row.get("genome_accession", "").strip()
                or row.get("virus_tax_id", "").strip()
                or protein_accession,
                "virus_family": derive_virus_family(row.get("virus_lineage", "")),
                "cds_start": maybe_int(row.get("cds_start", "0")),
                "cds_end": maybe_int(row.get("cds_end", "0")),
                "cds_strand": row.get("cds_strand", "").strip(),
                "protein_length_aa": maybe_int(row.get("protein_length_aa", "0")),
                "length_bin": context_row.get("context_prev_length_bin", "") or f"len_{maybe_int(row.get('protein_length_aa', '0'))}",
                "feature_type": row.get("protein_feature_type", "").strip() or "__MISSING__",
                "is_hypothetical": refinement_numeric["is_hypothetical"] > 0.0,
                "row_order": row_idx,
            }
            records.append(record)
            grouped_indices[record["genome_key"]].append(len(records) - 1)

            now = time.time()
            if accepted_rows % CACHE_PROGRESS_EVERY_ROWS == 0 or (now - last_progress_time) >= CACHE_PROGRESS_EVERY_SECONDS:
                elapsed = max(now - build_started, 1e-6)
                print(
                    f"[cache] rows={accepted_rows} source_rows={row_idx} "
                    f"tokens={len(flat_tokens)} genomes={len(grouped_indices)} elapsed={elapsed:.1f}s"
                )
                if status_path is not None:
                    write_status(
                        status_path,
                        "building_cache",
                        rows_loaded=accepted_rows,
                        source_rows_read=row_idx,
                        token_count=len(flat_tokens),
                        genome_groups=len(grouped_indices),
                        elapsed_seconds=elapsed,
                        task_mode=args.task_mode,
                        split_scheme=split_scheme,
                    )
                last_progress_time = now

    control_rng = np.random.default_rng(args.seed + 17)
    host_corrupted_count = 0
    if args.context_control == "shuffle_host_within_family":
        apply_host_shuffle_within_family(records, host_category_columns, host_numeric_rows, control_rng)
    if args.host_corruption_fraction > 0.0:
        host_corrupted_count = apply_host_corruption(
            records,
            host_category_columns,
            host_numeric_rows,
            args.host_corruption_fraction,
            control_rng,
        )

    active_label_indices = [idx for idx, _ in enumerate(LABEL_RULES) if label_counter[idx] >= args.min_label_count]
    if not active_label_indices:
        active_label_indices = [idx for idx, _ in enumerate(LABEL_RULES) if label_counter[idx] > 0]
    label_names = [LABEL_RULES[idx].name for idx in active_label_indices]
    label_matrix = np.zeros((len(label_masks), len(active_label_indices)), dtype=np.uint8)
    for row_idx, mask in enumerate(label_masks):
        for col_idx, label_idx in enumerate(active_label_indices):
            if mask & (1 << label_idx):
                label_matrix[row_idx, col_idx] = 1

    slot_offsets = [offset for offset in range(-args.neighbor_radius, args.neighbor_radius + 1) if offset != 0]
    neighbor_numeric_rows: list[np.ndarray] = [np.zeros((len(slot_offsets), len(neighbor_numeric_fields)), dtype=np.float32) for _ in records]
    neighbor_mask_rows: list[np.ndarray] = [np.zeros((len(slot_offsets),), dtype=np.bool_) for _ in records]
    neighbor_category_columns = {
        field: [np.zeros((len(slot_offsets),), dtype=np.int32) for _ in records] for field in neighbor_category_fields
    }

    def sort_key(index: int):
        record = records[index]
        start = int(record["cds_start"])
        end = int(record["cds_end"])
        return (
            0 if start > 0 else 1,
            start if start > 0 else 10**12,
            end if end > 0 else 10**12,
            int(record["row_order"]),
        )

    if enable_neighbor_branch:
        print(f"[cache] Building neighbor features for {len(grouped_indices)} genome groups")
        for indices in grouped_indices.values():
            ordered = sorted(indices, key=sort_key)
            index_to_pos = {record_idx: pos for pos, record_idx in enumerate(ordered)}
            for record_idx in ordered:
                position = index_to_pos[record_idx]
                center = records[record_idx]
                for slot_idx, offset in enumerate(slot_offsets):
                    neighbor_pos = position + offset
                    if neighbor_pos < 0 or neighbor_pos >= len(ordered):
                        for field in neighbor_category_fields:
                            neighbor_category_columns[field][record_idx][slot_idx] = register_category(category_vocabs["neighbor"][field], "__PAD__")
                        continue

                    neighbor_idx = ordered[neighbor_pos]
                    neighbor = records[neighbor_idx]
                    same_strand = 1.0 if neighbor["cds_strand"] and neighbor["cds_strand"] == center["cds_strand"] else 0.0
                    if offset < 0:
                        gap_nt = max(0, int(center["cds_start"]) - int(neighbor["cds_end"]) - 1)
                        overlap_nt = max(0, int(neighbor["cds_end"]) - int(center["cds_start"]) + 1)
                    else:
                        gap_nt = max(0, int(neighbor["cds_start"]) - int(center["cds_end"]) - 1)
                        overlap_nt = max(0, int(center["cds_end"]) - int(neighbor["cds_start"]) + 1)
                    values = [
                        float(offset) / max(1, args.neighbor_radius),
                        safe_log1p(int(neighbor["protein_length_aa"])),
                        safe_log1p(gap_nt),
                        safe_log1p(overlap_nt),
                        same_strand,
                        1.0,
                    ]
                    if args.task_mode == "annotation_refinement" and "annotation_context" in selected_context_blocks:
                        values.append(1.0 if neighbor["is_hypothetical"] else 0.0)
                    neighbor_numeric_rows[record_idx][slot_idx] = np.asarray(values, dtype=np.float32)
                    neighbor_mask_rows[record_idx][slot_idx] = True
                    neighbor_category_columns[NEIGHBOR_LENGTH_FIELD][record_idx][slot_idx] = register_category(
                        category_vocabs["neighbor"][NEIGHBOR_LENGTH_FIELD],
                        f"len2^{int(np.log2(max(1, int(neighbor['protein_length_aa']))))}-{int(np.log2(max(1, int(neighbor['protein_length_aa'])))) + 1}",
                    )
                    if NEIGHBOR_FEATURE_FIELD in neighbor_category_columns:
                        neighbor_category_columns[NEIGHBOR_FEATURE_FIELD][record_idx][slot_idx] = register_category(
                            category_vocabs["neighbor"][NEIGHBOR_FEATURE_FIELD],
                            str(neighbor["feature_type"]),
                        )

    if args.context_control == "shuffle_genome_relative_position":
        apply_genome_relative_position_shuffle(grouped_indices, global_numeric_rows, global_numeric_fields, control_rng)
    if args.context_control == "shuffle_local_order":
        apply_local_order_shuffle(
            global_category_columns,
            global_numeric_rows,
            global_category_fields,
            global_numeric_fields,
            neighbor_category_columns,
            neighbor_numeric_rows,
            neighbor_mask_rows,
            control_rng,
        )

    global_categories = np.stack(
        [np.asarray(global_category_columns[field], dtype=np.int32) for field in global_category_fields],
        axis=1,
    ) if global_category_fields else np.zeros((len(records), 0), dtype=np.int32)
    host_categories = np.stack(
        [np.asarray(host_category_columns[field], dtype=np.int32) for field in host_category_fields],
        axis=1,
    ) if host_category_fields else np.zeros((len(records), 0), dtype=np.int32)
    neighbor_categories = np.stack(
        [np.stack(neighbor_category_columns[field]).astype(np.int32) for field in neighbor_category_fields],
        axis=2,
    ) if neighbor_category_fields else np.zeros((len(records), len(slot_offsets), 0), dtype=np.int32)

    cache = {
        **signature,
        "flat_tokens": np.frombuffer(bytes(flat_tokens), dtype=np.uint8),
        "offsets": np.asarray(offsets, dtype=np.int64),
        "lengths": np.asarray(lengths, dtype=np.int32),
        "splits": np.asarray(splits, dtype=np.int8),
        "labels": label_matrix,
        "global_numeric": np.asarray(global_numeric_rows, dtype=np.float32) if global_numeric_rows else np.zeros((len(records), 0), dtype=np.float32),
        "host_numeric": np.asarray(host_numeric_rows, dtype=np.float32) if host_numeric_rows else np.zeros((len(records), 0), dtype=np.float32),
        "biophysics": np.asarray(biophysics_rows, dtype=np.float32) if biophysics_rows else np.zeros((len(records), 0), dtype=np.float32),
        "global_categories": global_categories,
        "host_categories": host_categories,
        "neighbor_numeric": np.stack(neighbor_numeric_rows).astype(np.float32),
        "neighbor_categories": neighbor_categories,
        "neighbor_mask": np.stack(neighbor_mask_rows).astype(np.bool_),
        "sequence_embeddings": np.stack(sequence_embedding_rows).astype(np.float16) if sequence_embedding_rows is not None else None,
        "global_category_fields": global_category_fields,
        "host_category_fields": host_category_fields,
        "global_numeric_fields": global_numeric_fields,
        "host_numeric_fields": host_numeric_fields,
        "biophysics_fields": feature_lists["biophysics_fields"],
        "selected_context_blocks": list(selected_context_blocks),
        "context_control": args.context_control,
        "host_corruption_fraction": float(args.host_corruption_fraction),
        "host_corrupted_count": int(host_corrupted_count),
        "neighbor_category_fields": neighbor_category_fields,
        "neighbor_numeric_fields": neighbor_numeric_fields,
        "category_vocabs": category_vocabs,
        "label_names": label_names,
        "label_support": [label_counter[idx] for idx in active_label_indices],
        "protein_accessions": protein_accessions,
        "virus_tax_ids": virus_tax_ids,
        "genome_versions": genome_versions,
        "descriptions": descriptions,
        "max_length": args.max_length,
        "created_at": timestamp(),
        "split_counts": {str(split_id): int(count) for split_id, count in split_counter.items()},
        "plm_embedding_dim": plm_dim,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, cache_path)
    elapsed = time.time() - build_started
    print(
        f"[cache] Saved {len(records)} examples, {len(label_names)} labels, {len(flat_tokens)} tokens "
        f"to {cache_path} in {elapsed:.1f}s"
    )
    if status_path is not None:
        write_status(
            status_path,
            "cache_ready",
            rows_loaded=len(records),
            label_count=len(label_names),
            token_count=len(flat_tokens),
            cache_path=str(cache_path),
            elapsed_seconds=elapsed,
            task_mode=args.task_mode,
            split_scheme=split_scheme,
        )
    return cache


def load_or_build_cache(
    args: argparse.Namespace,
    input_path: Path,
    cache_path: Path,
    split_scheme: str,
    split_column: str | None,
    split_manifest_path: Path | None,
    context_table_path: Path | None,
    plm_embedding_path: Path | None,
    status_path: Path | None = None,
) -> dict[str, Any]:
    signature = cache_signature(
        args,
        input_path,
        split_scheme,
        split_column,
        split_manifest_path,
        context_table_path,
        plm_embedding_path,
    )
    if cache_path.exists() and not args.force_rebuild_cache:
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        if cache_matches_request(cache, signature):
            print(f"[cache] Reusing existing cache from {cache_path}")
            if status_path is not None:
                write_status(
                    status_path,
                    "cache_ready",
                    cache_path=str(cache_path),
                    rows_loaded=int(cache["labels"].shape[0]),
                    label_count=len(cache["label_names"]),
                    reused_existing_cache=True,
                    task_mode=args.task_mode,
                    split_scheme=split_scheme,
                )
            return cache
    return build_cache(
        args,
        input_path,
        cache_path,
        split_scheme,
        split_column,
        split_manifest_path,
        context_table_path,
        plm_embedding_path,
        status_path=status_path,
    )


@torch.no_grad()
def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    autocast_dtype: torch.dtype | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    all_probs: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    all_indices: list[np.ndarray] = []
    use_autocast = autocast_dtype is not None and device.type == "cuda"
    for batch in loader:
        with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=use_autocast):
            logits = model(
                batch["tokens"].to(device, non_blocking=True),
                batch["sequence_embedding"].to(device, non_blocking=True),
                batch["global_categories"].to(device, non_blocking=True),
                batch["global_numeric"].to(device, non_blocking=True),
                batch["host_categories"].to(device, non_blocking=True),
                batch["host_numeric"].to(device, non_blocking=True),
                batch["biophysics"].to(device, non_blocking=True),
                batch["neighbor_categories"].to(device, non_blocking=True),
                batch["neighbor_numeric"].to(device, non_blocking=True),
                batch["neighbor_mask"].to(device, non_blocking=True),
            )
        probs = torch.sigmoid(logits).to(dtype=torch.float32).cpu().numpy()
        all_probs.append(probs)
        all_targets.append(batch["labels"].numpy())
        all_indices.append(batch["indices"].numpy())
    return np.concatenate(all_probs), np.concatenate(all_targets), np.concatenate(all_indices)


def main() -> int:
    args = parse_args()
    if not 0.0 <= float(args.host_corruption_fraction) <= 1.0:
        raise ValueError("--host-corruption-fraction must be between 0 and 1")
    resolve_context_blocks(args.task_mode, args.context_blocks)
    device = choose_device(args.device)
    set_seed(args.seed, device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    root = repo_root()
    input_path = (root / args.input).resolve()
    output_dir = (root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "run_status.json"
    cache_path = (Path(args.cache_path) if args.cache_path else output_dir / "dataset_cache.pt")
    if not cache_path.is_absolute():
        cache_path = (root / cache_path).resolve()

    split_scheme, split_column, split_manifest_path = resolve_split_config(root, args)
    context_table_path = resolve_context_table_path(root, args)
    plm_embedding_path = resolve_plm_embedding_path(root, args)
    write_status(
        status_path,
        "initializing",
        output_dir=str(output_dir),
        cache_path=str(cache_path),
        task_mode=args.task_mode,
        split_scheme=split_scheme,
        sequence_backbone=args.sequence_backbone,
    )

    cache = load_or_build_cache(
        args,
        input_path,
        cache_path,
        split_scheme,
        split_column,
        split_manifest_path,
        context_table_path,
        plm_embedding_path,
        status_path,
    )

    splits = cache["splits"]
    train_idx = np.where(splits == 0)[0]
    val_idx = np.where(splits == 1)[0]
    test_idx = np.where(splits == 2)[0]
    if train_idx.size == 0 or val_idx.size == 0 or test_idx.size == 0:
        raise RuntimeError(f"Split scheme '{split_scheme}' produced empty partitions.")

    train_dataset = TaskModeDataset(cache, train_idx)
    val_dataset = TaskModeDataset(cache, val_idx)
    test_dataset = TaskModeDataset(cache, test_idx)

    pin_memory = device.type == "cuda"
    print(
        f"[train] device={device} train={train_idx.shape[0]} val={val_idx.shape[0]} "
        f"test={test_idx.shape[0]} labels={len(cache['label_names'])} batch_size={args.batch_size}"
    )
    write_status(
        status_path,
        "training",
        device=str(device),
        train_size=int(train_idx.shape[0]),
        val_size=int(val_idx.shape[0]),
        test_size=int(test_idx.shape[0]),
        label_count=len(cache["label_names"]),
        task_mode=args.task_mode,
        split_scheme=split_scheme,
    )
    train_loader = make_dataloader(train_dataset, args.batch_size, True, args.num_workers, args.prefetch_factor, pin_memory)
    eval_num_workers = max(0, min(args.num_workers, 4))
    val_loader = make_dataloader(val_dataset, args.eval_batch_size, False, eval_num_workers, args.prefetch_factor, pin_memory)
    test_loader = make_dataloader(test_dataset, args.eval_batch_size, False, eval_num_workers, args.prefetch_factor, pin_memory)

    model = TaskModeV2Model(
        sequence_backbone=args.sequence_backbone,
        plm_embedding_dim=int(cache.get("plm_embedding_dim", 0)),
        global_category_sizes=[int(cache["global_categories"][:, idx].max()) + 1 for idx in range(cache["global_categories"].shape[1])],
        global_numeric_dim=int(cache["global_numeric"].shape[1]),
        host_category_sizes=[int(cache["host_categories"][:, idx].max()) + 1 for idx in range(cache["host_categories"].shape[1])],
        host_numeric_dim=int(cache["host_numeric"].shape[1]),
        neighbor_category_sizes=[int(cache["neighbor_categories"][:, :, idx].max()) + 1 for idx in range(cache["neighbor_categories"].shape[2])],
        neighbor_numeric_dim=int(cache["neighbor_numeric"].shape[2]),
        neighbor_slot_count=int(cache["neighbor_numeric"].shape[1]),
        biophysics_dim=int(cache["biophysics"].shape[1]),
        num_labels=len(cache["label_names"]),
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)

    if args.compile_model and hasattr(torch, "compile"):
        model = torch.compile(model)  # type: ignore[assignment]

    train_labels = cache["labels"][train_idx]
    pos_weight = compute_pos_weight(train_labels, args.max_pos_weight).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = args.epochs * max(1, len(train_loader))
    warmup_steps = int(total_steps * args.warmup_fraction)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: linear_warmup_cosine_decay(step, total_steps, warmup_steps))

    use_amp = device.type == "cuda"
    autocast_dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp and autocast_dtype == torch.float16)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp and autocast_dtype == torch.float16)

    run_manifest = {
        "created_at": timestamp(),
        "input": str(input_path),
        "cache_path": str(cache_path),
        "output_dir": str(output_dir),
        "device": str(device),
        "torch_version": torch.__version__,
        "config": vars(args),
        "label_names": cache["label_names"],
        "label_support": cache["label_support"],
        "task_mode": args.task_mode,
        "selected_context_blocks": cache.get("selected_context_blocks", []),
        "context_control": cache.get("context_control", "none"),
        "host_corruption_fraction": cache.get("host_corruption_fraction", 0.0),
        "host_corrupted_count": cache.get("host_corrupted_count", 0),
        "sequence_backbone": args.sequence_backbone,
        "global_category_fields": cache["global_category_fields"],
        "global_numeric_fields": cache["global_numeric_fields"],
        "host_category_fields": cache["host_category_fields"],
        "host_numeric_fields": cache["host_numeric_fields"],
        "biophysics_fields": cache["biophysics_fields"],
        "neighbor_category_fields": cache["neighbor_category_fields"],
        "neighbor_numeric_fields": cache["neighbor_numeric_fields"],
        "context_table_path": str(context_table_path) if context_table_path else "",
        "plm_embedding_path": str(plm_embedding_path) if plm_embedding_path else "",
        "split_strategy": {
            "scheme": split_scheme,
            "column": split_column or "",
            "manifest_path": str(split_manifest_path) if split_manifest_path else "",
            "counts": cache.get("split_counts", {}),
        },
        "split_sizes": {"train": int(train_idx.shape[0]), "val": int(val_idx.shape[0]), "test": int(test_idx.shape[0])},
        "seed": int(args.seed),
        **collect_git_metadata(root),
    }
    save_json(output_dir / "run_manifest.json", run_manifest)

    best_val_score = -1.0
    best_checkpoint_path = output_dir / "best_model.pt"
    history: list[dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        seen = 0
        for step, batch in enumerate(train_loader, start=1):
            optimizer.zero_grad(set_to_none=True)
            batch_tensors = {key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
            with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=use_amp):
                logits = model(
                    batch_tensors["tokens"],
                    batch_tensors["sequence_embedding"],
                    batch_tensors["global_categories"],
                    batch_tensors["global_numeric"],
                    batch_tensors["host_categories"],
                    batch_tensors["host_numeric"],
                    batch_tensors["biophysics"],
                    batch_tensors["neighbor_categories"],
                    batch_tensors["neighbor_numeric"],
                    batch_tensors["neighbor_mask"],
                )
                loss = criterion(logits, batch_tensors["labels"])

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
                optimizer.step()

            scheduler.step()
            batch_size = int(batch_tensors["tokens"].shape[0])
            running_loss += float(loss.item()) * batch_size
            seen += batch_size

        train_loss = running_loss / max(1, seen)
        y_val_prob, y_val_true, _ = predict(model, val_loader, device, autocast_dtype)
        thresholds = optimize_thresholds(y_val_true, y_val_prob)
        val_metrics = compute_metrics(y_val_true, y_val_prob, thresholds, cache["label_names"])
        history_entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_macro_average_precision": val_metrics["macro_average_precision"],
            "val_micro_average_precision": val_metrics["micro_average_precision"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_micro_f1": val_metrics["micro_f1"],
            "epoch_seconds": time.time() - epoch_start,
        }
        history.append(history_entry)
        print(
            f"[train] epoch={epoch} train_loss={train_loss:.4f} "
            f"val_macro_ap={val_metrics['macro_average_precision']:.4f} "
            f"val_macro_f1={val_metrics['macro_f1']:.4f} "
            f"epoch_seconds={history_entry['epoch_seconds']:.1f}"
        )
        write_status(
            status_path,
            "training",
            current_epoch=epoch,
            train_loss=train_loss,
            val_macro_average_precision=val_metrics["macro_average_precision"],
            val_micro_average_precision=val_metrics["micro_average_precision"],
            val_macro_f1=val_metrics["macro_f1"],
            val_micro_f1=val_metrics["micro_f1"],
            elapsed_epoch_seconds=history_entry["epoch_seconds"],
            best_val_macro_average_precision=best_val_score,
            task_mode=args.task_mode,
            split_scheme=split_scheme,
        )

        if val_metrics["macro_average_precision"] > best_val_score:
            best_val_score = val_metrics["macro_average_precision"]
            torch.save({"model_state": model.state_dict()}, best_checkpoint_path)
            save_json(output_dir / "best_thresholds.json", {"thresholds": dict(zip(cache["label_names"], thresholds.tolist()))})
            save_json(
                output_dir / "metrics_summary.json",
                {
                    "created_at": timestamp(),
                    "best_checkpoint": str(best_checkpoint_path),
                    "best_epoch": epoch,
                    "thresholds": dict(zip(cache["label_names"], thresholds.tolist())),
                    "validation": val_metrics,
                },
            )

    checkpoint = torch.load(best_checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    thresholds_payload = json.loads((output_dir / "best_thresholds.json").read_text(encoding="utf-8"))
    thresholds = np.asarray([float(thresholds_payload["thresholds"][name]) for name in cache["label_names"]], dtype=np.float32)
    y_val_prob, y_val_true, _ = predict(model, val_loader, device, autocast_dtype)
    y_test_prob, y_test_true, test_indices = predict(model, test_loader, device, autocast_dtype)
    val_metrics = compute_metrics(y_val_true, y_val_prob, thresholds, cache["label_names"])
    test_metrics = compute_metrics(y_test_true, y_test_prob, thresholds, cache["label_names"])

    summary = {
        "created_at": timestamp(),
        "best_checkpoint": str(best_checkpoint_path),
        "best_epoch": int(np.argmax([row["val_macro_average_precision"] for row in history]) + 1),
        "thresholds": dict(zip(cache["label_names"], thresholds.tolist())),
        "validation": val_metrics,
        "test": test_metrics,
    }
    save_json(output_dir / "metrics_summary.json", summary)
    save_history(output_dir / "history.jsonl", history)
    save_label_metrics(output_dir / "val_label_metrics.tsv", val_metrics["label_metrics"])
    save_label_metrics(output_dir / "test_label_metrics.tsv", test_metrics["label_metrics"])
    write_status(
        status_path,
        "completed",
        best_epoch=summary["best_epoch"],
        test_macro_average_precision=test_metrics["macro_average_precision"],
        test_micro_average_precision=test_metrics["micro_average_precision"],
        test_macro_f1=test_metrics["macro_f1"],
        test_micro_f1=test_metrics["micro_f1"],
        output_dir=str(output_dir),
        task_mode=args.task_mode,
        split_scheme=split_scheme,
    )
    if args.save_test_predictions:
        save_test_predictions(
            output_dir / "test_predictions.tsv.gz",
            {
                "protein_accessions": cache["protein_accessions"],
                "virus_tax_ids": cache["virus_tax_ids"],
                "genome_versions": cache["genome_versions"],
                "descriptions": cache["descriptions"],
            },
            test_indices,
            y_test_true,
            y_test_prob,
            thresholds,
            cache["label_names"],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
