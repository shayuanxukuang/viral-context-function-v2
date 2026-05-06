from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import random
import re
import subprocess
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset

from context_features import CONTEXT_CATEGORY_FIELDS, CONTEXT_NUMERIC_PREFIX_FIELDS

AA_VOCAB = "ACDEFGHIKLMNPQRSTVWYBXZJUO*"
AA_TO_ID = {aa: idx + 1 for idx, aa in enumerate(AA_VOCAB)}
PAD_ID = 0

BASE_CATEGORY_FIELDS = [
    "protein_feature_type",
    "source_mol_type",
    "division",
    "host_join_strategy",
]

CATEGORY_FIELDS = BASE_CATEGORY_FIELDS

BASE_NUMERIC_FIELD_NAMES = [
    "log_protein_length",
    "log_host_record_count",
    "log_uniprot_entries",
    "log_uniprot_go_entries",
    "log_uniprot_interpro_entries",
    "log_uniprot_ec_entries",
    "is_hypothetical",
    "is_mat_peptide",
]

NUMERIC_FIELD_NAMES = BASE_NUMERIC_FIELD_NAMES

SPLIT_ID_TO_NAME = {
    0: "train",
    1: "val",
    2: "test",
}

SPLIT_NAME_TO_ID = {name: split_id for split_id, name in SPLIT_ID_TO_NAME.items()}

SPLIT_SCHEME_TO_COLUMN = {
    "default_hash": None,
    "species_holdout": "species_holdout_split",
    "family_holdout": "family_holdout_split",
    "host_holdout": "host_taxid_holdout_split",
    "host_taxid_holdout": "host_taxid_holdout_split",
    "host_supergroup_holdout": "host_supergroup_holdout_split",
    "sequence_sketch_holdout": "sequence_sketch_holdout_split",
}


@dataclass(frozen=True)
class LabelRule:
    name: str
    patterns: tuple[str, ...]
    description: str


LABEL_RULES = [
    LabelRule(
        name="polymerase",
        patterns=(
            r"rna-dependent rna polymerase",
            r"dna polymerase",
            r"\bpolymerase\b",
            r"\breplicase\b",
            r"reverse transcriptase",
            r"\btranscriptase\b",
        ),
        description="Polymerase and replicase machinery",
    ),
    LabelRule(name="helicase", patterns=(r"\bhelicase\b",), description="Helicases and unwinding proteins"),
    LabelRule(name="protease", patterns=(r"\bprotease\b", r"\bproteinase\b"), description="Proteases and processing enzymes"),
    LabelRule(
        name="capsid_head",
        patterns=(r"\bcapsid\b", r"\bcoat protein\b", r"\bhead protein\b", r"\bmajor head protein\b"),
        description="Capsid and head structural proteins",
    ),
    LabelRule(
        name="tail_fiber_receptor",
        patterns=(r"tail fiber", r"tail spike", r"baseplate", r"receptor binding"),
        description="Tail fiber, tail spike, and receptor-binding proteins",
    ),
    LabelRule(
        name="tail_assembly",
        patterns=(r"\btail\b", r"tape measure", r"tail assembly"),
        description="Tail assembly and morphogenesis proteins",
    ),
    LabelRule(
        name="portal_terminase_packaging",
        patterns=(r"portal protein", r"terminase", r"packaging"),
        description="Portal, terminase, and genome packaging proteins",
    ),
    LabelRule(
        name="lysis",
        patterns=(r"endolysin", r"\blysin\b", r"\bholin\b", r"\bspanin\b", r"\blysozyme\b"),
        description="Lysis and cell wall disruption proteins",
    ),
    LabelRule(
        name="envelope_glycoprotein",
        patterns=(r"glycoprotein", r"envelope protein", r"spike protein"),
        description="Envelope and glycoprotein structural proteins",
    ),
    LabelRule(
        name="membrane_matrix",
        patterns=(r"membrane protein", r"matrix protein", r"\bmembrane\b"),
        description="Membrane and matrix-associated proteins",
    ),
    LabelRule(name="nucleocapsid", patterns=(r"nucleocapsid",), description="Nucleocapsid proteins"),
    LabelRule(
        name="integrase_recombinase",
        patterns=(r"\bintegrase\b", r"\brecombinase\b"),
        description="Integrases and recombinases",
    ),
    LabelRule(
        name="nuclease",
        patterns=(r"\bnuclease\b", r"endonuclease", r"exonuclease"),
        description="Nucleases and nucleic acid processing proteins",
    ),
    LabelRule(name="methyltransferase", patterns=(r"methyltransferase",), description="Methyltransferases"),
    LabelRule(name="ligase", patterns=(r"\bligase\b",), description="Ligases"),
    LabelRule(
        name="transcription_regulator",
        patterns=(r"transcriptional regulator", r"transcription regulator", r"transactivator"),
        description="Transcriptional regulators and activators",
    ),
    LabelRule(name="polyprotein", patterns=(r"\bpolyprotein\b",), description="Polyprotein precursors"),
]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def collect_git_metadata(root: Path) -> dict[str, Any]:
    metadata = {
        "git_commit": "unrecorded",
        "git_branch": "",
        "git_dirty": None,
        "git_available": False,
    }
    try:
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        branch = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--abbrev-ref", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty_proc = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return metadata

    metadata["git_available"] = True
    metadata["git_commit"] = commit or "unrecorded"
    metadata["git_branch"] = branch
    metadata["git_dirty"] = bool(dirty_proc.stdout.strip())
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an overnight ViruFunc-FM baseline on the processed index.")
    parser.add_argument("--input", default="data/processed/training/viral_protein_training_index.tsv.gz", help="Protein-level training index")
    parser.add_argument("--output-dir", default="runs/overnight_baseline", help="Directory for checkpoints, reports, and predictions")
    parser.add_argument("--cache-path", default="", help="Optional preprocessing cache path. Defaults to <output-dir>/dataset_cache.pt")
    parser.add_argument(
        "--split-manifest",
        default="",
        help="Optional strict split manifest. Defaults to data/processed/splits/viral_protein_strict_splits.tsv.gz when split-scheme is not default_hash",
    )
    parser.add_argument(
        "--split-scheme",
        default="default_hash",
        choices=sorted(SPLIT_SCHEME_TO_COLUMN),
        help="Split strategy: default_hash or one of the strict holdout schemes",
    )
    parser.add_argument(
        "--context-table",
        default="",
        help="Optional per-protein context feature table. When set, context numeric/category features are appended to the cache.",
    )
    parser.add_argument("--max-length", type=int, default=2048, help="Maximum sequence length after head-tail truncation")
    parser.add_argument("--epochs", type=int, default=12, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=1024, help="Training batch size")
    parser.add_argument("--eval-batch-size", type=int, default=2048, help="Evaluation batch size")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="AdamW learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-2, help="AdamW weight decay")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")
    parser.add_argument("--embed-dim", type=int, default=128, help="Sequence embedding size")
    parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden channel size")
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


def set_seed(seed: int, device: torch.device | None = None) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device is not None and device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

def visible_cuda_devices_from_env() -> list[str]:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not raw:
        return []
    return [token.strip() for token in raw.split(",") if token.strip()]


def normalize_requested_device(requested: str) -> str:
    value = requested.strip().lower()
    if value == "auto":
        return value
    if value == "cuda":
        return "cuda:0"
    return value


def try_cuda_device(device_index: int) -> tuple[bool, str | None]:
    try:
        device_count = int(torch.cuda.device_count())
        if device_count <= 0:
            return False, "torch.cuda.device_count() returned 0"
        if device_index < 0 or device_index >= device_count:
            return False, f"requested logical device {device_index} but only {device_count} CUDA device(s) are visible"
        torch.cuda.set_device(device_index)
        _ = torch.cuda.current_device()
        return True, None
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def choose_device(requested: str) -> torch.device:
    normalized = normalize_requested_device(requested)
    if normalized == "cpu":
        return torch.device("cpu")

    if normalized != "auto":
        device = torch.device(normalized)
        if device.type != "cuda":
            return device
        device_index = 0 if device.index is None else int(device.index)
        ok, reason = try_cuda_device(device_index)
        if not ok:
            visible = ",".join(visible_cuda_devices_from_env()) or "<all>"
            raise RuntimeError(
                f"Unable to initialize CUDA device '{normalized}'. "
                f"CUDA_VISIBLE_DEVICES={visible}. {reason}"
            )
        return torch.device(f"cuda:{device_index}")

    ok, reason = try_cuda_device(0)
    if ok:
        return torch.device("cuda:0")

    visible = ",".join(visible_cuda_devices_from_env()) or "<all>"
    print(
        f"[device] Falling back to CPU because CUDA auto-detection failed. "
        f"CUDA_VISIBLE_DEVICES={visible}. {reason}"
    )
    return torch.device("cpu")


def hash_bucket(value: str) -> int:
    digest = hashlib.md5(value.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 100


def assign_split(row: dict[str, str]) -> int:
    key = row.get("virus_tax_id", "").strip() or row.get("genome_version", "").strip() or row.get("protein_accession", "").strip()
    bucket = hash_bucket(key)
    if bucket < 80:
        return 0
    if bucket < 90:
        return 1
    return 2


def normalize_text(row: dict[str, str]) -> str:
    product = row.get("cds_product", "").strip()
    description = row.get("protein_description", "").strip()
    return " ".join(part for part in [product, description] if part).lower()


def label_hits(text: str) -> list[int]:
    hits: list[int] = []
    for idx, rule in enumerate(LABEL_RULES):
        if any(re.search(pattern, text) for pattern in rule.patterns):
            hits.append(idx)
    return hits


def encode_sequence(sequence: str, max_length: int) -> bytes:
    sequence = sequence.strip().upper()
    if len(sequence) > max_length:
        head = max_length // 2
        tail = max_length - head
        sequence = sequence[:head] + sequence[-tail:]
    encoded = bytearray(len(sequence))
    for idx, aa in enumerate(sequence):
        encoded[idx] = AA_TO_ID.get(aa, AA_TO_ID["X"])
    return bytes(encoded)


def maybe_int(value: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def safe_log1p(value: int | float) -> float:
    return math.log1p(max(float(value), 0.0))


def context_numeric_field_names() -> list[str]:
    fields = list(CONTEXT_NUMERIC_PREFIX_FIELDS)
    fields.extend(f"context_genome_{rule.name}_fraction" for rule in LABEL_RULES)
    fields.extend(f"context_local_{rule.name}_count" for rule in LABEL_RULES)
    return fields


def feature_schema(context_table_path: Path | None) -> tuple[list[str], list[str]]:
    category_fields = list(BASE_CATEGORY_FIELDS)
    numeric_field_names = list(BASE_NUMERIC_FIELD_NAMES)
    if context_table_path is not None:
        category_fields.extend(CONTEXT_CATEGORY_FIELDS)
        numeric_field_names.extend(context_numeric_field_names())
    return category_fields, numeric_field_names


def build_numeric_features(
    row: dict[str, str],
    text: str,
    numeric_field_names: list[str],
    context_row: dict[str, str] | None = None,
) -> list[float]:
    feature_type = row.get("protein_feature_type", "").strip()
    values = [
        safe_log1p(maybe_int(row.get("protein_length_aa", "0"))),
        safe_log1p(maybe_int(row.get("host_record_count", "0"))),
        safe_log1p(maybe_int(row.get("reviewed_uniprot_entries_for_taxon", "0"))),
        safe_log1p(maybe_int(row.get("reviewed_uniprot_entries_with_go_for_taxon", "0"))),
        safe_log1p(maybe_int(row.get("reviewed_uniprot_entries_with_interpro_for_taxon", "0"))),
        safe_log1p(maybe_int(row.get("reviewed_uniprot_entries_with_ec_for_taxon", "0"))),
        1.0 if ("hypothetical protein" in text or "uncharacterized" in text or "unknown protein" in text) else 0.0,
        1.0 if feature_type == "mat_peptide" else 0.0,
    ]
    if context_row is not None:
        for field in numeric_field_names[len(BASE_NUMERIC_FIELD_NAMES) :]:
            try:
                values.append(float(context_row.get(field, "0") or 0.0))
            except ValueError:
                values.append(0.0)
    return values


def register_category(vocab: dict[str, int], value: str) -> int:
    normalized = value.strip() if value else "__MISSING__"
    if normalized not in vocab:
        vocab[normalized] = len(vocab)
    return vocab[normalized]


def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def resolve_context_table_path(root: Path, args: argparse.Namespace) -> Path | None:
    if not args.context_table:
        return None
    path = Path(args.context_table)
    if not path.is_absolute():
        path = (root / path).resolve()
    else:
        path = path.resolve()
    return path


def resolve_split_config(root: Path, args: argparse.Namespace) -> tuple[str, str | None, Path | None]:
    split_scheme = args.split_scheme
    split_column = SPLIT_SCHEME_TO_COLUMN[split_scheme]
    if split_column is None:
        return split_scheme, None, None

    split_manifest_path = Path(args.split_manifest) if args.split_manifest else root / "data/processed/splits/viral_protein_strict_splits.tsv.gz"
    if not split_manifest_path.is_absolute():
        split_manifest_path = (root / split_manifest_path).resolve()
    else:
        split_manifest_path = split_manifest_path.resolve()
    return split_scheme, split_column, split_manifest_path


def load_split_assignments(split_manifest_path: Path, split_column: str) -> dict[str, int]:
    if not split_manifest_path.exists():
        raise FileNotFoundError(f"Split manifest does not exist: {split_manifest_path}")

    assignments: dict[str, int] = {}
    with open_text(split_manifest_path) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames or split_column not in reader.fieldnames:
            raise ValueError(f"Split column '{split_column}' was not found in {split_manifest_path}")
        for row in reader:
            protein_accession = row.get("protein_accession", "").strip()
            if not protein_accession:
                continue
            split_name = row.get(split_column, "").strip().lower()
            if split_name not in SPLIT_NAME_TO_ID:
                raise ValueError(
                    f"Unexpected split value '{split_name}' for protein '{protein_accession}' in column '{split_column}'"
                )
            split_id = SPLIT_NAME_TO_ID[split_name]
            previous = assignments.get(protein_accession)
            if previous is not None and previous != split_id:
                raise ValueError(f"Conflicting split assignment for protein '{protein_accession}' in {split_manifest_path}")
            assignments[protein_accession] = split_id
    return assignments


def load_context_rows(context_table_path: Path, category_fields: list[str], numeric_field_names: list[str]) -> dict[str, dict[str, str]]:
    if not context_table_path.exists():
        raise FileNotFoundError(f"Context feature table does not exist: {context_table_path}")

    rows: dict[str, dict[str, str]] = {}
    with open_text(context_table_path) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required_fields = ["protein_accession", *category_fields[len(BASE_CATEGORY_FIELDS) :], *numeric_field_names[len(BASE_NUMERIC_FIELD_NAMES) :]]
        if not reader.fieldnames:
            raise ValueError(f"Context feature table is missing a header: {context_table_path}")
        missing_fields = [field for field in required_fields if field not in reader.fieldnames]
        if missing_fields:
            preview = ", ".join(missing_fields[:10])
            raise ValueError(f"Context feature table is missing required columns: {preview}")
        for row in reader:
            protein_accession = row.get("protein_accession", "").strip()
            if not protein_accession:
                continue
            rows[protein_accession] = row
    return rows


def cache_matches_request(
    cache: dict[str, Any],
    split_scheme: str,
    split_column: str | None,
    split_manifest_path: Path | None,
    context_table_path: Path | None,
    category_fields: list[str],
    numeric_field_names: list[str],
) -> bool:
    expected_manifest = str(split_manifest_path) if split_manifest_path else ""
    expected_column = split_column or ""
    expected_context = str(context_table_path) if context_table_path else ""
    return (
        cache.get("split_scheme", "default_hash") == split_scheme
        and cache.get("split_column", "") == expected_column
        and cache.get("split_manifest_path", "") == expected_manifest
        and cache.get("context_table_path", "") == expected_context
        and cache.get("category_fields", list(BASE_CATEGORY_FIELDS)) == category_fields
        and cache.get("numeric_field_names", list(BASE_NUMERIC_FIELD_NAMES)) == numeric_field_names
    )


def build_cache(
    args: argparse.Namespace,
    input_path: Path,
    cache_path: Path,
    split_scheme: str,
    split_column: str | None,
    split_manifest_path: Path | None,
    context_table_path: Path | None,
    category_fields: list[str],
    numeric_field_names: list[str],
) -> dict[str, Any]:
    print(f"[cache] Building cache from {input_path}")
    start_time = time.time()

    split_assignments = None
    if split_column and split_manifest_path:
        print(f"[cache] Loading split assignments from {split_manifest_path} ({split_column})")
        split_assignments = load_split_assignments(split_manifest_path, split_column)

    context_rows = None
    if context_table_path is not None:
        print(f"[cache] Loading context features from {context_table_path}")
        context_rows = load_context_rows(context_table_path, category_fields, numeric_field_names)

    category_vocabs = {field: {"__MISSING__": 0} for field in category_fields}
    category_columns = {field: [] for field in category_fields}
    offsets = [0]
    lengths: list[int] = []
    splits: list[int] = []
    numeric_rows: list[list[float]] = []
    label_masks: list[int] = []
    protein_accessions: list[str] = []
    virus_tax_ids: list[str] = []
    genome_versions: list[str] = []
    descriptions: list[str] = []
    label_counter: Counter[int] = Counter()
    flat_tokens = bytearray()
    split_counter: Counter[int] = Counter()
    missing_split_accessions: list[str] = []
    missing_context_accessions: list[str] = []

    with open_text(input_path) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row_idx, row in enumerate(reader, start=1):
            if args.debug_limit and row_idx > args.debug_limit:
                break

            sequence = row.get("protein_sequence", "").strip()
            if not sequence:
                continue

            protein_accession = row.get("protein_accession", "").strip()
            if split_assignments is None:
                split_id = assign_split(row)
            else:
                split_id = split_assignments.get(protein_accession)
                if split_id is None:
                    missing_split_accessions.append(protein_accession or f"row_{row_idx}")
                    continue

            text = normalize_text(row)
            context_row = None
            if context_rows is not None:
                context_row = context_rows.get(protein_accession)
                if context_row is None:
                    missing_context_accessions.append(protein_accession or f"row_{row_idx}")
                    continue

            encoded = encode_sequence(sequence, args.max_length)
            flat_tokens.extend(encoded)
            offsets.append(len(flat_tokens))
            lengths.append(len(encoded))
            splits.append(split_id)
            split_counter[split_id] += 1

            descriptions.append(row.get("cds_product", "").strip() or row.get("protein_description", "").strip())
            protein_accessions.append(protein_accession)
            virus_tax_ids.append(row.get("virus_tax_id", "").strip())
            genome_versions.append(row.get("genome_version", "").strip())

            hits = label_hits(text)
            mask = 0
            for hit in hits:
                mask |= 1 << hit
                label_counter[hit] += 1
            label_masks.append(mask)
            numeric_rows.append(build_numeric_features(row, text, numeric_field_names, context_row))

            for field in category_fields:
                if field in row:
                    category_value = row.get(field, "")
                elif context_row is not None:
                    category_value = context_row.get(field, "")
                else:
                    category_value = ""
                category_columns[field].append(register_category(category_vocabs[field], category_value))

    if missing_split_accessions:
        preview = ", ".join(missing_split_accessions[:10])
        raise RuntimeError(
            f"Missing split assignments for {len(missing_split_accessions)} proteins under scheme '{split_scheme}'. "
            f"Examples: {preview}"
        )

    if missing_context_accessions:
        preview = ", ".join(missing_context_accessions[:10])
        raise RuntimeError(
            f"Missing context rows for {len(missing_context_accessions)} proteins in {context_table_path}. "
            f"Examples: {preview}"
        )

    active_label_indices = [idx for idx, _ in enumerate(LABEL_RULES) if label_counter[idx] >= args.min_label_count]
    if not active_label_indices:
        active_label_indices = [idx for idx, _ in enumerate(LABEL_RULES) if label_counter[idx] > 0]
        print(
            f"[cache] No labels met min_label_count={args.min_label_count}; "
            f"falling back to all observed labels in the current subset."
        )
    label_names = [LABEL_RULES[idx].name for idx in active_label_indices]
    label_descriptions = [LABEL_RULES[idx].description for idx in active_label_indices]
    selected_counts = [label_counter[idx] for idx in active_label_indices]

    label_matrix = np.zeros((len(label_masks), len(active_label_indices)), dtype=np.uint8)
    for row_idx, mask in enumerate(label_masks):
        for col_idx, label_idx in enumerate(active_label_indices):
            if mask & (1 << label_idx):
                label_matrix[row_idx, col_idx] = 1

    category_matrix = np.stack(
        [np.asarray(category_columns[field], dtype=np.int32) for field in category_fields],
        axis=1,
    )

    cache = {
        "flat_tokens": np.frombuffer(bytes(flat_tokens), dtype=np.uint8),
        "offsets": np.asarray(offsets, dtype=np.int64),
        "lengths": np.asarray(lengths, dtype=np.int32),
        "splits": np.asarray(splits, dtype=np.int8),
        "labels": label_matrix,
        "numeric": np.asarray(numeric_rows, dtype=np.float32),
        "categories": category_matrix,
        "category_fields": category_fields,
        "category_vocabs": category_vocabs,
        "label_names": label_names,
        "label_descriptions": label_descriptions,
        "label_support": selected_counts,
        "protein_accessions": protein_accessions,
        "virus_tax_ids": virus_tax_ids,
        "genome_versions": genome_versions,
        "descriptions": descriptions,
        "max_length": args.max_length,
        "numeric_field_names": numeric_field_names,
        "created_at": timestamp(),
        "split_scheme": split_scheme,
        "split_column": split_column or "",
        "split_manifest_path": str(split_manifest_path) if split_manifest_path else "",
        "context_table_path": str(context_table_path) if context_table_path else "",
        "split_counts": {SPLIT_ID_TO_NAME[split_id]: int(count) for split_id, count in split_counter.items()},
    }

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, cache_path)

    elapsed = time.time() - start_time
    print(
        f"[cache] Saved {len(lengths)} examples, {len(label_names)} labels, "
        f"{len(cache['flat_tokens']) / 1e6:.1f}M tokens to {cache_path} in {elapsed:.1f}s"
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
    category_fields: list[str],
    numeric_field_names: list[str],
) -> dict[str, Any]:
    if cache_path.exists() and not args.force_rebuild_cache:
        print(f"[cache] Loading existing cache from {cache_path}")
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        if cache_matches_request(
            cache,
            split_scheme,
            split_column,
            split_manifest_path,
            context_table_path,
            category_fields,
            numeric_field_names,
        ):
            return cache
        print("[cache] Existing cache configuration does not match the current request; rebuilding cache.")
    return build_cache(
        args,
        input_path,
        cache_path,
        split_scheme,
        split_column,
        split_manifest_path,
        context_table_path,
        category_fields,
        numeric_field_names,
    )


class ProteinSequenceDataset(Dataset):
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
        categories = self.cache["categories"][idx]
        numeric = self.cache["numeric"][idx]
        labels = self.cache["labels"][idx]
        return sequence, categories, numeric, labels, idx


def make_collate_fn():
    def collate(batch):
        sequences, categories, numerics, labels, indices = zip(*batch)
        max_len = max(len(seq) for seq in sequences)
        token_tensor = torch.zeros((len(batch), max_len), dtype=torch.long)
        for row_idx, seq in enumerate(sequences):
            token_tensor[row_idx, : len(seq)] = torch.from_numpy(seq.astype(np.int64, copy=False))
        return {
            "tokens": token_tensor,
            "categories": torch.as_tensor(np.stack(categories), dtype=torch.long),
            "numeric": torch.as_tensor(np.stack(numerics), dtype=torch.float32),
            "labels": torch.as_tensor(np.stack(labels), dtype=torch.float32),
            "indices": torch.as_tensor(indices, dtype=torch.long),
        }

    return collate


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


class ViralSequenceBaseline(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_labels: int,
        category_sizes: list[int],
        numeric_dim: int,
        embed_dim: int,
        hidden_dim: int,
        dropout: float,
    ):
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
        cat_embed_dim = 16
        self.category_embeddings = nn.ModuleList([nn.Embedding(size, cat_embed_dim) for size in category_sizes])
        self.numeric_encoder = nn.Sequential(
            nn.Linear(numeric_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        pooled_dim = hidden_dim * 3 + 64 + cat_embed_dim * len(category_sizes)
        self.classifier = nn.Sequential(
            nn.LayerNorm(pooled_dim),
            nn.Linear(pooled_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, num_labels),
        )

    def forward(self, tokens: torch.Tensor, categories: torch.Tensor, numeric: torch.Tensor) -> torch.Tensor:
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

        category_vectors = [embedding(categories[:, idx]) for idx, embedding in enumerate(self.category_embeddings)]
        numeric_vector = self.numeric_encoder(numeric)
        combined = torch.cat([mean_pool, max_pool, attention_pool, numeric_vector, *category_vectors], dim=-1)
        return self.classifier(combined)


def make_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    prefetch_factor: int,
    pin_memory: bool,
) -> DataLoader:
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


def compute_pos_weight(labels: np.ndarray, max_pos_weight: float) -> torch.Tensor:
    positives = labels.sum(axis=0).astype(np.float32)
    negatives = labels.shape[0] - positives
    pos_weight = negatives / np.clip(positives, 1.0, None)
    pos_weight = np.clip(pos_weight, 1.0, max_pos_weight)
    return torch.as_tensor(pos_weight, dtype=torch.float32)


def linear_warmup_cosine_decay(step: int, total_steps: int, warmup_steps: int) -> float:
    if total_steps <= 0:
        return 1.0
    if step < warmup_steps:
        return float(step + 1) / float(max(1, warmup_steps))
    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return 0.5 * (1.0 + math.cos(math.pi * progress))


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
        tokens = batch["tokens"].to(device, non_blocking=True)
        categories = batch["categories"].to(device, non_blocking=True)
        numeric = batch["numeric"].to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=use_autocast):
            logits = model(tokens, categories, numeric)
        probs = torch.sigmoid(logits).to(dtype=torch.float32).cpu().numpy()
        all_probs.append(probs)
        all_targets.append(batch["labels"].numpy())
        all_indices.append(batch["indices"].numpy())

    return (
        np.concatenate(all_probs, axis=0),
        np.concatenate(all_targets, axis=0),
        np.concatenate(all_indices, axis=0),
    )


def optimize_thresholds(y_true: np.ndarray, y_prob: np.ndarray) -> np.ndarray:
    thresholds = np.full(y_true.shape[1], 0.5, dtype=np.float32)
    grid = np.linspace(0.1, 0.9, 17)
    for label_idx in range(y_true.shape[1]):
        positives = y_true[:, label_idx].sum()
        if positives == 0:
            continue
        best_threshold = 0.5
        best_score = -1.0
        for threshold in grid:
            preds = (y_prob[:, label_idx] >= threshold).astype(np.uint8)
            score = f1_score(y_true[:, label_idx], preds, zero_division=0)
            if score > best_score:
                best_score = score
                best_threshold = float(threshold)
        thresholds[label_idx] = best_threshold
    return thresholds


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: np.ndarray,
    label_names: list[str],
) -> dict[str, Any]:
    y_pred = (y_prob >= thresholds.reshape(1, -1)).astype(np.uint8)
    label_metrics: list[dict[str, Any]] = []
    per_label_ap: list[float] = []
    per_label_f1: list[float] = []

    for label_idx, label_name in enumerate(label_names):
        support = int(y_true[:, label_idx].sum())
        ap = None
        if support > 0:
            ap = float(average_precision_score(y_true[:, label_idx], y_prob[:, label_idx]))
            per_label_ap.append(ap)
        f1 = float(f1_score(y_true[:, label_idx], y_pred[:, label_idx], zero_division=0))
        precision = float(precision_score(y_true[:, label_idx], y_pred[:, label_idx], zero_division=0))
        recall = float(recall_score(y_true[:, label_idx], y_pred[:, label_idx], zero_division=0))
        per_label_f1.append(f1)
        label_metrics.append(
            {
                "label": label_name,
                "support": support,
                "threshold": float(thresholds[label_idx]),
                "average_precision": ap,
                "f1": f1,
                "precision": precision,
                "recall": recall,
            }
        )

    macro_ap = float(np.mean(per_label_ap)) if per_label_ap else 0.0
    macro_f1 = float(np.mean(per_label_f1)) if per_label_f1 else 0.0
    micro_ap = float(average_precision_score(y_true.reshape(-1), y_prob.reshape(-1)))
    micro_f1 = float(f1_score(y_true.reshape(-1), y_pred.reshape(-1), zero_division=0))

    return {
        "macro_average_precision": macro_ap,
        "micro_average_precision": micro_ap,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "label_metrics": label_metrics,
    }


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def save_history(path: Path, history: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in history:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_label_metrics(path: Path, label_metrics: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["label", "support", "threshold", "average_precision", "f1", "precision", "recall"],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(label_metrics)


def save_test_predictions(
    path: Path,
    cache: dict[str, Any],
    indices: np.ndarray,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: np.ndarray,
    label_names: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "protein_accession",
                "virus_tax_id",
                "genome_version",
                "description",
                "true_labels",
                "predicted_labels",
                "top_label",
                "top_probability",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        for row_idx, sample_idx in enumerate(indices):
            true_labels = [label_names[j] for j in range(len(label_names)) if y_true[row_idx, j] == 1]
            predicted_labels = [label_names[j] for j in range(len(label_names)) if y_prob[row_idx, j] >= thresholds[j]]
            top_idx = int(np.argmax(y_prob[row_idx]))
            writer.writerow(
                {
                    "protein_accession": cache["protein_accessions"][int(sample_idx)],
                    "virus_tax_id": cache["virus_tax_ids"][int(sample_idx)],
                    "genome_version": cache["genome_versions"][int(sample_idx)],
                    "description": cache["descriptions"][int(sample_idx)],
                    "true_labels": json.dumps(true_labels, ensure_ascii=False),
                    "predicted_labels": json.dumps(predicted_labels, ensure_ascii=False),
                    "top_label": label_names[top_idx],
                    "top_probability": float(y_prob[row_idx, top_idx]),
                }
            )


def main() -> int:
    args = parse_args()
    device = choose_device(args.device)
    set_seed(args.seed, device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    root = repo_root()
    input_path = (root / args.input).resolve()
    output_dir = (root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = (Path(args.cache_path) if args.cache_path else output_dir / "dataset_cache.pt")
    if not cache_path.is_absolute():
        cache_path = (root / cache_path).resolve()
    split_scheme, split_column, split_manifest_path = resolve_split_config(root, args)
    context_table_path = resolve_context_table_path(root, args)
    category_fields, numeric_field_names = feature_schema(context_table_path)

    cache = load_or_build_cache(
        args,
        input_path,
        cache_path,
        split_scheme=split_scheme,
        split_column=split_column,
        split_manifest_path=split_manifest_path,
        context_table_path=context_table_path,
        category_fields=category_fields,
        numeric_field_names=numeric_field_names,
    )
    label_names: list[str] = cache["label_names"]
    if not label_names:
        raise RuntimeError("No labels passed the minimum support threshold.")

    splits = cache["splits"]
    train_idx = np.where(splits == 0)[0]
    val_idx = np.where(splits == 1)[0]
    test_idx = np.where(splits == 2)[0]
    if train_idx.size == 0 or val_idx.size == 0 or test_idx.size == 0:
        raise RuntimeError(
            f"Split scheme '{split_scheme}' produced empty partitions: "
            f"train={train_idx.size}, val={val_idx.size}, test={test_idx.size}"
        )

    train_dataset = ProteinSequenceDataset(cache, train_idx)
    val_dataset = ProteinSequenceDataset(cache, val_idx)
    test_dataset = ProteinSequenceDataset(cache, test_idx)

    pin_memory = device.type == "cuda"
    train_loader = make_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        pin_memory=pin_memory,
    )
    eval_num_workers = max(0, min(args.num_workers, 4))
    val_loader = make_dataloader(
        val_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=eval_num_workers,
        prefetch_factor=args.prefetch_factor,
        pin_memory=pin_memory,
    )
    test_loader = make_dataloader(
        test_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=eval_num_workers,
        prefetch_factor=args.prefetch_factor,
        pin_memory=pin_memory,
    )

    category_sizes = []
    for field_idx, _field in enumerate(cache["category_fields"]):
        values = cache["categories"][:, field_idx]
        category_sizes.append(int(values.max()) + 1)

    model = ViralSequenceBaseline(
        vocab_size=len(AA_TO_ID) + 1,
        num_labels=len(label_names),
        category_sizes=category_sizes,
        numeric_dim=cache["numeric"].shape[1],
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
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: linear_warmup_cosine_decay(step, total_steps, warmup_steps),
    )

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
        "label_names": label_names,
        "label_support": cache["label_support"],
        "category_fields": cache["category_fields"],
        "numeric_field_names": cache["numeric_field_names"],
        "context_table_path": str(context_table_path) if context_table_path else "",
        "split_strategy": {
            "scheme": split_scheme,
            "column": split_column or "",
            "manifest_path": str(split_manifest_path) if split_manifest_path else "",
            "counts": cache.get("split_counts", {}),
        },
        "split_sizes": {
            "train": int(train_idx.shape[0]),
            "val": int(val_idx.shape[0]),
            "test": int(test_idx.shape[0]),
        },
        "seed": int(args.seed),
        **collect_git_metadata(root),
    }
    save_json(output_dir / "run_manifest.json", run_manifest)

    best_val_score = -1.0
    best_checkpoint_path = output_dir / "best_model.pt"
    history: list[dict[str, Any]] = []

    print(
        f"[train] device={device} train={len(train_dataset)} val={len(val_dataset)} test={len(test_dataset)} "
        f"labels={len(label_names)} batch_size={args.batch_size} split_scheme={split_scheme} "
        f"context={'on' if context_table_path else 'off'}"
    )

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        seen = 0

        for step, batch in enumerate(train_loader, start=1):
            optimizer.zero_grad(set_to_none=True)
            tokens = batch["tokens"].to(device, non_blocking=True)
            categories = batch["categories"].to(device, non_blocking=True)
            numeric = batch["numeric"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=use_amp):
                logits = model(tokens, categories, numeric)
                loss = criterion(logits, labels)

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
            batch_size = int(tokens.shape[0])
            running_loss += float(loss.item()) * batch_size
            seen += batch_size

            if step % 100 == 0 or step == len(train_loader):
                avg_loss = running_loss / max(1, seen)
                current_lr = float(optimizer.param_groups[0]["lr"])
                print(f"[train] epoch={epoch} step={step}/{len(train_loader)} loss={avg_loss:.4f} lr={current_lr:.6g}")

        train_loss = running_loss / max(1, seen)
        val_probs, val_true, _ = predict(model, val_loader, device, autocast_dtype)
        val_thresholds = optimize_thresholds(val_true, val_probs)
        val_metrics = compute_metrics(val_true, val_probs, val_thresholds, label_names)
        epoch_time = time.time() - epoch_start

        history_row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_macro_average_precision": val_metrics["macro_average_precision"],
            "val_micro_average_precision": val_metrics["micro_average_precision"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_micro_f1": val_metrics["micro_f1"],
            "epoch_seconds": epoch_time,
        }
        history.append(history_row)
        save_history(output_dir / "history.jsonl", history)

        print(
            f"[val] epoch={epoch} train_loss={train_loss:.4f} "
            f"macro_ap={val_metrics['macro_average_precision']:.4f} "
            f"macro_f1={val_metrics['macro_f1']:.4f} time={epoch_time:.1f}s"
        )

        if val_metrics["macro_average_precision"] > best_val_score:
            best_val_score = val_metrics["macro_average_precision"]
            checkpoint = {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "epoch": epoch,
                "best_val_macro_average_precision": best_val_score,
                "label_names": label_names,
                "config": vars(args),
            }
            torch.save(checkpoint, best_checkpoint_path)
            save_json(output_dir / "best_val_metrics.json", val_metrics)
            save_label_metrics(output_dir / "best_val_label_metrics.tsv", val_metrics["label_metrics"])
            save_json(
                output_dir / "best_thresholds.json",
                {"thresholds": {label_names[i]: float(val_thresholds[i]) for i in range(len(label_names))}},
            )
            print(f"[checkpoint] Saved best model to {best_checkpoint_path}")

    checkpoint = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])

    val_probs, val_true, _ = predict(model, val_loader, device, autocast_dtype)
    test_probs, test_true, test_indices = predict(model, test_loader, device, autocast_dtype)
    final_thresholds = optimize_thresholds(val_true, val_probs)
    final_val_metrics = compute_metrics(val_true, val_probs, final_thresholds, label_names)
    final_test_metrics = compute_metrics(test_true, test_probs, final_thresholds, label_names)

    summary = {
        "created_at": timestamp(),
        "best_checkpoint": str(best_checkpoint_path),
        "best_epoch": int(checkpoint["epoch"]),
        "thresholds": {label_names[i]: float(final_thresholds[i]) for i in range(len(label_names))},
        "validation": final_val_metrics,
        "test": final_test_metrics,
    }
    save_json(output_dir / "metrics_summary.json", summary)
    save_label_metrics(output_dir / "test_label_metrics.tsv", final_test_metrics["label_metrics"])

    if args.save_test_predictions:
        save_test_predictions(
            output_dir / "test_predictions.tsv.gz",
            cache,
            test_indices,
            test_true,
            test_probs,
            final_thresholds,
            label_names,
        )

    print(
        f"[done] best_epoch={checkpoint['epoch']} "
        f"val_macro_ap={final_val_metrics['macro_average_precision']:.4f} "
        f"test_macro_ap={final_test_metrics['macro_average_precision']:.4f} "
        f"test_macro_f1={final_test_metrics['macro_f1']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
