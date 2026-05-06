from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


AA_VOCAB = "ACDEFGHIKLMNPQRSTVWYBXZJUO*"
AA_TO_INT = {aa: idx + 1 for idx, aa in enumerate(AA_VOCAB)}
MASK_64 = (1 << 64) - 1
MINHASH_PARAMS = (
    (11400714819323198485, 1),
    (14029467366897019727, 7046029254386353131),
    (1609587929392839161, 9650029242287828579),
)
FAMILY_SUFFIXES = ("viridae", "virinae", "viriformidae")
HOST_SUPERGROUP_RULES = (
    ("Bacteria", ("bacteria",)),
    ("Archaea", ("archaea",)),
    ("Viridiplantae", ("viridiplantae",)),
    ("Metazoa", ("metazoa",)),
    ("Fungi", ("fungi",)),
    ("SAR", ("sar", "stramenopiles", "alveolata", "rhizaria")),
    ("Amoebozoa", ("amoebozoa",)),
    ("Excavata", ("discoba", "excavata", "metamonada")),
)

SPLIT_FIELDS = [
    "protein_accession",
    "genome_version",
    "virus_tax_id",
    "virus_name",
    "protein_length_aa",
    "protein_feature_type",
    "virus_family",
    "virus_family_source",
    "host_taxid_key",
    "host_taxid_source",
    "host_supergroup",
    "host_supergroup_source",
    "sequence_sketch_key",
    "sequence_length_bin",
    "species_holdout_split",
    "family_holdout_split",
    "host_taxid_holdout_split",
    "host_supergroup_holdout_split",
    "sequence_sketch_holdout_split",
]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate stricter ViruFunc-FM protein split manifests.")
    parser.add_argument(
        "--input",
        default="data/processed/training/viral_protein_training_index.tsv.gz",
        help="Protein-level training index table",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed/splits",
        help="Directory for split manifests and reports",
    )
    parser.add_argument("--train-fraction", type=float, default=0.8, help="Train fraction for hash-based partitions")
    parser.add_argument("--val-fraction", type=float, default=0.1, help="Validation fraction for hash-based partitions")
    parser.add_argument("--seed", type=int, default=42, help="Stable seed used inside the hash assignment")
    parser.add_argument("--kmer-size", type=int, default=4, help="K-mer size for the sequence sketch proxy split")
    parser.add_argument("--debug-limit", type=int, default=0, help="Optional row cap for smoke tests")
    return parser.parse_args()


def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def open_tsv_writer(path: Path, fieldnames: list[str]) -> tuple[csv.DictWriter, gzip.GzipFile]:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = gzip.open(path, "wt", encoding="utf-8", newline="")
    writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
    writer.writeheader()
    return writer, handle


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


def stable_bucket(key: str) -> int:
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 10_000


def assign_partition(group_key: str, scheme_name: str, seed: int, train_fraction: float, val_fraction: float) -> str:
    bucket = stable_bucket(f"{seed}|{scheme_name}|{group_key}")
    train_cutoff = int(train_fraction * 10_000)
    val_cutoff = int((train_fraction + val_fraction) * 10_000)
    if bucket < train_cutoff:
        return "train"
    if bucket < val_cutoff:
        return "val"
    return "test"


def lineage_parts(lineage: str) -> list[str]:
    return [part.strip().rstrip(".") for part in lineage.split(";") if part.strip()]


def derive_virus_family(lineage: str) -> tuple[str, str]:
    parts = lineage_parts(lineage)
    for part in reversed(parts):
        lower = part.lower()
        if any(lower.endswith(suffix) for suffix in FAMILY_SUFFIXES):
            return part, "family_suffix"
    for part in reversed(parts):
        lower = part.lower()
        if lower not in {"viruses", "virus"} and lower.endswith("virus"):
            return part, "virus_suffix"
    if parts:
        return parts[-1], "deepest_lineage_term"
    return "unknown", "missing"


def normalize_host_text(source_host: str) -> str:
    normalized = " ".join(source_host.strip().lower().split())
    return normalized


def derive_host_taxid_key(host_tax_ids_json: str, source_host: str) -> tuple[str, str]:
    host_tax_ids = parse_json_list(host_tax_ids_json)
    if host_tax_ids:
        return host_tax_ids[0], "host_tax_ids_json"
    normalized_source = normalize_host_text(source_host)
    if normalized_source:
        return f"source_host::{normalized_source}", "source_host"
    return "unknown", "missing"


def derive_host_supergroup(host_lineages_json: str, source_host: str) -> tuple[str, str]:
    host_lineages = parse_json_list(host_lineages_json)
    if host_lineages:
        tokens = {token.strip().lower() for token in host_lineages[0].split(";") if token.strip()}
        for group_name, markers in HOST_SUPERGROUP_RULES:
            if any(marker in tokens for marker in markers):
                return group_name, "host_lineages_json"
        if "eukaryota" in tokens:
            return "OtherEukaryota", "host_lineages_json"
        if "root" in tokens:
            return "root", "host_lineages_json"
    normalized_source = normalize_host_text(source_host)
    if normalized_source:
        return "source_host_only", "source_host"
    return "unknown", "missing"


def sequence_length_bin(sequence_length: int) -> str:
    if sequence_length <= 0:
        return "len_0"
    lower = int(math.floor(math.log2(sequence_length)))
    upper = lower + 1
    return f"len2^{lower}-{upper}"


def update_minhash(minima: list[int], value: int) -> None:
    for idx, (multiplier, increment) in enumerate(MINHASH_PARAMS):
        hashed = (value * multiplier + increment) & MASK_64
        if hashed < minima[idx]:
            minima[idx] = hashed


def build_sequence_sketch_key(sequence: str, kmer_size: int) -> str:
    sequence = sequence.strip().upper()
    if len(sequence) < kmer_size:
        short_key = stable_bucket(f"short|{sequence}")
        return f"short:{len(sequence)}:{short_key:04d}"

    encoded = [AA_TO_INT.get(aa, AA_TO_INT["X"]) for aa in sequence]
    base = len(AA_TO_INT) + 1
    removal_factor = base ** (kmer_size - 1)
    minima = [MASK_64] * len(MINHASH_PARAMS)

    rolling = 0
    for idx in range(kmer_size):
        rolling = rolling * base + encoded[idx]
    update_minhash(minima, rolling)

    for idx in range(kmer_size, len(encoded)):
        rolling = (rolling - encoded[idx - kmer_size] * removal_factor) * base + encoded[idx]
        update_minhash(minima, rolling)

    coarse_bins = [f"{minimum >> 52:03x}" for minimum in minima]
    return f"{sequence_length_bin(len(sequence))}:{'-'.join(coarse_bins)}"


def main() -> int:
    args = parse_args()
    root = repo_root()
    input_path = (root / args.input).resolve()
    output_dir = (root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    split_path = output_dir / "viral_protein_strict_splits.tsv.gz"
    report_path = output_dir / "viral_protein_strict_splits_report.json"

    scheme_partition_counts = defaultdict(Counter)
    scheme_group_counts = defaultdict(Counter)
    family_counter: Counter[str] = Counter()
    host_supergroup_counter: Counter[str] = Counter()

    total_rows = 0
    with open_text(input_path) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        writer, writer_handle = open_tsv_writer(split_path, SPLIT_FIELDS)
        try:
            for row_idx, row in enumerate(reader, start=1):
                if args.debug_limit and row_idx > args.debug_limit:
                    break

                protein_accession = row.get("protein_accession", "").strip()
                genome_version = row.get("genome_version", "").strip()
                virus_tax_id = row.get("virus_tax_id", "").strip() or "unknown"
                virus_name = row.get("virus_name", "").strip()
                protein_length = row.get("protein_length_aa", "").strip() or "0"
                protein_feature_type = row.get("protein_feature_type", "").strip()

                virus_family, virus_family_source = derive_virus_family(row.get("virus_lineage", ""))
                host_taxid_key, host_taxid_source = derive_host_taxid_key(
                    row.get("host_tax_ids_json", ""),
                    row.get("source_host", ""),
                )
                host_supergroup, host_supergroup_source = derive_host_supergroup(
                    row.get("host_lineages_json", ""),
                    row.get("source_host", ""),
                )
                sequence = row.get("protein_sequence", "").strip()
                sequence_sketch_key = build_sequence_sketch_key(sequence, args.kmer_size)
                length_bin = sequence_length_bin(len(sequence))

                scheme_keys = {
                    "species_holdout": virus_tax_id,
                    "family_holdout": virus_family,
                    "host_taxid_holdout": host_taxid_key,
                    "host_supergroup_holdout": host_supergroup,
                    "sequence_sketch_holdout": sequence_sketch_key,
                }

                record = {
                    "protein_accession": protein_accession,
                    "genome_version": genome_version,
                    "virus_tax_id": virus_tax_id,
                    "virus_name": virus_name,
                    "protein_length_aa": protein_length,
                    "protein_feature_type": protein_feature_type,
                    "virus_family": virus_family,
                    "virus_family_source": virus_family_source,
                    "host_taxid_key": host_taxid_key,
                    "host_taxid_source": host_taxid_source,
                    "host_supergroup": host_supergroup,
                    "host_supergroup_source": host_supergroup_source,
                    "sequence_sketch_key": sequence_sketch_key,
                    "sequence_length_bin": length_bin,
                }

                for scheme_name, group_key in scheme_keys.items():
                    partition = assign_partition(
                        group_key=group_key,
                        scheme_name=scheme_name,
                        seed=args.seed,
                        train_fraction=args.train_fraction,
                        val_fraction=args.val_fraction,
                    )
                    record[f"{scheme_name}_split"] = partition
                    scheme_partition_counts[scheme_name][partition] += 1
                    scheme_group_counts[scheme_name][group_key] += 1

                family_counter[virus_family] += 1
                host_supergroup_counter[host_supergroup] += 1
                writer.writerow(record)
                total_rows += 1
        finally:
            writer_handle.close()

    report = {
        "created_at": timestamp(),
        "input": str(input_path),
        "output": str(split_path),
        "row_count": total_rows,
        "debug_limit": args.debug_limit,
        "config": {
            "train_fraction": args.train_fraction,
            "val_fraction": args.val_fraction,
            "seed": args.seed,
            "kmer_size": args.kmer_size,
        },
        "top_virus_families": family_counter.most_common(20),
        "top_host_supergroups": host_supergroup_counter.most_common(20),
        "schemes": {},
    }

    for scheme_name, partition_counts in scheme_partition_counts.items():
        group_counts = scheme_group_counts[scheme_name]
        report["schemes"][scheme_name] = {
            "partition_counts": dict(partition_counts),
            "unique_groups": len(group_counts),
            "top_groups": group_counts.most_common(20),
        }

    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["schemes"], ensure_ascii=False, indent=2))
    print(f"Wrote strict split manifest to {split_path}")
    print(f"Wrote report to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
