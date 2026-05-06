from __future__ import annotations

import json


CONTEXT_CATEGORY_FIELDS = [
    "context_prev_feature_type",
    "context_next_feature_type",
    "context_host_supergroup",
    "context_segment_bucket",
]

CONTEXT_NUMERIC_PREFIX_FIELDS = [
    "context_log_genome_protein_count",
    "context_genome_hypothetical_fraction",
    "context_genome_mat_peptide_fraction",
    "context_has_prev_neighbor",
    "context_has_next_neighbor",
    "context_same_strand_prev",
    "context_same_strand_next",
    "context_log_host_taxid_count",
    "context_log_host_lineage_count",
]

FAMILY_SUFFIXES = ("viridae", "virinae", "viriformidae")


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


def derive_baltimore_like_class(lineage: str, source_mol_type: str) -> str:
    source = source_mol_type.strip().lower()
    lineage_text = " ".join(part.lower() for part in lineage_parts(lineage))

    if "reverse" in lineage_text or "retro" in lineage_text or "rt" in source:
        return "retrotranscribing"
    if "double-stranded dna" in source or source.startswith("dsdna"):
        return "dsDNA"
    if "single-stranded dna" in source or source.startswith("ssdna"):
        return "ssDNA"
    if "double-stranded rna" in source or source.startswith("dsrna"):
        return "dsRNA"
    if "negative-strand rna" in source or "negarnaviricota" in lineage_text:
        return "ssRNA_negative"
    if "positive-strand rna" in source or "pisuviricota" in lineage_text or "kitrinoviricota" in lineage_text:
        return "ssRNA_positive"
    if "rna" in source:
        return "RNA_other"
    if "dna" in source:
        return "DNA_other"
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


def normalize_segment_bucket(segment: str) -> str:
    value = segment.strip()
    if not value:
        return "__UNSEGMENTED__"
    lowered = value.lower()
    if lowered.startswith("segment"):
        return lowered.replace(" ", "_")
    return value


def context_numeric_field_names(label_names: list[str]) -> list[str]:
    fields = list(CONTEXT_NUMERIC_PREFIX_FIELDS)
    fields.extend(f"context_genome_{label_name}_fraction" for label_name in label_names)
    fields.extend(f"context_local_{label_name}_count" for label_name in label_names)
    return fields
