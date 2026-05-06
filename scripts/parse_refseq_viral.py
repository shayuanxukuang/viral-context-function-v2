from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


ORGANISM_SUFFIX_RE = re.compile(r"\[(?P<organism>[^\[\]]+)\]\s*$")
LOCUS_RE = re.compile(
    r"^LOCUS\s+(?P<locus>\S+)\s+(?P<length>\d+)\s+bp\s+(?P<molecule>\S+)\s+(?P<topology>\S+)\s+(?P<division>\S+)"
)
LOCATION_RANGE_RE = re.compile(r"([<>]?\d+)\.\.([<>]?\d+)|([<>]?\d+)")

PROTEIN_FIELDS = [
    "protein_accession",
    "protein_description",
    "protein_organism",
    "protein_length_aa",
    "protein_sequence_sha256",
    "protein_sequence",
]

GENOME_SEQUENCE_FIELDS = [
    "genome_version",
    "genome_description",
    "genome_organism",
    "genome_length_nt",
    "genome_sequence_sha256",
    "genome_sequence",
]

GENOME_FIELDS = [
    "genome_accession",
    "genome_version",
    "locus",
    "definition",
    "organism",
    "taxonomy",
    "virus_tax_id",
    "genome_length_nt",
    "molecule_type",
    "topology",
    "division",
    "source_mol_type",
    "source_host",
    "source_isolate",
    "source_segment",
    "source_geo_loc_name",
    "source_collection_date",
    "source_note",
]

CDS_FIELDS = [
    "genome_accession",
    "genome_version",
    "virus_tax_id",
    "organism",
    "source_segment",
    "feature_type",
    "protein_accession",
    "gene",
    "locus_tag",
    "product",
    "location_raw",
    "location_kind",
    "cds_start",
    "cds_end",
    "cds_strand",
    "location_part_count",
    "location_partial_left",
    "location_partial_right",
    "codon_start",
    "translation_length_aa",
]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse NCBI RefSeq viral FASTA and GenBank files.")
    parser.add_argument(
        "--protein-fasta",
        default="data/raw/ncbi/refseq/viral.1.protein.faa.gz",
        help="Protein FASTA file to parse",
    )
    parser.add_argument(
        "--genome-fasta",
        default="data/raw/ncbi/refseq/viral.1.1.genomic.fna.gz",
        help="Genome FASTA file to parse",
    )
    parser.add_argument(
        "--genbank",
        default="data/raw/ncbi/refseq/viral.1.genomic.gbff.gz",
        help="GenBank flatfile to parse",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed/refseq",
        help="Directory for processed RefSeq tables",
    )
    return parser.parse_args()


def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def open_tsv_writer(path: Path, fieldnames: list[str]) -> tuple[csv.DictWriter, gzip.GzipFile]:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = gzip.open(path, "wt", encoding="utf-8", newline="")
    writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
    writer.writeheader()
    return writer, handle


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def clean_description(description: str) -> tuple[str, str]:
    description = description.strip()
    organism = ""
    match = ORGANISM_SUFFIX_RE.search(description)
    if match:
        organism = match.group("organism").strip()
    return description, organism


def parse_fasta(path: Path) -> Iterable[tuple[str, str]]:
    header: str | None = None
    sequence_chunks: list[str] = []

    with open_text(path) as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(sequence_chunks)
                header = line[1:]
                sequence_chunks = []
            else:
                sequence_chunks.append(line)

    if header is not None:
        yield header, "".join(sequence_chunks)


def parse_fasta_header(header: str) -> tuple[str, str]:
    header = header.strip()
    if not header:
        return "", ""
    if " " not in header:
        return header, ""
    accession, description = header.split(" ", 1)
    return accession.strip(), description.strip()


def write_protein_table(input_path: Path, output_path: Path) -> int:
    writer, handle = open_tsv_writer(output_path, PROTEIN_FIELDS)
    count = 0
    try:
        for header, sequence in parse_fasta(input_path):
            accession, description = parse_fasta_header(header)
            description, organism = clean_description(description)
            writer.writerow(
                {
                    "protein_accession": accession,
                    "protein_description": description,
                    "protein_organism": organism,
                    "protein_length_aa": len(sequence),
                    "protein_sequence_sha256": sha256_text(sequence),
                    "protein_sequence": sequence,
                }
            )
            count += 1
    finally:
        handle.close()
    return count


def write_genome_sequence_table(input_path: Path, output_path: Path) -> int:
    writer, handle = open_tsv_writer(output_path, GENOME_SEQUENCE_FIELDS)
    count = 0
    try:
        for header, sequence in parse_fasta(input_path):
            accession, description = parse_fasta_header(header)
            description, organism = clean_description(description)
            writer.writerow(
                {
                    "genome_version": accession,
                    "genome_description": description,
                    "genome_organism": organism,
                    "genome_length_nt": len(sequence),
                    "genome_sequence_sha256": sha256_text(sequence),
                    "genome_sequence": sequence,
                }
            )
            count += 1
    finally:
        handle.close()
    return count


def iter_genbank_records(path: Path) -> Iterable[list[str]]:
    record_lines: list[str] = []
    with open_text(path) as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if line.strip() == "//":
                if record_lines:
                    yield record_lines
                    record_lines = []
            else:
                record_lines.append(line)
    if record_lines:
        yield record_lines


def collect_field(lines: list[str], start_index: int) -> tuple[str, int]:
    first_line = lines[start_index]
    chunks = [first_line[12:].strip()]
    index = start_index + 1
    while index < len(lines):
        line = lines[index]
        if len(line) >= 12 and line[:12] == "            ":
            chunks.append(line.strip())
            index += 1
            continue
        break
    return " ".join(chunk for chunk in chunks if chunk), index


def add_qualifier(qualifiers: dict[str, list[str]], key: str, value: str) -> None:
    qualifiers.setdefault(key, []).append(value)


def join_qualifier_parts(key: str, parts: list[str]) -> str:
    if key == "translation":
        return "".join(part.strip() for part in parts)
    return " ".join(part.strip() for part in parts if part.strip())


def parse_feature_table(lines: list[str]) -> list[dict[str, object]]:
    features: list[dict[str, object]] = []
    current: dict[str, object] | None = None
    active_key: str | None = None
    active_parts: list[str] = []

    def flush_active() -> None:
        nonlocal active_key, active_parts
        if current is None or active_key is None:
            return
        qualifiers = current["qualifiers"]
        assert isinstance(qualifiers, dict)
        add_qualifier(qualifiers, active_key, join_qualifier_parts(active_key, active_parts))
        active_key = None
        active_parts = []

    for line in lines:
        feature_key = line[5:21].strip() if len(line) >= 21 else ""
        payload = line[21:].rstrip() if len(line) > 21 else line.strip()
        stripped = payload.strip()

        if feature_key:
            flush_active()
            if current is not None:
                features.append(current)
            current = {"type": feature_key, "location": payload.strip(), "qualifiers": {}}
            continue

        if current is None or not stripped:
            continue

        if active_key is not None:
            if stripped.endswith('"'):
                active_parts.append(stripped[:-1])
                flush_active()
            else:
                active_parts.append(stripped)
            continue

        if stripped.startswith("/"):
            body = stripped[1:]
            if "=" not in body:
                qualifiers = current["qualifiers"]
                assert isinstance(qualifiers, dict)
                add_qualifier(qualifiers, body, "true")
                continue

            key, raw_value = body.split("=", 1)
            if raw_value.startswith('"'):
                value = raw_value[1:]
                if value.endswith('"'):
                    qualifiers = current["qualifiers"]
                    assert isinstance(qualifiers, dict)
                    add_qualifier(qualifiers, key, value[:-1])
                else:
                    active_key = key
                    active_parts = [value]
            else:
                qualifiers = current["qualifiers"]
                assert isinstance(qualifiers, dict)
                add_qualifier(qualifiers, key, raw_value)
            continue

        current["location"] = f"{current['location']}{stripped}"

    flush_active()
    if current is not None:
        features.append(current)
    return features


def first_qualifier(feature: dict[str, object], key: str) -> str:
    qualifiers = feature.get("qualifiers", {})
    if not isinstance(qualifiers, dict):
        return ""
    values = qualifiers.get(key, [])
    if not values:
        return ""
    return values[0]


def parse_taxid(source_feature: dict[str, object]) -> str:
    qualifiers = source_feature.get("qualifiers", {})
    if not isinstance(qualifiers, dict):
        return ""
    for value in qualifiers.get("db_xref", []):
        if value.startswith("taxon:"):
            return value.split(":", 1)[1]
    return ""


def infer_location_kind(location: str) -> str:
    if "join(" in location:
        return "join"
    if "order(" in location:
        return "order"
    if "complement(" in location:
        return "complement"
    return "simple"


def parse_location(location: str) -> dict[str, object]:
    starts: list[int] = []
    ends: list[int] = []
    for match in LOCATION_RANGE_RE.finditer(location):
        if match.group(1) and match.group(2):
            left = int(match.group(1).lstrip("<>"))
            right = int(match.group(2).lstrip("<>"))
            starts.append(left)
            ends.append(right)
        elif match.group(3):
            value = int(match.group(3).lstrip("<>"))
            starts.append(value)
            ends.append(value)

    strand = -1 if "complement(" in location else 1
    return {
        "location_kind": infer_location_kind(location),
        "cds_start": min(starts) if starts else "",
        "cds_end": max(ends) if ends else "",
        "cds_strand": strand,
        "location_part_count": len(starts),
        "location_partial_left": "1" if "<" in location else "0",
        "location_partial_right": "1" if ">" in location else "0",
    }


def parse_locus(line: str) -> dict[str, str]:
    match = LOCUS_RE.match(line)
    if not match:
        return {
            "locus": "",
            "genome_length_nt": "",
            "molecule_type": "",
            "topology": "",
            "division": "",
        }
    return {
        "locus": match.group("locus"),
        "genome_length_nt": match.group("length"),
        "molecule_type": match.group("molecule"),
        "topology": match.group("topology"),
        "division": match.group("division"),
    }


def parse_genbank_record(lines: list[str]) -> tuple[dict[str, object], list[dict[str, object]]]:
    locus_info = {"locus": "", "genome_length_nt": "", "molecule_type": "", "topology": "", "division": ""}
    definition = ""
    accession = ""
    version = ""
    organism = ""
    taxonomy = ""
    features: list[dict[str, object]] = []

    index = 0
    while index < len(lines):
        line = lines[index]

        if line.startswith("LOCUS"):
            locus_info = parse_locus(line)
            index += 1
            continue

        if line.startswith("DEFINITION"):
            definition, index = collect_field(lines, index)
            continue

        if line.startswith("ACCESSION"):
            accession = line[12:].strip().split()[0]
            index += 1
            continue

        if line.startswith("VERSION"):
            version = line[12:].strip().split()[0]
            index += 1
            continue

        if line.startswith("SOURCE"):
            index += 1
            if index < len(lines) and lines[index].startswith("  ORGANISM"):
                organism = lines[index][12:].strip()
                index += 1
                taxonomy_chunks: list[str] = []
                while index < len(lines) and lines[index].startswith("            "):
                    taxonomy_chunks.append(lines[index].strip())
                    index += 1
                taxonomy = " ".join(taxonomy_chunks)
                continue
            continue

        if line.startswith("FEATURES"):
            index += 1
            feature_lines: list[str] = []
            while index < len(lines) and not lines[index].startswith("ORIGIN"):
                feature_lines.append(lines[index])
                index += 1
            features = parse_feature_table(feature_lines)
            continue

        index += 1

    source_feature = next((feature for feature in features if feature.get("type") == "source"), {})
    genome_record = {
        **locus_info,
        "genome_accession": accession,
        "genome_version": version or accession,
        "definition": definition,
        "organism": organism,
        "taxonomy": taxonomy,
        "virus_tax_id": parse_taxid(source_feature),
        "source_mol_type": first_qualifier(source_feature, "mol_type"),
        "source_host": first_qualifier(source_feature, "host"),
        "source_isolate": first_qualifier(source_feature, "isolate"),
        "source_segment": first_qualifier(source_feature, "segment"),
        "source_geo_loc_name": first_qualifier(source_feature, "geo_loc_name"),
        "source_collection_date": first_qualifier(source_feature, "collection_date"),
        "source_note": first_qualifier(source_feature, "note"),
    }

    cds_records: list[dict[str, object]] = []
    for feature in features:
        protein_accession = first_qualifier(feature, "protein_id")
        if not protein_accession:
            continue
        location_raw = str(feature.get("location", ""))
        location_info = parse_location(location_raw)
        translation = first_qualifier(feature, "translation")
        cds_records.append(
            {
                "genome_accession": genome_record["genome_accession"],
                "genome_version": genome_record["genome_version"],
                "virus_tax_id": genome_record["virus_tax_id"],
                "organism": genome_record["organism"],
                "source_segment": genome_record["source_segment"],
                "feature_type": str(feature.get("type", "")),
                "protein_accession": protein_accession,
                "gene": first_qualifier(feature, "gene"),
                "locus_tag": first_qualifier(feature, "locus_tag"),
                "product": first_qualifier(feature, "product"),
                "location_raw": location_raw,
                **location_info,
                "codon_start": first_qualifier(feature, "codon_start"),
                "translation_length_aa": len(translation) if translation else "",
            }
        )

    return genome_record, cds_records


def write_genbank_tables(input_path: Path, genome_output: Path, cds_output: Path) -> tuple[int, int]:
    genome_writer, genome_handle = open_tsv_writer(genome_output, GENOME_FIELDS)
    cds_writer, cds_handle = open_tsv_writer(cds_output, CDS_FIELDS)
    genome_count = 0
    cds_count = 0

    try:
        for record_lines in iter_genbank_records(input_path):
            genome_record, cds_records = parse_genbank_record(record_lines)
            genome_writer.writerow(genome_record)
            genome_count += 1
            for cds_record in cds_records:
                cds_writer.writerow(cds_record)
                cds_count += 1
    finally:
        genome_handle.close()
        cds_handle.close()

    return genome_count, cds_count


def main() -> int:
    args = parse_args()
    root = repo_root()

    protein_fasta = (root / args.protein_fasta).resolve()
    genome_fasta = (root / args.genome_fasta).resolve()
    genbank_path = (root / args.genbank).resolve()
    output_dir = (root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    proteins_output = output_dir / "viral_proteins.tsv.gz"
    genome_sequences_output = output_dir / "viral_genome_sequences.tsv.gz"
    genomes_output = output_dir / "viral_genomes.tsv.gz"
    cds_output = output_dir / "viral_cds.tsv.gz"

    protein_count = write_protein_table(protein_fasta, proteins_output)
    genome_sequence_count = write_genome_sequence_table(genome_fasta, genome_sequences_output)
    genome_count, cds_count = write_genbank_tables(genbank_path, genomes_output, cds_output)

    report = {
        "generated_at": timestamp(),
        "inputs": {
            "protein_fasta": str(protein_fasta),
            "genome_fasta": str(genome_fasta),
            "genbank": str(genbank_path),
        },
        "outputs": {
            "viral_proteins": str(proteins_output),
            "viral_genome_sequences": str(genome_sequences_output),
            "viral_genomes": str(genomes_output),
            "viral_cds": str(cds_output),
        },
        "counts": {
            "protein_records": protein_count,
            "genome_sequence_records": genome_sequence_count,
            "genome_records": genome_count,
            "cds_records": cds_count,
        },
    }

    report_path = output_dir / "parse_refseq_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(report["counts"], indent=2, ensure_ascii=False))
    print(f"Wrote report to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
