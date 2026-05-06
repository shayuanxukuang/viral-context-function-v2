from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


PROTEIN_INDEX_FIELDS = [
    "protein_accession",
    "protein_length_aa",
    "protein_sequence_sha256",
    "protein_sequence",
    "protein_description",
    "protein_organism",
    "genome_accession",
    "genome_version",
    "virus_tax_id",
    "virus_name",
    "virus_lineage",
    "source_mol_type",
    "source_segment",
    "source_isolate",
    "source_host",
    "source_geo_loc_name",
    "protein_feature_type",
    "cds_gene",
    "cds_locus_tag",
    "cds_product",
    "cds_location_raw",
    "cds_location_kind",
    "cds_start",
    "cds_end",
    "cds_strand",
    "cds_part_count",
    "cds_partial_left",
    "cds_partial_right",
    "host_join_strategy",
    "host_record_count",
    "host_tax_ids_json",
    "host_names_json",
    "host_lineages_json",
    "host_evidence_json",
    "host_pmids_json",
    "host_sample_types_json",
    "host_source_organisms_json",
    "reviewed_uniprot_entries_for_taxon",
    "reviewed_uniprot_entries_with_go_for_taxon",
    "reviewed_uniprot_entries_with_interpro_for_taxon",
    "reviewed_uniprot_entries_with_ec_for_taxon",
]

GENOME_INDEX_FIELDS = [
    "genome_accession",
    "genome_version",
    "virus_tax_id",
    "virus_name",
    "virus_lineage",
    "genome_length_nt",
    "molecule_type",
    "topology",
    "division",
    "source_mol_type",
    "source_segment",
    "source_isolate",
    "source_host",
    "source_geo_loc_name",
    "protein_count",
    "host_join_strategy",
    "host_record_count",
    "host_tax_ids_json",
    "host_names_json",
    "host_lineages_json",
    "host_evidence_json",
    "host_pmids_json",
    "host_sample_types_json",
    "host_source_organisms_json",
    "reviewed_uniprot_entries_for_taxon",
    "reviewed_uniprot_entries_with_go_for_taxon",
    "reviewed_uniprot_entries_with_interpro_for_taxon",
    "reviewed_uniprot_entries_with_ec_for_taxon",
]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build ViruFunc-FM training index tables.")
    parser.add_argument(
        "--proteins",
        default="data/processed/refseq/viral_proteins.tsv.gz",
        help="Processed protein FASTA table",
    )
    parser.add_argument(
        "--genomes",
        default="data/processed/refseq/viral_genomes.tsv.gz",
        help="Processed genome metadata table",
    )
    parser.add_argument(
        "--cds",
        default="data/processed/refseq/viral_cds.tsv.gz",
        help="Processed CDS table",
    )
    parser.add_argument(
        "--host-pairs",
        default="data/processed/taxonomy/virus_host_pairs.standardized.tsv.gz",
        help="Standardized Virus-Host association table",
    )
    parser.add_argument(
        "--host-refseq-links",
        default="data/processed/taxonomy/virus_host_refseq_links.tsv.gz",
        help="Exploded Virus-Host RefSeq link table",
    )
    parser.add_argument(
        "--uniprot",
        default="data/interim/uniprot_reviewed_virus_annotations/uniprot_reviewed_virus_annotations.tsv",
        help="Extracted UniProt reviewed viral annotation table",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed/training",
        help="Directory for training index outputs",
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


def read_tsv(path: Path):
    with open_text(path) as handle:
        yield from csv.DictReader(handle, delimiter="\t")


def sorted_json(values: set[str]) -> str:
    ordered = sorted(value for value in values if value)
    return json.dumps(ordered, ensure_ascii=False)


def new_host_aggregate() -> dict[str, object]:
    return {
        "host_tax_ids": set(),
        "host_names": set(),
        "host_lineages": set(),
        "host_evidence": set(),
        "host_pmids": set(),
        "host_sample_types": set(),
        "host_source_organisms": set(),
        "record_count": 0,
    }


def update_host_aggregate(aggregate: dict[str, object], row: dict[str, str]) -> None:
    aggregate["host_tax_ids"].add(row.get("host_tax_id", "").strip())
    aggregate["host_names"].add(row.get("host_name_standard", "").strip())
    aggregate["host_lineages"].add(row.get("host_lineage_standard", "").strip())
    aggregate["host_evidence"].add(row.get("evidence", "").strip())
    aggregate["host_pmids"].add(row.get("pmid", "").strip())
    aggregate["host_sample_types"].add(row.get("sample_type", "").strip())
    aggregate["host_source_organisms"].add(row.get("source_organism", "").strip())
    aggregate["record_count"] = int(aggregate["record_count"]) + 1


def finalize_host_aggregate(aggregate: dict[str, object] | None) -> dict[str, str]:
    if not aggregate:
        return {
            "host_record_count": "0",
            "host_tax_ids_json": "[]",
            "host_names_json": "[]",
            "host_lineages_json": "[]",
            "host_evidence_json": "[]",
            "host_pmids_json": "[]",
            "host_sample_types_json": "[]",
            "host_source_organisms_json": "[]",
        }

    return {
        "host_record_count": str(aggregate["record_count"]),
        "host_tax_ids_json": sorted_json(aggregate["host_tax_ids"]),
        "host_names_json": sorted_json(aggregate["host_names"]),
        "host_lineages_json": sorted_json(aggregate["host_lineages"]),
        "host_evidence_json": sorted_json(aggregate["host_evidence"]),
        "host_pmids_json": sorted_json(aggregate["host_pmids"]),
        "host_sample_types_json": sorted_json(aggregate["host_sample_types"]),
        "host_source_organisms_json": sorted_json(aggregate["host_source_organisms"]),
    }


def load_host_refseq_aggregates(path: Path) -> dict[str, dict[str, object]]:
    by_refseq: dict[str, dict[str, object]] = {}
    for row in read_tsv(path):
        refseq_accession = row.get("refseq_accession", "").strip()
        if not refseq_accession:
            continue
        aggregate = by_refseq.setdefault(refseq_accession, new_host_aggregate())
        update_host_aggregate(aggregate, row)
    return by_refseq


def load_host_taxid_aggregates(path: Path) -> dict[str, dict[str, object]]:
    by_taxid: dict[str, dict[str, object]] = {}
    for row in read_tsv(path):
        virus_tax_id = row.get("virus_tax_id", "").strip()
        if not virus_tax_id:
            continue
        aggregate = by_taxid.setdefault(virus_tax_id, new_host_aggregate())
        update_host_aggregate(aggregate, row)
    return by_taxid


def load_uniprot_taxon_stats(path: Path) -> dict[str, dict[str, int]]:
    stats: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "reviewed_uniprot_entries_for_taxon": 0,
            "reviewed_uniprot_entries_with_go_for_taxon": 0,
            "reviewed_uniprot_entries_with_interpro_for_taxon": 0,
            "reviewed_uniprot_entries_with_ec_for_taxon": 0,
        }
    )

    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            taxid = row.get("Organism (ID)", "").strip()
            if not taxid:
                continue
            taxon_stats = stats[taxid]
            taxon_stats["reviewed_uniprot_entries_for_taxon"] += 1
            if row.get("Gene Ontology IDs", "").strip():
                taxon_stats["reviewed_uniprot_entries_with_go_for_taxon"] += 1
            if row.get("InterPro", "").strip():
                taxon_stats["reviewed_uniprot_entries_with_interpro_for_taxon"] += 1
            if row.get("EC number", "").strip():
                taxon_stats["reviewed_uniprot_entries_with_ec_for_taxon"] += 1

    return stats


def load_genomes(path: Path) -> dict[str, dict[str, str]]:
    genomes: dict[str, dict[str, str]] = {}
    for row in read_tsv(path):
        genome_version = row.get("genome_version", "").strip()
        if genome_version:
            genomes[genome_version] = row
    return genomes


def load_cds(path: Path) -> tuple[dict[str, dict[str, str]], dict[str, int]]:
    cds_by_protein: dict[str, dict[str, str]] = {}
    protein_counts_by_genome: dict[str, int] = defaultdict(int)
    for row in read_tsv(path):
        protein_accession = row.get("protein_accession", "").strip()
        genome_version = row.get("genome_version", "").strip()
        if genome_version:
            protein_counts_by_genome[genome_version] += 1
        if protein_accession and protein_accession not in cds_by_protein:
            cds_by_protein[protein_accession] = row
    return cds_by_protein, protein_counts_by_genome


def choose_host_join(
    genome_accession: str,
    virus_tax_id: str,
    refseq_hosts: dict[str, dict[str, object]],
    taxid_hosts: dict[str, dict[str, object]],
) -> tuple[str, dict[str, str]]:
    if genome_accession and genome_accession in refseq_hosts:
        return "refseq_accession", finalize_host_aggregate(refseq_hosts[genome_accession])
    if virus_tax_id and virus_tax_id in taxid_hosts:
        return "virus_tax_id", finalize_host_aggregate(taxid_hosts[virus_tax_id])
    return "none", finalize_host_aggregate(None)


def main() -> int:
    args = parse_args()
    root = repo_root()

    proteins_path = (root / args.proteins).resolve()
    genomes_path = (root / args.genomes).resolve()
    cds_path = (root / args.cds).resolve()
    host_pairs_path = (root / args.host_pairs).resolve()
    host_refseq_links_path = (root / args.host_refseq_links).resolve()
    uniprot_path = (root / args.uniprot).resolve()
    output_dir = (root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    refseq_hosts = load_host_refseq_aggregates(host_refseq_links_path)
    taxid_hosts = load_host_taxid_aggregates(host_pairs_path)
    uniprot_stats = load_uniprot_taxon_stats(uniprot_path)
    genomes = load_genomes(genomes_path)
    cds_by_protein, protein_counts_by_genome = load_cds(cds_path)

    protein_index_path = output_dir / "viral_protein_training_index.tsv.gz"
    genome_index_path = output_dir / "viral_genome_training_index.tsv.gz"

    protein_writer, protein_handle = open_tsv_writer(protein_index_path, PROTEIN_INDEX_FIELDS)
    genome_writer, genome_handle = open_tsv_writer(genome_index_path, GENOME_INDEX_FIELDS)

    protein_count = 0
    genome_count = 0
    missing_genome_for_proteins = 0
    missing_feature_for_proteins = 0
    host_join_counts: dict[str, int] = defaultdict(int)

    try:
        for protein_row in read_tsv(proteins_path):
            protein_accession = protein_row.get("protein_accession", "").strip()
            cds_row = cds_by_protein.get(protein_accession, {})
            genome_version = cds_row.get("genome_version", "").strip()
            genome_row = genomes.get(genome_version, {})

            genome_accession = genome_row.get("genome_accession", "").strip() or cds_row.get("genome_accession", "").strip()
            virus_tax_id = genome_row.get("virus_tax_id", "").strip() or cds_row.get("virus_tax_id", "").strip()
            host_join_strategy, host_payload = choose_host_join(genome_accession, virus_tax_id, refseq_hosts, taxid_hosts)
            host_join_counts[host_join_strategy] += 1
            if not cds_row:
                missing_feature_for_proteins += 1
            if not genome_version:
                missing_genome_for_proteins += 1
            uniprot_payload = uniprot_stats.get(
                virus_tax_id,
                {
                    "reviewed_uniprot_entries_for_taxon": 0,
                    "reviewed_uniprot_entries_with_go_for_taxon": 0,
                    "reviewed_uniprot_entries_with_interpro_for_taxon": 0,
                    "reviewed_uniprot_entries_with_ec_for_taxon": 0,
                },
            )

            protein_writer.writerow(
                {
                    "protein_accession": protein_accession,
                    "protein_length_aa": protein_row.get("protein_length_aa", ""),
                    "protein_sequence_sha256": protein_row.get("protein_sequence_sha256", ""),
                    "protein_sequence": protein_row.get("protein_sequence", ""),
                    "protein_description": protein_row.get("protein_description", ""),
                    "protein_organism": protein_row.get("protein_organism", ""),
                    "genome_accession": genome_accession,
                    "genome_version": genome_version,
                    "virus_tax_id": virus_tax_id,
                    "virus_name": genome_row.get("organism", ""),
                    "virus_lineage": genome_row.get("taxonomy", ""),
                    "source_mol_type": genome_row.get("source_mol_type", ""),
                    "source_segment": genome_row.get("source_segment", "") or cds_row.get("source_segment", ""),
                    "source_isolate": genome_row.get("source_isolate", ""),
                    "source_host": genome_row.get("source_host", ""),
                    "source_geo_loc_name": genome_row.get("source_geo_loc_name", ""),
                    "protein_feature_type": cds_row.get("feature_type", ""),
                    "cds_gene": cds_row.get("gene", ""),
                    "cds_locus_tag": cds_row.get("locus_tag", ""),
                    "cds_product": cds_row.get("product", ""),
                    "cds_location_raw": cds_row.get("location_raw", ""),
                    "cds_location_kind": cds_row.get("location_kind", ""),
                    "cds_start": cds_row.get("cds_start", ""),
                    "cds_end": cds_row.get("cds_end", ""),
                    "cds_strand": cds_row.get("cds_strand", ""),
                    "cds_part_count": cds_row.get("location_part_count", ""),
                    "cds_partial_left": cds_row.get("location_partial_left", ""),
                    "cds_partial_right": cds_row.get("location_partial_right", ""),
                    "host_join_strategy": host_join_strategy,
                    **host_payload,
                    **{key: str(value) for key, value in uniprot_payload.items()},
                }
            )
            protein_count += 1

        for genome_version, genome_row in genomes.items():
            genome_accession = genome_row.get("genome_accession", "").strip()
            virus_tax_id = genome_row.get("virus_tax_id", "").strip()
            host_join_strategy, host_payload = choose_host_join(genome_accession, virus_tax_id, refseq_hosts, taxid_hosts)
            uniprot_payload = uniprot_stats.get(
                virus_tax_id,
                {
                    "reviewed_uniprot_entries_for_taxon": 0,
                    "reviewed_uniprot_entries_with_go_for_taxon": 0,
                    "reviewed_uniprot_entries_with_interpro_for_taxon": 0,
                    "reviewed_uniprot_entries_with_ec_for_taxon": 0,
                },
            )

            genome_writer.writerow(
                {
                    "genome_accession": genome_accession,
                    "genome_version": genome_version,
                    "virus_tax_id": virus_tax_id,
                    "virus_name": genome_row.get("organism", ""),
                    "virus_lineage": genome_row.get("taxonomy", ""),
                    "genome_length_nt": genome_row.get("genome_length_nt", ""),
                    "molecule_type": genome_row.get("molecule_type", ""),
                    "topology": genome_row.get("topology", ""),
                    "division": genome_row.get("division", ""),
                    "source_mol_type": genome_row.get("source_mol_type", ""),
                    "source_segment": genome_row.get("source_segment", ""),
                    "source_isolate": genome_row.get("source_isolate", ""),
                    "source_host": genome_row.get("source_host", ""),
                    "source_geo_loc_name": genome_row.get("source_geo_loc_name", ""),
                    "protein_count": str(protein_counts_by_genome.get(genome_version, 0)),
                    "host_join_strategy": host_join_strategy,
                    **host_payload,
                    **{key: str(value) for key, value in uniprot_payload.items()},
                }
            )
            genome_count += 1
    finally:
        protein_handle.close()
        genome_handle.close()

    report = {
        "generated_at": timestamp(),
        "inputs": {
            "proteins": str(proteins_path),
            "genomes": str(genomes_path),
            "cds": str(cds_path),
            "host_pairs": str(host_pairs_path),
            "host_refseq_links": str(host_refseq_links_path),
            "uniprot": str(uniprot_path),
        },
        "outputs": {
            "protein_index": str(protein_index_path),
            "genome_index": str(genome_index_path),
        },
        "counts": {
            "protein_index_rows": protein_count,
            "genome_index_rows": genome_count,
            "refseq_host_accessions": len(refseq_hosts),
            "taxid_host_groups": len(taxid_hosts),
            "uniprot_taxa": len(uniprot_stats),
            "protein_rows_missing_feature_join": missing_feature_for_proteins,
            "protein_rows_missing_genome_join": missing_genome_for_proteins,
            "host_join_refseq_accession": host_join_counts.get("refseq_accession", 0),
            "host_join_virus_tax_id": host_join_counts.get("virus_tax_id", 0),
            "host_join_none": host_join_counts.get("none", 0),
        },
    }

    report_path = output_dir / "build_training_index_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(report["counts"], indent=2, ensure_ascii=False))
    print(f"Wrote report to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
