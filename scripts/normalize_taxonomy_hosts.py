from __future__ import annotations

import argparse
import csv
import gzip
import json
from datetime import datetime, timezone
from pathlib import Path


OBSERVED_TAXONOMY_FIELDS = [
    "taxid",
    "scientific_name",
    "rank",
    "parent_taxid",
    "lineage_taxids",
    "lineage_names",
    "lineage_ranks",
    "lineage_source",
]

PAIR_FIELDS = [
    "virus_tax_id_raw",
    "virus_tax_id",
    "virus_name_raw",
    "virus_name_standard",
    "virus_lineage_raw",
    "virus_lineage_standard",
    "refseq_ids_json",
    "refseq_id_count",
    "kegg_genome",
    "kegg_disease",
    "disease",
    "host_tax_id_raw",
    "host_tax_id",
    "host_name_raw",
    "host_name_standard",
    "host_lineage_raw",
    "host_lineage_standard",
    "pmid",
    "evidence",
    "sample_type",
    "source_organism",
]

REFSEQ_LINK_FIELDS = [
    "refseq_accession",
    "virus_tax_id",
    "virus_name_standard",
    "host_tax_id",
    "host_name_standard",
    "host_lineage_standard",
    "pmid",
    "evidence",
    "sample_type",
    "source_organism",
]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize taxonomy and Virus-Host DB joins for ViruFunc-FM.")
    parser.add_argument(
        "--genomes",
        default="data/processed/refseq/viral_genomes.tsv.gz",
        help="Processed RefSeq genome metadata table",
    )
    parser.add_argument(
        "--virushostdb-pairs",
        default="data/raw/virushostdb/virushostdb.tsv",
        help="Virus-Host DB association table",
    )
    parser.add_argument(
        "--virushostdb-lineage",
        default="data/raw/virushostdb/taxid2lineage_full_VH.tsv",
        help="Virus-Host DB lineage table",
    )
    parser.add_argument(
        "--names",
        default="data/interim/ncbi_taxdump/names.dmp",
        help="NCBI names.dmp file",
    )
    parser.add_argument(
        "--nodes",
        default="data/interim/ncbi_taxdump/nodes.dmp",
        help="NCBI nodes.dmp file",
    )
    parser.add_argument(
        "--merged",
        default="data/interim/ncbi_taxdump/merged.dmp",
        help="NCBI merged.dmp file",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed/taxonomy",
        help="Directory for normalized taxonomy tables",
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


def parse_dmp_line(line: str) -> list[str]:
    line = line.rstrip("\n")
    parts = line.split("\t|\t")
    cleaned = [part.strip() for part in parts]
    if cleaned:
        cleaned[-1] = cleaned[-1].rstrip("\t|").strip()
    return cleaned


def canonicalize_taxid(taxid: str, merged_map: dict[str, str]) -> str:
    current = taxid.strip()
    seen: set[str] = set()
    while current in merged_map and current not in seen:
        seen.add(current)
        current = merged_map[current]
    return current


def parse_refseq_ids(value: str) -> list[str]:
    if not value.strip():
        return []
    normalized = value.replace(";", ",")
    items = [item.strip() for item in normalized.split(",")]
    return [item for item in items if item]


def load_lineage_fallback(path: Path) -> dict[str, dict[str, str]]:
    fallback: dict[str, dict[str, str]] = {}
    with open_text(path) as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t", 2)
            if len(parts) < 3:
                continue
            taxid, name, lineage = parts
            fallback[taxid.strip()] = {
                "name": name.strip(),
                "lineage": lineage.strip(),
            }
    return fallback


def load_merged(path: Path) -> dict[str, str]:
    merged: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            parts = parse_dmp_line(raw_line)
            if len(parts) >= 2:
                merged[parts[0]] = parts[1]
    return merged


def load_nodes(path: Path) -> dict[str, tuple[str, str]]:
    nodes: dict[str, tuple[str, str]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            parts = parse_dmp_line(raw_line)
            if len(parts) >= 3:
                nodes[parts[0]] = (parts[1], parts[2])
    return nodes


def collect_observed_taxids(genomes_path: Path, pairs_path: Path, merged_map: dict[str, str]) -> set[str]:
    observed: set[str] = set()

    for row in read_tsv(genomes_path):
        taxid = row.get("virus_tax_id", "").strip()
        if taxid:
            observed.add(canonicalize_taxid(taxid, merged_map))

    with pairs_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            virus_tax_id = row.get("virus tax id", "").strip()
            host_tax_id = row.get("host tax id", "").strip()
            if virus_tax_id:
                observed.add(canonicalize_taxid(virus_tax_id, merged_map))
            if host_tax_id:
                observed.add(canonicalize_taxid(host_tax_id, merged_map))

    return {taxid for taxid in observed if taxid}


def lineage_taxids(taxid: str, nodes: dict[str, tuple[str, str]], cache: dict[str, list[str]]) -> list[str]:
    if taxid in cache:
        return cache[taxid]

    lineage: list[str] = []
    current = taxid
    seen: set[str] = set()

    while current and current not in seen and current in nodes:
        seen.add(current)
        lineage.append(current)
        parent_taxid, _ = nodes[current]
        if parent_taxid == current:
            break
        current = parent_taxid

    lineage.reverse()
    cache[taxid] = lineage
    return lineage


def load_scientific_names(path: Path, wanted_taxids: set[str]) -> dict[str, str]:
    names: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            parts = parse_dmp_line(raw_line)
            if len(parts) < 4:
                continue
            taxid, name_txt, _, name_class = parts[:4]
            if taxid in wanted_taxids and name_class == "scientific name":
                names[taxid] = name_txt
    return names


def write_observed_taxonomy(
    output_path: Path,
    observed_taxids: set[str],
    nodes: dict[str, tuple[str, str]],
    names: dict[str, str],
    lineage_fallback: dict[str, dict[str, str]],
) -> dict[str, dict[str, str]]:
    lineage_cache: dict[str, list[str]] = {}
    taxonomy_lookup: dict[str, dict[str, str]] = {}
    writer, handle = open_tsv_writer(output_path, OBSERVED_TAXONOMY_FIELDS)

    try:
        for taxid in sorted(observed_taxids, key=lambda value: int(value)):
            lineage_ids = lineage_taxids(taxid, nodes, lineage_cache)
            lineage_names = [names.get(item, lineage_fallback.get(item, {}).get("name", "")) for item in lineage_ids]
            lineage_ranks = [nodes.get(item, ("", ""))[1] for item in lineage_ids]
            fallback = lineage_fallback.get(taxid, {})
            scientific_name = names.get(taxid, fallback.get("name", ""))
            lineage_name_text = "; ".join(name for name in lineage_names if name)
            lineage_source = "ncbi"
            if not lineage_name_text and fallback.get("lineage"):
                lineage_name_text = fallback["lineage"]
                lineage_source = "virushostdb"
            row = {
                "taxid": taxid,
                "scientific_name": scientific_name,
                "rank": nodes.get(taxid, ("", ""))[1],
                "parent_taxid": nodes.get(taxid, ("", ""))[0],
                "lineage_taxids": "; ".join(lineage_ids),
                "lineage_names": lineage_name_text,
                "lineage_ranks": "; ".join(rank for rank in lineage_ranks if rank),
                "lineage_source": lineage_source,
            }
            writer.writerow(row)
            taxonomy_lookup[taxid] = row
    finally:
        handle.close()

    return taxonomy_lookup


def write_standardized_pairs(
    pairs_path: Path,
    output_pairs: Path,
    output_links: Path,
    merged_map: dict[str, str],
    taxonomy_lookup: dict[str, dict[str, str]],
    lineage_fallback: dict[str, dict[str, str]],
) -> tuple[int, int]:
    pair_writer, pair_handle = open_tsv_writer(output_pairs, PAIR_FIELDS)
    link_writer, link_handle = open_tsv_writer(output_links, REFSEQ_LINK_FIELDS)

    pair_count = 0
    link_count = 0

    try:
        with pairs_path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                virus_tax_id_raw = row.get("virus tax id", "").strip()
                host_tax_id_raw = row.get("host tax id", "").strip()
                virus_tax_id = canonicalize_taxid(virus_tax_id_raw, merged_map) if virus_tax_id_raw else ""
                host_tax_id = canonicalize_taxid(host_tax_id_raw, merged_map) if host_tax_id_raw else ""

                virus_taxonomy = taxonomy_lookup.get(virus_tax_id, {})
                host_taxonomy = taxonomy_lookup.get(host_tax_id, {})
                virus_fallback = lineage_fallback.get(virus_tax_id, {})
                host_fallback = lineage_fallback.get(host_tax_id, {})

                refseq_ids = parse_refseq_ids(row.get("refseq id", ""))
                output_row = {
                    "virus_tax_id_raw": virus_tax_id_raw,
                    "virus_tax_id": virus_tax_id,
                    "virus_name_raw": row.get("virus name", "").strip(),
                    "virus_name_standard": virus_taxonomy.get("scientific_name") or virus_fallback.get("name", ""),
                    "virus_lineage_raw": row.get("virus lineage", "").strip(),
                    "virus_lineage_standard": virus_taxonomy.get("lineage_names") or virus_fallback.get("lineage", ""),
                    "refseq_ids_json": json.dumps(refseq_ids, ensure_ascii=False),
                    "refseq_id_count": len(refseq_ids),
                    "kegg_genome": row.get("KEGG GENOME", "").strip(),
                    "kegg_disease": row.get("KEGG DISEASE", "").strip(),
                    "disease": row.get("DISEASE", "").strip(),
                    "host_tax_id_raw": host_tax_id_raw,
                    "host_tax_id": host_tax_id,
                    "host_name_raw": row.get("host name", "").strip(),
                    "host_name_standard": host_taxonomy.get("scientific_name") or host_fallback.get("name", ""),
                    "host_lineage_raw": row.get("host lineage", "").strip(),
                    "host_lineage_standard": host_taxonomy.get("lineage_names") or host_fallback.get("lineage", ""),
                    "pmid": row.get("pmid", "").strip(),
                    "evidence": row.get("evidence", "").strip(),
                    "sample_type": row.get("sample type", "").strip(),
                    "source_organism": row.get("source organism", "").strip(),
                }
                pair_writer.writerow(output_row)
                pair_count += 1

                for refseq_accession in refseq_ids:
                    link_writer.writerow(
                        {
                            "refseq_accession": refseq_accession,
                            "virus_tax_id": output_row["virus_tax_id"],
                            "virus_name_standard": output_row["virus_name_standard"],
                            "host_tax_id": output_row["host_tax_id"],
                            "host_name_standard": output_row["host_name_standard"],
                            "host_lineage_standard": output_row["host_lineage_standard"],
                            "pmid": output_row["pmid"],
                            "evidence": output_row["evidence"],
                            "sample_type": output_row["sample_type"],
                            "source_organism": output_row["source_organism"],
                        }
                    )
                    link_count += 1
    finally:
        pair_handle.close()
        link_handle.close()

    return pair_count, link_count


def main() -> int:
    args = parse_args()
    root = repo_root()

    genomes_path = (root / args.genomes).resolve()
    pairs_path = (root / args.virushostdb_pairs).resolve()
    lineage_path = (root / args.virushostdb_lineage).resolve()
    names_path = (root / args.names).resolve()
    nodes_path = (root / args.nodes).resolve()
    merged_path = (root / args.merged).resolve()
    output_dir = (root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    merged_map = load_merged(merged_path)
    nodes = load_nodes(nodes_path)
    lineage_fallback = load_lineage_fallback(lineage_path)
    observed_taxids = collect_observed_taxids(genomes_path, pairs_path, merged_map)

    lineage_cache: dict[str, list[str]] = {}
    all_needed_taxids = set(observed_taxids)
    for taxid in observed_taxids:
        all_needed_taxids.update(lineage_taxids(taxid, nodes, lineage_cache))

    names = load_scientific_names(names_path, all_needed_taxids)

    observed_taxonomy_path = output_dir / "observed_taxonomy.tsv.gz"
    standardized_pairs_path = output_dir / "virus_host_pairs.standardized.tsv.gz"
    refseq_links_path = output_dir / "virus_host_refseq_links.tsv.gz"

    taxonomy_lookup = write_observed_taxonomy(
        observed_taxonomy_path,
        observed_taxids,
        nodes,
        names,
        lineage_fallback,
    )
    pair_count, link_count = write_standardized_pairs(
        pairs_path,
        standardized_pairs_path,
        refseq_links_path,
        merged_map,
        taxonomy_lookup,
        lineage_fallback,
    )

    report = {
        "generated_at": timestamp(),
        "inputs": {
            "genomes": str(genomes_path),
            "virushostdb_pairs": str(pairs_path),
            "virushostdb_lineage": str(lineage_path),
            "names": str(names_path),
            "nodes": str(nodes_path),
            "merged": str(merged_path),
        },
        "outputs": {
            "observed_taxonomy": str(observed_taxonomy_path),
            "virus_host_pairs_standardized": str(standardized_pairs_path),
            "virus_host_refseq_links": str(refseq_links_path),
        },
        "counts": {
            "observed_taxids": len(observed_taxids),
            "standardized_pairs": pair_count,
            "refseq_links": link_count,
        },
    }

    report_path = output_dir / "normalize_taxonomy_hosts_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(report["counts"], indent=2, ensure_ascii=False))
    print(f"Wrote report to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
