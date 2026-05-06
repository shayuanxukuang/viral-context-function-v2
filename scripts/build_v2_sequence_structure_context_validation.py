#!/usr/bin/env python3
"""Merge sequence, structure, genome-context, and module evidence for V2 cases.

The script builds manuscript-ready post hoc evidence tables for the 27
high-context-gain candidates and optional matched controls prepared by
prepare_v2_sequence_structure_validation.py. It does not use product text,
neighbor labels, MMseqs2 hits, Foldseek hits, or annotation-derived priors as
de novo model inputs.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


DEFAULT_TARGETS = Path("runs/v2_sequence_structure_validation/targets/validation_targets.tsv")
DEFAULT_CANDIDATES = Path(
    "artifacts/return/v2_plos_cb_supplementary_package_20260504/"
    "supplementary_tables/S16_high_context_gain_candidates.tsv"
)
DEFAULT_MODULE_CANDIDATES = Path("artifacts/return/extracted_v2_20260430_100225/module_discovery/module_candidates.tsv")
DEFAULT_MODULE_CLUSTERS = Path(
    "artifacts/return/v2_plos_cb_supplementary_package_20260504/supplementary_tables/S17_module_clusters.tsv"
)
DEFAULT_HOMOLOGY = Path(
    "artifacts/return/v2_plos_cb_supplementary_package_20260504/"
    "supplementary_tables/S21_homology_top_hit_assignments.tsv"
)
DEFAULT_STRUCTURE_SUMMARY = Path("artifacts/return/targeted_structure_validation_foldseek_summary/esmfold_quality.tsv")
DEFAULT_FOLDSEEK_HITS = Path(
    "artifacts/return/extracted_structure_foldseek_20260428_205525/"
    "runs/structure_validation_shortlist/foldseek/pdb_hits.tsv"
)

ACCESSION_COLUMNS = ("protein_accession", "protein_id", "candidate_id", "query")
LABEL_COLUMNS = ("predicted_label", "candidate_label", "top_label")
GENOME_COLUMNS = ("genome_id", "genome_version")
CONTEXT_PROB_COLUMNS = ("p_context", "top_probability_calibrated", "calibrated_probability")
PROTEIN_PROB_COLUMNS = ("p_protein_only", "p_protein_only_estimated")
CONTEXT_GAIN_COLUMNS = ("delta_p", "context_gain")

FOLDSEEK_FIELDS = [
    "query",
    "target",
    "evalue",
    "bits",
    "prob",
    "alnlen",
    "pident",
    "lddt",
    "alntmscore",
    "qtmscore",
    "ttmscore",
    "taxid",
    "taxname",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS, help="validation_targets.tsv from preparation.")
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES, help="Fallback S16 candidate TSV.")
    parser.add_argument("--module-candidates", type=Path, default=DEFAULT_MODULE_CANDIDATES)
    parser.add_argument("--module-clusters", type=Path, default=DEFAULT_MODULE_CLUSTERS)
    parser.add_argument("--homology-hits", type=Path, default=DEFAULT_HOMOLOGY)
    parser.add_argument("--homology-scheme", default="family_holdout")
    parser.add_argument("--homology-subset", default="all_test")
    parser.add_argument("--structure-summary", type=Path, default=DEFAULT_STRUCTURE_SUMMARY)
    parser.add_argument("--foldseek-hits", type=Path, default=DEFAULT_FOLDSEEK_HITS)
    parser.add_argument("--pdb-dir", type=Path, help="Optional predicted PDB directory for pLDDT parsing.")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/v2_sequence_structure_validation/evidence"))
    parser.add_argument("--top-foldseek-hits", type=int, default=20)
    parser.add_argument("--figure-top-n", type=int, default=12)
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_path(path: Path | None, root: Path) -> Path | None:
    if path is None:
        return None
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("r", encoding="utf-8", newline="")


def first_present(row: dict[str, Any], columns: Iterable[str], default: str = "") -> str:
    for column in columns:
        value = row.get(column)
        if value is not None and str(value).strip() != "":
            return str(value).strip()
    return default


def as_float(value: Any, default: float = math.nan) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def clean_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.12g}"
    return str(value)


def read_tsv(path: Path | None, required: bool = False, table_name: str = "table") -> list[dict[str, str]]:
    if path is None or not path.exists():
        if required:
            raise SystemExit(f"Required {table_name} not found: {path}")
        return []
    with open_text(path) as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: clean_cell(row.get(field, "")) for field in fieldnames})


def parse_json_list(value: str) -> list[str]:
    if not value:
        return []
    try:
        parsed = json.loads(value)
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
    except json.JSONDecodeError:
        return []
    return []


def normalize_target_row(row: dict[str, str], target_type: str = "high_context_candidate") -> dict[str, Any]:
    p_context = first_present(row, CONTEXT_PROB_COLUMNS)
    delta = first_present(row, CONTEXT_GAIN_COLUMNS)
    p_protein = first_present(row, PROTEIN_PROB_COLUMNS)
    if not p_protein and p_context and delta:
        p_protein_float = as_float(p_context) - as_float(delta)
        if not math.isnan(p_protein_float):
            p_protein = f"{max(0.0, min(1.0, p_protein_float)):.12g}"
    module_cluster_id = first_present(row, ("module_cluster_id", "cluster_id"))
    if module_cluster_id.lower() in {"-1", "none", "nan", "null"}:
        module_cluster_id = ""
    return {
        "target_type": row.get("target_type", target_type),
        "matched_candidate_id": row.get("matched_candidate_id", ""),
        "match_rank": row.get("match_rank", ""),
        "protein_accession": first_present(row, ACCESSION_COLUMNS),
        "predicted_label": first_present(row, LABEL_COLUMNS),
        "p_protein_only": p_protein,
        "p_context": p_context,
        "delta_p": delta,
        "family": first_present(row, ("family", "virus_family")),
        "host_group": first_present(row, ("host_group", "host_supergroup")),
        "genome_id": first_present(row, GENOME_COLUMNS),
        "description": first_present(row, ("description", "protein_description", "cds_product")),
        "hypothetical_or_uncharacterized": first_present(
            row, ("hypothetical_or_uncharacterized", "hypothetical_or_unknown"), "0"
        ),
        "module_cluster_id": module_cluster_id,
        "exact_transfer_flag": first_present(row, ("exact_transfer_flag",)),
        "sequence_length_aa": first_present(row, ("sequence_length_aa", "protein_length_aa")),
        "validation_gate_status": first_present(row, ("validation_gate_status", "fdr_gate_status")),
    }


def load_targets(targets_path: Path, candidates_path: Path) -> list[dict[str, Any]]:
    if targets_path.exists():
        rows = read_tsv(targets_path, required=True, table_name="validation targets")
        targets = [normalize_target_row(row) for row in rows]
    else:
        rows = read_tsv(candidates_path, required=True, table_name="high-context-gain candidates")
        targets = [normalize_target_row(row, "high_context_candidate") for row in rows]
    targets = [row for row in targets if row.get("protein_accession")]
    if not targets:
        raise SystemExit("No target accessions were found in the target or candidate table.")
    return targets


def index_first(rows: Iterable[dict[str, str]], keys: Iterable[str]) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for row in rows:
        for key in keys:
            value = row.get(key, "")
            if value:
                out.setdefault(value, row)
                break
    return out


def load_homology(path: Path, scheme: str, subset: str) -> dict[str, dict[str, str]]:
    rows = read_tsv(path, required=False, table_name="MMseqs2 top-hit table")
    preferred: dict[str, dict[str, str]] = {}
    fallback: dict[str, dict[str, str]] = {}
    for row in rows:
        query = row.get("query", "")
        if not query:
            continue
        fallback.setdefault(query, row)
        if (not scheme or row.get("scheme") == scheme) and (not subset or row.get("subset") == subset):
            preferred.setdefault(query, row)
    merged = dict(fallback)
    merged.update(preferred)
    return merged


def sequence_evidence_status(hit: dict[str, str] | None) -> str:
    if not hit:
        return "no_mmseqs2_top_hit_available"
    identity = as_float(hit.get("pident"))
    target_labels = parse_json_list(hit.get("target_labels", ""))
    suffix = "" if target_labels else "_without_target_function_label"
    if identity >= 90:
        return "near_exact_sequence_hit" + suffix
    if identity >= 50:
        return "high_identity_sequence_hit" + suffix
    if identity >= 30:
        return "remote_sequence_hit" + suffix
    if not math.isnan(identity):
        return "weak_sequence_hit" + suffix
    return "sequence_hit_identity_missing" + suffix


def normalize_query_id(value: str) -> str:
    text = str(value).replace("\\", "/").split("/")[-1]
    text = text.removesuffix(".pdb").removesuffix(".cif")
    return text.split("|")[0].split()[0]


def foldseek_sort_key(row: dict[str, str]) -> tuple[float, float, float, float]:
    prob = as_float(row.get("prob"), 0.0)
    qtmscore = as_float(row.get("qtmscore"), 0.0)
    bits = as_float(row.get("bits"), 0.0)
    evalue = as_float(row.get("evalue"))
    if math.isnan(evalue):
        evalue_score = 0.0
    elif evalue <= 0:
        evalue_score = 400.0
    else:
        evalue_score = -math.log10(evalue)
    return (prob, qtmscore, bits, evalue_score)


def read_foldseek_rows(path: Path | None) -> list[dict[str, str]]:
    if path is None or not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        first = handle.readline().rstrip("\n")
        if not first:
            return []
        first_parts = first.split("\t")
        if "query" in first_parts and "target" in first_parts:
            handle.seek(0)
            return list(csv.DictReader(handle, delimiter="\t"))
        rows = []
        rows.append({field: first_parts[i] if i < len(first_parts) else "" for i, field in enumerate(FOLDSEEK_FIELDS)})
        reader = csv.reader(handle, delimiter="\t")
        for parts in reader:
            if not parts:
                continue
            rows.append({field: parts[i] if i < len(parts) else "" for i, field in enumerate(FOLDSEEK_FIELDS)})
        return rows


def summarize_foldseek(path: Path | None, top_n: int) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    rows = read_foldseek_rows(path)
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        query = normalize_query_id(row.get("query", ""))
        if query:
            grouped[query].append(row)

    summary: dict[str, dict[str, Any]] = {}
    top_rows: list[dict[str, Any]] = []
    for query, hits in grouped.items():
        hits = sorted(hits, key=foldseek_sort_key, reverse=True)
        kept = hits[:top_n]
        top = kept[0]
        second = kept[1] if len(kept) > 1 else {}
        top_prob = as_float(top.get("prob"), 0.0)
        second_prob = as_float(second.get("prob"), math.nan)
        top_qtm = as_float(top.get("qtmscore"), 0.0)
        second_qtm = as_float(second.get("qtmscore"), math.nan)
        near_top = [
            hit
            for hit in kept
            if as_float(hit.get("prob"), 0.0) >= max(0.0, top_prob - 0.05)
            and as_float(hit.get("qtmscore"), 0.0) >= max(0.0, top_qtm - 0.05)
        ]
        high_conf_hits = [
            hit
            for hit in kept
            if as_float(hit.get("prob"), 0.0) >= 0.9 or as_float(hit.get("qtmscore"), 0.0) >= 0.4
        ]
        tax_counts = Counter(hit.get("taxname", "") for hit in kept if hit.get("taxname", ""))
        tax_diversity = len(tax_counts)
        top_tax_fraction = 0.0
        if kept and top.get("taxname"):
            top_tax_fraction = tax_counts[top.get("taxname", "")] / len(kept)
        extra_near_top_fraction = max(0, len(near_top) - 1) / max(1, len(kept) - 1)
        if len(kept) <= 1 or math.isnan(second_prob):
            margin_component = 0.0
        else:
            margin_component = max(0.0, 1.0 - min(1.0, (top_prob - second_prob) / 0.25))
        diversity_component = min(1.0, max(0, tax_diversity - 1) / max(1, min(len(kept), top_n) - 1))
        ambiguity_index = 0.45 * extra_near_top_fraction + 0.35 * margin_component + 0.20 * diversity_component
        summary[query] = {
            "foldseek_hit_count": len(hits),
            "foldseek_top_n": len(kept),
            "foldseek_high_conf_hit_count": len(high_conf_hits),
            "foldseek_near_top_hit_count": len(near_top),
            "foldseek_taxname_diversity_top_n": tax_diversity,
            "foldseek_top_taxname_fraction": top_tax_fraction,
            "foldseek_ambiguity_index": ambiguity_index,
            "foldseek_top_target": top.get("target", ""),
            "foldseek_top_taxname": top.get("taxname", ""),
            "foldseek_top_evalue": top.get("evalue", ""),
            "foldseek_top_bits": top.get("bits", ""),
            "foldseek_top_prob": top.get("prob", ""),
            "foldseek_second_prob": second.get("prob", ""),
            "foldseek_prob_margin": "" if math.isnan(second_prob) else top_prob - second_prob,
            "foldseek_top_qtmscore": top.get("qtmscore", ""),
            "foldseek_second_qtmscore": second.get("qtmscore", ""),
            "foldseek_qtmscore_margin": "" if math.isnan(second_qtm) else top_qtm - second_qtm,
            "foldseek_top_alnlen": top.get("alnlen", ""),
            "foldseek_top_pident": top.get("pident", ""),
            "foldseek_top_lddt": top.get("lddt", ""),
            "foldseek_top_ttmscore": top.get("ttmscore", ""),
            "foldseek_top_taxid": top.get("taxid", ""),
            "foldseek_top_targets_top_n": ";".join(hit.get("target", "") for hit in kept[:5]),
            "foldseek_top_taxnames_top_n": ";".join(dict.fromkeys(hit.get("taxname", "") for hit in kept if hit.get("taxname", ""))),
        }
        for rank, hit in enumerate(kept, start=1):
            row = {"protein_accession": query, "foldseek_rank": rank}
            row.update(hit)
            top_rows.append(row)
    return summary, top_rows


def parse_pdb_plddt(path: Path) -> dict[str, Any]:
    residues: set[tuple[str, str]] = set()
    ca_plddt: list[float] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.startswith("ATOM"):
                continue
            atom = line[12:16].strip()
            chain = line[21].strip()
            residue = line[22:26].strip()
            residues.add((chain, residue))
            if atom == "CA":
                ca_plddt.append(as_float(line[60:66]))
    values = [value for value in ca_plddt if not math.isnan(value)]
    return {
        "pdb_residue_count": len(residues),
        "pdb_ca_count": len(values),
        "mean_plddt": statistics.mean(values) if values else "",
        "median_plddt": statistics.median(values) if values else "",
        "frac_plddt_ge_70": sum(value >= 70 for value in values) / len(values) if values else "",
        "pdb_path": str(path),
    }


def load_pdb_quality(pdb_dir: Path | None) -> dict[str, dict[str, Any]]:
    if pdb_dir is None or not pdb_dir.exists():
        return {}
    out: dict[str, dict[str, Any]] = {}
    for path in sorted(pdb_dir.glob("*.pdb")):
        out[normalize_query_id(path.name)] = parse_pdb_plddt(path)
    return out


def normalize_cluster_row(row: dict[str, str]) -> dict[str, str]:
    return {
        "module_cluster_id": row.get("cluster_id", ""),
        "module_cluster_size": row.get("cluster_size") or row.get("module_count", ""),
        "module_cluster_family_count": row.get("number_of_families") or row.get("family_count", ""),
        "module_hypothetical_ratio_mean": row.get("hypothetical_ratio_mean", ""),
        "module_structural_membrane_enrichment": row.get("structural_membrane_enrichment")
        or row.get("structural_membrane_vote_fraction_mean", ""),
        "module_neighborhood_consistency": row.get("neighborhood_consistency", ""),
        "module_top_neighborhood_signature": row.get("top_neighborhood_signature", ""),
        "module_priority_score": row.get("priority_score", ""),
    }


def clean_module_cluster_id(value: Any) -> str:
    text = str(value or "").strip()
    return "" if text.lower() in {"", "-1", "none", "nan", "null"} else text


def load_structure_summary(path: Path | None) -> dict[str, dict[str, str]]:
    rows = read_tsv(path, required=False, table_name="structure summary")
    return index_first(rows, ("protein_accession", "query"))


def structure_status(row: dict[str, Any]) -> str:
    hit_count = as_float(row.get("foldseek_hit_count"), 0.0)
    if hit_count <= 0:
        return "no_foldseek_hit_available"
    mean_plddt = as_float(row.get("mean_plddt"))
    top_prob = as_float(row.get("foldseek_top_prob"), 0.0)
    top_qtm = as_float(row.get("foldseek_top_qtmscore"), 0.0)
    coverage = as_float(row.get("foldseek_query_coverage"), 0.0)
    ambiguity = as_float(row.get("foldseek_ambiguity_index"), 1.0)
    high_hits = as_float(row.get("foldseek_high_conf_hit_count"), 0.0)
    if not math.isnan(mean_plddt) and mean_plddt < 50:
        if top_prob >= 0.9:
            return "structure_hit_on_low_confidence_model"
        return "low_model_confidence"
    if top_prob >= 0.9 and top_qtm >= 0.4 and coverage >= 0.35:
        if ambiguity >= 0.5 or high_hits > 3:
            return "structure_consistent_but_ambiguous"
        return "structure_consistent"
    if top_prob >= 0.9 and coverage >= 0.25:
        return "local_structure_consistent"
    if top_prob >= 0.5 or top_qtm >= 0.3:
        return "ambiguous_or_weak_structure_signal"
    return "weak_or_inconclusive"


def structure_claim_phrase(status: str) -> str:
    if status in {"structure_consistent", "local_structure_consistent"}:
        return "structure-consistent post hoc evidence"
    if status == "structure_consistent_but_ambiguous":
        return "structure-consistent but ambiguous post hoc evidence"
    if status == "structure_hit_on_low_confidence_model":
        return "Foldseek hit on low-confidence predicted model"
    if status == "low_model_confidence":
        return "low-confidence predicted structure"
    if status == "no_foldseek_hit_available":
        return "structure evidence not yet available"
    return "ambiguous or weak post hoc structure evidence"


def clamp01(value: float) -> float:
    if math.isnan(value):
        return 0.0
    return max(0.0, min(1.0, value))


def case_priority_score(row: dict[str, Any]) -> float:
    delta = clamp01(as_float(row.get("delta_p"), 0.0))
    p_context = as_float(row.get("p_context"), 0.0)
    pident = as_float(row.get("nearest_homolog_identity"))
    exact_transfer = str(row.get("exact_transfer_flag", "")).lower() in {"1", "true", "yes"}
    module_supported = str(row.get("module_cluster_id", "")).lower() not in {"", "-1", "none", "nan", "null"}
    hypothetical = str(row.get("hypothetical_or_uncharacterized", "")).lower() in {"1", "true", "yes"}
    structure = str(row.get("structure_evidence_status", ""))
    structure_points = {
        "structure_consistent": 1.5,
        "structure_consistent_but_ambiguous": 1.1,
        "local_structure_consistent": 0.9,
        "ambiguous_or_weak_structure_signal": 0.4,
        "structure_hit_on_low_confidence_model": 0.3,
    }.get(structure, 0.0)
    low_sequence_bonus = 1.0 if math.isnan(pident) or pident < 30 else 0.5 if pident < 50 else 0.0
    score = 3.0 * delta
    score += 0.6 if p_context >= 0.8 else 0.0
    score += low_sequence_bonus
    score += structure_points
    score += 0.7 if module_supported else 0.0
    score += 0.4 if hypothetical else 0.0
    score -= 0.8 if exact_transfer else 0.0
    return score


def add_recommendations(rows: list[dict[str, Any]], max_selected: int = 6) -> None:
    candidate_rows = [row for row in rows if row.get("target_type") == "high_context_candidate"]
    candidate_rows.sort(key=lambda row: as_float(row.get("case_study_priority_score"), 0.0), reverse=True)
    selected_ids: set[str] = set()
    used_labels: set[str] = set()
    for row in candidate_rows:
        label = row.get("predicted_label", "")
        if label and label not in used_labels:
            selected_ids.add(row["protein_accession"])
            used_labels.add(label)
        if len(selected_ids) >= max_selected:
            break
    if len(selected_ids) < max_selected:
        for row in candidate_rows:
            selected_ids.add(row["protein_accession"])
            if len(selected_ids) >= max_selected:
                break
    for row in rows:
        if row.get("target_type") != "high_context_candidate":
            row["figure6_recommendation"] = "matched_control_reference"
        elif row.get("protein_accession") in selected_ids:
            row["figure6_recommendation"] = "candidate_case_study_panel"
        else:
            row["figure6_recommendation"] = "supporting_candidate_table"


def build_integrated_rows(
    targets: list[dict[str, Any]],
    module_map: dict[str, dict[str, str]],
    cluster_map: dict[str, dict[str, str]],
    homology_map: dict[str, dict[str, str]],
    foldseek_map: dict[str, dict[str, Any]],
    structure_summary: dict[str, dict[str, str]],
    pdb_quality: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for target in targets:
        accession = target["protein_accession"]
        row: dict[str, Any] = dict(target)

        homology = homology_map.get(accession, {})
        row.update(
            {
                "nearest_homolog_accession": homology.get("target", ""),
                "nearest_homolog_identity": homology.get("pident", ""),
                "nearest_homolog_bits": homology.get("bits", ""),
                "nearest_homolog_target_labels": ",".join(parse_json_list(homology.get("target_labels", ""))),
                "sequence_evidence_status": sequence_evidence_status(homology if homology else None),
            }
        )

        module = module_map.get(accession, {})
        if module:
            row["module_cluster_id"] = clean_module_cluster_id(row.get("module_cluster_id") or module.get("cluster_id", ""))
        else:
            row["module_cluster_id"] = clean_module_cluster_id(row.get("module_cluster_id", ""))
        row.update(
            {
                "module_neighborhood_signature": module.get("neighborhood_signature", ""),
                "module_weak_label_counts": module.get("weak_label_counts_json", ""),
                "module_member_count": module.get("member_count", ""),
                "module_hypothetical_ratio": module.get("hypothetical_ratio", ""),
                "module_structural_membrane_vote_fraction": module.get("structural_membrane_vote_fraction", ""),
                "bio_tm_helix_count": module.get("bio_tm_helix_count", ""),
                "bio_signal_peptide_score": module.get("bio_signal_peptide_score", ""),
                "bio_disorder_score": module.get("bio_disorder_score", ""),
            }
        )
        cluster = cluster_map.get(str(row.get("module_cluster_id", "")), {})
        row.update(cluster)

        structure = structure_summary.get(accession, {})
        pdb = pdb_quality.get(accession, {})
        foldseek = foldseek_map.get(accession, {})
        mean_plddt = first_present(structure, ("mean_plddt",)) or pdb.get("mean_plddt", "")
        median_plddt = first_present(structure, ("median_plddt",)) or pdb.get("median_plddt", "")
        frac70 = first_present(structure, ("frac_plddt_ge_70",)) or pdb.get("frac_plddt_ge_70", "")
        residue_count = first_present(structure, ("residue_count", "ca_count")) or pdb.get("pdb_residue_count", "")
        row.update(
            {
                "mean_plddt": mean_plddt,
                "median_plddt": median_plddt,
                "frac_plddt_ge_70": frac70,
                "structure_residue_count": residue_count,
                "pdb_path": first_present(structure, ("pdb_path",)) or pdb.get("pdb_path", ""),
            }
        )

        if not foldseek and structure:
            alnlen = as_float(structure.get("foldseek_alnlen"), 0.0)
            residues = as_float(row.get("structure_residue_count"), as_float(row.get("sequence_length_aa"), 0.0))
            coverage = alnlen / residues if residues else 0.0
            foldseek = {
                "foldseek_hit_count": 1 if structure.get("foldseek_target") else 0,
                "foldseek_top_n": 1 if structure.get("foldseek_target") else 0,
                "foldseek_high_conf_hit_count": 1
                if as_float(structure.get("foldseek_prob"), 0.0) >= 0.9
                or as_float(structure.get("foldseek_qtmscore"), 0.0) >= 0.4
                else 0,
                "foldseek_near_top_hit_count": 1 if structure.get("foldseek_target") else 0,
                "foldseek_taxname_diversity_top_n": 1 if structure.get("foldseek_taxname") else 0,
                "foldseek_top_taxname_fraction": 1.0 if structure.get("foldseek_taxname") else "",
                "foldseek_ambiguity_index": "",
                "foldseek_top_target": structure.get("foldseek_target", ""),
                "foldseek_top_taxname": structure.get("foldseek_taxname", ""),
                "foldseek_top_evalue": structure.get("foldseek_evalue", ""),
                "foldseek_top_bits": "",
                "foldseek_top_prob": structure.get("foldseek_prob", ""),
                "foldseek_second_prob": "",
                "foldseek_prob_margin": "",
                "foldseek_top_qtmscore": structure.get("foldseek_qtmscore", ""),
                "foldseek_second_qtmscore": "",
                "foldseek_qtmscore_margin": "",
                "foldseek_top_alnlen": structure.get("foldseek_alnlen", ""),
                "foldseek_top_pident": structure.get("foldseek_pident", ""),
                "foldseek_top_lddt": structure.get("foldseek_lddt", ""),
                "foldseek_top_ttmscore": structure.get("foldseek_ttmscore", ""),
                "foldseek_top_taxid": "",
                "foldseek_query_coverage": coverage,
            }
        if foldseek:
            row.update(foldseek)
            alnlen = as_float(row.get("foldseek_top_alnlen"), 0.0)
            residues = as_float(row.get("structure_residue_count"), as_float(row.get("sequence_length_aa"), 0.0))
            if "foldseek_query_coverage" not in row or row.get("foldseek_query_coverage", "") == "":
                row["foldseek_query_coverage"] = alnlen / residues if residues else ""
        else:
            row.update({"foldseek_hit_count": 0, "foldseek_top_n": 0, "foldseek_high_conf_hit_count": 0})

        row["structure_evidence_status"] = structure_status(row)
        row["structure_claim_phrase"] = structure_claim_phrase(row["structure_evidence_status"])
        row["case_study_priority_score"] = case_priority_score(row)
        row["candidate_claim_language"] = (
            "prioritized computational hypothesis; evidence is post hoc and requires independent experimental validation"
        )
        row["guardrail_note"] = (
            "Genome context complements sequence and structure; do not claim confirmed discovery or genome-context superiority."
        )
        rows.append(row)
    add_recommendations(rows)
    return sorted(rows, key=lambda row: as_float(row.get("case_study_priority_score"), 0.0), reverse=True)


def write_guardrails(path: Path) -> None:
    rows = [
        {
            "item": "main_claim",
            "manuscript_safe_wording": (
                "Genome context complements sequence and structure by prioritizing and sometimes disambiguating "
                "candidate viral protein functions under leakage-aware OOD evaluation."
            ),
        },
        {
            "item": "candidate_language",
            "manuscript_safe_wording": "Use prioritized, supported, structure-consistent, ambiguous, or hypothesis.",
        },
        {
            "item": "forbidden_candidate_language",
            "manuscript_safe_wording": "Do not call candidates validated or confirmed without independent experimental evidence.",
        },
        {
            "item": "posthoc_evidence",
            "manuscript_safe_wording": (
                "Product descriptions, MMseqs2 hits, Foldseek hits, and module weak labels are post hoc triage evidence only."
            ),
        },
        {
            "item": "gate_wording",
            "manuscript_safe_wording": "Use validation-targeted gate unless true held-out FDR control is demonstrated.",
        },
    ]
    write_tsv(path, rows)


def write_readme(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Sequence-Structure-Genome-Context Validation Module",
        "",
        "This directory contains manuscript-ready post hoc evidence tables and figure source data.",
        "",
        "Safe framing: genome context complements sequence and structure by prioritizing and sometimes disambiguating candidate viral protein functions under leakage-aware OOD evaluation.",
        "",
        "Candidates are computational hypotheses. Foldseek, MMseqs2, product text, and module summaries are evidence for triage, not de novo model inputs.",
        "",
        "Key outputs:",
        "- `tables/validation_integrated_evidence.tsv`",
        "- `tables/figure6_candidate_case_rankings.tsv`",
        "- `tables/foldseek_structural_ambiguity_metrics.tsv`",
        "- `figures/figure6_sequence_structure_context_matrix.png`",
        "- `figures/figure6_sequence_structure_context_scatter.png`",
        "",
        "Run summary:",
        "```json",
        json.dumps(report, indent=2, ensure_ascii=False),
        "```",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def make_figures(rows: list[dict[str, Any]], fig_dir: Path, top_n: int) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc:
        raise SystemExit(f"matplotlib and numpy are required to render figures: {exc}") from exc

    fig_dir.mkdir(parents=True, exist_ok=True)
    candidates = [row for row in rows if row.get("target_type") == "high_context_candidate"]
    candidates = sorted(candidates, key=lambda row: as_float(row.get("case_study_priority_score"), 0.0), reverse=True)
    top = candidates[:top_n]
    if not top:
        return

    matrix = []
    ylabels = []
    for row in top:
        pident = as_float(row.get("nearest_homolog_identity"))
        ambiguity = as_float(row.get("foldseek_ambiguity_index"), 1.0)
        matrix.append(
            [
                clamp01(as_float(row.get("delta_p"), 0.0)),
                clamp01(as_float(row.get("p_context"), 0.0)),
                1.0 if math.isnan(pident) else clamp01(1.0 - pident / 100.0),
                clamp01(as_float(row.get("foldseek_top_qtmscore"), 0.0)),
                clamp01(1.0 - ambiguity),
                1.0 if row.get("module_cluster_id") else 0.0,
            ]
        )
        ylabels.append(f"{row.get('protein_accession')} | {row.get('predicted_label')}")
    columns = ["context gain", "context prob.", "low seq. identity", "top qTM", "structure clarity", "module"]
    fig, ax = plt.subplots(figsize=(8.8, max(4.2, 0.32 * len(top) + 1.8)))
    im = ax.imshow(np.array(matrix), aspect="auto", cmap="YlGnBu", vmin=0, vmax=1)
    ax.set_xticks(range(len(columns)), labels=columns, rotation=30, ha="right")
    ax.set_yticks(range(len(ylabels)), labels=ylabels, fontsize=7)
    ax.set_title("Figure 6 source: sequence-structure-context candidate matrix", loc="left", fontsize=10, fontweight="bold")
    for y, values in enumerate(matrix):
        for x, value in enumerate(values):
            ax.text(x, y, f"{value:.2f}", ha="center", va="center", fontsize=6, color="#1f2933")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    fig.savefig(fig_dir / "figure6_sequence_structure_context_matrix.png", dpi=240, bbox_inches="tight")
    fig.savefig(fig_dir / "figure6_sequence_structure_context_matrix.pdf", bbox_inches="tight")
    plt.close(fig)

    x = [0.0 if math.isnan(as_float(row.get("nearest_homolog_identity"))) else as_float(row.get("nearest_homolog_identity")) for row in candidates]
    y = [0.0 if math.isnan(as_float(row.get("foldseek_top_qtmscore"))) else as_float(row.get("foldseek_top_qtmscore")) for row in candidates]
    c = [clamp01(as_float(row.get("delta_p"), 0.0)) for row in candidates]
    sizes = [40 + 180 * clamp01(as_float(row.get("p_context"), 0.0)) for row in candidates]
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    scatter = ax.scatter(x, y, c=c, s=sizes, cmap="viridis", alpha=0.78, edgecolors="#222222", linewidths=0.4)
    ax.set_xlabel("MMseqs2 nearest-hit identity (%)")
    ax.set_ylabel("Foldseek top qTM score")
    ax.set_title("Candidate evidence landscape", loc="left", fontsize=10, fontweight="bold")
    ax.grid(True, color="#D8DEE9", linewidth=0.6, alpha=0.8)
    for row in [r for r in candidates if r.get("figure6_recommendation") == "candidate_case_study_panel"][:6]:
        px = 0.0 if math.isnan(as_float(row.get("nearest_homolog_identity"))) else as_float(row.get("nearest_homolog_identity"))
        py = 0.0 if math.isnan(as_float(row.get("foldseek_top_qtmscore"))) else as_float(row.get("foldseek_top_qtmscore"))
        ax.annotate(row.get("protein_accession", ""), (px, py), xytext=(4, 4), textcoords="offset points", fontsize=6)
    cb = fig.colorbar(scatter, ax=ax)
    cb.set_label("context gain")
    fig.tight_layout()
    fig.savefig(fig_dir / "figure6_sequence_structure_context_scatter.png", dpi=240, bbox_inches="tight")
    fig.savefig(fig_dir / "figure6_sequence_structure_context_scatter.pdf", bbox_inches="tight")
    plt.close(fig)

    top_bar = top[: min(top_n, 12)]
    fig, ax = plt.subplots(figsize=(7.4, max(3.6, 0.28 * len(top_bar) + 1.4)))
    labels = [row.get("protein_accession", "") for row in reversed(top_bar)]
    scores = [as_float(row.get("case_study_priority_score"), 0.0) for row in reversed(top_bar)]
    ax.barh(labels, scores, color="#4C78A8")
    ax.set_xlabel("case-study priority score")
    ax.set_title("Ranked candidate case studies", loc="left", fontsize=10, fontweight="bold")
    ax.grid(True, axis="x", color="#D8DEE9", linewidth=0.6, alpha=0.8)
    fig.tight_layout()
    fig.savefig(fig_dir / "figure6_case_study_priority.png", dpi=240, bbox_inches="tight")
    fig.savefig(fig_dir / "figure6_case_study_priority.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    root = repo_root()
    targets_path = resolve_path(args.targets, root)
    candidates_path = resolve_path(args.candidates, root)
    module_candidates_path = resolve_path(args.module_candidates, root)
    module_clusters_path = resolve_path(args.module_clusters, root)
    homology_path = resolve_path(args.homology_hits, root)
    structure_summary_path = resolve_path(args.structure_summary, root)
    foldseek_hits_path = resolve_path(args.foldseek_hits, root)
    pdb_dir = resolve_path(args.pdb_dir, root)
    output_dir = resolve_path(args.output_dir, root)
    assert output_dir is not None
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    targets = load_targets(targets_path, candidates_path)
    module_map = index_first(read_tsv(module_candidates_path, required=False, table_name="module candidates"), ("center_accession",))
    cluster_map = {
        row.get("module_cluster_id", ""): row
        for row in (normalize_cluster_row(raw) for raw in read_tsv(module_clusters_path, required=False, table_name="module clusters"))
        if row.get("module_cluster_id")
    }
    homology_map = load_homology(homology_path, args.homology_scheme, args.homology_subset) if homology_path else {}
    foldseek_map, foldseek_top_rows = summarize_foldseek(foldseek_hits_path, args.top_foldseek_hits)
    target_accessions = {row["protein_accession"] for row in targets}
    foldseek_top_rows = [row for row in foldseek_top_rows if row.get("protein_accession") in target_accessions]
    structure_summary = load_structure_summary(structure_summary_path)
    pdb_quality = load_pdb_quality(pdb_dir)

    integrated = build_integrated_rows(
        targets,
        module_map=module_map,
        cluster_map=cluster_map,
        homology_map=homology_map,
        foldseek_map=foldseek_map,
        structure_summary=structure_summary,
        pdb_quality=pdb_quality,
    )

    candidate_rankings = [row for row in integrated if row.get("target_type") == "high_context_candidate"]
    ambiguity_rows = [
        {
            key: row.get(key, "")
            for key in [
                "protein_accession",
                "target_type",
                "foldseek_hit_count",
                "foldseek_top_n",
                "foldseek_high_conf_hit_count",
                "foldseek_near_top_hit_count",
                "foldseek_taxname_diversity_top_n",
                "foldseek_top_taxname_fraction",
                "foldseek_ambiguity_index",
                "foldseek_top_target",
                "foldseek_top_taxname",
                "foldseek_top_prob",
                "foldseek_second_prob",
                "foldseek_prob_margin",
                "foldseek_top_qtmscore",
                "foldseek_second_qtmscore",
                "foldseek_qtmscore_margin",
                "foldseek_query_coverage",
                "structure_evidence_status",
                "structure_claim_phrase",
            ]
        }
        for row in integrated
    ]

    ranking_fields = [
        "figure6_recommendation",
        "case_study_priority_score",
        "protein_accession",
        "predicted_label",
        "p_protein_only",
        "p_context",
        "delta_p",
        "family",
        "host_group",
        "genome_id",
        "description",
        "hypothetical_or_uncharacterized",
        "exact_transfer_flag",
        "nearest_homolog_accession",
        "nearest_homolog_identity",
        "sequence_evidence_status",
        "mean_plddt",
        "foldseek_top_target",
        "foldseek_top_taxname",
        "foldseek_top_prob",
        "foldseek_top_qtmscore",
        "foldseek_query_coverage",
        "foldseek_ambiguity_index",
        "structure_evidence_status",
        "module_cluster_id",
        "module_cluster_size",
        "module_cluster_family_count",
        "module_neighborhood_signature",
        "module_weak_label_counts",
        "candidate_claim_language",
        "guardrail_note",
    ]

    write_tsv(tables_dir / "validation_integrated_evidence.tsv", integrated)
    write_tsv(tables_dir / "figure6_candidate_case_rankings.tsv", candidate_rankings, ranking_fields)
    write_tsv(tables_dir / "foldseek_structural_ambiguity_metrics.tsv", ambiguity_rows)
    write_tsv(tables_dir / "foldseek_top_hits_long.tsv", foldseek_top_rows, ["protein_accession", "foldseek_rank", *FOLDSEEK_FIELDS])
    write_tsv(tables_dir / "figure6_validation_matrix_source.tsv", candidate_rankings, ranking_fields)
    write_guardrails(tables_dir / "validation_guardrails.tsv")
    make_figures(integrated, figures_dir, args.figure_top_n)

    report = {
        "claim_frame": (
            "Genome context complements sequence and structure by prioritizing and sometimes disambiguating "
            "candidate viral protein functions under leakage-aware OOD evaluation."
        ),
        "target_count": len(integrated),
        "candidate_count": len(candidate_rankings),
        "control_count": sum(1 for row in integrated if row.get("target_type") == "matched_control"),
        "foldseek_queries_with_hits_total": len(foldseek_map),
        "targets_with_foldseek_hits": sum(
            1 for row in integrated if as_float(row.get("foldseek_hit_count"), 0.0) > 0
        ),
        "structure_summary_rows": len(structure_summary),
        "homology_rows_indexed": len(homology_map),
        "outputs": {
            "integrated_evidence": str(tables_dir / "validation_integrated_evidence.tsv"),
            "case_rankings": str(tables_dir / "figure6_candidate_case_rankings.tsv"),
            "ambiguity_metrics": str(tables_dir / "foldseek_structural_ambiguity_metrics.tsv"),
            "figures": str(figures_dir),
        },
    }
    (output_dir / "sequence_structure_context_validation_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_readme(output_dir / "README.md", report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
