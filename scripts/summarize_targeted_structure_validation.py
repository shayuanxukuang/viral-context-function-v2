from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

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
    parser = argparse.ArgumentParser(
        description="Summarize targeted ESMFold validation outputs by protein and module cluster."
    )
    parser.add_argument("--pdb-dir", required=True, help="Directory containing ESMFold PDB files.")
    parser.add_argument("--representatives", required=True, help="Representative metadata TSV.")
    parser.add_argument("--output-dir", required=True, help="Output directory for summary tables and casebook.")
    parser.add_argument(
        "--foldseek-hits",
        default=None,
        help="Optional Foldseek convertalis TSV to link later structure-search evidence.",
    )
    return parser.parse_args()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_ca_plddt(pdb_path: Path) -> tuple[int, list[float]]:
    residues: set[tuple[str, str]] = set()
    ca_plddt: list[float] = []
    with pdb_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.startswith("ATOM"):
                continue
            atom = line[12:16].strip()
            chain = line[21].strip()
            residue = line[22:26].strip()
            residues.add((chain, residue))
            if atom == "CA":
                ca_plddt.append(float(line[60:66]))
    return len(residues), ca_plddt


def float_or_none(value: Any) -> float | None:
    try:
        if value == "" or value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def confidence_tier(max_mean_plddt: float, cluster_mean_plddt: float, frac_high_reps: float) -> str:
    if cluster_mean_plddt >= 70.0 and frac_high_reps >= 0.5:
        return "high_cluster_confidence"
    if max_mean_plddt >= 70.0:
        return "one_high_confidence_representative"
    if cluster_mean_plddt >= 55.0:
        return "moderate_cluster_confidence"
    return "low_esmfold_confidence"


def evidence_role(hypothetical_ratio: float, structural_vote: float, families: int) -> str:
    if hypothetical_ratio >= 0.5 and families >= 2 and structural_vote > 0:
        return "hypothetical_cross_family_structural_candidate"
    if hypothetical_ratio >= 0.8 and families >= 2:
        return "hypothetical_cross_family_context_candidate"
    if structural_vote >= 0.75:
        return "structural_membrane_positive_control"
    return "exploratory_context_candidate"


def structure_evidence_tier(mean_plddt: float, prob: float, qtmscore: float, query_coverage: float) -> str:
    if mean_plddt >= 70.0 and prob >= 0.5 and qtmscore >= 0.4 and query_coverage >= 0.35:
        return "strong_targeted_support"
    if mean_plddt >= 55.0 and prob >= 0.9 and qtmscore >= 0.3 and query_coverage >= 0.35:
        return "local_structural_support"
    if prob >= 0.9 and query_coverage >= 0.25:
        return "foldseek_only_low_model_confidence"
    if mean_plddt >= 70.0:
        return "model_confident_no_clear_pdb_analog"
    return "weak_or_inconclusive"


def summarize_foldseek(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        first_line = handle.readline().rstrip("\n")
    if not first_line:
        return {}
    first_parts = first_line.split("\t")
    if "query" in first_parts and "target" in first_parts:
        rows = read_tsv(path)
    else:
        rows = []
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle, delimiter="\t")
            for parts in reader:
                if not parts:
                    continue
                row = {field: parts[index] if index < len(parts) else "" for index, field in enumerate(FOLDSEEK_FIELDS)}
                rows.append(row)

    by_query_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        query = str(row.get("query", ""))
        if not query:
            continue
        by_query_rows[query].append(row)

    def hit_sort_key(row: dict[str, str]) -> tuple[float, float, float]:
        prob = float_or_none(row.get("prob")) or 0.0
        qtmscore = float_or_none(row.get("qtmscore")) or 0.0
        evalue = float_or_none(row.get("evalue"))
        negative_log_evalue = -1e9 if evalue is None or evalue <= 0 else -evalue
        return (prob, qtmscore, negative_log_evalue)

    return {query: max(query_rows, key=hit_sort_key) for query, query_rows in by_query_rows.items()}


def main() -> int:
    args = parse_args()
    pdb_dir = Path(args.pdb_dir).resolve()
    representatives_path = Path(args.representatives).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    representatives = read_tsv(representatives_path)
    metadata = {row["protein_accession"]: row for row in representatives}
    foldseek_by_query = summarize_foldseek(Path(args.foldseek_hits).resolve() if args.foldseek_hits else None)

    protein_rows: list[dict[str, Any]] = []
    for pdb_path in sorted(pdb_dir.glob("*.pdb")):
        accession = pdb_path.name.removesuffix(".pdb")
        residue_count, ca_plddt = parse_ca_plddt(pdb_path)
        row: dict[str, Any] = dict(metadata.get(accession, {}))
        row.update(
            {
                "protein_accession": accession,
                "pdb_path": str(pdb_path),
                "residue_count": residue_count,
                "ca_count": len(ca_plddt),
                "mean_plddt": round(mean(ca_plddt), 3) if ca_plddt else "",
                "median_plddt": round(median(ca_plddt), 3) if ca_plddt else "",
                "min_plddt": round(min(ca_plddt), 3) if ca_plddt else "",
                "max_plddt": round(max(ca_plddt), 3) if ca_plddt else "",
                "frac_plddt_ge_70": round(sum(v >= 70.0 for v in ca_plddt) / len(ca_plddt), 3) if ca_plddt else "",
                "frac_plddt_ge_90": round(sum(v >= 90.0 for v in ca_plddt) / len(ca_plddt), 3) if ca_plddt else "",
            }
        )
        foldseek_hit = foldseek_by_query.get(accession) or foldseek_by_query.get(pdb_path.stem)
        if foldseek_hit:
            alnlen = float_or_none(foldseek_hit.get("alnlen")) or 0.0
            query_coverage = alnlen / residue_count if residue_count else 0.0
            mean_plddt = mean(ca_plddt) if ca_plddt else 0.0
            prob = float_or_none(foldseek_hit.get("prob")) or 0.0
            qtmscore = float_or_none(foldseek_hit.get("qtmscore")) or 0.0
            row.update(
                {
                    "foldseek_target": foldseek_hit.get("target", ""),
                    "foldseek_evalue": foldseek_hit.get("evalue", ""),
                    "foldseek_prob": foldseek_hit.get("prob", ""),
                    "foldseek_alnlen": foldseek_hit.get("alnlen", ""),
                    "foldseek_pident": foldseek_hit.get("pident", ""),
                    "foldseek_lddt": foldseek_hit.get("lddt", ""),
                    "foldseek_alntmscore": foldseek_hit.get("alntmscore", ""),
                    "foldseek_qtmscore": foldseek_hit.get("qtmscore", ""),
                    "foldseek_ttmscore": foldseek_hit.get("ttmscore", ""),
                    "foldseek_taxname": foldseek_hit.get("taxname", ""),
                    "foldseek_query_coverage": round(query_coverage, 3),
                    "structure_evidence_tier": structure_evidence_tier(mean_plddt, prob, qtmscore, query_coverage),
                }
            )
        protein_rows.append(row)

    by_cluster: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in protein_rows:
        by_cluster[str(row.get("cluster_id", ""))].append(row)

    cluster_rows: list[dict[str, Any]] = []
    for cluster_id, rows in sorted(by_cluster.items(), key=lambda item: int(item[0]) if item[0].isdigit() else item[0]):
        mean_plddts = [v for v in (float_or_none(row.get("mean_plddt")) for row in rows) if v is not None]
        high_rep_count = sum(v >= 70.0 for v in mean_plddts)
        families = sorted({str(row.get("virus_family", "")) for row in rows if row.get("virus_family", "")})
        hypothetical = [v for v in (float_or_none(row.get("hypothetical_ratio")) for row in rows) if v is not None]
        structural = [
            v for v in (float_or_none(row.get("structural_membrane_vote_fraction")) for row in rows) if v is not None
        ]
        cluster_mean = mean(mean_plddts)
        max_mean = max(mean_plddts) if mean_plddts else 0.0
        frac_high_reps = high_rep_count / len(mean_plddts) if mean_plddts else 0.0
        hypothetical_ratio = mean(hypothetical)
        structural_vote = mean(structural)
        foldseek_supported = [
            row
            for row in rows
            if str(row.get("structure_evidence_tier", ""))
            in {"strong_targeted_support", "local_structural_support", "foldseek_only_low_model_confidence"}
        ]
        best_foldseek = max(
            rows,
            key=lambda row: (
                float_or_none(row.get("foldseek_prob")) or 0.0,
                float_or_none(row.get("foldseek_qtmscore")) or 0.0,
                float_or_none(row.get("foldseek_query_coverage")) or 0.0,
            ),
        )
        cluster_rows.append(
            {
                "cluster_id": cluster_id,
                "representative_count": len(rows),
                "family_count": len(families),
                "families": ",".join(families),
                "mean_representative_plddt": round(cluster_mean, 3),
                "max_representative_plddt": round(max_mean, 3),
                "high_confidence_representatives": high_rep_count,
                "frac_high_confidence_representatives": round(frac_high_reps, 3),
                "mean_hypothetical_ratio": round(hypothetical_ratio, 3),
                "mean_structural_membrane_vote_fraction": round(structural_vote, 3),
                "confidence_tier": confidence_tier(max_mean, cluster_mean, frac_high_reps),
                "evidence_role": evidence_role(hypothetical_ratio, structural_vote, len(families)),
                "top_accession": max(rows, key=lambda row: float_or_none(row.get("mean_plddt")) or -1).get(
                    "protein_accession", ""
                ),
                "foldseek_supported_representatives": len(foldseek_supported),
                "best_foldseek_accession": best_foldseek.get("protein_accession", ""),
                "best_foldseek_target": best_foldseek.get("foldseek_target", ""),
                "best_foldseek_prob": best_foldseek.get("foldseek_prob", ""),
                "best_foldseek_qtmscore": best_foldseek.get("foldseek_qtmscore", ""),
                "best_foldseek_query_coverage": best_foldseek.get("foldseek_query_coverage", ""),
                "best_structure_evidence_tier": best_foldseek.get("structure_evidence_tier", ""),
            }
        )

    protein_fields = [
        "cluster_id",
        "protein_accession",
        "virus_family",
        "description",
        "residue_count",
        "ca_count",
        "mean_plddt",
        "median_plddt",
        "min_plddt",
        "max_plddt",
        "frac_plddt_ge_70",
        "frac_plddt_ge_90",
        "hypothetical_ratio",
        "neighborhood_signature",
        "structural_membrane_vote_fraction",
        "foldseek_target",
        "foldseek_evalue",
        "foldseek_prob",
        "foldseek_alnlen",
        "foldseek_pident",
        "foldseek_lddt",
        "foldseek_alntmscore",
        "foldseek_qtmscore",
        "foldseek_ttmscore",
        "foldseek_query_coverage",
        "foldseek_taxname",
        "structure_evidence_tier",
        "pdb_path",
    ]
    cluster_fields = [
        "cluster_id",
        "representative_count",
        "family_count",
        "families",
        "mean_representative_plddt",
        "max_representative_plddt",
        "high_confidence_representatives",
        "frac_high_confidence_representatives",
        "mean_hypothetical_ratio",
        "mean_structural_membrane_vote_fraction",
        "confidence_tier",
        "evidence_role",
        "top_accession",
        "foldseek_supported_representatives",
        "best_foldseek_accession",
        "best_foldseek_target",
        "best_foldseek_prob",
        "best_foldseek_qtmscore",
        "best_foldseek_query_coverage",
        "best_structure_evidence_tier",
    ]

    protein_rows = sorted(
        protein_rows,
        key=lambda row: (str(row.get("cluster_id", "")), -(float_or_none(row.get("mean_plddt")) or 0.0)),
    )
    cluster_rows = sorted(
        cluster_rows,
        key=lambda row: (
            row["confidence_tier"] != "high_cluster_confidence",
            row["confidence_tier"] != "one_high_confidence_representative",
            -float(row["max_representative_plddt"]),
        ),
    )

    write_tsv(output_dir / "esmfold_quality.tsv", protein_rows, protein_fields)
    write_tsv(output_dir / "cluster_esmfold_summary.tsv", cluster_rows, cluster_fields)

    lines = [
        "# Targeted Structure Validation Casebook",
        "",
        f"- PDB directory: `{pdb_dir}`",
        f"- Representative metadata: `{representatives_path}`",
        f"- Proteins summarized: {len(protein_rows)}",
        f"- Clusters summarized: {len(cluster_rows)}",
        "",
        "## Cluster Summary",
        "",
    ]
    for cluster in cluster_rows:
        lines.extend(
            [
                f"### Cluster {cluster['cluster_id']}",
                "",
                f"- Confidence tier: {cluster['confidence_tier']}",
                f"- Evidence role: {cluster['evidence_role']}",
                f"- Families: {cluster['families']}",
                f"- Mean representative pLDDT: {cluster['mean_representative_plddt']}",
                f"- Max representative pLDDT: {cluster['max_representative_plddt']} ({cluster['top_accession']})",
                f"- Hypothetical ratio: {cluster['mean_hypothetical_ratio']}",
                f"- Structural/membrane vote fraction: {cluster['mean_structural_membrane_vote_fraction']}",
                f"- Best Foldseek hit: {cluster.get('best_foldseek_accession', '')} -> {cluster.get('best_foldseek_target', '')}; prob {cluster.get('best_foldseek_prob', '')}; qTM {cluster.get('best_foldseek_qtmscore', '')}; query coverage {cluster.get('best_foldseek_query_coverage', '')}",
                f"- Best structure evidence tier: {cluster.get('best_structure_evidence_tier', '')}",
                "",
                "| accession | family | length | mean pLDDT | Foldseek target | prob | qTM | query cov | evidence tier | description |",
                "|---|---|---:|---:|---|---:|---:|---:|---|---|",
            ]
        )
        for row in [r for r in protein_rows if str(r.get("cluster_id", "")) == str(cluster["cluster_id"])]:
            lines.append(
                "| {protein_accession} | {virus_family} | {residue_count} | {mean_plddt} | {foldseek_target} | {foldseek_prob} | {foldseek_qtmscore} | {foldseek_query_coverage} | {structure_evidence_tier} | {description} |".format(
                    **{key: str(row.get(key, "")).replace("|", "/") for key in protein_fields}
                )
            )
        lines.append("")

    (output_dir / "structure_validation_casebook.md").write_text("\n".join(lines), encoding="utf-8")
    report = {
        "pdb_dir": str(pdb_dir),
        "representatives": str(representatives_path),
        "output_dir": str(output_dir),
        "protein_count": len(protein_rows),
        "cluster_count": len(cluster_rows),
        "clusters_with_high_representative": sum(
            row["confidence_tier"] == "one_high_confidence_representative" for row in cluster_rows
        ),
        "clusters_with_low_confidence": sum(row["confidence_tier"] == "low_esmfold_confidence" for row in cluster_rows),
    }
    (output_dir / "structure_validation_summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
