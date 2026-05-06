from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    return (root / path).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Shortlist discovery clusters for paper-grade module case studies.")
    parser.add_argument("--linked-clusters", required=True)
    parser.add_argument("--module-candidates", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-background-cluster-size", type=int, default=500)
    parser.add_argument("--min-family-count", type=int, default=2)
    parser.add_argument("--top-clusters", type=int, default=20)
    parser.add_argument("--representatives-per-cluster", type=int, default=8)
    return parser.parse_args()


def read_tsv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def as_float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, "")
        if value in {"", None}:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(row: dict[str, Any], key: str, default: int = 0) -> int:
    try:
        value = row.get(key, "")
        if value in {"", None}:
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def parse_json_dict(value: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def cluster_theme(row: dict[str, Any]) -> str:
    hypothetical = as_float(row, "hypothetical_ratio_mean")
    structural_vote = as_float(row, "structural_membrane_vote_fraction_mean")
    high_conf = as_int(row, "high_confidence_structural_membrane_count")
    family_count = as_int(row, "family_count")
    if hypothetical >= 0.8 and family_count >= 2 and (structural_vote >= 0.1 or high_conf > 0):
        return "hypothetical_structural_membrane"
    if hypothetical >= 0.8 and family_count >= 2:
        return "hypothetical_cross_family"
    if structural_vote >= 0.2 or high_conf > 0:
        return "structural_membrane_control"
    if hypothetical >= 0.5 and family_count >= 2:
        return "hypothetical_context_module"
    return "lower_priority"


def discovery_score(row: dict[str, Any]) -> float:
    module_count = max(1, as_int(row, "module_count_joined", as_int(row, "module_count", 1)))
    family_count = as_int(row, "family_count")
    hypothetical = as_float(row, "hypothetical_ratio_mean")
    structural_vote = as_float(row, "structural_membrane_vote_fraction_mean")
    high_conf = as_int(row, "high_confidence_structural_membrane_count")
    consistency = as_float(row, "neighborhood_consistency")
    size_bonus = min(2.0, math.log10(module_count + 1.0))
    high_conf_bonus = min(3.0, math.log10(high_conf + 1.0)) if high_conf > 0 else 0.0
    return (
        (1.2 * min(family_count, 10))
        + (3.0 * hypothetical)
        + (3.0 * structural_vote)
        + (1.5 * consistency)
        + size_bonus
        + high_conf_bonus
    )


def representative_summary(rows: list[dict[str, Any]], limit: int) -> tuple[str, str, str, str]:
    sorted_rows = sorted(
        rows,
        key=lambda row: (
            parse_bool(row.get("high_confidence_structural_membrane_candidate", "")),
            as_float(row, "top_probability_calibrated"),
            as_float(row, "hypothetical_ratio"),
        ),
        reverse=True,
    )
    representatives = []
    families = []
    labels = []
    weak_counter: Counter[str] = Counter()
    seen_accessions: set[str] = set()
    for row in sorted_rows[:limit]:
        accession = str(row.get("center_accession", ""))
        if accession in seen_accessions:
            continue
        seen_accessions.add(accession)
        representatives.append(
            f"{accession}|{row.get('virus_family', '')}|{row.get('description', '')}"
        )
        families.append(str(row.get("virus_family", "")))
        if row.get("top_label"):
            labels.append(str(row.get("top_label", "")))
        weak_counter.update(parse_json_dict(str(row.get("weak_label_counts_json", "{}"))))
        if len(representatives) >= limit:
            break
    return (
        json.dumps(representatives, ensure_ascii=False),
        json.dumps(sorted(set(families)), ensure_ascii=False),
        Counter(labels).most_common(1)[0][0] if labels else "",
        json.dumps(dict(weak_counter.most_common(8)), ensure_ascii=False),
    )


def main() -> int:
    args = parse_args()
    root = repo_root()
    linked_clusters_path = resolve_path(root, args.linked_clusters)
    module_candidates_path = resolve_path(root, args.module_candidates)
    output_dir = resolve_path(root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    linked_rows = read_tsv(linked_clusters_path)
    module_rows = read_tsv(module_candidates_path)
    modules_by_cluster: dict[str, list[dict[str, Any]]] = defaultdict(list)
    high_confidence_hypothetical_rows: list[dict[str, Any]] = []
    for row in module_rows:
        modules_by_cluster[str(row.get("cluster_id", ""))].append(row)
        if (
            parse_bool(row.get("high_confidence_structural_membrane_candidate", ""))
            and as_float(row, "hypothetical_ratio") >= 0.5
            and str(row.get("cluster_id", "")) != "0"
        ):
            high_confidence_hypothetical_rows.append(
                {
                    "cluster_id": row.get("cluster_id", ""),
                    "center_accession": row.get("center_accession", ""),
                    "virus_family": row.get("virus_family", ""),
                    "description": row.get("description", ""),
                    "hypothetical_ratio": row.get("hypothetical_ratio", ""),
                    "structural_membrane_vote_fraction": row.get("structural_membrane_vote_fraction", ""),
                    "top_label": row.get("top_label", ""),
                    "top_probability_calibrated": row.get("top_probability_calibrated", ""),
                    "predicted_labels_at_precision_threshold": row.get("predicted_labels_at_precision_threshold", ""),
                    "neighborhood_signature": row.get("neighborhood_signature", ""),
                    "cluster_status": "noise_or_unclustered" if str(row.get("cluster_id", "")) == "-1" else "clustered",
                }
            )
    high_confidence_hypothetical_rows.sort(
        key=lambda row: (
            row["cluster_status"] == "clustered",
            as_float(row, "top_probability_calibrated"),
            as_float(row, "hypothetical_ratio"),
        ),
        reverse=True,
    )

    shortlisted: list[dict[str, Any]] = []
    excluded_background = 0
    for row in linked_rows:
        cluster_id = str(row.get("cluster_id", ""))
        module_count = as_int(row, "module_count_joined", as_int(row, "module_count"))
        family_count = as_int(row, "family_count")
        if cluster_id in {"", "-1"}:
            continue
        if module_count > args.max_background_cluster_size:
            excluded_background += 1
            continue
        if family_count < args.min_family_count:
            continue
        theme = cluster_theme(row)
        if theme == "lower_priority":
            continue
        reps_json, families_json, top_label, weak_counts = representative_summary(
            modules_by_cluster.get(cluster_id, []), args.representatives_per_cluster
        )
        score = discovery_score(row)
        shortlisted.append(
            {
                "cluster_id": cluster_id,
                "module_count": module_count,
                "family_count": family_count,
                "theme": theme,
                "paper_case_priority_score": score,
                "hypothetical_ratio_mean": row.get("hypothetical_ratio_mean", ""),
                "structural_membrane_vote_fraction_mean": row.get("structural_membrane_vote_fraction_mean", ""),
                "neighborhood_consistency": row.get("neighborhood_consistency", ""),
                "top_neighborhood_signature": row.get("top_neighborhood_signature", ""),
                "high_confidence_structural_membrane_count": row.get("high_confidence_structural_membrane_count", ""),
                "high_confidence_structural_membrane_fraction": row.get("high_confidence_structural_membrane_fraction", ""),
                "top_calibrated_label_among_gated": row.get("top_calibrated_label_among_gated", top_label),
                "representative_families_json": families_json,
                "representatives_json": reps_json,
                "weak_label_counts_top_json": weak_counts,
            }
        )
    shortlisted.sort(
        key=lambda row: (
            str(row["theme"]) == "hypothetical_structural_membrane",
            str(row["theme"]) == "hypothetical_cross_family",
            float(row["paper_case_priority_score"]),
        ),
        reverse=True,
    )
    top_rows = shortlisted[: args.top_clusters]
    write_tsv(output_dir / "paper_module_shortlist.tsv", top_rows)
    write_tsv(output_dir / "high_confidence_hypothetical_structural_candidates.tsv", high_confidence_hypothetical_rows)
    structure_rows = sorted(
        shortlisted,
        key=lambda row: (
            as_float(row, "hypothetical_ratio_mean") >= 0.5 and as_float(row, "structural_membrane_vote_fraction_mean") >= 0.2,
            as_int(row, "high_confidence_structural_membrane_count") > 0,
            as_float(row, "structural_membrane_vote_fraction_mean"),
            as_float(row, "hypothetical_ratio_mean"),
            as_int(row, "family_count"),
            float(row["paper_case_priority_score"]),
        ),
        reverse=True,
    )[: min(args.top_clusters, 10)]
    write_tsv(output_dir / "structure_case_study_clusters.tsv", structure_rows)

    lines = [
        "# Paper module shortlist",
        "",
        f"- created_at: `{timestamp()}`",
        f"- linked_clusters: `{linked_clusters_path}`",
        f"- module_candidates: `{module_candidates_path}`",
        f"- retained_clusters: `{len(shortlisted)}`",
        f"- excluded_background_clusters: `{excluded_background}`",
        "",
    ]
    for row in top_rows:
        lines.extend(
            [
                f"## cluster_{row['cluster_id']}",
                "",
                f"- theme: `{row['theme']}`",
                f"- module_count: `{row['module_count']}`",
                f"- family_count: `{row['family_count']}`",
                f"- hypothetical_ratio_mean: `{row['hypothetical_ratio_mean']}`",
                f"- structural_membrane_vote_fraction_mean: `{row['structural_membrane_vote_fraction_mean']}`",
                f"- high_confidence_structural_membrane_count: `{row['high_confidence_structural_membrane_count']}`",
                f"- neighborhood_consistency: `{row['neighborhood_consistency']}`",
                f"- top_neighborhood_signature: `{row['top_neighborhood_signature']}`",
                f"- representative_families_json: `{row['representative_families_json']}`",
                "",
                "Representative proteins:",
                "",
            ]
        )
        for item in json.loads(str(row["representatives_json"])):
            lines.append(f"- `{item}`")
        lines.append("")
    (output_dir / "paper_module_casebook.md").write_text("\n".join(lines), encoding="utf-8")

    report = {
        "created_at": timestamp(),
        "linked_clusters": str(linked_clusters_path),
        "module_candidates": str(module_candidates_path),
        "retained_cluster_count": len(shortlisted),
        "top_cluster_count": len(top_rows),
        "structure_case_study_cluster_count": len(structure_rows),
        "high_confidence_hypothetical_structural_candidate_count": len(high_confidence_hypothetical_rows),
        "excluded_background_clusters": excluded_background,
        "theme_counts": Counter(str(row["theme"]) for row in shortlisted),
        "output_dir": str(output_dir),
    }
    (output_dir / "paper_module_shortlist_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
