from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STRUCTURAL_MEMBRANE_LABELS = {
    "capsid_head",
    "portal_terminase_packaging",
    "tail_assembly",
    "nucleocapsid",
    "envelope_glycoprotein",
    "membrane_matrix",
    "tail_fiber_receptor",
}


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
    parser = argparse.ArgumentParser(description="Join selective-prediction candidates to discovered local modules.")
    parser.add_argument("--module-candidates", required=True)
    parser.add_argument("--ranked-clusters", required=True)
    parser.add_argument("--candidate-prioritization", required=True)
    parser.add_argument("--output-dir", required=True)
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


def parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def parse_labels(value: str) -> list[str]:
    try:
        parsed = json.loads(value)
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    except Exception:
        return []
    return []


def main() -> int:
    args = parse_args()
    root = repo_root()
    module_candidates_path = resolve_path(root, args.module_candidates)
    ranked_clusters_path = resolve_path(root, args.ranked_clusters)
    candidate_path = resolve_path(root, args.candidate_prioritization)
    output_dir = resolve_path(root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    module_rows = read_tsv(module_candidates_path)
    ranked_rows = read_tsv(ranked_clusters_path)
    candidate_rows = read_tsv(candidate_path)
    candidate_by_accession = {str(row["protein_accession"]): row for row in candidate_rows}

    joined_rows: list[dict[str, Any]] = []
    cluster_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in module_rows:
        accession = str(row.get("center_accession", ""))
        candidate = candidate_by_accession.get(accession, {})
        predicted_labels = parse_labels(str(candidate.get("predicted_labels_at_precision_threshold", "[]")))
        structural_membrane_prediction = any(label in STRUCTURAL_MEMBRANE_LABELS for label in predicted_labels)
        joined = {
            **row,
            "passes_fdr_gate": candidate.get("passes_fdr_gate", ""),
            "top_label": candidate.get("top_label", ""),
            "top_probability_calibrated": candidate.get("top_probability_calibrated", ""),
            "predicted_labels_at_precision_threshold": candidate.get("predicted_labels_at_precision_threshold", "[]"),
            "structural_membrane_prediction": structural_membrane_prediction,
            "high_confidence_structural_membrane_candidate": parse_bool(candidate.get("passes_fdr_gate", "")) and structural_membrane_prediction,
        }
        joined_rows.append(joined)
        cluster_rows[str(row.get("cluster_id", ""))].append(joined)

    ranked_by_cluster = {str(row["cluster_id"]): row for row in ranked_rows}
    linked_cluster_rows: list[dict[str, Any]] = []
    for cluster_id, rows in cluster_rows.items():
        if cluster_id in {"", "-1"}:
            continue
        high_conf_rows = [row for row in rows if parse_bool(row["high_confidence_structural_membrane_candidate"])]
        gated_rows = [row for row in rows if parse_bool(row["passes_fdr_gate"])]
        top_labels = Counter(str(row.get("top_label", "")) for row in gated_rows if row.get("top_label"))
        ranked = ranked_by_cluster.get(cluster_id, {})
        linked_cluster_rows.append(
            {
                **ranked,
                "cluster_id": cluster_id,
                "module_count_joined": len(rows),
                "fdr_gated_module_count": len(gated_rows),
                "high_confidence_structural_membrane_count": len(high_conf_rows),
                "high_confidence_structural_membrane_fraction": 0.0 if not rows else len(high_conf_rows) / len(rows),
                "top_calibrated_label_among_gated": top_labels.most_common(1)[0][0] if top_labels else "",
                "top_calibrated_label_count": top_labels.most_common(1)[0][1] if top_labels else 0,
                "priority_score_with_uncertainty": float(ranked.get("priority_score", 0.0) or 0.0) + len(high_conf_rows),
            }
        )
    linked_cluster_rows.sort(
        key=lambda row: (
            float(row.get("priority_score_with_uncertainty", 0.0) or 0.0),
            float(row.get("high_confidence_structural_membrane_fraction", 0.0) or 0.0),
        ),
        reverse=True,
    )

    write_tsv(output_dir / "module_candidates_with_uncertainty.tsv", joined_rows)
    write_tsv(output_dir / "ranked_clusters_with_uncertainty.tsv", linked_cluster_rows)
    report = {
        "created_at": timestamp(),
        "module_candidates": str(module_candidates_path),
        "ranked_clusters": str(ranked_clusters_path),
        "candidate_prioritization": str(candidate_path),
        "module_count": len(joined_rows),
        "cluster_count": len(linked_cluster_rows),
        "high_confidence_structural_membrane_modules": sum(parse_bool(row["high_confidence_structural_membrane_candidate"]) for row in joined_rows),
    }
    (output_dir / "uncertainty_module_link_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
