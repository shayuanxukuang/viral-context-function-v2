from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import normalize

from biophysics_features import compute_biophysics
from context_features import derive_virus_family
from label_rules import LABEL_RULES, label_hits, normalize_text


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


def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cluster pooled local modules from exported task-mode embeddings.")
    parser.add_argument("--embedding-file", required=True)
    parser.add_argument("--input", default="data/processed/training/viral_protein_training_index.tsv.gz")
    parser.add_argument("--output-dir", default="runs/module_discovery")
    parser.add_argument("--window-radius", type=int, default=1)
    parser.add_argument("--min-cluster-size", type=int, default=5)
    parser.add_argument("--dbscan-eps", type=float, default=0.15)
    parser.add_argument("--knn-k", type=int, default=15)
    parser.add_argument("--cluster-method", choices=("auto", "hdbscan", "leiden", "dbscan"), default="auto")
    parser.add_argument("--top-casebooks", type=int, default=10)
    return parser.parse_args()


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def load_embedding_payload(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "accessions" not in payload or "embeddings" not in payload:
        raise ValueError(f"Unsupported embedding payload: {path}")
    return payload


def choose_clusterer(method: str) -> str:
    if method != "auto":
        return method
    try:
        import hdbscan  # noqa: F401

        return "hdbscan"
    except Exception:
        pass
    try:
        import igraph  # noqa: F401
        import leidenalg  # noqa: F401

        return "leiden"
    except Exception:
        pass
    return "dbscan"


def cluster_embeddings(embeddings: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, str]:
    method = choose_clusterer(args.cluster_method)
    if method == "hdbscan":
        import hdbscan

        clusterer = hdbscan.HDBSCAN(min_cluster_size=args.min_cluster_size, metric="euclidean")
        return clusterer.fit_predict(embeddings), "hdbscan"

    if method == "leiden":
        import igraph as ig
        import leidenalg

        knn = NearestNeighbors(n_neighbors=min(args.knn_k, max(2, embeddings.shape[0] - 1)), metric="cosine")
        knn.fit(embeddings)
        distances, indices = knn.kneighbors(embeddings)
        edges: list[tuple[int, int]] = []
        weights: list[float] = []
        for src_idx in range(embeddings.shape[0]):
            for dst_idx, distance in zip(indices[src_idx][1:], distances[src_idx][1:]):
                edges.append((int(src_idx), int(dst_idx)))
                weights.append(float(max(0.0, 1.0 - distance)))
        graph = ig.Graph(n=embeddings.shape[0], edges=edges, directed=False)
        graph.es["weight"] = weights
        partition = leidenalg.find_partition(
            graph,
            leidenalg.RBConfigurationVertexPartition,
            weights=graph.es["weight"],
            resolution_parameter=0.8,
        )
        labels = np.asarray(partition.membership, dtype=np.int32)
        counts = Counter(labels.tolist())
        labels = np.asarray([label if counts[int(label)] >= args.min_cluster_size else -1 for label in labels], dtype=np.int32)
        return labels, "leiden"

    clusterer = DBSCAN(eps=args.dbscan_eps, min_samples=args.min_cluster_size, metric="cosine")
    return clusterer.fit_predict(embeddings), "dbscan"


def sort_key(row: dict[str, Any]) -> tuple[int, int, int]:
    start = int(row["cds_start"])
    end = int(row["cds_end"])
    return (0 if start > 0 else 1, start if start > 0 else 10**12, end if end > 0 else 10**12)


def load_metadata(path: Path, accessions: set[str]) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows_by_genome: dict[str, list[dict[str, Any]]] = defaultdict(list)
    accession_rows: dict[str, dict[str, Any]] = {}
    with open_text(path) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            accession = str(row.get("protein_accession", "") or "").strip()
            if accession not in accessions:
                continue
            genome_version = (
                str(row.get("genome_version", "") or "").strip()
                or str(row.get("genome_accession", "") or "").strip()
                or accession
            )
            text = normalize_text(row)
            label_ids = set(label_hits(text))
            label_names = [LABEL_RULES[idx].name for idx in sorted(label_ids)]
            record = {
                "protein_accession": accession,
                "genome_version": genome_version,
                "virus_family": derive_virus_family(str(row.get("virus_lineage", "") or "")),
                "description": str(row.get("protein_description", "") or row.get("cds_product", "") or "").strip(),
                "is_hypothetical": int("hypothetical protein" in text or "uncharacterized" in text or "unknown protein" in text),
                "cds_start": int(row.get("cds_start", "0") or 0),
                "cds_end": int(row.get("cds_end", "0") or 0),
                "protein_feature_type": str(row.get("protein_feature_type", "") or "").strip(),
                "sequence": str(row.get("protein_sequence", "") or "").strip(),
                "weak_labels_json": json.dumps(label_names, ensure_ascii=False),
            }
            rows_by_genome[genome_version].append(record)
            accession_rows[accession] = record

    ordered_modules: list[dict[str, Any]] = []
    for genome_version, rows in rows_by_genome.items():
        ordered = sorted(rows, key=sort_key)
        for idx, row in enumerate(ordered):
            ordered_modules.append(
                {
                    "center_accession": row["protein_accession"],
                    "genome_version": genome_version,
                    "members": ordered,
                    "center_index": idx,
                }
            )
    return ordered_modules, accession_rows


def main() -> int:
    args = parse_args()
    root = repo_root()
    embedding_path = (root / args.embedding_file).resolve() if not Path(args.embedding_file).is_absolute() else Path(args.embedding_file).resolve()
    input_path = (root / args.input).resolve()
    output_dir = (root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = load_embedding_payload(embedding_path)
    accessions = [str(item) for item in payload["accessions"]]
    embeddings = np.asarray(payload["embeddings"], dtype=np.float32)
    accession_to_embedding = {accession: embeddings[idx] for idx, accession in enumerate(accessions)}

    module_windows, accession_rows = load_metadata(input_path, set(accessions))
    module_rows: list[dict[str, Any]] = []
    pooled_embeddings: list[np.ndarray] = []
    for module in module_windows:
        center_idx = int(module["center_index"])
        members = module["members"]
        left = max(0, center_idx - args.window_radius)
        right = min(len(members), center_idx + args.window_radius + 1)
        selected = members[left:right]
        vectors = [accession_to_embedding[row["protein_accession"]] for row in selected if row["protein_accession"] in accession_to_embedding]
        if not vectors:
            continue
        pooled = np.mean(np.stack(vectors), axis=0)
        pooled_embeddings.append(pooled)
        center = selected[min(args.window_radius, len(selected) - 1)]
        hypothetical_ratio = sum(int(row["is_hypothetical"]) for row in selected) / len(selected)
        neighborhood_signature = "|".join(row["protein_feature_type"] or "__missing__" for row in selected)
        weak_label_counter = Counter()
        structural_votes = 0
        for row in selected:
            weak_ids = json.loads(row["weak_labels_json"])
            weak_label_counter.update(weak_ids)
            if any(label in STRUCTURAL_MEMBRANE_LABELS for label in weak_ids):
                structural_votes += 1
        bio = compute_biophysics(center["sequence"])
        module_rows.append(
            {
                "center_accession": center["protein_accession"],
                "genome_version": center["genome_version"],
                "virus_family": center["virus_family"],
                "member_accessions_json": json.dumps([row["protein_accession"] for row in selected], ensure_ascii=False),
                "member_count": len(selected),
                "hypothetical_ratio": hypothetical_ratio,
                "neighborhood_signature": neighborhood_signature,
                "weak_label_counts_json": json.dumps(dict(weak_label_counter), ensure_ascii=False, sort_keys=True),
                "structural_membrane_vote_fraction": structural_votes / len(selected),
                "bio_signal_peptide_score": bio["bio_signal_peptide_score"],
                "bio_tm_helix_count": bio["bio_tm_helix_count"],
                "bio_disorder_score": bio["bio_disorder_score"],
            }
        )

    normalized_embeddings = normalize(np.asarray(pooled_embeddings, dtype=np.float32))
    cluster_ids, cluster_method = cluster_embeddings(normalized_embeddings, args)
    for row, cluster_id in zip(module_rows, cluster_ids.tolist()):
        row["cluster_id"] = int(cluster_id)

    write_tsv(output_dir / "module_candidates.tsv", module_rows)

    cluster_buckets: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in module_rows:
        cluster_buckets[int(row["cluster_id"])].append(row)

    ranked_rows: list[dict[str, Any]] = []
    for cluster_id, rows in cluster_buckets.items():
        if cluster_id < 0:
            continue
        family_count = len({str(row["virus_family"]) for row in rows})
        signature_counter = Counter(str(row["neighborhood_signature"]) for row in rows)
        top_signature, top_signature_count = signature_counter.most_common(1)[0]
        ranked_rows.append(
            {
                "cluster_id": cluster_id,
                "module_count": len(rows),
                "family_count": family_count,
                "hypothetical_ratio_mean": float(np.mean([float(row["hypothetical_ratio"]) for row in rows])),
                "structural_membrane_vote_fraction_mean": float(np.mean([float(row["structural_membrane_vote_fraction"]) for row in rows])),
                "neighborhood_consistency": top_signature_count / len(rows),
                "top_neighborhood_signature": top_signature,
                "priority_score": (
                    family_count
                    + (2.0 * float(np.mean([float(row["hypothetical_ratio"]) for row in rows])))
                    + float(np.mean([float(row["structural_membrane_vote_fraction"]) for row in rows]))
                    + (top_signature_count / len(rows))
                ),
            }
        )
    ranked_rows.sort(key=lambda row: (float(row["priority_score"]), float(row["hypothetical_ratio_mean"])), reverse=True)
    write_tsv(output_dir / "ranked_hypothetical_clusters.tsv", ranked_rows)

    casebook_dir = output_dir / "casebooks"
    casebook_dir.mkdir(parents=True, exist_ok=True)
    for row in ranked_rows[: args.top_casebooks]:
        cluster_id = int(row["cluster_id"])
        members = cluster_buckets[cluster_id]
        lines = [
            f"# cluster_{cluster_id}",
            "",
            f"- module_count: `{row['module_count']}`",
            f"- family_count: `{row['family_count']}`",
            f"- hypothetical_ratio_mean: `{row['hypothetical_ratio_mean']}`",
            f"- structural_membrane_vote_fraction_mean: `{row['structural_membrane_vote_fraction_mean']}`",
            f"- neighborhood_consistency: `{row['neighborhood_consistency']}`",
            f"- top_neighborhood_signature: `{row['top_neighborhood_signature']}`",
            "",
            "## Representative Members",
            "",
        ]
        for member in members[:20]:
            lines.extend(
                [
                    f"- `{member['center_accession']}` | family=`{member['virus_family']}` | signature=`{member['neighborhood_signature']}` | hypothetical_ratio=`{member['hypothetical_ratio']}`",
                ]
            )
        (casebook_dir / f"cluster_{cluster_id}.casebook.md").write_text("\n".join(lines), encoding="utf-8")

    report = {
        "created_at": timestamp(),
        "embedding_file": str(embedding_path),
        "cluster_method": cluster_method,
        "module_count": len(module_rows),
        "cluster_count": sum(1 for cluster_id in cluster_buckets if cluster_id >= 0),
        "casebook_count": min(args.top_casebooks, len(ranked_rows)),
    }
    (output_dir / "module_discovery_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
