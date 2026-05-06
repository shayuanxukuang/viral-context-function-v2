#!/usr/bin/env python3
"""Classical homology top-hit label-transfer baselines for ViruFunc V2."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score

from label_rules import LABEL_RULES, label_hits, normalize_text


LABELS = [rule.name for rule in LABEL_RULES]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protein-index", type=Path, default=Path("data/processed/training/viral_protein_training_index.tsv.gz"))
    parser.add_argument("--split-manifest", type=Path, default=Path("data/processed/splits/viral_protein_strict_splits.tsv.gz"))
    parser.add_argument("--freeze-dir", type=Path, default=Path("data/v2_freeze"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--schemes", default="default,family_holdout,host_holdout")
    parser.add_argument("--tool", choices=["mmseqs"], default="mmseqs")
    parser.add_argument("--mmseqs-bin", default="mmseqs")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--sensitivity", default="7.5")
    parser.add_argument("--reuse-hits", action="store_true")
    parser.add_argument("--keep-fasta", action="store_true")
    parser.add_argument("--identity-thresholds", default="30,50,70,90,100")
    return parser.parse_args()


def resolve_executable(value: str) -> str | None:
    if any(sep in value for sep in ("/", "\\")):
        path = Path(value).expanduser()
        return str(path) if path.exists() else None
    return shutil.which(value)


def require_mmseqs(args: argparse.Namespace) -> None:
    resolved = resolve_executable(args.mmseqs_bin)
    if resolved:
        args.mmseqs_bin = resolved
        return
    message = f"""
MMseqs2 executable was not found: {args.mmseqs_bin}

Install MMseqs2 or pass its full executable path. Examples:

  conda install -c conda-forge -c bioconda mmseqs2
  python scripts/run_homology_label_transfer.py ... --mmseqs-bin /path/to/mmseqs

If resuming the full review-completion runner after completed multi-seed and
source-CI steps, use:

  python scripts/run_v2_review_completion.py --run-root <RUN_ROOT> \\
    --skip-multiseed --skip-source-ci --mmseqs-bin /path/to/mmseqs
""".strip()
    raise SystemExit(message)


def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("r", encoding="utf-8", newline="")


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_split_table(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    with open_text(path) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            accession = row.get("protein_accession", "")
            split = row.get("split", "")
            if accession and split:
                out[accession] = split
    return out


def load_split_maps(args: argparse.Namespace) -> dict[str, dict[str, str]]:
    schemes = [scheme.strip() for scheme in args.schemes.split(",") if scheme.strip()]
    maps: dict[str, dict[str, str]] = {}
    strict_needed = any(scheme in {"family_holdout", "host_holdout"} for scheme in schemes)
    if "default" in schemes:
        candidates = [
            args.freeze_dir / "splits" / "default_split.tsv",
            args.freeze_dir / "default_split.tsv",
        ]
        for path in candidates:
            maps["default"] = read_split_table(path)
            if maps["default"]:
                break
    if strict_needed:
        cols = {
            "family_holdout": "family_holdout_split",
            "host_holdout": "host_taxid_holdout_split",
        }
        wanted = {scheme: cols[scheme] for scheme in schemes if scheme in cols}
        maps.update({scheme: {} for scheme in wanted})
        with open_text(args.split_manifest) as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                accession = row.get("protein_accession", "")
                for scheme, col in wanted.items():
                    split = row.get(col, "")
                    if accession and split:
                        maps[scheme][accession] = split
    return {scheme: split_map for scheme, split_map in maps.items() if split_map}


def protein_rows(index_path: Path):
    with open_text(index_path) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            yield row


def fasta_write(handle, accession: str, sequence: str) -> None:
    handle.write(f">{accession}\n")
    for i in range(0, len(sequence), 80):
        handle.write(sequence[i : i + 80] + "\n")


def labels_for_row(row: dict[str, str]) -> list[str]:
    return [LABEL_RULES[idx].name for idx in label_hits(normalize_text(row))]


def prepare_scheme(args: argparse.Namespace, scheme: str, split_map: dict[str, str], out_dir: Path) -> dict[str, Any]:
    train_fasta = out_dir / f"{scheme}.train.faa"
    test_fasta = out_dir / f"{scheme}.test.faa"
    label_map: dict[str, list[str]] = {}
    test_accessions: list[str] = []
    train_sha: set[str] = set()
    test_sha: dict[str, str] = {}
    with train_fasta.open("w", encoding="utf-8") as train_handle, test_fasta.open("w", encoding="utf-8") as test_handle:
        for row in protein_rows(args.protein_index):
            accession = row.get("protein_accession", "")
            split = split_map.get(accession)
            if split not in {"train", "test"}:
                continue
            labels = labels_for_row(row)
            label_map[accession] = labels
            sequence = row.get("protein_sequence", "")
            if not sequence:
                continue
            if split == "train":
                fasta_write(train_handle, accession, sequence)
                train_sha.add(row.get("protein_sequence_sha256", ""))
            elif split == "test":
                fasta_write(test_handle, accession, sequence)
                test_accessions.append(accession)
                test_sha[accession] = row.get("protein_sequence_sha256", "")
    return {
        "train_fasta": train_fasta,
        "test_fasta": test_fasta,
        "label_map": label_map,
        "test_accessions": test_accessions,
        "train_sha": train_sha,
        "test_sha": test_sha,
    }


def run_mmseqs(args: argparse.Namespace, scheme: str, prep: dict[str, Any], out_dir: Path) -> Path:
    hits = out_dir / f"{scheme}.mmseqs_top_hits.tsv"
    if hits.exists() and args.reuse_hits:
        return hits
    tmp = out_dir / f"{scheme}.mmseqs_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    cmd = [
        args.mmseqs_bin,
        "easy-search",
        str(prep["test_fasta"]),
        str(prep["train_fasta"]),
        str(hits),
        str(tmp),
        "--format-output",
        "query,target,pident,evalue,bits",
        "--threads",
        str(args.threads),
        "-s",
        str(args.sensitivity),
    ]
    subprocess.run(cmd, check=True)
    return hits


def parse_top_hits(path: Path) -> dict[str, dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for parts in reader:
            if len(parts) < 5:
                continue
            query, target, pident, evalue, bits = parts[:5]
            row = {
                "query": query,
                "target": target,
                "pident": float(pident),
                "evalue": evalue,
                "bits": float(bits),
            }
            old = best.get(query)
            if old is None or (row["bits"], row["pident"]) > (old["bits"], old["pident"]):
                best[query] = row
    return best


def fmax(y_true: np.ndarray, y_score: np.ndarray) -> float:
    thresholds = np.unique(np.quantile(y_score, np.linspace(0.0, 1.0, 401)))
    best = 0.0
    for thr in thresholds:
        pred = y_score >= thr
        tp = float(np.logical_and(pred, y_true == 1).sum())
        fp = float(np.logical_and(pred, y_true == 0).sum())
        fn = float(np.logical_and(~pred, y_true == 1).sum())
        precision = 0.0 if tp + fp == 0 else tp / (tp + fp)
        recall = 0.0 if tp + fn == 0 else tp / (tp + fn)
        score = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        best = max(best, score)
    return best


def evaluate_scheme(
    scheme: str,
    subset: str,
    accessions: list[str],
    hits: dict[str, dict[str, Any]],
    label_map: dict[str, list[str]],
    identity_thresholds: list[float],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    label_to_idx = {label: idx for idx, label in enumerate(LABELS)}
    y_true = np.zeros((len(accessions), len(LABELS)), dtype=np.uint8)
    y_score = np.zeros((len(accessions), len(LABELS)), dtype=np.float32)
    hit_rows = []
    for i, accession in enumerate(accessions):
        for label in label_map.get(accession, []):
            if label in label_to_idx:
                y_true[i, label_to_idx[label]] = 1
        hit = hits.get(accession)
        if hit:
            hit_labels = label_map.get(hit["target"], [])
            for label in hit_labels:
                if label in label_to_idx:
                    y_score[i, label_to_idx[label]] = max(y_score[i, label_to_idx[label]], hit["pident"] / 100.0)
            hit_rows.append(
                {
                    "scheme": scheme,
                    "subset": subset,
                    "query": accession,
                    "target": hit["target"],
                    "pident": hit["pident"],
                    "bits": hit["bits"],
                    "query_labels": json.dumps(label_map.get(accession, [])),
                    "target_labels": json.dumps(hit_labels),
                }
            )
    aps = []
    fmaxes = []
    label_rows = []
    for label, idx in label_to_idx.items():
        if y_true[:, idx].sum() == 0:
            continue
        ap = float(average_precision_score(y_true[:, idx], y_score[:, idx]))
        fm = fmax(y_true[:, idx], y_score[:, idx])
        aps.append(ap)
        fmaxes.append(fm)
        label_rows.append({"scheme": scheme, "subset": subset, "label": label, "ap": ap, "fmax": fm, "positives": int(y_true[:, idx].sum())})
    micro_ap = float(average_precision_score(y_true.ravel(), y_score.ravel())) if y_true.sum() else math.nan
    micro_fmax = fmax(y_true.ravel(), y_score.ravel()) if y_true.sum() else math.nan
    metric = {
        "scheme": scheme,
        "subset": subset,
        "test_proteins": len(accessions),
        "covered_by_top_hit": len(hit_rows),
        "coverage": 0.0 if not accessions else len(hit_rows) / len(accessions),
        "macro_ap": float(np.mean(aps)) if aps else math.nan,
        "macro_fmax": float(np.mean(fmaxes)) if fmaxes else math.nan,
        "micro_ap": micro_ap,
        "micro_fmax": micro_fmax,
    }
    for thr in identity_thresholds:
        tp = fp = 0
        for row in hit_rows:
            if float(row["pident"]) < thr:
                continue
            query_labels = set(json.loads(row["query_labels"]))
            for label in json.loads(row["target_labels"]):
                if label in query_labels:
                    tp += 1
                else:
                    fp += 1
        metric[f"precision_at_identity_ge_{int(thr)}"] = "" if tp + fp == 0 else tp / (tp + fp)
    return metric, label_rows, hit_rows


def main() -> None:
    args = parse_args()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.tool == "mmseqs":
        require_mmseqs(args)
    split_maps = load_split_maps(args)
    identity_thresholds = [float(x) for x in args.identity_thresholds.split(",") if x.strip()]
    metrics: list[dict[str, Any]] = []
    label_metrics: list[dict[str, Any]] = []
    all_hits: list[dict[str, Any]] = []
    for scheme, split_map in split_maps.items():
        scheme_dir = out_dir / scheme
        scheme_dir.mkdir(parents=True, exist_ok=True)
        prep = prepare_scheme(args, scheme, split_map, scheme_dir)
        hits_path = run_mmseqs(args, scheme, prep, scheme_dir)
        hits = parse_top_hits(hits_path)
        accessions = prep["test_accessions"]
        metric, per_label, hit_rows = evaluate_scheme(scheme, "all_test", accessions, hits, prep["label_map"], identity_thresholds)
        metrics.append(metric)
        label_metrics.extend(per_label)
        all_hits.extend(hit_rows)
        if scheme == "family_holdout":
            strict = [acc for acc in accessions if prep["test_sha"].get(acc, "") not in prep["train_sha"]]
            metric, per_label, hit_rows = evaluate_scheme(scheme, "strict_zero_exact_transfer", strict, hits, prep["label_map"], identity_thresholds)
            metrics.append(metric)
            label_metrics.extend(per_label)
            all_hits.extend(hit_rows)
        if not args.keep_fasta:
            for fasta in (prep["train_fasta"], prep["test_fasta"]):
                fasta.unlink(missing_ok=True)
    write_tsv(out_dir / "homology_top_hit_metrics.tsv", metrics)
    write_tsv(out_dir / "homology_top_hit_label_metrics.tsv", label_metrics)
    write_tsv(out_dir / "homology_top_hit_assignments.tsv", all_hits)
    report = {
        "metrics": str(out_dir / "homology_top_hit_metrics.tsv"),
        "label_metrics": str(out_dir / "homology_top_hit_label_metrics.tsv"),
        "assignments": str(out_dir / "homology_top_hit_assignments.tsv"),
        "schemes": list(split_maps),
        "tool": args.tool,
        "mmseqs_bin": args.mmseqs_bin,
    }
    (out_dir / "homology_top_hit_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
