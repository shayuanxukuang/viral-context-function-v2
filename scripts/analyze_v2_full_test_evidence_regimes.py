#!/usr/bin/env python3
"""Build full-test sequence/structure evidence regimes for ViruFunc V2.

This script is intentionally evidence-table oriented. It does not use
post-hoc homology, Phold, Foldseek, or product text as model inputs. These
sources are only used to stratify family-heldout test proteins after model
training.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


CLAIM_FRAME = (
    "Genome context is evaluated as a label-specific prioritization signal; "
    "sequence, Phold/Foldseek-like, and module evidence are post hoc triage strata."
)

LABELS = [
    "polymerase",
    "helicase",
    "protease",
    "capsid_head",
    "tail_assembly",
    "portal_terminase_packaging",
    "lysis",
    "envelope_glycoprotein",
    "membrane_matrix",
    "nucleocapsid",
    "integrase_recombinase",
    "nuclease",
    "methyltransferase",
    "ligase",
    "transcription_regulator",
    "polyprotein",
    "tail_fiber_receptor",
]

DEFAULT_LABEL_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("portal_terminase_packaging", re.compile(r"\b(portal|terminase|packag|headful|large terminase|small terminase)\b", re.I)),
    ("tail_fiber_receptor", re.compile(r"\b(tail fiber|tail fibre|receptor[- ]binding|host[- ]range|attachment protein|tail spike|spike protein|baseplate fiber)\b", re.I)),
    ("envelope_glycoprotein", re.compile(r"\b(envelope|glycoprotein|viral spike|spike glycoprotein|fusion protein)\b", re.I)),
    ("membrane_matrix", re.compile(r"\b(membrane|matrix|matrix protein|M protein|transmembrane)\b", re.I)),
    ("integrase_recombinase", re.compile(r"\b(integrase|recombinase|resolvase|transposase)\b", re.I)),
    ("transcription_regulator", re.compile(r"\b(transcription|regulator|repressor|activator|anti[- ]terminator)\b", re.I)),
    ("methyltransferase", re.compile(r"\b(methyltransferase|methylase)\b", re.I)),
    ("nucleocapsid", re.compile(r"\b(nucleocapsid|nucleoprotein|\bN protein\b|capsid N)\b", re.I)),
    ("capsid_head", re.compile(r"\b(capsid|major capsid|head protein|coat protein|portal cap)\b", re.I)),
    ("tail_assembly", re.compile(r"\b(tail|baseplate|base plate|tape measure|sheath|tube|fiber|wedge)\b", re.I)),
    ("polymerase", re.compile(r"\b(polymerase|replicase|RNA-dependent|DNA polymerase|RdRp)\b", re.I)),
    ("helicase", re.compile(r"\b(helicase)\b", re.I)),
    ("protease", re.compile(r"\b(protease|peptidase)\b", re.I)),
    ("nuclease", re.compile(r"\b(nuclease|endonuclease|exonuclease|RNase|DNase)\b", re.I)),
    ("ligase", re.compile(r"\b(ligase)\b", re.I)),
    ("lysis", re.compile(r"\b(lysis|lysin|holin|endolysin|spanin|amidase|muramidase)\b", re.I)),
    ("polyprotein", re.compile(r"\b(polyprotein)\b", re.I)),
]


def open_text(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", newline="", encoding="utf-8")
    return path.open("r", newline="", encoding="utf-8")


def read_tsv(path: Path) -> list[dict[str, str]]:
    with open_text(path) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames:
            raise SystemExit(f"No header found in TSV: {path}")
        return [dict(row) for row in reader]


def stream_tsv(path: Path) -> Iterable[dict[str, str]]:
    with open_text(path) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames:
            raise SystemExit(f"No header found in TSV: {path}")
        for row in reader:
            yield dict(row)


def write_tsv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def as_float(value: Any, default: float = math.nan) -> float:
    if value is None:
        return default
    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "none", "null"}:
        return default
    try:
        return float(text)
    except ValueError:
        return default


def as_int(value: Any, default: int = 0) -> int:
    x = as_float(value, math.nan)
    return default if math.isnan(x) else int(x)


def first_present(row: dict[str, Any], names: Iterable[str]) -> str:
    for name in names:
        if name in row and str(row.get(name, "")).strip() != "":
            return str(row.get(name, "")).strip()
    return ""


def parse_label_list(value: Any) -> list[str]:
    text = str(value or "").strip()
    if not text or text in {"[]", "nan", "None"}:
        return []
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return sorted({str(x).strip() for x in parsed if str(x).strip()})
    except Exception:
        pass
    parts = re.split(r"[;,|]", text.strip("[]"))
    return sorted({p.strip().strip("'\"") for p in parts if p.strip().strip("'\"")})


def label_text(labels: Iterable[str]) -> str:
    return json.dumps(sorted({x for x in labels if x}), ensure_ascii=False)


def load_test_manifest(path: Path, split_column: str, test_value: str) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for row in stream_tsv(path):
        if split_column not in row:
            raise SystemExit(f"Required split column '{split_column}' not found in {path}")
        if row.get(split_column) == test_value:
            acc = first_present(row, ["protein_accession", "query", "accession"])
            if acc:
                out[acc] = row
    if not out:
        raise SystemExit(f"No test proteins found in {path} where {split_column} == {test_value}")
    return out


def valid_by_thresholds(row: dict[str, str], args: argparse.Namespace) -> bool:
    if not row:
        return False
    pident = as_float(row.get("pident"), math.nan)
    bits = as_float(row.get("bits"), math.nan)
    evalue = as_float(row.get("evalue"), math.nan)
    qcov = as_float(first_present(row, ["qcov", "query_coverage", "qcovs"]), math.nan)
    tcov = as_float(first_present(row, ["tcov", "target_coverage", "tcovs"]), math.nan)
    if not math.isnan(pident) and pident < args.min_seq_identity:
        return False
    if not math.isnan(bits) and bits < args.min_seq_bits:
        return False
    if args.max_seq_evalue is not None and not math.isnan(evalue) and evalue > args.max_seq_evalue:
        return False
    if not math.isnan(qcov) and qcov < args.min_seq_qcov:
        return False
    if not math.isnan(tcov) and tcov < args.min_seq_tcov:
        return False
    return True


def load_homology_top_hit(path: Path, scheme: str, subset: str) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for row in stream_tsv(path):
        if row.get("scheme", scheme) != scheme:
            continue
        if row.get("subset", subset) != subset:
            continue
        query = first_present(row, ["query", "protein_accession"])
        if query:
            out[query] = row
    return out


def load_homology_topk(path: Path | None, scheme: str, subset: str) -> dict[str, list[dict[str, str]]]:
    if not path:
        return {}
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in stream_tsv(path):
        if "scheme" in row and row.get("scheme") != scheme:
            continue
        if "subset" in row and row.get("subset") != subset:
            continue
        query = first_present(row, ["query", "protein_accession"])
        if query:
            grouped[query].append(row)
    for query, rows in grouped.items():
        rows.sort(key=lambda r: (as_float(r.get("bits"), 0.0), as_float(r.get("pident"), 0.0)), reverse=True)
    return grouped


def load_candidate_context(path: Path | None, min_gain: float) -> dict[str, dict[str, Any]]:
    if not path:
        return {}
    out: dict[str, dict[str, Any]] = {}
    for row in stream_tsv(path):
        acc = first_present(row, ["protein_accession", "query"])
        if not acc:
            continue
        gain = as_float(first_present(row, ["context_gain", "delta_p"]), math.nan)
        prev = out.get(acc)
        if prev is None or (not math.isnan(gain) and gain > as_float(prev.get("context_gain"), -math.inf)):
            out[acc] = {
                "candidate_label": first_present(row, ["candidate_label", "predicted_label", "top_label", "label"]),
                "top_probability_calibrated": first_present(row, ["top_probability_calibrated", "p_genome_aware", "p_context"]),
                "context_gain": "" if math.isnan(gain) else gain,
                "high_context_gain": int((not math.isnan(gain) and gain >= min_gain) or str(row.get("high_context_gain", "")) == "1"),
                "module_supported": row.get("module_supported", ""),
                "hypothetical_or_unknown": row.get("hypothetical_or_unknown", ""),
            }
    return out


def load_prediction_deltas(path: Path | None, min_gain: float) -> dict[str, dict[str, Any]]:
    """Load optional long-format full-test model deltas.

    Expected flexible columns include protein_accession/query, label/candidate_label,
    p_genome_aware/p_context/right_prob, p_protein_only/p_protein/left_prob,
    context_gain/delta_p, and optional y_true.
    """
    if not path:
        return {}
    by_protein: dict[str, dict[str, Any]] = {}
    for row in stream_tsv(path):
        acc = first_present(row, ["protein_accession", "query", "accession"])
        if not acc:
            continue
        label = first_present(row, ["label", "candidate_label", "predicted_label", "top_label"])
        p_context = as_float(first_present(row, ["p_genome_aware", "p_context", "right_prob", "genome_prob"]), math.nan)
        p_protein = as_float(first_present(row, ["p_protein_only", "p_protein", "left_prob", "protein_prob"]), math.nan)
        gain = as_float(first_present(row, ["context_gain", "delta_p"]), math.nan)
        if math.isnan(gain) and not math.isnan(p_context) and not math.isnan(p_protein):
            gain = p_context - p_protein
        y_true = as_float(first_present(row, ["y_true", "target", "truth"]), math.nan)
        rec = by_protein.setdefault(
            acc,
            {
                "max_context_gain": -math.inf,
                "top_context_label": "",
                "top_context_probability": "",
                "top_protein_probability": "",
                "high_context_gain": 0,
                "prediction_label_count": 0,
                "positive_label_count": 0,
            },
        )
        rec["prediction_label_count"] += 1
        if not math.isnan(y_true) and y_true > 0:
            rec["positive_label_count"] += 1
        if not math.isnan(gain) and gain > as_float(rec.get("max_context_gain"), -math.inf):
            rec["max_context_gain"] = gain
            rec["top_context_label"] = label
            rec["top_context_probability"] = "" if math.isnan(p_context) else p_context
            rec["top_protein_probability"] = "" if math.isnan(p_protein) else p_protein
            rec["high_context_gain"] = int(gain >= min_gain)
    for rec in by_protein.values():
        if rec["max_context_gain"] == -math.inf:
            rec["max_context_gain"] = ""
    return by_protein


def load_label_map(path: Path | None) -> list[tuple[str, re.Pattern[str]]]:
    if not path:
        return DEFAULT_LABEL_PATTERNS
    rows = read_tsv(path)
    patterns: list[tuple[str, re.Pattern[str]]] = []
    for row in rows:
        label = first_present(row, ["label", "candidate_label", "virufunc_label"])
        pattern = first_present(row, ["pattern", "regex", "text"])
        if not label or not pattern:
            raise SystemExit(f"Label map requires columns label and pattern/regex/text: {path}")
        patterns.append((label, re.compile(pattern, re.I)))
    return patterns


def map_text_to_labels(text: str, patterns: list[tuple[str, re.Pattern[str]]]) -> list[str]:
    labels = [label for label, pattern in patterns if pattern.search(text or "")]
    return sorted(set(labels))


def load_phold(path: Path | None, patterns: list[tuple[str, re.Pattern[str]]], args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    if not path:
        return {}
    out: dict[str, dict[str, Any]] = {}
    for row in stream_tsv(path):
        acc = first_present(row, ["protein_accession", "query", "cds", "ID", "id", "locus_tag"])
        if not acc:
            continue
        text = " ".join(
            first_present(row, [name])
            for name in [
                "product",
                "phrog",
                "phrog_name",
                "phrog_category",
                "category",
                "description",
                "target",
                "target_name",
                "annotation",
            ]
            if name in row
        )
        explicit_labels = parse_label_list(first_present(row, ["mapped_labels", "virufunc_labels", "target_labels", "labels"]))
        mapped = explicit_labels or map_text_to_labels(text, patterns)
        score = as_float(first_present(row, ["score", "bits", "prob", "probability"]), math.nan)
        evalue = as_float(first_present(row, ["evalue", "E-value", "eval"]), math.nan)
        margin = as_float(first_present(row, ["label_margin", "prob_margin", "score_margin"]), math.nan)
        no_hit_text = first_present(row, ["no_hit", "orphan", "hit"])
        has_hit = True
        if no_hit_text.lower() in {"1", "true", "yes", "no_hit", "orphan", "false"}:
            has_hit = no_hit_text.lower() not in {"1", "true", "yes", "no_hit", "orphan"}
        if args.max_structure_evalue is not None and not math.isnan(evalue) and evalue > args.max_structure_evalue:
            confident = False
        elif not math.isnan(score) and score < args.min_structure_score:
            confident = False
        else:
            confident = has_hit
        if not has_hit:
            regime = "viral_structure_orphan"
        elif len(mapped) > 1 or (not math.isnan(margin) and margin < args.min_structure_label_margin):
            regime = "structure_ambiguous"
        elif confident and len(mapped) == 1:
            regime = "structure_resolved"
        elif has_hit:
            regime = "structure_weak"
        else:
            regime = "viral_structure_orphan"
        out[acc] = {
            "structure_regime": regime,
            "structure_mapped_labels": label_text(mapped),
            "structure_top_label": mapped[0] if len(mapped) == 1 else "",
            "structure_score": "" if math.isnan(score) else score,
            "structure_evalue": "" if math.isnan(evalue) else evalue,
            "structure_label_margin": "" if math.isnan(margin) else margin,
            "phold_product": first_present(row, ["product", "annotation", "description"]),
        }
    return out


def classify_sequence(
    acc: str,
    top: dict[str, str],
    topk: list[dict[str, str]],
    candidate_label: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if not top:
        return {
            "sequence_regime": "sequence_orphan",
            "sequence_has_hit": 0,
            "sequence_label_agreement": 0,
            "sequence_ambiguous_label_count": 0,
            "query_labels": "[]",
            "target_labels": "[]",
        }
    query_labels = parse_label_list(top.get("query_labels", ""))
    target_labels = parse_label_list(top.get("target_labels", ""))
    valid_top = valid_by_thresholds(top, args)
    agreement_labels = set(query_labels) & set(target_labels)
    candidate_agreement = bool(candidate_label and candidate_label in target_labels)
    label_agreement = bool(agreement_labels) or candidate_agreement
    topk_labels: set[str] = set()
    for hit in topk[: args.sequence_ambiguity_top_k]:
        if valid_by_thresholds(hit, args):
            topk_labels.update(parse_label_list(hit.get("target_labels", "")))
    ambiguous = len(topk_labels) >= 2
    if ambiguous:
        regime = "sequence_ambiguous"
    elif valid_top and label_agreement:
        regime = "sequence_resolved"
    elif valid_top:
        regime = "sequence_weak"
    else:
        regime = "sequence_weak"
    return {
        "sequence_regime": regime,
        "sequence_has_hit": 1,
        "sequence_label_agreement": int(label_agreement),
        "sequence_candidate_label_agreement": int(candidate_agreement),
        "sequence_ambiguous_label_count": len(topk_labels),
        "query_labels": label_text(query_labels),
        "target_labels": label_text(target_labels),
        "homology_target": top.get("target", ""),
        "homology_pident": top.get("pident", ""),
        "homology_bits": top.get("bits", ""),
        "homology_evalue": top.get("evalue", ""),
        "homology_qcov": first_present(top, ["qcov", "query_coverage", "qcovs"]),
        "homology_tcov": first_present(top, ["tcov", "target_coverage", "tcovs"]),
    }


def combined_regime(sequence_regime: str, structure_regime: str) -> str:
    seq_strong = sequence_regime == "sequence_resolved"
    struct_strong = structure_regime == "structure_resolved"
    seq_orphan = sequence_regime == "sequence_orphan"
    struct_orphan = structure_regime in {"viral_structure_orphan", "structure_not_available", "structure_not_covered"}
    if seq_strong and struct_strong:
        return "conventional_resolved"
    if seq_strong and not struct_strong:
        return "sequence_resolved_only"
    if struct_strong and not seq_strong:
        return "structure_resolved_only"
    if seq_orphan and struct_orphan:
        return "conventional_orphan"
    return "weak_or_ambiguous_evidence"


def summarize(rows: list[dict[str, Any]], group_cols: list[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(str(row.get(col, "")) for col in group_cols)].append(row)
    out: list[dict[str, Any]] = []
    for key, vals in sorted(grouped.items()):
        gains = [as_float(v.get("max_context_gain"), math.nan) for v in vals]
        gains = [g for g in gains if not math.isnan(g)]
        row = {col: key[idx] for idx, col in enumerate(group_cols)}
        row.update(
            {
                "protein_count": len(vals),
                "with_context_gain_count": len(gains),
                "mean_context_gain": "" if not gains else sum(gains) / len(gains),
                "median_context_gain": "" if not gains else sorted(gains)[len(gains) // 2],
                "high_context_count": sum(as_int(v.get("high_context_gain"), 0) for v in vals),
                "candidate_context_count": sum(1 for v in vals if str(v.get("candidate_label", "")) != ""),
                "sequence_label_agreement_count": sum(as_int(v.get("sequence_label_agreement"), 0) for v in vals),
                "structure_resolved_count": sum(1 for v in vals if v.get("structure_regime") == "structure_resolved"),
            }
        )
        row["high_context_fraction"] = row["high_context_count"] / len(vals) if vals else ""
        out.append(row)
    return out


def maybe_plot(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    for col, name in [("sequence_regime", "sequence_regime_counts"), ("combined_evidence_regime", "combined_regime_counts")]:
        counts = Counter(str(row.get(col, "")) for row in rows)
        labels = list(counts)
        vals = [counts[x] for x in labels]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(labels, vals, color="#4C78A8")
        ax.set_ylabel("proteins")
        ax.set_title(name.replace("_", " "))
        ax.tick_params(axis="x", labelrotation=25)
        fig.tight_layout()
        fig.savefig(fig_dir / f"{name}.png", dpi=180)
        plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--homology-top-hit", type=Path, required=True, help="S21/MMseqs2 top-hit assignment table.")
    parser.add_argument("--homology-topk", type=Path, help="Optional top-K homology hits with target_labels for ambiguity.")
    parser.add_argument("--candidate-assignments", type=Path, help="Optional validation-targeted candidate assignments.")
    parser.add_argument("--prediction-deltas", type=Path, help="Optional full-test long-format model probability/delta table.")
    parser.add_argument("--phold-evidence", type=Path, help="Optional Phold/ProstT5/Foldseek-style annotation TSV.")
    parser.add_argument("--phold-label-map", type=Path, help="Optional TSV with label and regex/pattern columns.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scheme", default="family_holdout")
    parser.add_argument("--subset", default="all_test")
    parser.add_argument("--split-column", default="family_holdout_split")
    parser.add_argument("--test-value", default="test")
    parser.add_argument("--min-seq-identity", type=float, default=30.0)
    parser.add_argument("--min-seq-bits", type=float, default=50.0)
    parser.add_argument("--max-seq-evalue", type=float, default=None)
    parser.add_argument("--min-seq-qcov", type=float, default=0.0)
    parser.add_argument("--min-seq-tcov", type=float, default=0.0)
    parser.add_argument("--sequence-ambiguity-top-k", type=int, default=10)
    parser.add_argument("--min-context-gain", type=float, default=0.2)
    parser.add_argument("--min-structure-score", type=float, default=0.0)
    parser.add_argument("--max-structure-evalue", type=float, default=None)
    parser.add_argument("--min-structure-label-margin", type=float, default=0.05)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    out_dir = args.output_dir
    tables_dir = out_dir / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    test_manifest = load_test_manifest(args.split_manifest, args.split_column, args.test_value)
    homology = load_homology_top_hit(args.homology_top_hit, args.scheme, args.subset)
    topk = load_homology_topk(args.homology_topk, args.scheme, args.subset)
    candidates = load_candidate_context(args.candidate_assignments, args.min_context_gain)
    prediction_deltas = load_prediction_deltas(args.prediction_deltas, args.min_context_gain)
    label_map = load_label_map(args.phold_label_map)
    phold = load_phold(args.phold_evidence, label_map, args)

    rows: list[dict[str, Any]] = []
    for acc, manifest in sorted(test_manifest.items()):
        cand = candidates.get(acc, {})
        pred = prediction_deltas.get(acc, {})
        candidate_label = str(cand.get("candidate_label") or pred.get("top_context_label") or "")
        seq = classify_sequence(acc, homology.get(acc, {}), topk.get(acc, []), candidate_label, args)
        struct = phold.get(acc, {})
        if not struct:
            struct = {
                "structure_regime": "structure_not_available" if not args.phold_evidence else "structure_not_covered",
                "structure_mapped_labels": "[]",
            }
        max_gain = pred.get("max_context_gain", "")
        if max_gain == "" and cand.get("context_gain", "") != "":
            max_gain = cand.get("context_gain", "")
        high_context = int(as_int(pred.get("high_context_gain"), 0) or as_int(cand.get("high_context_gain"), 0))
        row: dict[str, Any] = {
            "protein_accession": acc,
            "genome_version": manifest.get("genome_version", ""),
            "virus_family": manifest.get("virus_family", ""),
            "host_supergroup": manifest.get("host_supergroup", ""),
            "protein_length_aa": manifest.get("protein_length_aa", ""),
            "length_bin": manifest.get("sequence_length_bin", ""),
            "candidate_label": candidate_label,
            "max_context_gain": max_gain,
            "top_context_label": pred.get("top_context_label", candidate_label),
            "top_context_probability": pred.get("top_context_probability", cand.get("top_probability_calibrated", "")),
            "top_protein_probability": pred.get("top_protein_probability", ""),
            "high_context_gain": high_context,
            "module_supported": cand.get("module_supported", ""),
            "prediction_label_count": pred.get("prediction_label_count", ""),
            "positive_label_count": pred.get("positive_label_count", ""),
        }
        row.update(seq)
        row.update(struct)
        row["combined_evidence_regime"] = combined_regime(row["sequence_regime"], row["structure_regime"])
        rows.append(row)

    fieldnames = [
        "protein_accession",
        "genome_version",
        "virus_family",
        "host_supergroup",
        "protein_length_aa",
        "length_bin",
        "query_labels",
        "candidate_label",
        "top_context_label",
        "max_context_gain",
        "top_context_probability",
        "top_protein_probability",
        "high_context_gain",
        "module_supported",
        "sequence_regime",
        "sequence_has_hit",
        "sequence_label_agreement",
        "sequence_candidate_label_agreement",
        "sequence_ambiguous_label_count",
        "homology_target",
        "homology_pident",
        "homology_bits",
        "homology_evalue",
        "homology_qcov",
        "homology_tcov",
        "target_labels",
        "structure_regime",
        "structure_mapped_labels",
        "structure_top_label",
        "structure_score",
        "structure_evalue",
        "structure_label_margin",
        "phold_product",
        "combined_evidence_regime",
        "prediction_label_count",
        "positive_label_count",
    ]
    write_tsv(tables_dir / "full_test_evidence_regimes.tsv", rows, fieldnames)
    write_tsv(tables_dir / "summary_by_sequence_regime.tsv", summarize(rows, ["sequence_regime"]))
    write_tsv(tables_dir / "summary_by_structure_regime.tsv", summarize(rows, ["structure_regime"]))
    write_tsv(tables_dir / "summary_by_combined_regime.tsv", summarize(rows, ["combined_evidence_regime"]))
    write_tsv(tables_dir / "summary_by_label_and_regime.tsv", summarize(rows, ["top_context_label", "combined_evidence_regime"]))
    maybe_plot(out_dir, rows)

    report = {
        "claim_frame": CLAIM_FRAME,
        "test_protein_count": len(rows),
        "homology_top_hit_rows": len(homology),
        "homology_topk_queries": len(topk),
        "candidate_context_rows_indexed": len(candidates),
        "prediction_delta_proteins_indexed": len(prediction_deltas),
        "phold_evidence_rows_indexed": len(phold),
        "sequence_regime_counts": dict(Counter(row["sequence_regime"] for row in rows)),
        "structure_regime_counts": dict(Counter(row["structure_regime"] for row in rows)),
        "combined_regime_counts": dict(Counter(row["combined_evidence_regime"] for row in rows)),
        "outputs": {
            "full_test_regimes": str(tables_dir / "full_test_evidence_regimes.tsv"),
            "sequence_summary": str(tables_dir / "summary_by_sequence_regime.tsv"),
            "structure_summary": str(tables_dir / "summary_by_structure_regime.tsv"),
            "combined_summary": str(tables_dir / "summary_by_combined_regime.tsv"),
        },
    }
    (out_dir / "full_test_evidence_regime_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
