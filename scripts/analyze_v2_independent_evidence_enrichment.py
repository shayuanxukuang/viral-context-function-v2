#!/usr/bin/env python3
"""Compare post hoc independent-evidence support in high-context candidates and matched controls.

This analysis is intentionally downstream of model training. Sequence hits, domain
hits, structure hits, product text, and manual evidence are evidence for triage,
not de novo model inputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_INTEGRATED = Path(
    "artifacts/return/v2_plos_cb_supplementary_package_20260506/"
    "supplementary_tables/S22_validation_integrated_evidence.tsv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--integrated-evidence", type=Path, default=DEFAULT_INTEGRATED)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate-type", default="high_context_candidate")
    parser.add_argument("--control-type", default="matched_control")
    parser.add_argument("--remote-identity-max", type=float, default=60.0)
    parser.add_argument("--module-consistency-min", type=float, default=0.75)
    parser.add_argument("--foldseek-prob-min", type=float, default=0.9)
    parser.add_argument("--foldseek-qtm-min", type=float, default=0.4)
    parser.add_argument("--foldseek-coverage-min", type=float, default=0.35)
    parser.add_argument("--plddt-min", type=float, default=70.0)
    parser.add_argument(
        "--phrog-phold-hits",
        type=Path,
        default=None,
        help="Optional TSV keyed by protein_accession/query with PHROG/Phold labels or annotations.",
    )
    parser.add_argument(
        "--domain-hits",
        type=Path,
        default=None,
        help="Optional InterPro/Pfam/HMM hit TSV keyed by protein_accession/query.",
    )
    parser.add_argument(
        "--manual-evidence",
        type=Path,
        default=None,
        help="Optional manual literature/support TSV keyed by protein_accession/query.",
    )
    return parser.parse_args()


def read_tsv(path: Path, *, required: bool = True, table_name: str = "table") -> list[dict[str, str]]:
    if not path.exists():
        if required:
            raise SystemExit(f"Required {table_name} not found: {path}")
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise SystemExit(f"{table_name} has no header: {path}")
        return list(reader)


def write_tsv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
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
        writer.writerows(rows)


def require_columns(rows: list[dict[str, str]], columns: list[str], table_name: str) -> None:
    fieldnames = set(rows[0].keys()) if rows else set()
    missing = [col for col in columns if col not in fieldnames]
    if missing:
        raise SystemExit(f"{table_name} is missing required columns: {', '.join(missing)}")


def as_float(value: Any, default: float = math.nan) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def as_bool(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return text in {"1", "true", "yes", "y", "supported", "support", "label_agreement", "agree"}


def present(value: Any) -> bool:
    text = str(value or "").strip()
    return bool(text) and text.lower() not in {"nan", "none", "null", "-1"}


def normalize_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


LABEL_TERMS = {
    "capsid_head": {"capsid", "major capsid", "head protein", "coat protein"},
    "tail_assembly": {"tail", "baseplate", "tail assembly", "tail fiber", "tail sheath", "tail tube"},
    "portal_terminase_packaging": {"portal", "terminase", "packaging"},
    "polymerase": {"polymerase", "replicase", "rna dependent", "dna polymerase"},
    "nucleocapsid": {"nucleocapsid", "nucleoprotein", "capsid protein n", "protein n"},
    "membrane_matrix": {"membrane", "matrix", "matrix protein", "m protein"},
    "envelope_glycoprotein": {"envelope", "glycoprotein", "spike"},
    "lysis": {"lysis", "lysin", "holin", "endolysin"},
    "protease": {"protease", "peptidase"},
    "helicase": {"helicase"},
    "nuclease": {"nuclease", "endonuclease", "exonuclease"},
    "ligase": {"ligase"},
    "methyltransferase": {"methyltransferase", "methylase"},
    "transcription_regulator": {"transcription", "regulator", "anti terminator", "antiterminator"},
    "polyprotein": {"polyprotein"},
    "integrase_recombinase": {"integrase", "recombinase", "resolvase"},
}


def label_agrees(predicted_label: str, evidence_text: str) -> bool:
    label = normalize_text(predicted_label).replace(" ", "_")
    text = normalize_text(evidence_text)
    if not label or not text:
        return False
    terms = set(LABEL_TERMS.get(label, set()))
    terms.add(label.replace("_", " "))
    return any(term in text for term in terms)


def index_rows(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    out: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        accession = (
            row.get("protein_accession")
            or row.get("protein_id")
            or row.get("query")
            or row.get("accession")
            or row.get("candidate_id")
            or ""
        )
        if accession:
            out[accession].append(row)
    return out


def row_text(row: dict[str, str]) -> str:
    useful = []
    for key, value in row.items():
        if key.lower() in {"sequence", "seq", "aa_sequence"}:
            continue
        if present(value):
            useful.append(str(value))
    return " | ".join(useful)


def optional_support(
    accession: str,
    predicted_label: str,
    table: dict[str, list[dict[str, str]]],
    *,
    agreement_columns: tuple[str, ...],
    support_columns: tuple[str, ...] = (),
) -> tuple[bool, bool, str]:
    rows = table.get(accession, [])
    if not rows:
        return False, False, ""
    any_hit = True
    label_hit = False
    snippets = []
    for row in rows:
        for col in support_columns:
            if col in row and as_bool(row.get(col)):
                label_hit = True
        texts = []
        for col in agreement_columns:
            if col in row and present(row.get(col)):
                texts.append(str(row.get(col)))
        if not texts:
            texts.append(row_text(row))
        joined = " | ".join(texts)
        snippets.append(joined[:200])
        if label_agrees(predicted_label, joined):
            label_hit = True
    return any_hit, label_hit, " || ".join(snippets[:3])


def fisher_two_sided(a: int, b: int, c: int, d: int) -> float:
    # Table [[a, b], [c, d]]. Condition on margins.
    row1 = a + b
    row2 = c + d
    col1 = a + c
    total = row1 + row2
    if total == 0:
        return math.nan

    def hypergeom(x: int) -> float:
        return math.comb(col1, x) * math.comb(total - col1, row1 - x) / math.comb(total, row1)

    lo = max(0, row1 - (total - col1))
    hi = min(row1, col1)
    p_obs = hypergeom(a)
    p = sum(hypergeom(x) for x in range(lo, hi + 1) if hypergeom(x) <= p_obs + 1e-12)
    return min(1.0, p)


def mcnemar_exact(candidate_only: int, control_only: int) -> float:
    n = candidate_only + control_only
    if n == 0:
        return math.nan
    k = min(candidate_only, control_only)
    p = 2.0 * sum(math.comb(n, i) for i in range(0, k + 1)) / (2.0**n)
    return min(1.0, p)


def make_flag_rows(
    rows: list[dict[str, str]],
    args: argparse.Namespace,
    phrog_phold: dict[str, list[dict[str, str]]],
    domains: dict[str, list[dict[str, str]]],
    manual: dict[str, list[dict[str, str]]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        accession = row.get("protein_accession", "")
        label = row.get("predicted_label", "")
        sequence_status = row.get("sequence_evidence_status", "")
        nearest_identity = as_float(row.get("nearest_homolog_identity"))
        nearest_hit = present(row.get("nearest_homolog_accession"))
        target_labels = row.get("nearest_homolog_target_labels", "")

        sequence_any_hit = nearest_hit
        mmseqs_remote_hit = nearest_hit and (
            "remote" in sequence_status
            or (not math.isnan(nearest_identity) and nearest_identity <= args.remote_identity_max)
        )
        mmseqs_label_agreement = nearest_hit and (
            label_agrees(label, target_labels)
            or (
                "without_target_function_label" not in sequence_status
                and "weak_sequence_hit" not in sequence_status
                and "missing" not in sequence_status
            )
        )

        mean_plddt = as_float(row.get("mean_plddt"))
        foldseek_prob = as_float(row.get("foldseek_top_prob"), 0.0)
        foldseek_qtm = as_float(row.get("foldseek_top_qtmscore"), 0.0)
        foldseek_cov = as_float(row.get("foldseek_query_coverage"), 0.0)
        structure_status = row.get("structure_evidence_status", "")
        foldseek_confident_hit = (
            not math.isnan(mean_plddt)
            and mean_plddt >= args.plddt_min
            and foldseek_prob >= args.foldseek_prob_min
            and foldseek_qtm >= args.foldseek_qtm_min
            and foldseek_cov >= args.foldseek_coverage_min
        )
        foldseek_functionally_informative = structure_status in {
            "structure_consistent",
            "structure_consistent_but_ambiguous",
            "local_structure_consistent",
        }

        module_consistency = as_float(row.get("module_neighborhood_consistency"))
        module_consistent = present(row.get("module_cluster_id")) and (
            math.isnan(module_consistency) or module_consistency >= args.module_consistency_min
        )

        phrog_any, phrog_agree, phrog_text = optional_support(
            accession,
            label,
            phrog_phold,
            agreement_columns=(
                "phrog",
                "phrog_label",
                "phrog_annotation",
                "phold_label",
                "phold_prediction",
                "function",
                "annotation",
                "description",
            ),
            support_columns=("label_agreement", "agrees_with_prediction", "supported"),
        )
        domain_any, domain_agree, domain_text = optional_support(
            accession,
            label,
            domains,
            agreement_columns=(
                "interpro",
                "interpro_name",
                "pfam",
                "pfam_name",
                "hmm_name",
                "hmm_description",
                "domain",
                "description",
                "signature_description",
            ),
            support_columns=("label_agreement", "agrees_with_prediction", "supported"),
        )
        manual_any, manual_supported, manual_text = optional_support(
            accession,
            label,
            manual,
            agreement_columns=("support_status", "evidence_status", "literature_note", "interpretation", "description"),
            support_columns=("label_agreement", "literature_supported", "supported"),
        )

        direct_label_support = any(
            [
                mmseqs_label_agreement,
                phrog_agree,
                domain_agree,
                manual_supported,
                foldseek_confident_hit and foldseek_functionally_informative,
            ]
        )
        any_independent_support = any(
            [
                mmseqs_remote_hit,
                mmseqs_label_agreement,
                phrog_agree,
                domain_agree,
                manual_supported,
                foldseek_confident_hit,
                foldseek_functionally_informative,
                module_consistent,
            ]
        )
        sequence_structure_resolved = any(
            [
                mmseqs_label_agreement,
                phrog_agree,
                domain_agree,
                manual_supported,
                foldseek_confident_hit,
            ]
        )
        context_complement_regime = module_consistent and not sequence_structure_resolved
        weak_or_ambiguous_sequence_structure = not sequence_structure_resolved or structure_status in {
            "structure_consistent_but_ambiguous",
            "ambiguous_or_weak_structure_signal",
            "structure_hit_on_low_confidence_model",
            "low_model_confidence",
            "no_foldseek_hit_available",
            "weak_or_inconclusive",
        }

        out.append(
            {
                "protein_accession": accession,
                "target_type": row.get("target_type", ""),
                "matched_candidate_id": row.get("matched_candidate_id", ""),
                "predicted_label": label,
                "delta_p": row.get("delta_p", ""),
                "p_context": row.get("p_context", ""),
                "nearest_homolog_identity": row.get("nearest_homolog_identity", ""),
                "sequence_evidence_status": sequence_status,
                "structure_evidence_status": structure_status,
                "module_cluster_id": row.get("module_cluster_id", ""),
                "module_neighborhood_consistency": row.get("module_neighborhood_consistency", ""),
                "sequence_any_mmseqs_hit": int(sequence_any_hit),
                "mmseqs_remote_hit": int(mmseqs_remote_hit),
                "mmseqs_label_agreement": int(mmseqs_label_agreement),
                "phrog_phold_any_hit": int(phrog_any),
                "phrog_phold_label_agreement": int(phrog_agree),
                "domain_hmm_any_hit": int(domain_any),
                "domain_hmm_label_agreement": int(domain_agree),
                "foldseek_confident_structural_hit": int(foldseek_confident_hit),
                "foldseek_functionally_informative": int(foldseek_functionally_informative),
                "module_consistent": int(module_consistent),
                "manual_literature_supported": int(manual_supported),
                "direct_label_support": int(direct_label_support),
                "any_independent_support": int(any_independent_support),
                "sequence_structure_resolved": int(sequence_structure_resolved),
                "weak_or_ambiguous_sequence_structure": int(weak_or_ambiguous_sequence_structure),
                "context_complement_regime": int(context_complement_regime),
                "phrog_phold_evidence_text": phrog_text,
                "domain_hmm_evidence_text": domain_text,
                "manual_evidence_text": manual_text,
            }
        )
    return out


def summarize_flags(
    flag_rows: list[dict[str, Any]],
    candidate_type: str,
    control_type: str,
    optional_available: dict[str, bool],
) -> list[dict[str, Any]]:
    features = [
        ("sequence_any_mmseqs_hit", "MMseqs2 any train/reference hit"),
        ("mmseqs_remote_hit", "MMseqs2 remote/low-identity hit"),
        ("mmseqs_label_agreement", "MMseqs2 top-hit label agreement"),
        ("phrog_phold_any_hit", "PHROG/Phold any hit"),
        ("phrog_phold_label_agreement", "PHROG/Phold label agreement"),
        ("domain_hmm_any_hit", "InterPro/Pfam/HMM any hit"),
        ("domain_hmm_label_agreement", "InterPro/Pfam/HMM label agreement"),
        ("foldseek_confident_structural_hit", "Foldseek confident structural hit"),
        ("foldseek_functionally_informative", "Foldseek functionally informative class"),
        ("module_consistent", "Module neighborhood consistency"),
        ("manual_literature_supported", "Manual literature-supported example"),
        ("direct_label_support", "Direct label support"),
        ("any_independent_support", "Any independent/post hoc support"),
        ("sequence_structure_resolved", "Sequence/structure-resolved evidence"),
        ("weak_or_ambiguous_sequence_structure", "Weak/ambiguous sequence-structure regime"),
        ("context_complement_regime", "Module-supported weak-evidence regime"),
    ]
    candidates = [row for row in flag_rows if row["target_type"] == candidate_type]
    controls = [row for row in flag_rows if row["target_type"] == control_type]
    out: list[dict[str, Any]] = []
    unavailable_features = {
        "phrog_phold_any_hit": not optional_available.get("phrog_phold", False),
        "phrog_phold_label_agreement": not optional_available.get("phrog_phold", False),
        "domain_hmm_any_hit": not optional_available.get("domain_hmm", False),
        "domain_hmm_label_agreement": not optional_available.get("domain_hmm", False),
        "manual_literature_supported": not optional_available.get("manual", False),
    }
    for feature, label in features:
        if unavailable_features.get(feature, False):
            out.append(
                {
                    "feature": feature,
                    "evidence_axis": label,
                    "source_available": 0,
                    "candidate_supported": "",
                    "candidate_total": "",
                    "candidate_rate": "",
                    "control_supported": "",
                    "control_total": "",
                    "control_rate": "",
                    "rate_difference_candidate_minus_control": "",
                    "rate_ratio": "",
                    "haldane_odds_ratio": "",
                    "fisher_exact_p": "",
                    "note": "optional evidence source not supplied",
                }
            )
            continue
        cand_pos = sum(int(row[feature]) for row in candidates)
        ctrl_pos = sum(int(row[feature]) for row in controls)
        cand_total = len(candidates)
        ctrl_total = len(controls)
        cand_neg = cand_total - cand_pos
        ctrl_neg = ctrl_total - ctrl_pos
        cand_rate = cand_pos / cand_total if cand_total else math.nan
        ctrl_rate = ctrl_pos / ctrl_total if ctrl_total else math.nan
        diff = cand_rate - ctrl_rate
        rr = (cand_rate / ctrl_rate) if ctrl_rate > 0 else math.inf
        odds_ratio = ((cand_pos + 0.5) * (ctrl_neg + 0.5)) / ((cand_neg + 0.5) * (ctrl_pos + 0.5))
        fisher_p = fisher_two_sided(cand_pos, cand_neg, ctrl_pos, ctrl_neg)
        out.append(
            {
                "feature": feature,
                "evidence_axis": label,
                "source_available": 1,
                "candidate_supported": cand_pos,
                "candidate_total": cand_total,
                "candidate_rate": round(cand_rate, 6),
                "control_supported": ctrl_pos,
                "control_total": ctrl_total,
                "control_rate": round(ctrl_rate, 6),
                "rate_difference_candidate_minus_control": round(diff, 6),
                "rate_ratio": "inf" if math.isinf(rr) else round(rr, 6),
                "haldane_odds_ratio": round(odds_ratio, 6),
                "fisher_exact_p": "" if math.isnan(fisher_p) else round(fisher_p, 6),
                "note": "",
            }
        )
    return out


def paired_rows(flag_rows: list[dict[str, Any]], candidate_type: str, control_type: str) -> list[dict[str, Any]]:
    by_candidate = {row["protein_accession"]: row for row in flag_rows if row["target_type"] == candidate_type}
    controls = [row for row in flag_rows if row["target_type"] == control_type]
    features = [
        "mmseqs_label_agreement",
        "foldseek_confident_structural_hit",
        "foldseek_functionally_informative",
        "module_consistent",
        "direct_label_support",
        "sequence_structure_resolved",
        "weak_or_ambiguous_sequence_structure",
        "context_complement_regime",
    ]
    out = []
    for control in controls:
        candidate = by_candidate.get(str(control.get("matched_candidate_id", "")))
        if candidate is None:
            continue
        row = {
            "candidate_accession": candidate["protein_accession"],
            "control_accession": control["protein_accession"],
            "predicted_label": candidate["predicted_label"],
        }
        for feature in features:
            row[f"candidate_{feature}"] = candidate[feature]
            row[f"control_{feature}"] = control[feature]
            row[f"delta_{feature}"] = int(candidate[feature]) - int(control[feature])
        out.append(row)
    return out


def summarize_pairs(pair_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    features = sorted(
        {
            key.removeprefix("candidate_")
            for row in pair_rows
            for key in row
            if key.startswith("candidate_") and key != "candidate_accession"
        }
    )
    out = []
    for feature in features:
        cand_only = 0
        ctrl_only = 0
        both = 0
        neither = 0
        for row in pair_rows:
            c = int(row[f"candidate_{feature}"])
            m = int(row[f"control_{feature}"])
            if c and m:
                both += 1
            elif c and not m:
                cand_only += 1
            elif m and not c:
                ctrl_only += 1
            else:
                neither += 1
        total = both + cand_only + ctrl_only + neither
        out.append(
            {
                "feature": feature,
                "matched_pairs": total,
                "both_supported": both,
                "candidate_only": cand_only,
                "control_only": ctrl_only,
                "neither_supported": neither,
                "paired_rate_difference": round((cand_only - ctrl_only) / total, 6) if total else "",
                "mcnemar_exact_p": "" if total == 0 else round(mcnemar_exact(cand_only, ctrl_only), 6),
            }
        )
    return out


def make_figures(summary: list[dict[str, Any]], out_dir: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    selected = [
        "mmseqs_label_agreement",
        "foldseek_confident_structural_hit",
        "foldseek_functionally_informative",
        "module_consistent",
        "direct_label_support",
        "sequence_structure_resolved",
        "context_complement_regime",
    ]
    rows = [row for row in summary if row["feature"] in selected]
    labels = [str(row["evidence_axis"]) for row in rows]
    cand = [float(row["candidate_rate"]) for row in rows]
    ctrl = [float(row["control_rate"]) for row in rows]
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(8.2, max(4.4, 0.42 * len(rows) + 1.2)))
    ax.barh(y + 0.18, cand, height=0.34, color="#2563EB", label="High-context candidates")
    ax.barh(y - 0.18, ctrl, height=0.34, color="#94A3B8", label="Matched controls")
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 1.12)
    ax.set_xlabel("Support rate")
    ax.set_title("Independent evidence support: candidates vs matched controls", loc="left", fontsize=10, fontweight="bold")
    ax.grid(axis="x", color="#D8DEE9", linewidth=0.6, alpha=0.8)
    ax.text(
        0.98,
        0.04,
        "blue = high-context candidates; gray = matched controls",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="#334155",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 2.0},
    )
    for idx, row in enumerate(rows):
        ax.text(cand[idx] + 0.015, idx + 0.18, f"{int(row['candidate_supported'])}/{int(row['candidate_total'])}", va="center", fontsize=7)
        ax.text(ctrl[idx] + 0.015, idx - 0.18, f"{int(row['control_supported'])}/{int(row['control_total'])}", va="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(fig_dir / "independent_evidence_support_rates.png", dpi=240, bbox_inches="tight")
    fig.savefig(fig_dir / "independent_evidence_support_rates.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    evidence_path = args.integrated_evidence.resolve()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_tsv(evidence_path, table_name="integrated evidence")
    if not rows:
        raise SystemExit(f"Integrated evidence table is empty: {evidence_path}")
    require_columns(rows, ["target_type", "protein_accession", "predicted_label"], "integrated evidence")

    selected = [row for row in rows if row.get("target_type") in {args.candidate_type, args.control_type}]
    if not selected:
        raise SystemExit(
            f"No rows found for target types {args.candidate_type!r} and {args.control_type!r} in {evidence_path}"
        )

    phrog_rows = read_tsv(args.phrog_phold_hits.resolve(), required=False, table_name="PHROG/Phold hits") if args.phrog_phold_hits else []
    domain_rows = read_tsv(args.domain_hits.resolve(), required=False, table_name="domain hits") if args.domain_hits else []
    manual_rows = read_tsv(args.manual_evidence.resolve(), required=False, table_name="manual evidence") if args.manual_evidence else []

    flags = make_flag_rows(
        selected,
        args,
        phrog_phold=index_rows(phrog_rows),
        domains=index_rows(domain_rows),
        manual=index_rows(manual_rows),
    )
    summary = summarize_flags(
        flags,
        args.candidate_type,
        args.control_type,
        optional_available={
            "phrog_phold": bool(phrog_rows),
            "domain_hmm": bool(domain_rows),
            "manual": bool(manual_rows),
        },
    )
    pairs = paired_rows(flags, args.candidate_type, args.control_type)
    pair_summary = summarize_pairs(pairs)
    make_figures(summary, out_dir)

    tables_dir = out_dir / "tables"
    write_tsv(tables_dir / "independent_evidence_flags.tsv", flags)
    write_tsv(tables_dir / "independent_evidence_enrichment.tsv", summary)
    write_tsv(tables_dir / "independent_evidence_matched_pairs.tsv", pairs)
    write_tsv(tables_dir / "independent_evidence_paired_summary.tsv", pair_summary)

    report = {
        "claim_frame": "Post hoc independent-evidence enrichment for candidate triage; evidence sources are not de novo model inputs.",
        "integrated_evidence": str(evidence_path),
        "candidate_type": args.candidate_type,
        "control_type": args.control_type,
        "candidate_count": sum(row["target_type"] == args.candidate_type for row in flags),
        "control_count": sum(row["target_type"] == args.control_type for row in flags),
        "optional_sources": {
            "phrog_phold_hits": str(args.phrog_phold_hits.resolve()) if args.phrog_phold_hits else None,
            "domain_hits": str(args.domain_hits.resolve()) if args.domain_hits else None,
            "manual_evidence": str(args.manual_evidence.resolve()) if args.manual_evidence else None,
            "phrog_phold_rows": len(phrog_rows),
            "domain_rows": len(domain_rows),
            "manual_rows": len(manual_rows),
        },
        "thresholds": {
            "remote_identity_max": args.remote_identity_max,
            "module_consistency_min": args.module_consistency_min,
            "foldseek_prob_min": args.foldseek_prob_min,
            "foldseek_qtm_min": args.foldseek_qtm_min,
            "foldseek_coverage_min": args.foldseek_coverage_min,
            "plddt_min": args.plddt_min,
        },
        "outputs": {
            "flags": str(tables_dir / "independent_evidence_flags.tsv"),
            "enrichment": str(tables_dir / "independent_evidence_enrichment.tsv"),
            "matched_pairs": str(tables_dir / "independent_evidence_matched_pairs.tsv"),
            "paired_summary": str(tables_dir / "independent_evidence_paired_summary.tsv"),
            "figures": str(out_dir / "figures"),
        },
        "interpretation_guardrail": "A lower direct sequence/structure support rate in high-context candidates is compatible with the paper's complementarity claim when these candidates are enriched for module-supported weak-evidence regimes.",
    }
    (out_dir / "independent_evidence_enrichment_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
