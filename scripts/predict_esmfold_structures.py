#!/usr/bin/env python3
"""Predict monomer structures from FASTA with the ESMFold Python API.

This is a small compatibility wrapper for environments where `import esm`
works but the historical `python -m esm.scripts.fold` entrypoint is absent.
It expects fair-esm style APIs, especially `esm.pretrained.esmfold_v1`.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable


SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--input", type=Path, required=True, help="Input FASTA.")
    parser.add_argument("-o", "--output-dir", type=Path, required=True, help="Output directory for PDB files.")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--max-tokens-per-batch", type=int, default=0, help="Reserved for CLI compatibility; sequences are processed one by one.")
    parser.add_argument("--cpu-offload", action="store_true", help="Accepted for CLI compatibility; this wrapper uses normal model.to(device).")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_fasta(path: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    name = ""
    seq_parts: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name:
                    records.append((name, "".join(seq_parts)))
                name = line[1:].split()[0].split("|")[0]
                seq_parts = []
            else:
                seq_parts.append(line)
    if name:
        records.append((name, "".join(seq_parts)))
    if not records:
        raise SystemExit(f"No FASTA records found: {path}")
    return records


def safe_name(name: str) -> str:
    cleaned = SAFE_NAME_RE.sub("_", name.strip())
    return cleaned or "sequence"


def choose_device(requested: str):
    import torch

    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise SystemExit("Requested --device cuda but torch.cuda.is_available() is false.")
    return requested


def mean_plddt_from_pdb(pdb: str) -> float:
    values: list[float] = []
    seen: set[tuple[str, str]] = set()
    for line in pdb.splitlines():
        if not line.startswith("ATOM"):
            continue
        atom = line[12:16].strip()
        chain = line[21].strip()
        residue = line[22:26].strip()
        key = (chain, residue)
        if atom == "CA" and key not in seen:
            seen.add(key)
            try:
                values.append(float(line[60:66]))
            except ValueError:
                pass
    return sum(values) / len(values) if values else 0.0


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import torch
        import esm
    except ImportError as exc:
        raise SystemExit(f"ESMFold dependencies are unavailable in this Python environment: {exc}") from exc

    pretrained = getattr(esm, "pretrained", None)
    if pretrained is None or not hasattr(pretrained, "esmfold_v1"):
        raise SystemExit(
            "This esm package does not expose esm.pretrained.esmfold_v1(). "
            "Install fair-esm with ESMFold support or pass --esmfold-bin to the runner."
        )

    device = choose_device(args.device)
    records = read_fasta(args.input)
    print(f"[esmfold] loading esmfold_v1 on {device}; records={len(records)}", flush=True)
    model = pretrained.esmfold_v1()
    model = model.eval()
    if hasattr(model, "set_chunk_size"):
        model.set_chunk_size(args.chunk_size)
    model = model.to(device)

    rows = []
    with torch.no_grad():
        for idx, (name, sequence) in enumerate(records, start=1):
            out_path = args.output_dir / f"{safe_name(name)}.pdb"
            if out_path.exists() and not args.overwrite:
                rows.append({"protein_accession": name, "pdb_path": str(out_path), "status": "reused_existing"})
                print(f"[esmfold] reuse {idx}/{len(records)} {name}", flush=True)
                continue
            print(f"[esmfold] predict {idx}/{len(records)} {name} length={len(sequence)}", flush=True)
            try:
                pdb = model.infer_pdb(sequence)
            except RuntimeError as exc:
                if device == "cuda":
                    torch.cuda.empty_cache()
                rows.append({"protein_accession": name, "pdb_path": str(out_path), "status": f"failed: {exc}"})
                print(f"[esmfold] failed {name}: {exc}", flush=True)
                continue
            out_path.write_text(pdb, encoding="utf-8")
            rows.append(
                {
                    "protein_accession": name,
                    "pdb_path": str(out_path),
                    "status": "predicted",
                    "sequence_length": len(sequence),
                    "mean_ca_plddt": round(mean_plddt_from_pdb(pdb), 3),
                }
            )
            if device == "cuda":
                torch.cuda.empty_cache()

    manifest = {
        "input": str(args.input),
        "output_dir": str(args.output_dir),
        "record_count": len(records),
        "predicted_or_reused": sum(1 for row in rows if not str(row.get("status", "")).startswith("failed")),
        "failed": sum(1 for row in rows if str(row.get("status", "")).startswith("failed")),
        "device": device,
    }
    (args.output_dir / "esmfold_prediction_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    with (args.output_dir / "esmfold_prediction_manifest.tsv").open("w", encoding="utf-8") as handle:
        fields = ["protein_accession", "pdb_path", "status", "sequence_length", "mean_ca_plddt"]
        handle.write("\t".join(fields) + "\n")
        for row in rows:
            handle.write("\t".join(str(row.get(field, "")) for field in fields) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
