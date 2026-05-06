from __future__ import annotations

import argparse
import csv
import gzip
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import torch


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cache frozen protein language model embeddings for downstream precomputed_plm task-mode training."
    )
    parser.add_argument("--input", default="data/processed/training/viral_protein_training_index.tsv.gz")
    parser.add_argument("--output", required=True, help="Torch file to write {accessions, embeddings, metadata}")
    parser.add_argument("--model-name", default="facebook/esm2_t33_650M_UR50D", help="Hugging Face model id")
    parser.add_argument("--accession-field", default="protein_accession")
    parser.add_argument("--sequence-field", default="protein_sequence")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--batch-size", type=int, default=4, help="Number of windows batched per forward pass")
    parser.add_argument("--window-size", type=int, default=1022, help="Max residues per PLM window before pooling")
    parser.add_argument("--window-overlap", type=int, default=128, help="Overlap between adjacent windows")
    parser.add_argument("--max-records", type=int, default=0, help="Optional cap for smoke tests")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing output file")
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    value = requested.strip().lower()
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if value == "cuda" and torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device(value)


def sliding_windows(sequence: str, window_size: int, overlap: int) -> list[str]:
    if window_size <= 0:
        raise ValueError("--window-size must be positive")
    if overlap < 0 or overlap >= window_size:
        raise ValueError("--window-overlap must be in [0, window_size)")
    if len(sequence) <= window_size:
        return [sequence]
    step = window_size - overlap
    windows: list[str] = []
    start = 0
    while start < len(sequence):
        chunk = sequence[start : start + window_size]
        if not chunk:
            break
        windows.append(chunk)
        if start + window_size >= len(sequence):
            break
        start += step
    return windows


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    # Drop special tokens at both ends when the tokenizer inserted them.
    token_mask = attention_mask.clone()
    if token_mask.shape[1] >= 2:
        token_mask[:, 0] = 0
        lengths = token_mask.sum(dim=1)
        for row_idx, length in enumerate(lengths.tolist()):
            if length > 0:
                last_token = int(attention_mask[row_idx].sum().item()) - 1
                if last_token >= 0:
                    token_mask[row_idx, last_token] = 0
    token_mask = token_mask.unsqueeze(-1).to(dtype=last_hidden_state.dtype)
    denom = token_mask.sum(dim=1).clamp_min(1.0)
    return (last_hidden_state * token_mask).sum(dim=1) / denom


def batched(iterable: list[str], size: int) -> Iterable[list[str]]:
    for idx in range(0, len(iterable), size):
        yield iterable[idx : idx + size]


def main() -> int:
    args = parse_args()
    root = repo_root()
    input_path = (root / args.input).resolve()
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = (root / output_path).resolve()

    if output_path.exists() and not args.force:
        raise FileExistsError(f"Output already exists: {output_path}. Pass --force to overwrite.")

    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "cache_plm_embeddings.py requires `transformers`. "
            "Install it first, for example: pip install transformers sentencepiece"
        ) from exc

    device = choose_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name)
    model.to(device)
    model.eval()

    accessions: list[str] = []
    embeddings: list[np.ndarray] = []
    sequence_lengths: list[int] = []

    with open_text(input_path) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row_idx, row in enumerate(reader, start=1):
            accession = str(row.get(args.accession_field, "") or "").strip()
            sequence = str(row.get(args.sequence_field, "") or "").strip().upper()
            if not accession or not sequence:
                continue
            accessions.append(accession)
            sequence_lengths.append(len(sequence))

            windows = sliding_windows(sequence, args.window_size, args.window_overlap)
            pooled_windows: list[torch.Tensor] = []
            with torch.no_grad():
                for batch in batched(windows, args.batch_size):
                    encoded = tokenizer(
                        batch,
                        padding=True,
                        truncation=True,
                        return_tensors="pt",
                    )
                    encoded = {key: value.to(device) for key, value in encoded.items()}
                    outputs = model(**encoded)
                    pooled = mean_pool(outputs.last_hidden_state, encoded["attention_mask"])
                    pooled_windows.extend(item.detach().cpu() for item in pooled)
            stacked = torch.stack(pooled_windows)
            embedding = stacked.mean(dim=0).to(dtype=torch.float32).numpy()
            embeddings.append(embedding)

            if row_idx % 1000 == 0:
                print(f"[plm] cached {row_idx} proteins")
            if args.max_records and len(accessions) >= args.max_records:
                break

    payload = {
        "created_at": timestamp(),
        "input": str(input_path),
        "model_name": args.model_name,
        "window_size": args.window_size,
        "window_overlap": args.window_overlap,
        "accessions": accessions,
        "embeddings": np.asarray(embeddings, dtype=np.float16),
        "sequence_lengths": sequence_lengths,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    print(
        json.dumps(
            {
                "created_at": payload["created_at"],
                "output": str(output_path),
                "protein_count": len(accessions),
                "embedding_dim": int(payload["embeddings"].shape[1]) if len(accessions) else 0,
                "device": str(device),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
