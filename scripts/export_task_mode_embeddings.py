from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from train_overnight_baseline import choose_device
from train_task_modes import TaskModeDataset, TaskModeV2Model, make_collate_fn


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export per-protein embeddings from a trained task-mode run.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--representation", choices=("sequence", "context", "host", "fused"), default="fused")
    parser.add_argument("--split", choices=("all", "train", "val", "test"), default="all")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    return parser.parse_args()


def build_model(cache: dict[str, Any], manifest: dict[str, Any]) -> TaskModeV2Model:
    config = manifest.get("config", {}) or {}
    return TaskModeV2Model(
        sequence_backbone=str(manifest.get("sequence_backbone", config.get("sequence_backbone", "cnn"))),
        plm_embedding_dim=int(cache.get("plm_embedding_dim", 0)),
        global_category_sizes=[int(cache["global_categories"][:, idx].max()) + 1 for idx in range(cache["global_categories"].shape[1])],
        global_numeric_dim=int(cache["global_numeric"].shape[1]),
        host_category_sizes=[int(cache["host_categories"][:, idx].max()) + 1 for idx in range(cache["host_categories"].shape[1])],
        host_numeric_dim=int(cache["host_numeric"].shape[1]),
        neighbor_category_sizes=[int(cache["neighbor_categories"][:, :, idx].max()) + 1 for idx in range(cache["neighbor_categories"].shape[2])],
        neighbor_numeric_dim=int(cache["neighbor_numeric"].shape[2]),
        neighbor_slot_count=int(cache["neighbor_categories"].shape[1]),
        biophysics_dim=int(cache["biophysics"].shape[1]),
        num_labels=int(cache["labels"].shape[1]),
        embed_dim=int(config.get("embed_dim", 128)),
        hidden_dim=int(config.get("hidden_dim", 256)),
        dropout=float(config.get("dropout", 0.2)),
    )


def dataloader_for_indices(cache: dict[str, Any], indices: np.ndarray, args: argparse.Namespace, device: torch.device) -> DataLoader:
    dataset = TaskModeDataset(cache, indices)
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "collate_fn": make_collate_fn(),
    }
    if args.num_workers > 0:
        kwargs["prefetch_factor"] = args.prefetch_factor
    return DataLoader(**kwargs)


@torch.no_grad()
def collect_representations(
    model: TaskModeV2Model,
    loader: DataLoader,
    device: torch.device,
    representation: str,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    vectors: list[np.ndarray] = []
    indices: list[np.ndarray] = []
    for batch in loader:
        tokens = batch["tokens"].to(device, non_blocking=True)
        sequence_embedding = batch["sequence_embedding"].to(device, non_blocking=True)
        global_categories = batch["global_categories"].to(device, non_blocking=True)
        global_numeric = batch["global_numeric"].to(device, non_blocking=True)
        host_categories = batch["host_categories"].to(device, non_blocking=True)
        host_numeric = batch["host_numeric"].to(device, non_blocking=True)
        biophysics = batch["biophysics"].to(device, non_blocking=True)
        neighbor_categories = batch["neighbor_categories"].to(device, non_blocking=True)
        neighbor_numeric = batch["neighbor_numeric"].to(device, non_blocking=True)
        neighbor_mask = batch["neighbor_mask"].to(device, non_blocking=True)

        sequence_vec = model.sequence_encoder(tokens, sequence_embedding)
        if model.biophysics_encoder is not None:
            bio_vec = model.biophysics_encoder(biophysics)
            sequence_vec = torch.cat([sequence_vec, bio_vec], dim=-1)
        sequence_vec = model.sequence_project(sequence_vec)
        global_vec = model.global_encoder(global_categories, global_numeric)
        neighbor_vec = model.neighbor_encoder(neighbor_categories, neighbor_numeric, neighbor_mask)
        context_vec = model.context_project(torch.cat([global_vec, neighbor_vec], dim=-1))
        host_vec = model.host_encoder(host_categories, host_numeric)
        gate_weights = torch.softmax(model.gate(torch.cat([sequence_vec, context_vec, host_vec], dim=-1)), dim=-1)
        fused_vec = torch.sum(torch.stack([sequence_vec, context_vec, host_vec], dim=1) * gate_weights.unsqueeze(-1), dim=1)

        chosen = {
            "sequence": sequence_vec,
            "context": context_vec,
            "host": host_vec,
            "fused": fused_vec,
        }[representation]
        vectors.append(chosen.detach().cpu().numpy().astype(np.float16))
        indices.append(batch["indices"].numpy().astype(np.int64))
    return np.concatenate(indices), np.concatenate(vectors)


def main() -> int:
    args = parse_args()
    root = repo_root()
    run_dir = (root / args.run_dir).resolve() if not Path(args.run_dir).is_absolute() else Path(args.run_dir).resolve()
    output_path = (root / args.output).resolve() if not Path(args.output).is_absolute() else Path(args.output).resolve()

    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    cache = torch.load(run_dir / "dataset_cache.pt", map_location="cpu", weights_only=False)
    checkpoint = torch.load(run_dir / "best_model.pt", map_location="cpu", weights_only=False)
    device = choose_device(args.device)

    splits = np.asarray(cache["splits"])
    if args.split == "all":
        indices = np.arange(splits.shape[0], dtype=np.int64)
    elif args.split == "train":
        indices = np.where(splits == 0)[0]
    elif args.split == "val":
        indices = np.where(splits == 1)[0]
    else:
        indices = np.where(splits == 2)[0]

    model = build_model(cache, manifest)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)

    loader = dataloader_for_indices(cache, indices, args, device)
    exported_indices, embeddings = collect_representations(model, loader, device, args.representation)
    payload = {
        "created_at": timestamp(),
        "run_dir": str(run_dir),
        "representation": args.representation,
        "split": args.split,
        "accessions": [str(cache["protein_accessions"][int(idx)]) for idx in exported_indices],
        "embeddings": embeddings,
        "genome_versions": [str(cache["genome_versions"][int(idx)]) for idx in exported_indices],
        "virus_tax_ids": [str(cache["virus_tax_ids"][int(idx)]) for idx in exported_indices],
        "descriptions": [str(cache["descriptions"][int(idx)]) for idx in exported_indices],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    print(json.dumps({"output": str(output_path), "count": len(payload["accessions"]), "embedding_dim": int(embeddings.shape[1])}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
