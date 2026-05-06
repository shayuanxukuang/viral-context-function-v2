from __future__ import annotations

import argparse
import csv
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
    parser = argparse.ArgumentParser(description="Post-hoc uncertainty calibration and selective prediction for task-mode runs.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--fdr-target", type=float, default=0.1, help="Target false discovery rate for candidate gating")
    parser.add_argument("--coverage-grid", default="0.05,0.1,0.2,0.3,0.5,0.7,1.0")
    return parser.parse_args()


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def predict_logits(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    logits_rows: list[np.ndarray] = []
    target_rows: list[np.ndarray] = []
    index_rows: list[np.ndarray] = []
    for batch in loader:
        logits = model(
            batch["tokens"].to(device, non_blocking=True),
            batch["sequence_embedding"].to(device, non_blocking=True),
            batch["global_categories"].to(device, non_blocking=True),
            batch["global_numeric"].to(device, non_blocking=True),
            batch["host_categories"].to(device, non_blocking=True),
            batch["host_numeric"].to(device, non_blocking=True),
            batch["biophysics"].to(device, non_blocking=True),
            batch["neighbor_categories"].to(device, non_blocking=True),
            batch["neighbor_numeric"].to(device, non_blocking=True),
            batch["neighbor_mask"].to(device, non_blocking=True),
        )
        logits_rows.append(logits.detach().cpu().numpy().astype(np.float32))
        target_rows.append(batch["labels"].numpy().astype(np.float32))
        index_rows.append(batch["indices"].numpy().astype(np.int64))
    return np.concatenate(logits_rows), np.concatenate(target_rows), np.concatenate(index_rows)


def fit_temperature(logits: np.ndarray, targets: np.ndarray, device: torch.device) -> float:
    logits_tensor = torch.as_tensor(logits, dtype=torch.float32, device=device)
    targets_tensor = torch.as_tensor(targets, dtype=torch.float32, device=device)
    temperature = torch.nn.Parameter(torch.ones((), dtype=torch.float32, device=device))
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.LBFGS([temperature], lr=0.25, max_iter=50)

    def closure():
        optimizer.zero_grad()
        scaled = logits_tensor / temperature.clamp_min(1e-3)
        loss = criterion(scaled, targets_tensor)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(temperature.detach().cpu().clamp_min(1e-3).item())


def sigmoid_with_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    scaled = np.clip(logits / max(temperature, 1e-6), -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-scaled))


def best_f1_threshold(probabilities: np.ndarray, targets: np.ndarray) -> float:
    candidate_thresholds = np.unique(probabilities)
    if candidate_thresholds.shape[0] > 512:
        candidate_thresholds = np.quantile(candidate_thresholds, np.linspace(0.0, 1.0, 257))
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in candidate_thresholds:
        predictions = probabilities >= threshold
        tp = float(np.sum((predictions == 1) & (targets == 1)))
        fp = float(np.sum((predictions == 1) & (targets == 0)))
        fn = float(np.sum((predictions == 0) & (targets == 1)))
        denominator = (2.0 * tp) + fp + fn
        f1 = 0.0 if denominator <= 0 else (2.0 * tp) / denominator
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)
    return best_threshold


def precision_target_threshold(probabilities: np.ndarray, targets: np.ndarray, precision_target: float) -> tuple[float, float]:
    ordering = np.argsort(probabilities)[::-1]
    sorted_probs = probabilities[ordering]
    sorted_targets = targets[ordering]
    tp = np.cumsum(sorted_targets)
    fp = np.cumsum(1 - sorted_targets)
    precision = tp / np.maximum(tp + fp, 1)
    valid = np.where(precision >= precision_target)[0]
    if valid.size == 0:
        return 1.0, 0.0
    best_idx = int(valid[-1])
    return float(sorted_probs[best_idx]), float(precision[best_idx])


def micro_counts(predictions: np.ndarray, targets: np.ndarray) -> tuple[int, int, int]:
    tp = int(np.sum((predictions == 1) & (targets == 1)))
    fp = int(np.sum((predictions == 1) & (targets == 0)))
    fn = int(np.sum((predictions == 0) & (targets == 1)))
    return tp, fp, fn


def precision_recall_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    if precision + recall == 0:
        return precision, recall, 0.0
    return precision, recall, (2.0 * precision * recall) / (precision + recall)


def build_model(cache: dict[str, Any], manifest: dict[str, Any]) -> TaskModeV2Model:
    config = manifest.get("config", {}) or {}
    model = TaskModeV2Model(
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
    return model


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


def main() -> int:
    args = parse_args()
    root = repo_root()
    run_dir = (root / args.run_dir).resolve() if not Path(args.run_dir).is_absolute() else Path(args.run_dir).resolve()
    output_dir = (Path(args.output_dir).resolve() if args.output_dir else run_dir / "uncertainty")
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    cache = torch.load(run_dir / "dataset_cache.pt", map_location="cpu", weights_only=False)
    checkpoint = torch.load(run_dir / "best_model.pt", map_location="cpu", weights_only=False)
    threshold_payload = json.loads((run_dir / "best_thresholds.json").read_text(encoding="utf-8"))
    label_names: list[str] = list(cache["label_names"])
    base_thresholds = np.asarray([float(threshold_payload["thresholds"][name]) for name in label_names], dtype=np.float32)

    device = choose_device(args.device)
    model = build_model(cache, manifest)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)

    splits = np.asarray(cache["splits"])
    val_idx = np.where(splits == 1)[0]
    test_idx = np.where(splits == 2)[0]
    val_loader = dataloader_for_indices(cache, val_idx, args, device)
    test_loader = dataloader_for_indices(cache, test_idx, args, device)

    val_logits, val_targets, _ = predict_logits(model, val_loader, device)
    test_logits, test_targets, test_indices = predict_logits(model, test_loader, device)
    temperature = fit_temperature(val_logits, val_targets, device)
    val_prob = sigmoid_with_temperature(val_logits, temperature)
    test_prob = sigmoid_with_temperature(test_logits, temperature)

    threshold_rows: list[dict[str, Any]] = []
    f1_thresholds: list[float] = []
    precision_thresholds: list[float] = []
    precision_target = 1.0 - float(args.fdr_target)
    for label_idx, label_name in enumerate(label_names):
        label_targets_val = val_targets[:, label_idx].astype(np.int8)
        label_prob_val = val_prob[:, label_idx]
        if int(label_targets_val.sum()) == 0:
            f1_threshold = 1.0
            precision_threshold = 1.0
            empirical_precision = 0.0
        else:
            f1_threshold = best_f1_threshold(label_prob_val, label_targets_val)
            precision_threshold, empirical_precision = precision_target_threshold(label_prob_val, label_targets_val, precision_target)
        f1_thresholds.append(f1_threshold)
        precision_thresholds.append(precision_threshold)
        threshold_rows.append(
            {
                "label": label_name,
                "base_threshold": float(base_thresholds[label_idx]),
                "temperature_scaled_best_f1_threshold": float(f1_threshold),
                "temperature_scaled_precision_target_threshold": float(precision_threshold),
                "precision_target": precision_target,
                "empirical_val_precision_at_target_threshold": empirical_precision,
                "val_support": int(label_targets_val.sum()),
            }
        )
    write_tsv(output_dir / "per_label_thresholds.tsv", threshold_rows)

    scaled_f1_thresholds = np.asarray(f1_thresholds, dtype=np.float32)
    scaled_precision_thresholds = np.asarray(precision_thresholds, dtype=np.float32)
    top_scores_val = val_prob.max(axis=1)
    top_scores_test = test_prob.max(axis=1)
    top_idx_val = val_prob.argmax(axis=1)
    top_idx_test = test_prob.argmax(axis=1)
    top_correct_val = val_targets[np.arange(val_targets.shape[0]), top_idx_val] == 1
    top_correct_test = test_targets[np.arange(test_targets.shape[0]), top_idx_test] == 1
    candidate_gate_threshold, empirical_val_precision = precision_target_threshold(top_scores_val, top_correct_val.astype(np.int8), precision_target)

    coverage_rows: list[dict[str, Any]] = []
    coverage_values = [float(token.strip()) for token in args.coverage_grid.split(",") if token.strip()]
    ordering = np.argsort(top_scores_test)[::-1]
    n_test = test_prob.shape[0]
    for coverage in coverage_values:
        keep = max(1, int(round(coverage * n_test)))
        selected = ordering[:keep]
        selected_mask = np.zeros(n_test, dtype=bool)
        selected_mask[selected] = True
        predictions = (test_prob >= scaled_f1_thresholds).astype(np.int8)
        predictions[~selected_mask] = 0
        tp, fp, fn = micro_counts(predictions, test_targets.astype(np.int8))
        precision, recall, f1 = precision_recall_f1(tp, fp, fn)
        coverage_rows.append(
            {
                "coverage": coverage,
                "selected_examples": int(selected_mask.sum()),
                "top1_precision": float(top_correct_test[selected_mask].mean()) if selected_mask.any() else 0.0,
                "micro_precision": precision,
                "micro_recall": recall,
                "micro_f1": f1,
            }
        )
    write_tsv(output_dir / "coverage_curves.tsv", coverage_rows)

    candidate_mask = top_scores_test >= candidate_gate_threshold
    candidate_predictions = (test_prob >= scaled_precision_thresholds).astype(np.int8)
    candidate_rows: list[dict[str, Any]] = []
    for local_idx in np.argsort(top_scores_test)[::-1]:
        global_idx = int(test_indices[local_idx])
        predicted_labels = [label_names[label_idx] for label_idx in np.where(candidate_predictions[local_idx] == 1)[0]]
        candidate_rows.append(
            {
                "protein_accession": str(cache["protein_accessions"][global_idx]),
                "virus_tax_id": str(cache["virus_tax_ids"][global_idx]),
                "genome_version": str(cache["genome_versions"][global_idx]),
                "description": str(cache["descriptions"][global_idx]),
                "top_label": label_names[int(top_idx_test[local_idx])],
                "top_probability_calibrated": float(top_scores_test[local_idx]),
                "passes_fdr_gate": bool(candidate_mask[local_idx]),
                "predicted_labels_at_precision_threshold": json.dumps(predicted_labels, ensure_ascii=False),
                "top_label_in_true_labels": bool(top_correct_test[local_idx]),
            }
        )
    write_tsv(output_dir / "candidate_prioritization.tsv", candidate_rows)

    report = {
        "created_at": timestamp(),
        "run_dir": str(run_dir),
        "temperature": temperature,
        "fdr_target": float(args.fdr_target),
        "precision_target": precision_target,
        "candidate_gate_threshold": candidate_gate_threshold,
        "empirical_val_top1_precision": empirical_val_precision,
        "selected_test_candidates": int(candidate_mask.sum()),
        "coverage_rows": coverage_rows,
    }
    (output_dir / "uncertainty_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
