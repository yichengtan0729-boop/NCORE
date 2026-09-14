from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ncore.calibration import binary_calibration_metrics
from ncore.checkpointing import load_compatible_model_state
from ncore.config import load_config
from ncore.metrics import binary_ranking_metrics, dual_metric_score, normalized_auprc
from ncore.models.model import NCORE
from ncore.training import (
    collect_final_logits,
    load_saved_correction_bound,
    load_saved_reason_threshold,
    make_loader,
)


def _load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def evaluate_ensemble(cfg, checkpoint_dir, split, device):
    checkpoint_dir = Path(checkpoint_dir)
    selection_path = checkpoint_dir / "ensemble_weights_v5.json"
    selection = json.loads(selection_path.read_text())
    if selection.get("fit_split") != "val":
        raise ValueError("refusing ensemble weights not selected on validation")
    checkpoint_names = selection["checkpoints"]
    weights = np.asarray(selection["selected"]["weights"], dtype=np.float64)
    if len(checkpoint_names) != 3 or weights.shape != (3,):
        raise ValueError("v5 ensemble selection must contain three checkpoints")
    loader = make_loader(cfg, split, False)
    logits_rows = []
    labels = masks = None
    for checkpoint_name in checkpoint_names:
        checkpoint_path = checkpoint_dir / checkpoint_name
        state = _load(checkpoint_path, device)
        model = NCORE(cfg).to(device)
        load_compatible_model_state(model, state)
        load_saved_correction_bound(model, checkpoint_dir)
        load_saved_reason_threshold(model, checkpoint_dir)
        logits, labels, masks = collect_final_logits(
            model, loader, device, state.get("stage", "grpo")
        )
        logits_rows.append(logits)
    ensemble_logits = torch.stack(logits_rows, 0)
    ensemble_logits = (
        ensemble_logits * torch.as_tensor(weights).view(3, 1, 1)
    ).sum(0)
    results = {
        "split": split,
        "selection_split": "val",
        "checkpoints": checkpoint_names,
        "weights": weights.tolist(),
    }
    for task_index, task_name in enumerate(cfg["experiment"]["task_names"]):
        valid = masks[:, task_index] > 0
        task_labels = labels[valid, task_index].numpy()
        task_logits = ensemble_logits[valid, task_index].numpy()
        auroc, auprc = binary_ranking_metrics(task_labels, task_logits)
        prevalence = float(np.mean(task_labels))
        probability = 1.0 / (1.0 + np.exp(-np.clip(task_logits, -80, 80)))
        results[f"ensemble_auroc_{task_name}"] = auroc
        results[f"ensemble_auprc_{task_name}"] = auprc
        results[f"ensemble_normalized_auprc_{task_name}"] = normalized_auprc(
            auprc, prevalence
        )
        results[f"ensemble_dual_score_{task_name}"] = dual_metric_score(
            auroc, auprc, prevalence
        )
        for metric, value in binary_calibration_metrics(
            task_labels, probability
        ).items():
            results[f"ensemble_{metric}_{task_name}"] = value
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    requested_device = cfg.get("device", "cuda")
    device = torch.device(requested_device if torch.cuda.is_available() else "cpu")
    results = evaluate_ensemble(cfg, args.checkpoint_dir, args.split, device)
    rendered = json.dumps(results, indent=2, sort_keys=True, allow_nan=True)
    print(rendered)
    if args.output_json:
        Path(args.output_json).write_text(rendered + "\n")


if __name__ == "__main__":
    main()
