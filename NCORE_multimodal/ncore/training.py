from __future__ import annotations
from pathlib import Path
from typing import Dict, Any
from contextlib import nullcontext
import copy
import json
import random
import shutil
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

from .calibration import (
    binary_calibration_metrics,
    fit_temperature,
    load_temperature,
    save_temperature,
)
from .data.dataset import ManifestDataset, collate_manifest
from .ema import ModelEMA, ModelSWA
from .metrics import (
    DEFAULT_ENSEMBLE_WEIGHTS,
    checkpoint_passes_direct_guard,
    dual_metric_score,
    metric_sort_key,
    normalized_auprc,
    performance_guard,
    routing_ranking_metrics,
    select_ensemble_weights,
)
from .sampling import PositiveAwareBatchSampler, apply_modality_dropout
from .models.model import NCORE
from .losses import (
    alpha_detached_no_harm_loss,
    auc_ranking_loss,
    direct_performance_objective,
    gate_supervision_loss,
    gate_utility_targets,
    hard_case_weights,
    no_harm_loss_from_per_sample,
    pairwise_ranking_loss,
    precision_ranking_loss,
    normalized_gate_supervision_loss,
    oracle_candidate_utility,
    patient_wise_utility_normalize,
    path_distillation_loss,
    reranker_objective,
    categorical_policy_kl,
    reason_gate_loss,
    residual_utility_ranking_loss,
)
from .rl.grpo import (
    per_sample_masked_bce,
    reward_components,
    group_advantages,
    clipped_grpo_loss,
)
from .numerics import assert_finite_tensor


def make_loader(cfg, split, shuffle=False):
    dataset = ManifestDataset(cfg["paths"]["manifest"], cfg, split)
    sampler_cfg = cfg.get("training", {}).get("positive_aware_sampler", {})
    if split == "train" and shuffle and bool(sampler_cfg.get("enabled", False)):
        task_name = cfg["experiment"]["task_names"][0]
        label_spec = dataset.labels[task_name]
        labels = dataset.df[label_spec["column"]].fillna(0).to_numpy(dtype=np.float32)
        mask_column = label_spec.get("mask_column")
        if mask_column in dataset.df.columns:
            valid = dataset.df[mask_column].fillna(0).to_numpy() > 0.5
            labels = np.where(valid, labels, 0.0)
        batch_sampler = PositiveAwareBatchSampler(
            labels,
            int(cfg["training"]["batch_size"]),
            min_positive_per_batch=int(sampler_cfg.get("min_positive_per_batch", 4)),
            positive_fraction_target=float(sampler_cfg.get("positive_fraction_target", 0.20)),
            seed=int(cfg.get("seed", 42)),
            drop_last=bool(sampler_cfg.get("drop_last", False)),
        )
        return DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=int(cfg["training"].get("num_workers", 0)),
            collate_fn=lambda batch: collate_manifest(batch, cfg),
            pin_memory=torch.cuda.is_available(),
        )
    return DataLoader(
        dataset,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=shuffle,
        num_workers=int(cfg["training"].get("num_workers", 0)),
        collate_fn=lambda batch: collate_manifest(batch, cfg),
        pin_memory=torch.cuda.is_available(),
    )


def move_batch(batch, device):
    output = dict(batch)
    modalities = {}
    for modality, value in batch["modalities"].items():
        if isinstance(value, dict):
            modalities[modality] = {
                key: tensor.to(device) for key, tensor in value.items()
            }
        elif torch.is_tensor(value):
            modalities[modality] = value.to(device)
        else:
            modalities[modality] = value
    output["modalities"] = modalities
    output["modality_mask"] = batch["modality_mask"].to(device)
    output["labels"] = batch["labels"].to(device)
    output["label_mask"] = batch["label_mask"].to(device)
    return output


def dataset_label_statistics(dataset):
    statistics = {}
    for task_name in dataset.labels:
        spec = dataset.labels[task_name]
        labels = dataset.df[spec["column"]].fillna(0).to_numpy(dtype=np.float32)
        if spec.get("mask_column") in dataset.df.columns:
            mask = dataset.df[spec["mask_column"]].fillna(0).to_numpy() > 0.5
        else:
            mask = np.ones(len(dataset.df), dtype=bool)
        valid_labels = labels[mask]
        positive = int((valid_labels > 0.5).sum())
        valid = int(mask.sum())
        negative = valid - positive
        statistics[task_name] = {
            "valid": valid,
            "positive": positive,
            "negative": negative,
            "prevalence": float(positive / valid) if valid else float("nan"),
        }
    return statistics


def compute_train_pos_weight(train_loader, device):
    statistics = dataset_label_statistics(train_loader.dataset)
    weights = []
    for task_name in train_loader.dataset.labels:
        stat = statistics[task_name]
        weight = stat["negative"] / stat["positive"] if stat["positive"] else 1.0
        weights.append(weight)
    return torch.tensor(weights, dtype=torch.float32, device=device), statistics


def log_data_statistics(loaders, pos_weight):
    for split, loader in loaders.items():
        stats = dataset_label_statistics(loader.dataset)
        for index, (task_name, stat) in enumerate(stats.items()):
            fields = {
                "split": split,
                "task": task_name,
                **stat,
            }
            if split == "train":
                fields["pos_weight"] = float(pos_weight[index].detach().cpu())
            print(f"[data] {json.dumps(fields, sort_keys=True)}")


@torch.no_grad()
def collect_direct_calibration_data(model, loader, device):
    logits, labels, masks = [], [], []
    model.eval()
    for batch in loader:
        batch = move_batch(batch, device)
        encoded = model.encode(batch)
        logits.append(model.direct_logits(encoded, batch["modality_mask"]).cpu())
        labels.append(batch["labels"].cpu())
        masks.append(batch["label_mask"].cpu())
    return torch.cat(logits), torch.cat(labels), torch.cat(masks)


def fit_and_save_direct_temperature(model, val_loader, device, output_dir, cfg):
    if not bool(cfg.get("calibration", {}).get("temperature_scaling", False)):
        model.set_direct_temperature(1.0)
        return 1.0
    split = getattr(val_loader.dataset, "split", "val")
    logits, labels, masks = collect_direct_calibration_data(
        model, val_loader, device
    )
    temperature = fit_temperature(logits, labels, masks, split=split)
    model.set_direct_temperature(temperature)
    save_temperature(
        Path(output_dir) / "direct_temperature.json", temperature, fit_split=split
    )
    print(f"[calibration] fit_split={split} direct_temperature={temperature:.6f}")
    return temperature


def load_saved_direct_temperature(model, output_dir):
    temperature = load_temperature(Path(output_dir) / "direct_temperature.json")
    if temperature is not None:
        model.set_direct_temperature(temperature)
        print(f"[calibration] loaded direct_temperature={temperature:.6f}")
    return temperature


@torch.no_grad()
def collect_final_calibration_data(model, loader, device, stage="grpo"):
    logits, labels, masks = [], [], []
    model.eval()
    for batch in loader:
        batch = move_batch(batch, device)
        if stage in {"policy_warmup", "grpo"}:
            outputs = model.sample_rollout(batch, deterministic=True)
        else:
            outputs = supervised_round_robin(model, batch)
        logits.append(outputs["logits"].cpu())
        labels.append(batch["labels"].cpu())
        masks.append(batch["label_mask"].cpu())
    return torch.cat(logits), torch.cat(labels), torch.cat(masks)


def fit_and_save_final_temperature(model, val_loader, device, output_dir, stage="grpo"):
    split = getattr(val_loader.dataset, "split", "val")
    logits, labels, masks = collect_final_calibration_data(
        model, val_loader, device, stage=stage
    )
    temperature = fit_temperature(logits, labels, masks, split=split)
    model.set_final_temperature(temperature)
    save_temperature(
        Path(output_dir) / "final_temperature.json", temperature, fit_split=split
    )
    print(f"[calibration] fit_split={split} final_temperature={temperature:.6f}")
    return temperature


def load_saved_final_temperature(model, output_dir):
    temperature = load_temperature(Path(output_dir) / "final_temperature.json")
    if temperature is not None:
        model.set_final_temperature(temperature)
        print(f"[calibration] loaded final_temperature={temperature:.6f}")
    return temperature


def operator_regularization(model, operators, cfg):
    training_cfg = cfg["training"]
    sparse = torch.stack([operator.abs().mean() for operator in operators.values()]).mean()
    stable = torch.stack(
        [
            torch.relu(torch.linalg.matrix_norm(operator.float(), ord=2) - 1)
            .square()
            .mean()
            for operator in operators.values()
        ]
    ).mean()
    values = list(operators.values())
    diversity_terms = []
    for left in range(len(values)):
        for right in range(left + 1, len(values)):
            diversity_terms.append((values[left] - values[right]).square().mean())
    diversity = (
        -torch.stack(diversity_terms).mean()
        if diversity_terms
        else sparse.new_tensor(0.0)
    )
    return (
        float(training_cfg.get("operator_sparse_weight", 0.0)) * sparse
        + float(training_cfg.get("operator_stability_weight", 0.0)) * stable
        + float(training_cfg.get("operator_diversity_weight", 0.0)) * diversity
    )


def _alpha_anchor_weight(cfg, epoch):
    gate_cfg = cfg["training"].get("gate", {})
    base_weight = float(gate_cfg.get("anchor_weight", 0.0))
    decay_epochs = int(gate_cfg.get("anchor_decay_epochs", 0))
    if epoch is None or base_weight <= 0 or decay_epochs <= 0:
        return 0.0
    plateau_epochs = min(3, decay_epochs)
    if epoch < plateau_epochs:
        return base_weight
    if epoch >= decay_epochs - 1:
        return 0.0
    remaining = decay_epochs - 1 - epoch
    decay_span = max(decay_epochs - plateau_epochs, 1)
    return base_weight * remaining / decay_span


def _safe_correlation(left, right):
    if left.numel() < 2:
        return left.new_tensor(0.0)
    left_centered = left - left.mean()
    right_centered = right - right.mean()
    denominator = torch.sqrt(
        left_centered.square().sum() * right_centered.square().sum()
    ).clamp_min(1e-8)
    return (left_centered * right_centered).sum() / denominator


def residual_supervised_objective(
    outputs, batch, pos_weight, cfg, include_auxiliary=True,
    delta_weight_default=0.0, stage="supervised", epoch=None
):
    """Prediction and anti-collapse objectives for residual training."""
    loss_cfg = cfg["training"].get("loss", {})
    final_per_sample = per_sample_masked_bce(
        outputs["logits"], batch["labels"], batch["label_mask"], pos_weight
    )
    direct_per_sample = per_sample_masked_bce(
        outputs["direct_logits"], batch["labels"], batch["label_mask"], pos_weight
    )
    valid_samples = batch["label_mask"].sum(1) > 0
    hard_gamma = float(loss_cfg.get("hard_case_gamma", 0.0))
    v5 = bool(cfg.get("model", {}).get("performance_v5", {}).get("enabled", False))
    if v5 and stage == "operator_warmup" and valid_samples.any():
        threshold = torch.quantile(direct_per_sample[valid_samples].detach(), 0.60)
        weights = torch.where(
            direct_per_sample.detach() >= threshold,
            direct_per_sample.new_tensor(1.75),
            direct_per_sample.new_tensor(1.0),
        )
    else:
        weights = hard_case_weights(
            direct_per_sample,
            gamma=hard_gamma,
            clip=float(loss_cfg.get("hard_case_clip", 2.0)),
            valid_samples=valid_samples,
        )
    if hard_gamma == 0.0 and not (v5 and stage == "operator_warmup"):
        prediction = final_per_sample.mean()
    elif valid_samples.any():
        prediction = (
            final_per_sample[valid_samples] * weights[valid_samples]
        ).sum() / weights[valid_samples].sum().clamp_min(1e-8)
    else:
        prediction = final_per_sample.sum() * 0.0
    if bool(loss_cfg.get("noharm_detach_alpha", False)):
        no_harm = alpha_detached_no_harm_loss(
            outputs["direct_logits"],
            outputs["operator_delta_logits"],
            outputs["operator_gate"],
            batch["labels"],
            batch["label_mask"],
            pos_weight,
        )
    else:
        no_harm = no_harm_loss_from_per_sample(
            final_per_sample, direct_per_sample, valid_samples
        )
    ranking = pairwise_ranking_loss(
        outputs["logits"],
        batch["labels"],
        batch["label_mask"],
        max_pairs=int(loss_cfg.get("ranking_max_pairs", 1024)),
    )
    auc_rank = auc_ranking_loss(
        outputs["logits"],
        batch["labels"],
        batch["label_mask"],
        margin=float(loss_cfg.get("margin_auc", 0.5)),
        max_pairs=int(loss_cfg.get("ranking_max_pairs", 1024)),
    )
    pr_rank = precision_ranking_loss(
        outputs["logits"],
        batch["labels"],
        batch["label_mask"],
        margin=float(loss_cfg.get("margin_pr", 0.75)),
        max_pairs=int(loss_cfg.get("ranking_max_pairs", 1024)),
    )
    delta_l2 = outputs["operator_delta_logits"].square().mean()
    no_harm_weight = float(
        loss_cfg.get("noharm_weight", loss_cfg.get("no_harm_weight", 0.0))
    )
    ranking_weight = float(loss_cfg.get("ranking_weight", 0.0))
    configured_delta_default = (
        cfg["training"].get("operator_delta_l2", delta_weight_default)
        if loss_cfg else delta_weight_default
    )
    delta_weight = float(
        loss_cfg.get("delta_l2_weight", configured_delta_default)
    )
    utility_rank = final_per_sample.sum() * 0.0
    improvement = direct_per_sample.detach() - final_per_sample.detach()
    utility_rank_weight = (
        float(loss_cfg.get("utility_rank_weight", 0.0))
        if stage == "operator_warmup" else 0.0
    )
    if utility_rank_weight > 0:
        utility_rank, improvement = residual_utility_ranking_loss(
            final_per_sample,
            direct_per_sample,
            weights,
            valid_samples,
        )

    utility, utility_targets = gate_utility_targets(
        outputs["direct_logits"],
        outputs["operator_delta_logits"],
        batch["labels"],
        batch["label_mask"],
        pos_weight,
        alpha_probe=float(loss_cfg.get("alpha_probe", 0.15)),
        utility_temperature=float(loss_cfg.get("utility_temperature", 0.10)),
    )
    residual_cfg = cfg.get("model", {}).get("residual", {})
    alpha_min = float(residual_cfg.get("alpha_min", 0.0))
    alpha_max = float(residual_cfg.get("alpha_max", 1.0))
    gate_loss = outputs["operator_gate"].sum() * 0.0
    gate_probability = outputs["operator_gate"].new_zeros(
        outputs["operator_gate"].size(0)
    )
    gate_weight = (
        float(loss_cfg.get("gate_supervision_weight", 0.0))
        if stage == "supervised" else 0.0
    )
    if gate_weight > 0 and alpha_max > alpha_min:
        span = max(alpha_max - alpha_min, 1e-6)
        gate_probability = (
            (outputs["operator_gate"].squeeze(-1) - alpha_min) / span
        ).clamp(1e-6, 1 - 1e-6)
        if v5 and outputs.get("utility_normalizer") is not None:
            normalized_utility = outputs["utility_normalizer"](
                utility, update=True
            )
            gate_loss = normalized_gate_supervision_loss(
                gate_probability[valid_samples], normalized_utility[valid_samples]
            )
        else:
            gate_loss, gate_probability = gate_supervision_loss(
                outputs["operator_gate"],
                utility_targets,
                alpha_min,
                alpha_max,
                valid_samples,
            )
    anchor_target = float(
        cfg["training"].get("gate", {}).get("alpha_anchor", 0.08)
    )
    alpha_anchor = (outputs["operator_gate"].mean() - anchor_target).square()
    anchor_weight = (
        _alpha_anchor_weight(cfg, epoch) if stage == "supervised" else 0.0
    )
    total = prediction
    if include_auxiliary:
        total = (
            total
            + no_harm_weight * no_harm
            + ranking_weight * ranking
            + float(loss_cfg.get("auc_rank_weight", 0.0)) * auc_rank
            + float(loss_cfg.get("pr_rank_weight", 0.0)) * pr_rank
            + delta_weight * delta_l2
            + utility_rank_weight * utility_rank
            + gate_weight * gate_loss
            + anchor_weight * alpha_anchor
        )
    scaled_delta = outputs["operator_gate"] * outputs["operator_delta_logits"]
    direct_scale = outputs["direct_logits"].abs().mean().clamp_min(1e-8)
    diagnostics = {
        "total": total,
        "prediction": prediction,
        "no_harm": no_harm,
        "ranking": ranking,
        "auc_rank": auc_rank,
        "pr_rank": pr_rank,
        "delta_l2": delta_l2,
        "utility_rank": utility_rank,
        "gate_supervision": gate_loss,
        "alpha_anchor": alpha_anchor,
        "alpha_anchor_weight": outputs["operator_gate"].new_tensor(anchor_weight),
        "hard_weight_mean": weights[valid_samples].mean() if valid_samples.any() else weights.mean(),
        "hard_weight_max": weights[valid_samples].max() if valid_samples.any() else weights.max(),
        "mean_abs_delta": outputs["operator_delta_logits"].abs().mean(),
        "std_delta": outputs["operator_delta_logits"].std(unbiased=False),
        "mean_abs_scaled_delta": scaled_delta.abs().mean(),
        "relative_scaled_delta": scaled_delta.abs().mean() / direct_scale,
        "mean_utility": utility[valid_samples].mean() if valid_samples.any() else utility.mean(),
        "positive_utility_fraction": (utility[valid_samples] > 0).float().mean() if valid_samples.any() else utility.new_tensor(0.0),
        "mean_gate_probability": gate_probability.mean() if gate_probability.numel() else utility.new_tensor(0.0),
        "gate_utility_correlation": _safe_correlation(
            outputs["operator_gate"][valid_samples].squeeze(-1),
            utility[valid_samples],
        ) if valid_samples.any() else utility.new_tensor(0.0),
        "mean_alpha_utility_positive": outputs["operator_gate"][utility > 0].mean() if (utility > 0).any() else utility.new_tensor(float("nan")),
        "mean_alpha_utility_negative": outputs["operator_gate"][utility <= 0].mean() if (utility <= 0).any() else utility.new_tensor(float("nan")),
        "mean_improvement": improvement[valid_samples].mean() if valid_samples.any() else improvement.mean(),
    }
    return total, diagnostics


def _mean_diagnostics(entries):
    if not entries:
        return {}
    output = {}
    for key in entries[0]:
        values = [entry[key] for entry in entries]
        finite = [
            float(value) for value in values
            if value is not None and np.isfinite(float(value))
        ]
        output[key] = float(np.mean(finite)) if finite else None
    return output


def supervised_round_robin(model, batch, fixed_alpha=None):
    encoded = model.encode(batch)
    operators = model.build_operators(encoded, batch["modality_mask"])
    batch_size = batch["labels"].size(0)
    state = model.initial(batch_size, batch["labels"].device)
    for step in range(model.max_steps):
        modality_index = step % model.M
        modality = model.modalities[modality_index]
        concept = encoded["conf"][modality].argmax(1)
        action = modality_index * model.K + concept
        action = action.clone()
        action[batch["modality_mask"][:, modality_index].eq(0)] = model.stop_idx
        state, _ = model.apply_action(state, encoded, operators, action)
    outputs = model.prediction_outputs(
        state,
        encoded,
        batch["modality_mask"],
        operators=operators,
        fixed_alpha=fixed_alpha,
        reason_mask=(
            torch.ones(batch_size, device=batch["labels"].device)
            if getattr(model, "performance_v5", False) and fixed_alpha is not None
            else None
        ),
    )
    comm_mean, comm_max = model._commutator_summary(
        operators, state, batch["modality_mask"]
    )
    outputs.update({
        "state": state,
        "encoded": encoded,
        "operators": operators,
        "mean_commutator": comm_mean,
        "max_commutator": comm_max,
    })
    return outputs


def _is_v6(model):
    return bool(getattr(model, "performance_v6", False))


def _load_torch_checkpoint(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _strong_direct_prefix(name):
    return name.startswith(
        (
            "encoders.",
            "direct_projections.",
            "direct_fusion.",
            "direct_head.",
            "direct_attention_enhancer.",
        )
    ) or name == "direct_temperature"


def _anchor_metric(metrics, task_name, metric):
    for key in (
        f"strong_direct_{metric}_{task_name}",
        f"base_{metric}_{task_name}",
        f"direct_{metric}_{task_name}",
        f"{metric}_{task_name}",
    ):
        value = metrics.get(key)
        if value is not None and np.isfinite(value):
            return float(value)
    return float("-inf")


def discover_strong_direct_checkpoint(cfg, output_dir, model=None):
    configured = cfg.get("training", {}).get("strong_direct_init_checkpoint")
    if configured and str(configured).lower() not in {"auto", "none"}:
        path = Path(configured).expanduser()
        if not path.exists():
            raise FileNotFoundError(
                f"Configured strong direct checkpoint not found: {path}"
            )
        return path
    root = Path(cfg["paths"]["output_root"]).expanduser()
    candidates = set()
    for pattern in (
        "**/best_direct.pt",
        "**/best_direct_v5.pt",
        "**/best_supervised.pt",
        "**/best.pt",
    ):
        candidates.update(root.glob(pattern))
    candidates = [
        path for path in candidates
        if path.is_file() and Path(output_dir) not in path.parents
    ]
    if not candidates:
        return None
    task_name = cfg["experiment"]["task_names"][0]
    ranked = []
    model_state = model.state_dict() if model is not None else None
    for path in candidates:
        try:
            state = _load_torch_checkpoint(path, "cpu")
            metrics = state.get("metrics", {})
            checkpoint_model = state.get("model", state)
            compatible = sum(
                1
                for raw_name, value in checkpoint_model.items()
                for name in [raw_name.removeprefix("module.")]
                if _strong_direct_prefix(name)
                and torch.is_tensor(value)
                and (
                    model_state is None
                    or (
                        name in model_state
                        and model_state[name].shape == value.shape
                    )
                )
            )
            ranked.append(
                (
                    _anchor_metric(metrics, task_name, "auroc"),
                    _anchor_metric(metrics, task_name, "auprc"),
                    compatible,
                    path,
                )
            )
        except (OSError, RuntimeError, ValueError, TypeError):
            continue
    return max(ranked, default=(None, None, None, None))[-1]


@torch.no_grad()
def initialize_strong_direct_anchor(
    model, val_loader, cfg, device, output_dir
):
    """Load only the known direct anchor and validate it before formal training."""
    checkpoint_path = discover_strong_direct_checkpoint(
        cfg, output_dir, model=model
    )
    required = bool(
        cfg.get("training", {}).get(
            "require_strong_direct_init", _is_v6(model)
        )
    )
    if checkpoint_path is None:
        message = "[direct-anchor-warning] no strong direct checkpoint found"
        print(message)
        if required:
            raise FileNotFoundError(
                message
                + "; set training.strong_direct_init_checkpoint to a known checkpoint"
            )
        return None
    state = _load_torch_checkpoint(checkpoint_path, device)
    source = state.get("model", state)
    current = model.state_dict()
    loaded, unexpected = [], []
    for raw_name, value in source.items():
        name = raw_name.removeprefix("module.")
        if not _strong_direct_prefix(name):
            continue
        if name in current and current[name].shape == value.shape:
            current[name] = value.to(
                device=current[name].device, dtype=current[name].dtype
            )
            loaded.append(name)
        else:
            unexpected.append(raw_name)
    model.load_state_dict(current, strict=True)
    anchor_names = [name for name in current if _strong_direct_prefix(name)]
    missing = sorted(set(anchor_names) - set(loaded))
    previous_fallback = bool(model.use_strong_direct_fallback.item())
    model.set_strong_direct_fallback(True)
    metrics = evaluate_model(
        model,
        val_loader,
        cfg,
        device,
        deterministic=True,
        split="val",
        stage="direct",
    )
    model.set_strong_direct_fallback(previous_fallback)
    task_name = cfg["experiment"]["task_names"][0]
    auroc = float(metrics[f"direct_auroc_{task_name}"])
    auprc = float(metrics[f"direct_auprc_{task_name}"])
    print(
        "[strong-direct-init] "
        f"path={checkpoint_path} loaded_keys={len(loaded)} "
        f"missing_keys={missing} unexpected_keys={unexpected} "
        f"val_auroc={auroc:.6f} val_auprc={auprc:.6f}"
    )
    report = {
        "checkpoint": str(checkpoint_path),
        "loaded_keys": sorted(loaded),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "val_auroc": auroc,
        "val_auprc": auprc,
    }
    (Path(output_dir) / "strong_direct_init_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    if auroc < 0.83:
        print("[direct-anchor-warning] strong direct warm start failed")
    if auroc < 0.80 and required:
        raise RuntimeError(
            "[direct-anchor-warning] strong direct warm start failed: "
            f"validation AUROC {auroc:.6f} < 0.80"
        )
    return metrics


def _direct_candidate_is_pareto_safe(metrics, cfg):
    task_name = cfg["experiment"]["task_names"][0]
    selection = cfg.get("selection", {})
    if bool(cfg.get("model", {}).get("performance_v6", {}).get("enabled", False)):
        reason_rate = metrics.get("reason_rate")
        if reason_rate is not None and not 0.10 <= float(reason_rate) <= 0.90:
            print(
                "[reason-rate-guard] checkpoint rejected: "
                f"reason_rate={float(reason_rate):.6f}"
            )
            return False
    return checkpoint_passes_direct_guard(
        metrics[f"direct_auroc_{task_name}"],
        metrics[f"direct_auprc_{task_name}"],
        metrics[f"strong_direct_auroc_{task_name}"],
        metrics[f"strong_direct_auprc_{task_name}"],
        float(selection.get("tolerance_auroc", 0.001)),
        float(selection.get("tolerance_auprc", 0.002)),
    )


def _set_module_trainable(module, enabled):
    if module is None:
        return
    for parameter in module.parameters():
        parameter.requires_grad = enabled


def _set_operator_trainable(model, enabled):
    for module in [
        model.projectors,
        model.op_generators,
        model.updaters,
        model.gates,
        model.context_proj,
        model.outcome_head,
        model.state_norm,
        model.operator_hidden_norm,
        model.operator_eps_logits,
        model.residual_correction,
    ]:
        _set_module_trainable(module, enabled)
    for parameter in [
        model.initial_state,
        model.operator_gate_logit,
        model.state_gate_logit,
    ]:
        parameter.requires_grad = enabled


def _set_ehr_last_layers_trainable(model, cfg, enabled=True):
    if "ehr" not in model.encoders:
        return
    encoder = model.encoders["ehr"]
    layer_count = int(
        cfg.get("training", {}).get("direct_phase", {}).get("ehr_last_layers", 2)
    )
    if hasattr(encoder, "transformer") and hasattr(encoder.transformer, "layers"):
        for layer in list(encoder.transformer.layers)[-layer_count:]:
            _set_module_trainable(layer, enabled)
        _set_module_trainable(getattr(encoder, "output_norm", None), enabled)
    elif hasattr(encoder, "lstm"):
        highest = max(
            int(cfg["model"]["encoders"]["ehr"].get("layers", 1)) - 1, 0
        )
        for name, parameter in encoder.named_parameters():
            if name.endswith(f"_l{highest}"):
                parameter.requires_grad = enabled
    else:
        parameters = list(encoder.parameters())
        for parameter in parameters[-max(layer_count, 1):]:
            parameter.requires_grad = enabled


def configure_trainable_parameters(model, cfg, stage, epoch=None):
    for parameter in model.parameters():
        parameter.requires_grad = False

    v5 = bool(getattr(model, "performance_v5", False))
    v6 = _is_v6(model)

    if stage == "direct" and v5:
        _set_module_trainable(model.pairwise_residual, True)
        _set_module_trainable(model.pooling_residual, True)
        freeze_epochs = int(
            cfg.get("training", {}).get("direct_phase", {}).get(
                "freeze_base_epochs", 5
            )
        )
        if epoch is not None and int(epoch) >= freeze_epochs:
            _set_module_trainable(model.direct_head, True)
            _set_ehr_last_layers_trainable(model, cfg, True)

    if stage in {"direct", "supervised"} and model.direct_enabled and not v5:
        _set_module_trainable(model.direct_projections, True)
        _set_module_trainable(model.direct_fusion, True)
        _set_module_trainable(model.direct_head, True)
        _set_module_trainable(model.direct_attention_enhancer, True)
        for modality, encoder in model.encoders.items():
            spec = cfg["model"]["encoders"][modality]
            if not bool(spec.get("freeze", False)):
                _set_module_trainable(encoder, True)

    if stage in {"operator_warmup", "supervised", "grpo"} and not (
        v5 and stage == "grpo"
    ):
        _set_operator_trainable(model, True)

    if stage in {"supervised", "grpo"} and not (v5 and stage == "grpo"):
        _set_module_trainable(model.patient_operator_gate, True)
    else:
        _set_module_trainable(model.patient_operator_gate, False)

    if stage in {"policy_warmup", "grpo"}:
        _set_module_trainable(model.policy, True)
    if v5 and stage in {"supervised", "policy_warmup", "grpo"}:
        _set_module_trainable(model.reason_gate, True)
    if v5 and stage == "supervised":
        _set_module_trainable(model.policy, True)
        if v6:
            _set_module_trainable(model.reranker, True)
    if v6 and stage == "policy_warmup":
        _set_module_trainable(model.reason_gate, epoch is not None and int(epoch) >= 7)
        _set_module_trainable(model.reranker, epoch is not None and int(epoch) >= 4)

    if stage == "grpo":
        if v5:
            # v6 GRPO is a small, KL-anchored routing refinement.
            _set_module_trainable(model.encoders, False)
            _set_operator_trainable(model, False)
            _set_module_trainable(model.patient_operator_gate, False)
            _set_module_trainable(model.pairwise_residual, False)
            _set_module_trainable(model.pooling_residual, False)
            _set_module_trainable(model.direct_projections, False)
            _set_module_trainable(model.direct_fusion, False)
            _set_module_trainable(model.direct_head, False)
            if v6:
                _set_module_trainable(model.policy, False)
                _set_module_trainable(model.reason_gate, False)
                _set_module_trainable(model.reranker, False)
                _set_module_trainable(model.policy.net[-1], True)
                _set_module_trainable(model.reason_gate.network[-1], True)
                _set_module_trainable(model.reranker.last_layer, True)
            else:
                _set_module_trainable(model.reason_gate, True)
                _set_module_trainable(model.policy, True)
            return
        if bool(cfg["training"].get("freeze_encoders_during_grpo", True)):
            _set_module_trainable(model.encoders, False)
        else:
            for modality, encoder in model.encoders.items():
                spec = cfg["model"]["encoders"][modality]
                if not bool(spec.get("freeze", False)):
                    _set_module_trainable(encoder, True)
        direct_lr = float(cfg["training"].get("lr_direct_grpo", 0.0))
        if direct_lr > 0 and model.direct_enabled:
            _set_module_trainable(model.direct_projections, True)
            _set_module_trainable(model.direct_fusion, True)
            _set_module_trainable(model.direct_head, True)
            _set_module_trainable(model.direct_attention_enhancer, True)

    if stage == "supervised" and not model.direct_enabled:
        _set_operator_trainable(model, True)


def _is_native_encoder_parameter(name, cfg):
    if not name.startswith("encoders."):
        return False
    parts = name.split(".")
    if len(parts) < 2:
        return False
    modality = parts[1]
    backend = cfg["model"]["encoders"].get(modality, {}).get("backend", "")
    return backend in {"native_lstm", "native_transformer", "medfuse_ehr"}


def make_optimizer(model, cfg, stage):
    training_cfg = cfg["training"]
    legacy_lr = float(training_cfg.get("lr", 1e-4))
    buckets = {}
    v6 = _is_v6(model)
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if stage == "grpo" and v6:
            key = (
                "grpo_routing",
                float(cfg.get("grpo", {}).get("lr_policy", 3e-6)),
            )
        elif name.startswith("reranker."):
            key = (
                "reranker",
                float(training_cfg.get("lr_reranker", training_cfg.get("lr_policy", 5e-5))),
            )
        elif name.startswith("pairwise_residual."):
            key = ("pairwise", float(training_cfg.get("lr_pairwise", 3e-4)))
        elif name.startswith("pooling_residual."):
            key = ("pool", float(training_cfg.get("lr_pool", 3e-4)))
        elif name.startswith("direct_head."):
            key = ("direct_head", float(training_cfg.get("lr_direct_head", 2e-5)))
        elif name.startswith("reason_gate."):
            key = (
                "reason_gate",
                float(
                    training_cfg.get(
                        "lr_reason_gate", training_cfg.get("lr_policy", 5e-5)
                    )
                ),
            )
        elif stage == "policy_warmup" or name.startswith("policy."):
            key = ("policy", float(training_cfg.get("lr_policy", training_cfg.get("policy_lr", legacy_lr))))
        elif stage == "grpo":
            if name.startswith("direct_"):
                key = ("direct_grpo", float(training_cfg.get("lr_direct_grpo", 0.0)))
            else:
                key = ("ncore_grpo", float(training_cfg.get("lr_ncore_grpo", legacy_lr)))
        elif _is_native_encoder_parameter(name, cfg):
            key = (
                "ehr",
                float(
                    training_cfg.get(
                        "lr_ehr_last_layers",
                        training_cfg.get("lr_ehr", training_cfg.get("lr_direct", legacy_lr)),
                    )
                ),
            )
        elif name.startswith("direct_"):
            key = ("direct", float(training_cfg.get("lr_direct", legacy_lr)))
        else:
            key = ("ncore", float(training_cfg.get("lr_ncore", legacy_lr)))
        if key[1] > 0:
            buckets.setdefault(key, []).append(parameter)
    groups = [
        {"params": parameters, "lr": lr, "group_name": group_name}
        for (group_name, lr), parameters in buckets.items()
    ]
    if not groups:
        raise ValueError(f"No trainable parameters configured for stage '{stage}'")
    return torch.optim.AdamW(
        groups, weight_decay=float(training_cfg.get("weight_decay", 0.0))
    )


def _save_checkpoint(path, model, optimizer, cfg, stage, epoch, metrics,
                     pos_weight, **extra_state):
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "cfg": cfg,
        "stage": stage,
        "epoch": epoch,
        "metrics": metrics,
        "pos_weight": pos_weight.detach().cpu(),
    }
    payload.update(extra_state)
    torch.save(payload, path)


def _parameters_finite(model):
    return all(
        torch.isfinite(parameter.detach()).all().item()
        for parameter in model.parameters()
    )


def _metrics_finite(metrics):
    for value in metrics.values():
        if value is None or isinstance(value, (str, bool, int)):
            continue
        if isinstance(value, (float, np.floating)) and not np.isfinite(value):
            return False
    return True


def _save_last_finite(
    output_dir, model, optimizer, cfg, stage, epoch, metrics, pos_weight,
    *, loss_is_finite=True, **extra_state
):
    if loss_is_finite and _parameters_finite(model) and _metrics_finite(metrics):
        _save_checkpoint(
            Path(output_dir) / "last_finite.pt",
            model,
            optimizer,
            cfg,
            stage,
            epoch,
            metrics,
            pos_weight,
            **extra_state,
        )
        return True
    print(
        f"[last-finite-skip] stage={stage} epoch={epoch} "
        f"loss_finite={loss_is_finite} "
        f"params_finite={_parameters_finite(model)} "
        f"metrics_finite={_metrics_finite(metrics)}"
    )
    return False


def _resume_optimizer(optimizer, resume_state, stage):
    if not resume_state or resume_state.get("stage") != stage:
        return 0
    if "optimizer" in resume_state:
        optimizer.load_state_dict(resume_state["optimizer"])
    return int(resume_state.get("epoch", -1)) + 1


def _primary_score(metrics, cfg):
    if bool(cfg.get("model", {}).get("performance_v5", {}).get("enabled", False)):
        task_name = cfg["experiment"]["task_names"][0]
        prefix = (
            "direct" if metrics.get("evaluation_stage") == "direct" else "final"
        )
        auroc = metrics.get(f"{prefix}_auroc_{task_name}")
        auprc = metrics.get(f"{prefix}_auprc_{task_name}")
        prevalence = metrics.get(f"prevalence_{task_name}")
        if auroc is not None and auprc is not None and prevalence is not None:
            return dual_metric_score(auroc, auprc, prevalence)
    requested = cfg.get("experiment", {}).get("primary_metric")
    if requested in metrics:
        return float(metrics[requested])
    task_name = cfg["experiment"]["task_names"][0]
    for candidate in [f"final_auprc_{task_name}", f"auprc_{task_name}"]:
        if candidate in metrics:
            return float(metrics[candidate])
    return float("-inf")


def _should_save_best(score, best, epoch, start_epoch):
    if epoch == start_epoch:
        return True
    return np.isfinite(score) and (not np.isfinite(best) or score > best)


def _passes_v5_direct_guard(metrics, cfg, output_dir):
    """Apply the validation Direct guard without changing legacy selection."""
    if not bool(
        cfg.get("model", {}).get("performance_v5", {}).get("enabled", False)
    ):
        return True
    direct_path = Path(output_dir) / "best_direct.pt"
    if not direct_path.exists():
        return False
    try:
        direct_state = torch.load(
            direct_path, map_location="cpu", weights_only=False
        )
    except TypeError:
        direct_state = torch.load(direct_path, map_location="cpu")
    task_name = cfg["experiment"]["task_names"][0]
    direct_metrics = direct_state.get("metrics", {})
    final_auroc = metrics.get(f"final_auroc_{task_name}")
    final_auprc = metrics.get(f"final_auprc_{task_name}")
    direct_auroc = direct_metrics.get(f"direct_auroc_{task_name}")
    direct_auprc = direct_metrics.get(f"direct_auprc_{task_name}")
    if any(
        value is None
        for value in [final_auroc, final_auprc, direct_auroc, direct_auprc]
    ):
        return False
    selection = cfg.get("selection", {})
    return checkpoint_passes_direct_guard(
        final_auroc,
        final_auprc,
        direct_auroc,
        direct_auprc,
        float(selection.get("tolerance_auroc", 0.001)),
        float(selection.get("tolerance_auprc", 0.002)),
    )


def _print_best_validation(stage, metrics, cfg, prefix="final"):
    if not bool(
        cfg.get("model", {}).get("performance_v5", {}).get("enabled", False)
    ):
        return
    task_name = cfg["experiment"]["task_names"][0]
    print(
        f"[best-validation] stage={stage} "
        f"auroc={metrics[f'{prefix}_auroc_{task_name}']:.6f} "
        f"auprc={metrics[f'{prefix}_auprc_{task_name}']:.6f} "
        f"dual_score={metrics[f'{prefix}_dual_score_{task_name}']:.6f}"
    )


def _gradient_clip_config(cfg):
    value = cfg["training"].get("grad_clip", 5.0)
    if isinstance(value, dict):
        return bool(value.get("enabled", True)), float(value.get("max_norm", 1.0))
    return True, float(value)


def _clip_and_step(
    loss,
    optimizer,
    parameters,
    cfg,
    *,
    model=None,
    stage=None,
    epoch=None,
    batch_idx=None,
    nonfinite_state=None,
):
    """Backpropagate safely, skipping isolated non-finite gradient batches."""
    state = nonfinite_state if nonfinite_state is not None else {}
    state.setdefault("consecutive", 0)
    state.setdefault("skipped", 0)
    state.setdefault("steps", 0)
    optimizer.zero_grad(set_to_none=True)
    if not torch.isfinite(loss).all():
        assert_finite_tensor(
            "loss", loss, stage=stage, epoch=epoch, batch_idx=batch_idx,
            raise_on_nonfinite=False,
        )
        state["consecutive"] += 1
        state["skipped"] += 1
        if state["consecutive"] > int(
            cfg["training"].get("max_consecutive_nonfinite_batches", 3)
        ):
            raise FloatingPointError("Too many consecutive non-finite batches")
        return False
    loss.backward()
    parameter_list = list(parameters)
    names = (
        {id(parameter): name for name, parameter in model.named_parameters()}
        if model is not None else {}
    )
    bad = []
    for index, parameter in enumerate(parameter_list):
        if parameter.grad is None:
            continue
        if not torch.isfinite(parameter.grad).all():
            name = names.get(id(parameter), f"parameter_{index}")
            print(f"[nonfinite-gradient] parameter={name}")
            assert_finite_tensor(
                f"gradient.{name}",
                parameter.grad,
                stage=stage,
                epoch=epoch,
                batch_idx=batch_idx,
                raise_on_nonfinite=False,
            )
            bad.append(name)
            break
    if bad:
        optimizer.zero_grad(set_to_none=True)
        state["consecutive"] += 1
        state["skipped"] += 1
        if state["consecutive"] > int(
            cfg["training"].get("max_consecutive_nonfinite_batches", 3)
        ):
            raise FloatingPointError(
                f"Too many consecutive non-finite gradient batches; first={bad[0]}"
            )
        return False
    enabled, max_norm = _gradient_clip_config(cfg)
    if enabled:
        total_norm = torch.nn.utils.clip_grad_norm_(parameter_list, max_norm)
        post_clip_finite = torch.isfinite(total_norm).all() and all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in parameter_list
        )
        if not post_clip_finite:
            assert_finite_tensor(
                "gradient.total_norm_after_clip",
                torch.as_tensor(total_norm),
                stage=stage,
                epoch=epoch,
                batch_idx=batch_idx,
                raise_on_nonfinite=False,
            )
            optimizer.zero_grad(set_to_none=True)
            state["consecutive"] += 1
            state["skipped"] += 1
            if state["consecutive"] > int(
                cfg["training"].get("max_consecutive_nonfinite_batches", 3)
            ):
                raise FloatingPointError(
                    "Too many consecutive non-finite post-clip gradient batches"
                )
            return False
    optimizer.step()
    state["consecutive"] = 0
    state["steps"] += 1
    interval = int(cfg.get("debug", {}).get("parameter_finite_check_interval", 0))
    if model is not None and interval > 0 and state["steps"] % interval == 0:
        guarded_prefixes = (
            "op_generators.", "updaters.", "gates.", "outcome_head.",
            "residual_correction.", "patient_operator_gate.", "policy.",
        )
        for name, parameter in model.named_parameters():
            if parameter.requires_grad and name.startswith(guarded_prefixes):
                assert_finite_tensor(
                    f"parameter.{name}", parameter,
                    stage=stage, epoch=epoch, batch_idx=batch_idx,
                    raise_on_nonfinite=True,
                )
    return True


def train_direct(model, train_loader, val_loader, cfg, device, output_dir,
                 pos_weight, resume_state=None):
    v5 = bool(getattr(model, "performance_v5", False))
    start_epoch = (
        int(resume_state.get("epoch", -1)) + 1
        if resume_state and resume_state.get("stage") == "direct" else 0
    )
    anchor_metrics = None
    if v5 and _is_v6(model) and start_epoch == 0:
        anchor_metrics = initialize_strong_direct_anchor(
            model, val_loader, cfg, device, output_dir
        )
    configure_trainable_parameters(model, cfg, "direct", epoch=start_epoch)
    optimizer = make_optimizer(model, cfg, "direct")
    if resume_state and resume_state.get("stage") == "direct" and "optimizer" in resume_state:
        try:
            optimizer.load_state_dict(resume_state["optimizer"])
        except (ValueError, KeyError):
            print("[resume-warning] direct phase changed; optimizer state restarted")
    parameters = [p for p in model.parameters() if p.requires_grad]
    best = raw_best = ema_best = swa_best = float("-inf")
    if anchor_metrics is not None:
        best = _primary_score(anchor_metrics, cfg)
        model.set_strong_direct_fallback(True)
        for name in ("best_strong_direct.pt", "best_direct.pt", "best_direct_v6.pt"):
            _save_checkpoint(
                output_dir / name,
                model,
                optimizer,
                cfg,
                "direct",
                -1,
                anchor_metrics,
                pos_weight,
                selected_variant="strong_direct_fallback",
            )
        model.set_strong_direct_fallback(False)
    nonfinite_state = {}
    epochs = int(cfg["training"].get("epochs_direct", cfg["training"].get("epochs_supervised", 1)))
    ema_cfg = cfg["training"].get("ema", {})
    ema = (
        ModelEMA(
            model,
            decay=float(ema_cfg.get("decay", 0.999)),
            include_frozen=v5,
        )
        if bool(ema_cfg.get("enabled", False)) else None
    )
    swa_cfg = cfg["training"].get("swa", {})
    swa = ModelSWA(
        model,
        parameter_prefixes=(
            "encoders.ehr.",
            "direct_head.",
            "pairwise_residual.",
            "pooling_residual.",
        ),
    ) if v5 and bool(swa_cfg.get("enabled", False)) else None
    if resume_state and ema is not None and "ema" in resume_state:
        ema.load_state_dict(resume_state["ema"])
    if resume_state and swa is not None and "swa" in resume_state:
        swa.load_state_dict(resume_state["swa"])
    freeze_epochs = int(
        cfg["training"].get("direct_phase", {}).get("freeze_base_epochs", 5)
    )
    swa_start = max(0, epochs - int(swa_cfg.get("last_epochs", 8)))
    for epoch in range(start_epoch, epochs):
        if v5 and epoch == freeze_epochs and start_epoch < freeze_epochs:
            configure_trainable_parameters(model, cfg, "direct", epoch=epoch)
            optimizer = make_optimizer(model, cfg, "direct")
            parameters = [p for p in model.parameters() if p.requires_grad]
        model.train()
        losses, gates, component_rows = [], [], []
        for batch_idx, batch in enumerate(train_loader):
            batch = move_batch(batch, device)
            batch = apply_modality_dropout(batch, cfg)
            model.set_numerics_context("direct", epoch, batch_idx)
            encoded = model.encode(batch)
            logits = model.direct_logits(encoded, batch["modality_mask"])
            if v5:
                loss, components = direct_performance_objective(
                    logits,
                    batch["labels"],
                    batch["label_mask"],
                    pos_weight,
                    cfg,
                )
            else:
                loss = per_sample_masked_bce(
                    logits, batch["labels"], batch["label_mask"], pos_weight
                ).mean()
                components = {"bce": loss}
            if not _clip_and_step(
                loss, optimizer, parameters, cfg, model=model,
                stage="direct", epoch=epoch, batch_idx=batch_idx,
                nonfinite_state=nonfinite_state,
            ):
                continue
            if ema is not None:
                ema.update(model)
            losses.append(loss.item())
            gates.append(float(model.direct_attention_gate_value().detach().cpu()))
            component_rows.append(
                {key: float(value.detach().cpu()) for key, value in components.items()}
            )
        if swa is not None and epoch >= swa_start:
            swa.update(model)
        raw_metrics = evaluate_model(
            model, val_loader, cfg, device, deterministic=True,
            split="val", pos_weight=pos_weight, stage="direct", epoch=epoch
        )
        raw_score = _primary_score(raw_metrics, cfg)
        ema_metrics, swa_metrics = None, None
        ema_score = swa_score = float("-inf")
        if ema is not None:
            with ema.average_parameters(model):
                ema_metrics = evaluate_model(
                    model, val_loader, cfg, device, deterministic=True,
                    split="val", pos_weight=pos_weight, stage="direct", epoch=epoch
                )
            ema_score = _primary_score(ema_metrics, cfg)
        if swa is not None and swa.count > 0:
            with swa.average_parameters(model):
                swa_metrics = evaluate_model(
                    model, val_loader, cfg, device, deterministic=True,
                    split="val", pos_weight=pos_weight, stage="direct", epoch=epoch
                )
            swa_score = _primary_score(swa_metrics, cfg)
        variants = [("raw", raw_score, raw_metrics)]
        if ema_metrics is not None:
            variants.append(("ema", ema_score, ema_metrics))
        if swa_metrics is not None:
            variants.append(("swa", swa_score, swa_metrics))
        safe_variants = [
            row for row in variants
            if (not _is_v6(model)) or _direct_candidate_is_pareto_safe(row[2], cfg)
        ]
        safe_to_save = bool(safe_variants)
        winner, score, metrics = max(
            safe_variants or variants, key=lambda row: row[1]
        )
        if _is_v6(model) and not safe_to_save:
            task_name = cfg["experiment"]["task_names"][0]
            print(
                "[direct-anchor-warning] full direct rejected; "
                f"strong_auroc={metrics[f'strong_direct_auroc_{task_name}']:.6f} "
                f"direct_auroc={metrics[f'direct_auroc_{task_name}']:.6f} "
                f"strong_auprc={metrics[f'strong_direct_auprc_{task_name}']:.6f} "
                f"direct_auprc={metrics[f'direct_auprc_{task_name}']:.6f}"
            )
        print(
            f"[direct] epoch={epoch} loss={np.mean(losses):.4f} "
            f"phase={'A' if epoch < freeze_epochs else 'B'} "
            f"components={_mean_diagnostics(component_rows)} "
            f"attention_gate={np.mean(gates):.6f} winner={winner} "
            f"raw_metrics={raw_metrics} ema_metrics={ema_metrics} swa_metrics={swa_metrics}"
        )
        extra = {}
        if ema is not None:
            extra["ema"] = ema.state_dict()
        if swa is not None:
            extra["swa"] = swa.state_dict()
        _save_checkpoint(output_dir / "last_direct.pt", model, optimizer, cfg, "direct", epoch, raw_metrics, pos_weight, **extra)
        _save_last_finite(
            output_dir, model, optimizer, cfg, "direct", epoch, raw_metrics,
            pos_weight, loss_is_finite=bool(losses), **extra
        )
        if (
            (not _is_v6(model) or _direct_candidate_is_pareto_safe(raw_metrics, cfg))
            and _should_save_best(raw_score, raw_best, epoch, start_epoch)
        ):
            raw_best = raw_score
            _save_checkpoint(output_dir / "best_direct_raw.pt", model, optimizer, cfg, "direct", epoch, raw_metrics, pos_weight, selected_variant="raw", **extra)
        if (
            ema is not None
            and (not _is_v6(model) or _direct_candidate_is_pareto_safe(ema_metrics, cfg))
            and _should_save_best(ema_score, ema_best, epoch, start_epoch)
        ):
            ema_best = ema_score
            with ema.average_parameters(model):
                _save_checkpoint(output_dir / "best_direct_ema.pt", model, optimizer, cfg, "direct", epoch, ema_metrics, pos_weight, selected_variant="ema", **extra)
        if (
            swa is not None
            and swa_metrics is not None
            and (not _is_v6(model) or _direct_candidate_is_pareto_safe(swa_metrics, cfg))
            and _should_save_best(swa_score, swa_best, epoch, start_epoch)
        ):
            swa_best = swa_score
            with swa.average_parameters(model):
                _save_checkpoint(output_dir / "best_direct_swa.pt", model, optimizer, cfg, "direct", epoch, swa_metrics, pos_weight, selected_variant="swa", **extra)
        if safe_to_save and _should_save_best(score, best, epoch, start_epoch):
            best = score
            context = (
                ema.average_parameters(model) if winner == "ema"
                else swa.average_parameters(model) if winner == "swa"
                else nullcontext()
            )
            with context:
                _save_checkpoint(output_dir / "best_direct.pt", model, optimizer, cfg, "direct", epoch, metrics, pos_weight, selected_variant=winner, **extra)
                if v5:
                    _save_checkpoint(output_dir / "best_direct_v5.pt", model, optimizer, cfg, "direct", epoch, metrics, pos_weight, selected_variant=winner, **extra)
            _print_best_validation("direct", metrics, cfg, prefix="direct")

    best_path = output_dir / "best_direct.pt"
    if best_path.exists():
        try:
            best_state = torch.load(
                best_path, map_location=device, weights_only=False
            )
        except TypeError:
            best_state = torch.load(best_path, map_location=device)
        model.load_state_dict(best_state["model"])
        temperature = fit_and_save_direct_temperature(
            model, val_loader, device, output_dir, cfg
        )
        best_state["model"] = model.state_dict()
        best_state["direct_temperature"] = float(temperature)
        torch.save(best_state, best_path)
        if v5:
            torch.save(best_state, output_dir / "best_direct_v5.pt")


def train_operator_warmup(model, train_loader, val_loader, cfg, device,
                          output_dir, pos_weight, resume_state=None):
    configure_trainable_parameters(model, cfg, "operator_warmup")
    optimizer = make_optimizer(model, cfg, "operator_warmup")
    parameters = [p for p in model.parameters() if p.requires_grad]
    start_epoch = _resume_optimizer(optimizer, resume_state, "operator_warmup")
    best = float("-inf")
    nonfinite_state = {}
    epochs = int(cfg["training"].get("epochs_operator_warmup", 5))
    fixed_alpha = cfg["training"].get("operator_warmup", {}).get(
        "fixed_alpha"
    )
    for epoch in range(start_epoch, epochs):
        model.train()
        model.reset_numerics_observations()
        losses, diagnostic_rows = [], []
        skipped_before = int(nonfinite_state.get("skipped", 0))
        for batch_idx, batch in enumerate(train_loader):
            batch = move_batch(batch, device)
            batch = apply_modality_dropout(batch, cfg)
            model.set_numerics_context("operator_warmup", epoch, batch_idx)
            outputs = supervised_round_robin(
                model, batch, fixed_alpha=fixed_alpha
            )
            loss, diagnostics = residual_supervised_objective(
                outputs,
                batch,
                pos_weight,
                cfg,
                delta_weight_default=float(
                    cfg["training"].get("operator_delta_l2", 0.1)
                ),
                stage="operator_warmup",
                epoch=epoch,
            )
            loss = loss + operator_regularization(model, outputs["operators"], cfg)
            if not _clip_and_step(
                loss, optimizer, parameters, cfg, model=model,
                stage="operator_warmup", epoch=epoch, batch_idx=batch_idx,
                nonfinite_state=nonfinite_state,
            ):
                continue
            losses.append(loss.item())
            diagnostic_rows.append(
                {key: float(value.detach().cpu()) for key, value in diagnostics.items()}
            )
        metrics = evaluate_model(
            model, val_loader, cfg, device, deterministic=True,
            split="val", pos_weight=pos_weight,
            stage="operator_warmup", epoch=epoch
        )
        score = _primary_score(metrics, cfg)
        print(
            f"[operator_warmup] epoch={epoch} loss={np.mean(losses):.4f} "
            f"fixed_alpha={fixed_alpha} components={_mean_diagnostics(diagnostic_rows)} "
            f"metrics={metrics}"
        )
        numerics = model.numerics_summary()
        numerics["skipped_batches"] = int(
            nonfinite_state.get("skipped", 0)
        ) - skipped_before
        print(
            "[numerics] "
            + " ".join(f"{key}={value}" for key, value in numerics.items())
        )
        _save_checkpoint(output_dir / "last_operator_warmup.pt", model, optimizer, cfg, "operator_warmup", epoch, metrics, pos_weight)
        _save_last_finite(
            output_dir, model, optimizer, cfg, "operator_warmup", epoch,
            metrics, pos_weight, loss_is_finite=bool(losses),
            skipped_batches=numerics["skipped_batches"],
        )
        if (
            _should_save_best(score, best, epoch, start_epoch)
            and _passes_v5_direct_guard(metrics, cfg, output_dir)
        ):
            best = score
            _save_checkpoint(output_dir / "best_operator_warmup.pt", model, optimizer, cfg, "operator_warmup", epoch, metrics, pos_weight)
            _print_best_validation("operator_warmup", metrics, cfg)


def train_supervised(model, train_loader, val_loader, cfg, device, output_dir,
                     pos_weight=None, resume_state=None):
    if pos_weight is None:
        pos_weight, _ = compute_train_pos_weight(train_loader, device)
    configure_trainable_parameters(model, cfg, "supervised")
    optimizer = make_optimizer(model, cfg, "supervised")
    parameters = [p for p in model.parameters() if p.requires_grad]
    start_epoch = _resume_optimizer(optimizer, resume_state, "supervised")
    ema_cfg = cfg["training"].get("ema", {})
    ema = (
        ModelEMA(model, decay=float(ema_cfg.get("decay", 0.999)))
        if bool(ema_cfg.get("enabled", False)) else None
    )
    if ema is not None and resume_state and "ema" in resume_state:
        ema.load_state_dict(resume_state["ema"])
    best = raw_best = ema_best = float("-inf")
    nonfinite_state = {}
    alpha_floor_epochs = residual_collapse_epochs = 0
    for epoch in range(start_epoch, int(cfg["training"]["epochs_supervised"])):
        model.train()
        model.reset_numerics_observations()
        losses, diagnostic_rows = [], []
        for batch_idx, batch in enumerate(train_loader):
            batch = move_batch(batch, device)
            batch = apply_modality_dropout(batch, cfg)
            model.set_numerics_context("supervised", epoch, batch_idx)
            oracle_diagnostics = {}
            oracle_set = None
            if getattr(model, "performance_v5", False):
                model.eval()
                if _is_v6(model):
                    routing_cfg = model.performance_v6_cfg.get(
                        "routing", cfg.get("routing", {})
                    )
                    with torch.no_grad():
                        oracle_set = build_oracle_candidate_set(
                            model,
                            batch,
                            pos_weight,
                            candidate_count=int(
                                routing_cfg.get("num_candidates", 24)
                            ),
                        )
                    best_index = oracle_set["utility"].argmax(1)
                    best_actions = torch.stack(
                        [
                            oracle_set["paths"][index][sample]
                            for sample, index in enumerate(best_index.tolist())
                        ],
                        0,
                    )
                    best_gain = oracle_set["bce_gain"].gather(
                        1, best_index[:, None]
                    ).squeeze(1)
                    oracle_diagnostics = routing_ranking_metrics(
                        oracle_set["policy_scores"].cpu().numpy(),
                        oracle_set["utility"].cpu().numpy(),
                    )
                else:
                    best_actions, best_gain, _, oracle_diagnostics = select_oracle_candidate(
                        model,
                        batch,
                        pos_weight,
                        candidate_count=int(
                            cfg["training"].get("oracle_candidates", 12)
                        ),
                    )
                model.train()
                outputs = model.rollout_actions(batch, best_actions)
            else:
                outputs = supervised_round_robin(model, batch)
            loss, diagnostics = residual_supervised_objective(
                outputs,
                batch,
                pos_weight,
                cfg,
                stage="supervised",
                epoch=epoch,
            )
            if getattr(model, "performance_v5", False):
                reason_loss, reason_parts = reason_gate_loss(
                    outputs["reason_probability"],
                    best_gain,
                    focal_gamma=float(
                        cfg["training"].get("reason_gate", {}).get("focal_gamma", 1.5)
                    ),
                )
                useful = best_gain > 0
                if _is_v6(model) and oracle_set is not None:
                    policy_scores = differentiable_candidate_scores(
                        model,
                        batch,
                        oracle_set["paths"],
                        oracle_set["encoded"],
                        oracle_set["operators"],
                    )
                    routing_cfg = model.performance_v6_cfg.get(
                        "routing", cfg.get("routing", {})
                    )
                    path_loss, path_parts = path_distillation_loss(
                        policy_scores,
                        oracle_set["utility"],
                        temperature=float(
                            routing_cfg.get("oracle_temperature", 0.20)
                        ),
                        listwise_weight=float(
                            routing_cfg.get("listwise_weight", 0.50)
                        ),
                        top1_weight=float(
                            routing_cfg.get("top1_ce_weight", 0.30)
                        ),
                        pairwise_weight=float(
                            routing_cfg.get("pairwise_weight", 0.20)
                        ),
                    )
                    reranker_scores = candidate_reranker_scores(
                        model,
                        batch,
                        oracle_set["paths"],
                        oracle_set["encoded"],
                        oracle_set["operators"],
                    )
                    reranker_cfg = model.performance_v6_cfg.get(
                        "reranker", cfg.get("reranker", {})
                    )
                    reranker_loss, reranker_parts = reranker_objective(
                        reranker_scores,
                        oracle_set["utility"],
                        policy_scores.detach(),
                        regression_weight=float(
                            reranker_cfg.get("regression_weight", 0.50)
                        ),
                        ranking_weight=float(
                            reranker_cfg.get("ranking_weight", 0.50)
                        ),
                    )
                elif useful.any():
                    path_loss = -outputs["path_logprob"][useful].mean()
                    path_parts, reranker_parts = {}, {}
                    reranker_loss = path_loss * 0.0
                else:
                    path_loss = outputs["path_logprob"].sum() * 0.0
                    path_parts, reranker_parts = {}, {}
                    reranker_loss = path_loss * 0.0
                oracle_weight = float(
                    cfg["training"].get("oracle_loss_weight", 0.60)
                )
                loss = loss + oracle_weight * (
                    reason_loss + path_loss + reranker_loss
                )
                diagnostics.update(
                    {
                        "reason_loss": reason_loss,
                        "reason_bce": reason_parts["bce"],
                        "path_distillation": path_loss,
                        "reranker_loss": reranker_loss,
                        "oracle_useful_fraction": useful.float().mean(),
                        **path_parts,
                        **reranker_parts,
                    }
                )
            loss = loss + operator_regularization(model, outputs["operators"], cfg)
            if not _clip_and_step(
                loss, optimizer, parameters, cfg, model=model,
                stage="supervised", epoch=epoch, batch_idx=batch_idx,
                nonfinite_state=nonfinite_state,
            ):
                continue
            if ema is not None:
                ema.update(model)
            losses.append(loss.item())
            diagnostic_rows.append(
                {key: float(value.detach().cpu()) for key, value in diagnostics.items()}
            )
            if oracle_diagnostics:
                print(
                    "[candidate-diversity] "
                    + json.dumps(oracle_diagnostics, sort_keys=True)
                )
        raw_metrics = evaluate_model(
            model,
            val_loader,
            cfg,
            device,
            deterministic=True,
            split="val",
            pos_weight=pos_weight,
            stage="supervised",
            epoch=epoch,
        )
        raw_score = _primary_score(raw_metrics, cfg)
        ema_metrics = None
        ema_score = float("-inf")
        if ema is not None:
            with ema.average_parameters(model):
                ema_metrics = evaluate_model(
                    model,
                    val_loader,
                    cfg,
                    device,
                    deterministic=True,
                    split="val",
                    pos_weight=pos_weight,
                    stage="supervised",
                    epoch=epoch,
                )
            ema_score = _primary_score(ema_metrics, cfg)
        winner_is_ema = ema_metrics is not None and ema_score > raw_score
        metrics = ema_metrics if winner_is_ema else raw_metrics
        score = ema_score if winner_is_ema else raw_score
        print(
            f"[supervised] epoch={epoch} loss={np.mean(losses):.4f} "
            f"components={_mean_diagnostics(diagnostic_rows)} "
            f"raw_metrics={raw_metrics} ema_metrics={ema_metrics} "
            f"winner={'ema' if winner_is_ema else 'raw'}"
        )
        checkpoint_extra = {"ema": ema.state_dict()} if ema is not None else {}
        _save_checkpoint(
            output_dir / "last_supervised.pt",
            model,
            optimizer,
            cfg,
            "supervised",
            epoch,
            raw_metrics,
            pos_weight,
            **checkpoint_extra,
        )
        if getattr(model, "performance_v5", False):
            _save_checkpoint(
                output_dir / f"supervised_epoch_{epoch}.pt",
                model, optimizer, cfg, "supervised", epoch, raw_metrics,
                pos_weight, **checkpoint_extra,
            )
        _save_last_finite(
            output_dir, model, optimizer, cfg, "supervised", epoch,
            raw_metrics, pos_weight, loss_is_finite=bool(losses),
            skipped_batches=int(nonfinite_state.get("skipped", 0)),
            **checkpoint_extra,
        )
        if _should_save_best(raw_score, raw_best, epoch, start_epoch):
            raw_best = raw_score
            _save_checkpoint(
                output_dir / "best_supervised_raw.pt",
                model,
                optimizer,
                cfg,
                "supervised",
                epoch,
                raw_metrics,
                pos_weight,
                **checkpoint_extra,
            )
        if ema is not None and _should_save_best(
            ema_score, ema_best, epoch, start_epoch
        ):
            ema_best = ema_score
            with ema.average_parameters(model):
                _save_checkpoint(
                    output_dir / "best_supervised_ema.pt",
                    model,
                    optimizer,
                    cfg,
                    "supervised",
                    epoch,
                    ema_metrics,
                    pos_weight,
                    **checkpoint_extra,
                )
        if (
            _should_save_best(score, best, epoch, start_epoch)
            and _passes_v5_direct_guard(metrics, cfg, output_dir)
        ):
            best = score
            if winner_is_ema:
                with ema.average_parameters(model):
                    _save_checkpoint(
                        output_dir / "best_supervised.pt",
                        model,
                        optimizer,
                        cfg,
                        "supervised",
                        epoch,
                        metrics,
                        pos_weight,
                        selected_variant="ema",
                        **checkpoint_extra,
                    )
            else:
                _save_checkpoint(
                    output_dir / "best_supervised.pt",
                    model,
                    optimizer,
                    cfg,
                    "supervised",
                    epoch,
                    metrics,
                    pos_weight,
                    selected_variant="raw",
                    **checkpoint_extra,
                )
            _print_best_validation("supervised", metrics, cfg)

        alpha_min = float(cfg["model"].get("residual", {}).get("alpha_min", 0.0))
        alpha_floor_epochs = (
            alpha_floor_epochs + 1
            if raw_metrics.get("mean_operator_gate_alpha", 1.0) <= alpha_min + 0.005
            else 0
        )
        residual_collapse_epochs = (
            residual_collapse_epochs + 1
            if raw_metrics.get("relative_scaled_operator_delta", 1.0) < 1e-4
            else 0
        )
        if alpha_floor_epochs >= 2:
            print("[collapse-warning] patient gate near floor")
        if residual_collapse_epochs >= 2:
            print("[collapse-warning] operator residual collapsed")

    if getattr(model, "performance_v5", False):
        best_path = Path(output_dir) / "best_supervised.pt"
        if best_path.exists():
            try:
                best_state = torch.load(best_path, map_location=device, weights_only=False)
            except TypeError:
                best_state = torch.load(best_path, map_location=device)
            model.load_state_dict(best_state["model"])
        select_correction_bound_from_validation(
            model, val_loader, cfg, device, output_dir, pos_weight,
            stage="supervised"
        )


def _index_batch(batch, indices):
    output = {}
    for key, value in batch.items():
        if key == "modalities":
            output[key] = {}
            for modality, modality_value in value.items():
                if isinstance(modality_value, dict):
                    output[key][modality] = {
                        name: tensor.index_select(0, indices)
                        for name, tensor in modality_value.items()
                    }
                elif torch.is_tensor(modality_value):
                    output[key][modality] = modality_value.index_select(0, indices)
                else:
                    output[key][modality] = [
                        modality_value[index] for index in indices.tolist()
                    ]
        elif torch.is_tensor(value):
            output[key] = value.index_select(0, indices)
        elif isinstance(value, list):
            output[key] = [value[index] for index in indices.tolist()]
        else:
            output[key] = value
    return output


def balance_policy_targets(batch, best_nonstop_actions, useful_mask, stop_idx,
                           target_nonstop_fraction=0.5):
    """Build a class-balanced training batch; useful cases may be oversampled."""
    batch_size = useful_mask.numel()
    desired_nonstop = int(round(batch_size * float(target_nonstop_fraction)))
    desired_nonstop = min(max(desired_nonstop, 0), batch_size)
    useful_indices = useful_mask.nonzero(as_tuple=False).flatten()
    stop_indices = (~useful_mask).nonzero(as_tuple=False).flatten()
    all_indices = torch.arange(batch_size, device=useful_mask.device)

    def draw(pool, count):
        if count <= 0:
            return all_indices[:0]
        if pool.numel() == 0:
            pool = all_indices
        selected = torch.randint(pool.numel(), (count,), device=pool.device)
        return pool[selected]

    if useful_indices.numel() == 0:
        desired_nonstop = 0
    nonstop_source = draw(useful_indices, desired_nonstop)
    stop_source = draw(stop_indices, batch_size - desired_nonstop)
    source = torch.cat([nonstop_source, stop_source])
    actions = torch.full(
        (batch_size, best_nonstop_actions.size(1)),
        int(stop_idx),
        dtype=torch.long,
        device=best_nonstop_actions.device,
    )
    if desired_nonstop:
        actions[:desired_nonstop] = best_nonstop_actions.index_select(
            0, nonstop_source
        )
    permutation = torch.randperm(batch_size, device=source.device)
    source = source[permutation]
    actions = actions[permutation]
    return _index_batch(batch, source), actions, source


def _sample_nonstop_paths(model, batch, candidate_count):
    paths = []
    modality_mask = batch["modality_mask"]
    batch_size = modality_mask.size(0)
    for _ in range(int(candidate_count)):
        actions = torch.full(
            (batch_size, model.max_steps),
            model.stop_idx,
            dtype=torch.long,
            device=modality_mask.device,
        )
        for sample in range(batch_size):
            available = (modality_mask[sample] > 0).nonzero(
                as_tuple=False
            ).flatten()
            if available.numel() == 0:
                continue
            for step in range(model.max_steps):
                modality = available[
                    torch.randint(available.numel(), (1,), device=available.device)
                ].item()
                concept = torch.randint(
                    model.K, (1,), device=available.device
                ).item()
                actions[sample, step] = modality * model.K + concept
        paths.append(actions)
    return paths


def sample_diverse_candidate_paths(
    model, batch, encoded=None, operators=None, candidate_count=12
):
    """Create top, random, and modality-balanced oracle candidates."""
    if int(candidate_count) != 12:
        top_count = int(candidate_count) // 3
        random_count = int(candidate_count) // 3
        balanced_count = int(candidate_count) - top_count - random_count
    else:
        top_count = random_count = balanced_count = 4
    encoded = encoded or model.encode(batch)
    operators = operators or model.build_operators(encoded, batch["modality_mask"])
    mask = batch["modality_mask"]
    batch_size = mask.size(0)

    def empty_path():
        return torch.full(
            (batch_size, model.max_steps),
            model.stop_idx,
            dtype=torch.long,
            device=mask.device,
        )

    paths = []
    # Actual top-policy paths with rank offsets for controlled diversity.
    for offset in range(top_count):
        actions = empty_path()
        state = model.initial(batch_size, mask.device)
        previous = torch.full(
            (batch_size,), model.policy.num_actions, dtype=torch.long,
            device=mask.device,
        )
        for step in range(model.max_steps):
            logits = model._policy_inputs(
                state, encoded, operators, mask, previous, step
            )
            rank = min(offset, logits.size(1) - 1)
            action = logits.topk(rank + 1, dim=1).indices[:, rank]
            actions[:, step] = action
            state, _ = model.apply_action(state, encoded, operators, action)
            previous = action
        paths.append(actions)
    paths.extend(_sample_nonstop_paths(model, batch, random_count))
    # Balanced candidates deliberately rotate the start modality.
    for offset in range(balanced_count):
        actions = empty_path()
        for sample in range(batch_size):
            available = (mask[sample] > 0).nonzero(as_tuple=False).flatten()
            if available.numel() == 0:
                continue
            for step in range(model.max_steps):
                modality_index = int(available[(offset + step) % available.numel()])
                concept = (offset + step) % model.K
                actions[sample, step] = modality_index * model.K + concept
        paths.append(actions)
    return paths


@torch.no_grad()
def build_oracle_candidate_set(model, batch, pos_weight, candidate_count=24):
    """Build label-aware train/diagnostic targets; never called by inference."""
    encoded = model.encode(batch)
    operators = model.build_operators(encoded, batch["modality_mask"])
    paths = sample_diverse_candidate_paths(
        model,
        batch,
        encoded=encoded,
        operators=operators,
        candidate_count=candidate_count,
    )
    direct = model.direct_outputs(encoded, batch["modality_mask"])["direct_logits"]
    direct_loss = per_sample_masked_bce(
        direct, batch["labels"], batch["label_mask"], pos_weight
    )
    candidate_logits, bce_gains, rank_gains, pr_gains = [], [], [], []
    policy_scores = []
    signed = batch["labels"] * 2.0 - 1.0
    valid_count = batch["label_mask"].sum(1).clamp_min(1.0)
    direct_probability = torch.sigmoid(direct.detach())
    for actions in paths:
        rollout = model.rollout_actions(
            batch, actions, encoded=encoded, operators=operators
        )
        logits = rollout["logits"]
        candidate_logits.append(logits)
        policy_scores.append(rollout["path_logprob"])
        candidate_loss = per_sample_masked_bce(
            logits, batch["labels"], batch["label_mask"], pos_weight
        )
        bce_gains.append(direct_loss - candidate_loss)
        signed_gain = (
            signed * (logits - direct) * batch["label_mask"]
        ).sum(1) / valid_count
        rank_gains.append(signed_gain)
        hard_factor = torch.where(
            batch["labels"] > 0.5,
            1.0 + (1.0 - direct_probability),
            1.0 + direct_probability,
        )
        pr_gains.append(
            (signed * (logits - direct) * hard_factor * batch["label_mask"]).sum(1)
            / valid_count
        )
    bce_gain = torch.stack(bce_gains, 1)
    rank_gain = torch.stack(rank_gains, 1)
    pr_gain = torch.stack(pr_gains, 1)
    weights = (
        (0.45, 0.30, 0.25)
        if _is_v6(model)
        else (0.50, 0.25, 0.25)
    )
    utility = oracle_candidate_utility(
        bce_gain, rank_gain, pr_gain, weights=weights
    )
    return {
        "encoded": encoded,
        "operators": operators,
        "paths": paths,
        "actions": torch.stack(paths, 1),
        "candidate_logits": torch.stack(candidate_logits, 1),
        "policy_scores": torch.stack(policy_scores, 1),
        "bce_gain": bce_gain,
        "rank_gain": rank_gain,
        "pr_gain": pr_gain,
        "utility": patient_wise_utility_normalize(utility),
    }


@torch.no_grad()
def select_oracle_candidate(model, batch, pos_weight, candidate_count=12):
    details = build_oracle_candidate_set(
        model, batch, pos_weight, candidate_count=candidate_count
    )
    utility = details["utility"]
    best_utility, best_index = utility.max(1)
    paths = details["paths"]
    best_actions = torch.stack(
        [paths[index][sample] for sample, index in enumerate(best_index.tolist())],
        0,
    )
    best_bce_gain = details["bce_gain"].gather(
        1, best_index.unsqueeze(1)
    ).squeeze(1)
    all_actions = details["actions"]
    all_modalities = all_actions.clamp_max(model.stop_idx - 1) // model.K
    best_modalities = best_actions.clamp_max(model.stop_idx - 1) // model.K
    coverage, best_distribution = [], []
    for modality_index in range(model.M):
        coverage.append(float((all_modalities == modality_index).float().mean()))
        best_distribution.append(
            float((best_modalities == modality_index).float().mean())
        )
    diagnostics = {
        "candidate_utility_std": float(utility.std(unbiased=False)),
        "candidate_modality_coverage": coverage,
        "best_path_modality_distribution": best_distribution,
    }
    diagnostics.update(
        routing_ranking_metrics(
            details["policy_scores"].cpu().numpy(),
            utility.cpu().numpy(),
        )
    )
    return best_actions, best_bce_gain, best_utility, diagnostics


def differentiable_candidate_scores(model, batch, paths, encoded, operators):
    scores = []
    for actions in paths:
        rollout = model.rollout_actions(
            batch, actions, encoded=encoded, operators=operators
        )
        scores.append(rollout["path_logprob"])
    return torch.stack(scores, 1)


def candidate_reranker_scores(model, batch, paths, encoded, operators):
    if model.reranker is None:
        raise RuntimeError("v6 reranker is required")
    scores = []
    for actions in paths:
        with torch.no_grad():
            rollout = model.rollout_actions(
                batch, actions, encoded=encoded, operators=operators
            )
        scores.append(
            model.score_candidate_rollout(rollout, batch["modality_mask"])
        )
    return torch.stack(scores, 1)


def select_policy_checkpoint(output_dir, cfg=None):
    records = []
    for path in sorted(Path(output_dir).glob("policy_epoch_*.pt")):
        try:
            state = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            state = torch.load(path, map_location="cpu")
        metrics = state.get("metrics", {})
        stop_rate = float(metrics.get("stop_rate", 1.0))
        dual_values = [
            float(value)
            for key, value in metrics.items()
            if key.startswith("final_dual_score_") and value is not None
        ]
        selection_score = (
            dual_values[0]
            if dual_values
            else float(metrics.get("mean_total_reward", float("-inf")))
        )
        if cfg is not None and not _passes_v5_direct_guard(
            metrics, cfg, output_dir
        ):
            continue
        records.append((path, stop_rate, selection_score))
    if not records:
        return None
    eligible = [record for record in records if 0.1 <= record[1] <= 0.9]
    if cfg is not None and bool(
        cfg.get("model", {}).get("performance_v6", {}).get("enabled", False)
    ) and not eligible:
        print("[reason-rate-guard] no eligible policy checkpoint")
        return None
    selected = (
        max(eligible, key=lambda item: item[2])
        if eligible
        else min(records, key=lambda item: (item[1], -item[2]))
    )
    shutil.copyfile(selected[0], Path(output_dir) / "best_policy_warmup.pt")
    print(
        f"[policy-selection] checkpoint={selected[0].name} "
        f"stop_rate={selected[1]:.6f} validation_score={selected[2]:.6f}"
    )
    if cfg is not None:
        try:
            selected_state = torch.load(
                selected[0], map_location="cpu", weights_only=False
            )
        except TypeError:
            selected_state = torch.load(selected[0], map_location="cpu")
        _print_best_validation(
            "policy_warmup", selected_state.get("metrics", {}), cfg
        )
    return selected[0]


def select_reason_threshold_from_validation(
    model, val_loader, cfg, device, output_dir, pos_weight, stage="policy_warmup"
):
    if not getattr(model, "performance_v5", False):
        return None
    split = getattr(val_loader.dataset, "split", "val")
    if split != "val":
        raise ValueError("reason threshold selection is validation-only")
    candidates = cfg.get("selection", {}).get(
        "reason_thresholds", [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
    )
    task_name = cfg["experiment"]["task_names"][0]
    rows = []
    for threshold in candidates:
        model.reason_threshold.fill_(float(threshold))
        metrics = evaluate_model(
            model,
            val_loader,
            cfg,
            device,
            deterministic=True,
            split="val",
            pos_weight=pos_weight,
            stage=stage,
        )
        rows.append(
            {
                "threshold": float(threshold),
                "reason_rate": float(metrics["reason_rate"]),
                "auroc": float(metrics[f"final_auroc_{task_name}"]),
                "auprc": float(metrics[f"final_auprc_{task_name}"]),
                "score": float(metrics[f"final_dual_score_{task_name}"]),
                "nll": float(metrics[f"final_nll_{task_name}"]),
            }
        )
    eligible = [row for row in rows if 0.10 <= row["reason_rate"] <= 0.90]
    selected = max(eligible or rows, key=metric_sort_key)
    model.reason_threshold.fill_(selected["threshold"])
    payload = {"fit_split": "val", "selected": selected, "candidates": rows}
    (Path(output_dir) / "reason_threshold_v5.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    print(
        "[reason-threshold] "
        f"threshold={selected['threshold']:.2f} "
        f"reason_rate={selected['reason_rate']:.6f} "
        f"validation_score={selected['score']:.6f}"
    )
    return selected


def load_saved_reason_threshold(model, output_dir):
    path = Path(output_dir) / "reason_threshold_v5.json"
    if not path.exists() or not getattr(model, "performance_v5", False):
        return None
    payload = json.loads(path.read_text())
    if payload.get("fit_split") != "val":
        raise ValueError("refusing reason threshold not selected on validation")
    threshold = float(payload["selected"]["threshold"])
    model.reason_threshold.fill_(threshold)
    print(f"[reason-threshold] loaded={threshold:.2f} fit_split=val")
    return threshold


def select_correction_bound_from_validation(
    model, val_loader, cfg, device, output_dir, pos_weight, stage="supervised"
):
    if not getattr(model, "performance_v5", False):
        return None
    split = getattr(val_loader.dataset, "split", "val")
    if split != "val":
        raise ValueError("correction bound selection is validation-only")
    candidates = cfg.get("model", {}).get("performance_v5", {}).get(
        "bounded_correction", {}
    ).get("candidates", [0.20, 0.35, 0.50])
    task_name = cfg["experiment"]["task_names"][0]
    rows = []
    for bound in candidates:
        model.set_correction_bound(float(bound))
        metrics = evaluate_model(
            model, val_loader, cfg, device, deterministic=True, split="val",
            pos_weight=pos_weight, stage=stage,
        )
        rows.append(
            {
                "c": float(bound),
                "auroc": float(metrics[f"final_auroc_{task_name}"]),
                "auprc": float(metrics[f"final_auprc_{task_name}"]),
                "score": float(metrics[f"final_dual_score_{task_name}"]),
                "nll": float(metrics[f"final_nll_{task_name}"]),
            }
        )
    selected = max(rows, key=metric_sort_key)
    model.set_correction_bound(selected["c"])
    (Path(output_dir) / "correction_bound_v5.json").write_text(
        json.dumps(
            {"fit_split": "val", "selected": selected, "candidates": rows},
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(
        f"[correction-bound] c={selected['c']:.2f} "
        f"validation_score={selected['score']:.6f}"
    )
    return selected


def load_saved_correction_bound(model, output_dir):
    path = Path(output_dir) / "correction_bound_v5.json"
    if not path.exists() or not getattr(model, "performance_v5", False):
        return None
    payload = json.loads(path.read_text())
    if payload.get("fit_split") != "val":
        raise ValueError("refusing correction bound not selected on validation")
    value = float(payload["selected"]["c"])
    model.set_correction_bound(value)
    print(f"[correction-bound] loaded={value:.2f} fit_split=val")
    return value


def _train_policy_warmup_v6(
    model, train_loader, val_loader, cfg, device, output_dir,
    pos_weight=None, resume_state=None,
):
    if pos_weight is None:
        pos_weight, _ = compute_train_pos_weight(train_loader, device)
    start_epoch = (
        int(resume_state.get("epoch", -1)) + 1
        if resume_state and resume_state.get("stage") == "policy_warmup"
        else 0
    )
    routing_cfg = model.performance_v6_cfg.get(
        "routing", cfg.get("routing", {})
    )
    reranker_cfg = model.performance_v6_cfg.get(
        "reranker", cfg.get("reranker", {})
    )
    candidate_count = int(routing_cfg.get("num_candidates", 24))
    nonfinite_state = {}
    optimizer = None
    parameters = []
    current_phase = None
    for epoch in range(start_epoch, int(cfg["training"]["epochs_policy_warmup"])):
        phase = 1 if epoch < 4 else 2 if epoch < 7 else 3
        if phase != current_phase:
            configure_trainable_parameters(
                model, cfg, "policy_warmup", epoch=epoch
            )
            optimizer = make_optimizer(model, cfg, "policy_warmup")
            parameters = [
                parameter
                for parameter in model.parameters()
                if parameter.requires_grad
            ]
            current_phase = phase
        model.train()
        model.reset_numerics_observations()
        losses, statistic_rows = [], []
        for batch_idx, batch in enumerate(train_loader):
            batch = move_batch(batch, device)
            batch = apply_modality_dropout(batch, cfg)
            model.set_numerics_context("policy_warmup", epoch, batch_idx)
            model.eval()
            with torch.no_grad():
                oracle = build_oracle_candidate_set(
                    model,
                    batch,
                    pos_weight,
                    candidate_count=candidate_count,
                )
            model.train()
            policy_scores = differentiable_candidate_scores(
                model,
                batch,
                oracle["paths"],
                oracle["encoded"],
                oracle["operators"],
            )
            path_loss, path_parts = path_distillation_loss(
                policy_scores,
                oracle["utility"],
                temperature=float(routing_cfg.get("oracle_temperature", 0.20)),
                listwise_weight=float(routing_cfg.get("listwise_weight", 0.50)),
                top1_weight=float(routing_cfg.get("top1_ce_weight", 0.30)),
                pairwise_weight=float(routing_cfg.get("pairwise_weight", 0.20)),
            )
            total = path_loss
            reranker_scores = None
            reranker_parts = {}
            if epoch >= 4:
                reranker_scores = candidate_reranker_scores(
                    model,
                    batch,
                    oracle["paths"],
                    oracle["encoded"],
                    oracle["operators"],
                )
                reranker_loss, reranker_parts = reranker_objective(
                    reranker_scores,
                    oracle["utility"],
                    policy_scores.detach(),
                    regression_weight=float(
                        reranker_cfg.get("regression_weight", 0.50)
                    ),
                    ranking_weight=float(
                        reranker_cfg.get("ranking_weight", 0.50)
                    ),
                )
                total = total + reranker_loss
            reason_parts = {}
            if epoch >= 7:
                best_index = oracle["utility"].argmax(1)
                best_actions = torch.stack(
                    [
                        oracle["paths"][index][sample]
                        for sample, index in enumerate(best_index.tolist())
                    ],
                    0,
                )
                reason_rollout = model.rollout_actions(
                    batch,
                    best_actions,
                    encoded=oracle["encoded"],
                    operators=oracle["operators"],
                )
                best_gain = oracle["bce_gain"].gather(
                    1, best_index[:, None]
                ).squeeze(1)
                reason_loss, reason_parts = reason_gate_loss(
                    reason_rollout["reason_probability"],
                    best_gain,
                    focal_gamma=float(
                        cfg["training"].get("policy_warmup", {}).get(
                            "focal_gamma", 1.5
                        )
                    ),
                )
                total = total + reason_loss
            if not _clip_and_step(
                total,
                optimizer,
                parameters,
                cfg,
                model=model,
                stage="policy_warmup",
                epoch=epoch,
                batch_idx=batch_idx,
                nonfinite_state=nonfinite_state,
            ):
                continue
            losses.append(float(total.detach()))
            policy_diag = routing_ranking_metrics(
                policy_scores.detach().cpu().numpy(),
                oracle["utility"].cpu().numpy(),
            )
            row = dict(policy_diag)
            row.update(
                {
                    f"path_{key}": float(value.detach())
                    for key, value in path_parts.items()
                }
            )
            if reranker_scores is not None:
                reranker_diag = routing_ranking_metrics(
                    reranker_scores.detach().cpu().numpy(),
                    oracle["utility"].cpu().numpy(),
                )
                row.update(
                    {
                        "reranker_top1_agreement": reranker_diag[
                            "oracle_top1_agreement"
                        ],
                        "reranker_top3_recall": reranker_diag[
                            "oracle_top3_recall"
                        ],
                        "reranker_mean_utility_regret": reranker_diag[
                            "mean_utility_regret"
                        ],
                    }
                )
                row.update(
                    {
                        key: float(value.detach())
                        for key, value in reranker_parts.items()
                    }
                )
            if reason_parts:
                row["reason_bce"] = float(reason_parts["bce"].detach())
            statistic_rows.append(row)
        metrics = evaluate_model(
            model,
            val_loader,
            cfg,
            device,
            deterministic=True,
            split="val",
            pos_weight=pos_weight,
            stage="policy_warmup",
            epoch=epoch,
            include_oracle=True,
        )
        statistics = _mean_diagnostics(statistic_rows)
        metrics.update({f"train_{key}": value for key, value in statistics.items()})
        print(
            f"[policy_warmup] epoch={epoch} phase={phase} "
            f"loss={np.mean(losses):.4f} routing={statistics} metrics={metrics}"
        )
        _save_checkpoint(
            output_dir / "last_policy_warmup.pt",
            model,
            optimizer,
            cfg,
            "policy_warmup",
            epoch,
            metrics,
            pos_weight,
        )
        _save_checkpoint(
            output_dir / f"policy_epoch_{epoch}.pt",
            model,
            optimizer,
            cfg,
            "policy_warmup",
            epoch,
            metrics,
            pos_weight,
        )
        _save_last_finite(
            output_dir,
            model,
            optimizer,
            cfg,
            "policy_warmup",
            epoch,
            metrics,
            pos_weight,
            loss_is_finite=bool(losses),
            skipped_batches=int(nonfinite_state.get("skipped", 0)),
        )
    selected_path = select_policy_checkpoint(output_dir, cfg)
    if selected_path is not None:
        selected_state = _load_torch_checkpoint(selected_path, device)
        model.load_state_dict(selected_state["model"])
    select_reason_threshold_from_validation(
        model,
        val_loader,
        cfg,
        device,
        output_dir,
        pos_weight,
        stage="policy_warmup",
    )


def train_policy_warmup(model, train_loader, val_loader, cfg, device,
                        output_dir, pos_weight=None, resume_state=None):
    if _is_v6(model):
        return _train_policy_warmup_v6(
            model,
            train_loader,
            val_loader,
            cfg,
            device,
            output_dir,
            pos_weight=pos_weight,
            resume_state=resume_state,
        )
    if pos_weight is None:
        pos_weight, _ = compute_train_pos_weight(train_loader, device)
    configure_trainable_parameters(model, cfg, "policy_warmup")
    optimizer = make_optimizer(model, cfg, "policy_warmup")
    parameters = [p for p in model.parameters() if p.requires_grad]
    start_epoch = _resume_optimizer(optimizer, resume_state, "policy_warmup")
    candidates_count = int(cfg["training"].get("policy_warmup_candidates", 8))
    warmup_cfg = cfg["training"].get("policy_warmup", {})
    balance_enabled = bool(warmup_cfg.get("balance_stop_nonstop", False))
    target_fraction = float(warmup_cfg.get("target_nonstop_fraction", 0.5))
    utility_margin = float(warmup_cfg.get("utility_margin", 0.0))
    entropy_base = float(warmup_cfg.get("entropy_weight", 0.0))
    entropy_epochs = int(warmup_cfg.get("entropy_epochs", 3))
    nonfinite_state = {}
    for epoch in range(start_epoch, int(cfg["training"]["epochs_policy_warmup"])):
        model.reset_numerics_observations()
        losses, statistic_rows = [], []
        for batch_idx, batch in enumerate(train_loader):
            batch = move_batch(batch, device)
            batch = apply_modality_dropout(batch, cfg)
            model.set_numerics_context("policy_warmup", epoch, batch_idx)
            if getattr(model, "performance_v5", False):
                model.eval()
                with torch.no_grad():
                    actions, best_gain, _, candidate_diagnostics = select_oracle_candidate(
                        model,
                        batch,
                        pos_weight,
                        candidate_count=int(
                            cfg["training"].get("oracle_candidates", 12)
                        ),
                    )
                model.train()
                rollout = model.rollout_actions(batch, actions)
                useful = best_gain > float(utility_margin)
                reason_loss, reason_parts = reason_gate_loss(
                    rollout["reason_probability"],
                    best_gain,
                    focal_gamma=float(warmup_cfg.get("focal_gamma", 1.5)),
                )
                if useful.any():
                    path_nll = -rollout["path_logprob"][useful].mean()
                    path_entropy = rollout["entropy"][useful].mean()
                else:
                    path_nll = rollout["path_logprob"].sum() * 0.0
                    path_entropy = rollout["entropy"].sum() * 0.0
                entropy_weight = float(warmup_cfg.get("entropy_weight", 0.005))
                loss = reason_loss + path_nll - entropy_weight * path_entropy
                if not _clip_and_step(
                    loss, optimizer, parameters, cfg, model=model,
                    stage="policy_warmup", epoch=epoch, batch_idx=batch_idx,
                    nonfinite_state=nonfinite_state,
                ):
                    continue
                losses.append(loss.item())
                with torch.no_grad():
                    predicted = model.sample_rollout(batch, deterministic=True)
                statistic_rows.append(
                    {
                        "target_stop_fraction": float((~useful).float().mean()),
                        "target_nonstop_fraction": float(useful.float().mean()),
                        "predicted_stop_fraction": float(predicted["length"].eq(0).float().mean()),
                        "useful_candidate_fraction": float(useful.float().mean()),
                        "mean_best_delta_score": float(best_gain.mean()),
                        "reason_bce": float(reason_parts["bce"]),
                        "path_nll": float(path_nll.detach()),
                        "policy_entropy": float(path_entropy.detach()),
                        "entropy_weight": entropy_weight,
                        "policy_logits_min": float(predicted["policy_logits_min"]),
                        "policy_logits_max": float(predicted["policy_logits_max"]),
                        "policy_state_norm_mean": float(predicted["policy_state_norm_mean"]),
                        "policy_state_norm_max": float(predicted["policy_state_norm_max"]),
                        "fallback_count": float(predicted["fallback_count"]),
                        "candidate_utility_std": candidate_diagnostics["candidate_utility_std"],
                    }
                )
                continue
            model.eval()
            with torch.no_grad():
                encoded = model.encode(batch)
                operators = model.build_operators(encoded, batch["modality_mask"])
                stop_actions = torch.full(
                    (batch["labels"].size(0), model.max_steps),
                    model.stop_idx,
                    dtype=torch.long,
                    device=device,
                )
                if balance_enabled:
                    path_actions = _sample_nonstop_paths(
                        model, batch, candidates_count
                    )
                    candidates = [
                        model.rollout_actions(
                            batch,
                            actions,
                            encoded=encoded,
                            operators=operators,
                        )
                        for actions in path_actions
                    ]
                    raw_gains = torch.stack(
                        [
                            reward_components(
                                model,
                                batch,
                                candidate,
                                cfg,
                                pos_weight=pos_weight,
                                compute_faithfulness=False,
                            )["raw_prediction_gain"]
                            for candidate in candidates
                        ],
                        1,
                    )
                    best_gain, best_index = raw_gains.max(1)
                    best_paths = torch.stack(
                        [
                            path_actions[candidate_index][sample_index]
                            for sample_index, candidate_index in enumerate(
                                best_index.tolist()
                            )
                        ],
                        0,
                    )
                    useful = (best_gain > utility_margin) & best_paths[
                        :, 0
                    ].ne(model.stop_idx)
                    batch, actions, _ = balance_policy_targets(
                        batch,
                        best_paths,
                        useful,
                        model.stop_idx,
                        target_fraction,
                    )
                    useful_fraction = useful.float().mean()
                    mean_best_gain = best_gain.mean()
                else:
                    candidates = [
                        model.rollout_actions(
                            batch,
                            stop_actions,
                            encoded=encoded,
                            operators=operators,
                        )
                    ]
                    for _ in range(max(0, candidates_count - 1)):
                        candidates.append(
                            model.sample_rollout(
                                batch, encoded=encoded, operators=operators
                            )
                        )
                    rewards = torch.stack(
                        [
                            reward_components(
                                model,
                                batch,
                                candidate,
                                cfg,
                                pos_weight=pos_weight,
                                compute_faithfulness=False,
                            )["total"]
                            for candidate in candidates
                        ],
                        1,
                    )
                    best_reward, best_candidate = rewards.max(1)
                    actions = torch.stack(
                        [
                            candidates[candidate_index]["actions"][sample_index]
                            for sample_index, candidate_index in enumerate(
                                best_candidate.tolist()
                            )
                        ],
                        0,
                    )
                    useful_fraction = actions[:, 0].ne(model.stop_idx).float().mean()
                    mean_best_gain = best_reward.mean()
            model.train()
            encoded = model.encode(batch)
            operators = model.build_operators(encoded, batch["modality_mask"])
            rollout = model.rollout_actions(
                batch, actions, encoded=encoded, operators=operators
            )
            target_nonstop = actions[:, 0].ne(model.stop_idx)
            if balance_enabled and target_nonstop.any() and (~target_nonstop).any():
                nonstop_fraction = target_nonstop.float().mean()
                sample_weights = torch.where(
                    target_nonstop,
                    0.5 / nonstop_fraction.clamp_min(1e-8),
                    0.5 / (1.0 - nonstop_fraction).clamp_min(1e-8),
                )
            else:
                sample_weights = torch.ones_like(rollout["logprob"])
            entropy_weight = (
                entropy_base * max(0.0, 1.0 - epoch / max(entropy_epochs, 1))
                if epoch < entropy_epochs else 0.0
            )
            loss = -(
                sample_weights * rollout["logprob"]
            ).mean() - entropy_weight * rollout["entropy"].mean()
            if not _clip_and_step(
                loss, optimizer, parameters, cfg, model=model,
                stage="policy_warmup", epoch=epoch, batch_idx=batch_idx,
                nonfinite_state=nonfinite_state,
            ):
                continue
            losses.append(loss.item())
            with torch.no_grad():
                predicted = model.sample_rollout(batch, deterministic=True)
            statistic_rows.append(
                {
                    "target_stop_fraction": float((~target_nonstop).float().mean()),
                    "target_nonstop_fraction": float(target_nonstop.float().mean()),
                    "predicted_stop_fraction": float(predicted["length"].eq(0).float().mean()),
                    "useful_candidate_fraction": float(useful_fraction),
                    "mean_best_delta_score": float(mean_best_gain),
                    "policy_entropy": float(rollout["entropy"].mean().detach()),
                    "entropy_weight": float(entropy_weight),
                    "policy_logits_min": float(predicted["policy_logits_min"]),
                    "policy_logits_max": float(predicted["policy_logits_max"]),
                    "policy_state_norm_mean": float(predicted["policy_state_norm_mean"]),
                    "policy_state_norm_max": float(predicted["policy_state_norm_max"]),
                    "fallback_count": float(predicted["fallback_count"]),
                }
            )
        metrics = evaluate_model(
            model,
            val_loader,
            cfg,
            device,
            deterministic=True,
            split="val",
            pos_weight=pos_weight,
            stage="policy_warmup",
            epoch=epoch,
        )
        policy_statistics = _mean_diagnostics(statistic_rows)
        metrics.update(
            {f"train_{key}": value for key, value in policy_statistics.items()}
        )
        print(
            f"[policy_warmup] epoch={epoch} nll={np.mean(losses):.4f} "
            f"policy={policy_statistics} metrics={metrics}"
        )
        _save_checkpoint(output_dir / "last_policy_warmup.pt", model, optimizer, cfg, "policy_warmup", epoch, metrics, pos_weight)
        _save_last_finite(
            output_dir, model, optimizer, cfg, "policy_warmup", epoch,
            metrics, pos_weight, loss_is_finite=bool(losses),
            skipped_batches=int(nonfinite_state.get("skipped", 0)),
        )
        _save_checkpoint(output_dir / f"policy_epoch_{epoch}.pt", model, optimizer, cfg, "policy_warmup", epoch, metrics, pos_weight)
        if metrics.get("stop_rate", 0.0) > 0.95:
            print("[collapse-warning] policy STOP collapse")
    selected_path = select_policy_checkpoint(output_dir, cfg)
    if selected_path is not None and getattr(model, "performance_v5", False):
        try:
            selected_state = torch.load(
                selected_path, map_location=device, weights_only=False
            )
        except TypeError:
            selected_state = torch.load(selected_path, map_location=device)
        model.load_state_dict(selected_state["model"])
    select_reason_threshold_from_validation(
        model, val_loader, cfg, device, output_dir, pos_weight,
        stage="policy_warmup"
    )


def _repeat_batch(batch, group_size):
    output = {
        "modalities": {},
        "meta": sum(([item] * group_size for item in batch["meta"]), []),
    }
    for modality, value in batch["modalities"].items():
        if isinstance(value, dict):
            output["modalities"][modality] = {
                key: tensor.repeat_interleave(group_size, 0)
                for key, tensor in value.items()
            }
        elif torch.is_tensor(value):
            output["modalities"][modality] = value.repeat_interleave(group_size, 0)
        else:
            output["modalities"][modality] = sum(
                ([item] * group_size for item in value), []
            )
    for key in ["modality_mask", "labels", "label_mask"]:
        output[key] = batch[key].repeat_interleave(group_size, 0)
    return output


def select_top_v5_checkpoints(output_dir, cfg, limit=5):
    """Select validation-only candidates from supervised, policy, and GRPO."""
    output_dir = Path(output_dir)
    direct_path = output_dir / "best_direct.pt"
    if not direct_path.exists():
        return []
    try:
        direct_state = torch.load(direct_path, map_location="cpu", weights_only=False)
    except TypeError:
        direct_state = torch.load(direct_path, map_location="cpu")
    task_name = cfg["experiment"]["task_names"][0]
    direct_metrics = direct_state.get("metrics", {})
    direct_auroc = float(direct_metrics[f"direct_auroc_{task_name}"])
    direct_auprc = float(direct_metrics[f"direct_auprc_{task_name}"])
    direct_score = float(direct_metrics[f"direct_dual_score_{task_name}"])
    selection_cfg = cfg.get("selection", {})
    rows = []
    paths = set()
    for pattern in ["supervised_epoch_*.pt", "policy_epoch_*.pt", "grpo_epoch_*.pt"]:
        for path in output_dir.glob(pattern):
            if path in paths:
                continue
            paths.add(path)
            try:
                state = torch.load(path, map_location="cpu", weights_only=False)
            except TypeError:
                state = torch.load(path, map_location="cpu")
            metrics = state.get("metrics", {})
            auroc = metrics.get(f"final_auroc_{task_name}")
            auprc = metrics.get(f"final_auprc_{task_name}")
            score = metrics.get(f"final_dual_score_{task_name}")
            if auroc is None or auprc is None or score is None:
                continue
            if not checkpoint_passes_direct_guard(
                auroc,
                auprc,
                direct_auroc,
                direct_auprc,
                float(selection_cfg.get("tolerance_auroc", 0.001)),
                float(selection_cfg.get("tolerance_auprc", 0.002)),
            ):
                continue
            rows.append(
                {
                    "path": str(path),
                    "checkpoint": path.name,
                    "stage": state.get("stage"),
                    "epoch": int(state.get("epoch", -1)),
                    "auroc": float(auroc),
                    "auprc": float(auprc),
                    "score": float(score),
                    "nll": float(metrics.get(f"final_nll_{task_name}", float("inf"))),
                    "both_improved": bool(
                        float(auroc) >= direct_auroc
                        and float(auprc) >= direct_auprc
                    ),
                    "score_improved": bool(float(score) > direct_score),
                }
            )
    rows.sort(
        key=lambda row: (int(row["both_improved"]), *metric_sort_key(row)),
        reverse=True,
    )
    selected = rows[: int(limit)]
    (output_dir / "top5_checkpoints_v5.json").write_text(
        json.dumps({"fit_split": "val", "checkpoints": selected}, indent=2, sort_keys=True)
        + "\n"
    )
    improved = [row for row in selected if row["score_improved"]]
    if improved:
        best_row = improved[0]
        is_v6 = bool(
            cfg.get("model", {}).get("performance_v6", {}).get("enabled", False)
        )
        if not is_v6:
            shutil.copyfile(best_row["path"], output_dir / "best_final_v5.pt")
        elif best_row["both_improved"]:
            shutil.copyfile(best_row["path"], output_dir / "best_final_v6.pt")
        if not best_row["both_improved"]:
            print(
                "[performance-warning] learned routing still below required gain"
            )
        print(
            "[final-selection] "
            f"checkpoint={best_row['checkpoint']} "
            f"validation_auroc={best_row['auroc']:.6f} "
            f"validation_auprc={best_row['auprc']:.6f} "
            f"validation_score={best_row['score']:.6f}"
        )
    else:
        print(
            "[performance-guard] no guarded checkpoint improved Direct dual "
            "score; best_final_v5.pt was not created"
        )
    return selected


@torch.no_grad()
def collect_final_logits(model, loader, device, stage):
    logits, labels, masks = [], [], []
    for batch in loader:
        batch = move_batch(batch, device)
        if stage in {"policy_warmup", "grpo"}:
            outputs = model.sample_rollout(batch, deterministic=True)
        else:
            outputs = supervised_round_robin(model, batch)
        logits.append(outputs["logits"].cpu())
        labels.append(batch["labels"].cpu())
        masks.append(batch["label_mask"].cpu())
    return torch.cat(logits), torch.cat(labels), torch.cat(masks)


def fit_v5_ensemble_on_validation(model, val_loader, cfg, device, output_dir):
    split = getattr(val_loader.dataset, "split", "val")
    if split != "val":
        raise ValueError("ensemble fitting is validation-only")
    selected = select_top_v5_checkpoints(output_dir, cfg, limit=5)
    if len(selected) < 3:
        print("[ensemble-warning] fewer than three guarded validation checkpoints")
        return None
    chosen = selected[:3]
    logits_rows = []
    labels = masks = None
    for row in chosen:
        try:
            state = torch.load(row["path"], map_location=device, weights_only=False)
        except TypeError:
            state = torch.load(row["path"], map_location=device)
        model.load_state_dict(state["model"])
        load_saved_correction_bound(model, output_dir)
        load_saved_reason_threshold(model, output_dir)
        logits, labels, masks = collect_final_logits(
            model, val_loader, device, state.get("stage", "grpo")
        )
        logits_rows.append(logits)
    task_index = 0
    valid = masks[:, task_index] > 0
    selected_weights, candidates = select_ensemble_weights(
        torch.stack(logits_rows, 0)[:, valid, task_index].numpy(),
        labels[valid, task_index].numpy(),
        split="val",
        weight_candidates=cfg.get("selection", {}).get(
            "ensemble_weights", DEFAULT_ENSEMBLE_WEIGHTS
        ),
    )
    payload = {
        "fit_split": "val",
        "checkpoints": [row["checkpoint"] for row in chosen],
        "selected": selected_weights,
        "candidates": candidates,
    }
    (Path(output_dir) / "ensemble_weights_v5.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    print(
        f"[ensemble-selection] weights={selected_weights['weights']} "
        f"validation_score={selected_weights['score']:.6f}"
    )
    return payload


def train_grpo(model, train_loader, val_loader, cfg, device, output_dir,
               pos_weight=None, resume_state=None):
    if pos_weight is None:
        pos_weight, _ = compute_train_pos_weight(train_loader, device)
    configure_trainable_parameters(model, cfg, "grpo")
    optimizer = make_optimizer(model, cfg, "grpo")
    parameters = [p for p in model.parameters() if p.requires_grad]
    start_epoch = _resume_optimizer(optimizer, resume_state, "grpo")
    reference_policy = None
    beta_kl = float(cfg.get("grpo", {}).get("kl_anchor_weight", 0.05))
    previous_validation_score = None
    if _is_v6(model):
        reference_policy = copy.deepcopy(model.policy).to(device).eval()
        if resume_state and "pi_ref_policy" in resume_state:
            reference_policy.load_state_dict(resume_state["pi_ref_policy"])
        for parameter in reference_policy.parameters():
            parameter.requires_grad = False
        if resume_state:
            beta_kl = float(resume_state.get("beta_kl", beta_kl))
            previous_validation_score = resume_state.get(
                "previous_validation_score"
            )
    if getattr(model, "performance_v5", False) and start_epoch == 0:
        for name in [
            "best.pt",
            "best_unconfirmed.pt",
            "best_final_v5.pt",
            "best_final_v6.pt",
            "top5_checkpoints_v5.json",
            "ensemble_weights_v5.json",
        ]:
            (Path(output_dir) / name).unlink(missing_ok=True)
    group_size = int(cfg.get("grpo", {}).get("group_size", cfg.get("rl", {}).get("group_size", 8)))
    rl_cfg = cfg.get("grpo", cfg.get("rl", {}))
    entropy_weight = float(rl_cfg.get("entropy_weight", 0.005))
    if resume_state:
        entropy_weight = float(
            resume_state.get("adaptive_entropy_weight", entropy_weight)
        )
    best = unconfirmed_best = float("-inf")
    best_category = -1
    task_name = cfg["experiment"]["task_names"][0]
    direct_reference = None
    direct_path = Path(output_dir) / "best_direct.pt"
    if getattr(model, "performance_v5", False) and direct_path.exists():
        try:
            direct_state = torch.load(direct_path, map_location="cpu", weights_only=False)
        except TypeError:
            direct_state = torch.load(direct_path, map_location="cpu")
        direct_metrics = direct_state.get("metrics", {})
        direct_reference = {
            "auroc": float(direct_metrics[f"direct_auroc_{task_name}"]),
            "auprc": float(direct_metrics[f"direct_auprc_{task_name}"]),
            "score": float(direct_metrics[f"direct_dual_score_{task_name}"]),
        }
    stop_collapse_epochs = 0
    nonfinite_state = {}
    for epoch in range(start_epoch, int(cfg["training"]["epochs_grpo"])):
        model.train()
        model.reset_numerics_observations()
        losses, reward_rows = [], []
        for batch_idx, original_batch in enumerate(train_loader):
            original_batch = move_batch(original_batch, device)
            original_batch = apply_modality_dropout(original_batch, cfg)
            model.set_numerics_context("grpo", epoch, batch_idx)
            batch = _repeat_batch(original_batch, group_size)
            rollout = model.sample_rollout(batch)
            reward_cfg = cfg.get("reward", cfg.get("rl", {}).get("reward", {}))
            faithfulness_probability = float(
                rl_cfg.get("faithfulness_probability", cfg.get("rl", {}).get("faithfulness_probability", 0.0))
            )
            faithfulness = random.random() < faithfulness_probability and float(
                reward_cfg.get("faithfulness", 0.0)
            ) > 0
            with torch.no_grad():
                components = reward_components(
                    model,
                    batch,
                    rollout,
                    cfg,
                    pos_weight=pos_weight,
                    compute_faithfulness=faithfulness,
                )
                advantage = group_advantages(components["total"], group_size)
                old_logprob = rollout["logprob"].detach()
                actions = rollout["actions"].detach()
                stop = rollout["length"].eq(0)
                reward_rows.append(
                    {
                        "mean_raw_prediction_gain": float(components["raw_prediction_gain"].mean()),
                        "std_raw_prediction_gain": float(components["raw_prediction_gain"].std(unbiased=False)),
                        "mean_scaled_prediction_gain": float(components["scaled_prediction_gain"].mean()),
                        "mean_order_contribution": float(components["order_contribution"].mean()),
                        "mean_length_penalty": float(components["length_penalty"].mean()),
                        "mean_total_reward": float(components["total"].mean()),
                        "mean_stop_reward": float(components["total"][stop].mean()) if stop.any() else float("nan"),
                        "mean_nonstop_reward": float(components["total"][~stop].mean()) if (~stop).any() else float("nan"),
                        "mean_abs_raw_prediction_gain": float(components["raw_prediction_gain"].abs().mean()),
                        "mean_reranker_consistency": float(
                            components.get(
                                "reranker_consistency",
                                torch.zeros_like(components["total"]),
                            ).mean()
                        ),
                    }
                )
            for _ in range(int(rl_cfg.get("grpo_epochs_per_batch", 1))):
                encoded = model.encode(batch)
                operators = model.build_operators(encoded, batch["modality_mask"])
                current = model.rollout_actions(
                    batch, actions, encoded=encoded, operators=operators
                )
                if reference_policy is not None:
                    with torch.no_grad():
                        reference = model.rollout_actions(
                            batch,
                            actions,
                            encoded=encoded,
                            operators=operators,
                            policy_module=reference_policy,
                        )
                    kl_anchor = categorical_policy_kl(
                        current["policy_logits_steps"],
                        reference["policy_logits_steps"],
                    )
                    reranker_prediction = model.score_candidate_rollout(
                        current, batch["modality_mask"]
                    )
                    reranker_refinement = torch.nn.functional.smooth_l1_loss(
                        reranker_prediction.float(),
                        components["total"].detach().float(),
                    )
                else:
                    kl_anchor = current["logprob"].sum() * 0.0
                    reranker_refinement = kl_anchor
                policy_loss = clipped_grpo_loss(
                    current["logprob"],
                    old_logprob,
                    advantage,
                    current["entropy"],
                    float(rl_cfg.get("clip_eps", 0.2)),
                    entropy_weight,
                )
                anchor = per_sample_masked_bce(
                    current["logits"],
                    batch["labels"],
                    batch["label_mask"],
                    pos_weight,
                ).mean()
                total = (
                    policy_loss
                    + float(cfg["training"].get("grpo_supervised_anchor", 0.25)) * anchor
                    + beta_kl * kl_anchor
                    + 0.10 * reranker_refinement
                    + operator_regularization(model, operators, cfg)
                )
                if not _clip_and_step(
                    total, optimizer, parameters, cfg, model=model,
                    stage="grpo", epoch=epoch, batch_idx=batch_idx,
                    nonfinite_state=nonfinite_state,
                ):
                    continue
                losses.append(total.item())
        metrics = evaluate_model(
            model,
            val_loader,
            cfg,
            device,
            deterministic=True,
            split="val",
            pos_weight=pos_weight,
            stage="grpo",
            epoch=epoch,
        )
        reward_statistics = _mean_diagnostics(reward_rows)
        metrics["adaptive_entropy_weight"] = entropy_weight
        score = _primary_score(metrics, cfg)
        if (
            _is_v6(model)
            and previous_validation_score is not None
            and score < float(previous_validation_score)
        ):
            beta_kl = max(beta_kl, 0.10)
            print(f"[grpo-kl-anchor] validation declined; beta_kl={beta_kl:.3f}")
        previous_validation_score = score
        metrics["grpo_beta_kl"] = beta_kl
        print(
            f"[grpo] epoch={epoch} loss={np.mean(losses):.4f} "
            f"reward={reward_statistics} metrics={metrics}"
        )
        if _is_v6(model):
            delta_auroc = float(
                metrics.get(
                    f"delta_auroc_final_minus_direct_{task_name}", float("-inf")
                )
            )
            delta_auprc = float(
                metrics.get(
                    f"delta_auprc_final_minus_direct_{task_name}", float("-inf")
                )
            )
            if delta_auroc < 0.003 or delta_auprc < 0.005:
                print(
                    "[performance-warning] learned routing still below required gain "
                    f"delta_auroc={delta_auroc:.6f} "
                    f"delta_auprc={delta_auprc:.6f}"
                )
        print(
            "[policy-numerics] "
            f"policy_logits_min={metrics.get('policy_logits_min')} "
            f"policy_logits_max={metrics.get('policy_logits_max')} "
            f"policy_state_norm_mean={metrics.get('policy_state_norm_mean')} "
            f"policy_state_norm_max={metrics.get('policy_state_norm_max')} "
            f"fallback_count={metrics.get('fallback_count', 0)} "
            f"skipped_batches={nonfinite_state.get('skipped', 0)}"
        )
        if reward_statistics and (
            reward_statistics["mean_abs_raw_prediction_gain"]
            < reward_statistics["mean_length_penalty"]
        ):
            print(
                "[collapse-warning] mean |raw R_pred| is smaller than "
                "mean length penalty"
            )
        stop_collapse_epochs = (
            stop_collapse_epochs + 1
            if metrics.get("stop_rate", 0.0) > 0.98 else 0
        )
        if stop_collapse_epochs >= 2:
            entropy_weight = min(entropy_weight * 2.0, 0.02)
            print(
                "[collapse-warning] GRPO STOP collapse; "
                f"entropy_weight={entropy_weight:.6f}"
            )
        _save_checkpoint(
            output_dir / "last.pt",
            model,
            optimizer,
            cfg,
            "grpo",
            epoch,
            metrics,
            pos_weight,
            adaptive_entropy_weight=entropy_weight,
            beta_kl=beta_kl,
            previous_validation_score=previous_validation_score,
            pi_ref_policy=(
                reference_policy.state_dict() if reference_policy is not None else None
            ),
        )
        if getattr(model, "performance_v5", False):
            _save_checkpoint(
                output_dir / f"grpo_epoch_{epoch}.pt",
                model, optimizer, cfg, "grpo", epoch, metrics, pos_weight,
                adaptive_entropy_weight=entropy_weight,
            )
        _save_last_finite(
            output_dir, model, optimizer, cfg, "grpo", epoch, metrics,
            pos_weight, loss_is_finite=bool(losses),
            adaptive_entropy_weight=entropy_weight,
            skipped_batches=int(nonfinite_state.get("skipped", 0)),
        )
        save_main = _should_save_best(score, best, epoch, start_epoch)
        if getattr(model, "performance_v5", False) and direct_reference is not None:
            final_row = {
                "auroc": metrics[f"final_auroc_{task_name}"],
                "auprc": metrics[f"final_auprc_{task_name}"],
                "score": metrics[f"final_dual_score_{task_name}"],
            }
            guard = performance_guard(final_row, direct_reference)
            direct_guard_passed = _passes_v5_direct_guard(
                metrics, cfg, output_dir
            )
            category = 1 if guard.both_improved else 0
            save_main = guard.save and direct_guard_passed and (
                category > best_category
                or (category == best_category and score > best)
            )
            if guard.tradeoff_warning:
                print(
                    "[tradeoff-warning] validation dual score improved but "
                    "AUROC/AUPRC did not both improve"
                )
            if (not guard.save or not direct_guard_passed) and score > unconfirmed_best:
                unconfirmed_best = score
                _save_checkpoint(
                    output_dir / "best_unconfirmed.pt", model, optimizer, cfg,
                    "grpo", epoch, metrics, pos_weight,
                    adaptive_entropy_weight=entropy_weight,
                    performance_guard_passed=False,
                )
            if save_main:
                best_category = category
        if save_main:
            best = score
            _save_checkpoint(
                output_dir / "best.pt",
                model,
                optimizer,
                cfg,
                "grpo",
                epoch,
                metrics,
                pos_weight,
                adaptive_entropy_weight=entropy_weight,
                performance_guard_passed=True,
            )
            _print_best_validation("grpo", metrics, cfg)

    if getattr(model, "performance_v5", False):
        if bool(cfg.get("ensemble", {}).get("enabled", True)):
            fit_v5_ensemble_on_validation(
                model, val_loader, cfg, device, output_dir
            )
        elif _is_v6(model):
            select_top_v5_checkpoints(output_dir, cfg, limit=5)
            print("[ensemble] disabled for v6 single-model optimization")
        selected_path = Path(output_dir) / (
            "best_final_v6.pt" if _is_v6(model) else "best_final_v5.pt"
        )
        calibration_is_unconfirmed = False
        if not selected_path.exists():
            selected_path = Path(output_dir) / "best_unconfirmed.pt"
            calibration_is_unconfirmed = True
        if selected_path.exists():
            try:
                selected_state = torch.load(
                    selected_path, map_location=device, weights_only=False
                )
            except TypeError:
                selected_state = torch.load(selected_path, map_location=device)
            model.load_state_dict(selected_state["model"])
            load_saved_correction_bound(model, output_dir)
            load_saved_reason_threshold(model, output_dir)
            final_temperature = fit_and_save_final_temperature(
                model, val_loader, device, output_dir, stage="grpo"
            )
            selected_state["model"] = model.state_dict()
            selected_state["final_temperature"] = final_temperature
            selected_state["performance_guard_passed"] = bool(
                not calibration_is_unconfirmed
            )
            torch.save(selected_state, selected_path)
        else:
            print(
                "[performance-guard] final temperature was not fitted because "
                "no final candidate checkpoint exists"
            )


def direct_initialization_error(model, loader, device):
    model.eval()
    with torch.no_grad():
        batch = move_batch(next(iter(loader)), device)
        outputs = supervised_round_robin(model, batch)
        return float((outputs["logits"] - outputs["direct_logits"]).abs().max().cpu())


def _binary_metrics(labels, probabilities):
    if len(np.unique(labels)) < 2:
        return float("nan"), float("nan")
    return (
        float(roc_auc_score(labels, probabilities)),
        float(average_precision_score(labels, probabilities)),
    )


@torch.no_grad()
def evaluate_model(model, loader, cfg, device, deterministic=True, split=None,
                   pos_weight=None, stage=None, epoch=None,
                   include_oracle=False):
    """Evaluate predictions; optional label-aware oracle output is diagnostic only."""
    model.eval()
    evaluation_stage = stage or "grpo"
    uses_policy = evaluation_stage in {"policy_warmup", "grpo"}
    if (
        evaluation_stage == "operator_warmup"
        and bool(cfg.get("evaluation", {}).get(
            "rollout_during_operator_warmup", False
        ))
    ):
        uses_policy = True
    labels, masks = [], []
    base_logits_values, strong_logits_values = [], []
    pair_logits_values, pool_logits_values = [], []
    direct_logits_values, final_logits_values = [], []
    alpha_values, delta_values, raw_corrections, bounded_corrections = [], [], [], []
    reason_probabilities = []
    oracle_logits_values = []
    oracle_at_k_logits_values = {1: [], 3: [], 5: []}
    oracle_agreement_values = []
    oracle_diagnostic_rows = []
    policy_ranking_rows = []
    reranker_ranking_rows = []
    selected_utility_regrets = []
    lengths, policy_entropies, policy_logprobs = [], [], []
    action_counts = torch.zeros(model.num_actions, dtype=torch.long)
    reward_rows = []
    for batch_idx, batch in enumerate(loader):
        batch = move_batch(batch, device)
        model.set_numerics_context(
            stage=evaluation_stage, epoch=epoch, batch_idx=batch_idx
        )
        if uses_policy:
            rollout = model.sample_rollout(
                batch, deterministic=deterministic
            )
        elif evaluation_stage == "direct":
            encoded = model.encode(batch)
            direct = model.direct_outputs(encoded, batch["modality_mask"])
            model._check("direct_hidden", direct["direct_hidden"])
            model._check("direct_logits", direct["direct_logits"])
            zeros = torch.zeros_like(direct["direct_logits"])
            rollout = dict(direct)
            rollout.update(
                {
                    "logits": direct["direct_logits"],
                    "operator_gate": zeros[:, :1],
                    "operator_delta_logits": zeros,
                }
            )
        else:
            fixed_alpha = (
                cfg["training"].get("operator_warmup", {}).get("fixed_alpha")
                if evaluation_stage == "operator_warmup" else None
            )
            rollout = supervised_round_robin(
                model, batch, fixed_alpha=fixed_alpha
            )
        labels.append(batch["labels"].cpu())
        masks.append(batch["label_mask"].cpu())
        base_logits_values.append(rollout["direct_base_logits"].cpu())
        strong_logits_values.append(
            rollout.get("strong_direct_logits", rollout["direct_base_logits"]).cpu()
        )
        pair_logits_values.append(
            rollout.get("direct_pair_logits", rollout["direct_logits"]).cpu()
        )
        pool_logits_values.append(
            rollout.get("direct_pool_logits", rollout["direct_logits"]).cpu()
        )
        direct_logits_values.append(rollout["direct_logits"].cpu())
        final_logits_values.append(rollout["logits"].cpu())
        alpha_values.append(rollout["operator_gate"].cpu())
        delta_values.append(rollout["operator_delta_logits"].cpu())
        raw_corrections.append(
            rollout.get(
                "raw_correction",
                rollout["operator_gate"] * rollout["operator_delta_logits"],
            ).cpu()
        )
        bounded_corrections.append(
            rollout.get(
                "bounded_correction",
                rollout["operator_gate"] * rollout["operator_delta_logits"],
            ).cpu()
        )
        reason_probabilities.append(
            rollout.get(
                "reason_probability", torch.ones_like(rollout["operator_gate"])
            ).cpu()
        )
        if include_oracle and getattr(model, "performance_v5", False):
            oracle_pos_weight = pos_weight
            if oracle_pos_weight is None:
                oracle_pos_weight = torch.ones(
                    batch["labels"].size(1), device=batch["labels"].device
                )
            if _is_v6(model):
                routing_cfg = model.performance_v6_cfg.get(
                    "routing", cfg.get("routing", {})
                )
                oracle_set = build_oracle_candidate_set(
                    model,
                    batch,
                    oracle_pos_weight,
                    candidate_count=int(routing_cfg.get("num_candidates", 24)),
                )
                utility = oracle_set["utility"]
                policy_scores = oracle_set["policy_scores"]
                best_index = utility.argmax(1)
                oracle_actions = torch.stack(
                    [
                        oracle_set["paths"][index][sample]
                        for sample, index in enumerate(best_index.tolist())
                    ],
                    0,
                )
                task_width = oracle_set["candidate_logits"].size(-1)
                gather = best_index[:, None, None].expand(-1, 1, task_width)
                oracle_logits_values.append(
                    oracle_set["candidate_logits"].gather(1, gather).squeeze(1).cpu()
                )
                policy_diag = routing_ranking_metrics(
                    policy_scores.cpu().numpy(), utility.cpu().numpy()
                )
                policy_ranking_rows.append(policy_diag)
                reranker_scores = candidate_reranker_scores(
                    model,
                    batch,
                    oracle_set["paths"],
                    oracle_set["encoded"],
                    oracle_set["operators"],
                )
                policy_order = policy_scores.argsort(1, descending=True)
                for width in (1, 3, 5):
                    top = policy_order[:, : min(width, policy_order.size(1))]
                    top_utility = utility.gather(1, top)
                    local = top_utility.argmax(1)
                    chosen = top.gather(1, local[:, None]).squeeze(1)
                    chosen_gather = chosen[:, None, None].expand(
                        -1, 1, task_width
                    )
                    oracle_at_k_logits_values[width].append(
                        oracle_set["candidate_logits"]
                        .gather(1, chosen_gather)
                        .squeeze(1)
                        .cpu()
                    )
                topk_width = min(
                    int(routing_cfg.get("policy_topk", 5)),
                    policy_order.size(1),
                )
                allowed = torch.zeros_like(reranker_scores, dtype=torch.bool)
                allowed.scatter_(1, policy_order[:, :topk_width], True)
                restricted_reranker = reranker_scores.masked_fill(
                    ~allowed, -torch.inf
                )
                reranker_diag = routing_ranking_metrics(
                    restricted_reranker.cpu().numpy(), utility.cpu().numpy()
                )
                reranker_ranking_rows.append(reranker_diag)
                selected_index = restricted_reranker.argmax(1)
                selected_regret = (
                    utility.max(1).values
                    - utility.gather(1, selected_index[:, None]).squeeze(1)
                )
                selected_utility_regrets.append(selected_regret.cpu())
                oracle_diagnostics = {
                    "candidate_utility_std": float(
                        utility.std(unbiased=False)
                    ),
                    "candidate_modality_coverage": [],
                    "best_path_modality_distribution": [],
                }
            else:
                oracle_actions, _, _, oracle_diagnostics = select_oracle_candidate(
                    model,
                    batch,
                    oracle_pos_weight,
                    candidate_count=int(
                        cfg["training"].get("oracle_candidates", 12)
                    ),
                )
                oracle_rollout = model.rollout_actions(batch, oracle_actions)
                oracle_logits_values.append(oracle_rollout["logits"].cpu())
            oracle_diagnostic_rows.append(oracle_diagnostics)
            if uses_policy and "actions" in rollout:
                oracle_agreement_values.append(
                    rollout["actions"].eq(oracle_actions).all(1).float().cpu()
                )
        if uses_policy:
            lengths.append(rollout["length"].cpu())
            policy_entropies.append(rollout["entropy"].cpu())
            policy_logprobs.append(rollout["logprob"].cpu())
            valid_actions = rollout["actions"].detach().cpu().reshape(-1)
            valid_actions = valid_actions[valid_actions < model.num_actions]
            action_counts += torch.bincount(
                valid_actions, minlength=model.num_actions
            )[:model.num_actions]
            components = reward_components(
                model,
                batch,
                rollout,
                cfg,
                pos_weight=pos_weight,
                compute_faithfulness=False,
            )
            stop = rollout["length"].eq(0)
            reward_rows.append(
                {
                    "raw_prediction_gain": float(components["raw_prediction_gain"].mean()),
                    "std_raw_prediction_gain": float(
                        components["raw_prediction_gain"].std(unbiased=False)
                    ),
                    "scaled_prediction_gain": float(components["scaled_prediction_gain"].mean()),
                    "order_contribution": float(components["order_contribution"].mean()),
                    "length_penalty": float(components["length_penalty"].mean()),
                    "total_reward": float(components["total"].mean()),
                    "stop_reward": float(components["total"][stop].mean()) if stop.any() else float("nan"),
                    "nonstop_reward": float(components["total"][~stop].mean()) if (~stop).any() else float("nan"),
                }
            )

    labels_np = torch.cat(labels).numpy()
    masks_np = torch.cat(masks).numpy()
    base_logits_np = torch.cat(base_logits_values).numpy()
    strong_logits_np = torch.cat(strong_logits_values).numpy()
    pair_logits_np = torch.cat(pair_logits_values).numpy()
    pool_logits_np = torch.cat(pool_logits_values).numpy()
    direct_logits_np = torch.cat(direct_logits_values).numpy()
    final_logits_np = torch.cat(final_logits_values).numpy()
    oracle_logits_np = (
        torch.cat(oracle_logits_values).numpy()
        if oracle_logits_values else None
    )
    oracle_at_k_logits_np = {
        width: torch.cat(rows).numpy()
        for width, rows in oracle_at_k_logits_values.items()
        if rows
    }
    temperature = float(model.direct_temperature.detach().cpu())
    if not np.isfinite(temperature):
        raise FloatingPointError("direct_temperature is non-finite during evaluation")
    temperature = float(np.clip(temperature, 1e-3, 100.0))
    def sigmoid_array(values):
        values = np.clip(values, -80.0, 80.0)
        return 1.0 / (1.0 + np.exp(-values))

    base_uncalibrated_np = sigmoid_array(base_logits_np)
    direct_uncalibrated_np = sigmoid_array(direct_logits_np)
    final_uncalibrated_np = sigmoid_array(final_logits_np)
    base_np = sigmoid_array(np.clip(base_logits_np / temperature, -30.0, 30.0))
    strong_np = sigmoid_array(
        np.clip(strong_logits_np / temperature, -30.0, 30.0)
    )
    direct_np = sigmoid_array(
        np.clip(direct_logits_np / temperature, -30.0, 30.0)
    )
    final_temperature = float(model.final_temperature.detach().cpu())
    if not np.isfinite(final_temperature):
        raise FloatingPointError("final_temperature is non-finite during evaluation")
    final_temperature = float(np.clip(final_temperature, 1e-3, 100.0))
    pair_np = sigmoid_array(np.clip(pair_logits_np / temperature, -30.0, 30.0))
    pool_np = sigmoid_array(np.clip(pool_logits_np / temperature, -30.0, 30.0))
    final_np = sigmoid_array(
        np.clip(final_logits_np / final_temperature, -30.0, 30.0)
    )
    alpha_np = torch.cat(alpha_values).numpy().reshape(-1)
    delta_tensor = torch.cat(delta_values)
    raw_correction_tensor = torch.cat(raw_corrections)
    bounded_correction_tensor = torch.cat(bounded_corrections)
    reason_probability_np = torch.cat(reason_probabilities).numpy().reshape(-1)
    metrics = {
        "split": split or getattr(loader.dataset, "split", "unknown"),
        "evaluation_stage": evaluation_stage,
        "direct_temperature": temperature,
        "final_temperature": final_temperature,
    }
    for task_index, task_name in enumerate(cfg["experiment"]["task_names"]):
        valid = masks_np[:, task_index] > 0.5
        task_labels = labels_np[valid, task_index]
        base_auroc, base_auprc = _binary_metrics(
            task_labels, base_np[valid, task_index]
        )
        strong_auroc, strong_auprc = _binary_metrics(
            task_labels, strong_np[valid, task_index]
        )
        direct_auroc, direct_auprc = _binary_metrics(
            task_labels, direct_np[valid, task_index]
        )
        final_auroc, final_auprc = _binary_metrics(
            task_labels, final_np[valid, task_index]
        )
        positive = int((task_labels > 0.5).sum())
        prevalence = float(positive / valid.sum()) if valid.sum() else float("nan")
        pair_auroc, pair_auprc = _binary_metrics(
            task_labels, pair_np[valid, task_index]
        )
        pool_auroc, pool_auprc = _binary_metrics(
            task_labels, pool_np[valid, task_index]
        )
        metrics[f"n_valid_{task_name}"] = int(valid.sum())
        metrics[f"n_positive_{task_name}"] = positive
        metrics[f"prevalence_{task_name}"] = prevalence
        metrics[f"base_auroc_{task_name}"] = base_auroc
        metrics[f"base_auprc_{task_name}"] = base_auprc
        metrics[f"strong_direct_auroc_{task_name}"] = strong_auroc
        metrics[f"strong_direct_auprc_{task_name}"] = strong_auprc
        metrics[f"direct_auroc_{task_name}"] = direct_auroc
        metrics[f"direct_auprc_{task_name}"] = direct_auprc
        metrics[f"pairwise_auroc_{task_name}"] = pair_auroc
        metrics[f"pairwise_auprc_{task_name}"] = pair_auprc
        metrics[f"pooled_auroc_{task_name}"] = pool_auroc
        metrics[f"pooled_auprc_{task_name}"] = pool_auprc
        metrics[f"final_auroc_{task_name}"] = final_auroc
        metrics[f"final_auprc_{task_name}"] = final_auprc
        metrics[f"auroc_{task_name}"] = final_auroc
        metrics[f"auprc_{task_name}"] = final_auprc
        metrics[f"delta_auroc_final_minus_direct_{task_name}"] = final_auroc - direct_auroc
        metrics[f"delta_auprc_final_minus_direct_{task_name}"] = final_auprc - direct_auprc
        metrics[f"delta_auroc_direct_minus_base_{task_name}"] = direct_auroc - base_auroc
        metrics[f"delta_auprc_direct_minus_base_{task_name}"] = direct_auprc - base_auprc
        for prefix, auroc, auprc in [
            ("base", base_auroc, base_auprc),
            ("strong_direct", strong_auroc, strong_auprc),
            ("pairwise", pair_auroc, pair_auprc),
            ("pooled", pool_auroc, pool_auprc),
            ("direct", direct_auroc, direct_auprc),
            ("final", final_auroc, final_auprc),
        ]:
            metrics[f"{prefix}_normalized_auprc_{task_name}"] = normalized_auprc(
                auprc, prevalence
            )
            metrics[f"{prefix}_dual_score_{task_name}"] = dual_metric_score(
                auroc, auprc, prevalence
            )
        if oracle_logits_np is not None:
            oracle_probability = sigmoid_array(
                oracle_logits_np[valid, task_index]
            )
            oracle_auroc, oracle_auprc = _binary_metrics(
                task_labels, oracle_probability
            )
            metrics[f"oracle_auroc_{task_name}"] = oracle_auroc
            metrics[f"oracle_auprc_{task_name}"] = oracle_auprc
            metrics[f"learned_path_auroc_{task_name}"] = final_auroc
            metrics[f"learned_path_auprc_{task_name}"] = final_auprc
            metrics[f"oracle_minus_learned_auroc_{task_name}"] = (
                oracle_auroc - final_auroc
            )
            metrics[f"oracle_minus_learned_auprc_{task_name}"] = (
                oracle_auprc - final_auprc
            )
            for width, values in oracle_at_k_logits_np.items():
                topk_probability = sigmoid_array(values[valid, task_index])
                topk_auroc, topk_auprc = _binary_metrics(
                    task_labels, topk_probability
                )
                metrics[f"oracle_at_{width}_auroc_{task_name}"] = topk_auroc
                metrics[f"oracle_at_{width}_auprc_{task_name}"] = topk_auprc
            if task_index == 0:
                metrics["oracle_auroc"] = oracle_auroc
                metrics["oracle_auprc"] = oracle_auprc
                metrics["learned_routing_gap"] = oracle_auroc - final_auroc
                metrics["learned_routing_gap_auroc"] = oracle_auroc - final_auroc
                metrics["learned_routing_gap_auprc"] = oracle_auprc - final_auprc
        probability_sets = {
            "base": (base_np, base_uncalibrated_np),
            "direct": (direct_np, direct_uncalibrated_np),
            "final": (final_np, final_uncalibrated_np),
        }
        for prefix, (calibrated, uncalibrated) in probability_sets.items():
            calibrated_metrics = binary_calibration_metrics(
                task_labels, calibrated[valid, task_index]
            )
            uncalibrated_metrics = binary_calibration_metrics(
                task_labels, uncalibrated[valid, task_index]
            )
            for name, value in calibrated_metrics.items():
                metrics[f"{prefix}_{name}_{task_name}"] = value
            for name, value in uncalibrated_metrics.items():
                metrics[f"{prefix}_uncalibrated_{name}_{task_name}"] = value

    metrics["mean_operator_gate_alpha"] = float(np.mean(alpha_np))
    metrics["std_operator_gate_alpha"] = float(np.std(alpha_np))
    for percentile in [10, 50, 90]:
        metrics[f"p{percentile}_operator_gate_alpha"] = float(
            np.percentile(alpha_np, percentile)
        )
    metrics["direct_attention_gate"] = float(
        model.direct_attention_gate_value().detach().cpu()
    )
    metrics["mean_direct_attention_gate"] = metrics["direct_attention_gate"]
    metrics["mean_abs_operator_delta"] = float(delta_tensor.abs().mean())
    metrics["std_operator_delta"] = float(delta_tensor.std(unbiased=False))
    metrics["mean_abs_scaled_operator_delta"] = float(
        bounded_correction_tensor.abs().mean()
    )
    metrics["mean_scaled_operator_delta"] = float(bounded_correction_tensor.mean())
    metrics["mean_raw_correction"] = float(raw_correction_tensor.mean())
    metrics["mean_bounded_correction"] = float(bounded_correction_tensor.mean())
    metrics["p95_abs_correction"] = float(
        torch.quantile(bounded_correction_tensor.abs().float(), 0.95)
    )
    correction_bound = (
        float(model.correction_bound.detach().cpu())
        if getattr(model, "performance_v5", False)
        else float(
            cfg.get("model", {}).get("performance_v5", {})
            .get("bounded_correction", {}).get("selected_c", 0.35)
        )
    )
    metrics["correction_saturation_fraction"] = float(
        (bounded_correction_tensor.abs() >= 0.95 * correction_bound).float().mean()
    )
    direct_scale = torch.from_numpy(direct_logits_np).abs().mean().clamp_min(1e-8)
    metrics["relative_scaled_operator_delta"] = float(
        bounded_correction_tensor.abs().mean() / direct_scale
    )
    metrics["reason_rate"] = (
        float(
            np.mean(
                reason_probability_np
                >= float(model.reason_threshold.detach().cpu())
            )
        )
        if evaluation_stage != "direct" else None
    )
    if oracle_logits_np is not None:
        metrics["oracle_diagnostics_only"] = True
        metrics["diagnostic_only"] = True
        metrics["oracle_path_agreement"] = (
            float(torch.cat(oracle_agreement_values).mean())
            if oracle_agreement_values else None
        )
        for key in [
            "candidate_utility_std",
            "candidate_modality_coverage",
            "best_path_modality_distribution",
        ]:
            values = [row[key] for row in oracle_diagnostic_rows]
            metrics[key] = np.asarray(values, dtype=np.float64).mean(0).tolist()
            if key == "candidate_utility_std":
                metrics[key] = float(metrics[key])
        if policy_ranking_rows:
            for key in policy_ranking_rows[0]:
                metrics[key] = float(
                    np.mean([row[key] for row in policy_ranking_rows])
                )
        if reranker_ranking_rows:
            metrics["reranker_top1_agreement"] = float(
                np.mean(
                    [
                        row["oracle_top1_agreement"]
                        for row in reranker_ranking_rows
                    ]
                )
            )
            metrics["reranker_top3_recall"] = float(
                np.mean(
                    [row["oracle_top3_recall"] for row in reranker_ranking_rows]
                )
            )
        if selected_utility_regrets:
            regret = torch.cat(selected_utility_regrets).numpy()
            metrics["mean_utility_regret"] = float(np.mean(regret))
            metrics["median_utility_regret"] = float(np.median(regret))
            metrics["p90_utility_regret"] = float(np.percentile(regret, 90))

    expanded_alpha = np.repeat(alpha_np[:, None], direct_np.shape[1], axis=1)
    valid_all = masks_np > 0.5
    direct_correct = (direct_np >= 0.5) == (labels_np > 0.5)
    confident = np.abs(direct_np - 0.5) * 2.0 >= 0.5
    direct_element_loss = (
        np.logaddexp(0.0, direct_logits_np) - labels_np * direct_logits_np
    )
    final_element_loss = (
        np.logaddexp(0.0, final_logits_np) - labels_np * final_logits_np
    )
    utility_positive = direct_element_loss - final_element_loss > 0
    sample_utility = (
        ((direct_element_loss - final_element_loss) * masks_np).sum(1)
        / np.clip(masks_np.sum(1), 1.0, None)
    )
    reason_target = sample_utility > 0
    if evaluation_stage == "direct":
        metrics["reason_auroc"] = None
        metrics["reason_auprc"] = None
        metrics["gate_utility_correlation"] = None
        metrics["reason_precision"] = None
        metrics["reason_recall"] = None
    elif np.unique(reason_target.astype(np.int64)).size >= 2:
        metrics["reason_auroc"] = float(
            roc_auc_score(reason_target, reason_probability_np)
        )
        metrics["reason_auprc"] = float(
            average_precision_score(reason_target, reason_probability_np)
        )
    elif evaluation_stage != "direct":
        metrics["reason_auroc"] = None
        metrics["reason_auprc"] = None
    if evaluation_stage != "direct" and np.std(sample_utility) > 0 and np.std(reason_probability_np) > 0:
        metrics["gate_utility_correlation"] = float(
            np.corrcoef(reason_probability_np, sample_utility)[0, 1]
        )
    elif evaluation_stage != "direct":
        metrics["gate_utility_correlation"] = 0.0
    routed = reason_probability_np >= float(model.reason_threshold.detach().cpu())
    true_positive = int(np.sum(routed & reason_target))
    if evaluation_stage != "direct":
        metrics["reason_precision"] = float(
            true_positive / max(int(routed.sum()), 1)
        )
        metrics["reason_recall"] = float(
            true_positive / max(int(reason_target.sum()), 1)
        )
    if evaluation_stage != "direct" and metrics["gate_utility_correlation"] < 0.05:
        print(
            "[gate-warning] utility correlation below 0.05: "
            f"{metrics['gate_utility_correlation']:.6f}"
        )
    alpha_groups = {
        "positive_label": valid_all & (labels_np > 0.5),
        "negative_label": valid_all & (labels_np <= 0.5),
        "direct_correct": valid_all & direct_correct,
        "direct_error": valid_all & ~direct_correct,
        "direct_confident": valid_all & confident,
        "direct_uncertain": valid_all & ~confident,
        "utility_positive": valid_all & utility_positive,
        "utility_negative": valid_all & ~utility_positive,
    }
    for name, selector in alpha_groups.items():
        values = expanded_alpha[selector]
        metrics[f"mean_alpha_{name}"] = (
            float(values.mean()) if values.size else None
        )
    epsilon = model.identity_mix_values()
    for modality, value in epsilon.items():
        metrics[f"identity_mix_epsilon_{modality}"] = float(value.detach().cpu())
    metrics["mean_identity_mix_epsilon"] = float(
        np.mean([float(value.detach().cpu()) for value in epsilon.values()])
    )
    metrics["mean_state_gate_beta"] = float(model.state_gate_value().cpu())
    if uses_policy:
        metrics["mean_rollout_length"] = float(torch.cat(lengths).mean())
        metrics["mean_policy_entropy"] = float(torch.cat(policy_entropies).mean())
        metrics["mean_policy_logprob"] = float(torch.cat(policy_logprobs).mean())
        reward_statistics = _mean_diagnostics(reward_rows)
        for name, value in reward_statistics.items():
            metric_name = name if name.startswith("std_") else f"mean_{name}"
            metrics[metric_name] = value
        total_action_slots = int(action_counts.sum())
        if model.policy.include_stop:
            metrics["stop_rate"] = (
                float(action_counts[model.stop_idx] / total_action_slots)
                if total_action_slots else 0.0
            )
            non_stop_total = int(action_counts[:model.stop_idx].sum())
        else:
            metrics["stop_rate"] = float(torch.cat(lengths).eq(0).float().mean())
            non_stop_total = total_action_slots
        for modality_index, modality in enumerate(model.modalities):
            start = modality_index * model.K
            count = int(action_counts[start:start + model.K].sum())
            metrics[f"action_frequency_{modality}"] = (
                float(count / non_stop_total) if non_stop_total else 0.0
            )
    else:
        metrics.update(
            {
                "mean_rollout_length": None,
                "mean_policy_entropy": None,
                "mean_policy_logprob": None,
                "stop_rate": None,
                "mean_raw_prediction_gain": None,
                "std_raw_prediction_gain": None,
                "mean_scaled_prediction_gain": None,
                "mean_order_contribution": None,
                "mean_length_penalty": None,
                "mean_total_reward": None,
                "mean_stop_reward": None,
                "mean_nonstop_reward": None,
            }
        )
        for modality in model.modalities:
            metrics[f"action_frequency_{modality}"] = None
    metrics.update(model.numerics_summary())
    if uses_policy:
        metrics.update(model._last_policy_diagnostics)
    return metrics


def print_target_report(metrics, cfg):
    task_name = cfg["experiment"]["task_names"][0]
    direct_auroc = float(metrics[f"direct_auroc_{task_name}"])
    direct_auprc = float(metrics[f"direct_auprc_{task_name}"])
    final_auroc = float(metrics[f"final_auroc_{task_name}"])
    final_auprc = float(metrics[f"final_auprc_{task_name}"])
    print("[target-report]")
    print(f"direct_auroc={direct_auroc:.8f}")
    print(f"direct_auprc={direct_auprc:.8f}")
    print(f"final_single_auroc={final_auroc:.8f}")
    print(f"final_single_auprc={final_auprc:.8f}")
    print(f"final_ensemble_auroc={metrics.get('ensemble_auroc', 'not_evaluated')}")
    print(f"final_ensemble_auprc={metrics.get('ensemble_auprc', 'not_evaluated')}")
    print(f"delta_auroc={final_auroc - direct_auroc:.8f}")
    print(f"delta_auprc={final_auprc - direct_auprc:.8f}")
    print("target_auroc=0.90")
    if final_auroc < 0.90:
        print("[target-gap]")
        print(f"auroc_gap_to_0.90={0.90 - final_auroc:.8f}")
        print(f"direct_bottleneck={0.90 - direct_auroc:.8f}")
        print(
            "operator_oracle_ceiling="
            f"{metrics.get('oracle_auroc', 'not_measured_in_this_evaluation')}"
        )
        print(
            "learned_routing_gap="
            f"{metrics.get('learned_routing_gap', 'not_measured_in_this_evaluation')}"
        )
        print(f"auprc_bottleneck={final_auprc - direct_auprc:.8f}")
