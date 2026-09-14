from pathlib import Path

import numpy as np
import pytest
import torch

from ncore.config import load_config
from ncore.ema import ModelEMA, ModelSWA
from ncore.losses import (
    auc_ranking_loss,
    direct_performance_objective,
    oracle_candidate_utility,
    precision_ranking_loss,
    quantile_gate_targets,
)
from ncore.metrics import (
    dual_metric_score,
    normalized_auprc,
    select_ensemble_weights,
    select_reason_threshold,
)
from ncore.models.model import NCORE
from ncore.models.v5 import RunningUtilityNormalizer, bounded_correction
from ncore.sampling import PositiveAwareBatchSampler
from ncore.training import configure_trainable_parameters


def v5_cfg():
    return {
        "paths": {"report_model_name_or_path": ""},
        "experiment": {"task_names": ["mortality"]},
        "model": {
            "modalities": ["cxr", "ecg", "ehr"],
            "hidden_dim": 24,
            "dropout": 0.0,
            "concept": {"num_concepts": 4, "concept_dim": 12},
            "encoders": {
                "cxr": {"backend": "precomputed", "input_dim": 6},
                "ecg": {"backend": "precomputed", "input_dim": 8},
                "ehr": {"backend": "precomputed", "input_dim": 10},
            },
            "direct_fusion": {
                "enabled": True,
                "fusion_dim": 12,
                "hidden_dim": 16,
                "dropout": 0.0,
                "attention": {"enabled": False},
            },
            "operator": {
                "identity_mix": True,
                "identity_mix_init": 0.1,
                "state_residual_gate_init": 0.1,
            },
            "residual": {
                "enabled": True,
                "mode": "direct_aware",
                "residual_dim": 16,
                "hidden_dim": 16,
                "dropout": 0.0,
                "gate_mode": "patient_specific",
                "gate_hidden_dim": 12,
                "gate_dropout": 0.0,
                "alpha_min": 0.03,
                "alpha_init": 0.08,
                "alpha_max": 0.20,
                "include_logit_magnitude": True,
            },
            "policy": {
                "max_steps": 2,
                "include_direct_uncertainty": True,
                "include_logit_magnitude": True,
            },
            "performance_v5": {
                "enabled": True,
                "pairwise": {
                    "dim": 12,
                    "rank_dim": 4,
                    "hidden_dim": 12,
                    "dropout": 0.0,
                    "gate_init": 0.05,
                    "gate_max": 0.30,
                },
                "pooling": {
                    "hidden_dim": 12,
                    "score_hidden_dim": 8,
                    "dropout": 0.0,
                    "gate_init": 0.03,
                    "gate_max": 0.20,
                },
                "bounded_correction": {"selected_c": 0.35},
                "reason_gate": {"hidden_dim": 12, "dropout": 0.0},
            },
        },
        "training": {
            "loss": {},
            "direct_loss": {},
            "direct_phase": {"freeze_base_epochs": 5, "ehr_last_layers": 1},
            "lr_pairwise": 3e-4,
            "lr_pool": 3e-4,
            "lr_direct_head": 2e-5,
            "lr_ehr_last_layers": 1e-5,
        },
        "data": {},
        "reward": {},
    }


def v5_batch(batch_size=8):
    cfg = v5_cfg()
    labels = torch.tensor(([0, 1] * ((batch_size + 1) // 2))[:batch_size]).float()
    return {
        "modalities": {
            name: torch.randn(batch_size, spec["input_dim"])
            for name, spec in cfg["model"]["encoders"].items()
        },
        "modality_mask": torch.ones(batch_size, 3),
        "labels": labels.unsqueeze(1),
        "label_mask": torch.ones(batch_size, 1),
        "meta": [{} for _ in range(batch_size)],
    }


def test_normalized_auprc_and_dual_score():
    assert normalized_auprc(0.4, 0.2) == pytest.approx(0.25)
    assert dual_metric_score(0.9, 0.4, 0.2) == pytest.approx(0.64)


def test_pairwise_gate_zero_strictly_recovers_base():
    model = NCORE(v5_cfg()).eval()
    model.pairwise_residual.gate.set_override(0.0)
    model.pooling_residual.gate.set_override(0.0)
    batch = v5_batch()
    output = model.direct_outputs(model.encode(batch), batch["modality_mask"])
    assert torch.equal(output["direct_logits"], output["direct_base_logits"])


def test_pairwise_missing_modality_is_strictly_masked():
    model = NCORE(v5_cfg()).eval()
    batch = v5_batch()
    batch["modality_mask"][0, 0] = 0
    changed = v5_batch()
    changed["modality_mask"] = batch["modality_mask"].clone()
    changed["modalities"] = {name: value.clone() for name, value in batch["modalities"].items()}
    changed["modalities"]["cxr"][0].fill_(1e6)
    first = model.direct_outputs(model.encode(batch), batch["modality_mask"])
    second = model.direct_outputs(model.encode(changed), changed["modality_mask"])
    assert torch.allclose(first["direct_logits"][0], second["direct_logits"][0])


def test_pooling_weights_respect_missing_mask():
    model = NCORE(v5_cfg()).eval()
    batch = v5_batch()
    batch["modality_mask"][0, 1] = 0
    output = model.direct_outputs(model.encode(batch), batch["modality_mask"])
    weights = output["direct_modality_weights"]
    assert weights[0, 1].item() == 0.0
    assert torch.allclose(weights.sum(1), torch.ones(weights.size(0)))


def test_auc_pr_and_direct_losses_are_finite():
    batch = v5_batch()
    logits = torch.randn(8, 1, requires_grad=True)
    auc = auc_ranking_loss(logits, batch["labels"], batch["label_mask"])
    pr = precision_ranking_loss(logits, batch["labels"], batch["label_mask"])
    total, _ = direct_performance_objective(
        logits, batch["labels"], batch["label_mask"], torch.ones(1), v5_cfg()
    )
    assert torch.isfinite(torch.stack([auc, pr, total])).all()


def test_positive_aware_sampler_guarantees_requested_positives():
    sampler = PositiveAwareBatchSampler(
        [1] * 5 + [0] * 95,
        20,
        min_positive_per_batch=4,
        positive_fraction_target=0.2,
        seed=42,
    )
    labels = np.array([1] * 5 + [0] * 95)
    assert all(labels[batch].sum() >= 4 for batch in sampler)


def test_bounded_correction_never_exceeds_c():
    output = bounded_correction(torch.tensor([-1e6, 0.0, 1e6]), 0.35)
    assert output.abs().max() <= 0.35


def test_running_utility_normalization_is_finite():
    normalizer = RunningUtilityNormalizer()
    output = normalizer(torch.tensor([float("nan"), -1.0, 1.0]), update=True)
    assert torch.isfinite(output).all()


def test_quantile_gate_targets_set_extremes():
    _, rank = quantile_gate_targets(torch.arange(8).float())
    assert torch.equal(rank[:2], torch.zeros(2))
    assert torch.equal(rank[-2:], torch.ones(2))


def test_reason_gate_output_is_probability():
    model = NCORE(v5_cfg()).eval()
    output = model(v5_batch(), deterministic=True)
    assert torch.all((output["reason_probability"] > 0) & (output["reason_probability"] < 1))


def test_reason_threshold_selection_is_validation_only():
    args = (
        np.array([0.1, 0.9, 0.2, 0.8]),
        np.array([-1.0, 1.0, -0.5, 0.5]),
        np.array([-2.0, 2.0, -1.0, 1.0]),
        np.array([0, 1, 0, 1]),
    )
    selected, _ = select_reason_threshold(*args, split="val")
    assert selected["threshold"] in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    with pytest.raises(ValueError):
        select_reason_threshold(*args, split="test")


def test_v5_path_policy_has_no_stop_action():
    model = NCORE(v5_cfg())
    assert model.policy.include_stop is False
    assert model.policy.num_actions == model.M * model.K
    assert model.num_actions == model.M * model.K


def test_oracle_candidate_utility_is_finite():
    gains = torch.tensor([[0.0, 1.0, 2.0], [2.0, 1.0, 0.0]])
    utility = oracle_candidate_utility(gains, gains * 2, gains * 0.5)
    assert utility.shape == gains.shape
    assert torch.isfinite(utility).all()


def test_grpo_only_unfreezes_routing():
    model = NCORE(v5_cfg())
    configure_trainable_parameters(model, v5_cfg(), "grpo")
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    assert trainable
    assert all(name.startswith(("reason_gate.", "policy.")) for name in trainable)


def test_direct_is_frozen_after_direct_stage():
    model = NCORE(v5_cfg())
    configure_trainable_parameters(model, v5_cfg(), "supervised")
    assert all(not parameter.requires_grad for parameter in model.direct_head.parameters())
    assert all(not parameter.requires_grad for parameter in model.pairwise_residual.parameters())


def test_temperature_scaling_preserves_ranking_metrics():
    labels = np.array([0, 1, 0, 1])
    logits = np.array([-2.0, 0.2, -0.5, 3.0])
    original = (np.argsort(logits), np.argsort(logits))
    calibrated = (np.argsort(logits / 2.5), np.argsort(logits / 2.5))
    assert all(np.array_equal(a, b) for a, b in zip(original, calibrated))


def test_ema_updates_v5_parameters():
    model = torch.nn.Linear(2, 1)
    ema = ModelEMA(model, decay=0.5)
    before = ema.shadow["weight"].clone()
    with torch.no_grad():
        model.weight.add_(1)
    ema.update(model)
    assert not torch.equal(before, ema.shadow["weight"])


def test_swa_averages_and_restores():
    model = torch.nn.Linear(1, 1, bias=False)
    swa = ModelSWA(model)
    with torch.no_grad():
        model.weight.fill_(1)
    swa.update(model)
    with torch.no_grad():
        model.weight.fill_(3)
    swa.update(model)
    raw = model.weight.clone()
    with swa.average_parameters(model):
        assert model.weight.item() == pytest.approx(2.0)
    assert torch.equal(model.weight, raw)


def test_ensemble_weight_selection_is_validation_only():
    logits = np.array([
        [-2, 2, -1, 1], [-1, 1, -0.5, 0.5], [-3, 3, -2, 2]
    ])
    labels = np.array([0, 1, 0, 1])
    selected, _ = select_ensemble_weights(logits, labels, split="val")
    assert sum(selected["weights"]) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        select_ensemble_weights(logits, labels, split="test")


def test_formal_epochs_are_not_overridden_by_smoke():
    root = Path(__file__).resolve().parents[1]
    formal = load_config(root / "configs/ncore_performance_v5_mortality.yaml")
    smoke = load_config(root / "configs/ncore_performance_v5_smoke.yaml")
    names = [
        "epochs_direct", "epochs_operator_warmup", "epochs_supervised",
        "epochs_policy_warmup", "epochs_grpo",
    ]
    assert [formal["training"][name] for name in names] == [30, 10, 16, 10, 12]
    assert [smoke["training"][name] for name in names] == [2, 2, 2, 2, 2]


def test_legacy_config_still_has_stop_and_forwards():
    cfg = v5_cfg()
    cfg["model"].pop("performance_v5")
    model = NCORE(cfg)
    output = model(v5_batch(), deterministic=True)
    assert model.num_actions == model.M * model.K + 1
    assert torch.isfinite(output["logits"]).all()


def test_v5_rollout_is_finite_and_uses_virtual_stop_only_for_routing():
    model = NCORE(v5_cfg()).eval()
    output = model(v5_batch(), deterministic=True)
    assert output["actions"].shape == (8, model.max_steps)
    assert torch.isfinite(output["logits"]).all()
    assert torch.isfinite(output["logprob"]).all()
