import inspect
from pathlib import Path

import numpy as np
import pytest
import torch

import ncore.training as training
from ncore.config import load_config
from ncore.losses import (
    categorical_policy_kl,
    listwise_kl_loss,
    path_pairwise_preference_loss,
    patient_wise_utility_normalize,
    reranker_objective,
)
from ncore.metrics import oracle_within_topk_indices, routing_ranking_metrics
from ncore.models.model import NCORE
from ncore.models.v6 import CandidateReranker
from ncore.training import configure_trainable_parameters


def v6_cfg():
    return {
        "seed": 42,
        "paths": {"report_model_name_or_path": "", "output_root": "outputs"},
        "experiment": {"task_names": ["mortality"], "name": "test"},
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
                    "gate_init": 0.03,
                    "gate_max": 0.15,
                },
                "pooling": {
                    "hidden_dim": 12,
                    "score_hidden_dim": 8,
                    "dropout": 0.0,
                    "gate_init": 0.02,
                    "gate_max": 0.10,
                },
                "bounded_correction": {"selected_c": 0.35},
                "reason_gate": {"hidden_dim": 12, "dropout": 0.0},
            },
            "performance_v6": {
                "enabled": True,
                "routing": {"num_candidates": 6, "policy_topk": 3},
                "reranker": {
                    "enabled": True,
                    "hidden_dim": 16,
                    "path_embedding_dim": 8,
                    "dropout": 0.0,
                },
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
            "lr_policy": 5e-5,
            "lr_reranker": 5e-5,
        },
        "grpo": {"lr_policy": 3e-6, "kl_anchor_weight": 0.05},
        "data": {},
        "reward": {},
        "selection": {"tolerance_auroc": 0.001, "tolerance_auprc": 0.002},
    }


def batch(batch_size=6):
    cfg = v6_cfg()
    labels = torch.tensor(([0, 1] * 4)[:batch_size]).float()
    return {
        "modalities": {
            name: torch.randn(batch_size, spec["input_dim"])
            for name, spec in cfg["model"]["encoders"].items()
        },
        "modality_mask": torch.ones(batch_size, 3),
        "labels": labels[:, None],
        "label_mask": torch.ones(batch_size, 1),
        "meta": [{} for _ in range(batch_size)],
    }


def test_strong_direct_warm_start_loads_anchor(tmp_path, monkeypatch):
    cfg = v6_cfg()
    source = NCORE(cfg)
    with torch.no_grad():
        source.direct_head.weight.fill_(0.25)
    checkpoint = tmp_path / "strong.pt"
    torch.save({"model": source.state_dict(), "metrics": {}}, checkpoint)
    cfg["training"]["strong_direct_init_checkpoint"] = str(checkpoint)
    target = NCORE(cfg)
    monkeypatch.setattr(
        training,
        "evaluate_model",
        lambda *args, **kwargs: {
            "direct_auroc_mortality": 0.86,
            "direct_auprc_mortality": 0.30,
        },
    )
    report = training.initialize_strong_direct_anchor(
        target, object(), cfg, torch.device("cpu"), tmp_path
    )
    assert report["direct_auroc_mortality"] == pytest.approx(0.86)
    assert torch.equal(target.direct_head.weight, source.direct_head.weight)


def test_strong_direct_failed_sanity_stops_formal_run(tmp_path, monkeypatch):
    cfg = v6_cfg()
    checkpoint = tmp_path / "weak.pt"
    torch.save({"model": NCORE(cfg).state_dict(), "metrics": {}}, checkpoint)
    cfg["training"]["strong_direct_init_checkpoint"] = str(checkpoint)
    cfg["training"]["require_strong_direct_init"] = True
    monkeypatch.setattr(
        training,
        "evaluate_model",
        lambda *args, **kwargs: {
            "direct_auroc_mortality": 0.79,
            "direct_auprc_mortality": 0.20,
        },
    )
    with pytest.raises(RuntimeError, match="strong direct warm start failed"):
        training.initialize_strong_direct_anchor(
            NCORE(cfg), object(), cfg, torch.device("cpu"), tmp_path
        )


def test_zero_residual_gates_strictly_recover_strong_direct():
    model = NCORE(v6_cfg()).eval()
    model.pairwise_residual.gate.set_override(0.0)
    model.pooling_residual.gate.set_override(0.0)
    sample = batch()
    output = model.direct_outputs(model.encode(sample), sample["modality_mask"])
    assert torch.equal(output["direct_logits"], output["strong_direct_logits"])


def test_patient_wise_utility_normalization():
    values = torch.tensor([[1.0, 2.0, 3.0], [100.0, 200.0, 300.0]])
    normalized = patient_wise_utility_normalize(values)
    assert torch.allclose(normalized.mean(1), torch.zeros(2), atol=1e-6)
    assert torch.allclose(normalized[0], normalized[1], atol=1e-6)


def test_listwise_kl_prefers_oracle_order():
    utility = torch.tensor([[0.0, 1.0, 2.0]])
    normalized = patient_wise_utility_normalize(utility)
    good = listwise_kl_loss(normalized / 0.2, utility, 0.2)
    bad = listwise_kl_loss(-normalized / 0.2, utility, 0.2)
    assert good < bad


def test_pairwise_path_preference_loss_prefers_correct_order():
    utility = torch.tensor([[0.0, 1.0, 2.0]])
    good = path_pairwise_preference_loss(utility, utility)
    bad = path_pairwise_preference_loss(-utility, utility)
    assert good < bad


def test_topk_candidate_paths_are_valid():
    model = NCORE(v6_cfg()).eval()
    sample = batch()
    paths = model.generate_policy_topk_paths(sample, topk=3)
    assert len(paths) == 3
    assert all(path.shape == (6, model.max_steps) for path in paths)
    assert all(torch.all(path < model.num_actions) for path in paths)


def test_reranker_forward_has_no_label_input():
    parameters = inspect.signature(CandidateReranker.forward).parameters
    assert "labels" not in parameters
    assert "targets" not in parameters
    assert "label_mask" not in parameters


def test_reranker_training_target_can_use_train_utility():
    scores = torch.randn(3, 8, requires_grad=True)
    utility_from_train_labels = torch.randn(3, 8)
    policy = torch.randn(3, 8)
    loss, _ = reranker_objective(scores, utility_from_train_labels, policy)
    loss.backward()
    assert scores.grad is not None and torch.isfinite(scores.grad).all()


def test_deterministic_inference_is_repeatable():
    model = NCORE(v6_cfg()).eval()
    sample = batch()
    first = model(sample, deterministic=True)
    second = model(sample, deterministic=True)
    assert torch.equal(first["actions"], second["actions"])
    assert torch.allclose(first["logits"], second["logits"])


def test_reason_false_strictly_returns_direct():
    model = NCORE(v6_cfg()).eval()
    model.reason_threshold.fill_(2.0)
    sample = batch()
    output = model(sample, deterministic=True)
    assert torch.equal(output["logits"], output["direct_logits"])
    assert torch.all(output["actions"] == model.stop_idx)


def test_reason_true_uses_topk_and_reranker():
    model = NCORE(v6_cfg()).eval()
    model.reason_threshold.fill_(-1.0)
    output = model(batch(), deterministic=True)
    assert "policy_topk_actions" in output
    assert "reranker_scores" in output
    assert torch.all(output["actions"][:, 0] != model.stop_idx)


def test_inference_does_not_read_labels():
    model = NCORE(v6_cfg()).eval()
    first_batch = batch()
    second_batch = {
        key: (
            {name: value.clone() for name, value in item.items()}
            if key == "modalities"
            else item.clone() if torch.is_tensor(item) else list(item)
        )
        for key, item in first_batch.items()
    }
    second_batch["labels"] = 1.0 - first_batch["labels"]
    first = model(first_batch, deterministic=True)["logits"]
    second = model(second_batch, deterministic=True)["logits"]
    assert torch.equal(first, second)


def test_oracle_at_k_is_restricted_to_policy_candidates():
    policy = np.array([[3.0, 2.0, 1.0, 0.0]])
    utility = np.array([[0.0, 1.0, 5.0, 9.0]])
    assert oracle_within_topk_indices(policy, utility, 1).item() == 0
    assert oracle_within_topk_indices(policy, utility, 3).item() == 2


def test_utility_regret_is_correct():
    diagnostics = routing_ranking_metrics(
        np.array([[3.0, 2.0, 1.0]]),
        np.array([[0.0, 2.0, 5.0]]),
    )
    assert diagnostics["mean_utility_regret"] == pytest.approx(5.0)


def test_grpo_kl_anchor_is_active():
    current = torch.tensor([[[2.0, 0.0], [1.0, 0.0]]], requires_grad=True)
    reference = torch.zeros_like(current)
    loss = categorical_policy_kl(current, reference)
    loss.backward()
    assert loss > 0
    assert current.grad is not None


def test_grpo_only_updates_routing_last_layers():
    model = NCORE(v6_cfg())
    configure_trainable_parameters(model, v6_cfg(), "grpo")
    trainable = [
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    assert trainable
    assert all(
        name.startswith(
            ("policy.net.3.", "reason_gate.network.4.", "reranker.last_layer.")
        )
        for name in trainable
    )


def test_formal_and_smoke_epochs_remain_separate():
    root = Path(__file__).resolve().parents[1]
    formal = load_config(root / "configs/ncore_performance_v6_mortality.yaml")
    smoke = load_config(root / "configs/ncore_performance_v6_smoke.yaml")
    names = [
        "epochs_direct",
        "epochs_operator_warmup",
        "epochs_supervised",
        "epochs_policy_warmup",
        "epochs_grpo",
    ]
    assert [formal["training"][name] for name in names] == [30, 10, 16, 10, 12]
    assert [smoke["training"][name] for name in names] == [2, 2, 2, 2, 2]


def test_legacy_v5_config_still_forwards():
    cfg = v6_cfg()
    cfg["model"].pop("performance_v6")
    model = NCORE(cfg).eval()
    output = model(batch(), deterministic=True)
    assert "reranker_scores" not in output
    assert torch.isfinite(output["logits"]).all()


def test_v6_forward_has_no_nan():
    model = NCORE(v6_cfg()).eval()
    output = model(batch(), deterministic=True)
    tensors = [
        output["logits"],
        output["direct_logits"],
        output["reason_probability"],
        output["reranker_scores"],
    ]
    assert all(torch.isfinite(value).all() for value in tensors)
