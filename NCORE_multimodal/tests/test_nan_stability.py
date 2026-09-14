import copy
from pathlib import Path

import pytest
import torch

from ncore.checkpointing import load_compatible_model_state
from ncore.models.model import NCORE
from ncore.models.operators import (
    PatientOperatorGenerator,
    commutator_features,
    symmetric_normalize,
)
from ncore.models.v2 import binary_uncertainty_features
from ncore.training import _clip_and_step, evaluate_model
from test_v3_model import v3_batch, v3_cfg
from train import _automatic_checkpoint


def stable_cfg(*, fail_fast=True):
    cfg = v3_cfg()
    cfg["training"].update(
        {
            "operator_warmup": {"fixed_alpha": 0.15},
            "grad_clip": {"enabled": True, "max_norm": 1.0},
            "max_consecutive_nonfinite_batches": 3,
        }
    )
    cfg["numerics"] = {
        "eps": 1e-6,
        "operator_clip": 10.0,
        "commutator_clip": 20.0,
        "commutator_norm_clip": 20.0,
        "delta_clip": 10.0,
        "state_update_clip": 10.0,
    }
    cfg["debug"] = {
        "finite_checks": True,
        "finite_checks_raise": fail_fast,
        "fail_fast_on_nonfinite": fail_fast,
        "parameter_finite_check_interval": 1,
    }
    cfg["evaluation"] = {"rollout_during_operator_warmup": False}
    return cfg


def stable_batch(cfg=None, batch_size=4):
    batch = v3_batch(cfg or stable_cfg(), batch_size=batch_size)
    mask_patterns = torch.tensor(
        [[1, 1, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]],
        dtype=torch.float32,
    )
    repeats = (batch_size + len(mask_patterns) - 1) // len(mask_patterns)
    batch["modality_mask"] = mask_patterns.repeat(repeats, 1)[:batch_size]
    batch["labels"] = (torch.arange(batch_size) % 2).float().unsqueeze(1)
    batch["label_mask"] = torch.ones(batch_size, 1)
    batch["meta"] = [{} for _ in range(batch_size)]
    return batch


def policy_inputs(model, batch):
    encoded = model.encode(batch)
    operators = model.build_operators(encoded)
    state = model.initial(batch["labels"].size(0), torch.device("cpu"))
    previous = torch.full(
        (batch["labels"].size(0),), model.stop_idx + 1, dtype=torch.long
    )
    logits = model._policy_inputs(
        state, encoded, operators, batch["modality_mask"], previous, 0
    )
    return logits, model._last_policy_details


def test_extreme_binary_entropy_is_finite():
    _, entropy, _ = binary_uncertainty_features(
        torch.tensor([[-1e30], [1e30], [0.0]])
    )
    assert torch.isfinite(entropy).all()


@pytest.mark.parametrize("temperature", [1e-12, 1e12])
def test_extreme_temperature_produces_finite_calibrated_probability(temperature):
    model = NCORE(stable_cfg())
    model.set_direct_temperature(temperature)
    probability, entropy, margin = model.direct_uncertainty_features(
        torch.tensor([[-1e20], [1e20]])
    )
    assert torch.isfinite(probability).all()
    assert torch.isfinite(entropy).all()
    assert torch.isfinite(margin).all()
    assert probability.min() >= 1e-6
    assert probability.max() <= 1.0 - 1e-6


def test_zero_valid_operator_pairs_have_zero_finite_summary():
    cfg = stable_cfg()
    model = NCORE(cfg)
    batch = stable_batch(cfg)
    batch["modality_mask"] = torch.tensor(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]],
        dtype=torch.float32,
    )
    encoded = model.encode(batch)
    operators = model.build_operators(encoded)
    state = model.initial(4, torch.device("cpu"))
    mean, maximum = model._commutator_summary(
        operators, state, batch["modality_mask"]
    )
    assert torch.equal(mean, torch.zeros_like(mean))
    assert torch.equal(maximum, torch.zeros_like(maximum))
    assert torch.isfinite(mean).all() and torch.isfinite(maximum).all()


def test_operator_matrix_is_finite_after_normalization():
    generator = PatientOperatorGenerator(
        12, numerics={"eps": 1e-6, "operator_clip": 10.0}
    )
    operator = generator(torch.randn(4, 8, 12) * 1e6)
    assert operator.shape == (4, 8, 8)
    assert torch.isfinite(operator).all()
    assert operator.abs().max() <= 10.0


def test_normalization_avoids_old_infinity_times_zero_nan():
    adjacency = torch.full((1, 8, 8), float("inf"))
    old_degree = adjacency.sum(-1)
    old_inverse = old_degree.rsqrt()
    old = old_inverse.unsqueeze(-1) * adjacency * old_inverse.unsqueeze(-2)
    assert torch.isnan(old).any()
    with pytest.raises(FloatingPointError, match="symmetric_normalize.input"):
        symmetric_normalize(adjacency, eta=1.0)
    finite_extreme = torch.full(
        (1, 8, 8), torch.finfo(torch.float32).max
    )
    stable = symmetric_normalize(finite_extreme, eta=1.0)
    assert torch.isfinite(stable).all()


def test_commutator_features_are_finite_and_clipped():
    operators = [torch.randn(4, 8, 8) * 100 for _ in range(3)]
    features = commutator_features(operators, torch.randn(4, 8, 12) * 100)
    assert features.shape == (4, 3)
    assert torch.isfinite(features).all()
    assert features.max() <= 20.0


def test_policy_state_is_finite():
    cfg = stable_cfg()
    model = NCORE(cfg)
    _, details = policy_inputs(model, stable_batch(cfg))
    assert torch.isfinite(details["policy_state"]).all()


def test_policy_logits_pre_mask_are_finite():
    cfg = stable_cfg()
    model = NCORE(cfg)
    _, details = policy_inputs(model, stable_batch(cfg))
    assert torch.isfinite(details["policy_logits_pre_mask"]).all()


def test_masked_policy_stop_logit_is_finite():
    cfg = stable_cfg()
    model = NCORE(cfg)
    logits, _ = policy_inputs(model, stable_batch(cfg))
    assert torch.isfinite(logits[:, model.stop_idx]).all()


def test_stop_only_categorical_is_valid():
    cfg = stable_cfg()
    model = NCORE(cfg)
    batch = stable_batch(cfg)
    batch["modality_mask"].zero_()
    logits, _ = policy_inputs(model, batch)
    distribution = torch.distributions.Categorical(logits=logits)
    assert torch.equal(distribution.probs[:, model.stop_idx], torch.ones(4))
    assert torch.equal(distribution.sample(), torch.full((4,), model.stop_idx))


def test_operator_warmup_evaluation_does_not_call_sample_rollout(monkeypatch):
    cfg = stable_cfg()
    model = NCORE(cfg).eval()
    monkeypatch.setattr(
        model,
        "sample_rollout",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("called")),
    )
    metrics = evaluate_model(
        model,
        [stable_batch(cfg)],
        cfg,
        torch.device("cpu"),
        split="val",
        stage="operator_warmup",
    )
    assert metrics["stop_rate"] is None
    assert torch.isfinite(torch.tensor(metrics["mean_operator_gate_alpha"]))


def test_policy_warmup_evaluation_calls_rollout(monkeypatch):
    cfg = stable_cfg()
    model = NCORE(cfg).eval()
    original = model.sample_rollout
    calls = {"count": 0}

    def tracked(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(model, "sample_rollout", tracked)
    evaluate_model(
        model,
        [stable_batch(cfg)],
        cfg,
        torch.device("cpu"),
        split="val",
        stage="policy_warmup",
    )
    assert calls["count"] == 1


def malformed_policy_details(model, *, logits_finite=False):
    batch_size, width = 2, model.policy.input_dim
    policy_state = torch.full((batch_size, width), float("nan"))
    logits = torch.zeros(batch_size, model.num_actions)
    return {
        "policy_state_pre_norm": policy_state.clone(),
        "policy_state": policy_state,
        "policy_logits_pre_mask": logits.clone() if logits_finite else logits / 0,
        "policy_logits_post_mask": logits.clone() if logits_finite else logits / 0,
        "valid_action_mask": torch.ones(
            batch_size, model.num_actions, dtype=torch.bool
        ),
    }


def test_nonfinite_policy_state_fails_fast():
    model = NCORE(stable_cfg(fail_fast=True))
    with pytest.raises(FloatingPointError, match="policy_state"):
        model._validated_policy_logits(
            malformed_policy_details(model), {"modality_mask": torch.ones(2, 3)}
        )


def test_nonfinite_policy_state_uses_explicit_stop_fallback():
    model = NCORE(stable_cfg(fail_fast=False))
    logits = model._validated_policy_logits(
        malformed_policy_details(model, logits_finite=True),
        {"modality_mask": torch.ones(2, 3)},
    )
    assert torch.equal(logits[:, model.stop_idx], torch.zeros(2))
    assert torch.equal(
        logits[:, :model.stop_idx],
        torch.full((2, model.stop_idx), -1e9),
    )
    assert model.policy_fallback_count == 2


def test_gradient_clipping_respects_configured_norm():
    model = torch.nn.Linear(2, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    cfg = {
        "training": {
            "grad_clip": {"enabled": True, "max_norm": 0.1},
            "max_consecutive_nonfinite_batches": 3,
        },
        "debug": {},
    }
    loss = model(torch.full((4, 2), 100.0)).sum()
    assert _clip_and_step(
        loss, optimizer, list(model.parameters()), cfg, model=model
    )
    norm = torch.sqrt(
        sum(parameter.grad.square().sum() for parameter in model.parameters())
    )
    assert norm <= 0.1001


def test_nonfinite_gradient_batch_is_skipped():
    model = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    cfg = {
        "training": {
            "grad_clip": {"enabled": True, "max_norm": 1.0},
            "max_consecutive_nonfinite_batches": 3,
        },
        "debug": {},
    }
    before = model.weight.detach().clone()
    state = {}
    loss = model.weight.sum() * torch.tensor(float("nan"))
    assert not _clip_and_step(
        loss,
        optimizer,
        list(model.parameters()),
        cfg,
        model=model,
        stage="operator_warmup",
        nonfinite_state=state,
    )
    assert torch.equal(model.weight, before)
    assert state["skipped"] == 1


def test_amp_operator_and_commutator_compute_float32():
    generator = PatientOperatorGenerator(12)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        operator = generator(torch.randn(2, 8, 12))
        features = commutator_features(
            [operator, operator.transpose(1, 2)], torch.randn(2, 8, 12)
        )
    assert operator.dtype == torch.float32
    assert features.dtype == torch.float32


def test_old_v3_checkpoint_loads_with_new_norm_parameters_missing():
    cfg = stable_cfg()
    original = NCORE(cfg)
    old_state = {
        key: value
        for key, value in original.state_dict().items()
        if "input_norm" not in key and "operator_hidden_norm" not in key
    }
    restored = NCORE(cfg)
    report = load_compatible_model_state(restored, {"model": old_state})
    assert report["loaded_keys"]
    assert any("input_norm" in key for key in report["missing_keys"])


def test_best_direct_to_operator_warmup_checkpoint_chain(tmp_path):
    path = tmp_path / "best_direct.pt"
    torch.save({"stage": "direct"}, path)
    assert _automatic_checkpoint("operator_warmup", tmp_path, False) == path


def test_action_count_is_25_for_three_modalities_and_eight_concepts():
    model = NCORE(stable_cfg())
    assert model.M == 3 and model.K == 8 and model.num_actions == 25


def test_mortality_logits_shape_is_stable():
    cfg = stable_cfg()
    model = NCORE(cfg)
    assert model(stable_batch(cfg), deterministic=True)["logits"].shape == (4, 1)


def test_extreme_crash_regression_policy_distribution_is_finite():
    cfg = stable_cfg()
    model = NCORE(cfg).eval()
    batch = stable_batch(cfg, batch_size=16)
    mask_patterns = torch.tensor(
        [[1, 1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 0]],
        dtype=torch.float32,
    )
    batch["modality_mask"] = mask_patterns.repeat(4, 1)
    with torch.no_grad():
        model.direct_head.weight.mul_(1e6)
        model.direct_head.bias.fill_(1e6)
        for generator in model.op_generators.values():
            generator.q.weight.mul_(1e5)
            generator.k.weight.mul_(1e5)
    logits, _ = policy_inputs(model, batch)
    assert logits.shape == (16, 25)
    assert torch.isfinite(logits).all()
    distribution = torch.distributions.Categorical(logits=logits)
    assert torch.isfinite(distribution.probs).all()
