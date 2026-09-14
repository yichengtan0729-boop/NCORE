import torch

from ncore.models.model import NCORE
from ncore.rl.grpo import prediction_delta_reward


def residual_cfg(task_names=None):
    task_names = task_names or ["vent24"]
    return {
        "paths": {"report_model_name_or_path": ""},
        "experiment": {"task_names": task_names},
        "model": {
            "modalities": ["cxr", "ecg", "ehr"],
            "hidden_dim": 32,
            "dropout": 0.0,
            "concept": {"num_concepts": 8, "concept_dim": 16},
            "policy": {"max_steps": 2},
            "direct_fusion": {
                "enabled": True,
                "fusion_dim": 12,
                "hidden_dim": 16,
                "dropout": 0.0,
            },
            "operator": {
                "identity_mix": True,
                "identity_mix_init": 0.10,
                "state_residual_gate_init": 0.10,
                "rho": 0.98,
            },
            "residual": {"operator_gate_init": 0.05},
            "encoders": {
                "cxr": {"backend": "precomputed", "input_dim": 8},
                "ecg": {"backend": "precomputed", "input_dim": 10},
                "ehr": {"backend": "precomputed", "input_dim": 12},
            },
        },
        "data": {"report_max_length": 64},
        "reward": {
            "prediction_delta": 1.0,
            "order": 0.05,
            "evidence": 0.0,
            "faithfulness": 0.0,
            "length": 0.01,
        },
    }


def residual_batch(num_tasks=1):
    return {
        "modalities": {
            "cxr": torch.randn(4, 8),
            "ecg": torch.randn(4, 10),
            "ehr": torch.randn(4, 12),
        },
        "modality_mask": torch.tensor(
            [[1, 1, 1], [0, 1, 1], [0, 0, 1], [1, 0, 1]],
            dtype=torch.float32,
        ),
        "labels": torch.randint(0, 2, (4, num_tasks)).float(),
        "label_mask": torch.ones(4, num_tasks),
        "meta": [{}] * 4,
    }


def test_zero_operator_gate_recovers_direct_baseline():
    model = NCORE(residual_cfg())
    batch = residual_batch()
    with torch.no_grad():
        model.outcome_head[-1].weight.normal_()
        model.outcome_head[-1].bias.normal_()
        model.operator_gate_logit.fill_(float("-inf"))
    rollout = model.sample_rollout(batch, deterministic=True)
    assert torch.allclose(
        rollout["logits"], rollout["direct_logits"], atol=1e-6, rtol=0
    )


def test_missing_modality_uses_identity_operator_and_zero_commutator():
    model = NCORE(residual_cfg())
    batch = residual_batch()
    encoded = model.encode(batch)
    operators = model.build_operators(encoded)
    identity = torch.eye(model.K)
    assert torch.allclose(operators["cxr"][1], identity, atol=1e-7)
    available = operators["ecg"][1]
    commutator = available @ operators["cxr"][1] - operators["cxr"][1] @ available
    assert torch.allclose(commutator, torch.zeros_like(commutator), atol=1e-7)


def test_missing_actions_are_masked():
    model = NCORE(residual_cfg())
    batch = residual_batch()
    encoded = model.encode(batch)
    operators = model.build_operators(encoded)
    state = model.initial(4, torch.device("cpu"))
    previous = torch.full((4,), model.stop_idx + 1, dtype=torch.long)
    logits = model._policy_inputs(
        state, encoded, operators, batch["modality_mask"], previous, 0
    )
    probabilities = torch.softmax(logits, dim=-1)
    assert torch.equal(probabilities[1, :model.K], torch.zeros(model.K))


def test_direct_fusion_ignores_values_of_missing_modalities():
    model = NCORE(residual_cfg()).eval()
    batch = residual_batch()
    changed = {
        **batch,
        "modalities": {key: value.clone() for key, value in batch["modalities"].items()},
    }
    changed["modalities"]["cxr"][1].fill_(1e6)
    changed["modalities"]["cxr"][2].fill_(-1e6)
    changed["modalities"]["ecg"][2].fill_(1e6)
    with torch.no_grad():
        original_logits = model.direct_logits(model.encode(batch), batch["modality_mask"])
        changed_logits = model.direct_logits(
            model.encode(changed), changed["modality_mask"]
        )
    assert torch.allclose(original_logits[1:3], changed_logits[1:3], atol=1e-6)


def test_action_space_is_dynamic():
    model = NCORE(residual_cfg())
    assert model.M == 3
    assert model.K == 8
    assert model.num_actions == 25
    assert model.policy.num_actions == 25
    assert model.stop_idx == 24


def test_prediction_delta_reward_sign():
    targets = torch.ones(2, 1)
    mask = torch.ones_like(targets)
    direct = torch.zeros(2, 1)
    improved = torch.full((2, 1), 2.0)
    degraded = torch.full((2, 1), -2.0)
    assert torch.all(prediction_delta_reward(improved, direct, targets, mask) > 0)
    assert torch.all(prediction_delta_reward(degraded, direct, targets, mask) < 0)


def test_stop_path_preserves_state_and_initial_residual_is_direct():
    model = NCORE(residual_cfg())
    batch = residual_batch()
    actions = torch.full((4, model.max_steps), model.stop_idx, dtype=torch.long)
    rollout = model.rollout_actions(batch, actions)
    initial = model.initial(4, torch.device("cpu"))
    assert torch.allclose(rollout["state"], initial, atol=0, rtol=0)
    assert torch.allclose(
        rollout["logits"], rollout["direct_logits"], atol=1e-6, rtol=0
    )


def test_single_and_multitask_output_shapes():
    for task_names, width in [(["vent24"], 1), (["mortality"], 1), (["vent24", "mortality"], 2)]:
        model = NCORE(residual_cfg(task_names))
        rollout = model.sample_rollout(
            residual_batch(num_tasks=width), deterministic=True
        )
        assert rollout["logits"].shape == (4, width)
        assert rollout["direct_logits"].shape == (4, width)
        assert rollout["operator_delta_logits"].shape == (4, width)
