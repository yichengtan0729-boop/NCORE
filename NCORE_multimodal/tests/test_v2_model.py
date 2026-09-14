import copy

import torch

from ncore.models.model import NCORE


def v2_cfg(task_names=None, modalities=None):
    task_names = task_names or ["mortality"]
    modalities = modalities or ["cxr", "ecg", "ehr"]
    dimensions = {name: 6 + 2 * index for index, name in enumerate(modalities)}
    return {
        "paths": {"report_model_name_or_path": ""},
        "experiment": {"task_names": task_names},
        "model": {
            "modalities": modalities,
            "hidden_dim": 24,
            "dropout": 0.0,
            "concept": {"num_concepts": 4, "concept_dim": 12},
            "encoders": {
                name: {"backend": "precomputed", "input_dim": width}
                for name, width in dimensions.items()
            },
            "direct_fusion": {
                "enabled": True,
                "mode": "residual_attention",
                "fusion_dim": 12,
                "hidden_dim": 16,
                "dropout": 0.0,
                "attention": {
                    "enabled": True,
                    "num_heads": 4,
                    "ff_dim": 24,
                    "dropout": 0.0,
                    "residual_gate_init": 0.10,
                },
            },
            "operator": {
                "identity_mix": True,
                "identity_mix_init": 0.10,
                "state_residual_gate_init": 0.10,
                "rho": 0.98,
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
                "alpha_max": 0.35,
                "alpha_init": 0.08,
            },
            "policy": {
                "max_steps": 2,
                "include_direct_uncertainty": True,
            },
        },
        "data": {},
        "training": {"loss": {}},
        "reward": {"prediction_delta": 1.0},
    }


def v2_batch(cfg=None, batch_size=5):
    cfg = cfg or v2_cfg()
    modalities = cfg["model"]["modalities"]
    mask = torch.ones(batch_size, len(modalities))
    if len(modalities) > 1:
        mask[1, 0] = 0
    return {
        "modalities": {
            name: torch.randn(batch_size, spec["input_dim"])
            for name, spec in cfg["model"]["encoders"].items()
        },
        "modality_mask": mask,
        "labels": torch.randint(
            0, 2, (batch_size, len(cfg["experiment"]["task_names"]))
        ).float(),
        "label_mask": torch.ones(
            batch_size, len(cfg["experiment"]["task_names"])
        ),
        "meta": [{}] * batch_size,
    }


def test_attention_gate_zero_exactly_recovers_base_direct():
    model = NCORE(v2_cfg()).eval()
    batch = v2_batch()
    with torch.no_grad():
        model.direct_attention_enhancer.gate_logit.fill_(float("-inf"))
        outputs = model.direct_outputs(model.encode(batch), batch["modality_mask"])
    assert torch.equal(outputs["direct_logits"], outputs["direct_base_logits"])


def test_attention_respects_missing_modality_mask():
    model = NCORE(v2_cfg()).eval()
    batch = v2_batch()
    changed = copy.deepcopy(batch)
    changed["modalities"]["cxr"][1].fill_(1e5)
    with torch.no_grad():
        original = model.direct_logits(model.encode(batch), batch["modality_mask"])
        modified = model.direct_logits(model.encode(changed), changed["modality_mask"])
    assert torch.allclose(original[1], modified[1], atol=1e-6)


def test_patient_alpha_initialization_and_bounds():
    model = NCORE(v2_cfg()).eval()
    rollout = model.sample_rollout(v2_batch(), deterministic=True)
    alpha = rollout["operator_gate"]
    assert alpha.shape == (5, 1)
    assert torch.allclose(alpha, torch.full_like(alpha, 0.08), atol=1e-6)
    assert torch.all((alpha >= 0) & (alpha <= 0.35))


def test_patient_alpha_can_vary_by_patient():
    model = NCORE(v2_cfg()).eval()
    with torch.no_grad():
        model.patient_operator_gate.network[-1].weight.normal_()
    alpha = model.sample_rollout(v2_batch(), deterministic=True)["operator_gate"]
    assert alpha.std() > 0


def test_fixed_alpha_bypasses_patient_gate():
    model = NCORE(v2_cfg()).eval()
    batch = v2_batch()
    encoded = model.encode(batch)
    operators = model.build_operators(encoded)
    state = model.initial(5, torch.device("cpu"))
    outputs = model.prediction_outputs(
        state, encoded, batch["modality_mask"], operators, fixed_alpha=0.15
    )
    assert torch.equal(outputs["operator_gate"], torch.full((5, 1), 0.15))


def test_zero_alpha_exactly_recovers_enhanced_direct():
    model = NCORE(v2_cfg()).eval()
    batch = v2_batch()
    encoded = model.encode(batch)
    operators = model.build_operators(encoded)
    state = model.initial(5, torch.device("cpu"))
    with torch.no_grad():
        model.residual_correction.delta_head.weight.normal_()
        outputs = model.prediction_outputs(
            state, encoded, batch["modality_mask"], operators, fixed_alpha=0.0
        )
    assert torch.equal(outputs["logits"], outputs["direct_logits"])


def test_direct_aware_output_contract_and_initial_noop():
    model = NCORE(v2_cfg()).eval()
    rollout = model.sample_rollout(v2_batch(), deterministic=True)
    assert rollout["operator_hidden"].shape == (5, model.D)
    assert rollout["residual_interaction_hidden"].shape == (5, 16)
    assert rollout["operator_delta_logits"].shape == (5, 1)
    assert torch.allclose(rollout["logits"], rollout["direct_logits"], atol=1e-7)


def test_residual_disable_is_exact_direct_ablation():
    cfg = v2_cfg()
    cfg["model"]["residual"]["enabled"] = False
    model = NCORE(cfg).eval()
    rollout = model.sample_rollout(v2_batch(cfg), deterministic=True)
    assert torch.equal(rollout["logits"], rollout["direct_logits"])
    assert torch.count_nonzero(rollout["operator_gate"]) == 0


def test_force_identity_operator_ablation():
    cfg = v2_cfg()
    cfg["model"]["operator"]["force_identity_operator"] = True
    model = NCORE(cfg)
    batch = v2_batch(cfg)
    operators = model.build_operators(model.encode(batch))
    expected = torch.eye(model.K).expand(5, -1, -1)
    for operator in operators.values():
        assert torch.equal(operator, expected)


def test_policy_state_contains_direct_uncertainty_features():
    cfg = v2_cfg(task_names=["vent24", "mortality"])
    model = NCORE(cfg)
    expected_extra = 4 * 2 + 3 + model.M
    assert model.policy.extra_dim == expected_extra
    assert model.policy.net[0].in_features == model.policy.base_input_dim + expected_extra


def test_v2_rollout_has_dynamic_action_space_and_finite_values():
    cfg = v2_cfg(modalities=["cxr", "ehr"])
    model = NCORE(cfg)
    rollout = model.sample_rollout(v2_batch(cfg), deterministic=True)
    assert model.num_actions == 2 * model.K + 1
    assert torch.isfinite(rollout["logits"]).all()
    assert torch.isfinite(rollout["operator_gate"]).all()


def test_model_predictions_do_not_read_labels():
    model = NCORE(v2_cfg()).eval()
    batch = v2_batch()
    changed = dict(batch)
    changed["labels"] = 1.0 - batch["labels"]
    with torch.no_grad():
        first = model.sample_rollout(batch, deterministic=True)["logits"]
        second = model.sample_rollout(changed, deterministic=True)["logits"]
    assert torch.equal(first, second)
