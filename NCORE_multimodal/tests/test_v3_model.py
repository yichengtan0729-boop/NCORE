import copy

import torch

from ncore.models.model import NCORE


def v3_cfg(task_names=None):
    task_names = task_names or ["mortality"]
    return {
        "paths": {"report_model_name_or_path": ""},
        "experiment": {"task_names": task_names},
        "model": {
            "modalities": ["cxr", "ecg", "ehr"],
            "hidden_dim": 24,
            "dropout": 0.0,
            "concept": {"num_concepts": 8, "concept_dim": 12},
            "encoders": {
                "cxr": {"backend": "precomputed", "input_dim": 6},
                "ecg": {"backend": "precomputed", "input_dim": 8},
                "ehr": {"backend": "precomputed", "input_dim": 10},
            },
            "direct_fusion": {
                "enabled": True,
                "mode": "residual_attention",
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
        },
        "training": {"loss": {}},
        "data": {},
        "reward": {"prediction_temperature": 0.01, "length": 0.0001},
        "calibration": {"temperature_scaling": True},
    }


def v3_batch(cfg=None, batch_size=4):
    cfg = cfg or v3_cfg()
    return {
        "modalities": {
            name: torch.randn(batch_size, spec["input_dim"])
            for name, spec in cfg["model"]["encoders"].items()
        },
        "modality_mask": torch.tensor(
            [[1, 1, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]],
            dtype=torch.float32,
        )[:batch_size],
        "labels": torch.randint(
            0, 2, (batch_size, len(cfg["experiment"]["task_names"]))
        ).float(),
        "label_mask": torch.ones(
            batch_size, len(cfg["experiment"]["task_names"])
        ),
        "meta": [{}] * batch_size,
    }


def test_v3_alpha_floor_ceiling_and_initial_mean():
    model = NCORE(v3_cfg()).eval()
    alpha = model(v3_batch(), deterministic=True)["operator_gate"]
    assert alpha.shape == (4, 1)
    assert alpha.min() >= 0.03
    assert alpha.max() <= 0.20
    assert torch.allclose(alpha.mean(), torch.tensor(0.08), atol=1e-6)


def test_alpha_min_zero_recovers_v2_gate_formula():
    cfg = v3_cfg()
    cfg["model"]["residual"]["alpha_min"] = 0.0
    cfg["model"]["residual"]["alpha_max"] = 0.35
    model = NCORE(cfg).eval()
    alpha = model(v3_batch(cfg), deterministic=True)["operator_gate"]
    assert torch.allclose(alpha, torch.full_like(alpha, 0.08), atol=1e-6)


def test_v3_attention_default_is_exact_base_direct():
    model = NCORE(v3_cfg()).eval()
    batch = v3_batch()
    direct = model.direct_outputs(model.encode(batch), batch["modality_mask"])
    assert model.direct_attention_enhancer is None
    assert torch.equal(direct["direct_logits"], direct["direct_base_logits"])


def test_calibrated_uncertainty_changes_probability_not_logit_order():
    model = NCORE(v3_cfg())
    logits = torch.tensor([[-2.0], [0.5], [3.0]])
    original, _, _ = model.direct_uncertainty_features(logits)
    model.set_direct_temperature(2.0)
    calibrated, _, _ = model.direct_uncertainty_features(logits)
    assert not torch.equal(original, calibrated)
    assert torch.equal(original.argsort(0), calibrated.argsort(0))


def test_v3_missing_operator_identity_and_missing_action_mask():
    model = NCORE(v3_cfg())
    batch = v3_batch()
    encoded = model.encode(batch)
    operators = model.build_operators(encoded)
    assert torch.equal(operators["cxr"][1], torch.eye(model.K))
    state = model.initial(4, torch.device("cpu"))
    previous = torch.full((4,), model.stop_idx + 1, dtype=torch.long)
    logits = model._policy_inputs(
        state, encoded, operators, batch["modality_mask"], previous, 0
    )
    assert torch.equal(torch.softmax(logits, -1)[1, :model.K], torch.zeros(model.K))


def test_v3_action_space_stop_and_mortality_shape():
    model = NCORE(v3_cfg())
    rollout = model(v3_batch(), deterministic=True)
    assert model.M == 3 and model.K == 8 and model.num_actions == 25
    assert model.stop_idx == 24
    assert rollout["logits"].shape == (4, 1)


def test_old_v2_style_config_still_forwards():
    cfg = v3_cfg()
    cfg["calibration"] = {"temperature_scaling": False}
    cfg["model"]["residual"].pop("alpha_min")
    cfg["model"]["residual"].pop("include_logit_magnitude")
    cfg["model"]["residual"]["alpha_max"] = 0.35
    cfg["model"]["policy"].pop("include_logit_magnitude")
    output = NCORE(copy.deepcopy(cfg))(v3_batch(cfg), deterministic=True)
    assert torch.isfinite(output["logits"]).all()

