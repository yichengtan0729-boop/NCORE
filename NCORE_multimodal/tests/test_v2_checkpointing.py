import torch

from ncore.checkpointing import load_compatible_model_state
from ncore.config import load_config


def test_compatible_loader_loads_matches_and_skips_shape_changes():
    model = torch.nn.Sequential(torch.nn.Linear(3, 2))
    checkpoint = {
        "model": {
            "0.weight": torch.full((2, 3), 2.0),
            "0.bias": torch.zeros(3),
            "retired.weight": torch.ones(1),
        }
    }
    report = load_compatible_model_state(model, checkpoint)
    assert torch.equal(model[0].weight, torch.full((2, 3), 2.0))
    assert report["loaded_keys"] == ["0.weight"]
    assert report["skipped_incompatible_keys"][0]["key"] == "0.bias"
    assert report["unexpected_keys"] == ["retired.weight"]


def test_recursive_v2_mortality_config_preserves_v1_and_overrides_v2():
    cfg = load_config("configs/ncore_strong_residual_v2_mortality.yaml")
    assert cfg["experiment"]["name"] == "ncore_strong_residual_v2"
    assert cfg["experiment"]["task_names"] == ["mortality"]
    assert cfg["model"]["encoders"]["cxr"]["input_dim"] == 512
    assert cfg["model"]["residual"]["gate_mode"] == "patient_specific"
    assert cfg["training"]["lr_ncore"] == 1e-3
