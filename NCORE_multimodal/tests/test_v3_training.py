import numpy as np
import pytest
import torch

from ncore.calibration import (
    binary_calibration_metrics,
    fit_temperature,
    load_temperature,
    save_temperature,
)
from ncore.ema import ModelEMA
from ncore.rl.grpo import prediction_delta_reward, reward_components
from ncore.training import balance_policy_targets


def _simple_batch(size=10):
    return {
        "modalities": {"x": torch.arange(size).float().unsqueeze(1)},
        "modality_mask": torch.ones(size, 1),
        "labels": torch.zeros(size, 1),
        "label_mask": torch.ones(size, 1),
        "meta": [{"index": index} for index in range(size)],
    }


def test_balanced_policy_targets_hit_requested_nonstop_fraction():
    batch = _simple_batch()
    paths = torch.zeros(10, 2, dtype=torch.long)
    useful = torch.tensor([True, True] + [False] * 8)
    balanced, actions, _ = balance_policy_targets(
        batch, paths, useful, stop_idx=24, target_nonstop_fraction=0.5
    )
    assert len(balanced["meta"]) == 10
    assert actions[:, 0].ne(24).float().mean().item() == 0.5


def test_balanced_policy_targets_use_all_stop_when_no_useful_case():
    batch = _simple_batch()
    paths = torch.zeros(10, 2, dtype=torch.long)
    _, actions, _ = balance_policy_targets(
        batch,
        paths,
        torch.zeros(10, dtype=torch.bool),
        stop_idx=24,
        target_nonstop_fraction=0.5,
    )
    assert torch.equal(actions, torch.full_like(actions, 24))


def test_prediction_gain_sign_is_preserved():
    target = torch.ones(2, 1)
    mask = torch.ones_like(target)
    direct = torch.zeros_like(target)
    assert torch.all(
        prediction_delta_reward(torch.ones_like(target), direct, target, mask) > 0
    )
    assert torch.all(
        prediction_delta_reward(-torch.ones_like(target), direct, target, mask) < 0
    )


class _RewardModel:
    max_steps = 2

    def reverse_actions(self, actions):
        return actions

    def rollout_actions(self, batch, actions, encoded=None, operators=None):
        return self.rollout


def test_reward_temperature_scaling_and_length_penalty():
    model = _RewardModel()
    batch = {
        "labels": torch.ones(1, 1),
        "label_mask": torch.ones(1, 1),
        "modality_mask": torch.ones(1, 1),
    }
    rollout = {
        "logits": torch.ones(1, 1),
        "direct_logits": torch.zeros(1, 1),
        "actions": torch.zeros(1, 2, dtype=torch.long),
        "encoded": {},
        "operators": {},
        "evidence": torch.zeros(1),
        "length": torch.ones(1),
        "entropy": torch.zeros(1),
    }
    model.rollout = rollout
    cfg = {
        "reward": {
            "prediction_temperature": 0.01,
            "prediction_delta": 1.0,
            "length": 0.0001,
        }
    }
    components = reward_components(
        model, batch, rollout, cfg, compute_faithfulness=False
    )
    assert torch.allclose(
        components["scaled_prediction_gain"],
        components["raw_prediction_gain"] / 0.01,
    )
    assert components["length_penalty"].item() == pytest.approx(0.00005)


def test_temperature_fit_rejects_non_validation_split():
    with pytest.raises(ValueError):
        fit_temperature(
            torch.zeros(2, 1),
            torch.zeros(2, 1),
            torch.ones(2, 1),
            split="test",
        )


def test_temperature_round_trip_requires_validation_metadata(tmp_path):
    path = tmp_path / "direct_temperature.json"
    save_temperature(path, 1.7, fit_split="val")
    assert load_temperature(path) == pytest.approx(1.7)
    save_temperature(path, 1.7, fit_split="test")
    with pytest.raises(ValueError):
        load_temperature(path)


def test_temperature_scaling_preserves_auroc_ordering():
    logits = np.array([-2.0, -0.5, 0.2, 3.0])
    original = 1.0 / (1.0 + np.exp(-logits))
    calibrated = 1.0 / (1.0 + np.exp(-logits / 2.5))
    assert np.array_equal(np.argsort(original), np.argsort(calibrated))


def test_calibration_metrics_are_finite():
    metrics = binary_calibration_metrics(
        np.array([0, 1, 0, 1]), np.array([0.1, 0.8, 0.3, 0.9])
    )
    assert set(metrics) == {"ece", "brier", "nll"}
    assert all(np.isfinite(value) for value in metrics.values())


def test_ema_update_and_policy_exclusion():
    model = torch.nn.Module()
    model.predictor = torch.nn.Linear(1, 1, bias=False)
    model.policy = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.predictor.weight.zero_()
        model.policy.weight.zero_()
    ema = ModelEMA(model, decay=0.5)
    with torch.no_grad():
        model.predictor.weight.fill_(2.0)
        model.policy.weight.fill_(2.0)
    ema.update(model)
    assert ema.shadow["predictor.weight"].item() == pytest.approx(1.0)
    assert "policy.weight" not in ema.shadow


def test_ema_context_restores_raw_parameters():
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.zero_()
    ema = ModelEMA(model, decay=0.5)
    ema.shadow["weight"].fill_(3.0)
    with ema.average_parameters(model):
        assert model.weight.item() == pytest.approx(3.0)
    assert model.weight.item() == pytest.approx(0.0)
