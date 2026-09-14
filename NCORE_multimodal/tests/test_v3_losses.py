import torch

from ncore.losses import (
    alpha_detached_no_harm_loss,
    gate_utility_targets,
    residual_utility_ranking_loss,
)
from ncore.training import _alpha_anchor_weight


def _harmful_case():
    direct = torch.tensor([[3.0]], requires_grad=True)
    delta = torch.tensor([[-20.0]], requires_grad=True)
    alpha_logit = torch.tensor([[0.0]], requires_grad=True)
    alpha = 0.03 + 0.17 * torch.sigmoid(alpha_logit)
    target = torch.ones(1, 1)
    mask = torch.ones_like(target)
    return direct, delta, alpha_logit, alpha, target, mask


def test_alpha_detached_no_harm_has_no_alpha_gradient():
    direct, delta, alpha_logit, alpha, target, mask = _harmful_case()
    loss = alpha_detached_no_harm_loss(
        direct, delta, alpha, target, mask
    )
    loss.backward()
    assert alpha_logit.grad is None


def test_alpha_detached_no_harm_still_trains_delta():
    direct, delta, _, alpha, target, mask = _harmful_case()
    loss = alpha_detached_no_harm_loss(
        direct, delta, alpha, target, mask
    )
    loss.backward()
    assert delta.grad is not None and delta.grad.abs().sum() > 0


def test_positive_utility_produces_target_above_half():
    utility, target = gate_utility_targets(
        torch.zeros(1, 1),
        torch.ones(1, 1) * 5,
        torch.ones(1, 1),
        torch.ones(1, 1),
        alpha_probe=0.15,
        utility_temperature=0.1,
    )
    assert utility.item() > 0 and target.item() > 0.5


def test_negative_utility_produces_target_below_half():
    utility, target = gate_utility_targets(
        torch.zeros(1, 1),
        torch.ones(1, 1) * 5,
        torch.zeros(1, 1),
        torch.ones(1, 1),
        alpha_probe=0.15,
        utility_temperature=0.1,
    )
    assert utility.item() < 0 and target.item() < 0.5


def test_utility_ranking_trains_final_loss_and_detaches_direct():
    final = torch.tensor([1.0, 2.0], requires_grad=True)
    direct = torch.tensor([0.5, 1.0], requires_grad=True)
    loss, _ = residual_utility_ranking_loss(final, direct)
    loss.backward()
    assert final.grad is not None
    assert direct.grad is None


def test_alpha_anchor_schedule_is_early_only():
    cfg = {
        "training": {
            "gate": {
                "anchor_weight": 0.05,
                "anchor_decay_epochs": 6,
            }
        }
    }
    weights = [_alpha_anchor_weight(cfg, epoch) for epoch in range(7)]
    assert weights[:3] == [0.05, 0.05, 0.05]
    assert weights[3] > weights[4] > weights[5]
    assert weights[5] == weights[6] == 0.0

