import torch

from ncore.losses import (
    hard_case_weights,
    no_harm_loss,
    pairwise_ranking_loss,
)
from ncore.rl.grpo import per_sample_masked_bce
from ncore.training import residual_supervised_objective


def test_no_harm_is_zero_when_final_is_better():
    target = torch.ones(2, 1)
    mask = torch.ones_like(target)
    assert no_harm_loss(torch.ones(2, 1), torch.zeros(2, 1), target, mask) == 0


def test_no_harm_penalizes_degradation_and_detaches_direct():
    target = torch.ones(2, 1)
    mask = torch.ones_like(target)
    direct = torch.ones(2, 1, requires_grad=True)
    final = torch.zeros(2, 1, requires_grad=True)
    loss = no_harm_loss(final, direct, target, mask)
    loss.backward()
    assert loss > 0
    assert direct.grad is None
    assert final.grad is not None


def test_ranking_loss_is_zero_without_positive_negative_pair():
    logits = torch.randn(4, 1)
    target = torch.ones(4, 1)
    mask = torch.ones_like(target)
    assert pairwise_ranking_loss(logits, target, mask) == 0


def test_ranking_loss_rewards_correct_order():
    target = torch.tensor([[1.0], [0.0]])
    mask = torch.ones_like(target)
    correct = pairwise_ranking_loss(torch.tensor([[2.0], [-2.0]]), target, mask)
    reversed_order = pairwise_ranking_loss(
        torch.tensor([[-2.0], [2.0]]), target, mask
    )
    assert correct < reversed_order


def test_hard_case_gamma_zero_returns_exact_unit_weights():
    losses = torch.tensor([0.1, 1.0, 3.0])
    assert torch.equal(hard_case_weights(losses, gamma=0), torch.ones(3))


def test_hard_case_weights_are_detached_and_clipped():
    losses = torch.tensor([0.1, 1.0, 30.0], requires_grad=True)
    weights = hard_case_weights(losses, gamma=1.0, clip=2.0)
    assert not weights.requires_grad
    assert weights.max() <= 3.0
    assert weights[-1] > weights[0]


def test_gamma_zero_objective_matches_original_masked_bce():
    logits = torch.tensor([[0.2], [-0.5]], requires_grad=True)
    outputs = {
        "logits": logits,
        "direct_logits": torch.zeros_like(logits),
        "operator_delta_logits": torch.ones_like(logits),
        "operator_gate": torch.full((2, 1), 0.1),
    }
    batch = {
        "labels": torch.tensor([[1.0], [0.0]]),
        "label_mask": torch.ones(2, 1),
    }
    cfg = {"training": {"loss": {"hard_case_gamma": 0.0}}}
    actual, _ = residual_supervised_objective(
        outputs, batch, torch.ones(1), cfg
    )
    expected = per_sample_masked_bce(
        logits, batch["labels"], batch["label_mask"], torch.ones(1)
    ).mean()
    assert torch.equal(actual, expected)

