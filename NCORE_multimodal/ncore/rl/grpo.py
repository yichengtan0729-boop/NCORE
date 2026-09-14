from __future__ import annotations
import torch
import torch.nn.functional as F


def per_sample_masked_bce(logits, targets, mask, pos_weight=None):
    """Per-sample BCE with a fixed, training-derived positive-class weight."""
    if pos_weight is not None:
        pos_weight = torch.as_tensor(
            pos_weight, device=logits.device, dtype=logits.dtype
        )
    loss = F.binary_cross_entropy_with_logits(
        logits, targets, reduction="none", pos_weight=pos_weight
    )
    return (loss * mask).sum(1) / (mask.sum(1).clamp_min(1.0) + 1e-6)


def _reward_config(cfg):
    return cfg.get("reward", cfg.get("rl", {}).get("reward", {}))


def prediction_delta_reward(final_logits, direct_logits, targets, mask,
                            pos_weight=None):
    final_score = -per_sample_masked_bce(
        final_logits, targets, mask, pos_weight=pos_weight
    )
    direct_score = -per_sample_masked_bce(
        direct_logits, targets, mask, pos_weight=pos_weight
    )
    return final_score - direct_score


def reward_components(model, batch, rollout, cfg, pos_weight=None,
                      compute_faithfulness=True):
    reward_cfg = _reward_config(cfg)
    targets, label_mask = batch["labels"], batch["label_mask"]
    final_score = -per_sample_masked_bce(
        rollout["logits"], targets, label_mask, pos_weight=pos_weight
    )
    direct_score = -per_sample_masked_bce(
        rollout["direct_logits"], targets, label_mask, pos_weight=pos_weight
    )
    raw_prediction_delta = final_score - direct_score
    if bool(cfg.get("model", {}).get("performance_v5", {}).get("enabled", False)):
        valid_count = label_mask.sum(1).clamp_min(1.0)
        signed = targets * 2.0 - 1.0
        logit_gain = rollout["logits"] - rollout["direct_logits"]
        auc_proxy_gain = (signed * logit_gain * label_mask).sum(1) / valid_count
        direct_probability = torch.sigmoid(rollout["direct_logits"].detach())
        pr_factor = torch.where(
            targets > 0.5,
            1.0 + (1.0 - direct_probability),
            1.0 + direct_probability,
        )
        pr_proxy_gain = (
            signed * logit_gain * pr_factor * label_mask
        ).sum(1) / valid_count
        direct_loss = per_sample_masked_bce(
            rollout["direct_logits"], targets, label_mask, pos_weight=pos_weight
        )
        hard_threshold = torch.quantile(direct_loss.detach(), 0.60)
        hard_case_bonus = (direct_loss >= hard_threshold).to(logit_gain.dtype) * torch.relu(
            raw_prediction_delta
        )
        length = rollout["length"]
        length_penalty = 0.0001 * length
        total = (
            float(reward_cfg.get("bce_gain", 0.45)) * raw_prediction_delta
            + float(reward_cfg.get("auroc_proxy_gain", 0.30)) * auc_proxy_gain
            + float(reward_cfg.get("pr_proxy_gain", 0.20)) * pr_proxy_gain
            + float(reward_cfg.get("hard_case_bonus", 0.05)) * hard_case_bonus
            - length_penalty
        )
        zeros = torch.zeros_like(total)
        return {
            "total": total,
            "pred": raw_prediction_delta,
            "prediction_delta": raw_prediction_delta,
            "raw_prediction_gain": raw_prediction_delta,
            "scaled_prediction_gain": raw_prediction_delta,
            "auroc_proxy_gain": auc_proxy_gain,
            "pr_proxy_gain": pr_proxy_gain,
            "hard_case_bonus": hard_case_bonus,
            "order": zeros,
            "order_contribution": zeros,
            "evidence": rollout["evidence"],
            "faithfulness": zeros,
            "length": length,
            "length_penalty": length_penalty,
            "entropy_bonus": zeros,
            "final_score": final_score,
            "direct_score": direct_score,
        }
    prediction_temperature = max(
        float(reward_cfg.get("prediction_temperature", 1.0)), 1e-6
    )
    scaled_prediction_delta = raw_prediction_delta / prediction_temperature

    reversed_actions = model.reverse_actions(rollout["actions"])
    reversed_rollout = model.rollout_actions(
        batch,
        reversed_actions,
        encoded=rollout["encoded"],
        operators=rollout["operators"],
    )
    reversed_score = -per_sample_masked_bce(
        reversed_rollout["logits"], targets, label_mask, pos_weight=pos_weight
    )
    order = final_score - reversed_score

    evidence = rollout["evidence"]
    length = rollout["length"] / float(max(model.max_steps, 1))
    faithfulness = torch.zeros_like(final_score)
    if compute_faithfulness and float(reward_cfg.get("faithfulness", 0.0)) > 0:
        counterfactual_encoded = model.ablate_selected_evidence(
            rollout["encoded"], rollout["actions"]
        )
        counterfactual_operators = model.build_operators(
            counterfactual_encoded, batch["modality_mask"]
        )
        counterfactual = model.rollout_actions(
            batch,
            rollout["actions"],
            encoded=counterfactual_encoded,
            operators=counterfactual_operators,
        )
        counterfactual_score = -per_sample_masked_bce(
            counterfactual["logits"], targets, label_mask, pos_weight=pos_weight
        )
        faithfulness = final_score - counterfactual_score

    prediction_weight = float(
        reward_cfg.get("prediction_delta", reward_cfg.get("pred", 1.0))
    )
    order_contribution = float(reward_cfg.get("order", 0.0)) * order
    length_penalty = float(reward_cfg.get("length", 0.0)) * length
    entropy_bonus = float(reward_cfg.get("entropy_bonus", 0.0)) * rollout[
        "entropy"
    ]
    total = (
        prediction_weight * scaled_prediction_delta
        + order_contribution
        + float(reward_cfg.get("evidence", 0.0)) * evidence
        + float(reward_cfg.get("faithfulness", 0.0)) * faithfulness
        - length_penalty
        + entropy_bonus
    )
    return {
        "total": total,
        "pred": raw_prediction_delta,
        "prediction_delta": raw_prediction_delta,
        "raw_prediction_gain": raw_prediction_delta,
        "scaled_prediction_gain": scaled_prediction_delta,
        "order": order,
        "order_contribution": order_contribution,
        "evidence": evidence,
        "faithfulness": faithfulness,
        "length": length,
        "length_penalty": length_penalty,
        "entropy_bonus": entropy_bonus,
        "final_score": final_score,
        "direct_score": direct_score,
    }


def group_advantages(rewards: torch.Tensor, group_size: int, eps=1e-6):
    rewards = rewards.view(-1, group_size)
    mean = rewards.mean(1, keepdim=True)
    std = rewards.std(1, keepdim=True, unbiased=False)
    return ((rewards - mean) / (std + eps)).reshape(-1)


def clipped_grpo_loss(new_logprob, old_logprob, advantage, entropy,
                      clip_eps=0.2, entropy_weight=0.005):
    ratio = torch.exp(new_logprob - old_logprob)
    unclipped = ratio * advantage
    clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantage
    return -torch.min(unclipped, clipped).mean() - entropy_weight * entropy.mean()
