from __future__ import annotations
import torch
import torch.nn.functional as F

from .rl.grpo import per_sample_masked_bce


def no_harm_loss_from_per_sample(final_loss, direct_loss, valid_samples=None):
    penalty = torch.relu(final_loss - direct_loss.detach())
    if valid_samples is not None:
        penalty = penalty[valid_samples]
    if penalty.numel() == 0:
        return final_loss.sum() * 0.0
    return penalty.mean()


def no_harm_loss(final_logits, direct_logits, targets, label_mask,
                 pos_weight=None):
    final_loss = per_sample_masked_bce(
        final_logits, targets, label_mask, pos_weight
    )
    direct_loss = per_sample_masked_bce(
        direct_logits, targets, label_mask, pos_weight
    )
    valid = label_mask.sum(1) > 0
    return no_harm_loss_from_per_sample(final_loss, direct_loss, valid)


def alpha_detached_no_harm_loss(direct_logits, delta_logits, alpha, targets,
                                label_mask, pos_weight=None):
    logits_no_harm = direct_logits + alpha.detach() * delta_logits
    final_loss = per_sample_masked_bce(
        logits_no_harm, targets, label_mask, pos_weight
    )
    direct_loss = per_sample_masked_bce(
        direct_logits, targets, label_mask, pos_weight
    ).detach()
    valid = label_mask.sum(1) > 0
    return no_harm_loss_from_per_sample(final_loss, direct_loss, valid)


def gate_utility_targets(direct_logits, delta_logits, targets, label_mask,
                         pos_weight=None, alpha_probe=0.15,
                         utility_temperature=0.10):
    direct_loss = per_sample_masked_bce(
        direct_logits, targets, label_mask, pos_weight
    )
    probe_logits = direct_logits + float(alpha_probe) * delta_logits.detach()
    probe_loss = per_sample_masked_bce(
        probe_logits, targets, label_mask, pos_weight
    )
    utility = (direct_loss - probe_loss).detach()
    temperature = max(float(utility_temperature), 1e-6)
    targets = torch.sigmoid(utility / (temperature + 1e-6)).detach()
    return utility, targets


def gate_supervision_loss(alpha, utility_targets, alpha_min, alpha_max,
                          valid_samples=None):
    span = max(float(alpha_max) - float(alpha_min), 1e-6)
    gate_probability = ((alpha - float(alpha_min)) / (span + 1e-6)).clamp(
        min=1e-6, max=1.0 - 1e-6
    ).squeeze(-1)
    if valid_samples is not None:
        gate_probability = gate_probability[valid_samples]
        utility_targets = utility_targets[valid_samples]
    if gate_probability.numel() == 0:
        return alpha.sum() * 0.0, gate_probability
    return (
        F.binary_cross_entropy(gate_probability, utility_targets),
        gate_probability,
    )


def residual_utility_ranking_loss(final_per_sample_loss,
                                  direct_per_sample_loss,
                                  hard_weights=None,
                                  valid_samples=None):
    improvement = direct_per_sample_loss.detach() - final_per_sample_loss
    penalties = F.softplus(-improvement)
    if hard_weights is not None:
        penalties = penalties * hard_weights.detach()
    if valid_samples is not None:
        penalties = penalties[valid_samples]
    if penalties.numel() == 0:
        return final_per_sample_loss.sum() * 0.0, improvement.detach()
    return penalties.mean(), improvement.detach()


def pairwise_ranking_loss(logits, targets, label_mask, max_pairs=1024):
    losses = []
    for task_index in range(logits.size(1)):
        valid = label_mask[:, task_index] > 0
        task_logits = logits[valid, task_index]
        task_targets = targets[valid, task_index]
        positive = task_logits[task_targets > 0.5]
        negative = task_logits[task_targets <= 0.5]
        if positive.numel() == 0 or negative.numel() == 0:
            continue
        differences = positive.unsqueeze(1) - negative.unsqueeze(0)
        pair_losses = F.softplus(-differences).reshape(-1)
        if max_pairs is not None and int(max_pairs) > 0:
            pair_losses = pair_losses[:int(max_pairs)]
        losses.append(pair_losses.mean())
    if not losses:
        return logits.sum() * 0.0
    return torch.stack(losses).mean()


def auc_ranking_loss(
    logits, targets, label_mask, margin=0.5, max_pairs=1024
):
    """Margin AUROC surrogate over valid positive/negative pairs."""
    losses = []
    for task_index in range(logits.size(1)):
        valid = label_mask[:, task_index] > 0
        scores = logits[valid, task_index]
        labels = targets[valid, task_index]
        positive = scores[labels > 0.5]
        negative = scores[labels <= 0.5]
        if positive.numel() == 0 or negative.numel() == 0:
            continue
        pair_losses = F.softplus(
            float(margin) - (positive.unsqueeze(1) - negative.unsqueeze(0))
        ).reshape(-1)
        if max_pairs is not None and int(max_pairs) > 0:
            pair_losses = pair_losses[: int(max_pairs)]
        losses.append(pair_losses.mean())
    return torch.stack(losses).mean() if losses else logits.sum() * 0.0


def precision_ranking_loss(
    logits,
    targets,
    label_mask,
    margin=0.75,
    top_fraction=0.25,
    positive_weight=2.0,
    max_pairs=1024,
):
    """A differentiable high-score-region ranking surrogate for AUPRC."""
    losses = []
    for task_index in range(logits.size(1)):
        valid = label_mask[:, task_index] > 0
        scores = logits[valid, task_index]
        labels = targets[valid, task_index]
        positive = scores[labels > 0.5]
        negative = scores[labels <= 0.5]
        if positive.numel() == 0 or negative.numel() == 0:
            continue
        hard_positive_count = max(1, int(round(positive.numel() * float(top_fraction))))
        hard_negative_count = max(1, int(round(negative.numel() * float(top_fraction))))
        hard_positive = torch.topk(
            positive, min(hard_positive_count, positive.numel()), largest=False
        ).values
        hard_negative = torch.topk(
            negative, min(hard_negative_count, negative.numel()), largest=True
        ).values
        pair_losses = F.softplus(
            float(margin)
            - (hard_positive.unsqueeze(1) - hard_negative.unsqueeze(0))
        ).reshape(-1)
        if max_pairs is not None and int(max_pairs) > 0:
            pair_losses = pair_losses[: int(max_pairs)]
        losses.append(float(positive_weight) * pair_losses.mean())
    return torch.stack(losses).mean() if losses else logits.sum() * 0.0


def hard_pair_mining_loss(
    logits, targets, label_mask, fraction=0.25, margin=0.5, max_pairs=1024
):
    return precision_ranking_loss(
        logits,
        targets,
        label_mask,
        margin=margin,
        top_fraction=fraction,
        positive_weight=1.0,
        max_pairs=max_pairs,
    )


def direct_performance_objective(logits, targets, label_mask, pos_weight, cfg):
    """v5 BCE + AUROC + AUPRC + hard-pair objective."""
    loss_cfg = cfg.get("training", {}).get("direct_loss", {})
    bce = per_sample_masked_bce(logits, targets, label_mask, pos_weight).mean()
    auc = auc_ranking_loss(
        logits,
        targets,
        label_mask,
        margin=float(loss_cfg.get("margin_auc", 0.5)),
        max_pairs=int(loss_cfg.get("max_pairs", 1024)),
    )
    pr = precision_ranking_loss(
        logits,
        targets,
        label_mask,
        margin=float(loss_cfg.get("margin_pr", 0.75)),
        top_fraction=float(loss_cfg.get("top_fraction", 0.25)),
        positive_weight=float(loss_cfg.get("positive_rank_weight", 2.0)),
        max_pairs=int(loss_cfg.get("max_pairs", 1024)),
    )
    hard = hard_pair_mining_loss(
        logits,
        targets,
        label_mask,
        fraction=float(loss_cfg.get("hard_fraction", 0.25)),
        margin=float(loss_cfg.get("margin_auc", 0.5)),
        max_pairs=int(loss_cfg.get("max_pairs", 1024)),
    )
    total = (
        bce
        + float(loss_cfg.get("auc_weight", 0.20)) * auc
        + float(loss_cfg.get("pr_weight", 0.15)) * pr
        + float(loss_cfg.get("hardpair_weight", 0.10)) * hard
    )
    return total, {"bce": bce, "auc_rank": auc, "pr_rank": pr, "hardpair": hard}


def quantile_gate_targets(normalized_utility, temperature=0.5):
    """Top quartile=1, bottom quartile=0, middle uses a soft normalized target."""
    utility = torch.nan_to_num(
        normalized_utility.detach().float(), nan=0.0, posinf=10.0, neginf=-10.0
    )
    soft = torch.sigmoid(utility / max(float(temperature), 1e-6))
    if utility.numel() < 2:
        return soft, soft
    lower = torch.quantile(utility, 0.25)
    upper = torch.quantile(utility, 0.75)
    rank = soft.clone()
    rank[utility <= lower] = 0.0
    rank[utility >= upper] = 1.0
    return soft.detach(), rank.detach()


def normalized_gate_supervision_loss(gate_probability, normalized_utility):
    soft, rank = quantile_gate_targets(normalized_utility)
    gate_probability = gate_probability.reshape(-1).clamp(1e-6, 1 - 1e-6)
    return (
        0.4 * F.binary_cross_entropy(gate_probability, soft)
        + 0.6 * F.binary_cross_entropy(gate_probability, rank)
    )


def reason_gate_loss(
    reason_probability,
    candidate_gain,
    *,
    focal_gamma=1.5,
    ranking_weight=0.10,
):
    """Weighted BCE + focal + ordering loss for WHETHER-to-reason."""
    probability = reason_probability.reshape(-1).clamp(1e-6, 1 - 1e-6)
    gain = torch.nan_to_num(candidate_gain.detach().reshape(-1).float())
    soft = torch.sigmoid(gain / 0.5)
    if gain.numel() < 2:
        target = soft.detach()
    else:
        median = torch.quantile(gain, 0.50)
        upper = torch.quantile(gain, 0.75)
        target = soft.clone()
        target[gain <= median] = 0.0
        target[gain >= upper] = 1.0
        target = target.detach()
    positive_weight = (target.numel() - target.sum()) / target.sum().clamp_min(1.0)
    weights = torch.where(target >= 0.5, positive_weight.clamp_min(1.0), 1.0)
    bce_each = F.binary_cross_entropy(probability, target, reduction="none")
    bce = (weights * bce_each).mean()
    pt = torch.where(target >= 0.5, probability, 1.0 - probability)
    focal = (weights * (1.0 - pt).pow(float(focal_gamma)) * bce_each).mean()
    ranking = auc_ranking_loss(
        torch.logit(probability).unsqueeze(1),
        (gain > 0).float().unsqueeze(1),
        torch.ones_like(gain).unsqueeze(1),
        margin=0.5,
    )
    return bce + focal + float(ranking_weight) * ranking, {
        "bce": bce,
        "focal": focal,
        "ranking": ranking,
        "soft_target": soft,
        "rank_target": target,
    }


def oracle_candidate_utility(bce_gain, rank_gain, pr_gain, eps=1e-6):
    """Combine per-sample candidate gains after within-group normalization."""
    values = []
    for gain in [bce_gain, rank_gain, pr_gain]:
        gain = torch.nan_to_num(gain.float())
        mean = gain.mean(dim=-1, keepdim=True)
        std = gain.std(dim=-1, keepdim=True, unbiased=False).clamp_min(eps)
        values.append((gain - mean) / std)
    return 0.50 * values[0] + 0.25 * values[1] + 0.25 * values[2]


def hard_case_weights(direct_per_sample_loss, gamma=1.0, clip=2.0,
                      valid_samples=None, eps=1e-6):
    difficulty = direct_per_sample_loss.detach()
    if valid_samples is None:
        valid_samples = torch.ones_like(difficulty, dtype=torch.bool)
    valid_difficulty = difficulty[valid_samples]
    if valid_difficulty.numel() == 0:
        return torch.ones_like(difficulty)
    mean_difficulty = valid_difficulty.mean().clamp_min(eps)
    hardness = (difficulty / (mean_difficulty + eps) - 1.0).clamp(
        min=0.0, max=float(clip)
    )
    weights = 1.0 + float(gamma) * hardness
    return torch.where(valid_samples, weights, torch.ones_like(weights)).detach()
