from __future__ import annotations

import itertools
import math

import torch
import torch.nn as nn

from .v2 import probability_to_logit


class BoundedScalarGate(nn.Module):
    def __init__(self, maximum: float, initial: float):
        super().__init__()
        self.maximum = float(maximum)
        if not 0.0 <= float(initial) <= self.maximum:
            raise ValueError("bounded gate initial value must lie inside its range")
        ratio = float(initial) / max(self.maximum, 1e-12)
        self.logit = nn.Parameter(torch.tensor(probability_to_logit(ratio)))
        self.override = None

    def value(self):
        if self.override is not None:
            return self.logit.new_tensor(float(self.override))
        return self.maximum * torch.sigmoid(self.logit)

    def set_override(self, value=None):
        if value is not None and not 0.0 <= float(value) <= self.maximum:
            raise ValueError("gate override is outside its bounded range")
        self.override = value

    def forward(self, delta):
        return self.value() * delta


class ResidualAdapter(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
        )
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, value):
        return self.norm(self.network(value))


class PairwiseInteractionResidual(nn.Module):
    def __init__(self, input_dims, num_tasks: int, cfg):
        super().__init__()
        self.modalities = list(input_dims)
        self.pairs = list(itertools.combinations(self.modalities, 2))
        dim = int(cfg.get("dim", 256))
        rank_dim = int(cfg.get("rank_dim", 64))
        dropout = float(cfg.get("dropout", 0.1))
        self.adapters = nn.ModuleDict(
            {name: ResidualAdapter(input_dims[name], dim, dropout) for name in self.modalities}
        )
        self.low_rank_left = nn.ModuleDict()
        self.low_rank_right = nn.ModuleDict()
        for index, _ in enumerate(self.pairs):
            self.low_rank_left[str(index)] = nn.Linear(dim, rank_dim, bias=False)
            self.low_rank_right[str(index)] = nn.Linear(dim, rank_dim, bias=False)
        pair_dim = dim * 2 + rank_dim + 1
        hidden_dim = int(cfg.get("hidden_dim", dim))
        self.head = nn.Sequential(
            nn.LayerNorm(len(self.pairs) * pair_dim + len(self.pairs)),
            nn.Linear(len(self.pairs) * pair_dim + len(self.pairs), hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_tasks),
        )
        self.gate = BoundedScalarGate(
            float(cfg.get("gate_max", 0.30)), float(cfg.get("gate_init", 0.05))
        )

    def forward(self, features, modality_mask):
        adapted = {name: self.adapters[name](features[name]) for name in self.modalities}
        pair_features, pair_masks = [], []
        for pair_index, (left_name, right_name) in enumerate(self.pairs):
            left_index = self.modalities.index(left_name)
            right_index = self.modalities.index(right_name)
            pair_mask = (
                modality_mask[:, left_index:left_index + 1]
                * modality_mask[:, right_index:right_index + 1]
            ).to(adapted[left_name].dtype)
            left, right = adapted[left_name], adapted[right_name]
            bilinear = self.low_rank_left[str(pair_index)](left) * self.low_rank_right[
                str(pair_index)
            ](right)
            cosine = torch.nn.functional.cosine_similarity(
                left.float(), right.float(), dim=-1, eps=1e-6
            ).to(left.dtype).unsqueeze(-1)
            pair = torch.cat(
                [left * right, (left - right).abs(), bilinear, cosine], dim=-1
            )
            pair_features.append(pair * pair_mask)
            pair_masks.append(pair_mask)
        head_input = torch.cat(pair_features + pair_masks, dim=-1)
        delta = self.head(head_input)
        any_pair = torch.cat(pair_masks, dim=1).sum(1, keepdim=True).gt(0)
        delta = torch.where(any_pair, delta, torch.zeros_like(delta))
        return self.gate(delta), delta, adapted, torch.cat(pair_masks, dim=1)


class CrossModalGatedPoolingResidual(nn.Module):
    def __init__(self, dim: int, num_modalities: int, num_tasks: int, cfg):
        super().__init__()
        self.num_modalities = int(num_modalities)
        score_hidden = int(cfg.get("score_hidden_dim", 128))
        self.score = nn.Sequential(
            nn.LayerNorm(dim + num_tasks + 1),
            nn.Linear(dim + num_tasks + 1, score_hidden),
            nn.GELU(),
            nn.Linear(score_hidden, 1),
        )
        head_hidden = int(cfg.get("hidden_dim", dim))
        self.head = nn.Sequential(
            nn.LayerNorm(dim * 3 + num_modalities),
            nn.Linear(dim * 3 + num_modalities, head_hidden),
            nn.GELU(),
            nn.Dropout(float(cfg.get("dropout", 0.1))),
            nn.Linear(head_hidden, num_tasks),
        )
        self.gate = BoundedScalarGate(
            float(cfg.get("gate_max", 0.20)), float(cfg.get("gate_init", 0.03))
        )

    def forward(self, adapted_features, modality_mask, direct_probability):
        tokens = torch.stack(list(adapted_features.values()), dim=1)
        batch_size, modalities, _ = tokens.shape
        confidence = direct_probability.unsqueeze(1).expand(-1, modalities, -1)
        availability = modality_mask.unsqueeze(-1).to(tokens.dtype)
        score_input = torch.cat([tokens, confidence, availability], dim=-1)
        scores = self.score(score_input).squeeze(-1)
        scores = scores.masked_fill(modality_mask <= 0, -1e9)
        weights = torch.softmax(scores.float(), dim=1).to(tokens.dtype)
        weights = weights * modality_mask.to(weights.dtype)
        weights = weights / weights.sum(1, keepdim=True).clamp_min(1e-6)
        pooled = (tokens * weights.unsqueeze(-1)).sum(1)
        valid = modality_mask.unsqueeze(-1).gt(0)
        maximum = tokens.masked_fill(~valid, -torch.inf).max(1).values
        maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
        mean = (tokens * valid.to(tokens.dtype)).sum(1) / valid.sum(1).clamp_min(1)
        delta = self.head(
            torch.cat([pooled, maximum, mean, modality_mask.to(tokens.dtype)], dim=-1)
        )
        has_modality = modality_mask.sum(1, keepdim=True).gt(0)
        delta = torch.where(has_modality, delta, torch.zeros_like(delta))
        return self.gate(delta), delta, weights


class ReasonGate(nn.Module):
    def __init__(self, input_dim: int, cfg):
        super().__init__()
        hidden_dim = int(cfg.get("hidden_dim", 128))
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(float(cfg.get("dropout", 0.1))),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state):
        return torch.sigmoid(self.network(state)).clamp(1e-6, 1.0 - 1e-6)


class RunningUtilityNormalizer(nn.Module):
    def __init__(self, momentum: float = 0.95, eps: float = 1e-6):
        super().__init__()
        self.momentum = float(momentum)
        self.eps = float(eps)
        self.register_buffer("mean", torch.tensor(0.0))
        self.register_buffer("variance", torch.tensor(1.0))
        self.register_buffer("initialized", torch.tensor(False))

    @torch.no_grad()
    def update(self, utility):
        finite = utility.detach().float()[torch.isfinite(utility.detach().float())]
        if finite.numel() == 0:
            return
        batch_mean = finite.mean()
        batch_variance = finite.var(unbiased=False).clamp_min(self.eps)
        if not bool(self.initialized):
            self.mean.copy_(batch_mean)
            self.variance.copy_(batch_variance)
            self.initialized.fill_(True)
        else:
            self.mean.mul_(self.momentum).add_(batch_mean, alpha=1 - self.momentum)
            self.variance.mul_(self.momentum).add_(
                batch_variance, alpha=1 - self.momentum
            )

    def forward(self, utility, *, update=False):
        if update:
            self.update(utility)
        normalized = (utility - self.mean) / torch.sqrt(self.variance + self.eps)
        return torch.nan_to_num(normalized, nan=0.0, posinf=10.0, neginf=-10.0)


def bounded_correction(raw_correction, bound: float = 0.35):
    bound = max(float(bound), 1e-6)
    return bound * torch.tanh(raw_correction / bound)
