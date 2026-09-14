from __future__ import annotations
import math
import torch
import torch.nn as nn


def probability_to_logit(probability: float) -> float:
    probability = min(max(float(probability), 1e-6), 1.0 - 1e-6)
    return math.log(probability / (1.0 - probability))


def binary_uncertainty_features(logits: torch.Tensor, eps: float = 1e-6):
    scaled = logits.float().clamp(-30.0, 30.0)
    probabilities = torch.sigmoid(scaled).clamp(eps, 1.0 - eps)
    entropy = -(
        probabilities * torch.log(probabilities)
        + (1.0 - probabilities)
        * torch.log(1.0 - probabilities)
    )
    margin = ((probabilities - 0.5).abs() * 2.0).clamp(0.0, 1.0)
    return probabilities, entropy, margin


class ResidualModalityAttentionEnhancer(nn.Module):
    """Light attention block that is exactly the identity when its gate is zero."""

    def __init__(self, fusion_dim: int, hidden_dim: int, cfg):
        super().__init__()
        num_heads = int(cfg.get("num_heads", 4))
        if fusion_dim % num_heads != 0:
            raise ValueError("direct fusion_dim must be divisible by attention num_heads")
        layer = nn.TransformerEncoderLayer(
            d_model=fusion_dim,
            nhead=num_heads,
            dim_feedforward=int(cfg.get("ff_dim", fusion_dim * 2)),
            dropout=float(cfg.get("dropout", 0.1)),
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=int(cfg.get("num_layers", 1))
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, fusion_dim))
        nn.init.normal_(self.cls_token, std=0.02)
        self.to_hidden = nn.Linear(fusion_dim, hidden_dim)
        self.candidate_norm = nn.LayerNorm(hidden_dim)
        gate_init = float(cfg.get("residual_gate_init", 0.10))
        self.gate_logit = nn.Parameter(torch.tensor(probability_to_logit(gate_init)))

    def gate_value(self):
        return torch.sigmoid(self.gate_logit)

    def forward(self, projected_features, modality_mask, base_hidden):
        tokens = torch.stack(projected_features, dim=1)
        tokens = tokens * modality_mask.unsqueeze(-1).to(tokens.dtype)
        cls = self.cls_token.expand(tokens.size(0), -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        cls_valid = torch.zeros(
            tokens.size(0), 1, dtype=torch.bool, device=tokens.device
        )
        padding_mask = torch.cat([cls_valid, modality_mask <= 0], dim=1)
        attention_hidden = self.encoder(
            tokens, src_key_padding_mask=padding_mask
        )[:, 0]
        candidate = self.candidate_norm(
            base_hidden + self.to_hidden(attention_hidden)
        )
        gate = self.gate_value()
        enhanced = base_hidden + gate * (candidate - base_hidden)
        return enhanced, attention_hidden, gate


class DirectAwareResidualHead(nn.Module):
    """Predict an operator correction from direct/operator feature interactions."""

    def __init__(self, direct_dim: int, operator_dim: int, num_tasks: int, cfg):
        super().__init__()
        residual_dim = int(cfg.get("residual_dim", 256))
        hidden_dim = int(cfg.get("hidden_dim", residual_dim))
        dropout = float(cfg.get("dropout", 0.1))
        self.direct_projection = nn.Linear(direct_dim, residual_dim)
        self.operator_projection = nn.Linear(operator_dim, residual_dim)
        self.input_norm = nn.LayerNorm(residual_dim * 4)
        self.interaction = nn.Sequential(
            nn.Linear(residual_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, residual_dim),
            nn.GELU(),
        )
        self.delta_head = nn.Linear(residual_dim, num_tasks)
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)

    def forward(self, direct_hidden, operator_hidden):
        direct = self.direct_projection(direct_hidden)
        operator = self.operator_projection(operator_hidden)
        interaction_input = torch.cat(
            [direct, operator, direct * operator, (direct - operator).abs()], dim=-1
        )
        interaction_hidden = self.interaction(self.input_norm(interaction_input))
        return self.delta_head(interaction_hidden), interaction_hidden


class PatientSpecificOperatorGate(nn.Module):
    """Bounded label-free per-patient residual gate."""

    def __init__(self, input_dim: int, cfg):
        super().__init__()
        hidden_dim = int(cfg.get("gate_hidden_dim", 128))
        dropout = float(cfg.get("gate_dropout", 0.1))
        self.alpha_min = float(cfg.get("alpha_min", 0.0))
        self.alpha_max = float(cfg.get("alpha_max", 0.35))
        self.alpha_init = float(cfg.get("alpha_init", 0.08))
        if not 0.0 <= self.alpha_min < self.alpha_max:
            raise ValueError("alpha_min must be non-negative and below alpha_max")
        if not self.alpha_min <= self.alpha_init <= self.alpha_max:
            raise ValueError("alpha_init must be between alpha_min and alpha_max")
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.input_norm = nn.LayerNorm(input_dim)
        ratio = (self.alpha_init - self.alpha_min) / (
            self.alpha_max - self.alpha_min
        )
        with torch.no_grad():
            nn.init.zeros_(self.network[-1].weight)
            self.network[-1].bias.fill_(probability_to_logit(ratio))

    def forward(self, features):
        return self.alpha_min + (self.alpha_max - self.alpha_min) * torch.sigmoid(
            self.network(self.input_norm(features))
        )

    def normalized(self, alpha):
        return (alpha - self.alpha_min) / (
            self.alpha_max - self.alpha_min + 1e-6
        )
