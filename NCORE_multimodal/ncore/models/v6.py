from __future__ import annotations

import torch
import torch.nn as nn


class CandidateReranker(nn.Module):
    """Label-free scorer for complete operator paths.

    Labels are deliberately absent from the public forward signature. Training
    code may derive utility targets from train labels, but validation/test
    inference can only pass model states and candidate metadata.
    """

    def __init__(
        self,
        direct_dim: int,
        operator_dim: int,
        num_tasks: int,
        num_modalities: int,
        num_concepts: int,
        max_steps: int,
        cfg,
    ):
        super().__init__()
        self.num_modalities = int(num_modalities)
        self.num_concepts = int(num_concepts)
        self.num_actions = self.num_modalities * self.num_concepts
        self.stop_idx = self.num_actions
        self.max_steps = int(max_steps)
        embedding_dim = int(cfg.get("path_embedding_dim", 64))
        hidden_dim = int(cfg.get("hidden_dim", 256))
        dropout = float(cfg.get("dropout", 0.1))
        self.action_embedding = nn.Embedding(self.num_actions + 1, embedding_dim)
        self.modality_embedding = nn.Embedding(self.num_modalities + 1, embedding_dim)
        self.position_embedding = nn.Embedding(self.max_steps, embedding_dim)
        feature_dim = (
            int(direct_dim)
            + 2 * int(num_tasks)
            + int(num_modalities)
            + embedding_dim
            + int(operator_dim)
            + 2
            + 2 * int(num_tasks)
            + 1
        )
        self.feature_norm = nn.LayerNorm(feature_dim)
        self.backbone = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )
        self.last_layer = nn.Linear(hidden_dim, 1)

    def encode_path(self, actions: torch.Tensor) -> torch.Tensor:
        actions = actions.long()
        valid = actions.ne(self.stop_idx)
        safe_actions = actions.clamp(0, self.stop_idx)
        modalities = torch.div(
            actions.clamp_max(self.stop_idx - 1),
            self.num_concepts,
            rounding_mode="floor",
        )
        modalities = torch.where(
            valid, modalities, torch.full_like(modalities, self.num_modalities)
        )
        positions = torch.arange(
            actions.size(1), device=actions.device, dtype=torch.long
        ).unsqueeze(0)
        tokens = (
            self.action_embedding(safe_actions)
            + self.modality_embedding(modalities)
            + self.position_embedding(positions)
        )
        weights = valid.to(tokens.dtype).unsqueeze(-1)
        return (tokens * weights).sum(1) / weights.sum(1).clamp_min(1.0)

    def build_features(
        self,
        direct_hidden: torch.Tensor,
        calibrated_probability: torch.Tensor,
        entropy: torch.Tensor,
        modality_mask: torch.Tensor,
        actions: torch.Tensor,
        operator_hidden: torch.Tensor,
        commutator_summary: torch.Tensor,
        residual_magnitude: torch.Tensor,
        confidence_change: torch.Tensor,
        path_length: torch.Tensor,
    ) -> torch.Tensor:
        path_embedding = self.encode_path(actions)
        features = torch.cat(
            [
                direct_hidden,
                calibrated_probability,
                entropy,
                modality_mask.to(direct_hidden.dtype),
                path_embedding,
                operator_hidden,
                commutator_summary,
                residual_magnitude,
                confidence_change,
                path_length.reshape(-1, 1).to(direct_hidden.dtype),
            ],
            dim=-1,
        )
        return torch.nan_to_num(
            self.feature_norm(features.float()).to(direct_hidden.dtype),
            nan=0.0,
            posinf=10.0,
            neginf=-10.0,
        )

    def forward(
        self,
        direct_hidden: torch.Tensor,
        calibrated_probability: torch.Tensor,
        entropy: torch.Tensor,
        modality_mask: torch.Tensor,
        actions: torch.Tensor,
        operator_hidden: torch.Tensor,
        commutator_summary: torch.Tensor,
        residual_magnitude: torch.Tensor,
        confidence_change: torch.Tensor,
        path_length: torch.Tensor,
    ) -> torch.Tensor:
        features = self.build_features(
            direct_hidden,
            calibrated_probability,
            entropy,
            modality_mask,
            actions,
            operator_hidden,
            commutator_summary,
            residual_magnitude,
            confidence_change,
            path_length,
        )
        return self.last_layer(self.backbone(features)).squeeze(-1)


def select_candidate_actions(
    candidate_actions: torch.Tensor, selected_index: torch.Tensor
) -> torch.Tensor:
    """Gather one complete path per patient from [B, N, S] candidates."""
    if candidate_actions.ndim != 3:
        raise ValueError("candidate_actions must have shape [batch, candidates, steps]")
    if selected_index.shape != (candidate_actions.size(0),):
        raise ValueError("selected_index must have shape [batch]")
    gather_index = selected_index[:, None, None].expand(
        -1, 1, candidate_actions.size(-1)
    )
    return candidate_actions.gather(1, gather_index).squeeze(1)
