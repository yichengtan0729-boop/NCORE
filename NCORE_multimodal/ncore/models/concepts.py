from __future__ import annotations
import torch
import torch.nn as nn


class ConceptProjector(nn.Module):
    """Map one modality vector to K concept nodes and confidence scores."""
    def __init__(self, in_dim: int, num_concepts: int, concept_dim: int, dropout=0.1):
        super().__init__()
        self.num_concepts = num_concepts
        self.concept_dim = concept_dim
        self.norm = nn.LayerNorm(in_dim)
        self.to_nodes = nn.Sequential(
            nn.Linear(in_dim, num_concepts * concept_dim),
            nn.GELU(), nn.Dropout(dropout)
        )
        self.conf_head = nn.Linear(concept_dim, 1)
    def forward(self, h):
        b = h.shape[0]
        z = self.to_nodes(self.norm(h)).view(b, self.num_concepts, self.concept_dim)
        conf = torch.sigmoid(self.conf_head(z).squeeze(-1))
        return z, conf
