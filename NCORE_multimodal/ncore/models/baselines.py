from __future__ import annotations
import torch
import torch.nn as nn


class ConcatBaseline(nn.Module):
    """MedFuse-style projected feature concatenation generalized to M modalities."""
    def __init__(self, encoder_dims, hidden_dim, num_tasks):
        super().__init__()
        self.proj = nn.ModuleList([nn.Linear(d, hidden_dim) for d in encoder_dims])
        self.head = nn.Sequential(nn.Linear(hidden_dim*len(encoder_dims), hidden_dim), nn.GELU(), nn.Linear(hidden_dim, num_tasks))
    def forward(self, feats, modality_mask):
        xs = []
        for i, (x,p) in enumerate(zip(feats, self.proj)):
            xs.append(p(x) * modality_mask[:, i:i+1])
        return self.head(torch.cat(xs, dim=-1))


class SequentialLSTMBaseline(nn.Module):
    """Fixed modality-order recurrent fusion, close in spirit to MedFuse fusion."""
    def __init__(self, encoder_dims, hidden_dim, num_tasks):
        super().__init__()
        self.proj = nn.ModuleList([nn.Linear(d, hidden_dim) for d in encoder_dims])
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, num_tasks)
    def forward(self, feats, modality_mask):
        x = torch.stack([p(f) for p,f in zip(self.proj, feats)], 1)
        x = x * modality_mask.unsqueeze(-1)
        lengths = modality_mask.sum(1).long().clamp_min(1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        _, (h,_) = self.lstm(packed)
        return self.head(h[-1])
