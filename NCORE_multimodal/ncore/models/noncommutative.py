from __future__ import annotations
import itertools
from typing import List, Sequence
import torch
import torch.nn as nn


def enumerate_words(num_modalities: int, max_degree: int, include_identity=True):
    words = [()] if include_identity else []
    for d in range(1, max_degree+1):
        words.extend(itertools.product(range(num_modalities), repeat=d))
    return words


def apply_word(state: torch.Tensor, operators: Sequence[torch.Tensor], word):
    x = state
    for idx in word:
        x = torch.bmm(operators[idx], x)
    return x


class GlobalNonCommutativeFilter(nn.Module):
    """GtNN-like baseline: globally learned coefficients over ordered modality words."""
    def __init__(self, num_modalities: int, max_degree: int = 2):
        super().__init__()
        self.words = enumerate_words(num_modalities, max_degree, include_identity=True)
        self.coeff = nn.Parameter(torch.zeros(len(self.words)))
        with torch.no_grad(): self.coeff[0] = 1.0
    def forward(self, state, operators):
        weights = torch.softmax(self.coeff, dim=0)
        out = 0.0
        for w, c in zip(self.words, weights):
            out = out + c * apply_word(state, operators, w)
        return out
