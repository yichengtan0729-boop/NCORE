from __future__ import annotations
import random
import numpy as np
import torch


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int, eps: float = 1e-6):
    while mask.ndim < x.ndim:
        mask = mask.unsqueeze(-1)
    num = (x * mask).sum(dim=dim)
    den = mask.sum(dim=dim)
    return num / (den + eps)
