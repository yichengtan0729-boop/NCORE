from __future__ import annotations

import math
from typing import Iterator, Sequence

import numpy as np
import torch
from torch.utils.data import Sampler


class PositiveAwareBatchSampler(Sampler[list[int]]):
    """Training-only sampler that injects positives without changing eval splits."""

    def __init__(
        self,
        labels: Sequence[float],
        batch_size: int,
        *,
        min_positive_per_batch: int = 4,
        positive_fraction_target: float = 0.20,
        seed: int = 42,
        drop_last: bool = False,
    ):
        self.labels = np.asarray(labels).reshape(-1)
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.epoch = 0
        self.positive = np.flatnonzero(self.labels > 0.5)
        self.negative = np.flatnonzero(self.labels <= 0.5)
        requested = max(
            int(min_positive_per_batch),
            int(math.ceil(self.batch_size * float(positive_fraction_target))),
        )
        self.positive_per_batch = min(requested, self.batch_size)
        if self.positive.size == 0 or self.negative.size == 0:
            self.positive_per_batch = 0

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __len__(self):
        if self.drop_last:
            return len(self.labels) // self.batch_size
        return math.ceil(len(self.labels) / self.batch_size)

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        if self.positive_per_batch == 0:
            order = rng.permutation(len(self.labels))
            for start in range(0, len(order), self.batch_size):
                batch = order[start:start + self.batch_size].tolist()
                if len(batch) == self.batch_size or not self.drop_last:
                    yield batch
            return
        negative_per_batch = self.batch_size - self.positive_per_batch
        for _ in range(len(self)):
            positives = rng.choice(
                self.positive,
                size=self.positive_per_batch,
                replace=self.positive.size < self.positive_per_batch,
            )
            negatives = rng.choice(
                self.negative,
                size=negative_per_batch,
                replace=self.negative.size < negative_per_batch,
            )
            batch = np.concatenate([positives, negatives])
            rng.shuffle(batch)
            yield batch.tolist()


def apply_modality_dropout(batch, cfg, *, generator=None):
    """Apply structured train-only modality dropout and keep one modality."""
    dropout_cfg = cfg.get("training", {}).get("modality_dropout", {})
    if not bool(dropout_cfg.get("enabled", False)):
        return batch
    mask = batch["modality_mask"]
    if mask.numel() == 0:
        return batch
    modalities = list(cfg["model"]["modalities"])
    global_probability = float(dropout_cfg.get("probability", 0.10))
    per_modality = dropout_cfg.get("per_modality", {})
    random_values = torch.rand(
        mask.shape, device=mask.device, generator=generator
    )
    probabilities = mask.new_tensor(
        [float(per_modality.get(name, global_probability)) for name in modalities]
    )
    dropped = (random_values < probabilities.unsqueeze(0)) & mask.gt(0)
    new_mask = mask * (~dropped).to(mask.dtype)
    lost_all = (new_mask.sum(1) == 0) & (mask.sum(1) > 0)
    if lost_all.any():
        available_scores = random_values.masked_fill(mask <= 0, float("inf"))
        restore = available_scores.argmin(1)
        rows = torch.arange(mask.size(0), device=mask.device)[lost_all]
        new_mask[rows, restore[lost_all]] = 1.0
    output = dict(batch)
    output["modality_mask"] = new_mask
    output["modalities"] = dict(batch["modalities"])
    for index, modality in enumerate(modalities):
        keep_shape = [new_mask.size(0)] + [1] * 3
        value = batch["modalities"][modality]
        if isinstance(value, dict):
            copied = dict(value)
            values = value.get("values")
            if torch.is_tensor(values):
                keep = new_mask[:, index].view(
                    new_mask.size(0), *([1] * (values.ndim - 1))
                ).to(values.dtype)
                copied["values"] = values * keep
            output["modalities"][modality] = copied
        elif torch.is_tensor(value):
            keep = new_mask[:, index].view(
                new_mask.size(0), *([1] * (value.ndim - 1))
            ).to(value.dtype)
            output["modalities"][modality] = value * keep
    return output
