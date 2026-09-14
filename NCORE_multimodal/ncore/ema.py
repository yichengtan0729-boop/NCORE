from __future__ import annotations
from contextlib import contextmanager

import torch


class ModelEMA:
    """EMA for trainable prediction parameters, excluding the routing policy."""

    def __init__(self, model, decay=0.999, include_frozen=False):
        self.decay = float(decay)
        if not 0.0 <= self.decay < 1.0:
            raise ValueError("EMA decay must be in [0, 1)")
        self.shadow = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if (include_frozen or parameter.requires_grad)
            and not name.startswith("policy.")
        }

    @torch.no_grad()
    def update(self, model):
        parameters = dict(model.named_parameters())
        for name, shadow in self.shadow.items():
            current = parameters[name].detach()
            shadow.mul_(self.decay).add_(current, alpha=1.0 - self.decay)

    def state_dict(self):
        return {
            "decay": self.decay,
            "shadow": {name: value.clone() for name, value in self.shadow.items()},
        }

    def load_state_dict(self, state):
        self.decay = float(state.get("decay", self.decay))
        saved = state.get("shadow", {})
        for name in self.shadow:
            if name in saved and tuple(saved[name].shape) == tuple(self.shadow[name].shape):
                self.shadow[name].copy_(saved[name])

    @contextmanager
    def average_parameters(self, model):
        parameters = dict(model.named_parameters())
        backup = {}
        with torch.no_grad():
            for name, shadow in self.shadow.items():
                backup[name] = parameters[name].detach().clone()
                parameters[name].copy_(shadow)
        try:
            yield
        finally:
            with torch.no_grad():
                for name, value in backup.items():
                    parameters[name].copy_(value)


class ModelSWA:
    """Small state-dict SWA helper with the same safe swap API as ModelEMA."""

    def __init__(self, model, parameter_prefixes=()):
        self.parameter_prefixes = tuple(parameter_prefixes)
        self.average = {}
        self.count = 0
        for name, parameter in model.named_parameters():
            if self.parameter_prefixes and not name.startswith(self.parameter_prefixes):
                continue
            self.average[name] = parameter.detach().clone()

    @torch.no_grad()
    def update(self, model):
        parameters = dict(model.named_parameters())
        self.count += 1
        for name, average in self.average.items():
            current = parameters[name].detach()
            if self.count == 1:
                average.copy_(current)
            else:
                average.add_((current - average) / float(self.count))

    def state_dict(self):
        return {
            "count": self.count,
            "average": {name: value.clone() for name, value in self.average.items()},
        }

    def load_state_dict(self, state):
        self.count = int(state.get("count", 0))
        for name, value in state.get("average", {}).items():
            if name in self.average and self.average[name].shape == value.shape:
                self.average[name].copy_(value)

    @contextmanager
    def average_parameters(self, model):
        parameters = dict(model.named_parameters())
        backup = {}
        with torch.no_grad():
            for name, average in self.average.items():
                backup[name] = parameters[name].detach().clone()
                parameters[name].copy_(average)
        try:
            yield
        finally:
            with torch.no_grad():
                for name, value in backup.items():
                    parameters[name].copy_(value)
