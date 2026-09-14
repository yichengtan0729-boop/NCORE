from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .numerics import assert_finite_tensor


def fit_temperature(logits, targets, mask, split="val", max_iter=50):
    """Fit one positive temperature; fitting on non-validation data is rejected."""
    if str(split).lower() not in {"val", "validation"}:
        raise ValueError("Temperature scaling may only be fit on validation data")
    valid = mask > 0
    if not valid.any():
        return 1.0
    logits = logits.detach()
    targets = targets.detach()
    mask = mask.detach().to(logits.dtype)
    assert_finite_tensor("calibration_logits", logits)
    assert_finite_tensor("calibration_targets", targets)
    log_temperature = torch.nn.Parameter(logits.new_tensor(0.0))
    optimizer = torch.optim.LBFGS(
        [log_temperature], lr=0.1, max_iter=int(max_iter), line_search_fn="strong_wolfe"
    )

    def closure():
        optimizer.zero_grad()
        temperature = log_temperature.exp().clamp(1e-3, 100.0)
        assert_finite_tensor("calibration_temperature", temperature)
        scaled_logits = (logits.float() / temperature.float()).clamp(-30.0, 30.0)
        assert_finite_tensor("calibration_scaled_logits", scaled_logits)
        elementwise = F.binary_cross_entropy_with_logits(
            scaled_logits, targets.float(), reduction="none"
        )
        loss = (elementwise * mask.float()).sum() / (
            mask.float().sum() + 1e-6
        )
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.detach().exp().clamp(1e-3, 100.0).cpu())


def save_temperature(path, temperature, fit_split="val"):
    path = Path(path)
    path.write_text(
        json.dumps(
            {"temperature": float(temperature), "fit_split": str(fit_split)},
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def load_temperature(path):
    path = Path(path)
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    if str(payload.get("fit_split", "")).lower() not in {"val", "validation"}:
        raise ValueError("Refusing temperature not fitted on validation data")
    temperature = float(payload["temperature"])
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Saved direct temperature must be finite and positive")
    return temperature


def binary_calibration_metrics(labels, probabilities, bins=15):
    labels = np.asarray(labels, dtype=np.float64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if labels.size == 0:
        return {"ece": float("nan"), "brier": float("nan"), "nll": float("nan")}
    probabilities = np.clip(probabilities, 1e-7, 1.0 - 1e-7)
    brier = float(np.mean((probabilities - labels) ** 2))
    nll = float(
        -np.mean(
            labels * np.log(probabilities)
            + (1.0 - labels) * np.log(1.0 - probabilities)
        )
    )
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    ece = 0.0
    for index in range(int(bins)):
        if index == int(bins) - 1:
            selected = (probabilities >= edges[index]) & (
                probabilities <= edges[index + 1]
            )
        else:
            selected = (probabilities >= edges[index]) & (
                probabilities < edges[index + 1]
            )
        if selected.any():
            confidence = probabilities[selected].mean()
            accuracy = labels[selected].mean()
            ece += selected.mean() * abs(confidence - accuracy)
    return {"ece": float(ece), "brier": brier, "nll": nll}
