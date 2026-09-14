from __future__ import annotations

from typing import Any, Mapping

import torch


def tensor_finite_stats(name: str, x: torch.Tensor) -> dict[str, Any]:
    """Return finite/non-finite diagnostics without reducing an empty tensor."""
    if not torch.is_tensor(x):
        raise TypeError(f"{name} must be a torch.Tensor")
    detached = x.detach()
    finite_mask = torch.isfinite(detached)
    finite = detached[finite_mask].float()
    stats: dict[str, Any] = {
        "tensor": name,
        "shape": list(detached.shape),
        "dtype": str(detached.dtype),
        "device": str(detached.device),
        "num_nan": int(torch.isnan(detached).sum().item()),
        "num_posinf": int(torch.isposinf(detached).sum().item()),
        "num_neginf": int(torch.isneginf(detached).sum().item()),
        "finite_min": None,
        "finite_max": None,
        "finite_mean": None,
        "finite_std": None,
    }
    if finite.numel():
        stats.update(
            {
                "finite_min": float(finite.min().item()),
                "finite_max": float(finite.max().item()),
                "finite_mean": float(finite.mean().item()),
                "finite_std": float(finite.std(unbiased=False).item()),
            }
        )
    return stats


def _print_stats(stats: Mapping[str, Any], prefix: str = "") -> None:
    for key, value in stats.items():
        print(f"{prefix}{key}={value}")


def assert_finite_tensor(
    name: str,
    x: torch.Tensor,
    stage: str | None = None,
    epoch: int | None = None,
    batch_idx: int | None = None,
    extra: Mapping[str, Any] | None = None,
    raise_on_nonfinite: bool = True,
) -> bool:
    """Print the first bad tensor precisely and optionally fail fast."""
    if torch.isfinite(x).all():
        return True
    print("[nonfinite-error]")
    print(f"stage={stage}")
    print(f"epoch={epoch}")
    print(f"batch_idx={batch_idx}")
    _print_stats(tensor_finite_stats(name, x))
    for extra_name, extra_value in (extra or {}).items():
        if torch.is_tensor(extra_value):
            _print_stats(
                tensor_finite_stats(str(extra_name), extra_value),
                prefix="extra_",
            )
        else:
            print(f"extra_{extra_name}={extra_value}")
    if raise_on_nonfinite:
        raise FloatingPointError(
            f"Non-finite tensor '{name}' at stage={stage}, "
            f"epoch={epoch}, batch_idx={batch_idx}"
        )
    return False


def finite_or_fallback(
    name: str,
    x: torch.Tensor,
    *,
    fallback: float = 0.0,
    stage: str | None = None,
    epoch: int | None = None,
    batch_idx: int | None = None,
    extra: Mapping[str, Any] | None = None,
    fail_fast: bool = True,
) -> torch.Tensor:
    """Use an explicit, logged fallback only when fail-fast is disabled."""
    if assert_finite_tensor(
        name,
        x,
        stage=stage,
        epoch=epoch,
        batch_idx=batch_idx,
        extra=extra,
        raise_on_nonfinite=fail_fast,
    ):
        return x
    print(f"[nonfinite-fallback] tensor={name} fallback={fallback}")
    return torch.nan_to_num(x, nan=fallback, posinf=fallback, neginf=fallback)
