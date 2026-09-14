from __future__ import annotations
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from ncore.numerics import assert_finite_tensor, finite_or_fallback


def symmetric_normalize(a: torch.Tensor, eta: float = 0.0, eps: float = 1e-6):
    k = a.size(-1)
    assert_finite_tensor("symmetric_normalize.input", a)
    with torch.autocast(device_type=a.device.type, enabled=False):
        a32 = a.float().clamp(-10.0, 10.0)
        eye = torch.eye(k, device=a.device, dtype=torch.float32).expand(
            a.shape[0], k, k
        )
        aa = a32 + float(eta) * eye
        degree = aa.sum(-1).clamp_min(float(eps))
        inv = torch.rsqrt(degree + float(eps))
        normalized = inv.unsqueeze(-1) * aa * inv.unsqueeze(-2)
    return normalized.to(a.dtype)


class PatientOperatorGenerator(nn.Module):
    def __init__(self, concept_dim: int, op_dim: int | None = None,
                 rho: float = 0.98, numerics=None, debug=None):
        super().__init__()
        op_dim = op_dim or concept_dim
        self.q = nn.Linear(concept_dim, op_dim, bias=False)
        self.k = nn.Linear(concept_dim, op_dim, bias=False)
        self.scale = op_dim ** -0.5
        self.rho = rho
        self.numerics = dict(numerics or {})
        self.debug = dict(debug or {})

    def forward(self, z, context=None, name="operator"):
        context = context or {}
        fail_fast = bool(self.debug.get("fail_fast_on_nonfinite", True))
        eps = float(self.numerics.get("eps", 1e-6))
        operator_clip = float(self.numerics.get("operator_clip", 10.0))
        q, k = self.q(z), self.k(z)
        q = finite_or_fallback(
            f"{name}.q", q, fail_fast=fail_fast, **context
        )
        k = finite_or_fallback(
            f"{name}.k", k, fail_fast=fail_fast, **context
        )
        with torch.autocast(device_type=z.device.type, enabled=False):
            s = torch.bmm(q.float(), k.float().transpose(1, 2)) * self.scale
            s = 0.5 * (s + s.transpose(1, 2))
        s = finite_or_fallback(
            f"{name}.similarity", s, fail_fast=fail_fast, **context
        )
        # Bounding finite similarities prevents adjacency-degree overflow.
        adjacency = F.softplus(s.clamp(-30.0, 30.0))
        adjacency = finite_or_fallback(
            f"{name}.adjacency", adjacency, fail_fast=fail_fast, **context
        ).clamp(0.0, operator_clip)
        normalized = symmetric_normalize(adjacency, eta=1.0, eps=eps).float()
        normalized = finite_or_fallback(
            f"{name}.normalized", normalized, fail_fast=fail_fast, **context
        )
        with torch.autocast(device_type=z.device.type, enabled=False):
            spectral = torch.linalg.matrix_norm(normalized, ord=2).clamp_min(eps)
            scale = torch.minimum(
                torch.ones_like(spectral),
                normalized.new_tensor(self.rho) / (spectral + eps),
            )
            operator = normalized * scale.view(-1, 1, 1)
        operator = finite_or_fallback(
            f"{name}.spectral_scaled", operator,
            fail_fast=fail_fast, **context
        ).clamp(-operator_clip, operator_clip)
        assert_finite_tensor(
            f"{name}.final", operator, raise_on_nonfinite=fail_fast, **context
        )
        return operator


class ConceptGateBank(nn.Module):
    def __init__(self, num_concepts: int):
        super().__init__()
        # Diagonal initialized high, off-diagonal low but learnable.
        logits = torch.full((num_concepts, num_concepts), -2.0)
        logits.fill_diagonal_(2.0)
        self.logits = nn.Parameter(logits)
    def all_gates(self):
        return torch.sigmoid(self.logits)
    def gate(self, concept_idx: torch.Tensor):
        return self.all_gates()[concept_idx]


def localize_operator(t: torch.Tensor, gate: torch.Tensor, eta: float = 0.05,
                      eps: float = 1e-6):
    # t: [B,K,K], gate: [B,K]
    loc = t * gate.unsqueeze(-1) * gate.unsqueeze(-2)
    return symmetric_normalize(loc, eta=eta, eps=eps)


def commutator_features(
    operators: List[torch.Tensor],
    state: torch.Tensor,
    *,
    eps: float = 1e-6,
    commutator_clip: float = 20.0,
    norm_clip: float = 20.0,
    fail_fast: bool = True,
    context=None,
    return_stats: bool = False,
):
    """Float32 per-patient commutator RMS features for modality pairs."""
    context = context or {}
    feats = []
    max_abs = state.new_tensor(0.0, dtype=torch.float32)
    for i in range(len(operators)):
        for j in range(i+1, len(operators)):
            ti, tj = operators[i], operators[j]
            assert_finite_tensor(
                f"operator_matrix_{i}", ti,
                raise_on_nonfinite=fail_fast, **context
            )
            assert_finite_tensor(
                f"operator_matrix_{j}", tj,
                raise_on_nonfinite=fail_fast, **context
            )
            with torch.autocast(device_type=state.device.type, enabled=False):
                ti32, tj32 = ti.float(), tj.float()
                comm = torch.bmm(ti32, tj32) - torch.bmm(tj32, ti32)
            comm = finite_or_fallback(
                f"commutator_{i}_{j}", comm,
                fail_fast=fail_fast, **context
            )
            max_abs = torch.maximum(max_abs, comm.detach().abs().max())
            comm = comm.clamp(-commutator_clip, commutator_clip)
            with torch.autocast(device_type=state.device.type, enabled=False):
                transformed = torch.bmm(comm, state.float())
                feature = torch.sqrt(
                    transformed.square().mean(dim=(1, 2)) + float(eps)
                ).clamp_max(norm_clip)
            feature = finite_or_fallback(
                f"commutator_norm_{i}_{j}", feature,
                fail_fast=fail_fast, **context
            )
            feats.append(feature)
    if not feats:
        output = state.new_zeros(state.size(0), 0, dtype=torch.float32)
        stats = {
            "max_abs_commutator": 0.0,
            "max_commutator_norm": 0.0,
        }
    else:
        output = torch.stack(feats, dim=1)
        stats = {
            "max_abs_commutator": float(max_abs.item()),
            "max_commutator_norm": float(output.detach().max().item()),
        }
    return (output, stats) if return_stats else output
