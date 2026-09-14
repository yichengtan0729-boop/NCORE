from __future__ import annotations
from typing import Dict, Any
import math
import torch
import torch.nn as nn

from .encoders import build_encoder
from .concepts import ConceptProjector
from .operators import (
    PatientOperatorGenerator,
    ConceptGateBank,
    localize_operator,
    commutator_features,
)
from .policy import PathPolicy
from .v2 import (
    DirectAwareResidualHead,
    PatientSpecificOperatorGate,
    ResidualModalityAttentionEnhancer,
    binary_uncertainty_features,
)
from .v5 import (
    CrossModalGatedPoolingResidual,
    PairwiseInteractionResidual,
    ReasonGate,
    RunningUtilityNormalizer,
    bounded_correction,
)
from ncore.numerics import assert_finite_tensor


def _as_logit(probability: float) -> float:
    p = min(max(float(probability), 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


class NCORE(nn.Module):
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg
        model_cfg = cfg["model"]
        concept_cfg = model_cfg.get("concept", {})
        policy_cfg = model_cfg.get("policy", {})
        direct_cfg = model_cfg.get("direct_fusion", {})
        operator_cfg = model_cfg.get("operator", {})
        residual_cfg = model_cfg.get("residual", {})
        performance_cfg = model_cfg.get("performance_v5", {})
        self.performance_v5 = bool(performance_cfg.get("enabled", False))
        self.numerics_cfg = dict(cfg.get("numerics", {}))
        self.debug_cfg = dict(cfg.get("debug", {}))
        self.numerics_eps = float(self.numerics_cfg.get("eps", 1e-6))
        self.finite_checks = bool(self.debug_cfg.get("finite_checks", False))
        self.fail_fast_on_nonfinite = bool(
            self.debug_cfg.get("fail_fast_on_nonfinite", True)
        ) and bool(self.debug_cfg.get("finite_checks_raise", True))
        self._numerics_context = {
            "stage": None,
            "epoch": None,
            "batch_idx": None,
        }
        self._numerics_observations = {}
        self.policy_fallback_count = 0
        self._last_policy_diagnostics = {}
        self._last_policy_details = {}

        self.modalities = list(model_cfg["modalities"])
        self.M = len(self.modalities)
        self.K = int(concept_cfg.get("num_concepts", model_cfg.get("num_concepts", 16)))
        self.D = int(concept_cfg.get("concept_dim", model_cfg.get("concept_dim", 128)))
        self.max_steps = int(policy_cfg.get("max_steps", model_cfg.get("max_steps", 3)))
        self.eta = float(model_cfg.get("operator_eta", operator_cfg.get("localization_eta", 0.05)))
        self.num_tasks = len(cfg["experiment"]["task_names"])
        self.direct_enabled = bool(direct_cfg.get("enabled", False))
        self.direct_mode = str(direct_cfg.get("mode", "mlp"))
        attention_cfg = direct_cfg.get("attention", {})
        self.attention_enabled = bool(
            attention_cfg.get("enabled", self.direct_mode == "residual_attention")
        )
        self.residual_enabled = bool(residual_cfg.get("enabled", True))
        self.residual_mode = str(residual_cfg.get("mode", "operator_only"))
        self.operator_gate_mode = str(residual_cfg.get("gate_mode", "global"))
        self.force_identity_operator = bool(
            operator_cfg.get("force_identity_operator", False)
        )
        self.policy_direct_features = bool(
            policy_cfg.get("include_direct_uncertainty", False)
        )
        self.gate_logit_magnitude = bool(
            residual_cfg.get("include_logit_magnitude", False)
        )
        self.policy_logit_magnitude = bool(
            policy_cfg.get("include_logit_magnitude", False)
        )
        self.identity_mix_enabled = bool(operator_cfg.get("identity_mix", False))
        self.state_gate_enabled = "state_residual_gate_init" in operator_cfg

        self.encoders = nn.ModuleDict()
        self.projectors = nn.ModuleDict()
        self.op_generators = nn.ModuleDict()
        self.updaters = nn.ModuleDict()
        dropout = float(model_cfg.get("dropout", 0.1))
        rho = float(model_cfg.get("operator_rho", operator_cfg.get("rho", 0.98)))
        for modality in self.modalities:
            encoder = build_encoder(modality, model_cfg["encoders"][modality], cfg)
            self.encoders[modality] = encoder
            self.projectors[modality] = ConceptProjector(
                encoder.output_dim, self.K, self.D, dropout
            )
            self.op_generators[modality] = PatientOperatorGenerator(
                self.D,
                rho=rho,
                numerics=self.numerics_cfg,
                debug=self.debug_cfg,
            )
            self.updaters[modality] = nn.Sequential(
                nn.Linear(self.D, self.D), nn.GELU(), nn.Dropout(dropout)
            )

        self.gates = ConceptGateBank(self.K)
        self.initial_state = nn.Parameter(torch.randn(self.K, self.D) * 0.02)
        self.context_proj = nn.ModuleDict(
            {modality: nn.Linear(self.D, self.D) for modality in self.modalities}
        )
        self.outcome_head = nn.Sequential(
            nn.LayerNorm(self.D), nn.Linear(self.D, self.num_tasks)
        )
        self.state_norm = nn.LayerNorm(self.D)
        self.operator_hidden_norm = nn.LayerNorm(self.D)
        self.register_buffer("direct_temperature", torch.tensor(1.0))
        self.register_buffer(
            "final_temperature", torch.tensor(1.0), persistent=self.performance_v5
        )
        self.register_buffer(
            "reason_threshold",
            torch.tensor(float(performance_cfg.get("reason_threshold", 0.5))),
            persistent=self.performance_v5,
        )
        self.register_buffer(
            "correction_bound",
            torch.tensor(
                float(
                    performance_cfg.get("bounded_correction", {}).get(
                        "selected_c", 0.35
                    )
                )
            ),
            persistent=self.performance_v5,
        )

        eps_init = float(operator_cfg.get("identity_mix_init", 0.10))
        self.operator_eps_logits = nn.ParameterDict(
            {
                modality: nn.Parameter(torch.tensor(_as_logit(eps_init)))
                for modality in self.modalities
            }
        )
        beta_init = float(operator_cfg.get("state_residual_gate_init", 0.10))
        self.state_gate_logit = nn.Parameter(torch.tensor(_as_logit(beta_init)))
        alpha_init = float(residual_cfg.get("operator_gate_init", 0.05))
        self.operator_gate_logit = nn.Parameter(torch.tensor(_as_logit(alpha_init)))

        self.direct_projections = nn.ModuleDict()
        self.direct_attention_enhancer = None
        self.residual_correction = None
        self.patient_operator_gate = None
        self.pairwise_residual = None
        self.pooling_residual = None
        self.reason_gate = None
        self.utility_normalizer = None
        if self.direct_enabled:
            fusion_dim = int(direct_cfg.get("fusion_dim", 256))
            hidden_dim = int(direct_cfg.get("hidden_dim", fusion_dim))
            direct_dropout = float(direct_cfg.get("dropout", 0.2))
            for modality in self.modalities:
                self.direct_projections[modality] = nn.Linear(
                    self.encoders[modality].output_dim, fusion_dim
                )
            self.direct_fusion = nn.Sequential(
                nn.Linear(self.M * fusion_dim + self.M, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(direct_dropout),
            )
            self.direct_head = nn.Linear(hidden_dim, self.num_tasks)
            if self.performance_v5:
                input_dims = {
                    modality: self.encoders[modality].output_dim
                    for modality in self.modalities
                }
                pair_cfg = performance_cfg.get("pairwise", {})
                pool_cfg = performance_cfg.get("pooling", {})
                self.pairwise_residual = PairwiseInteractionResidual(
                    input_dims, self.num_tasks, pair_cfg
                )
                pair_dim = int(pair_cfg.get("dim", 256))
                self.pooling_residual = CrossModalGatedPoolingResidual(
                    pair_dim, self.M, self.num_tasks, pool_cfg
                )
            if self.attention_enabled:
                self.direct_attention_enhancer = ResidualModalityAttentionEnhancer(
                    fusion_dim, hidden_dim, attention_cfg
                )
            if self.residual_enabled and self.residual_mode == "direct_aware":
                self.residual_correction = DirectAwareResidualHead(
                    hidden_dim, self.D, self.num_tasks, residual_cfg
                )
            if self.operator_gate_mode == "patient_specific":
                gate_input_dim = (
                    hidden_dim
                    + (4 if self.gate_logit_magnitude else 3) * self.num_tasks
                    + 2
                    + self.M
                )
                self.patient_operator_gate = PatientSpecificOperatorGate(
                    gate_input_dim, residual_cfg
                )
            if self.performance_v5:
                reason_input_dim = (
                    hidden_dim
                    + 4 * self.num_tasks
                    + 2
                    + self.M
                    + 1
                    + 2 * self.num_tasks
                )
                self.reason_gate = ReasonGate(
                    reason_input_dim, performance_cfg.get("reason_gate", {})
                )
                self.utility_normalizer = RunningUtilityNormalizer(
                    momentum=float(
                        performance_cfg.get("utility", {}).get("momentum", 0.95)
                    ),
                    eps=self.numerics_eps,
                )
            nn.init.zeros_(self.outcome_head[-1].weight)
            nn.init.zeros_(self.outcome_head[-1].bias)
        else:
            self.direct_fusion = nn.Identity()
            self.direct_head = nn.Identity()

        num_pairs = self.M * (self.M - 1) // 2
        policy_extra_dim = (
            (5 if self.policy_logit_magnitude else 4) * self.num_tasks
            + 3
            + self.M
            if self.policy_direct_features else 0
        )
        self.policy = PathPolicy(
            self.D,
            self.D,
            self.M,
            self.K,
            self.num_tasks,
            num_pairs,
            hidden_dim=int(model_cfg.get("hidden_dim", 256)),
            max_steps=self.max_steps,
            extra_dim=policy_extra_dim,
            include_stop=not self.performance_v5,
        )
        # v5 uses this as a virtual rollout marker, never as a policy action.
        self.stop_idx = self.M * self.K

    @property
    def num_actions(self) -> int:
        return self.policy.num_actions

    def set_numerics_context(self, stage=None, epoch=None, batch_idx=None):
        self._numerics_context = {
            "stage": stage,
            "epoch": epoch,
            "batch_idx": batch_idx,
        }

    def reset_numerics_observations(self):
        self._numerics_observations = {
            "max_abs_operator": 0.0,
            "max_abs_commutator": 0.0,
            "max_commutator_norm": 0.0,
            "max_abs_delta": 0.0,
            "max_abs_state_update": 0.0,
            "nonfinite_count": 0,
        }
        self.policy_fallback_count = 0

    def _observe_max(self, name, value):
        scalar = float(value.detach().abs().max().item()) if torch.is_tensor(value) else float(value)
        self._numerics_observations[name] = max(
            float(self._numerics_observations.get(name, 0.0)), scalar
        )

    def numerics_summary(self):
        summary = {
            "max_abs_operator": 0.0,
            "max_abs_commutator": 0.0,
            "max_commutator_norm": 0.0,
            "max_abs_delta": 0.0,
            "max_abs_state_update": 0.0,
            "nonfinite_count": 0,
            **self._numerics_observations,
        }
        temperature = float(self.direct_temperature.detach().cpu())
        summary.update(
            {
                "min_temperature": temperature,
                "max_temperature": temperature,
                "fallback_count": int(self.policy_fallback_count),
            }
        )
        return summary

    def _check(self, name, tensor, extra=None, force=False):
        if not (self.finite_checks or force):
            return True
        try:
            valid = assert_finite_tensor(
                name,
                tensor,
                extra=extra,
                raise_on_nonfinite=self.fail_fast_on_nonfinite,
                **self._numerics_context,
            )
            if not valid:
                self._numerics_observations["nonfinite_count"] = int(
                    self._numerics_observations.get("nonfinite_count", 0)
                ) + 1
            return valid
        except FloatingPointError:
            self._numerics_observations["nonfinite_count"] = int(
                self._numerics_observations.get("nonfinite_count", 0)
            ) + 1
            raise

    def operator_gate_value(self) -> torch.Tensor:
        if not self.direct_enabled:
            return self.initial_state.new_tensor(1.0)
        if self.operator_gate_mode == "patient_specific":
            return self.initial_state.new_tensor(
                float(self.cfg["model"].get("residual", {}).get("alpha_init", 0.08))
            )
        return torch.sigmoid(self.operator_gate_logit)

    def direct_attention_gate_value(self) -> torch.Tensor:
        if self.direct_attention_enhancer is None:
            return self.initial_state.new_tensor(0.0)
        return self.direct_attention_enhancer.gate_value()

    def set_direct_temperature(self, temperature: float) -> None:
        value = float(temperature)
        if not math.isfinite(value) or value <= 0:
            raise ValueError("direct temperature must be finite and positive")
        self.direct_temperature.fill_(value)

    def set_final_temperature(self, temperature: float) -> None:
        value = float(temperature)
        if not math.isfinite(value) or value <= 0:
            raise ValueError("final temperature must be finite and positive")
        self.final_temperature.fill_(value)

    def set_correction_bound(self, value: float) -> None:
        value = float(value)
        if not math.isfinite(value) or value <= 0:
            raise ValueError("correction bound must be finite and positive")
        self.correction_bound.fill_(value)

    def direct_uncertainty_features(self, logits: torch.Tensor):
        self._check("direct_logits", logits)
        temperature = self.direct_temperature.to(
            device=logits.device, dtype=torch.float32
        )
        self._check("direct_temperature", temperature, force=True)
        temperature_safe = temperature.clamp(1e-3, 100.0)
        with torch.autocast(device_type=logits.device.type, enabled=False):
            scaled_logits = (logits.float() / temperature_safe).clamp(
                -30.0, 30.0
            )
            probability, entropy, margin = binary_uncertainty_features(
                scaled_logits, eps=self.numerics_eps
            )
            logit_magnitude = scaled_logits.abs().clamp_max(30.0)
        self._check("calibrated_probability", probability)
        self._check("direct_entropy", entropy)
        self._check("direct_margin", margin)
        self._check("direct_logit_magnitude", logit_magnitude)
        return probability, entropy, margin

    def direct_logit_magnitude_feature(self, logits: torch.Tensor):
        temperature = self.direct_temperature.to(
            device=logits.device, dtype=torch.float32
        ).clamp(1e-3, 100.0)
        magnitude = (logits.float() / temperature).clamp(-30.0, 30.0).abs()
        self._check("direct_logit_magnitude", magnitude)
        return magnitude.to(logits.dtype)

    def state_gate_value(self) -> torch.Tensor:
        if not self.state_gate_enabled:
            return self.initial_state.new_tensor(1.0)
        return torch.sigmoid(self.state_gate_logit)

    def identity_mix_values(self) -> Dict[str, torch.Tensor]:
        if not self.identity_mix_enabled:
            return {
                modality: self.initial_state.new_tensor(1.0)
                for modality in self.modalities
            }
        return {
            modality: torch.sigmoid(self.operator_eps_logits[modality])
            for modality in self.modalities
        }

    def encode(self, batch):
        z, confidence, features = {}, {}, {}
        for modality in self.modalities:
            feature = self.encoders[modality](batch["modalities"][modality])
            concepts, conf = self.projectors[modality](feature)
            features[modality] = feature
            z[modality] = concepts
            confidence[modality] = conf
        return {
            "h": features,
            "z": z,
            "conf": confidence,
            "modality_mask": batch["modality_mask"],
        }

    def build_operators(self, encoded, modality_mask=None):
        modality_mask = (
            encoded.get("modality_mask") if modality_mask is None else modality_mask
        )
        eps_values = self.identity_mix_values()
        operators = {}
        for index, modality in enumerate(self.modalities):
            raw = self.op_generators[modality](
                encoded["z"][modality],
                context=self._numerics_context,
                name=f"operator.{modality}",
            )
            self._check(f"operator_matrix_raw.{modality}", raw)
            eye = torch.eye(
                self.K, device=raw.device, dtype=raw.dtype
            ).unsqueeze(0).expand(raw.size(0), -1, -1)
            if self.force_identity_operator:
                operator = eye
            elif self.identity_mix_enabled:
                eps = eps_values[modality]
                operator = (1.0 - eps) * eye + eps * raw
            else:
                operator = raw
            if modality_mask is not None:
                available = modality_mask[:, index].view(-1, 1, 1) > 0
                operator = torch.where(available, operator, eye)
            operator = operator.clamp(
                -float(self.numerics_cfg.get("operator_clip", 10.0)),
                float(self.numerics_cfg.get("operator_clip", 10.0)),
            )
            self._check(f"operator_matrix.{modality}", operator)
            self._observe_max("max_abs_operator", operator)
            operators[modality] = operator
        return operators

    def direct_outputs(self, encoded, modality_mask):
        if not self.direct_enabled:
            zeros = self.initial_state.new_zeros(
                modality_mask.size(0), self.num_tasks
            )
            return {
                "direct_base_logits": zeros,
                "direct_logits": zeros,
                "direct_base_hidden": None,
                "direct_hidden": None,
                "direct_attention_hidden": None,
                "direct_attention_gate": zeros.new_tensor(0.0),
                "direct_pair_logits": zeros,
                "direct_pool_logits": zeros,
                "direct_pair_delta": zeros,
                "direct_pool_delta": zeros,
            }
        projected = []
        masked_projected = []
        for index, modality in enumerate(self.modalities):
            feature = self.direct_projections[modality](encoded["h"][modality])
            projected.append(feature)
            masked_projected.append(
                feature * modality_mask[:, index:index + 1]
            )
        fusion_input = torch.cat(
            masked_projected + [modality_mask.to(projected[0].dtype)], dim=1
        )
        base_hidden = self.direct_fusion(fusion_input)
        base_logits = self.direct_head(base_hidden)
        direct_hidden = base_hidden
        attention_hidden = None
        attention_gate = base_logits.new_tensor(0.0)
        if self.direct_attention_enhancer is not None:
            direct_hidden, attention_hidden, attention_gate = (
                self.direct_attention_enhancer(
                    projected, modality_mask, base_hidden
                )
            )
        direct_logits = self.direct_head(direct_hidden)
        pair_delta = torch.zeros_like(direct_logits)
        pool_delta = torch.zeros_like(direct_logits)
        pair_logits = direct_logits
        pool_logits = direct_logits
        pair_gate = direct_logits.new_tensor(0.0)
        pool_gate = direct_logits.new_tensor(0.0)
        modality_weights = direct_logits.new_zeros(modality_mask.shape)
        if self.pairwise_residual is not None:
            gated_pair, pair_delta, adapted, pair_masks = self.pairwise_residual(
                encoded["h"], modality_mask
            )
            base_probability, _, _ = self.direct_uncertainty_features(base_logits)
            gated_pool, pool_delta, modality_weights = self.pooling_residual(
                adapted, modality_mask, base_probability
            )
            pair_logits = base_logits + gated_pair
            pool_logits = base_logits + gated_pool
            direct_logits = base_logits + gated_pair + gated_pool
            pair_gate = self.pairwise_residual.gate.value()
            pool_gate = self.pooling_residual.gate.value()
            self._check("direct_pair_delta", pair_delta)
            self._check("direct_pool_delta", pool_delta)
            self._check("direct_modality_weights", modality_weights)
        return {
            "direct_base_logits": base_logits,
            "direct_logits": direct_logits,
            "direct_base_hidden": base_hidden,
            "direct_hidden": direct_hidden,
            "direct_attention_hidden": attention_hidden,
            "direct_attention_gate": attention_gate,
            "direct_pair_logits": pair_logits,
            "direct_pool_logits": pool_logits,
            "direct_pair_delta": pair_delta,
            "direct_pool_delta": pool_delta,
            "direct_pair_gate": pair_gate,
            "direct_pool_gate": pool_gate,
            "direct_modality_weights": modality_weights,
        }

    def direct_logits(self, encoded, modality_mask):
        return self.direct_outputs(encoded, modality_mask)["direct_logits"]

    def _commutator_features(self, operators, state):
        features, stats = commutator_features(
            [operators[modality] for modality in self.modalities],
            state,
            eps=self.numerics_eps,
            commutator_clip=float(
                self.numerics_cfg.get("commutator_clip", 20.0)
            ),
            norm_clip=float(
                self.numerics_cfg.get("commutator_norm_clip", 20.0)
            ),
            fail_fast=self.fail_fast_on_nonfinite,
            context=self._numerics_context,
            return_stats=True,
        )
        self._check("commutator_tensors", features)
        self._observe_max(
            "max_abs_commutator", stats["max_abs_commutator"]
        )
        self._observe_max(
            "max_commutator_norm", stats["max_commutator_norm"]
        )
        return features

    def _commutator_summary(self, operators, state, modality_mask):
        batch_size = state.size(0)
        if operators is None or self.M < 2:
            zeros = state.new_zeros(batch_size, 1)
            return zeros, zeros
        features = self._commutator_features(operators, state)
        pair_masks = []
        for left in range(self.M):
            for right in range(left + 1, self.M):
                pair_masks.append(
                    modality_mask[:, left] * modality_mask[:, right]
                )
        available = torch.stack(pair_masks, dim=1).to(features.dtype)
        masked = features * available
        pair_count = available.sum(1, keepdim=True)
        has_pair = pair_count > 0
        mean = torch.where(
            has_pair,
            masked.sum(1, keepdim=True) / (pair_count + self.numerics_eps),
            torch.zeros_like(pair_count),
        )
        maximum = features.masked_fill(available <= 0, -torch.inf).max(
            1, keepdim=True
        ).values
        maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
        limit = float(self.numerics_cfg.get("commutator_norm_clip", 20.0))
        mean = mean.clamp(0.0, limit)
        maximum = maximum.clamp(0.0, limit)
        self._check("mean_commutator", mean)
        self._check("max_commutator", maximum)
        return mean, maximum

    def _patient_gate_features(
        self, direct, operators, state, modality_mask
    ):
        if direct["direct_hidden"] is not None:
            self._check("direct_hidden", direct["direct_hidden"])
        probability, entropy, margin = self.direct_uncertainty_features(
            direct["direct_logits"]
        )
        comm_mean, comm_max = self._commutator_summary(
            operators, state, modality_mask
        )
        features = [direct["direct_hidden"], probability, entropy, margin]
        if self.gate_logit_magnitude:
            features.append(
                self.direct_logit_magnitude_feature(direct["direct_logits"])
            )
        features.extend(
            [comm_mean, comm_max, modality_mask.to(probability.dtype)]
        )
        gate_state = torch.cat(features, dim=-1)
        self._check(
            "patient_gate_state",
            gate_state,
            extra={
                "modality_mask": modality_mask,
                "direct_logits": direct["direct_logits"],
                "calibrated_probability": probability,
                "mean_commutator": comm_mean,
            },
        )
        return gate_state

    def _reason_gate_features(self, direct, operators, state, modality_mask, alpha):
        probability, entropy, margin = self.direct_uncertainty_features(
            direct["direct_logits"]
        )
        magnitude = self.direct_logit_magnitude_feature(direct["direct_logits"])
        comm_mean, comm_max = self._commutator_summary(
            operators, state, modality_mask
        )
        disagreement = [
            direct.get("direct_pair_delta", torch.zeros_like(probability)),
            direct.get("direct_pool_delta", torch.zeros_like(probability)),
        ]
        features = torch.cat(
            [
                direct["direct_hidden"],
                probability,
                entropy,
                margin,
                magnitude,
                comm_mean,
                comm_max,
                modality_mask.to(probability.dtype),
                alpha,
                *disagreement,
            ],
            dim=-1,
        )
        self._check("reason_gate_state", features)
        return features

    def prediction_outputs(
        self, state, encoded, modality_mask, operators=None, fixed_alpha=None,
        direct=None, reason_mask=None
    ):
        operator_hidden = self.operator_hidden_norm(state.mean(1))
        self._check("operator_state", state)
        self._check("operator_hidden", operator_hidden)
        direct = direct or self.direct_outputs(encoded, modality_mask)
        if direct["direct_hidden"] is not None:
            self._check("direct_hidden", direct["direct_hidden"])
        self._check("direct_logits", direct["direct_logits"])
        if self.residual_correction is not None:
            operator_delta, interaction_hidden = self.residual_correction(
                direct["direct_hidden"], operator_hidden
            )
        else:
            operator_delta = self.outcome_head(operator_hidden)
            interaction_hidden = operator_hidden
        self._check("operator_delta_logits_pre_clip", operator_delta)
        operator_delta = operator_delta.clamp(
            -float(self.numerics_cfg.get("delta_clip", 10.0)),
            float(self.numerics_cfg.get("delta_clip", 10.0)),
        )
        self._check("operator_delta_logits", operator_delta)
        self._observe_max("max_abs_delta", operator_delta)

        if self.direct_enabled:
            if not self.residual_enabled:
                operator_delta = torch.zeros_like(operator_delta)
                alpha = operator_delta.new_zeros(operator_delta.size(0), 1)
            elif fixed_alpha is not None:
                alpha = operator_delta.new_full(
                    (operator_delta.size(0), 1), float(fixed_alpha)
                )
            elif self.patient_operator_gate is not None:
                alpha = self.patient_operator_gate(
                    self._patient_gate_features(
                        direct, operators, state, modality_mask
                    )
                )
            else:
                alpha = self.operator_gate_value().expand(
                    operator_delta.size(0), 1
                )
            self._check("patient_alpha", alpha)
            raw_correction = alpha * operator_delta
            if self.performance_v5:
                correction_bound = float(self.correction_bound.detach().cpu())
                correction = bounded_correction(raw_correction, correction_bound)
                reason_probability = self.reason_gate(
                    self._reason_gate_features(
                        direct, operators, state, modality_mask, alpha
                    )
                )
                if reason_mask is None:
                    route = reason_probability
                else:
                    route = reason_mask.to(correction.dtype).reshape(-1, 1)
                final = direct["direct_logits"] + route * correction
            else:
                correction = raw_correction
                reason_probability = torch.ones_like(alpha)
                final = direct["direct_logits"] + correction
        else:
            alpha = operator_delta.new_ones(operator_delta.size(0), 1)
            raw_correction = operator_delta
            correction = operator_delta
            reason_probability = torch.ones_like(alpha)
            final = operator_delta
        self._check("final_logits", final)
        outputs = dict(direct)
        outputs.update({
            "logits": final,
            "operator_hidden": operator_hidden,
            "operator_delta_logits": operator_delta,
            "residual_interaction_hidden": interaction_hidden,
            "operator_gate": alpha,
            "operator_gate_alpha_per_sample": alpha,
            "raw_correction": raw_correction,
            "bounded_correction": correction,
            "reason_probability": reason_probability,
            "utility_normalizer": self.utility_normalizer,
            "state_residual_gate": self.state_gate_value(),
        })
        return outputs

    def _context(self, encoded, modality_mask):
        pools = torch.stack(
            [
                self.context_proj[modality](encoded["z"][modality].mean(1))
                for modality in self.modalities
            ],
            1,
        )
        mask = modality_mask.unsqueeze(-1)
        denominator = mask.sum(1).clamp_min(1.0)
        return (pools * mask).sum(1) / (denominator + self.numerics_eps)

    def _policy_extra_features(self, outputs, operators, state, modality_mask):
        probability, entropy, margin = self.direct_uncertainty_features(
            outputs["direct_logits"]
        )
        comm_mean, comm_max = self._commutator_summary(
            operators, state, modality_mask
        )
        temperature = self.direct_temperature.to(
            device=outputs["direct_logits"].device, dtype=torch.float32
        ).clamp(1e-3, 100.0)
        scaled_logits = (
            outputs["direct_logits"].float() / temperature
        ).clamp(-30.0, 30.0).to(outputs["direct_logits"].dtype)
        features = [scaled_logits, probability, entropy, margin]
        if self.policy_logit_magnitude:
            features.append(
                self.direct_logit_magnitude_feature(outputs["direct_logits"])
            )
        features.extend(
            [
                outputs["operator_gate"],
                comm_mean,
                comm_max,
                modality_mask.to(probability.dtype),
            ]
        )
        extra = torch.cat(features, dim=-1)
        self._check("policy_extra_features", extra)
        return extra

    def _validated_policy_logits(self, details, diagnostic_extra):
        names = [
            "policy_state_pre_norm",
            "policy_state",
            "policy_logits_pre_mask",
            "policy_logits_post_mask",
        ]
        all_finite = True
        for name in names:
            all_finite = self._check(
                name, details[name], extra=diagnostic_extra
            ) and all_finite
        logits = details["policy_logits_post_mask"]
        action_mask = details["valid_action_mask"]
        if not action_mask.any(dim=1).all():
            raise FloatingPointError("A sample has no valid policy action")
        if self.policy.include_stop:
            stop_finite = torch.isfinite(logits[:, self.stop_idx])
            if not stop_finite.all():
                all_finite = self._check(
                    "policy_stop_logit", logits[:, self.stop_idx],
                    extra=diagnostic_extra,
                ) and all_finite
        if all_finite:
            return logits
        bad_rows = torch.zeros(
            logits.size(0), dtype=torch.bool, device=logits.device
        )
        for name in names:
            value = details[name]
            bad_rows |= ~torch.isfinite(value.reshape(value.size(0), -1)).all(dim=1)
        fallback = logits.new_full(logits.shape, -1e9)
        if self.policy.include_stop:
            fallback[:, self.stop_idx] = 0.0
        else:
            safe_action = action_mask.float().argmax(1)
            fallback.scatter_(1, safe_action.unsqueeze(1), 0.0)
        logits = torch.where(bad_rows.unsqueeze(1), fallback, logits)
        count = int(bad_rows.sum().item())
        self.policy_fallback_count += count
        print(f"[policy-fallback] count={count}")
        assert_finite_tensor(
            "policy_logits_fallback",
            logits,
            raise_on_nonfinite=True,
            **self._numerics_context,
        )
        return logits

    def _policy_inputs(self, state, encoded, operators, modality_mask, prev_action, step):
        state_pool = state.mean(1)
        self._check("operator_state", state)
        context = self._context(encoded, modality_mask)
        self._check("operator_hidden", state_pool)
        commutators = self._commutator_features(operators, state)
        outputs = self.prediction_outputs(
            state, encoded, modality_mask, operators=operators
        )
        current_pred, _, _ = self.direct_uncertainty_features(outputs["logits"])
        concept_conf = torch.stack(
            [encoded["conf"][modality] for modality in self.modalities], 1
        ) * modality_mask.unsqueeze(-1)
        step_tensor = torch.full(
            (state.size(0),), int(step), dtype=torch.long, device=state.device
        )
        extra_features = None
        if self.policy_direct_features:
            extra_features = self._policy_extra_features(
                outputs, operators, state, modality_mask
            )
        details = self.policy(
            state_pool,
            context,
            commutators,
            current_pred,
            concept_conf,
            prev_action,
            step_tensor,
            modality_mask,
            extra_features,
            return_details=True,
        )
        self._last_policy_details = details
        diagnostic_extra = {
            "modality_mask": modality_mask,
            "direct_logits": outputs["direct_logits"],
            "calibrated_probability": current_pred,
            "commutator": commutators,
            "alpha": outputs["operator_gate"],
        }
        logits = self._validated_policy_logits(details, diagnostic_extra)
        state_norm = details["policy_state"].float().norm(dim=1)
        diagnostics = {
            "policy_logits_min": float(logits.detach().min().item()),
            "policy_logits_max": float(logits.detach().max().item()),
            "policy_state_norm_mean": float(state_norm.detach().mean().item()),
            "policy_state_norm_max": float(state_norm.detach().max().item()),
            "fallback_count": int(self.policy_fallback_count),
        }
        self._last_policy_diagnostics = diagnostics
        return logits

    def _decode_action(self, action):
        is_stop = action.eq(self.stop_idx)
        safe = action.clamp_max(self.stop_idx - 1)
        modality = torch.div(safe, self.K, rounding_mode="floor")
        concept = safe % self.K
        return modality, concept, is_stop

    def apply_action(self, state, encoded, operators, action):
        modality_index, concept_index, is_stop = self._decode_action(action)
        new_state = state.clone()
        evidence_strength = state.new_zeros(state.size(0))
        gates = self.gates.gate(concept_index)
        modality_mask = encoded.get("modality_mask")
        beta = self.state_gate_value()
        self._check("state_gate_beta", beta)
        for index, modality in enumerate(self.modalities):
            selected = (~is_stop) & modality_index.eq(index)
            if modality_mask is not None:
                selected = selected & modality_mask[:, index].gt(0)
            if not selected.any():
                continue
            operator = operators[modality][selected]
            gate = gates[selected]
            localized = localize_operator(
                operator, gate, eta=self.eta, eps=self.numerics_eps
            )
            self._check(f"localized_operator.{modality}", localized)
            old_state = state[selected]
            with torch.autocast(device_type=state.device.type, enabled=False):
                propagated = torch.bmm(localized.float(), old_state.float())
                evidence = (
                    encoded["z"][modality][selected].float()
                    * gate.float().unsqueeze(-1)
                )
            delta = self.updaters[modality](propagated + evidence)
            self._check(f"state_delta.{modality}", delta)
            state_update = (beta.float() * delta.float()).clamp(
                -float(self.numerics_cfg.get("state_update_clip", 10.0)),
                float(self.numerics_cfg.get("state_update_clip", 10.0)),
            )
            self._check(f"state_update.{modality}", state_update)
            self._observe_max("max_abs_state_update", state_update)
            updated = self.state_norm(old_state.float() + state_update)
            self._check(f"operator_state_updated.{modality}", updated)
            new_state[selected] = updated.to(new_state.dtype)
            chosen_confidence = encoded["conf"][modality][selected].gather(
                1, concept_index[selected].unsqueeze(1)
            ).squeeze(1)
            evidence_strength[selected] = chosen_confidence
        return new_state, evidence_strength

    def initial(self, batch_size, device):
        return self.initial_state.unsqueeze(0).expand(batch_size, -1, -1).to(device)

    def _finish_rollout(self, state, encoded, modality_mask, actions, logps,
                        entropies, evidence_steps, operators, reason_mask=None,
                        reason_probability=None):
        outputs = self.prediction_outputs(
            state, encoded, modality_mask, operators=operators,
            reason_mask=reason_mask,
        )
        non_stop = actions.ne(self.stop_idx)
        length = non_stop.sum(1).float()
        evidence = torch.stack(evidence_steps, 1).sum(1) / (
            length.clamp_min(1.0) + self.numerics_eps
        )
        outputs.update(
            {
                "actions": actions,
                "logprob": torch.stack(logps, 1).sum(1),
                "entropy": torch.stack(entropies, 1).sum(1),
                "length": length,
                "evidence": evidence,
                "state": state,
                "encoded": encoded,
                "operators": operators,
            }
        )
        if reason_probability is not None:
            outputs["reason_probability"] = reason_probability
        outputs.update(self._last_policy_diagnostics)
        return outputs

    def sample_rollout(self, batch, encoded=None, operators=None, deterministic=False):
        if encoded is None:
            encoded = self.encode(batch)
        if operators is None:
            operators = self.build_operators(encoded, batch["modality_mask"])
        batch_size = batch["modality_mask"].size(0)
        device = batch["modality_mask"].device
        state = self.initial(batch_size, device)
        previous = torch.full(
            (batch_size,), self.policy.num_actions, dtype=torch.long, device=device
        )
        done = torch.zeros(batch_size, dtype=torch.bool, device=device)
        actions, logps, entropies, evidence_steps = [], [], [], []
        reason_probability = None
        reason_mask = None
        if self.performance_v5:
            direct = self.direct_outputs(encoded, batch["modality_mask"])
            initial_outputs = self.prediction_outputs(
                state,
                encoded,
                batch["modality_mask"],
                operators=operators,
                direct=direct,
                reason_mask=torch.ones(batch_size, device=device),
            )
            reason_probability = initial_outputs["reason_probability"].squeeze(-1)
            reason_distribution = torch.distributions.Bernoulli(
                probs=reason_probability
            )
            reason_mask = (
                reason_probability >= self.reason_threshold
                if deterministic
                else reason_distribution.sample().bool()
            )
            done = ~reason_mask
            reason_action = reason_mask.to(reason_probability.dtype)
            reason_logprob = reason_distribution.log_prob(reason_action)
            reason_entropy = reason_distribution.entropy()
        for step in range(self.max_steps):
            policy_logits = self._policy_inputs(
                state, encoded, operators, batch["modality_mask"], previous, step
            )
            self._check("policy_logits_before_categorical", policy_logits, force=True)
            distribution = torch.distributions.Categorical(logits=policy_logits)
            policy_action = policy_logits.argmax(-1) if deterministic else distribution.sample()
            action = torch.where(done, torch.full_like(policy_action, self.stop_idx), policy_action)
            logprob = distribution.log_prob(policy_action)
            entropy = distribution.entropy()
            proposed, evidence = self.apply_action(state, encoded, operators, action)
            state = torch.where(done.view(-1, 1, 1), state, proposed)
            actions.append(action)
            logps.append(torch.where(done, torch.zeros_like(logprob), logprob))
            entropies.append(torch.where(done, torch.zeros_like(entropy), entropy))
            evidence_steps.append(torch.where(done, torch.zeros_like(evidence), evidence))
            done = done | action.eq(self.stop_idx)
            previous = action
        stacked_actions = torch.stack(actions, 1)
        if self.performance_v5:
            path_logprob = torch.stack(logps, 1).sum(1)
            logps[0] = logps[0] + reason_logprob
            entropies[0] = entropies[0] + reason_entropy
        result = self._finish_rollout(
            state, encoded, batch["modality_mask"], stacked_actions, logps,
            entropies, evidence_steps, operators, reason_mask, reason_probability
        )
        if self.performance_v5:
            result["path_logprob"] = path_logprob
            result["reason_logprob"] = reason_logprob
        return result

    def rollout_actions(self, batch, actions, encoded=None, operators=None):
        if encoded is None:
            encoded = self.encode(batch)
        if operators is None:
            operators = self.build_operators(encoded, batch["modality_mask"])
        batch_size = actions.size(0)
        device = actions.device
        state = self.initial(batch_size, device)
        previous = torch.full(
            (batch_size,), self.policy.num_actions, dtype=torch.long, device=device
        )
        done = torch.zeros(batch_size, dtype=torch.bool, device=device)
        logps, entropies, evidence_steps = [], [], []
        reason_mask = actions[:, 0].ne(self.stop_idx) if self.performance_v5 else None
        reason_probability = None
        if self.performance_v5:
            direct = self.direct_outputs(encoded, batch["modality_mask"])
            initial_outputs = self.prediction_outputs(
                state,
                encoded,
                batch["modality_mask"],
                operators=operators,
                direct=direct,
                reason_mask=torch.ones(batch_size, device=device),
            )
            reason_probability = initial_outputs["reason_probability"].squeeze(-1)
            reason_distribution = torch.distributions.Bernoulli(probs=reason_probability)
            done = ~reason_mask
        for step in range(actions.size(1)):
            action = actions[:, step]
            policy_logits = self._policy_inputs(
                state, encoded, operators, batch["modality_mask"], previous, step
            )
            self._check("policy_logits_before_categorical", policy_logits, force=True)
            distribution = torch.distributions.Categorical(logits=policy_logits)
            safe_action = action.clamp_max(self.policy.num_actions - 1)
            logprob, entropy = distribution.log_prob(safe_action), distribution.entropy()
            proposed, evidence = self.apply_action(state, encoded, operators, action)
            active = ~done
            state = torch.where(active.view(-1, 1, 1), proposed, state)
            logps.append(torch.where(active, logprob, torch.zeros_like(logprob)))
            entropies.append(torch.where(active, entropy, torch.zeros_like(entropy)))
            evidence_steps.append(torch.where(active, evidence, torch.zeros_like(evidence)))
            done = done | action.eq(self.stop_idx)
            previous = action
        if self.performance_v5:
            path_logprob = torch.stack(logps, 1).sum(1)
            reason_action = reason_mask.to(reason_probability.dtype)
            reason_logprob = reason_distribution.log_prob(reason_action)
            logps[0] = logps[0] + reason_logprob
            entropies[0] = entropies[0] + reason_distribution.entropy()
        result = self._finish_rollout(
            state, encoded, batch["modality_mask"], actions, logps, entropies,
            evidence_steps, operators, reason_mask, reason_probability
        )
        if self.performance_v5:
            result["path_logprob"] = path_logprob
            result["reason_logprob"] = reason_logprob
        return result

    def forward(self, batch, deterministic=True):
        return self.sample_rollout(batch, deterministic=deterministic)

    def reverse_actions(self, actions):
        reversed_actions = torch.full_like(actions, self.stop_idx)
        for index in range(actions.size(0)):
            sequence = actions[index][actions[index] != self.stop_idx]
            sequence = torch.flip(sequence, dims=[0])
            reversed_actions[index, :len(sequence)] = sequence
        return reversed_actions

    def ablate_selected_evidence(self, encoded, actions):
        z = {modality: value.clone() for modality, value in encoded["z"].items()}
        confidence = {
            modality: value.clone() for modality, value in encoded["conf"].items()
        }
        for index in range(actions.size(0)):
            for action in actions[index].tolist():
                if action == self.stop_idx:
                    continue
                modality_index, concept = divmod(action, self.K)
                modality = self.modalities[modality_index]
                z[modality][index, concept] = 0
                confidence[modality][index, concept] = 0
        return {
            "h": encoded["h"],
            "z": z,
            "conf": confidence,
            "modality_mask": encoded.get("modality_mask"),
        }
