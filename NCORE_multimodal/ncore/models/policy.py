from __future__ import annotations
import torch
import torch.nn as nn


class PathPolicy(nn.Module):
    def __init__(self, state_dim, context_dim, num_modalities, num_concepts,
                 num_tasks, num_pairs, hidden_dim=256, max_steps=3,
                 action_emb_dim=32, extra_dim=0, include_stop=True):
        super().__init__()
        self.num_modalities = num_modalities
        self.num_concepts = num_concepts
        self.include_stop = bool(include_stop)
        self.num_actions = num_modalities * num_concepts + int(self.include_stop)
        self.stop_idx = self.num_actions - 1 if self.include_stop else None
        self.action_emb = nn.Embedding(self.num_actions + 1, action_emb_dim)
        self.step_emb = nn.Embedding(max_steps + 1, action_emb_dim)
        self.extra_dim = int(extra_dim)
        self.base_input_dim = state_dim + context_dim + num_pairs + num_tasks + num_modalities*num_concepts + 2*action_emb_dim
        self.input_dim = self.base_input_dim + self.extra_dim
        self.input_norm = nn.LayerNorm(self.input_dim)
        self.net = nn.Sequential(nn.Linear(self.input_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, self.num_actions))

    def valid_action_mask(self, modality_mask):
        action_mask = (
            modality_mask.gt(0)
            .unsqueeze(-1)
            .expand(-1, -1, self.num_concepts)
            .reshape(modality_mask.size(0), -1)
        )
        if not self.include_stop:
            return action_mask
        stop_valid = torch.ones(
            modality_mask.size(0), 1, dtype=torch.bool,
            device=modality_mask.device,
        )
        return torch.cat([action_mask, stop_valid], dim=1)

    def forward(self, state_pool, context, comm, current_pred, concept_conf,
                prev_action, step, modality_mask, extra_features=None,
                return_details=False):
        pa = self.action_emb(prev_action)
        se = self.step_emb(step)
        inputs = [state_pool, context, comm, current_pred, concept_conf.flatten(1), pa, se]
        if self.extra_dim:
            if extra_features is None or extra_features.size(-1) != self.extra_dim:
                raise ValueError(
                    f"Policy expected {self.extra_dim} direct-aware features"
                )
            inputs.append(extra_features)
        policy_state_pre_norm = torch.cat(inputs, dim=-1)
        policy_state = self.input_norm(policy_state_pre_norm)
        logits_pre_mask = self.net(policy_state)
        # Mask all K actions for unavailable modalities; legacy STOP is always valid.
        action_mask = self.valid_action_mask(modality_mask)
        if not action_mask.any(dim=1).all():
            raise RuntimeError("Every sample must have at least one valid action")
        logits_post_mask = logits_pre_mask.masked_fill(~action_mask, -1e9)
        if return_details:
            return {
                "policy_state_pre_norm": policy_state_pre_norm,
                "policy_state": policy_state,
                "policy_logits_pre_mask": logits_pre_mask,
                "policy_logits_post_mask": logits_post_mask,
                "valid_action_mask": action_mask,
            }
        return logits_post_mask
