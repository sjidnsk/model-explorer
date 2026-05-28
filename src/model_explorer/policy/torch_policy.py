from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .features import PolicyObservation


@dataclass(frozen=True)
class PolicyNetworkOutput:
    logits: torch.Tensor
    masked_logits: torch.Tensor
    action_probs: torch.Tensor
    value: torch.Tensor


class MaskedCandidatePolicyNetwork(nn.Module):
    def __init__(self, *, candidate_feature_count: int, global_feature_count: int, hidden_size: int = 64) -> None:
        super().__init__()
        self.candidate_encoder = nn.Sequential(
            nn.Linear(candidate_feature_count, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
        )
        self.global_encoder = nn.Sequential(
            nn.Linear(global_feature_count, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
        )
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(
        self,
        *,
        candidate_features: torch.Tensor,
        global_features: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> PolicyNetworkOutput:
        if candidate_features.ndim != 3:
            raise ValueError("candidate_features must have shape [batch, candidates, features]")
        if global_features.ndim != 2:
            raise ValueError("global_features must have shape [batch, features]")

        mask = action_mask.bool()
        if mask.ndim != 2 or mask.shape != candidate_features.shape[:2]:
            raise ValueError("action_mask must have shape [batch, candidates]")
        if not torch.all(mask.any(dim=1)):
            raise ValueError("each policy batch item must contain at least one valid action")

        candidate_embedding = self.candidate_encoder(candidate_features)
        global_embedding = self.global_encoder(global_features)
        expanded_global = global_embedding.unsqueeze(1).expand(-1, candidate_embedding.shape[1], -1)

        logits = self.policy_head(torch.cat((candidate_embedding, expanded_global), dim=-1)).squeeze(-1)
        masked_logits = logits.masked_fill(~mask, -1.0e9)
        action_probs = torch.softmax(masked_logits, dim=-1)

        masked_embedding = candidate_embedding * mask.unsqueeze(-1)
        valid_counts = mask.sum(dim=1, keepdim=True).clamp_min(1).to(candidate_embedding.dtype)
        pooled_candidates = masked_embedding.sum(dim=1) / valid_counts
        value = self.value_head(torch.cat((pooled_candidates, global_embedding), dim=-1)).squeeze(-1)

        return PolicyNetworkOutput(
            logits=logits,
            masked_logits=masked_logits,
            action_probs=action_probs,
            value=value,
        )


class TorchPolicyScorer:
    def __init__(self, network: MaskedCandidatePolicyNetwork) -> None:
        self.network = network

    def score(self, observation: PolicyObservation) -> tuple[float, ...]:
        device = next(self.network.parameters()).device
        tensors = observation_to_tensors(observation, device=device)
        self.network.eval()
        with torch.no_grad():
            output = self.network(**tensors)
        return tuple(float(value) for value in output.masked_logits[0].detach().cpu())


def observation_to_tensors(
    observation: PolicyObservation,
    *,
    device: torch.device | str | None = None,
) -> dict[str, torch.Tensor]:
    return {
        "candidate_features": torch.tensor(
            [observation.candidate_features],
            dtype=torch.float32,
            device=device,
        ),
        "global_features": torch.tensor(
            [observation.global_features],
            dtype=torch.float32,
            device=device,
        ),
        "action_mask": torch.tensor(
            [observation.action_mask],
            dtype=torch.bool,
            device=device,
        ),
    }
