from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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
    architecture_name = "mlp_v1"
    uses_missing_indicators = False

    def __init__(
        self,
        *,
        candidate_feature_count: int,
        global_feature_count: int,
        hidden_size: int = 64,
        missing_indicator_count: int = 0,
        dropout: float = 0.0,
        architecture_config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.candidate_feature_count = int(candidate_feature_count)
        self.global_feature_count = int(global_feature_count)
        self.hidden_size = int(hidden_size)
        self.missing_indicator_count = int(missing_indicator_count)
        self.dropout = float(dropout)
        self.architecture_config = dict(
            architecture_config
            if architecture_config is not None
            else {"hidden_dim": self.hidden_size, "dropout": self.dropout}
        )
        candidate_encoder_input_count = self.candidate_feature_count
        if self.uses_missing_indicators:
            candidate_encoder_input_count += self.missing_indicator_count
        self.candidate_encoder = nn.Sequential(
            nn.Linear(candidate_encoder_input_count, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(self.dropout),
        )
        self.global_encoder = nn.Sequential(
            nn.Linear(global_feature_count, hidden_size),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(self.dropout),
        )
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(hidden_size, 1),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(
        self,
        *,
        candidate_features: torch.Tensor,
        global_features: torch.Tensor,
        action_mask: torch.Tensor,
        candidate_missing_indicators: torch.Tensor | None = None,
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

        candidate_embedding = self.candidate_encoder(
            self._candidate_encoder_inputs(
                candidate_features,
                candidate_missing_indicators=candidate_missing_indicators,
            )
        )
        candidate_embedding = self._contextualize_candidates(candidate_embedding, mask=mask)
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

    def _candidate_encoder_inputs(
        self,
        candidate_features: torch.Tensor,
        *,
        candidate_missing_indicators: torch.Tensor | None,
    ) -> torch.Tensor:
        if not self.uses_missing_indicators:
            return candidate_features
        if self.missing_indicator_count <= 0:
            raise ValueError("missing indicator architecture requires missing indicators")
        if candidate_missing_indicators is None:
            candidate_missing_indicators = torch.zeros(
                (*candidate_features.shape[:2], self.missing_indicator_count),
                dtype=candidate_features.dtype,
                device=candidate_features.device,
            )
        if candidate_missing_indicators.ndim != 3:
            raise ValueError("candidate_missing_indicators must have shape [batch, candidates, indicators]")
        if candidate_missing_indicators.shape[:2] != candidate_features.shape[:2]:
            raise ValueError("candidate_missing_indicators must match candidate batch and count")
        if candidate_missing_indicators.shape[2] != self.missing_indicator_count:
            raise ValueError("candidate_missing_indicators feature count does not match network metadata")
        return torch.cat((candidate_features, candidate_missing_indicators.to(candidate_features.dtype)), dim=-1)

    def _contextualize_candidates(self, candidate_embedding: torch.Tensor, *, mask: torch.Tensor) -> torch.Tensor:
        return candidate_embedding


class MissingIndicatorCandidatePolicyNetwork(MaskedCandidatePolicyNetwork):
    architecture_name = "mlp_missing_v1"
    uses_missing_indicators = True


class CandidateAttentionPolicyNetwork(MaskedCandidatePolicyNetwork):
    architecture_name = "candidate_attention_v1"

    def __init__(
        self,
        *,
        candidate_feature_count: int,
        global_feature_count: int,
        hidden_size: int = 64,
        missing_indicator_count: int = 0,
        dropout: float = 0.0,
        attention_heads: int = 1,
        architecture_config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            candidate_feature_count=candidate_feature_count,
            global_feature_count=global_feature_count,
            hidden_size=hidden_size,
            missing_indicator_count=missing_indicator_count,
            dropout=dropout,
            architecture_config=architecture_config,
        )
        self.attention_heads = int(attention_heads)
        self.architecture_config = dict(self.architecture_config)
        self.architecture_config["attention_heads"] = self.attention_heads
        self.candidate_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=self.attention_heads,
            dropout=self.dropout,
            batch_first=True,
        )
        self.attention_norm = nn.LayerNorm(hidden_size)

    def _contextualize_candidates(self, candidate_embedding: torch.Tensor, *, mask: torch.Tensor) -> torch.Tensor:
        attended, _ = self.candidate_attention(
            candidate_embedding,
            candidate_embedding,
            candidate_embedding,
            key_padding_mask=~mask,
            need_weights=False,
        )
        return self.attention_norm(candidate_embedding + attended)


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
        "candidate_missing_indicators": torch.tensor(
            [observation.candidate_missing_indicators],
            dtype=torch.float32,
            device=device,
        ),
    }
