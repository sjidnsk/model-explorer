from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from .torch_policy import MaskedCandidatePolicyNetwork


@dataclass(frozen=True)
class PpoLosses:
    total_loss: torch.Tensor
    policy_loss: torch.Tensor
    value_loss: torch.Tensor
    entropy: torch.Tensor


def compute_masked_ppo_loss(
    network: MaskedCandidatePolicyNetwork,
    *,
    candidate_features: torch.Tensor,
    global_features: torch.Tensor,
    action_mask: torch.Tensor,
    actions: torch.Tensor,
    old_log_probs: torch.Tensor,
    returns: torch.Tensor,
    advantages: torch.Tensor,
    clip_ratio: float = 0.2,
    value_loss_coefficient: float = 0.5,
    entropy_coefficient: float = 0.01,
) -> PpoLosses:
    actions = actions.to(device=candidate_features.device, dtype=torch.long)
    old_log_probs = old_log_probs.to(device=candidate_features.device, dtype=torch.float32)
    returns = returns.to(device=candidate_features.device, dtype=torch.float32)
    advantages = advantages.to(device=candidate_features.device, dtype=torch.float32)

    if actions.ndim != 1:
        raise ValueError("actions must have shape [batch]")
    if actions.shape[0] != candidate_features.shape[0]:
        raise ValueError("actions batch size must match candidate_features")
    if not action_mask[torch.arange(actions.shape[0], device=actions.device), actions].all():
        raise ValueError("actions must reference unmasked candidates")

    output = network(
        candidate_features=candidate_features,
        global_features=global_features,
        action_mask=action_mask,
    )
    distribution = torch.distributions.Categorical(logits=output.masked_logits)
    log_probs = distribution.log_prob(actions)
    ratios = torch.exp(log_probs - old_log_probs)
    unclipped = ratios * advantages
    clipped = torch.clamp(ratios, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages

    policy_loss = -torch.min(unclipped, clipped).mean()
    value_loss = F.mse_loss(output.value, returns)
    entropy = distribution.entropy().mean()
    total_loss = policy_loss + value_loss_coefficient * value_loss - entropy_coefficient * entropy

    return PpoLosses(
        total_loss=total_loss,
        policy_loss=policy_loss,
        value_loss=value_loss,
        entropy=entropy,
    )
