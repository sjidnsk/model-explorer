from __future__ import annotations

from pathlib import Path

import torch

from .ppo import compute_masked_ppo_loss
from .rollout import RolloutEpisode
from .torch_policy import MaskedCandidatePolicyNetwork, TorchPolicyScorer


def train_policy_on_episode(
    episode: RolloutEpisode,
    *,
    checkpoint_path: str | Path | None = None,
    seed: int = 0,
    hidden_size: int = 64,
    learning_rate: float = 1.0e-3,
    epochs: int = 1,
) -> dict[str, float]:
    if not episode.transitions:
        raise ValueError("episode must contain at least one transition")

    torch.manual_seed(seed)
    first_observation = episode.transitions[0].observation
    network = MaskedCandidatePolicyNetwork(
        candidate_feature_count=len(first_observation.candidate_feature_names),
        global_feature_count=len(first_observation.global_feature_names),
        hidden_size=hidden_size,
    )
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
    batch = _episode_to_batch(episode)

    losses = None
    for _ in range(epochs):
        optimizer.zero_grad()
        losses = compute_masked_ppo_loss(network, **batch)
        losses.total_loss.backward()
        optimizer.step()

    if losses is None:
        raise ValueError("epochs must be at least 1")

    if checkpoint_path is not None:
        _save_policy_checkpoint(
            checkpoint_path,
            network=network,
            hidden_size=hidden_size,
            candidate_feature_names=first_observation.candidate_feature_names,
            global_feature_names=first_observation.global_feature_names,
        )

    return {
        "total_loss": float(losses.total_loss.detach()),
        "policy_loss": float(losses.policy_loss.detach()),
        "value_loss": float(losses.value_loss.detach()),
        "entropy": float(losses.entropy.detach()),
    }


def load_policy_checkpoint(path: str | Path) -> TorchPolicyScorer:
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    network = MaskedCandidatePolicyNetwork(
        candidate_feature_count=len(checkpoint["candidate_feature_names"]),
        global_feature_count=len(checkpoint["global_feature_names"]),
        hidden_size=int(checkpoint["hidden_size"]),
    )
    network.load_state_dict(checkpoint["state_dict"])
    network.eval()
    return TorchPolicyScorer(network)


def _episode_to_batch(episode: RolloutEpisode) -> dict[str, torch.Tensor]:
    observations = tuple(transition.observation for transition in episode.transitions)
    return {
        "candidate_features": torch.tensor(
            [observation.candidate_features for observation in observations],
            dtype=torch.float32,
        ),
        "global_features": torch.tensor(
            [observation.global_features for observation in observations],
            dtype=torch.float32,
        ),
        "action_mask": torch.tensor(
            [observation.action_mask for observation in observations],
            dtype=torch.bool,
        ),
        "actions": torch.tensor(
            [transition.action_index for transition in episode.transitions],
            dtype=torch.long,
        ),
        "old_log_probs": torch.tensor(
            [0.0 if transition.log_prob is None else transition.log_prob for transition in episode.transitions],
            dtype=torch.float32,
        ),
        "returns": torch.tensor(
            [transition.reward for transition in episode.transitions],
            dtype=torch.float32,
        ),
        "advantages": torch.tensor(
            [transition.reward for transition in episode.transitions],
            dtype=torch.float32,
        ),
    }


def _save_policy_checkpoint(
    path: str | Path,
    *,
    network: MaskedCandidatePolicyNetwork,
    hidden_size: int,
    candidate_feature_names: tuple[str, ...],
    global_feature_names: tuple[str, ...],
) -> None:
    torch.save(
        {
            "state_dict": network.state_dict(),
            "hidden_size": hidden_size,
            "candidate_feature_names": tuple(candidate_feature_names),
            "global_feature_names": tuple(global_feature_names),
        },
        Path(path),
    )
