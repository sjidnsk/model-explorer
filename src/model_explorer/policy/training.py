from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import torch

from .ppo import compute_masked_ppo_loss
from .rollout import RolloutEpisode, RolloutTransition
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
    return train_policy_on_episodes(
        (episode,),
        checkpoint_path=checkpoint_path,
        seed=seed,
        hidden_size=hidden_size,
        learning_rate=learning_rate,
        epochs=epochs,
    )


def train_policy_on_episodes(
    episodes: Iterable[RolloutEpisode],
    *,
    checkpoint_path: str | Path | None = None,
    seed: int = 0,
    hidden_size: int = 64,
    learning_rate: float = 1.0e-3,
    epochs: int = 1,
) -> dict[str, float]:
    trainable_transitions = _trainable_transitions(tuple(episodes))
    if not trainable_transitions:
        raise ValueError("episodes must contain at least one trainable transition")

    torch.manual_seed(seed)
    first_observation = trainable_transitions[0].observation
    network = MaskedCandidatePolicyNetwork(
        candidate_feature_count=len(first_observation.candidate_feature_names),
        global_feature_count=len(first_observation.global_feature_names),
        hidden_size=hidden_size,
    )
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
    batch = _transitions_to_batch(trainable_transitions)

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
            action_count=max(len(transition.observation.action_mask) for transition in trainable_transitions),
            seed=seed,
            sample_count=len(trainable_transitions),
        )

    return {
        "loss": float(losses.total_loss.detach()),
        "total_loss": float(losses.total_loss.detach()),
        "policy_loss": float(losses.policy_loss.detach()),
        "value_loss": float(losses.value_loss.detach()),
        "entropy": float(losses.entropy.detach()),
        "sample_count": len(trainable_transitions),
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


def _trainable_transitions(episodes: tuple[RolloutEpisode, ...]) -> tuple[RolloutTransition, ...]:
    transitions = tuple(
        transition
        for episode in episodes
        for transition in episode.transitions
        if transition.action_index >= 0
    )
    if transitions:
        _validate_transition_shapes(transitions)
    return transitions


def _validate_transition_shapes(transitions: tuple[RolloutTransition, ...]) -> None:
    first = transitions[0].observation
    candidate_feature_count = len(first.candidate_feature_names)
    global_feature_count = len(first.global_feature_names)
    for transition in transitions:
        observation = transition.observation
        if len(observation.candidate_feature_names) != candidate_feature_count:
            raise ValueError("all training observations must use the same candidate feature count")
        if len(observation.global_feature_names) != global_feature_count:
            raise ValueError("all training observations must use the same global feature count")


def _transitions_to_batch(transitions: tuple[RolloutTransition, ...]) -> dict[str, torch.Tensor]:
    observations = tuple(transition.observation for transition in transitions)
    action_count = max(len(observation.action_mask) for observation in observations)
    return {
        "candidate_features": torch.tensor(
            [_padded_candidate_features(observation, action_count) for observation in observations],
            dtype=torch.float32,
        ),
        "global_features": torch.tensor(
            [observation.global_features for observation in observations],
            dtype=torch.float32,
        ),
        "action_mask": torch.tensor(
            [_padded_action_mask(observation, action_count) for observation in observations],
            dtype=torch.bool,
        ),
        "actions": torch.tensor(
            [transition.action_index for transition in transitions],
            dtype=torch.long,
        ),
        "old_log_probs": torch.tensor(
            [0.0 if transition.log_prob is None else transition.log_prob for transition in transitions],
            dtype=torch.float32,
        ),
        "returns": torch.tensor(
            [transition.reward for transition in transitions],
            dtype=torch.float32,
        ),
        "advantages": torch.tensor(
            [transition.reward for transition in transitions],
            dtype=torch.float32,
        ),
    }


def _padded_candidate_features(observation, action_count: int) -> tuple[tuple[float, ...], ...]:
    rows = tuple(tuple(float(value) for value in row) for row in observation.candidate_features)
    if len(rows) >= action_count:
        return rows
    zero_row = tuple(0.0 for _ in observation.candidate_feature_names)
    return rows + tuple(zero_row for _ in range(action_count - len(rows)))


def _padded_action_mask(observation, action_count: int) -> tuple[bool, ...]:
    values = tuple(bool(value) for value in observation.action_mask)
    if len(values) >= action_count:
        return values
    return values + tuple(False for _ in range(action_count - len(values)))


def _save_policy_checkpoint(
    path: str | Path,
    *,
    network: MaskedCandidatePolicyNetwork,
    hidden_size: int,
    candidate_feature_names: tuple[str, ...],
    global_feature_names: tuple[str, ...],
    action_count: int,
    seed: int,
    sample_count: int,
) -> None:
    torch.save(
        {
            "state_dict": network.state_dict(),
            "hidden_size": hidden_size,
            "candidate_feature_names": tuple(candidate_feature_names),
            "global_feature_names": tuple(global_feature_names),
            "metadata": {
                "format": "model-explorer-masked-policy/v1",
                "action_count": int(action_count),
                "seed": int(seed),
                "sample_count": int(sample_count),
            },
        },
        Path(path),
    )
