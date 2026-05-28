from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any

from .dataset import validate_rollout_dataset
from .rollout import RolloutEpisode, RolloutTransition


@dataclass(frozen=True)
class ReturnAdvantageBatch:
    returns: tuple[float, ...]
    advantages: tuple[float, ...]


def train_policy_on_episode(
    episode: RolloutEpisode,
    *,
    checkpoint_path: str | Path | None = None,
    seed: int = 0,
    hidden_size: int = 64,
    learning_rate: float = 1.0e-3,
    epochs: int = 1,
    return_mode: str = "reward_as_return",
    discount_factor: float = 0.99,
) -> dict[str, Any]:
    return train_policy_on_episodes(
        (episode,),
        checkpoint_path=checkpoint_path,
        seed=seed,
        hidden_size=hidden_size,
        learning_rate=learning_rate,
        epochs=epochs,
        return_mode=return_mode,
        discount_factor=discount_factor,
    )


def train_policy_on_episodes(
    episodes: Iterable[RolloutEpisode],
    *,
    checkpoint_path: str | Path | None = None,
    seed: int = 0,
    hidden_size: int = 64,
    learning_rate: float = 1.0e-3,
    epochs: int = 1,
    return_mode: str = "reward_as_return",
    discount_factor: float = 0.99,
) -> dict[str, Any]:
    torch = _load_torch()
    from .ppo import compute_masked_ppo_loss
    from .torch_policy import MaskedCandidatePolicyNetwork

    episode_tuple = tuple(episodes)
    dataset_summary = validate_rollout_dataset(episode_tuple)
    trainable_transitions = _trainable_transitions(episode_tuple)
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
    batch = _transitions_to_batch(
        trainable_transitions,
        return_mode=return_mode,
        discount_factor=discount_factor,
    )

    losses = None
    epoch_losses: list[dict[str, Any]] = []
    for epoch_index in range(epochs):
        optimizer.zero_grad()
        losses = compute_masked_ppo_loss(network, **batch)
        losses.total_loss.backward()
        optimizer.step()
        epoch_losses.append(_loss_record(losses, epoch=epoch_index + 1))

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
            epoch_count=epochs,
            learning_rate=learning_rate,
            return_mode=return_mode,
            discount_factor=discount_factor,
        )

    result = {
        "loss": float(losses.total_loss.detach()),
        "total_loss": float(losses.total_loss.detach()),
        "policy_loss": float(losses.policy_loss.detach()),
        "value_loss": float(losses.value_loss.detach()),
        "entropy": float(losses.entropy.detach()),
        "epoch_losses": epoch_losses,
        "sample_count": len(trainable_transitions),
        "epochs": int(epochs),
        "seed": int(seed),
        "hidden_size": int(hidden_size),
        "learning_rate": float(learning_rate),
        "return_mode": str(return_mode),
        "discount_factor": float(discount_factor),
        "dataset_summary": dataset_summary,
    }
    result["warnings"] = _training_quality_warnings(result)
    return result


def load_policy_checkpoint(path: str | Path):
    torch = _load_torch()
    from .torch_policy import MaskedCandidatePolicyNetwork, TorchPolicyScorer

    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    metadata = checkpoint.get("metadata", {})
    candidate_feature_names = checkpoint.get("candidate_feature_names", metadata.get("candidate_feature_names"))
    global_feature_names = checkpoint.get("global_feature_names", metadata.get("global_feature_names"))
    hidden_size = checkpoint.get("hidden_size", metadata.get("hidden_size"))
    if candidate_feature_names is None or global_feature_names is None or hidden_size is None:
        raise ValueError("checkpoint is missing policy feature metadata")
    network = MaskedCandidatePolicyNetwork(
        candidate_feature_count=len(candidate_feature_names),
        global_feature_count=len(global_feature_names),
        hidden_size=int(hidden_size),
    )
    network.load_state_dict(checkpoint["state_dict"])
    network.eval()
    return TorchPolicyScorer(network)


def compute_returns_and_advantages(
    *,
    rewards: Iterable[float],
    dones: Iterable[bool] | None = None,
    values: Iterable[float] | None = None,
    discount_factor: float = 0.99,
    mode: str = "reward_as_return",
) -> ReturnAdvantageBatch:
    reward_values = tuple(float(reward) for reward in rewards)
    done_values = tuple(False for _ in reward_values) if dones is None else tuple(bool(done) for done in dones)
    if len(done_values) != len(reward_values):
        raise ValueError("dones length must match rewards length")

    normalized_mode = "reward_as_return" if mode in {"reward", "reward_as_return"} else mode
    if normalized_mode == "reward_as_return":
        returns = reward_values
    elif normalized_mode == "discounted":
        returns_list = [0.0 for _ in reward_values]
        running_return = 0.0
        for index in range(len(reward_values) - 1, -1, -1):
            if done_values[index]:
                running_return = 0.0
            running_return = reward_values[index] + float(discount_factor) * running_return
            returns_list[index] = running_return
        returns = tuple(returns_list)
    else:
        raise ValueError(f"unknown return mode: {mode}")

    if values is None:
        advantages = returns
    else:
        value_tuple = tuple(float(value) for value in values)
        if len(value_tuple) != len(returns):
            raise ValueError("values length must match rewards length")
        advantages = tuple(return_value - value for return_value, value in zip(returns, value_tuple))

    return ReturnAdvantageBatch(
        returns=tuple(float(value) for value in returns),
        advantages=tuple(float(value) for value in advantages),
    )


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


def _loss_record(losses, *, epoch: int) -> dict[str, Any]:
    return {
        "epoch": int(epoch),
        "loss": float(losses.total_loss.detach()),
        "total_loss": float(losses.total_loss.detach()),
        "policy_loss": float(losses.policy_loss.detach()),
        "value_loss": float(losses.value_loss.detach()),
        "entropy": float(losses.entropy.detach()),
    }


def _training_quality_warnings(result: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    dataset_summary = result.get("dataset_summary", {})
    trainable_count = int(dataset_summary.get("trainable_transition_count", result.get("sample_count", 0)))
    if trainable_count < 2:
        warnings.append("trainable_transition_count_too_low")
    reward_summary = dataset_summary.get("reward", {})
    if (
        isinstance(reward_summary, dict)
        and float(reward_summary.get("min", 0.0)) == 0.0
        and float(reward_summary.get("max", 0.0)) == 0.0
    ):
        warnings.append("reward_all_zero")
    entropy = float(result.get("entropy", 0.0))
    if entropy < 1.0e-3:
        warnings.append("entropy_too_low")
    value_loss = float(result.get("value_loss", 0.0))
    if not isfinite(value_loss):
        warnings.append("value_loss_non_finite")
    elif value_loss > 1.0e6:
        warnings.append("value_loss_abnormally_large")
    return warnings


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


def _transitions_to_batch(
    transitions: tuple[RolloutTransition, ...],
    *,
    return_mode: str,
    discount_factor: float,
) -> dict[str, Any]:
    torch = _load_torch()
    observations = tuple(transition.observation for transition in transitions)
    action_count = max(len(observation.action_mask) for observation in observations)
    return_advantage_batch = compute_returns_and_advantages(
        rewards=(transition.reward for transition in transitions),
        dones=(transition.done for transition in transitions),
        values=(0.0 if transition.value is None else transition.value for transition in transitions),
        discount_factor=discount_factor,
        mode=return_mode,
    )
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
            return_advantage_batch.returns,
            dtype=torch.float32,
        ),
        "advantages": torch.tensor(
            return_advantage_batch.advantages,
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
    epoch_count: int,
    learning_rate: float,
    return_mode: str,
    discount_factor: float,
) -> None:
    torch = _load_torch()
    torch.save(
        {
            "state_dict": network.state_dict(),
            "hidden_size": hidden_size,
            "candidate_feature_names": tuple(candidate_feature_names),
            "global_feature_names": tuple(global_feature_names),
            "metadata": {
                "format": "model-explorer-masked-policy",
                "version": 2,
                "format_version": "model-explorer-masked-policy/v2",
                "candidate_feature_names": tuple(candidate_feature_names),
                "global_feature_names": tuple(global_feature_names),
                "action_count": int(action_count),
                "seed": int(seed),
                "sample_count": int(sample_count),
                "epoch_count": int(epoch_count),
                "hidden_size": int(hidden_size),
                "learning_rate": float(learning_rate),
                "return_mode": str(return_mode),
                "discount_factor": float(discount_factor),
            },
        },
        Path(path),
    )


def _load_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for masked policy training") from exc
    return torch
