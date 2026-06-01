from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any

from .dataset import validate_rollout_dataset
from .features import OBSERVATION_SCHEMA_VERSION
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
    architecture: str | None = None,
    architecture_config: dict[str, Any] | None = None,
    teacher_imitation_weight: float = 0.0,
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
        architecture=architecture,
        architecture_config=architecture_config,
        teacher_imitation_weight=teacher_imitation_weight,
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
    architecture: str | None = None,
    architecture_config: dict[str, Any] | None = None,
    teacher_imitation_weight: float = 0.0,
) -> dict[str, Any]:
    torch = _load_torch()
    from .architectures import build_policy_network
    from .ppo import compute_masked_ppo_loss

    episode_tuple = tuple(episodes)
    dataset_summary = validate_rollout_dataset(episode_tuple)
    training_source = _training_source_summary(dataset_summary)
    teacher_margin_summary = _teacher_margin_summary(dataset_summary)
    teacher_imitation_weight = max(0.0, float(teacher_imitation_weight))
    trainable_transitions = _trainable_transitions(episode_tuple)
    if not trainable_transitions:
        raise ValueError("episodes must contain at least one trainable transition")

    torch.manual_seed(seed)
    first_observation = trainable_transitions[0].observation
    network = build_policy_network(
        architecture,
        observation=first_observation,
        hidden_size=hidden_size,
        architecture_config=architecture_config,
    )
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
    batch = _transitions_to_batch(
        trainable_transitions,
        return_mode=return_mode,
        discount_factor=discount_factor,
    )

    losses = None
    total_loss = None
    epoch_losses: list[dict[str, Any]] = []
    for epoch_index in range(epochs):
        optimizer.zero_grad()
        losses = compute_masked_ppo_loss(network, **_ppo_batch(batch))
        teacher_loss = (
            _teacher_imitation_loss(network, batch)
            if teacher_imitation_weight > 0.0
            else losses.total_loss.detach().new_tensor(0.0)
        )
        total_loss = losses.total_loss + teacher_imitation_weight * teacher_loss
        total_loss.backward()
        optimizer.step()
        epoch_losses.append(
            _loss_record(
                losses,
                epoch=epoch_index + 1,
                total_loss=total_loss,
                teacher_loss=teacher_loss,
                teacher_imitation_weight=teacher_imitation_weight,
            )
        )

    if losses is None or total_loss is None:
        raise ValueError("epochs must be at least 1")
    teacher_imitation = _teacher_imitation_metrics(
        network,
        batch,
        weight=teacher_imitation_weight,
    )

    if checkpoint_path is not None:
        _save_policy_checkpoint(
            checkpoint_path,
            network=network,
            hidden_size=network.hidden_size,
            candidate_feature_names=first_observation.candidate_feature_names,
            global_feature_names=first_observation.global_feature_names,
            candidate_missing_indicator_names=first_observation.candidate_missing_indicator_names,
            action_count=max(len(transition.observation.action_mask) for transition in trainable_transitions),
            seed=seed,
            sample_count=len(trainable_transitions),
            epoch_count=epochs,
            learning_rate=learning_rate,
            return_mode=return_mode,
            discount_factor=discount_factor,
            training_source=training_source,
            teacher_imitation=teacher_imitation,
            teacher_margin_summary=teacher_margin_summary,
        )

    result = {
        "architecture": network.architecture_name,
        "architecture_config": dict(network.architecture_config),
        "architecture_diagnostics": _architecture_diagnostics(
            network=network,
            candidate_feature_names=first_observation.candidate_feature_names,
            global_feature_names=first_observation.global_feature_names,
            candidate_missing_indicator_names=first_observation.candidate_missing_indicator_names,
            dataset_summary=dataset_summary,
        ),
        "loss": float(total_loss.detach()),
        "total_loss": float(total_loss.detach()),
        "ppo_total_loss": float(losses.total_loss.detach()),
        "policy_loss": float(losses.policy_loss.detach()),
        "value_loss": float(losses.value_loss.detach()),
        "entropy": float(losses.entropy.detach()),
        "epoch_losses": epoch_losses,
        "sample_count": len(trainable_transitions),
        "epochs": int(epochs),
        "seed": int(seed),
        "hidden_size": int(network.hidden_size),
        "learning_rate": float(learning_rate),
        "return_mode": str(return_mode),
        "discount_factor": float(discount_factor),
        "dataset_summary": dataset_summary,
        "training_source": training_source,
        "teacher_margin_summary": teacher_margin_summary,
        "teacher_imitation": teacher_imitation,
    }
    result["warnings"] = _training_quality_warnings(result)
    return result


def load_policy_checkpoint(path: str | Path):
    torch = _load_torch()
    from .architectures import MLP_V1, build_policy_network_from_metadata
    from .torch_policy import TorchPolicyScorer

    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    metadata = checkpoint.get("metadata", {})
    candidate_feature_names = checkpoint.get("candidate_feature_names", metadata.get("candidate_feature_names"))
    global_feature_names = checkpoint.get("global_feature_names", metadata.get("global_feature_names"))
    candidate_missing_indicator_names = checkpoint.get(
        "candidate_missing_indicator_names",
        metadata.get("candidate_missing_indicator_names", ()),
    )
    hidden_size = checkpoint.get("hidden_size", metadata.get("hidden_size"))
    if candidate_feature_names is None or global_feature_names is None or hidden_size is None:
        raise ValueError("checkpoint is missing policy feature metadata")
    network = build_policy_network_from_metadata(
        metadata.get("architecture", MLP_V1),
        candidate_feature_count=len(candidate_feature_names),
        global_feature_count=len(global_feature_names),
        missing_indicator_count=len(candidate_missing_indicator_names),
        hidden_size=int(hidden_size),
        architecture_config=metadata.get("architecture_config"),
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


def _loss_record(
    losses,
    *,
    epoch: int,
    total_loss=None,
    teacher_loss=None,
    teacher_imitation_weight: float = 0.0,
) -> dict[str, Any]:
    total = losses.total_loss if total_loss is None else total_loss
    teacher = losses.total_loss.detach().new_tensor(0.0) if teacher_loss is None else teacher_loss
    return {
        "epoch": int(epoch),
        "loss": float(total.detach()),
        "total_loss": float(total.detach()),
        "ppo_total_loss": float(losses.total_loss.detach()),
        "policy_loss": float(losses.policy_loss.detach()),
        "value_loss": float(losses.value_loss.detach()),
        "entropy": float(losses.entropy.detach()),
        "teacher_imitation_loss": float(teacher.detach()),
        "teacher_imitation_weight": float(teacher_imitation_weight),
        "teacher_imitation_weighted_loss": float((teacher * teacher_imitation_weight).detach()),
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


def _architecture_diagnostics(
    *,
    network,
    candidate_feature_names: tuple[str, ...],
    global_feature_names: tuple[str, ...],
    candidate_missing_indicator_names: tuple[str, ...],
    dataset_summary: dict[str, Any],
) -> dict[str, Any]:
    mask_distribution = dataset_summary.get("reachable_action_count_distribution", {})
    if not isinstance(mask_distribution, dict):
        mask_distribution = {}
    return {
        "architecture": network.architecture_name,
        "architecture_config": dict(network.architecture_config),
        "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
        "candidate_feature_dim": len(candidate_feature_names),
        "global_feature_dim": len(global_feature_names),
        "missing_indicator_dim": len(candidate_missing_indicator_names),
        "mask_valid_action_count": dict(mask_distribution),
    }


def _training_source_summary(dataset_summary: dict[str, Any]) -> dict[str, Any]:
    selection_counts = dataset_summary.get("selection_strategy_counts", {})
    if not isinstance(selection_counts, dict):
        selection_counts = {}
    requested_counts = dataset_summary.get("requested_selection_strategy_counts", {})
    if not isinstance(requested_counts, dict):
        requested_counts = {}
    fractions = dataset_summary.get("selection_strategy_fractions", {})
    if not isinstance(fractions, dict):
        fractions = {}
    primary = dataset_summary.get("primary_selection_strategy")
    transition_count = sum(_safe_int(value) for value in selection_counts.values())
    return {
        "primary_selection_strategy": None if primary is None else str(primary),
        "selection_strategy_counts": {
            str(key): _safe_int(value) for key, value in sorted(selection_counts.items())
        },
        "selection_strategy_fractions": {
            str(key): _safe_float(value) for key, value in sorted(fractions.items())
        },
        "requested_selection_strategy_counts": {
            str(key): _safe_int(value) for key, value in sorted(requested_counts.items())
        },
        "teacher_transition_count": transition_count,
    }


def _teacher_margin_summary(dataset_summary: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "feedback_aware_transition_count",
        "feedback_aware_transition_fraction",
        "teacher_margin_sample_count",
        "teacher_score_margin",
        "teacher_low_margin_threshold",
        "teacher_high_margin_threshold",
        "teacher_low_margin_sample_count",
        "teacher_medium_margin_sample_count",
        "teacher_high_margin_sample_count",
        "teacher_low_margin_sample_rate",
        "missing_teacher_signal_count",
        "missing_teacher_signal_rate",
        "low_margin_only_dataset_rate",
        "teacher_margin_bucket_counts",
    )
    return {key: dataset_summary[key] for key in keys if key in dataset_summary}


def _ppo_batch(batch: dict[str, Any]) -> dict[str, Any]:
    return {
        key: batch[key]
        for key in (
            "candidate_features",
            "global_features",
            "action_mask",
            "actions",
            "old_log_probs",
            "returns",
            "advantages",
            "candidate_missing_indicators",
        )
    }


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _safe_float(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    return numeric if isfinite(numeric) else 0.0


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
    teacher_labels = tuple(_teacher_action_label(transition, action_count) for transition in transitions)
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
        "candidate_missing_indicators": torch.tensor(
            [_padded_missing_indicators(observation, action_count) for observation in observations],
            dtype=torch.float32,
        ),
        "actions": torch.tensor(
            [transition.action_index for transition in transitions],
            dtype=torch.long,
        ),
        "teacher_actions": torch.tensor(
            [0 if label is None else label for label in teacher_labels],
            dtype=torch.long,
        ),
        "teacher_action_valid_mask": torch.tensor(
            [label is not None for label in teacher_labels],
            dtype=torch.bool,
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


def _teacher_action_label(transition: RolloutTransition, action_count: int) -> int | None:
    raw_index = transition.info.extra.get("teacher_action_index")
    if raw_index is None:
        raw_index = transition.action_index
    if isinstance(raw_index, bool):
        return None
    try:
        index = int(raw_index)
    except (TypeError, ValueError):
        return None
    if index < 0 or index >= action_count:
        return None
    mask = transition.observation.action_mask
    if index >= len(mask) or not bool(mask[index]):
        return None
    return index


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


def _padded_missing_indicators(observation, action_count: int) -> tuple[tuple[float, ...], ...]:
    width = len(observation.candidate_missing_indicator_names)
    rows = tuple(tuple(float(value) for value in row) for row in observation.candidate_missing_indicators)
    if not rows:
        rows = tuple(tuple(0.0 for _ in range(width)) for _ in observation.candidate_features)
    if len(rows) >= action_count:
        return rows
    zero_row = tuple(0.0 for _ in range(width))
    return rows + tuple(zero_row for _ in range(action_count - len(rows)))


def _teacher_imitation_loss(network, batch: dict[str, Any]):
    torch = _load_torch()
    from torch.nn import functional as F

    valid_mask = batch["teacher_action_valid_mask"].to(dtype=torch.bool)
    if not bool(valid_mask.any()):
        return batch["candidate_features"].new_tensor(0.0)
    output = network(
        candidate_features=batch["candidate_features"],
        global_features=batch["global_features"],
        action_mask=batch["action_mask"],
        candidate_missing_indicators=batch.get("candidate_missing_indicators"),
    )
    valid_mask = valid_mask.to(device=output.masked_logits.device)
    teacher_actions = batch["teacher_actions"].to(device=output.masked_logits.device, dtype=torch.long)
    return F.cross_entropy(output.masked_logits[valid_mask], teacher_actions[valid_mask])


def _teacher_imitation_metrics(network, batch: dict[str, Any], *, weight: float) -> dict[str, Any]:
    torch = _load_torch()
    valid_mask = batch["teacher_action_valid_mask"].to(dtype=torch.bool)
    valid_count = int(valid_mask.sum().item())
    ignored_count = int(valid_mask.numel() - valid_count)
    enabled = float(weight) > 0.0
    metrics = {
        "enabled": enabled,
        "weight": float(weight),
        "valid_teacher_label_count": valid_count,
        "ignored_teacher_label_count": ignored_count,
        "teacher_imitation_loss": 0.0,
        "teacher_action_accuracy": 0.0,
    }
    if valid_count == 0 or not enabled:
        return metrics

    from torch.nn import functional as F

    network.eval()
    with torch.no_grad():
        output = network(
            candidate_features=batch["candidate_features"],
            global_features=batch["global_features"],
            action_mask=batch["action_mask"],
            candidate_missing_indicators=batch.get("candidate_missing_indicators"),
        )
        valid_mask = valid_mask.to(device=output.masked_logits.device)
        teacher_actions = batch["teacher_actions"].to(device=output.masked_logits.device, dtype=torch.long)
        logits = output.masked_logits[valid_mask]
        labels = teacher_actions[valid_mask]
        loss = F.cross_entropy(logits, labels)
        predictions = torch.argmax(logits, dim=-1)
        accuracy = (predictions == labels).to(dtype=torch.float32).mean()
    metrics["teacher_imitation_loss"] = float(loss.detach().cpu())
    metrics["teacher_action_accuracy"] = float(accuracy.detach().cpu())
    return metrics


def _save_policy_checkpoint(
    path: str | Path,
    *,
    network,
    hidden_size: int,
    candidate_feature_names: tuple[str, ...],
    global_feature_names: tuple[str, ...],
    candidate_missing_indicator_names: tuple[str, ...],
    action_count: int,
    seed: int,
    sample_count: int,
    epoch_count: int,
    learning_rate: float,
    return_mode: str,
    discount_factor: float,
    training_source: dict[str, Any],
    teacher_imitation: dict[str, Any],
    teacher_margin_summary: dict[str, Any],
) -> None:
    torch = _load_torch()
    torch.save(
        {
            "state_dict": network.state_dict(),
            "hidden_size": hidden_size,
            "candidate_feature_names": tuple(candidate_feature_names),
            "global_feature_names": tuple(global_feature_names),
            "candidate_missing_indicator_names": tuple(candidate_missing_indicator_names),
            "metadata": {
                "format": "model-explorer-masked-policy",
                "version": 2,
                "format_version": "model-explorer-masked-policy/v2",
                "architecture": network.architecture_name,
                "architecture_config": dict(network.architecture_config),
                "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
                "candidate_feature_names": tuple(candidate_feature_names),
                "global_feature_names": tuple(global_feature_names),
                "candidate_missing_indicator_names": tuple(candidate_missing_indicator_names),
                "action_count": int(action_count),
                "seed": int(seed),
                "sample_count": int(sample_count),
                "epoch_count": int(epoch_count),
                "hidden_size": int(hidden_size),
                "learning_rate": float(learning_rate),
                "return_mode": str(return_mode),
                "discount_factor": float(discount_factor),
                "training_source": dict(training_source),
                "teacher_imitation": dict(teacher_imitation),
                "teacher_margin_summary": dict(teacher_margin_summary),
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
