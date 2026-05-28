from __future__ import annotations

from collections.abc import Iterable
from math import isfinite
from statistics import mean
from typing import Any

from .rollout import RolloutEpisode, RolloutTransition


def summarize_rollout_dataset(episodes: Iterable[RolloutEpisode]) -> dict[str, Any]:
    episode_tuple = tuple(episodes)
    transitions = tuple(transition for episode in episode_tuple for transition in episode.transitions)
    reachable_counts = tuple(_reachable_action_count(transition) for transition in transitions)
    trainable_transitions = tuple(transition for transition in transitions if _is_trainable_transition(transition))
    rewards = tuple(float(transition.reward) for transition in transitions if isfinite(float(transition.reward)))
    invalid_action_mask_count = sum(1 for transition in transitions if _has_invalid_action_mask(transition))
    empty_action_mask_count = sum(1 for transition in transitions if _reachable_action_count(transition) == 0)
    no_op_transition_count = sum(1 for transition in transitions if transition.action_index < 0)
    failure_transition_count = sum(1 for transition in transitions if transition.info.failure_reason is not None)
    risks = tuple(float(transition.info.risk) for transition in transitions if isfinite(float(transition.info.risk)))

    summary = {
        "episode_count": len(episode_tuple),
        "transition_count": len(transitions),
        "trainable_transition_count": len(trainable_transitions),
        "no_op_transition_count": no_op_transition_count,
        "failure_transition_count": failure_transition_count,
        "reachable_action_count_distribution": _distribution(reachable_counts),
        "invalid_action_mask_count": invalid_action_mask_count,
        "empty_action_mask_count": empty_action_mask_count,
        "reward": _numeric_summary(rewards),
        "failure_count": _aggregate_failure_count(episode_tuple),
        "replan_count": _aggregate_replan_count(episode_tuple),
        "coverage_delta_total": _aggregate_coverage_delta(episode_tuple),
        "total_path_cost": _aggregate_total_path_cost(episode_tuple),
        "average_risk": mean(risks) if risks else 0.0,
        "warnings": [],
        "errors": [],
    }
    warnings: list[str] = summary["warnings"]
    errors: list[str] = summary["errors"]
    if not transitions:
        warnings.append("empty_dataset")
    if not trainable_transitions:
        warnings.append("no_trainable_transitions")
        errors.append("no trainable transitions")
    if empty_action_mask_count:
        warnings.append("empty_action_mask")
    if invalid_action_mask_count:
        warnings.append("invalid_action_mask")
        errors.append("invalid action mask")
    if len(rewards) != len(transitions):
        warnings.append("non_finite_reward")
        errors.append("non-finite reward")
    return summary


def validate_rollout_dataset(episodes: Iterable[RolloutEpisode]) -> dict[str, Any]:
    summary = summarize_rollout_dataset(episodes)
    if summary["errors"]:
        raise ValueError("; ".join(summary["errors"]))
    return summary


def _is_trainable_transition(transition: RolloutTransition) -> bool:
    if transition.action_index < 0:
        return False
    if _has_invalid_action_mask(transition):
        return False
    return bool(transition.observation.action_mask[transition.action_index])


def _has_invalid_action_mask(transition: RolloutTransition) -> bool:
    observation = transition.observation
    mask = tuple(bool(value) for value in observation.action_mask)
    if len(mask) != len(observation.candidate_features):
        return True
    if len(mask) != len(observation.candidate_cells):
        return True
    if transition.action_index >= len(mask):
        return True
    if transition.action_index >= 0 and not mask[transition.action_index]:
        return True
    return False


def _reachable_action_count(transition: RolloutTransition) -> int:
    return sum(1 for value in transition.observation.action_mask if value)


def _distribution(values: tuple[int, ...]) -> dict[str, Any]:
    return {
        "counts": list(values),
        "min": min(values) if values else 0,
        "max": max(values) if values else 0,
        "mean": mean(values) if values else 0.0,
    }


def _numeric_summary(values: tuple[float, ...]) -> dict[str, float]:
    return {
        "min": min(values) if values else 0.0,
        "max": max(values) if values else 0.0,
        "mean": mean(values) if values else 0.0,
    }


def _aggregate_failure_count(episodes: tuple[RolloutEpisode, ...]) -> int:
    count = sum(int(episode.metrics.failure_count) for episode in episodes)
    if count:
        return count
    return sum(1 for episode in episodes for transition in episode.transitions if transition.info.failure_reason is not None)


def _aggregate_replan_count(episodes: tuple[RolloutEpisode, ...]) -> int:
    count = sum(int(episode.metrics.replan_count) for episode in episodes)
    if count:
        return count
    return sum(_max_transition_counter(episode.transitions, "replan_count") for episode in episodes)


def _aggregate_coverage_delta(episodes: tuple[RolloutEpisode, ...]) -> float:
    total = sum(float(episode.metrics.cumulative_coverage_rate_delta) for episode in episodes)
    if total:
        return total
    return sum(float(transition.info.coverage_rate_delta) for episode in episodes for transition in episode.transitions)


def _aggregate_total_path_cost(episodes: tuple[RolloutEpisode, ...]) -> float:
    total = sum(float(episode.metrics.total_path_cost) for episode in episodes)
    if total:
        return total
    return sum(float(transition.info.path_cost) for episode in episodes for transition in episode.transitions)


def _max_transition_counter(transitions: tuple[RolloutTransition, ...], field: str) -> int:
    if not transitions:
        return 0
    return max(int(getattr(transition.info, field)) for transition in transitions)
