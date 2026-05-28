from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from math import isfinite
from statistics import mean
from typing import Any

from .rollout import RolloutEpisode, RolloutTransition


@dataclass(frozen=True)
class DatasetValidationGates:
    min_episode_count: int | None = None
    min_transition_count: int | None = None
    min_trainable_transition_count: int | None = None
    max_empty_action_mask_count: int | None = None
    max_invalid_action_mask_count: int | None = None
    max_failure_rate: float | None = None
    require_finite_reward: bool = False
    max_missing_experimental_feature_rate: float | None = None
    min_action_mask_valid_mean: float | None = None
    max_unreachable_candidate_rate: float | None = None
    min_reward_std: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            key: value
            for key, value in {
                "min_episode_count": self.min_episode_count,
                "min_transition_count": self.min_transition_count,
                "min_trainable_transition_count": self.min_trainable_transition_count,
                "max_empty_action_mask_count": self.max_empty_action_mask_count,
                "max_invalid_action_mask_count": self.max_invalid_action_mask_count,
                "max_failure_rate": self.max_failure_rate,
                "require_finite_reward": self.require_finite_reward,
                "max_missing_experimental_feature_rate": self.max_missing_experimental_feature_rate,
                "min_action_mask_valid_mean": self.min_action_mask_valid_mean,
                "max_unreachable_candidate_rate": self.max_unreachable_candidate_rate,
                "min_reward_std": self.min_reward_std,
            }.items()
            if value is not None and value is not False
        }


def summarize_rollout_dataset(episodes: Iterable[RolloutEpisode]) -> dict[str, Any]:
    episode_tuple = tuple(episodes)
    transitions = tuple(transition for episode in episode_tuple for transition in episode.transitions)
    reachable_counts = tuple(_reachable_action_count(transition) for transition in transitions)
    trainable_transitions = tuple(transition for transition in transitions if _is_trainable_transition(transition))
    rewards = tuple(float(transition.reward) for transition in transitions if isfinite(float(transition.reward)))
    non_finite_reward_count = len(transitions) - len(rewards)
    candidate_quality = _candidate_quality_summary(transitions)
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
        "failure_rate": failure_transition_count / len(transitions) if transitions else 0.0,
        "reachable_action_count_distribution": _distribution(reachable_counts),
        "invalid_action_mask_count": invalid_action_mask_count,
        "empty_action_mask_count": empty_action_mask_count,
        "finite_reward_count": len(rewards),
        "non_finite_reward_count": non_finite_reward_count,
        "reward": _numeric_summary(rewards),
        "candidate_count": candidate_quality["candidate_count"],
        "reachable_candidate_count": candidate_quality["reachable_candidate_count"],
        "unreachable_candidate_count": candidate_quality["unreachable_candidate_count"],
        "action_mask_valid_mean": candidate_quality["action_mask_valid_mean"],
        "unreachable_candidate_rate": candidate_quality["unreachable_candidate_rate"],
        "experimental_feature_count": candidate_quality["experimental_feature_count"],
        "missing_experimental_feature_count": candidate_quality["missing_experimental_feature_count"],
        "missing_experimental_feature_rate": candidate_quality["missing_experimental_feature_rate"],
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


def validate_rollout_dataset(
    episodes: Iterable[RolloutEpisode],
    *,
    gates: dict[str, Any] | DatasetValidationGates | None = None,
) -> dict[str, Any]:
    summary = summarize_rollout_dataset(episodes)
    gate_config = _coerce_validation_gates(gates)
    gate_violations = _validation_gate_violations(summary, gate_config)
    summary["validation_gates"] = {
        "status": "failed" if summary["errors"] or gate_violations else "passed",
        "configured": gate_config.to_dict(),
        "violations": gate_violations,
    }
    if summary["errors"] or gate_violations:
        messages = list(summary["errors"])
        messages.extend(str(violation["message"]) for violation in gate_violations)
        raise ValueError("; ".join(messages))
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
    average = mean(values) if values else 0.0
    return {
        "min": min(values) if values else 0.0,
        "max": max(values) if values else 0.0,
        "mean": average,
        "std": _population_std(values, average=average),
    }


def _population_std(values: tuple[float, ...], *, average: float | None = None) -> float:
    if not values:
        return 0.0
    center = mean(values) if average is None else average
    variance = sum((value - center) ** 2 for value in values) / len(values)
    return variance ** 0.5


def _candidate_quality_summary(transitions: tuple[RolloutTransition, ...]) -> dict[str, Any]:
    candidate_count = 0
    reachable_count = 0
    missing_feature_count = 0
    experimental_feature_count = 0
    for transition in transitions:
        observation = transition.observation
        missing_rows = observation.candidate_missing_feature_names
        if len(missing_rows) != len(observation.candidate_cells):
            missing_rows = tuple(() for _ in observation.candidate_cells)
        for index, cell in enumerate(observation.candidate_cells):
            if cell is None:
                continue
            candidate_count += 1
            if index < len(observation.action_mask) and bool(observation.action_mask[index]):
                reachable_count += 1
            if index < len(missing_rows):
                missing_feature_count += len(missing_rows[index])
            experimental_feature_count += len(missing_rows[index]) + _present_experimental_feature_count(
                observation,
                index,
                missing_rows[index] if index < len(missing_rows) else (),
            )
    unreachable_count = candidate_count - reachable_count
    return {
        "candidate_count": candidate_count,
        "reachable_candidate_count": reachable_count,
        "unreachable_candidate_count": unreachable_count,
        "action_mask_valid_mean": reachable_count / candidate_count if candidate_count else 0.0,
        "unreachable_candidate_rate": unreachable_count / candidate_count if candidate_count else 0.0,
        "experimental_feature_count": experimental_feature_count,
        "missing_experimental_feature_count": missing_feature_count,
        "missing_experimental_feature_rate": (
            missing_feature_count / experimental_feature_count if experimental_feature_count else 0.0
        ),
    }


def _present_experimental_feature_count(
    observation,
    index: int,
    missing_feature_names: tuple[str, ...],
) -> int:
    experimental_names = (
        "expected_coverage_rate_delta",
        "expected_new_coverage_area",
        "information_gain",
        "confidence_gain",
        "value",
        "risk",
        "path_cost",
        "energy_cost",
    )
    if index >= len(observation.candidate_features):
        return 0
    feature_names = observation.candidate_feature_names
    available = 0
    missing = set(missing_feature_names)
    for name in experimental_names:
        if name in feature_names and name not in missing:
            available += 1
    return available


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


def _coerce_validation_gates(value: dict[str, Any] | DatasetValidationGates | None) -> DatasetValidationGates:
    if value is None:
        return DatasetValidationGates()
    if isinstance(value, DatasetValidationGates):
        return value
    if not isinstance(value, dict):
        raise ValueError("dataset validation gates must be an object")

    allowed = set(DatasetValidationGates.__dataclass_fields__)
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"unknown dataset validation gate: {unknown[0]}")

    return DatasetValidationGates(
        min_episode_count=_optional_int(value, "min_episode_count"),
        min_transition_count=_optional_int(value, "min_transition_count"),
        min_trainable_transition_count=_optional_int(value, "min_trainable_transition_count"),
        max_empty_action_mask_count=_optional_int(value, "max_empty_action_mask_count"),
        max_invalid_action_mask_count=_optional_int(value, "max_invalid_action_mask_count"),
        max_failure_rate=_optional_float(value, "max_failure_rate"),
        require_finite_reward=bool(value.get("require_finite_reward", False)),
        max_missing_experimental_feature_rate=_optional_float(value, "max_missing_experimental_feature_rate"),
        min_action_mask_valid_mean=_optional_float(value, "min_action_mask_valid_mean"),
        max_unreachable_candidate_rate=_optional_float(value, "max_unreachable_candidate_rate"),
        min_reward_std=_optional_float(value, "min_reward_std"),
    )


def _optional_int(value: dict[str, Any], key: str) -> int | None:
    return None if value.get(key) is None else int(value[key])


def _optional_float(value: dict[str, Any], key: str) -> float | None:
    return None if value.get(key) is None else float(value[key])


def _validation_gate_violations(
    summary: dict[str, Any],
    gates: DatasetValidationGates,
) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    _append_min_violation(violations, "min_episode_count", summary["episode_count"], gates.min_episode_count)
    _append_min_violation(violations, "min_transition_count", summary["transition_count"], gates.min_transition_count)
    _append_min_violation(
        violations,
        "min_trainable_transition_count",
        summary["trainable_transition_count"],
        gates.min_trainable_transition_count,
    )
    _append_max_violation(
        violations,
        "max_empty_action_mask_count",
        summary["empty_action_mask_count"],
        gates.max_empty_action_mask_count,
    )
    _append_max_violation(
        violations,
        "max_invalid_action_mask_count",
        summary["invalid_action_mask_count"],
        gates.max_invalid_action_mask_count,
    )
    _append_max_violation(violations, "max_failure_rate", summary["failure_rate"], gates.max_failure_rate)
    _append_max_violation(
        violations,
        "max_missing_experimental_feature_rate",
        summary["missing_experimental_feature_rate"],
        gates.max_missing_experimental_feature_rate,
    )
    _append_min_violation(
        violations,
        "min_action_mask_valid_mean",
        summary["action_mask_valid_mean"],
        gates.min_action_mask_valid_mean,
    )
    _append_max_violation(
        violations,
        "max_unreachable_candidate_rate",
        summary["unreachable_candidate_rate"],
        gates.max_unreachable_candidate_rate,
    )
    _append_min_violation(violations, "min_reward_std", summary["reward"]["std"], gates.min_reward_std)
    if gates.require_finite_reward and summary["non_finite_reward_count"] > 0:
        violations.append(
            {
                "gate": "require_finite_reward",
                "expected": True,
                "actual": False,
                "message": (
                    "require_finite_reward expected all rewards finite, "
                    f"actual non_finite_reward_count {summary['non_finite_reward_count']}"
                ),
            }
        )
    return violations


def _append_min_violation(
    violations: list[dict[str, Any]],
    gate: str,
    actual: int | float,
    expected: int | float | None,
) -> None:
    if expected is None or actual >= expected:
        return
    violations.append(
        {
            "gate": gate,
            "expected": expected,
            "actual": actual,
            "message": f"{gate} expected >= {expected}, actual {actual}",
        }
    )


def _append_max_violation(
    violations: list[dict[str, Any]],
    gate: str,
    actual: int | float,
    expected: int | float | None,
) -> None:
    if expected is None or actual <= expected:
        return
    violations.append(
        {
            "gate": gate,
            "expected": expected,
            "actual": actual,
            "message": f"{gate} expected <= {expected}, actual {actual}",
        }
    )
