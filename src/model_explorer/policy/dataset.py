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
    min_unreachable_candidate_count: int | None = None
    min_mask_stress_sample_count: int | None = None
    min_roi_group_count: int | None = None
    min_reward_std: float | None = None
    min_feedback_aware_sample_count: int | None = None
    min_teacher_high_margin_sample_count: int | None = None
    max_missing_teacher_signal_rate: float | None = None
    max_low_margin_only_dataset_rate: float | None = None

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
                "min_unreachable_candidate_count": self.min_unreachable_candidate_count,
                "min_mask_stress_sample_count": self.min_mask_stress_sample_count,
                "min_roi_group_count": self.min_roi_group_count,
                "min_reward_std": self.min_reward_std,
                "min_feedback_aware_sample_count": self.min_feedback_aware_sample_count,
                "min_teacher_high_margin_sample_count": self.min_teacher_high_margin_sample_count,
                "max_missing_teacher_signal_rate": self.max_missing_teacher_signal_rate,
                "max_low_margin_only_dataset_rate": self.max_low_margin_only_dataset_rate,
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
    provenance_summary = _provenance_summary(transitions)
    selection_source_summary = _selection_source_summary(transitions)
    teacher_margin_summary = _teacher_margin_summary(transitions)

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
        "observation_slot_count": candidate_quality["observation_slot_count"],
        "reachable_candidate_count": candidate_quality["reachable_candidate_count"],
        "unreachable_candidate_count": candidate_quality["unreachable_candidate_count"],
        "padding_candidate_count": candidate_quality["padding_candidate_count"],
        "padding_candidate_rate": candidate_quality["padding_candidate_rate"],
        "action_mask_valid_mean": candidate_quality["action_mask_valid_mean"],
        "unreachable_candidate_rate": candidate_quality["unreachable_candidate_rate"],
        "experimental_feature_count": candidate_quality["experimental_feature_count"],
        "missing_experimental_feature_count": candidate_quality["missing_experimental_feature_count"],
        "missing_experimental_feature_candidate_count": candidate_quality[
            "missing_experimental_feature_candidate_count"
        ],
        "missing_experimental_feature_rate": candidate_quality["missing_experimental_feature_rate"],
        "mask_stress_sample_count": candidate_quality["mask_stress_sample_count"],
        "mask_stress_sample_rate": candidate_quality["mask_stress_sample_rate"],
        "failure_count": _aggregate_failure_count(episode_tuple),
        "replan_count": _aggregate_replan_count(episode_tuple),
        "coverage_delta_total": _aggregate_coverage_delta(episode_tuple),
        "total_path_cost": _aggregate_total_path_cost(episode_tuple),
        "average_risk": mean(risks) if risks else 0.0,
        "warnings": [],
        "errors": [],
    }
    summary.update(provenance_summary)
    summary.update(selection_source_summary)
    summary.update(teacher_margin_summary)
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


def summarize_teacher_quality_gates(
    dataset_summary: dict[str, Any],
    gates: dict[str, Any] | DatasetValidationGates | None = None,
) -> dict[str, Any]:
    if gates is None:
        return {"status": "not_configured", "configured": {}, "violations": []}
    gate_config = _coerce_validation_gates(gates)
    violations = _teacher_quality_gate_violations(dataset_summary, gate_config)
    return {
        "status": "failed" if violations else "passed",
        "configured": {
            key: value
            for key, value in gate_config.to_dict().items()
            if key in _TEACHER_QUALITY_GATE_NAMES
        },
        "violations": violations,
    }


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
    observation_slot_count = 0
    reachable_count = 0
    padding_count = 0
    missing_feature_count = 0
    missing_feature_candidate_count = 0
    experimental_feature_count = 0
    mask_stress_sample_count = 0
    for transition in transitions:
        observation = transition.observation
        missing_rows = observation.candidate_missing_feature_names
        if len(missing_rows) != len(observation.candidate_cells):
            missing_rows = tuple(() for _ in observation.candidate_cells)
        transition_has_unreachable = False
        transition_has_padding = False
        transition_has_missing = False
        for index, cell in enumerate(observation.candidate_cells):
            observation_slot_count += 1
            if cell is None:
                padding_count += 1
                transition_has_padding = True
                continue
            candidate_count += 1
            if index < len(observation.action_mask) and bool(observation.action_mask[index]):
                reachable_count += 1
            else:
                transition_has_unreachable = True
            if index < len(missing_rows):
                missing_count = len(missing_rows[index])
                missing_feature_count += missing_count
                if missing_count:
                    missing_feature_candidate_count += 1
                    transition_has_missing = True
            experimental_feature_count += len(missing_rows[index]) + _present_experimental_feature_count(
                observation,
                index,
                missing_rows[index] if index < len(missing_rows) else (),
            )
        provenance = _transition_provenance(transition)
        if (
            transition_has_unreachable
            or transition_has_padding
            or transition_has_missing
            or bool(provenance.get("mask_stress_augmented"))
        ):
            mask_stress_sample_count += 1
    unreachable_count = candidate_count - reachable_count
    return {
        "candidate_count": candidate_count,
        "observation_slot_count": observation_slot_count,
        "reachable_candidate_count": reachable_count,
        "unreachable_candidate_count": unreachable_count,
        "padding_candidate_count": padding_count,
        "padding_candidate_rate": padding_count / observation_slot_count if observation_slot_count else 0.0,
        "action_mask_valid_mean": reachable_count / candidate_count if candidate_count else 0.0,
        "unreachable_candidate_rate": unreachable_count / candidate_count if candidate_count else 0.0,
        "experimental_feature_count": experimental_feature_count,
        "missing_experimental_feature_count": missing_feature_count,
        "missing_experimental_feature_candidate_count": missing_feature_candidate_count,
        "missing_experimental_feature_rate": (
            missing_feature_count / experimental_feature_count if experimental_feature_count else 0.0
        ),
        "mask_stress_sample_count": mask_stress_sample_count,
        "mask_stress_sample_rate": mask_stress_sample_count / len(transitions) if transitions else 0.0,
    }


def _provenance_summary(transitions: tuple[RolloutTransition, ...]) -> dict[str, Any]:
    records = tuple(_transition_provenance(transition) for transition in transitions)
    records = tuple(record for record in records if record)
    data_classes = _unique_strings(record.get("data_class") for record in records)
    dataset_ids = _unique_strings(record.get("dataset_id") for record in records)
    regions = _unique_strings(record.get("region") for record in records)
    generator_versions = _unique_strings(record.get("generator_version") for record in records)
    roi_names = _unique_strings(record.get("roi_name") for record in records)
    splits = _unique_strings(record.get("split") for record in records)
    mask_stress_labels = _unique_strings(record.get("mask_stress_label") for record in records)
    mask_stress_profiles = _unique_strings(record.get("mask_stress_profile") for record in records)
    summary: dict[str, Any] = {
        "data_classes": list(data_classes),
        "dataset_ids": list(dataset_ids),
        "regions": list(regions),
        "generator_versions": list(generator_versions),
        "roi_names": list(roi_names),
        "roi_count": len(roi_names),
        "splits": list(splits),
        "split_counts": _counts_by_key(records, "split"),
        "mask_stress_augmented": any(bool(record.get("mask_stress_augmented")) for record in records),
        "mask_stress_labels": list(mask_stress_labels),
        "mask_stress_profiles": list(mask_stress_profiles),
    }
    if len(data_classes) == 1:
        summary["data_class"] = data_classes[0]
    if len(dataset_ids) == 1:
        summary["dataset_id"] = dataset_ids[0]
    if len(regions) == 1:
        summary["region"] = regions[0]
    if len(generator_versions) == 1:
        summary["generator_version"] = generator_versions[0]
    if len(mask_stress_labels) == 1:
        summary["mask_stress_label"] = mask_stress_labels[0]
    if len(mask_stress_profiles) == 1:
        summary["mask_stress_profile"] = mask_stress_profiles[0]
    return summary


def _selection_source_summary(transitions: tuple[RolloutTransition, ...]) -> dict[str, Any]:
    strategies = tuple(_extra_string(transition, "selection_strategy") or "unknown" for transition in transitions)
    requested = tuple(
        _extra_string(transition, "requested_selection_strategy") or "unknown"
        for transition in transitions
    )
    strategy_counts = _counts(strategies)
    total = sum(strategy_counts.values())
    return {
        "selection_strategy_counts": strategy_counts,
        "selection_strategy_fractions": {
            strategy: count / total if total else 0.0
            for strategy, count in strategy_counts.items()
        },
        "requested_selection_strategy_counts": _counts(requested),
        "primary_selection_strategy": _primary_count_key(strategy_counts),
    }


_TEACHER_LOW_MARGIN_THRESHOLD = 0.05
_TEACHER_HIGH_MARGIN_THRESHOLD = 0.20


def _teacher_margin_summary(transitions: tuple[RolloutTransition, ...]) -> dict[str, Any]:
    margins = tuple(
        value
        for transition in transitions
        for value in [_extra_float(transition, "teacher_score_margin")]
        if value is not None
    )
    bucket_counts = {"high": 0, "low": 0, "medium": 0, "missing": len(transitions) - len(margins)}
    for margin in margins:
        bucket_counts[_teacher_margin_bucket(margin)] += 1
    low_count = bucket_counts["low"]
    high_count = bucket_counts["high"]
    medium_count = bucket_counts["medium"]
    feedback_aware_count = sum(
        1 for transition in transitions if _extra_string(transition, "selection_strategy") == "feedback_aware"
    )
    missing_teacher_signal_count = bucket_counts["missing"]
    teacher_margin_sample_count = len(margins)
    low_margin_only_dataset_rate = (
        1.0
        if teacher_margin_sample_count > 0 and low_count == teacher_margin_sample_count and high_count == 0 and medium_count == 0
        else 0.0
    )
    return {
        "feedback_aware_transition_count": feedback_aware_count,
        "feedback_aware_transition_fraction": feedback_aware_count / len(transitions) if transitions else 0.0,
        "teacher_margin_sample_count": teacher_margin_sample_count,
        "teacher_score_margin": _numeric_summary(margins),
        "teacher_low_margin_threshold": _TEACHER_LOW_MARGIN_THRESHOLD,
        "teacher_high_margin_threshold": _TEACHER_HIGH_MARGIN_THRESHOLD,
        "teacher_low_margin_sample_count": low_count,
        "teacher_medium_margin_sample_count": medium_count,
        "teacher_high_margin_sample_count": high_count,
        "teacher_low_margin_sample_rate": low_count / teacher_margin_sample_count if teacher_margin_sample_count else 0.0,
        "missing_teacher_signal_count": missing_teacher_signal_count,
        "missing_teacher_signal_rate": missing_teacher_signal_count / len(transitions) if transitions else 0.0,
        "low_margin_only_dataset_rate": low_margin_only_dataset_rate,
        "teacher_margin_bucket_counts": bucket_counts,
    }


def _teacher_margin_bucket(margin: float | None) -> str:
    if margin is None:
        return "missing"
    if margin <= _TEACHER_LOW_MARGIN_THRESHOLD:
        return "low"
    if margin <= _TEACHER_HIGH_MARGIN_THRESHOLD:
        return "medium"
    return "high"


def _extra_string(transition: RolloutTransition, key: str) -> str | None:
    value = transition.info.extra.get(key)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _extra_float(transition: RolloutTransition, key: str) -> float | None:
    value = transition.info.extra.get(key)
    if isinstance(value, bool) or value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if isfinite(numeric) else None


def _counts(values: tuple[str, ...]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _primary_count_key(counts: dict[str, int]) -> str | None:
    if not counts:
        return None
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def _transition_provenance(transition: RolloutTransition) -> dict[str, Any]:
    extra = transition.info.extra
    provenance = extra.get("provenance")
    if isinstance(provenance, dict):
        return dict(provenance)
    return {
        key: extra[key]
        for key in ("data_class", "dataset_id", "region", "generator_version", "roi_name", "split")
        if key in extra
    }


def _unique_strings(values) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value is None:
            continue
        text = str(value)
        if text in seen:
            continue
        seen.add(text)
        result.append(text)
    return tuple(result)


def _counts_by_key(records: tuple[dict[str, Any], ...], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        value = record.get(key)
        if value is None:
            continue
        text = str(value)
        counts[text] = counts.get(text, 0) + 1
    return counts


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
        min_unreachable_candidate_count=_optional_int(value, "min_unreachable_candidate_count"),
        min_mask_stress_sample_count=_optional_int(value, "min_mask_stress_sample_count"),
        min_roi_group_count=_optional_int(value, "min_roi_group_count"),
        min_reward_std=_optional_float(value, "min_reward_std"),
        min_feedback_aware_sample_count=_optional_int(value, "min_feedback_aware_sample_count"),
        min_teacher_high_margin_sample_count=_optional_int(value, "min_teacher_high_margin_sample_count"),
        max_missing_teacher_signal_rate=_optional_float(value, "max_missing_teacher_signal_rate"),
        max_low_margin_only_dataset_rate=_optional_float(value, "max_low_margin_only_dataset_rate"),
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
    _append_min_violation(
        violations,
        "min_unreachable_candidate_count",
        summary["unreachable_candidate_count"],
        gates.min_unreachable_candidate_count,
    )
    _append_min_violation(
        violations,
        "min_mask_stress_sample_count",
        summary["mask_stress_sample_count"],
        gates.min_mask_stress_sample_count,
    )
    _append_min_violation(violations, "min_roi_group_count", summary["roi_count"], gates.min_roi_group_count)
    _append_min_violation(violations, "min_reward_std", summary["reward"]["std"], gates.min_reward_std)
    violations.extend(_teacher_quality_gate_violations(summary, gates))
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


_TEACHER_QUALITY_GATE_NAMES = {
    "min_feedback_aware_sample_count",
    "min_teacher_high_margin_sample_count",
    "max_missing_teacher_signal_rate",
    "max_low_margin_only_dataset_rate",
}


def _teacher_quality_gate_violations(
    summary: dict[str, Any],
    gates: DatasetValidationGates,
) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    _append_min_violation(
        violations,
        "min_feedback_aware_sample_count",
        int(summary.get("feedback_aware_transition_count", 0)),
        gates.min_feedback_aware_sample_count,
    )
    _append_min_violation(
        violations,
        "min_teacher_high_margin_sample_count",
        int(summary.get("teacher_high_margin_sample_count", 0)),
        gates.min_teacher_high_margin_sample_count,
    )
    _append_max_violation(
        violations,
        "max_missing_teacher_signal_rate",
        float(summary.get("missing_teacher_signal_rate", 0.0)),
        gates.max_missing_teacher_signal_rate,
    )
    _append_max_violation(
        violations,
        "max_low_margin_only_dataset_rate",
        float(summary.get("low_margin_only_dataset_rate", 0.0)),
        gates.max_low_margin_only_dataset_rate,
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
