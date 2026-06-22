from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..core.interfaces import GoalCandidate
from .canonical_reward import CanonicalRewardGuardProfile, compute_canonical_reward_components


@dataclass(frozen=True)
class RewardInfo:
    reward: float
    coverage_rate_delta: float
    path_cost: float
    risk: float
    failure_reason: str | None = None
    reward_components: dict[str, float] | None = None
    profile_id: str | None = None
    profile_version: str | None = None
    profile_hash: str | None = None


def compute_step_reward(
    selected_goal: GoalCandidate | None,
    observation_update: dict[str, Any],
    *,
    failure_reason: str | None = None,
    path_cost_weight: float = 0.10,
    path_cost_normalizer: float = 100.0,
    risk_weight: float = 0.20,
    failure_penalty: float = 1.0,
    path_cost_override: float | None = None,
    risk_override: float | None = None,
    canonical_profile: CanonicalRewardGuardProfile | None = None,
) -> RewardInfo:
    coverage_rate_delta = _numeric_mapping_value(observation_update, "coverage_rate_delta") or 0.0
    path_cost = (
        float(path_cost_override)
        if path_cost_override is not None
        else _numeric_experimental(selected_goal, "path_cost")
        if selected_goal is not None
        else 0.0
    )
    risk = (
        float(risk_override)
        if risk_override is not None
        else _numeric_experimental(selected_goal, "risk")
        if selected_goal is not None
        else 0.0
    )
    valuable_coverage = (
        _numeric_mapping_value(observation_update, "valuable_coverage")
        or _numeric_mapping_value(observation_update, "valuable_area_covered")
        or _numeric_mapping_value(observation_update, "value_coverage")
        or _numeric_experimental(selected_goal, "valuable_coverage")
        or _numeric_experimental(selected_goal, "valuable_area_covered")
    )
    information_gain = (
        _numeric_mapping_value(observation_update, "information_gain")
        or _numeric_mapping_value(observation_update, "information_value")
        or _numeric_experimental(selected_goal, "information_gain")
    )
    roi_coverage = (
        _numeric_mapping_value(observation_update, "roi_coverage")
        or _numeric_mapping_value(observation_update, "roi_weighted_coverage")
        or _numeric_mapping_value(observation_update, "roi_weighted_coverage_delta")
        or _numeric_experimental(selected_goal, "roi_coverage")
        or _numeric_experimental(selected_goal, "roi_weighted_coverage")
        or _numeric_experimental(selected_goal, "roi_weighted_coverage_delta")
    )
    soft_risk_exposure = (
        _numeric_mapping_value(observation_update, "soft_risk_exposure")
        or _numeric_mapping_value(observation_update, "path_risk_exposure")
        or _numeric_experimental(selected_goal, "soft_risk_exposure")
        or _numeric_experimental(selected_goal, "path_risk_exposure")
        or risk
    )
    path_allowed_by_risk = (
        _bool_mapping_value(observation_update, "path_allowed_by_risk")
        if "path_allowed_by_risk" in observation_update
        else _bool_experimental(selected_goal, "path_allowed_by_risk")
    )
    hard_risk_violation_count = (
        _numeric_mapping_value(observation_update, "hard_risk_violation_count")
        or _numeric_experimental(selected_goal, "hard_risk_violation_count")
        or (0.0 if path_allowed_by_risk is not False else 1.0)
    )

    if canonical_profile is not None:
        metrics = {
                "coverage_gain_rate": coverage_rate_delta,
                "valuable_coverage": valuable_coverage,
                "information_gain": information_gain,
                "path_cost_m": path_cost,
                "risk_proxy": risk,
                "failure": failure_reason is not None,
                "failure_reason": failure_reason,
        }
        if canonical_profile.profile_version == "v3":
            metrics.update(
                {
                    "roi_coverage": roi_coverage,
                    "soft_risk_exposure": soft_risk_exposure,
                    "path_allowed_by_risk": path_allowed_by_risk,
                    "hard_risk_violation_count": hard_risk_violation_count,
                }
            )
        component_result = compute_canonical_reward_components(metrics, canonical_profile)
        return RewardInfo(
            reward=float(component_result.reward),
            coverage_rate_delta=float(coverage_rate_delta),
            path_cost=float(path_cost),
            risk=float(risk),
            failure_reason=failure_reason,
            reward_components=dict(component_result.components),
            profile_id=component_result.profile_id,
            profile_version=component_result.profile_version,
            profile_hash=component_result.profile_hash,
        )

    normalized_path_cost = path_cost / max(path_cost_normalizer, 1.0)

    reward = coverage_rate_delta - path_cost_weight * normalized_path_cost - risk_weight * risk
    if failure_reason is not None:
        reward -= failure_penalty

    return RewardInfo(
        reward=float(reward),
        coverage_rate_delta=float(coverage_rate_delta),
        path_cost=float(path_cost),
        risk=float(risk),
        failure_reason=failure_reason,
        reward_components={},
    )


def _numeric_experimental(goal: GoalCandidate | None, field: str) -> float:
    if goal is None:
        return 0.0
    return _numeric_mapping_value(goal.experimental, field) or 0.0


def _numeric_mapping_value(mapping: dict[str, Any], field: str) -> float | None:
    value = mapping.get(field)
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _bool_experimental(goal: GoalCandidate | None, field: str) -> bool | None:
    if goal is None:
        return None
    return _bool_mapping_value(goal.experimental, field)


def _bool_mapping_value(mapping: dict[str, Any], field: str) -> bool | None:
    value = mapping.get(field)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
    return None
