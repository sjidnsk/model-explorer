from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..core.interfaces import GoalCandidate


@dataclass(frozen=True)
class RewardInfo:
    reward: float
    coverage_rate_delta: float
    path_cost: float
    risk: float
    failure_reason: str | None = None


def compute_step_reward(
    selected_goal: GoalCandidate | None,
    observation_update: dict[str, Any],
    *,
    failure_reason: str | None = None,
    path_cost_weight: float = 0.10,
    path_cost_normalizer: float = 100.0,
    risk_weight: float = 0.20,
    failure_penalty: float = 1.0,
) -> RewardInfo:
    coverage_rate_delta = _numeric_mapping_value(observation_update, "coverage_rate_delta") or 0.0
    path_cost = _numeric_experimental(selected_goal, "path_cost") if selected_goal is not None else 0.0
    risk = _numeric_experimental(selected_goal, "risk") if selected_goal is not None else 0.0
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
