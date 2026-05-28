from __future__ import annotations

from collections.abc import Sequence
from math import isfinite
from typing import Any

from ..core.interfaces import ExplorerDecision, GoalCandidate, ModelExplorerContract
from ..policy.features import extract_policy_observation


_BENEFIT_FIELDS = ("information_gain", "confidence_gain", "value")
_COST_FIELDS = ("risk", "path_cost", "energy_cost")
_SCORE_WEIGHTS = {
    "coverage": 0.35,
    "information_gain": 0.20,
    "confidence_gain": 0.20,
    "value": 0.15,
    "risk": -0.15,
    "path_cost": -0.10,
    "energy_cost": -0.05,
}


def select_goal(contract: ModelExplorerContract, *, policy: Any | None = None) -> ExplorerDecision:
    reachable_goals = tuple(goal for goal in contract.top_goals if goal.reachable)
    ranked_goals = _rank_goals_with_policy(contract, policy) if policy is not None else None
    if ranked_goals is None:
        ranked_goals = _rank_goals(reachable_goals)

    if not ranked_goals:
        return ExplorerDecision(status="no_reachable_goal", selected_goal=None, ranked_goals=ranked_goals)

    return ExplorerDecision(status="selected", selected_goal=ranked_goals[0], ranked_goals=ranked_goals)


def _rank_goals_with_policy(contract: ModelExplorerContract, policy: Any) -> tuple[GoalCandidate, ...] | None:
    score_method = getattr(policy, "score", None)
    if score_method is None:
        return None

    observation = extract_policy_observation(contract)
    try:
        raw_scores = score_method(observation)
    except Exception:
        return None

    scores = _coerce_policy_scores(raw_scores, expected_count=len(contract.top_goals))
    if scores is None:
        return None

    scored_goals = tuple(
        (scores[index], goal)
        for index, goal in enumerate(contract.top_goals)
        if observation.action_mask[index]
    )
    return tuple(
        goal
        for _, goal in sorted(
            scored_goals,
            key=lambda item: (-item[0], -item[1].utility, item[1].cell[0], item[1].cell[1]),
        )
    )


def _rank_goals(goals: tuple[GoalCandidate, ...]) -> tuple[GoalCandidate, ...]:
    if not _has_coverage_signal(goals):
        return tuple(sorted(goals, key=lambda goal: (-goal.utility, goal.cell[0], goal.cell[1])))

    scored_goals = tuple((_policy_score(goal, goals), goal) for goal in goals)
    return tuple(
        goal
        for _, goal in sorted(
            scored_goals,
            key=lambda item: (-item[0], -item[1].utility, item[1].cell[0], item[1].cell[1]),
        )
    )


def _has_coverage_signal(goals: tuple[GoalCandidate, ...]) -> bool:
    return any(_coverage_value(goal) is not None for goal in goals)


def _policy_score(goal: GoalCandidate, goals: tuple[GoalCandidate, ...]) -> float:
    normalized = _normalized_feature_values(goals)
    goal_index = goals.index(goal)
    return sum(weight * values[goal_index] for field, weight in _SCORE_WEIGHTS.items() for values in [normalized[field]])


def _normalized_feature_values(goals: tuple[GoalCandidate, ...]) -> dict[str, tuple[float, ...]]:
    raw_values: dict[str, tuple[float, ...]] = {
        "coverage": tuple(_coverage_value(goal) or 0.0 for goal in goals),
    }

    for field in _BENEFIT_FIELDS:
        raw_values[field] = tuple(_numeric_experimental(goal, field) or 0.0 for goal in goals)

    for field in _COST_FIELDS:
        available_values = tuple(
            value for goal in goals for value in [_numeric_experimental(goal, field)] if value is not None
        )
        missing_default = max(available_values) if available_values else 0.0
        raw_values[field] = tuple(
            _numeric_experimental(goal, field) if _numeric_experimental(goal, field) is not None else missing_default
            for goal in goals
        )

    return {field: _min_max_normalize(values) for field, values in raw_values.items()}


def _coverage_value(goal: GoalCandidate) -> float | None:
    explicit_rate_delta = _numeric_experimental(goal, "expected_coverage_rate_delta")
    if explicit_rate_delta is not None:
        return explicit_rate_delta
    return _numeric_experimental(goal, "expected_new_coverage_area")


def _numeric_experimental(goal: GoalCandidate, field: str) -> float | None:
    value = goal.experimental.get(field)
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _min_max_normalize(values: tuple[float, ...]) -> tuple[float, ...]:
    if not values:
        return ()
    min_value = min(values)
    max_value = max(values)
    if max_value == min_value:
        return tuple(0.0 for _ in values)
    return tuple((value - min_value) / (max_value - min_value) for value in values)


def _coerce_policy_scores(raw_scores: Any, *, expected_count: int) -> tuple[float, ...] | None:
    if not isinstance(raw_scores, Sequence) or isinstance(raw_scores, (str, bytes)):
        return None
    if len(raw_scores) != expected_count:
        return None

    scores: list[float] = []
    for value in raw_scores:
        if isinstance(value, bool):
            return None
        try:
            score = float(value)
        except (TypeError, ValueError):
            return None
        if not isfinite(score):
            return None
        scores.append(score)
    return tuple(scores)
