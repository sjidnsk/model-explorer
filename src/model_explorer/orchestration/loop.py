from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from ..core.interfaces import ExplorerDecision, ExplorerStepResult, GoalCandidate, ModelExplorerContract
from ..decision.selector import select_goal
from ..io.scenario import Scenario


def run_exploration_loop(
    scenario_or_snapshots: Scenario | Iterable[ModelExplorerContract],
    *,
    observation_delta_threshold: float = 0.1,
) -> tuple[ExplorerStepResult, ...]:
    snapshots = (
        scenario_or_snapshots.snapshots
        if isinstance(scenario_or_snapshots, Scenario)
        else tuple(scenario_or_snapshots)
    )

    results: list[ExplorerStepResult] = []
    previous_decision: ExplorerDecision | None = None
    for step_index, contract in enumerate(snapshots):
        decision = select_goal(contract)
        replan_reasons = _replan_reasons(
            contract,
            decision,
            previous_decision,
            observation_delta_threshold=observation_delta_threshold,
        )
        results.append(
            ExplorerStepResult(
                step_index=step_index,
                decision=decision,
                observation_update=dict(contract.observation_update),
                replan_reasons=tuple(replan_reasons),
            )
        )
        previous_decision = decision

    return tuple(results)


def _replan_reasons(
    contract: ModelExplorerContract,
    decision: ExplorerDecision,
    previous_decision: ExplorerDecision | None,
    *,
    observation_delta_threshold: float,
) -> list[str]:
    reasons: list[str] = []

    if decision.status == "no_reachable_goal":
        reasons.append("no_reachable_goal")

    if _previous_goal_is_unreachable_or_missing(contract.top_goals, previous_decision):
        reasons.append("current_goal_unreachable")

    previous_goal = previous_decision.selected_goal if previous_decision is not None else None
    current_goal = decision.selected_goal
    if previous_goal is not None and current_goal is not None and current_goal.cell != previous_goal.cell:
        reasons.append("goal_changed")

    delta_c = _numeric_observation_delta(contract.observation_update)
    if delta_c is not None and delta_c >= observation_delta_threshold:
        reasons.append("observation_delta")

    return reasons


def _previous_goal_is_unreachable_or_missing(
    current_goals: tuple[GoalCandidate, ...],
    previous_decision: ExplorerDecision | None,
) -> bool:
    if previous_decision is None or previous_decision.selected_goal is None:
        return False

    previous_cell = previous_decision.selected_goal.cell
    for goal in current_goals:
        if goal.cell == previous_cell:
            return not goal.reachable
    return True


def _numeric_observation_delta(observation_update: dict[str, Any]) -> float | None:
    value = observation_update.get("delta_c")
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
