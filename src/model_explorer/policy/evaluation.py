from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from ..core.interfaces import GoalCandidate, ModelExplorerContract
from ..decision.selector import select_goal
from ..io.scenario import Scenario
from .reward import compute_step_reward


def evaluate_policy_baselines(
    scenario_or_snapshots: Scenario | Iterable[ModelExplorerContract],
    *,
    torch_policy=None,
) -> dict[str, dict[str, Any]]:
    snapshots = (
        scenario_or_snapshots.snapshots
        if isinstance(scenario_or_snapshots, Scenario)
        else tuple(scenario_or_snapshots)
    )

    report = {
        "utility": _evaluate_strategy(snapshots, _select_utility_goal),
        "coverage_heuristic": _evaluate_strategy(
            snapshots,
            lambda contract: select_goal(contract).selected_goal,
        ),
    }
    if torch_policy is not None:
        report["torch_policy"] = _evaluate_strategy(
            snapshots,
            lambda contract: select_goal(contract, policy=torch_policy).selected_goal,
        )
    return report


def _evaluate_strategy(
    snapshots: tuple[ModelExplorerContract, ...],
    selector,
) -> dict[str, Any]:
    selected_cells: list[list[int] | None] = []
    cumulative_coverage_rate_delta = 0.0
    total_path_cost = 0.0
    total_risk = 0.0
    failure_count = 0
    final_coverage_rate = 0.0
    selected_count = 0

    for contract in snapshots:
        selected_goal = selector(contract)
        if selected_goal is None:
            failure_count += 1
            selected_cells.append(None)
            continue

        reward_info = compute_step_reward(selected_goal, contract.observation_update)
        selected_cells.append([selected_goal.cell[0], selected_goal.cell[1]])
        cumulative_coverage_rate_delta += reward_info.coverage_rate_delta
        total_path_cost += reward_info.path_cost
        total_risk += reward_info.risk
        final_coverage_rate = _coverage_rate(contract.observation_update, fallback=final_coverage_rate)
        selected_count += 1

    return {
        "selected_cells": selected_cells,
        "final_coverage_rate": final_coverage_rate,
        "cumulative_coverage_rate_delta": cumulative_coverage_rate_delta,
        "total_path_cost": total_path_cost,
        "average_risk": total_risk / selected_count if selected_count else 0.0,
        "failure_count": failure_count,
        "replan_count": 0,
    }


def _select_utility_goal(contract: ModelExplorerContract) -> GoalCandidate | None:
    reachable_goals = tuple(goal for goal in contract.top_goals if goal.reachable)
    if not reachable_goals:
        return None
    return sorted(reachable_goals, key=lambda goal: (-goal.utility, goal.cell[0], goal.cell[1]))[0]


def _coverage_rate(observation_update: dict[str, Any], *, fallback: float) -> float:
    value = observation_update.get("coverage_rate")
    if isinstance(value, bool) or value is None:
        return fallback
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback
