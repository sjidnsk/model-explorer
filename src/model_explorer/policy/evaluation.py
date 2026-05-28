from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from ..core.interfaces import GoalCandidate, ModelExplorerContract
from ..decision.selector import select_goal
from ..io.scenario import Scenario
from .planning import PathPlanRequest, PathPlanningAdapter
from .reward import compute_step_reward


def evaluate_policy_baselines(
    scenario_or_snapshots: Scenario | Iterable[ModelExplorerContract],
    *,
    torch_policy=None,
    planning_adapter: PathPlanningAdapter | None = None,
) -> dict[str, dict[str, Any]]:
    snapshots = (
        scenario_or_snapshots.snapshots
        if isinstance(scenario_or_snapshots, Scenario)
        else tuple(scenario_or_snapshots)
    )

    report = {
        "utility": _evaluate_strategy(snapshots, _select_utility_goal, planning_adapter=planning_adapter),
        "coverage_heuristic": _evaluate_strategy(
            snapshots,
            lambda contract: select_goal(contract).selected_goal,
            planning_adapter=planning_adapter,
        ),
    }
    if torch_policy is not None:
        report["torch_policy"] = _evaluate_strategy(
            snapshots,
            lambda contract: select_goal(contract, policy=torch_policy).selected_goal,
            planning_adapter=planning_adapter,
        )
    return report


def evaluate_policy_baseline_scenarios(
    scenarios_or_snapshots: Iterable[Scenario | Iterable[ModelExplorerContract]],
    *,
    torch_policy=None,
    planning_adapter: PathPlanningAdapter | None = None,
) -> dict[str, dict[str, Any]]:
    reports = [
        evaluate_policy_baselines(
            scenario_or_snapshots,
            torch_policy=torch_policy,
            planning_adapter=planning_adapter,
        )
        for scenario_or_snapshots in scenarios_or_snapshots
    ]
    if not reports:
        return {}

    strategy_names = tuple(reports[0].keys())
    aggregate: dict[str, dict[str, Any]] = {}
    for strategy_name in strategy_names:
        strategy_reports = tuple(report[strategy_name] for report in reports if strategy_name in report)
        episode_count = len(strategy_reports)
        final_coverage_sum = sum(float(report["final_coverage_rate"]) for report in strategy_reports)
        aggregate[strategy_name] = {
            "episode_count": episode_count,
            "average_final_coverage_rate": final_coverage_sum / episode_count if episode_count else 0.0,
            "final_coverage_rate": final_coverage_sum / episode_count if episode_count else 0.0,
            "cumulative_coverage_rate_delta": sum(
                float(report["cumulative_coverage_rate_delta"]) for report in strategy_reports
            ),
            "total_path_cost": sum(float(report["total_path_cost"]) for report in strategy_reports),
            "average_risk": (
                sum(float(report["average_risk"]) for report in strategy_reports) / episode_count
                if episode_count
                else 0.0
            ),
            "failure_count": sum(int(report["failure_count"]) for report in strategy_reports),
            "replan_count": sum(int(report["replan_count"]) for report in strategy_reports),
            "value_coverage": sum(float(report["value_coverage"]) for report in strategy_reports),
        }
    return aggregate


def _evaluate_strategy(
    snapshots: tuple[ModelExplorerContract, ...],
    selector,
    *,
    planning_adapter: PathPlanningAdapter | None,
) -> dict[str, Any]:
    selected_cells: list[list[int] | None] = []
    cumulative_coverage_rate_delta = 0.0
    total_path_cost = 0.0
    total_risk = 0.0
    failure_count = 0
    final_coverage_rate = 0.0
    selected_count = 0
    replan_count = 0
    value_coverage = 0.0
    previous_goal: GoalCandidate | None = None
    current_cell = (0, 0)

    for step_index, contract in enumerate(snapshots):
        selected_goal = selector(contract)
        if selected_goal is None:
            failure_count += 1
            replan_count += 1
            selected_cells.append(None)
            previous_goal = None
            continue

        planning_result = None
        action_index = _selected_action_index(contract, selected_goal)
        if planning_adapter is not None:
            planning_result = planning_adapter.plan(
                PathPlanRequest(
                    contract=contract,
                    step_index=step_index,
                    action_index=action_index,
                    selected_goal=selected_goal,
                    current_cell=current_cell,
                )
            )
            if not planning_result.feasible:
                failure_count += 1

        reward_info = compute_step_reward(
            selected_goal,
            contract.observation_update,
            path_cost_override=None if planning_result is None else planning_result.path_cost,
            risk_override=None if planning_result is None else planning_result.risk,
        )
        selected_cells.append([selected_goal.cell[0], selected_goal.cell[1]])
        cumulative_coverage_rate_delta += reward_info.coverage_rate_delta
        total_path_cost += reward_info.path_cost
        total_risk += reward_info.risk
        final_coverage_rate = _coverage_rate(contract.observation_update, fallback=final_coverage_rate)
        value_coverage += _value_coverage(contract.observation_update)
        if _should_count_replan(contract, selected_goal, previous_goal) or (
            planning_result is not None and planning_result.replan_required
        ):
            replan_count += 1
        selected_count += 1
        if planning_result is None or planning_result.feasible:
            current_cell = selected_goal.cell
        previous_goal = selected_goal

    return {
        "selected_cells": selected_cells,
        "final_coverage_rate": final_coverage_rate,
        "cumulative_coverage_rate_delta": cumulative_coverage_rate_delta,
        "total_path_cost": total_path_cost,
        "average_risk": total_risk / selected_count if selected_count else 0.0,
        "failure_count": failure_count,
        "replan_count": replan_count,
        "value_coverage": value_coverage,
    }


def _select_utility_goal(contract: ModelExplorerContract) -> GoalCandidate | None:
    reachable_goals = tuple(goal for goal in contract.top_goals if goal.reachable)
    if not reachable_goals:
        return None
    return sorted(reachable_goals, key=lambda goal: (-goal.utility, goal.cell[0], goal.cell[1]))[0]


def _selected_action_index(contract: ModelExplorerContract, selected_goal: GoalCandidate) -> int:
    for index, goal in enumerate(contract.top_goals):
        if goal.cell == selected_goal.cell:
            return index
    return -1


def _coverage_rate(observation_update: dict[str, Any], *, fallback: float) -> float:
    value = observation_update.get("coverage_rate")
    if isinstance(value, bool) or value is None:
        return fallback
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _value_coverage(observation_update: dict[str, Any]) -> float:
    return (
        _numeric_mapping_value(observation_update, "value_coverage")
        or _numeric_mapping_value(observation_update, "value_coverage_reward")
        or 0.0
    )


def _should_count_replan(
    contract: ModelExplorerContract,
    selected_goal: GoalCandidate,
    previous_goal: GoalCandidate | None,
) -> bool:
    if previous_goal is not None and previous_goal.cell != selected_goal.cell:
        return True
    delta_c = _numeric_mapping_value(contract.observation_update, "delta_c")
    return delta_c is not None and delta_c >= 0.1


def _numeric_mapping_value(mapping: dict[str, Any], field: str) -> float | None:
    value = mapping.get(field)
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
