from __future__ import annotations

from collections.abc import Iterable, Sequence
from math import exp, isfinite, log
from typing import Any

from ..core.interfaces import GoalCandidate, ModelExplorerContract
from ..decision.selector import select_goal
from ..io.scenario import Scenario
from .features import extract_policy_observation
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

    utility_metrics = _evaluate_strategy(snapshots, _select_utility_goal, planning_adapter=planning_adapter)
    coverage_metrics = _evaluate_strategy(
        snapshots,
        lambda contract: select_goal(contract).selected_goal,
        planning_adapter=planning_adapter,
    )
    report = {
        "utility": utility_metrics,
        "coverage_heuristic": coverage_metrics,
    }
    if torch_policy is not None:
        torch_metrics = _evaluate_strategy(
            snapshots,
            lambda contract: select_goal(contract, policy=torch_policy).selected_goal,
            planning_adapter=planning_adapter,
        )
        torch_metrics["action_diagnostics"] = _policy_action_diagnostics(
            snapshots,
            torch_policy,
            utility_metrics=utility_metrics,
            coverage_metrics=coverage_metrics,
        )
        report["torch_policy"] = torch_metrics
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
        aggregate[strategy_name]["action_sensitive_metrics"] = _aggregate_nested_numeric(
            strategy_reports,
            "action_sensitive_metrics",
            average_fields={"selected_risk"},
        )
        aggregate[strategy_name]["oracle_metrics"] = _aggregate_nested_numeric(
            strategy_reports,
            "oracle_metrics",
            average_fields={"low_risk_oracle_risk"},
        )
        aggregate[strategy_name]["oracle_regret"] = _aggregate_nested_numeric(
            strategy_reports,
            "oracle_regret",
            average_fields={"risk_regret"},
        )
        aggregate[strategy_name]["sample_discriminativeness"] = _aggregate_nested_numeric(
            strategy_reports,
            "sample_discriminativeness",
            average_fields={
                "candidate_coverage_spread",
                "risk_spread",
                "path_cost_spread",
                "value_spread",
                "oracle_vs_heuristic_action_disagreement_rate",
            },
        )
    return aggregate


def _aggregate_nested_numeric(
    reports: tuple[dict[str, Any], ...],
    section: str,
    *,
    average_fields: set[str],
) -> dict[str, float]:
    values: dict[str, list[float]] = {}
    for report in reports:
        nested = report.get(section, {})
        if not isinstance(nested, dict):
            continue
        for key, value in nested.items():
            if isinstance(value, bool):
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if isfinite(numeric):
                values.setdefault(str(key), []).append(numeric)
    aggregate: dict[str, float] = {}
    for key, numbers in values.items():
        total = sum(numbers)
        aggregate[key] = total / len(numbers) if key in average_fields and numbers else total
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
    selected_expected_coverage_delta = 0.0
    selected_value_coverage = 0.0
    selected_path_cost = 0.0
    selected_risk_total = 0.0
    selected_composite_utility = 0.0
    coverage_oracle_expected_coverage_delta = 0.0
    low_risk_oracle_risk_total = 0.0
    cost_oracle_path_cost = 0.0
    composite_oracle_utility = 0.0
    coverage_regret = 0.0
    risk_regret_total = 0.0
    path_cost_regret = 0.0
    composite_regret = 0.0
    oracle_count = 0
    coverage_spread_total = 0.0
    risk_spread_total = 0.0
    path_cost_spread_total = 0.0
    value_spread_total = 0.0
    oracle_heuristic_disagreement_count = 0
    sample_discriminativeness_count = 0
    oracle_action_cells: dict[str, list[list[int] | None]] = {
        "coverage_oracle_cells": [],
        "low_risk_oracle_cells": [],
        "cost_oracle_cells": [],
        "composite_oracle_cells": [],
    }
    previous_goal: GoalCandidate | None = None
    current_cell = (0, 0)

    for step_index, contract in enumerate(snapshots):
        oracle_info = _oracle_info(contract)
        if oracle_info is not None:
            oracle_count += 1
            coverage_oracle_expected_coverage_delta += oracle_info["coverage_delta"]
            low_risk_oracle_risk_total += oracle_info["low_risk"]
            cost_oracle_path_cost += oracle_info["path_cost"]
            composite_oracle_utility += oracle_info["composite_utility"]
            coverage_spread_total += oracle_info["candidate_coverage_spread"]
            risk_spread_total += oracle_info["risk_spread"]
            path_cost_spread_total += oracle_info["path_cost_spread"]
            value_spread_total += oracle_info["value_spread"]
            oracle_heuristic_disagreement_count += int(oracle_info["heuristic_disagreement"])
            sample_discriminativeness_count += 1
            for key in oracle_action_cells:
                oracle_action_cells[key].append(oracle_info[key[:-1]])
        else:
            for key in oracle_action_cells:
                oracle_action_cells[key].append(None)

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
        selected_coverage_delta = _goal_coverage_delta(selected_goal)
        selected_value = _goal_value(selected_goal)
        selected_composite = _goal_composite_utility(
            selected_goal,
            path_cost_override=reward_info.path_cost,
            risk_override=reward_info.risk,
        )
        selected_expected_coverage_delta += selected_coverage_delta
        selected_value_coverage += selected_value
        selected_path_cost += reward_info.path_cost
        selected_risk_total += reward_info.risk
        selected_composite_utility += selected_composite
        if oracle_info is not None:
            coverage_regret += oracle_info["coverage_delta"] - selected_coverage_delta
            risk_regret_total += reward_info.risk - oracle_info["low_risk"]
            path_cost_regret += reward_info.path_cost - oracle_info["path_cost"]
            composite_regret += oracle_info["composite_utility"] - selected_composite
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
        "action_sensitive_metrics": {
            "selected_expected_coverage_delta": _finite_float(selected_expected_coverage_delta),
            "selected_value_coverage": _finite_float(selected_value_coverage),
            "selected_risk": _finite_float(selected_risk_total / selected_count if selected_count else 0.0),
            "selected_path_cost": _finite_float(selected_path_cost),
            "selected_composite_utility": _finite_float(selected_composite_utility),
            "selected_count": selected_count,
        },
        "oracle_metrics": {
            "coverage_oracle_expected_coverage_delta": _finite_float(coverage_oracle_expected_coverage_delta),
            "low_risk_oracle_risk": _finite_float(low_risk_oracle_risk_total / oracle_count if oracle_count else 0.0),
            "cost_oracle_path_cost": _finite_float(cost_oracle_path_cost),
            "composite_oracle_utility": _finite_float(composite_oracle_utility),
            "oracle_count": oracle_count,
        },
        "oracle_regret": {
            "coverage_regret": _finite_float(coverage_regret),
            "risk_regret": _finite_float(risk_regret_total / oracle_count if oracle_count else 0.0),
            "path_cost_regret": _finite_float(path_cost_regret),
            "composite_regret": _finite_float(composite_regret),
        },
        "sample_discriminativeness": {
            "candidate_coverage_spread": _finite_float(
                coverage_spread_total / sample_discriminativeness_count
                if sample_discriminativeness_count
                else 0.0
            ),
            "risk_spread": _finite_float(
                risk_spread_total / sample_discriminativeness_count
                if sample_discriminativeness_count
                else 0.0
            ),
            "path_cost_spread": _finite_float(
                path_cost_spread_total / sample_discriminativeness_count
                if sample_discriminativeness_count
                else 0.0
            ),
            "value_spread": _finite_float(
                value_spread_total / sample_discriminativeness_count
                if sample_discriminativeness_count
                else 0.0
            ),
            "oracle_vs_heuristic_action_disagreement_rate": _finite_float(
                oracle_heuristic_disagreement_count / sample_discriminativeness_count
                if sample_discriminativeness_count
                else 0.0
            ),
        },
        "oracle_actions": {
            "coverage_oracle_cell": _first_cell(oracle_action_cells["coverage_oracle_cells"]),
            "low_risk_oracle_cell": _first_cell(oracle_action_cells["low_risk_oracle_cells"]),
            "cost_oracle_cell": _first_cell(oracle_action_cells["cost_oracle_cells"]),
            "composite_oracle_cell": _first_cell(oracle_action_cells["composite_oracle_cells"]),
            **oracle_action_cells,
        },
    }


def _oracle_info(contract: ModelExplorerContract) -> dict[str, Any] | None:
    reachable_goals = tuple(goal for goal in contract.top_goals if goal.reachable)
    if not reachable_goals:
        return None

    coverage_oracle = _best_goal(
        reachable_goals,
        key=lambda goal: _goal_coverage_delta(goal),
        reverse=True,
    )
    low_risk_oracle = _best_goal(
        reachable_goals,
        key=lambda goal: _goal_risk(goal),
        reverse=False,
    )
    cost_oracle = _best_goal(
        reachable_goals,
        key=lambda goal: _goal_path_cost(goal),
        reverse=False,
    )
    composite_oracle = _best_goal(
        reachable_goals,
        key=lambda goal: _goal_composite_utility(goal),
        reverse=True,
    )
    heuristic_goal = select_goal(contract).selected_goal
    return {
        "coverage_delta": _goal_coverage_delta(coverage_oracle),
        "low_risk": _goal_risk(low_risk_oracle),
        "path_cost": _goal_path_cost(cost_oracle),
        "composite_utility": _goal_composite_utility(composite_oracle),
        "candidate_coverage_spread": _spread(_goal_coverage_delta(goal) for goal in reachable_goals),
        "risk_spread": _spread(_goal_risk(goal) for goal in reachable_goals),
        "path_cost_spread": _spread(_goal_path_cost(goal) for goal in reachable_goals),
        "value_spread": _spread(_goal_value(goal) for goal in reachable_goals),
        "heuristic_disagreement": _cell(heuristic_goal) != _cell(composite_oracle),
        "coverage_oracle_cell": _cell(coverage_oracle),
        "low_risk_oracle_cell": _cell(low_risk_oracle),
        "cost_oracle_cell": _cell(cost_oracle),
        "composite_oracle_cell": _cell(composite_oracle),
    }


def _best_goal(
    goals: tuple[GoalCandidate, ...],
    *,
    key,
    reverse: bool,
) -> GoalCandidate:
    return sorted(
        goals,
        key=lambda goal: (
            -key(goal) if reverse else key(goal),
            -goal.utility,
            goal.cell[0],
            goal.cell[1],
        ),
    )[0]


def _goal_coverage_delta(goal: GoalCandidate) -> float:
    explicit_rate_delta = _numeric_mapping_value(goal.experimental, "expected_coverage_rate_delta")
    if explicit_rate_delta is not None:
        return _finite_float(explicit_rate_delta)
    return _numeric_goal_value(goal, "expected_new_coverage_area")


def _goal_value(goal: GoalCandidate) -> float:
    return _numeric_goal_value(goal, "value")


def _goal_risk(goal: GoalCandidate) -> float:
    return _numeric_goal_value(goal, "risk")


def _goal_path_cost(goal: GoalCandidate) -> float:
    return _numeric_goal_value(goal, "path_cost")


def _goal_composite_utility(
    goal: GoalCandidate,
    *,
    path_cost_override: float | None = None,
    risk_override: float | None = None,
) -> float:
    path_cost = _finite_float(path_cost_override) if path_cost_override is not None else _goal_path_cost(goal)
    risk = _finite_float(risk_override) if risk_override is not None else _goal_risk(goal)
    return _finite_float(_goal_coverage_delta(goal) + 0.25 * _goal_value(goal) - 0.25 * risk - 0.05 * path_cost)


def _numeric_goal_value(goal: GoalCandidate, field: str) -> float:
    return _numeric_mapping_value(goal.experimental, field) or 0.0


def _spread(values: Iterable[float]) -> float:
    finite_values = tuple(_finite_float(value) for value in values)
    return max(finite_values) - min(finite_values) if finite_values else 0.0


def _cell(goal: GoalCandidate | None) -> list[int] | None:
    return None if goal is None else [goal.cell[0], goal.cell[1]]


def _first_cell(cells: list[list[int] | None]) -> list[int] | None:
    for cell in cells:
        if cell is not None:
            return cell
    return None


def _select_utility_goal(contract: ModelExplorerContract) -> GoalCandidate | None:
    reachable_goals = tuple(goal for goal in contract.top_goals if goal.reachable)
    if not reachable_goals:
        return None
    return sorted(reachable_goals, key=lambda goal: (-goal.utility, goal.cell[0], goal.cell[1]))[0]


def _policy_action_diagnostics(
    snapshots: tuple[ModelExplorerContract, ...],
    policy,
    *,
    utility_metrics: dict[str, Any],
    coverage_metrics: dict[str, Any],
) -> list[dict[str, Any]]:
    diagnostics: list[dict[str, Any]] = []
    utility_cells = utility_metrics.get("selected_cells", [])
    coverage_cells = coverage_metrics.get("selected_cells", [])
    for step_index, contract in enumerate(snapshots):
        observation = extract_policy_observation(contract)
        selected_goal = select_goal(contract, policy=policy).selected_goal
        scores = _policy_scores(policy, observation, expected_count=len(observation.action_mask))
        probabilities = _masked_probabilities(scores, observation.action_mask)
        selected_index = -1 if selected_goal is None else _selected_action_index(contract, selected_goal)
        selected_cell = None if selected_goal is None else [selected_goal.cell[0], selected_goal.cell[1]]
        selected_probability = (
            probabilities[selected_index]
            if selected_index >= 0 and selected_index < len(probabilities)
            else 0.0
        )
        diagnostics.append(
            {
                "step_index": step_index,
                "selected_index": None if selected_index < 0 else selected_index,
                "selected_cell": selected_cell,
                "selected_action_probability": selected_probability,
                "action_rank": _action_rank(contract, scores, selected_index, observation.action_mask),
                "entropy": _entropy(probabilities),
                "valid_action_count": sum(1 for is_valid in observation.action_mask if is_valid),
                "agrees_with_utility": selected_cell == _cell_at(utility_cells, step_index),
                "agrees_with_coverage_heuristic": selected_cell == _cell_at(coverage_cells, step_index),
                "selected_action_mask_valid": (
                    selected_index >= 0
                    and selected_index < len(observation.action_mask)
                    and bool(observation.action_mask[selected_index])
                ),
                "max_masked_action_probability": _max_masked_probability(
                    probabilities,
                    observation.action_mask,
                ),
            }
        )
    return diagnostics


def _policy_scores(policy, observation, *, expected_count: int) -> tuple[float, ...] | None:
    score_method = getattr(policy, "score", None)
    if score_method is None:
        return None
    try:
        raw_scores = score_method(observation)
    except Exception:
        return None
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


def _masked_probabilities(
    scores: tuple[float, ...] | None,
    action_mask: tuple[bool, ...],
) -> tuple[float, ...]:
    if scores is None or len(scores) != len(action_mask):
        return tuple(0.0 for _ in action_mask)
    valid_indices = [index for index, is_valid in enumerate(action_mask) if is_valid]
    if not valid_indices:
        return tuple(0.0 for _ in action_mask)
    max_score = max(scores[index] for index in valid_indices)
    weighted = {
        index: exp(max(min(scores[index] - max_score, 80.0), -80.0))
        for index in valid_indices
    }
    denominator = sum(weighted.values())
    if denominator <= 0.0 or not isfinite(denominator):
        return tuple(0.0 for _ in action_mask)
    return tuple(weighted.get(index, 0.0) / denominator for index in range(len(action_mask)))


def _action_rank(
    contract: ModelExplorerContract,
    scores: tuple[float, ...] | None,
    selected_index: int,
    action_mask: tuple[bool, ...],
) -> int | None:
    if scores is None or selected_index < 0 or selected_index >= len(scores):
        return None
    ranked_indices = sorted(
        (index for index, is_valid in enumerate(action_mask) if is_valid),
        key=lambda index: (
            -scores[index],
            -contract.top_goals[index].utility,
            contract.top_goals[index].cell[0],
            contract.top_goals[index].cell[1],
        ),
    )
    for rank, index in enumerate(ranked_indices, start=1):
        if index == selected_index:
            return rank
    return None


def _entropy(probabilities: tuple[float, ...]) -> float:
    return -sum(probability * log(probability) for probability in probabilities if probability > 0.0)


def _max_masked_probability(probabilities: tuple[float, ...], action_mask: tuple[bool, ...]) -> float:
    masked = [probability for probability, is_valid in zip(probabilities, action_mask) if not is_valid]
    return max(masked) if masked else 0.0


def _cell_at(cells: Any, index: int) -> list[int] | None:
    if not isinstance(cells, list) or index >= len(cells):
        return None
    cell = cells[index]
    if not isinstance(cell, list) or len(cell) != 2:
        return None
    return [int(cell[0]), int(cell[1])]


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
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if isfinite(numeric) else None


def _finite_float(value: Any) -> float:
    if isinstance(value, bool):
        return 0.0
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    return numeric if isfinite(numeric) else 0.0
