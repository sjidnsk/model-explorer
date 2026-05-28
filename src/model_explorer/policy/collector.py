from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from ..core.interfaces import ExplorerDecision, GoalCandidate, ModelExplorerContract
from ..decision.selector import select_goal
from ..io.scenario import Scenario
from .execution import ExecutionFeasibilityAdapter, ExecutionFeasibilityRequest
from .features import extract_policy_observation
from .provider import ContractProvider, ProviderStepRequest, SequenceContractProvider
from .reward import compute_step_reward
from .rollout import EpisodeMetrics, RolloutEpisode, RolloutInfo, RolloutTransition


def collect_rollout_episode(
    scenario_or_snapshots: Scenario | Iterable[ModelExplorerContract],
    *,
    policy=None,
    max_candidates: int | None = None,
) -> RolloutEpisode:
    snapshots = (
        scenario_or_snapshots.snapshots
        if isinstance(scenario_or_snapshots, Scenario)
        else tuple(scenario_or_snapshots)
    )
    if not snapshots:
        return RolloutEpisode(transitions=(), metrics=EpisodeMetrics())

    return collect_dynamic_rollout_episode(
        SequenceContractProvider(snapshots),
        policy=policy,
        max_steps=len(snapshots),
        max_candidates=max_candidates,
    )


def collect_dynamic_rollout_episode(
    provider: ContractProvider,
    *,
    policy=None,
    max_steps: int | None = None,
    max_candidates: int | None = None,
    execution_adapter: ExecutionFeasibilityAdapter | None = None,
) -> RolloutEpisode:
    transitions: list[RolloutTransition] = []
    total_path_cost = 0.0
    total_risk = 0.0
    selected_count = 0
    cumulative_coverage_rate_delta = 0.0
    failure_count = 0
    replan_count = 0
    final_coverage_rate = 0.0
    value_coverage = 0.0
    previous_decision: ExplorerDecision | None = None
    current_contract = provider.initial_contract()
    step_limit = max_steps if max_steps is not None else getattr(provider, "total_steps", None)
    step_index = 0

    while current_contract is not None and (step_limit is None or step_index < step_limit):
        remaining_steps = _remaining_steps(step_limit, step_index)
        observation = extract_policy_observation(
            current_contract,
            step_index=step_index,
            remaining_steps=remaining_steps,
            max_candidates=max_candidates,
        )
        decision = select_goal(current_contract, policy=policy)
        selected_goal = decision.selected_goal
        failure_reason = None if selected_goal is not None else "no_reachable_goal"
        action_index = -1 if selected_goal is None else _selected_action_index(current_contract, selected_goal.cell)
        extra_info: dict[str, Any] = {}

        execution_response = None
        if selected_goal is not None and execution_adapter is not None:
            execution_response = execution_adapter.check_feasibility(
                ExecutionFeasibilityRequest(
                    schema_version=current_contract.schema_version,
                    step_index=step_index,
                    action_index=action_index,
                    selected_cell=selected_goal.cell,
                    selected_world=current_contract.cell_to_world(selected_goal.cell),
                )
            )
            extra_info["execution_feasible"] = bool(execution_response.feasible)
            extra_info["execution_metrics"] = dict(execution_response.metrics)
            if not execution_response.feasible:
                failure_reason = execution_response.failure_reason or "execution_infeasible"

        provider_result = provider.advance(
            ProviderStepRequest(
                step_index=step_index,
                contract=current_contract,
                action_index=action_index,
                selected_goal=selected_goal,
                failure_reason=failure_reason,
                info=dict(extra_info),
            )
        )
        if provider_result.failure_reason is not None:
            if failure_reason is None:
                failure_reason = provider_result.failure_reason
            else:
                extra_info["provider_failure_reason"] = provider_result.failure_reason
        extra_info.update(provider_result.info)

        replan_reasons = _combined_replan_reasons(
            current_contract,
            decision,
            previous_decision,
            provider_result.replan_reasons,
            execution_replan_required=(
                bool(execution_response.replan_required) if execution_response is not None else False
            ),
        )
        extra_info["replan_reasons"] = list(replan_reasons)

        if failure_reason is not None:
            failure_count += 1
        if replan_reasons:
            replan_count += 1

        reward_info = compute_step_reward(
            selected_goal,
            current_contract.observation_update,
            failure_reason=failure_reason,
        )
        final_coverage_rate = _coverage_rate(current_contract.observation_update, fallback=final_coverage_rate)
        value_coverage += _value_coverage(current_contract.observation_update)
        cumulative_coverage_rate_delta += reward_info.coverage_rate_delta
        total_path_cost += reward_info.path_cost
        total_risk += reward_info.risk
        if selected_goal is not None:
            selected_count += 1

        reached_step_limit = step_limit is not None and step_index + 1 >= step_limit
        next_contract = None if reached_step_limit else provider_result.next_contract
        done = bool(provider_result.done or reached_step_limit or next_contract is None)
        next_observation = (
            None
            if next_contract is None
            else extract_policy_observation(
                next_contract,
                step_index=step_index + 1,
                remaining_steps=_remaining_steps(step_limit, step_index + 1),
                max_candidates=max_candidates,
            )
        )

        transitions.append(
            RolloutTransition(
                observation=observation,
                action_index=action_index,
                log_prob=None,
                value=None,
                reward=reward_info.reward,
                next_observation=next_observation,
                done=done,
                info=RolloutInfo(
                    selected_cell=None if selected_goal is None else selected_goal.cell,
                    coverage_rate_delta=reward_info.coverage_rate_delta,
                    path_cost=reward_info.path_cost,
                    risk=reward_info.risk,
                    failure_reason=reward_info.failure_reason,
                    final_coverage_rate=final_coverage_rate if done else None,
                    total_cost=total_path_cost,
                    failure_count=failure_count,
                    replan_count=replan_count,
                    extra=extra_info,
                ),
            )
        )

        previous_decision = decision
        current_contract = next_contract
        step_index += 1

    average_risk = total_risk / selected_count if selected_count else 0.0
    return RolloutEpisode(
        transitions=tuple(transitions),
        metrics=EpisodeMetrics(
            final_coverage_rate=final_coverage_rate,
            cumulative_coverage_rate_delta=cumulative_coverage_rate_delta,
            total_path_cost=total_path_cost,
            average_risk=average_risk,
            failure_count=failure_count,
            replan_count=replan_count,
            value_coverage=value_coverage,
        ),
    )


def _selected_action_index(contract: ModelExplorerContract, selected_cell: tuple[int, int]) -> int:
    for index, goal in enumerate(contract.top_goals):
        if goal.cell == selected_cell:
            return index
    raise ValueError(f"selected cell {selected_cell!r} is not present in top_goals")


def _coverage_rate(observation_update: dict, *, fallback: float) -> float:
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


def _remaining_steps(step_limit: int | None, step_index: int) -> int:
    if step_limit is None:
        return 0
    return max(step_limit - step_index - 1, 0)


def _combined_replan_reasons(
    contract: ModelExplorerContract,
    decision: ExplorerDecision,
    previous_decision: ExplorerDecision | None,
    provider_reasons: tuple[str, ...],
    *,
    execution_replan_required: bool,
) -> tuple[str, ...]:
    reasons = _local_replan_reasons(contract, decision, previous_decision)
    reasons.extend(provider_reasons)
    if execution_replan_required:
        reasons.append("execution_infeasible")
    return _dedupe_strings(reasons)


def _local_replan_reasons(
    contract: ModelExplorerContract,
    decision: ExplorerDecision,
    previous_decision: ExplorerDecision | None,
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

    delta_c = _numeric_mapping_value(contract.observation_update, "delta_c")
    if delta_c is not None and delta_c >= 0.1:
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


def _numeric_mapping_value(mapping: dict[str, Any], field: str) -> float | None:
    value = mapping.get(field)
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _dedupe_strings(values: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return tuple(result)
