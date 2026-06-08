from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from ..core.interfaces import ExplorerDecision, GoalCandidate, ModelExplorerContract
from ..decision.selector import select_goal
from ..io.scenario import Scenario
from .execution import ExecutionFeasibilityAdapter, ExecutionFeasibilityRequest
from .features import extract_policy_observation
from .feedback_selection import select_goal_with_path_feedback
from .planning import (
    AnchorProjectionCandidateConfig,
    ContractCostPlanner,
    PathPlanRequest,
    PathPlanResult,
    PathPlanningAdapter,
)
from .provider import ContractProvider, ProviderStepRequest, SequenceContractProvider
from .reward import compute_step_reward
from .rollout import EpisodeMetrics, RolloutEpisode, RolloutInfo, RolloutTransition


def collect_rollout_episode(
    scenario_or_snapshots: Scenario | Iterable[ModelExplorerContract],
    *,
    policy=None,
    max_candidates: int | None = None,
    planning_adapter: PathPlanningAdapter | None = None,
    anchor_projection_candidate_config: AnchorProjectionCandidateConfig | dict[str, Any] | None = None,
    reward_config: dict[str, Any] | None = None,
    selection_strategy: str = "auto",
) -> RolloutEpisode:
    if isinstance(scenario_or_snapshots, Scenario):
        snapshots = scenario_or_snapshots.snapshots
        rollout_metadata = dict(scenario_or_snapshots.metadata)
    else:
        snapshots = tuple(scenario_or_snapshots)
        rollout_metadata = {}
    if not snapshots:
        return RolloutEpisode(transitions=(), metrics=EpisodeMetrics())

    return collect_dynamic_rollout_episode(
        SequenceContractProvider(snapshots),
        policy=policy,
        max_steps=len(snapshots),
        max_candidates=max_candidates,
        planning_adapter=planning_adapter,
        anchor_projection_candidate_config=anchor_projection_candidate_config,
        reward_config=reward_config,
        rollout_metadata=rollout_metadata,
        selection_strategy=selection_strategy,
    )


def collect_dynamic_rollout_episode(
    provider: ContractProvider,
    *,
    policy=None,
    max_steps: int | None = None,
    max_candidates: int | None = None,
    planning_adapter: PathPlanningAdapter | None = None,
    anchor_projection_candidate_config: AnchorProjectionCandidateConfig | dict[str, Any] | None = None,
    execution_adapter: ExecutionFeasibilityAdapter | None = None,
    reward_config: dict[str, Any] | None = None,
    rollout_metadata: dict[str, Any] | None = None,
    selection_strategy: str = "auto",
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
    current_cell = (0, 0)
    default_planner = ContractCostPlanner()
    reward_kwargs = _reward_kwargs(reward_config)
    requested_selection_strategy = _normalize_selection_strategy(selection_strategy)

    while current_contract is not None and (step_limit is None or step_index < step_limit):
        remaining_steps = _remaining_steps(step_limit, step_index)
        observation = extract_policy_observation(
            current_contract,
            step_index=step_index,
            remaining_steps=remaining_steps,
            max_candidates=max_candidates,
        )
        decision, effective_selection_strategy, feedback_selection = _select_rollout_goal(
            current_contract,
            policy=policy,
            planning_adapter=planning_adapter,
            current_cell=current_cell,
            step_index=step_index,
            max_candidates=max_candidates,
            selection_strategy=requested_selection_strategy,
            anchor_projection_candidate_config=anchor_projection_candidate_config,
        )
        selected_goal = decision.selected_goal
        failure_reason = None if selected_goal is not None else "no_reachable_goal"
        action_index = -1 if selected_goal is None else _selected_action_index(current_contract, selected_goal.cell)
        extra_info: dict[str, Any] = _rollout_metadata_info(rollout_metadata)
        extra_info.update(
            {
                "requested_selection_strategy": requested_selection_strategy,
                "selection_strategy": effective_selection_strategy,
                "teacher_action_index": None if action_index < 0 else action_index,
                "teacher_selected_cell": None
                if selected_goal is None
                else [selected_goal.cell[0], selected_goal.cell[1]],
            }
        )
        if feedback_selection is not None:
            extra_info.update(_feedback_teacher_info(feedback_selection))

        planning_result: PathPlanResult | None = None
        planner = planning_adapter if planning_adapter is not None else default_planner
        if selected_goal is not None:
            if (
                feedback_selection is not None
                and feedback_selection.selected_evaluation is not None
                and feedback_selection.selected_evaluation.action_index == action_index
            ):
                planning_result = feedback_selection.selected_evaluation.result
                extra_info["selection_score"] = feedback_selection.scores_by_action_index.get(action_index)
            else:
                planning_result = planner.plan(
                    PathPlanRequest(
                        contract=current_contract,
                        step_index=step_index,
                        action_index=action_index,
                        selected_goal=selected_goal,
                        current_cell=current_cell,
                    )
                )
            extra_info["planning_feasible"] = bool(planning_result.feasible)
            extra_info["planning_metadata"] = dict(planning_result.metadata)
            extra_info["path_length"] = float(planning_result.path_length)
            if not planning_result.feasible:
                failure_reason = planning_result.failure_reason or "path_planning_failed"

        execution_response = None
        if selected_goal is not None and execution_adapter is not None and planning_adapter is None:
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
            planner_replan_required=bool(planning_result.replan_required) if planning_result is not None else False,
            execution_replan_required=bool(execution_response.replan_required) if execution_response is not None else False,
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
            path_cost_override=None if planning_result is None else planning_result.path_cost,
            risk_override=None if planning_result is None else planning_result.risk,
            **reward_kwargs,
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
        if selected_goal is not None and failure_reason is None:
            current_cell = selected_goal.cell
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


_SELECTION_STRATEGIES = {"auto", "utility", "coverage_heuristic", "feedback_aware"}


def _normalize_selection_strategy(value: str | None) -> str:
    strategy = "auto" if value is None else str(value).strip().lower()
    if strategy not in _SELECTION_STRATEGIES:
        allowed = ", ".join(sorted(_SELECTION_STRATEGIES))
        raise ValueError(f"selection_strategy must be one of: {allowed}")
    return strategy


def _select_rollout_goal(
    contract: ModelExplorerContract,
    *,
    policy,
    planning_adapter: PathPlanningAdapter | None,
    current_cell: tuple[int, int],
    step_index: int,
    max_candidates: int | None,
    selection_strategy: str,
    anchor_projection_candidate_config: AnchorProjectionCandidateConfig | dict[str, Any] | None,
):
    if policy is not None:
        return select_goal(contract, policy=policy), "external_policy", None
    if selection_strategy == "utility":
        return _select_utility_goal(contract), "utility", None
    if selection_strategy == "coverage_heuristic":
        return select_goal(contract), "coverage_heuristic", None
    if selection_strategy == "feedback_aware":
        if planning_adapter is not None:
            feedback_selection = select_goal_with_path_feedback(
                contract,
                planner=planning_adapter,
                current_cell=current_cell,
                step_index=step_index,
                top_k=max_candidates,
                anchor_projection_candidate_config=anchor_projection_candidate_config,
            )
            return feedback_selection.decision, "feedback_aware", feedback_selection
        return select_goal(contract), "coverage_heuristic", None

    if planning_adapter is not None:
        feedback_selection = select_goal_with_path_feedback(
            contract,
            planner=planning_adapter,
            current_cell=current_cell,
            step_index=step_index,
            top_k=max_candidates,
            anchor_projection_candidate_config=anchor_projection_candidate_config,
        )
        return feedback_selection.decision, "feedback_aware", feedback_selection
    return select_goal(contract), "coverage_heuristic", None


def _select_utility_goal(contract: ModelExplorerContract) -> ExplorerDecision:
    ranked_goals = tuple(
        sorted(
            (goal for goal in contract.top_goals if goal.reachable),
            key=lambda goal: (-goal.utility, goal.cell[0], goal.cell[1]),
        )
    )
    if not ranked_goals:
        return ExplorerDecision(status="no_reachable_goal", selected_goal=None, ranked_goals=())
    return ExplorerDecision(status="selected", selected_goal=ranked_goals[0], ranked_goals=ranked_goals)


def _feedback_teacher_info(feedback_selection) -> dict[str, Any]:
    info: dict[str, Any] = {}
    if feedback_selection.selected_action_index is not None:
        info["teacher_action_index"] = int(feedback_selection.selected_action_index)
    if feedback_selection.decision.selected_goal is not None:
        cell = feedback_selection.decision.selected_goal.cell
        info["teacher_selected_cell"] = [int(cell[0]), int(cell[1])]
    if feedback_selection.selected_score is not None:
        info["teacher_score"] = float(feedback_selection.selected_score)
        info["selection_score"] = float(feedback_selection.selected_score)
    if feedback_selection.runner_up_action_index is not None:
        info["teacher_runner_up_action_index"] = int(feedback_selection.runner_up_action_index)
    if feedback_selection.runner_up_score is not None:
        info["teacher_runner_up_score"] = float(feedback_selection.runner_up_score)
    if feedback_selection.score_margin is not None:
        info["teacher_score_margin"] = float(feedback_selection.score_margin)
    info["teacher_ranked_action_indices"] = [
        int(index) for index in feedback_selection.ranked_action_indices
    ]
    return info


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
    planner_replan_required: bool,
    execution_replan_required: bool,
) -> tuple[str, ...]:
    reasons = _local_replan_reasons(contract, decision, previous_decision)
    reasons.extend(provider_reasons)
    if planner_replan_required:
        reasons.append("path_planning_failed")
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


def _reward_kwargs(config: dict[str, Any] | None) -> dict[str, float]:
    allowed = ("path_cost_weight", "path_cost_normalizer", "risk_weight", "failure_penalty")
    if config is None:
        return {}
    return {key: float(config[key]) for key in allowed if key in config}


def _rollout_metadata_info(metadata: dict[str, Any] | None) -> dict[str, Any]:
    if not metadata:
        return {}
    info = {"provenance": dict(metadata)}
    for key in ("dataset_id", "data_class", "region", "generator_version"):
        if key in metadata:
            info[key] = metadata[key]
    return info
