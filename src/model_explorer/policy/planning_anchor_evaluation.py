"""Candidate path evaluation and anchor projection candidate generation."""

from __future__ import annotations

from typing import Any

from ..core.interfaces import GoalCandidate, ModelExplorerContract
from .planning_anchor_projection import (
    anchor_projection_candidate_config_from_mapping,
    _anchor_projection_candidate_generation_payload,
    _planner_validated_distance_exception_safe,
    _trainability_distance_reject_reasons,
)
from .planning_platform_feasibility import _platform_goal_feasibility
from .planning_types import (
    AnchorProjectionCandidateConfig,
    PathCandidateEvaluation,
    PathPlanningAdapter,
    PathPlanRequest,
    PathPlanResult,
)
from .planning_utils import (
    _cell_pair,
    _optional_nonnegative_float,
    _optional_nonnegative_int,
)


def evaluate_candidate_paths(
    contract: ModelExplorerContract,
    *,
    current_cell: tuple[int, int],
    top_k: int,
    planner: PathPlanningAdapter,
    step_index: int = 0,
    anchor_projection_candidate_config: AnchorProjectionCandidateConfig | dict[str, Any] | None = None,
) -> tuple[PathCandidateEvaluation, ...]:
    evaluations: list[PathCandidateEvaluation] = []
    projection_config = anchor_projection_candidate_config_from_mapping(anchor_projection_candidate_config)
    evaluated_policy_candidate_count = 0
    next_synthetic_action_index = len(contract.top_goals)
    for action_index, goal in enumerate(contract.top_goals):
        if evaluated_policy_candidate_count >= top_k:
            break
        if not goal.reachable:
            continue
        result = planner.plan(
            PathPlanRequest(
                contract=contract,
                step_index=step_index,
                action_index=action_index,
                selected_goal=goal,
                current_cell=current_cell,
            )
        )
        evaluation = PathCandidateEvaluation(
            action_index=action_index,
            cell=goal.cell,
            utility=goal.utility,
            result=result,
            selection_goal=goal,
        )
        evaluations.append(evaluation)
        same_action = _same_action_anchor_projection_candidate_evaluation(
            contract=contract,
            current_cell=current_cell,
            step_index=step_index,
            source_action_index=action_index,
            goal=goal,
            source_result=result,
            planner=planner,
            config=projection_config,
        )
        if same_action is not None:
            evaluations.append(same_action)
        evaluated_policy_candidate_count += 1
        projected = _projected_anchor_candidate_evaluation(
            contract=contract,
            current_cell=current_cell,
            step_index=step_index,
            source_action_index=action_index,
            synthetic_action_index=next_synthetic_action_index,
            goal=goal,
            source_result=result,
            planner=planner,
            config=projection_config,
        )
        if projected is not None:
            evaluations.append(projected)
            next_synthetic_action_index += 1
    return tuple(evaluations)


def _same_action_anchor_projection_candidate_evaluation(
    *,
    contract: ModelExplorerContract,
    current_cell: tuple[int, int],
    step_index: int,
    source_action_index: int,
    goal: GoalCandidate,
    source_result: PathPlanResult,
    planner: PathPlanningAdapter,
    config: AnchorProjectionCandidateConfig,
) -> PathCandidateEvaluation | None:
    if not config.enabled or not config.contract_aware_trainable_target_generation:
        return None
    source_feasibility = _platform_goal_feasibility(cell=goal.cell, result=source_result)
    if source_feasibility.get("classification") != "platform_inflated_goal_blocked":
        return None
    projection = source_feasibility.get("anchor_projection")
    projection = projection if isinstance(projection, dict) else {}
    anchor = _cell_pair(
        projection.get("projected_anchor_cell")
        or projection.get("nearest_inflated_passable_anchor")
        or source_feasibility.get("nearest_inflated_passable_anchor")
    )
    if anchor is None:
        return None
    anchor_reachable = bool(projection.get("anchor_reachable"))
    if config.require_anchor_reachable and not anchor_reachable:
        return None
    distance_cells = _optional_nonnegative_int(projection.get("projection_distance_cells"))
    distance_m = _optional_nonnegative_float(projection.get("projection_distance_m"))
    contract_reasons = _trainability_distance_reject_reasons(
        distance_cells=distance_cells,
        distance_m=distance_m,
        config=config,
    )
    default_distance_contract_safe = not contract_reasons
    planner_exception_safe = _planner_validated_distance_exception_safe(
        distance_cells=distance_cells,
        distance_m=distance_m,
        config=config,
    )
    planner_validated_exception = bool(
        not default_distance_contract_safe and planner_exception_safe
    )
    if contract_reasons and not planner_validated_exception:
        return None

    execution_goal = GoalCandidate(
        cell=anchor,
        utility=goal.utility,
        reachable=True,
        experimental={
            **goal.experimental,
            "anchor_projection_source_action_index": source_action_index,
            "anchor_projection_policy_target_cell": [goal.cell[0], goal.cell[1]],
            "anchor_projection_execution_goal_cell": [anchor[0], anchor[1]],
            "anchor_projection_target_binding_mode": "same_action_execution_substitute",
        },
    )
    projected_result = planner.plan(
        PathPlanRequest(
            contract=contract,
            step_index=step_index,
            action_index=source_action_index,
            selected_goal=execution_goal,
            current_cell=current_cell,
            metadata={
                "anchor_projection_candidate_generation": True,
                "contract_aware_trainable_target_generation": True,
                "planner_validated_trainable_target_mining": (
                    config.planner_validated_trainable_target_mining
                ),
                "allow_planner_validated_distance_exception": (
                    config.allow_planner_validated_distance_exception
                ),
                "target_binding_mode": "same_action_execution_substitute",
                "source_action_index": source_action_index,
                "policy_target_cell": [goal.cell[0], goal.cell[1]],
                "execution_goal_cell": [anchor[0], anchor[1]],
            },
        )
    )
    candidate_generation = _anchor_projection_candidate_generation_payload(
        source_action_index=source_action_index,
        policy_target=goal.cell,
        execution_goal=anchor,
        projection=projection,
        distance_cells=distance_cells,
        distance_m=distance_m,
        anchor_reachable=anchor_reachable,
        target_binding_mode="same_action_execution_substitute",
        ppo_consumable_action=True,
        contract_safe=default_distance_contract_safe,
        default_distance_contract_safe=default_distance_contract_safe,
        planner_validated_distance_exception=planner_validated_exception,
        planner_validated_exception_safe=planner_exception_safe,
        default_distance_contract_reject_reasons=contract_reasons,
    )
    return PathCandidateEvaluation(
        action_index=source_action_index,
        cell=goal.cell,
        utility=goal.utility,
        result=projected_result,
        selection_goal=goal,
        source_action_index=source_action_index,
        candidate_generation=candidate_generation,
    )


def _projected_anchor_candidate_evaluation(
    *,
    contract: ModelExplorerContract,
    current_cell: tuple[int, int],
    step_index: int,
    source_action_index: int,
    synthetic_action_index: int,
    goal: GoalCandidate,
    source_result: PathPlanResult,
    planner: PathPlanningAdapter,
    config: AnchorProjectionCandidateConfig,
) -> PathCandidateEvaluation | None:
    if not config.enabled:
        return None
    source_feasibility = _platform_goal_feasibility(cell=goal.cell, result=source_result)
    if source_feasibility.get("classification") != "platform_inflated_goal_blocked":
        return None
    projection = source_feasibility.get("anchor_projection")
    projection = projection if isinstance(projection, dict) else {}
    anchor = _cell_pair(
        projection.get("projected_anchor_cell")
        or projection.get("nearest_inflated_passable_anchor")
        or source_feasibility.get("nearest_inflated_passable_anchor")
    )
    if anchor is None:
        return None
    anchor_reachable = bool(projection.get("anchor_reachable"))
    if config.require_anchor_reachable and not anchor_reachable:
        return None
    distance_cells = _optional_nonnegative_int(projection.get("projection_distance_cells"))
    distance_m = _optional_nonnegative_float(projection.get("projection_distance_m"))
    if (
        config.max_projection_distance_cells is not None
        and distance_cells is not None
        and distance_cells > config.max_projection_distance_cells
    ):
        return None
    if (
        config.max_projection_distance_m is not None
        and distance_m is not None
        and distance_m > config.max_projection_distance_m
    ):
        return None

    projected_goal = GoalCandidate(
        cell=anchor,
        utility=goal.utility,
        reachable=True,
        experimental={
            **goal.experimental,
            "anchor_projection_source_action_index": source_action_index,
            "anchor_projection_policy_target_cell": [goal.cell[0], goal.cell[1]],
            "anchor_projection_execution_goal_cell": [anchor[0], anchor[1]],
        },
    )
    projected_result = planner.plan(
        PathPlanRequest(
            contract=contract,
            step_index=step_index,
            action_index=synthetic_action_index,
            selected_goal=projected_goal,
            current_cell=current_cell,
            metadata={
                "anchor_projection_candidate_generation": True,
                "source_action_index": source_action_index,
                "policy_target_cell": [goal.cell[0], goal.cell[1]],
                "execution_goal_cell": [anchor[0], anchor[1]],
            },
        )
    )
    contract_reasons = _trainability_distance_reject_reasons(
        distance_cells=distance_cells,
        distance_m=distance_m,
        config=config,
    )
    candidate_generation = _anchor_projection_candidate_generation_payload(
        source_action_index=source_action_index,
        policy_target=goal.cell,
        execution_goal=anchor,
        projection=projection,
        distance_cells=distance_cells,
        distance_m=distance_m,
        anchor_reachable=anchor_reachable,
        target_binding_mode="synthetic_projection",
        ppo_consumable_action=False,
        contract_safe=not contract_reasons,
        default_distance_contract_safe=not contract_reasons,
        planner_validated_distance_exception=False,
        planner_validated_exception_safe=False,
        default_distance_contract_reject_reasons=contract_reasons,
        extra_reject_reasons=contract_reasons,
    )
    return PathCandidateEvaluation(
        action_index=synthetic_action_index,
        cell=anchor,
        utility=goal.utility,
        result=projected_result,
        selection_goal=projected_goal,
        source_action_index=source_action_index,
        candidate_generation=candidate_generation,
    )


__all__ = (
    "evaluate_candidate_paths",
    "_same_action_anchor_projection_candidate_evaluation",
    "_projected_anchor_candidate_evaluation",
)
