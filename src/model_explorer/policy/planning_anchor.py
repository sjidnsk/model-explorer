"""Anchor-projection candidate generation helpers."""

from __future__ import annotations

from collections import deque
from math import ceil, hypot
from typing import Any

from ..core.interfaces import GoalCandidate, ModelExplorerContract
from .planning_types import (
    AnchorProjectionCandidateConfig,
    PathCandidateEvaluation,
    PathPlanningAdapter,
    PathPlanRequest,
    PathPlanResult,
)
from .planning_utils import (
    _cell_pair,
    _grid_distance,
    _manhattan_distance,
    _optional_nonnegative_float,
    _optional_nonnegative_int,
    _reconstruct_path,
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


def anchor_projection_candidate_config_from_mapping(
    value: AnchorProjectionCandidateConfig | dict[str, Any] | None,
) -> AnchorProjectionCandidateConfig:
    if isinstance(value, AnchorProjectionCandidateConfig):
        return value
    if not isinstance(value, dict):
        return AnchorProjectionCandidateConfig()
    return AnchorProjectionCandidateConfig(
        enabled=bool(value.get("enabled", False)),
        max_projection_distance_cells=_optional_nonnegative_int(
            value.get("max_projection_distance_cells")
        ),
        max_projection_distance_m=_optional_nonnegative_float(
            value.get("max_projection_distance_m")
        ),
        require_anchor_reachable=bool(value.get("require_anchor_reachable", True)),
        source_selection_path_cost_bonus=_optional_nonnegative_float(
            value.get("source_selection_path_cost_bonus")
        )
        or 0.0,
        max_source_selection_path_cost_regression=_optional_nonnegative_float(
            value.get("max_source_selection_path_cost_regression")
        ),
        max_source_selection_risk_regression=_optional_nonnegative_float(
            value.get("max_source_selection_risk_regression")
        ),
        contract_aware_trainable_target_generation=bool(
            value.get("contract_aware_trainable_target_generation", False)
        ),
        prefer_contract_safe_trainable_targets=bool(
            value.get("prefer_contract_safe_trainable_targets", False)
        ),
        max_trainable_projection_distance_cells=_optional_nonnegative_int(
            value.get("max_trainable_projection_distance_cells")
        )
        if value.get("max_trainable_projection_distance_cells") is not None
        else 2,
        max_trainable_projection_distance_m=_optional_nonnegative_float(
            value.get("max_trainable_projection_distance_m")
        )
        if value.get("max_trainable_projection_distance_m") is not None
        else 1.0,
        planner_validated_trainable_target_mining=bool(
            value.get("planner_validated_trainable_target_mining", False)
        ),
        allow_planner_validated_distance_exception=bool(
            value.get("allow_planner_validated_distance_exception", False)
        ),
        max_planner_validated_distance_cells=_optional_nonnegative_int(
            value.get("max_planner_validated_distance_cells")
        )
        if value.get("max_planner_validated_distance_cells") is not None
        else 3,
        max_planner_validated_distance_m=_optional_nonnegative_float(
            value.get("max_planner_validated_distance_m")
        )
        if value.get("max_planner_validated_distance_m") is not None
        else 1.5,
    )


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
    from .planning_diagnostics import _platform_goal_feasibility

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
    from .planning_diagnostics import _platform_goal_feasibility

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


def _anchor_projection_candidate_generation_payload(
    *,
    source_action_index: int,
    policy_target: tuple[int, int],
    execution_goal: tuple[int, int],
    projection: dict[str, Any],
    distance_cells: int | None,
    distance_m: float | None,
    anchor_reachable: bool,
    target_binding_mode: str,
    ppo_consumable_action: bool,
    contract_safe: bool,
    default_distance_contract_safe: bool | None = None,
    planner_validated_distance_exception: bool = False,
    planner_validated_exception_safe: bool = False,
    default_distance_contract_reject_reasons: list[str] | None = None,
    extra_reject_reasons: list[str] | None = None,
) -> dict[str, Any]:
    reason_codes = list(extra_reject_reasons or [])
    default_distance_contract_safe = (
        bool(contract_safe)
        if default_distance_contract_safe is None
        else bool(default_distance_contract_safe)
    )
    eligible_if_selected = bool(contract_safe or planner_validated_exception_safe)
    trainability_status = "eligible_if_source_selected" if eligible_if_selected else "rejected"
    return {
        "schema_version": "anchor-projection-candidate/v1",
        "candidate_role": "projected_execution_target",
        "target_binding_mode": target_binding_mode,
        "source": "anchor_projection_candidate_generation",
        "source_action_index": source_action_index,
        "policy_target_cell": [policy_target[0], policy_target[1]],
        "execution_goal_cell": [execution_goal[0], execution_goal[1]],
        "projected_anchor_cell": [execution_goal[0], execution_goal[1]],
        "projection_distance_cells": distance_cells,
        "projection_distance_m": distance_m,
        "anchor_reachable": anchor_reachable,
        "nearest_inflated_passable_anchor": projection.get("nearest_inflated_passable_anchor"),
        "nearest_anchor_reachable": projection.get("nearest_anchor_reachable"),
        "nearest_anchor_distance_cells": projection.get("nearest_anchor_distance_cells"),
        "nearest_anchor_distance_m": projection.get("nearest_anchor_distance_m"),
        "anchor_selection_status": projection.get("anchor_selection_status"),
        "start_component_id": projection.get("start_component_id"),
        "target_component_id": projection.get("target_component_id"),
        "nearest_anchor_component_id": projection.get("nearest_anchor_component_id"),
        "projected_anchor_component_id": projection.get("projected_anchor_component_id"),
        "start_component_size": projection.get("start_component_size"),
        "target_component_size": projection.get("target_component_size"),
        "nearest_anchor_component_size": projection.get("nearest_anchor_component_size"),
        "projected_anchor_component_size": projection.get("projected_anchor_component_size"),
        "reachable_substitute_anchor_available": projection.get(
            "reachable_substitute_anchor_available",
            False,
        ),
        "reachable_substitute_anchor_count": projection.get("reachable_substitute_anchor_count", 0),
        "comparison_scope": "projected_target_anchor_contrast",
        "scope": "projected_target_anchor_contrast",
        "training_use": "not_positive_evidence",
        "sample_weight": 0.0,
        "reject_reason": "pending_source_selection",
        "source_selection_status": "pending_source_selection",
        "evidence_boundary": "source_candidate_pending_selection_not_audit_proxy",
        "audit_proxy_positive_evidence": False,
        "ppo_consumable_action": bool(ppo_consumable_action),
        "contract_safe": bool(contract_safe),
        "default_distance_contract_safe": default_distance_contract_safe,
        "default_distance_contract_reject_reasons": list(
            default_distance_contract_reject_reasons or []
        ),
        "planner_validated_distance_exception": bool(planner_validated_distance_exception),
        "planner_validated_exception_safe": bool(planner_validated_exception_safe),
        "planner_validated_trainable_target_mining": bool(
            planner_validated_distance_exception or planner_validated_exception_safe
        ),
        "trainability_gate": {
            "status": trainability_status,
            "reason_codes": reason_codes,
            "ppo_consumable_action": bool(ppo_consumable_action),
            "source_action_index": source_action_index,
            "policy_target_cell": [policy_target[0], policy_target[1]],
            "execution_goal_cell": [execution_goal[0], execution_goal[1]],
            "contract_safe": bool(contract_safe),
            "default_distance_contract_safe": default_distance_contract_safe,
            "planner_validated_distance_exception": bool(planner_validated_distance_exception),
            "planner_validated_exception_safe": bool(planner_validated_exception_safe),
        },
    }


def _trainability_distance_reject_reasons(
    *,
    distance_cells: int | None,
    distance_m: float | None,
    config: AnchorProjectionCandidateConfig,
) -> list[str]:
    reasons: list[str] = []
    if distance_cells is None:
        reasons.append("projection_distance_cells_missing")
    elif distance_cells > config.max_trainable_projection_distance_cells:
        reasons.append("projection_distance_cells_exceeds_contract")
    if distance_m is None:
        reasons.append("projection_distance_m_missing")
    elif distance_m > config.max_trainable_projection_distance_m:
        reasons.append("projection_distance_m_exceeds_contract")
    return reasons


def _planner_validated_distance_exception_safe(
    *,
    distance_cells: int | None,
    distance_m: float | None,
    config: AnchorProjectionCandidateConfig,
) -> bool:
    if (
        not config.planner_validated_trainable_target_mining
        or not config.allow_planner_validated_distance_exception
    ):
        return False
    if distance_cells is None or distance_m is None:
        return False
    return (
        distance_cells <= config.max_planner_validated_distance_cells
        and distance_m <= float(config.max_planner_validated_distance_m)
    )

def _anchor_projection_analysis(
    *,
    request_payload: dict[str, Any],
    inflated_mask: tuple[tuple[bool, ...], ...],
    target_cell: tuple[int, int],
    resolution: float,
) -> dict[str, Any]:
    nearest_anchor = _nearest_inflated_passable_anchor(inflated_mask, target_cell)
    nearest_route = _proxy_anchor_route_comparison(
        request_payload=request_payload,
        inflated_mask=inflated_mask,
        anchor=nearest_anchor,
        resolution=resolution,
    )
    labels, component_sizes = _connected_component_labels(inflated_mask)
    start = _cell_pair(request_payload.get("start"))
    start_component_id = _component_id_at(labels, start)
    target_component_id = _component_id_at(labels, target_cell)
    nearest_component_id = _component_id_at(labels, nearest_anchor)
    projected_anchor = nearest_anchor
    projected_component_id = nearest_component_id
    proxy_route_comparison = nearest_route
    nearest_reachable = bool(nearest_route.get("anchor_route_feasible"))
    reachable_substitute_count = 0
    reachable_substitute_available = False
    anchor_selection_status = "nearest_anchor_reachable" if nearest_reachable else "anchor_not_reachable"

    if nearest_anchor is None:
        anchor_selection_status = "no_inflated_passable_anchor"
    elif not nearest_reachable:
        substitute, reachable_substitute_count = _best_reachable_anchor_in_component(
            inflated_mask,
            target_cell=target_cell,
            start=start,
            start_component_id=start_component_id,
            component_labels=labels,
        )
        if substitute is not None:
            projected_anchor = substitute
            projected_component_id = _component_id_at(labels, projected_anchor)
            proxy_route_comparison = _proxy_anchor_route_comparison(
                request_payload=request_payload,
                inflated_mask=inflated_mask,
                anchor=projected_anchor,
                resolution=resolution,
            )
            reachable_substitute_available = True
            anchor_selection_status = "reachable_substitute_anchor_found"
        elif start_component_id is None:
            anchor_selection_status = "true_geometry_unreachable"
        else:
            anchor_selection_status = "true_geometry_unreachable"

    return {
        "nearest_inflated_passable_anchor": _cell_list(nearest_anchor),
        "projected_anchor_cell": _cell_list(projected_anchor),
        "nearest_anchor_distance_cells": _cell_manhattan_or_none(target_cell, nearest_anchor),
        "nearest_anchor_distance_m": _cell_distance_m_or_none(target_cell, nearest_anchor, resolution),
        "projection_distance_cells": _cell_manhattan_or_none(target_cell, projected_anchor),
        "projection_distance_m": _cell_distance_m_or_none(target_cell, projected_anchor, resolution),
        "nearest_anchor_reachable": nearest_reachable,
        "anchor_selection_status": anchor_selection_status,
        "start_component_id": start_component_id,
        "target_component_id": target_component_id,
        "nearest_anchor_component_id": nearest_component_id,
        "projected_anchor_component_id": projected_component_id,
        "start_component_size": _component_size(component_sizes, start_component_id),
        "target_component_size": _component_size(component_sizes, target_component_id),
        "nearest_anchor_component_size": _component_size(component_sizes, nearest_component_id),
        "projected_anchor_component_size": _component_size(component_sizes, projected_component_id),
        "reachable_substitute_anchor_available": reachable_substitute_available,
        "reachable_substitute_anchor_count": reachable_substitute_count,
        "proxy_route_comparison": proxy_route_comparison,
    }


def _connected_component_labels(
    mask: tuple[tuple[bool, ...], ...],
) -> tuple[tuple[tuple[int | None, ...], ...], dict[int, int]]:
    height = len(mask)
    width = len(mask[0]) if height else 0
    labels: list[list[int | None]] = [[None for _ in range(width)] for _ in range(height)]
    component_sizes: dict[int, int] = {}
    next_component_id = 0
    for y, row in enumerate(mask):
        for x, passable in enumerate(row):
            if not passable or labels[y][x] is not None:
                continue
            component_id = next_component_id
            next_component_id += 1
            frontier: deque[tuple[int, int]] = deque([(x, y)])
            labels[y][x] = component_id
            size = 0
            while frontier:
                current = frontier.popleft()
                size += 1
                for neighbor in _mask_neighbors(mask, current):
                    nx, ny = neighbor
                    if labels[ny][nx] is not None:
                        continue
                    labels[ny][nx] = component_id
                    frontier.append(neighbor)
            component_sizes[component_id] = size
    return tuple(tuple(row) for row in labels), component_sizes


def _best_reachable_anchor_in_component(
    mask: tuple[tuple[bool, ...], ...],
    *,
    target_cell: tuple[int, int],
    start: tuple[int, int] | None,
    start_component_id: int | None,
    component_labels: tuple[tuple[int | None, ...], ...],
) -> tuple[tuple[int, int] | None, int]:
    if start is None or start_component_id is None:
        return None, 0
    distances = _grid_distance_map(mask, start=start)
    candidates: list[tuple[int, float, int, int, int]] = []
    for y, row in enumerate(mask):
        for x, passable in enumerate(row):
            if not passable or _component_id_at(component_labels, (x, y)) != start_component_id:
                continue
            start_distance = distances.get((x, y))
            if start_distance is None:
                continue
            manhattan = abs(x - target_cell[0]) + abs(y - target_cell[1])
            euclidean = hypot(x - target_cell[0], y - target_cell[1])
            candidates.append((manhattan, euclidean, start_distance, y, x))
    if not candidates:
        return None, 0
    _, _, _, y, x = min(candidates)
    return (x, y), len(candidates)


def _grid_distance_map(
    mask: tuple[tuple[bool, ...], ...],
    *,
    start: tuple[int, int],
) -> dict[tuple[int, int], int]:
    if _mask_value(mask, start) is not True:
        return {}
    frontier: deque[tuple[int, int]] = deque([start])
    distances: dict[tuple[int, int], int] = {start: 0}
    while frontier:
        current = frontier.popleft()
        for neighbor in _mask_neighbors(mask, current):
            if neighbor in distances:
                continue
            distances[neighbor] = distances[current] + 1
            frontier.append(neighbor)
    return distances


def _component_id_at(
    labels: tuple[tuple[int | None, ...], ...],
    cell: tuple[int, int] | None,
) -> int | None:
    if cell is None:
        return None
    x, y = cell
    if y < 0 or y >= len(labels):
        return None
    if x < 0 or x >= len(labels[y]):
        return None
    return labels[y][x]


def _component_size(component_sizes: dict[int, int], component_id: int | None) -> int | None:
    if component_id is None:
        return None
    return component_sizes.get(component_id)


def _cell_list(cell: tuple[int, int] | None) -> list[int] | None:
    if cell is None:
        return None
    return [cell[0], cell[1]]


def _cell_manhattan_or_none(
    origin: tuple[int, int],
    cell: tuple[int, int] | None,
) -> int | None:
    if cell is None:
        return None
    return _manhattan_distance(origin, cell)


def _cell_distance_m_or_none(
    origin: tuple[int, int],
    cell: tuple[int, int] | None,
    resolution: float,
) -> float | None:
    if cell is None:
        return None
    return float(hypot(cell[0] - origin[0], cell[1] - origin[1]) * resolution)


def _anchor_projection_reject_reason(
    *,
    nearest_anchor: tuple[int, int] | None,
    anchor_reachable: bool,
    comparison_scope: str,
) -> str:
    if nearest_anchor is None:
        return "no_inflated_passable_anchor"
    if not anchor_reachable:
        return "anchor_not_reachable"
    if comparison_scope == "audit_proxy_anchor_not_same_cell":
        return "audit_proxy_scope_not_positive_evidence"
    return "not_positive_evidence"

def _inflated_passable_mask(
    mask: tuple[tuple[bool, ...], ...],
    *,
    resolution: float,
    footprint_radius_m: float | None,
) -> tuple[tuple[bool, ...], ...]:
    if footprint_radius_m is None or footprint_radius_m <= 0.0:
        return tuple(tuple(row) for row in mask)
    height = len(mask)
    width = len(mask[0]) if height else 0
    safe = [[bool(value) for value in row] for row in mask]
    radius_cells = int(ceil(footprint_radius_m / max(resolution, 1.0e-12)))
    for blocked_y, row in enumerate(mask):
        for blocked_x, passable in enumerate(row):
            if passable:
                continue
            min_y = max(0, blocked_y - radius_cells)
            max_y = min(height - 1, blocked_y + radius_cells)
            min_x = max(0, blocked_x - radius_cells)
            max_x = min(width - 1, blocked_x + radius_cells)
            for y in range(min_y, max_y + 1):
                for x in range(min_x, max_x + 1):
                    distance_m = hypot((x - blocked_x) * resolution, (y - blocked_y) * resolution)
                    if distance_m <= footprint_radius_m:
                        safe[y][x] = False
    return tuple(tuple(row) for row in safe)


def _nearest_inflated_passable_anchor(
    mask: tuple[tuple[bool, ...], ...],
    cell: tuple[int, int],
) -> tuple[int, int] | None:
    candidates: list[tuple[int, float, int, int]] = []
    for y, row in enumerate(mask):
        for x, passable in enumerate(row):
            if passable:
                manhattan = abs(x - cell[0]) + abs(y - cell[1])
                euclidean = hypot(x - cell[0], y - cell[1])
                candidates.append((manhattan, euclidean, y, x))
    if not candidates:
        return None
    _, _, y, x = min(candidates)
    return (x, y)


def _proxy_anchor_route_comparison(
    *,
    request_payload: dict[str, Any],
    inflated_mask: tuple[tuple[bool, ...], ...],
    anchor: tuple[int, int] | None,
    resolution: float,
) -> dict[str, Any]:
    if anchor is None:
        return _proxy_route_unavailable("no_inflated_passable_anchor")
    start = _cell_pair(request_payload.get("start"))
    if start is None:
        return _proxy_route_unavailable("missing_start")
    path = _grid_path(inflated_mask, start=start, goal=anchor)
    if path is None:
        return {
            "scope": "audit_proxy_anchor_not_same_cell",
            "anchor_route_feasible": False,
            "anchor_path_cost": None,
            "anchor_path_length_cells": None,
            "anchor_path_length_m": None,
            "same_cell_positive_evidence": False,
            "failure_reason": "anchor_unreachable",
        }
    cost = _path_cost_for_payload(request_payload.get("cost"), path)
    return {
        "scope": "audit_proxy_anchor_not_same_cell",
        "anchor_route_feasible": True,
        "anchor_path_cost": cost,
        "anchor_path_length_cells": max(len(path) - 1, 0),
        "anchor_path_length_m": float(max(len(path) - 1, 0) * resolution),
        "same_cell_positive_evidence": False,
        "failure_reason": None,
    }


def _proxy_route_unavailable(reason: str) -> dict[str, Any]:
    return {
        "scope": "audit_proxy_anchor_not_same_cell",
        "anchor_route_feasible": False,
        "anchor_path_cost": None,
        "anchor_path_length_cells": None,
        "anchor_path_length_m": None,
        "same_cell_positive_evidence": False,
        "failure_reason": reason,
    }


def _grid_path(
    mask: tuple[tuple[bool, ...], ...],
    *,
    start: tuple[int, int],
    goal: tuple[int, int],
) -> tuple[tuple[int, int], ...] | None:
    if _mask_value(mask, start) is not True or _mask_value(mask, goal) is not True:
        return None
    frontier: deque[tuple[int, int]] = deque([start])
    came_from: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    while frontier:
        current = frontier.popleft()
        if current == goal:
            return _reconstruct_path(came_from, current)
        for neighbor in _mask_neighbors(mask, current):
            if neighbor in came_from:
                continue
            came_from[neighbor] = current
            frontier.append(neighbor)
    return None


def _mask_neighbors(
    mask: tuple[tuple[bool, ...], ...],
    cell: tuple[int, int],
) -> tuple[tuple[int, int], ...]:
    x, y = cell
    candidates = ((x + 1, y), (x, y + 1), (x - 1, y), (x, y - 1))
    return tuple(candidate for candidate in candidates if _mask_value(mask, candidate) is True)


def _path_cost_for_payload(cost_payload: Any, path: tuple[tuple[int, int], ...]) -> float | None:
    if len(path) < 2:
        return 0.0
    total = 0.0
    for _, cell in zip(path[:-1], path[1:]):
        value = _grid_value(cost_payload, cell)
        if value is None:
            return None
        total += float(value)
    return float(total)


def _bool_grid(value: Any, *, width: int | None, height: int | None) -> tuple[tuple[bool, ...], ...] | None:
    if width is None or height is None or not isinstance(value, list):
        return None
    rows: list[tuple[bool, ...]] = []
    if len(value) != height:
        return None
    for row in value:
        if not isinstance(row, list) or len(row) != width:
            return None
        rows.append(tuple(bool(item) for item in row))
    return tuple(rows)


def _mask_value(mask: tuple[tuple[bool, ...], ...], cell: tuple[int, int]) -> bool | None:
    x, y = cell
    if y < 0 or y >= len(mask):
        return None
    if x < 0 or x >= len(mask[y]):
        return None
    return bool(mask[y][x])


def _grid_value(value: Any, cell: tuple[int, int]) -> float | None:
    x, y = cell
    if not isinstance(value, list) or y < 0 or y >= len(value):
        return None
    row = value[y]
    if not isinstance(row, list) or x < 0 or x >= len(row):
        return None
    try:
        return float(row[x])
    except (TypeError, ValueError):
        return None


__all__ = [name for name in globals() if not name.startswith("__")]
