"""Anchor projection configuration and diagnostic payload helpers."""

from __future__ import annotations

from typing import Any

from .planning_anchor_grid import (
    _best_reachable_anchor_in_component,
    _bool_grid,
    _cell_distance_m_or_none,
    _cell_list,
    _cell_manhattan_or_none,
    _component_id_at,
    _component_size,
    _connected_component_labels,
    _grid_path,
    _inflated_passable_mask,
    _nearest_inflated_passable_anchor,
    _path_cost_for_payload,
)
from .planning_types import AnchorProjectionCandidateConfig
from .planning_utils import (
    _cell_pair,
    _optional_nonnegative_float,
    _optional_nonnegative_int,
)


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


__all__ = (
    "anchor_projection_candidate_config_from_mapping",
    "_anchor_projection_candidate_generation_payload",
    "_trainability_distance_reject_reasons",
    "_planner_validated_distance_exception_safe",
    "_anchor_projection_analysis",
    "_anchor_projection_reject_reason",
    "_proxy_anchor_route_comparison",
    "_proxy_route_unavailable",
)
