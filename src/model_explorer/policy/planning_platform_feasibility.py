"""Platform-footprint feasibility helpers for planning diagnostics."""

from __future__ import annotations

from typing import Any

from .planning_anchor_grid import _bool_grid, _inflated_passable_mask, _mask_value
from .planning_anchor_projection import (
    _anchor_projection_analysis,
    _anchor_projection_reject_reason,
    _proxy_anchor_route_comparison,
    _proxy_route_unavailable,
)
from .planning_types import PathPlanResult
from .planning_utils import (
    _cell_pair,
    _optional_nonnegative_float,
    _optional_nonnegative_int,
    _optional_positive_float,
    _positive_float,
    _positive_int,
)


def _platform_goal_feasibility(*, cell: tuple[int, int], result: PathPlanResult) -> dict[str, Any]:
    request_payload = result.metadata.get("request_payload")
    request_payload = request_payload if isinstance(request_payload, dict) else {}
    diagnostics = result.metadata.get("diagnostics")
    diagnostics = diagnostics if isinstance(diagnostics, dict) else {}
    grid = request_payload.get("grid") if isinstance(request_payload.get("grid"), dict) else {}
    width = _positive_int(grid.get("width"))
    height = _positive_int(grid.get("height"))
    resolution = _positive_float(grid.get("resolution"), default=1.0)
    mask = _bool_grid(request_payload.get("passable_mask"), width=width, height=height)
    footprint_radius_m = _optional_positive_float(diagnostics.get("footprint_radius_m"))
    if width is None or height is None or mask is None:
        return _platform_goal_feasibility_payload(
            classification="unknown_contract_mismatch",
            cell=cell,
            contract_reachable=True,
            original_passable=None,
            inflated_passable=None,
            footprint_radius_m=footprint_radius_m,
            nearest_anchor=None,
            anchor_distance_cells=None,
            anchor_distance_m=None,
            proxy_route_comparison=_proxy_route_unavailable("missing_request_passable_mask"),
        )

    original_passable = _mask_value(mask, cell)
    inflated_mask = _inflated_passable_mask(
        mask,
        resolution=resolution,
        footprint_radius_m=footprint_radius_m,
    )
    inflated_passable = _mask_value(inflated_mask, cell)
    classification = _platform_goal_classification(
        original_passable=original_passable,
        inflated_passable=inflated_passable,
    )
    anchor_projection_analysis = (
        _anchor_projection_analysis(
            request_payload=request_payload,
            inflated_mask=inflated_mask,
            target_cell=cell,
            resolution=resolution,
        )
        if classification == "platform_inflated_goal_blocked"
        else {}
    )
    nearest_anchor = _cell_pair(anchor_projection_analysis.get("nearest_inflated_passable_anchor"))
    anchor_distance_cells = _optional_nonnegative_int(
        anchor_projection_analysis.get("nearest_anchor_distance_cells")
    )
    anchor_distance_m = _optional_nonnegative_float(
        anchor_projection_analysis.get("nearest_anchor_distance_m")
    )
    proxy_route_comparison = anchor_projection_analysis.get("proxy_route_comparison")
    proxy_route_comparison = (
        proxy_route_comparison
        if isinstance(proxy_route_comparison, dict)
        else _proxy_anchor_route_comparison(
            request_payload=request_payload,
            inflated_mask=inflated_mask,
            anchor=nearest_anchor,
            resolution=resolution,
        )
    )
    return _platform_goal_feasibility_payload(
        classification=classification,
        cell=cell,
        contract_reachable=True,
        original_passable=original_passable,
        inflated_passable=inflated_passable,
        footprint_radius_m=footprint_radius_m,
        nearest_anchor=nearest_anchor,
        anchor_distance_cells=anchor_distance_cells,
        anchor_distance_m=anchor_distance_m,
        proxy_route_comparison=proxy_route_comparison,
        anchor_projection_analysis=anchor_projection_analysis,
    )


def _platform_goal_feasibility_payload(
    *,
    classification: str,
    cell: tuple[int, int],
    contract_reachable: bool,
    original_passable: bool | None,
    inflated_passable: bool | None,
    footprint_radius_m: float | None,
    nearest_anchor: tuple[int, int] | None,
    anchor_distance_cells: int | None,
    anchor_distance_m: float | None,
    proxy_route_comparison: dict[str, Any],
    anchor_projection_analysis: dict[str, Any] | None = None,
) -> dict[str, Any]:
    anchor_projection_analysis = (
        anchor_projection_analysis if isinstance(anchor_projection_analysis, dict) else {}
    )
    anchor_payload = None if nearest_anchor is None else [nearest_anchor[0], nearest_anchor[1]]
    projected_anchor = _cell_pair(anchor_projection_analysis.get("projected_anchor_cell")) or nearest_anchor
    projected_anchor_payload = (
        None if projected_anchor is None else [projected_anchor[0], projected_anchor[1]]
    )
    projection_distance_cells = _optional_nonnegative_int(
        anchor_projection_analysis.get("projection_distance_cells")
    )
    if projection_distance_cells is None:
        projection_distance_cells = anchor_distance_cells
    projection_distance_m = _optional_nonnegative_float(
        anchor_projection_analysis.get("projection_distance_m")
    )
    if projection_distance_m is None:
        projection_distance_m = anchor_distance_m
    same_cell_positive_evidence = bool(proxy_route_comparison.get("same_cell_positive_evidence"))
    anchor_reachable = bool(proxy_route_comparison.get("anchor_route_feasible"))
    comparison_scope = str(proxy_route_comparison.get("scope") or "unavailable")
    reject_reason = _anchor_projection_reject_reason(
        nearest_anchor=nearest_anchor,
        anchor_reachable=anchor_reachable,
        comparison_scope=comparison_scope,
    )
    return {
        "schema_version": "platform-goal-feasibility/v1",
        "cell": [cell[0], cell[1]],
        "policy_target_cell": [cell[0], cell[1]],
        "execution_goal_cell": [cell[0], cell[1]] if inflated_passable is True else None,
        "contract_reachable": bool(contract_reachable),
        "original_passable": original_passable,
        "inflated_passable": inflated_passable,
        "blocked_by_platform_footprint": bool(
            original_passable is True and inflated_passable is False
        ),
        "footprint_radius_m": footprint_radius_m,
        "nearest_inflated_passable_anchor": anchor_payload,
        "anchor_distance_cells": anchor_distance_cells,
        "anchor_distance_m": anchor_distance_m,
        "anchor_projection": {
            "nearest_inflated_passable_anchor": anchor_payload,
            "projected_anchor_cell": projected_anchor_payload,
            "projection_distance_cells": projection_distance_cells,
            "projection_distance_m": projection_distance_m,
            "nearest_anchor_distance_cells": anchor_distance_cells,
            "nearest_anchor_distance_m": anchor_distance_m,
            "anchor_reachable": anchor_reachable,
            "nearest_anchor_reachable": bool(
                anchor_projection_analysis.get("nearest_anchor_reachable", anchor_reachable)
            ),
            "anchor_selection_status": anchor_projection_analysis.get("anchor_selection_status"),
            "start_component_id": anchor_projection_analysis.get("start_component_id"),
            "target_component_id": anchor_projection_analysis.get("target_component_id"),
            "nearest_anchor_component_id": anchor_projection_analysis.get(
                "nearest_anchor_component_id"
            ),
            "projected_anchor_component_id": anchor_projection_analysis.get(
                "projected_anchor_component_id"
            ),
            "start_component_size": anchor_projection_analysis.get("start_component_size"),
            "target_component_size": anchor_projection_analysis.get("target_component_size"),
            "nearest_anchor_component_size": anchor_projection_analysis.get(
                "nearest_anchor_component_size"
            ),
            "projected_anchor_component_size": anchor_projection_analysis.get(
                "projected_anchor_component_size"
            ),
            "reachable_substitute_anchor_available": bool(
                anchor_projection_analysis.get("reachable_substitute_anchor_available", False)
            ),
            "reachable_substitute_anchor_count": int(
                anchor_projection_analysis.get("reachable_substitute_anchor_count", 0) or 0
            ),
            "comparison_scope": comparison_scope,
            "scope": comparison_scope,
            "same_cell_positive_evidence": same_cell_positive_evidence,
            "training_use": "not_positive_evidence",
            "sample_weight": 0.0,
            "reject_reason": reject_reason,
            "source_selection_status": "not_source_candidate",
            "evidence_boundary": "audit_projection_not_same_cell_positive_evidence",
            "audit_proxy_positive_evidence": False,
        },
        "classification": classification,
        "proxy_route_comparison": proxy_route_comparison,
    }


def _with_projected_anchor_feasibility(
    feasibility: dict[str, Any],
    *,
    candidate_generation: dict[str, Any],
) -> dict[str, Any]:
    payload = dict(feasibility)
    policy_target_cell = _cell_pair(candidate_generation.get("policy_target_cell"))
    execution_goal_cell = _cell_pair(candidate_generation.get("execution_goal_cell"))
    projected_anchor_cell = _cell_pair(candidate_generation.get("projected_anchor_cell"))
    if policy_target_cell is not None:
        payload["policy_target_cell"] = [policy_target_cell[0], policy_target_cell[1]]
    if execution_goal_cell is not None:
        payload["execution_goal_cell"] = [execution_goal_cell[0], execution_goal_cell[1]]
    projection = dict(payload.get("anchor_projection") if isinstance(payload.get("anchor_projection"), dict) else {})
    if projected_anchor_cell is not None:
        projection["projected_anchor_cell"] = [projected_anchor_cell[0], projected_anchor_cell[1]]
    projection.update(
        {
            "projection_distance_cells": candidate_generation.get("projection_distance_cells"),
            "projection_distance_m": candidate_generation.get("projection_distance_m"),
            "anchor_reachable": bool(candidate_generation.get("anchor_reachable")),
            "comparison_scope": str(candidate_generation.get("comparison_scope") or "projected_target_anchor_contrast"),
            "scope": str(candidate_generation.get("scope") or "projected_target_anchor_contrast"),
            "same_cell_positive_evidence": False,
            "training_use": str(candidate_generation.get("training_use") or "not_positive_evidence"),
            "sample_weight": float(candidate_generation.get("sample_weight") or 0.0),
            "reject_reason": candidate_generation.get("reject_reason"),
            "source_selection_status": candidate_generation.get("source_selection_status"),
            "evidence_boundary": candidate_generation.get(
                "evidence_boundary",
                "source_candidate_pending_selection_not_audit_proxy",
            ),
            "audit_proxy_positive_evidence": False,
            "target_binding_mode": candidate_generation.get("target_binding_mode"),
            "ppo_consumable_action": bool(candidate_generation.get("ppo_consumable_action", False)),
            "contract_safe": bool(candidate_generation.get("contract_safe", False)),
            "default_distance_contract_safe": bool(
                candidate_generation.get("default_distance_contract_safe", False)
            ),
            "default_distance_contract_reject_reasons": candidate_generation.get(
                "default_distance_contract_reject_reasons",
                [],
            ),
            "planner_validated_distance_exception": bool(
                candidate_generation.get("planner_validated_distance_exception", False)
            ),
            "planner_validated_exception_safe": bool(
                candidate_generation.get("planner_validated_exception_safe", False)
            ),
            "planner_validated_trainable_target_mining": bool(
                candidate_generation.get("planner_validated_trainable_target_mining", False)
            ),
            "trainability_gate": candidate_generation.get("trainability_gate"),
        }
    )
    payload["anchor_projection"] = projection
    return payload


def _platform_goal_classification(
    *,
    original_passable: bool | None,
    inflated_passable: bool | None,
) -> str:
    if original_passable is None or inflated_passable is None:
        return "out_of_bounds"
    if not original_passable:
        return "original_goal_blocked"
    if not inflated_passable:
        return "platform_inflated_goal_blocked"
    return "goal_passable"


def _platform_goal_contract_mismatch(feasibility: dict[str, Any]) -> bool:
    return bool(feasibility.get("contract_reachable")) and feasibility.get("classification") in {
        "platform_inflated_goal_blocked",
        "original_goal_blocked",
        "out_of_bounds",
        "unknown_contract_mismatch",
    }


__all__ = (
    "_platform_goal_feasibility",
    "_platform_goal_feasibility_payload",
    "_with_projected_anchor_feasibility",
    "_platform_goal_classification",
    "_platform_goal_contract_mismatch",
)
