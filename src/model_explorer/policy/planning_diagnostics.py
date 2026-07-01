"""Planning diagnostic summary and interpretation helpers."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Any

from .planning_anchor import (
    _anchor_projection_analysis,
    _anchor_projection_reject_reason,
    _bool_grid,
    _inflated_passable_mask,
    _mask_value,
    _proxy_anchor_route_comparison,
    _proxy_route_unavailable,
)
from .planning_utils import (
    _cell_pair,
    _optional_nonnegative_float,
    _optional_nonnegative_int,
    _optional_positive_float,
    _positive_float,
    _positive_int,
)


def path_feedback_summary(evaluations: Sequence[PathCandidateEvaluation]) -> dict[str, Any]:
    items = []
    for evaluation in evaluations:
        item = evaluation.to_dict()
        item["diagnostic_interpretation"] = _candidate_diagnostic_interpretation(item)
        items.append(item)
    failure_reasons = [
        item["failure_reason"] for item in items if item.get("failure_reason") is not None
    ]
    replan_count = sum(1 for item in items if item["replan_required"])
    feasible_items = [item for item in items if item["reachable"]]
    platform_counts = Counter(
        item.get("platform_goal_feasibility", {}).get("classification", "unavailable")
        for item in items
    )
    return {
        "candidate_count": len(items),
        "reachable_count": len(feasible_items),
        "failure_count": len(items) - len(feasible_items),
        "replan_count": replan_count,
        "failure_reasons": failure_reasons,
        "best_by_path_cost": min(
            feasible_items,
            key=lambda item: (float(item["path_cost"]), -float(item["utility"])),
            default=None,
        ),
        "candidates": items,
        "platform_goal_feasibility_class_counts": dict(sorted(platform_counts.items())),
        "platform_goal_contract_mismatch_count": sum(
            1
            for item in items
            if _platform_goal_contract_mismatch(
                item.get("platform_goal_feasibility")
                if isinstance(item.get("platform_goal_feasibility"), dict)
                else {}
            )
        ),
        "platform_goal_anchor_available_count": sum(
            1
            for item in items
            if isinstance(item.get("platform_goal_feasibility"), dict)
            and item["platform_goal_feasibility"].get("nearest_inflated_passable_anchor") is not None
        ),
        "platform_goal_unresolved_count": platform_counts["unknown_contract_mismatch"],
    }

def _postprocess_summary(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    tracking_safety = value.get("tracking_safety_report")
    return {
        "fallback_status": value.get("fallback_status"),
        "has_smoothed_path": bool(value.get("smoothed_path")),
        "tracking_safety_violation_count": (
            None
            if not isinstance(tracking_safety, dict)
            else int(tracking_safety.get("violation_count", 0) or 0)
        ),
    }


def _optimization_summary(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    return {
        "solver_status": value.get("solver_status"),
        "fallback_status": value.get("fallback_status"),
        "has_optimized_path": bool(value.get("optimized_path")),
    }


def _planning_backend_summary(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    sampled = value.get("sampled_region_path_report")
    return {
        "requested_backend": value.get("requested_backend"),
        "selected_backend": value.get("selected_backend"),
        "status": value.get("status"),
        "fallback_reason": value.get("fallback_reason"),
        "segment_count": value.get("segment_count"),
        "comparison": value.get("comparison") if isinstance(value.get("comparison"), dict) else {},
        "execution_alignment": (
            value.get("execution_alignment")
            if isinstance(value.get("execution_alignment"), dict)
            else {}
        ),
        "region_graph_candidate": (
            value.get("region_graph_candidate") if isinstance(value.get("region_graph_candidate"), dict) else {}
        ),
        "sampled_region_path": sampled if isinstance(sampled, dict) else {},
    }


def _region_graph_summary(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    quality = value.get("quality_metrics")
    quality = quality if isinstance(quality, dict) else {}
    quality_summary = {
        "requested_region_source": quality.get("requested_region_source"),
        "graph_source": quality.get("graph_source", value.get("region_source")),
        "fallback_ratio": quality.get("fallback_ratio"),
        "connected_component_count": quality.get("connected_component_count"),
        "fallback_reason": quality.get("fallback_reason") or value.get("failure_reason"),
        "start_goal_connected": quality.get("start_goal_connected"),
    }
    return {
        "status": value.get("status"),
        "region_source": value.get("region_source"),
        "vertex_count": value.get("vertex_count"),
        "edge_count": value.get("edge_count"),
        **quality_summary,
        "quality_metrics": quality_summary,
        "fallback_used": value.get("fallback_used"),
    }


def _iris_region_summary(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    return {
        "backend": value.get("backend"),
        "status": value.get("status"),
        "region_count": value.get("region_count"),
        "fallback_used": value.get("fallback_used"),
        "failure_status": value.get("failure_status"),
        "failure_reason": value.get("failure_reason"),
    }


def _convex_region_route_report(route: dict[str, Any]) -> dict[str, Any] | None:
    if "convex_region_count" not in route and "gcs_ready" not in route:
        return None
    return {
        "schema_version": route.get("convex_region_sequence_schema_version"),
        "region_count": route.get("convex_region_count"),
        "backend": route.get("convex_region_backend"),
        "fallback_used": route.get("convex_region_fallback_used"),
        "coverage_status": route.get("convex_region_coverage_status"),
        "start_contained": route.get("convex_region_start_contained"),
        "goal_contained": route.get("convex_region_goal_contained"),
        "adjacent_overlap_count": route.get("convex_region_adjacent_overlap_count"),
        "portal_count": route.get("convex_region_portal_count"),
        "blocked_cell_violation_count": route.get("convex_region_blocked_cell_violation_count"),
        "pydrake_available": route.get("convex_region_pydrake_available"),
        "gcs_ready": route.get("gcs_ready"),
        "gcs_ready_reason": route.get("gcs_ready_reason"),
        "sequence": route.get("convex_region_sequence") if isinstance(route.get("convex_region_sequence"), list) else [],
    }


def _convex_region_summary(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    return {
        "schema_version": value.get("schema_version"),
        "region_count": value.get("region_count"),
        "backend": value.get("backend"),
        "fallback_used": value.get("fallback_used"),
        "coverage_status": value.get("coverage_status"),
        "start_contained": value.get("start_contained"),
        "goal_contained": value.get("goal_contained"),
        "adjacent_overlap_count": value.get("adjacent_overlap_count"),
        "portal_count": value.get("portal_count"),
        "blocked_cell_violation_count": value.get("blocked_cell_violation_count"),
        "pydrake_available": value.get("pydrake_available"),
        "gcs_ready": value.get("gcs_ready"),
        "gcs_ready_reason": value.get("gcs_ready_reason"),
    }


def _gcs_trajectory_route_report(route: dict[str, Any]) -> dict[str, Any] | None:
    if "gcs_trajectory_report_schema_version" not in route:
        return None
    return {
        "schema_version": route.get("gcs_trajectory_report_schema_version"),
        "attempted": route.get("gcs_trajectory_attempted"),
        "success": route.get("gcs_trajectory_success"),
        "backend": route.get("gcs_trajectory_backend"),
        "result_status": route.get("gcs_trajectory_result_status"),
        "reason": route.get("gcs_trajectory_reason"),
        "sample_count": route.get("gcs_trajectory_sample_count"),
        "collision_count": route.get("gcs_trajectory_collision_count"),
        "path_length": route.get("gcs_trajectory_path_length"),
        "region_count": route.get("gcs_trajectory_region_count"),
        "sampled_points": (
            route.get("gcs_trajectory_sampled_points")
            if isinstance(route.get("gcs_trajectory_sampled_points"), list)
            else []
        ),
        "constraint_summary": (
            route.get("gcs_trajectory_constraint_summary")
            if isinstance(route.get("gcs_trajectory_constraint_summary"), dict)
            else {}
        ),
        "cost_summary": (
            route.get("gcs_trajectory_cost_summary")
            if isinstance(route.get("gcs_trajectory_cost_summary"), dict)
            else {}
        ),
    }


def _gcs_trajectory_summary(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    return {
        "schema_version": value.get("schema_version"),
        "attempted": value.get("attempted"),
        "success": value.get("success"),
        "backend": value.get("backend"),
        "result_status": value.get("result_status"),
        "reason": value.get("reason"),
        "sample_count": value.get("sample_count"),
        "collision_count": value.get("collision_count"),
        "path_length": value.get("path_length"),
        "region_count": value.get("region_count"),
        "constraint_summary": (
            value.get("constraint_summary") if isinstance(value.get("constraint_summary"), dict) else {}
        ),
        "cost_summary": value.get("cost_summary") if isinstance(value.get("cost_summary"), dict) else {},
    }


def _gcs_candidate_route_report(route: dict[str, Any]) -> dict[str, Any] | None:
    if "gcs_candidate_report_schema_version" not in route:
        return None
    return {
        "schema_version": route.get("gcs_candidate_report_schema_version"),
        "attempted": route.get("gcs_candidate_attempted"),
        "available": route.get("gcs_candidate_available"),
        "selected": route.get("gcs_candidate_selected"),
        "selection_reason": route.get("gcs_candidate_selection_reason"),
        "fallback_reason": route.get("gcs_candidate_fallback_reason"),
        "path_length": route.get("gcs_candidate_path_length"),
        "path_cost": route.get("gcs_candidate_path_cost"),
        "collision_count": route.get("gcs_candidate_collision_count"),
        "high_cost_exposure": route.get("gcs_candidate_high_cost_exposure"),
        "baseline_overlap_ratio": route.get("gcs_candidate_baseline_overlap_ratio"),
        "cost_delta_vs_baseline": route.get("gcs_candidate_cost_delta_vs_baseline"),
        "cost_delta_vs_postprocess": route.get("gcs_candidate_cost_delta_vs_postprocess"),
        "cost_summary": (
            route.get("gcs_candidate_cost_summary")
            if isinstance(route.get("gcs_candidate_cost_summary"), dict)
            else {}
        ),
    }


def _gcs_candidate_summary(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    return {
        "schema_version": value.get("schema_version"),
        "attempted": value.get("attempted"),
        "available": value.get("available"),
        "selected": value.get("selected"),
        "selection_reason": value.get("selection_reason"),
        "fallback_reason": value.get("fallback_reason"),
        "path_length": value.get("path_length"),
        "path_cost": value.get("path_cost"),
        "collision_count": value.get("collision_count"),
        "high_cost_exposure": value.get("high_cost_exposure"),
        "baseline_overlap_ratio": value.get("baseline_overlap_ratio"),
        "cost_delta_vs_baseline": value.get("cost_delta_vs_baseline"),
        "cost_delta_vs_postprocess": value.get("cost_delta_vs_postprocess"),
        "cost_summary": value.get("cost_summary") if isinstance(value.get("cost_summary"), dict) else {},
    }


def _gcs_motion_feasibility_route_report(route: dict[str, Any]) -> dict[str, Any] | None:
    if "gcs_motion_feasibility_report_schema_version" not in route:
        return None
    return {
        "schema_version": route.get("gcs_motion_feasibility_report_schema_version"),
        "evaluated": route.get("gcs_motion_feasibility_evaluated"),
        "trajectory_source": route.get("gcs_motion_feasibility_trajectory_source"),
        "motion_model": route.get("gcs_motion_feasibility_motion_model"),
        "feasibility_status": route.get("gcs_motion_feasibility_feasibility_status"),
        "fallback_reason": route.get("gcs_motion_feasibility_fallback_reason"),
        "min_turning_radius_m": route.get("gcs_motion_feasibility_min_turning_radius_m"),
        "max_heading_change_deg": route.get("gcs_motion_feasibility_max_heading_change_deg"),
        "curvature_violation_count": route.get("gcs_motion_feasibility_curvature_violation_count"),
        "heading_violation_count": route.get("gcs_motion_feasibility_heading_violation_count"),
        "violation_indices": (
            route.get("gcs_motion_feasibility_violation_indices")
            if isinstance(route.get("gcs_motion_feasibility_violation_indices"), list)
            else []
        ),
        "sample_count": route.get("gcs_motion_feasibility_sample_count"),
        "path_length": route.get("gcs_motion_feasibility_path_length"),
        "constraint_summary": (
            route.get("gcs_motion_feasibility_constraint_summary")
            if isinstance(route.get("gcs_motion_feasibility_constraint_summary"), dict)
            else {}
        ),
    }


def _gcs_motion_feasibility_summary(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    return {
        "schema_version": value.get("schema_version"),
        "evaluated": value.get("evaluated"),
        "trajectory_source": value.get("trajectory_source"),
        "motion_model": value.get("motion_model"),
        "feasibility_status": value.get("feasibility_status"),
        "fallback_reason": value.get("fallback_reason"),
        "min_turning_radius_m": value.get("min_turning_radius_m"),
        "max_heading_change_deg": value.get("max_heading_change_deg"),
        "curvature_violation_count": value.get("curvature_violation_count"),
        "heading_violation_count": value.get("heading_violation_count"),
        "violation_indices": value.get("violation_indices") if isinstance(value.get("violation_indices"), list) else [],
        "sample_count": value.get("sample_count"),
        "path_length": value.get("path_length"),
        "constraint_summary": value.get("constraint_summary") if isinstance(value.get("constraint_summary"), dict) else {},
    }


def _gcs_curvature_constrained_candidate_route_report(route: dict[str, Any]) -> dict[str, Any] | None:
    if "gcs_curvature_constrained_report_schema_version" not in route:
        return None
    return {
        "schema_version": route.get("gcs_curvature_constrained_report_schema_version"),
        "attempted": route.get("gcs_curvature_constrained_attempted"),
        "available": route.get("gcs_curvature_constrained_available"),
        "selected": route.get("gcs_curvature_constrained_selected"),
        "repair_success": route.get("gcs_curvature_constrained_repair_success"),
        "source": route.get("gcs_curvature_constrained_source"),
        "repair_strategy": route.get("gcs_curvature_constrained_repair_strategy"),
        "status_before": route.get("gcs_curvature_constrained_status_before"),
        "status_after": route.get("gcs_curvature_constrained_status_after"),
        "fallback_reason": route.get("gcs_curvature_constrained_fallback_reason"),
        "curvature_violation_count_before": route.get(
            "gcs_curvature_constrained_curvature_violation_count_before"
        ),
        "curvature_violation_count_after": route.get(
            "gcs_curvature_constrained_curvature_violation_count_after"
        ),
        "heading_violation_count_before": route.get(
            "gcs_curvature_constrained_heading_violation_count_before"
        ),
        "heading_violation_count_after": route.get(
            "gcs_curvature_constrained_heading_violation_count_after"
        ),
        "violation_indices_before": (
            route.get("gcs_curvature_constrained_violation_indices_before")
            if isinstance(route.get("gcs_curvature_constrained_violation_indices_before"), list)
            else []
        ),
        "violation_indices_after": (
            route.get("gcs_curvature_constrained_violation_indices_after")
            if isinstance(route.get("gcs_curvature_constrained_violation_indices_after"), list)
            else []
        ),
        "region_containment_violation_count": route.get(
            "gcs_curvature_constrained_region_containment_violation_count"
        ),
        "collision_count": route.get("gcs_curvature_constrained_collision_count"),
        "path_length": route.get("gcs_curvature_constrained_path_length"),
        "path_cost": route.get("gcs_curvature_constrained_path_cost"),
        "cost_delta_vs_baseline": route.get("gcs_curvature_constrained_cost_delta_vs_baseline"),
        "constraint_summary": (
            route.get("gcs_curvature_constrained_constraint_summary")
            if isinstance(route.get("gcs_curvature_constrained_constraint_summary"), dict)
            else {}
        ),
    }


def _gcs_curvature_constrained_candidate_summary(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    return {
        "schema_version": value.get("schema_version"),
        "attempted": value.get("attempted"),
        "available": value.get("available"),
        "selected": value.get("selected"),
        "repair_success": value.get("repair_success"),
        "source": value.get("source"),
        "repair_strategy": value.get("repair_strategy"),
        "status_before": value.get("status_before"),
        "status_after": value.get("status_after"),
        "fallback_reason": value.get("fallback_reason"),
        "curvature_violation_count_before": value.get("curvature_violation_count_before"),
        "curvature_violation_count_after": value.get("curvature_violation_count_after"),
        "heading_violation_count_before": value.get("heading_violation_count_before"),
        "heading_violation_count_after": value.get("heading_violation_count_after"),
        "violation_indices_before": (
            value.get("violation_indices_before")
            if isinstance(value.get("violation_indices_before"), list)
            else []
        ),
        "violation_indices_after": (
            value.get("violation_indices_after")
            if isinstance(value.get("violation_indices_after"), list)
            else []
        ),
        "region_containment_violation_count": value.get("region_containment_violation_count"),
        "collision_count": value.get("collision_count"),
        "path_length": value.get("path_length"),
        "path_cost": value.get("path_cost"),
        "cost_delta_vs_baseline": value.get("cost_delta_vs_baseline"),
        "constraint_summary": value.get("constraint_summary") if isinstance(value.get("constraint_summary"), dict) else {},
    }


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

def _input_source_summary(value: Any) -> dict[str, Any]:
    metadata = value.get("metadata") if isinstance(value, dict) else {}
    metadata = metadata if isinstance(metadata, dict) else {}
    cost_source = metadata.get("cost_source")
    passable_mask_source = metadata.get("passable_mask_source")
    return {
        "cost_source": cost_source,
        "passable_mask_source": passable_mask_source,
        "open_grid_fallback_used": (
            cost_source == "open_grid_fallback"
            or passable_mask_source == "open_grid_fallback"
        ),
    }


def _candidate_diagnostic_interpretation(item: dict[str, Any]) -> dict[str, Any]:
    flags: list[str] = []
    if item.get("failure_reason") is not None or not bool(item.get("reachable")):
        flags.append("path_planning_failure")
    if bool(item.get("replan_required")):
        flags.append("replan_required")
    if bool(item.get("open_grid_fallback_used")):
        flags.append("open_grid_fallback")

    optimization = item.get("trajectory_optimization")
    if isinstance(optimization, dict):
        fallback = optimization.get("fallback_status")
        if isinstance(fallback, str) and fallback not in {"ok", "not_needed", "none"}:
            flags.append("trajectory_optimization_fallback")

    postprocess = item.get("postprocess")
    if isinstance(postprocess, dict) and int(postprocess.get("tracking_safety_violation_count") or 0) > 0:
        flags.append("tracking_safety_violation")

    iris = item.get("iris_region")
    iris_status = None
    iris_fallback_used = False
    if isinstance(iris, dict):
        iris_status = iris.get("status")
        iris_fallback_used = bool(iris.get("fallback_used"))
        if iris_fallback_used:
            flags.append("iris_fallback")
        if iris_status == "failed":
            flags.append("iris_failure")

    graph = item.get("region_graph")
    graph_source = None
    graph_fallback_used = False
    graph_connected = None
    if isinstance(graph, dict):
        graph_source = graph.get("graph_source") or graph.get("region_source")
        graph_fallback_used = bool(graph.get("fallback_used"))
        graph_connected = graph.get("start_goal_connected")
        if graph_fallback_used:
            flags.append("region_graph_fallback")
        if graph_connected is False:
            flags.append("region_graph_disconnected")

    sampled_status = None
    sampled_fallback_reason = None
    planning_backend = item.get("planning_backend")
    if isinstance(planning_backend, dict):
        sampled = planning_backend.get("sampled_region_path")
        if isinstance(sampled, dict) and sampled:
            sampled_status = sampled.get("status")
            sampled_fallback_reason = sampled.get("fallback_reason")
            if sampled_status == "fallback" or sampled_fallback_reason:
                flags.append("sampled_region_path_fallback")

    ordered_flags = _dedupe(flags)
    return {
        "primary_source": _primary_diagnostic_source(ordered_flags),
        "diagnostic_flags": ordered_flags,
        "iris_status": iris_status,
        "iris_fallback_used": iris_fallback_used,
        "region_graph_source": graph_source,
        "region_graph_fallback_used": graph_fallback_used,
        "region_graph_start_goal_connected": graph_connected,
        "sampled_region_path_status": sampled_status,
        "sampled_region_path_fallback_reason": sampled_fallback_reason,
        "open_grid_fallback_used": bool(item.get("open_grid_fallback_used")),
    }


def _primary_diagnostic_source(flags: Sequence[str]) -> str:
    priority = (
        "path_planning_failure",
        "region_graph_disconnected",
        "region_graph_fallback",
        "iris_failure",
        "iris_fallback",
        "sampled_region_path_fallback",
        "tracking_safety_violation",
        "trajectory_optimization_fallback",
        "open_grid_fallback",
        "replan_required",
    )
    for flag in priority:
        if flag in flags:
            return flag
    return "none"


def _dedupe(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _report_present(value: Any) -> bool:
    return isinstance(value, dict)


__all__ = [name for name in globals() if not name.startswith("__")]
