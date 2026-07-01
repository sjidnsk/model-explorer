"""Backend and candidate summary helpers for planning diagnostics."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Any

from .planning_diagnostic_interpretation import _candidate_diagnostic_interpretation
from .planning_platform_feasibility import _platform_goal_contract_mismatch
from .planning_types import PathCandidateEvaluation


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


__all__ = (
    "path_feedback_summary",
    "_postprocess_summary",
    "_optimization_summary",
    "_planning_backend_summary",
    "_region_graph_summary",
    "_iris_region_summary",
    "_convex_region_route_report",
    "_convex_region_summary",
    "_gcs_trajectory_route_report",
    "_gcs_trajectory_summary",
    "_gcs_candidate_route_report",
    "_gcs_candidate_summary",
    "_gcs_motion_feasibility_route_report",
    "_gcs_motion_feasibility_summary",
    "_gcs_curvature_constrained_candidate_route_report",
    "_gcs_curvature_constrained_candidate_summary",
)
