from __future__ import annotations
from typing import Any
from .path_feedback_diagnostic_aggregate import (
    GCS_CONTROL_POINT_BACKEND,
    _first_present,
    _int_value,
)
def _sampled_region_path_candidate_audit(evaluations, *, scenario_id: str) -> list[dict[str, Any]]:
    audit: list[dict[str, Any]] = []
    for item in evaluations:
        candidate = item.to_dict()
        planning_backend = candidate.get("planning_backend")
        if not isinstance(planning_backend, dict):
            continue
        sampled = planning_backend.get("sampled_region_path")
        if not isinstance(sampled, dict) or not sampled:
            continue
        graph = candidate.get("region_graph")
        region_source = "unknown"
        if isinstance(graph, dict):
            region_source = str(graph.get("graph_source") or graph.get("region_source") or "unknown")
        rankings = sampled.get("candidate_rankings")
        rankings = rankings if isinstance(rankings, list) else []
        metrics = sampled.get("candidate_comparison")
        metrics = metrics if isinstance(metrics, dict) else {}
        audit.append(
            {
                "scenario_id": scenario_id,
                "action_index": candidate["action_index"],
                "cell": candidate["cell"],
                "region_source": region_source,
                "selected_backend": planning_backend.get("selected_backend"),
                "status": sampled.get("status") or planning_backend.get("status") or "unknown",
                "fallback_reason": sampled.get("fallback_reason"),
                "region_sequence": sampled.get("region_sequence", []),
                "start_goal_anchoring": sampled.get("start_goal_anchoring", {}),
                "edge_transition_count": _int_value(sampled.get("edge_transition_count")),
                "sample_attempt_count": _int_value(sampled.get("sample_attempt_count")),
                "candidate_ranking_count": len(rankings),
                "candidate_metrics": metrics,
                "terminal_adjustment_report": sampled.get("terminal_adjustment_report", {}),
                "execution_tie_break": sampled.get("execution_tie_break", {}),
                "best_candidate_ranking": rankings[0] if rankings else {},
            }
        )
    return audit


def _convex_region_candidate_audit(evaluations, *, scenario_id: str) -> list[dict[str, Any]]:
    audit: list[dict[str, Any]] = []
    for item in evaluations:
        candidate = item.to_dict()
        convex = candidate.get("convex_region")
        if not isinstance(convex, dict):
            continue
        audit.append(
            {
                "scenario_id": scenario_id,
                "action_index": candidate["action_index"],
                "cell": candidate["cell"],
                "backend": convex.get("backend"),
                "region_count": _int_value(convex.get("region_count")),
                "fallback_used": convex.get("fallback_used"),
                "coverage_status": convex.get("coverage_status"),
                "start_contained": convex.get("start_contained"),
                "goal_contained": convex.get("goal_contained"),
                "adjacent_overlap_count": _int_value(convex.get("adjacent_overlap_count")),
                "portal_count": _int_value(convex.get("portal_count")),
                "blocked_cell_violation_count": _int_value(convex.get("blocked_cell_violation_count")),
                "gcs_ready": convex.get("gcs_ready"),
                "gcs_ready_reason": convex.get("gcs_ready_reason"),
            }
        )
    return audit


def _gcs_trajectory_candidate_audit(evaluations, *, scenario_id: str) -> list[dict[str, Any]]:
    audit: list[dict[str, Any]] = []
    for item in evaluations:
        candidate = item.to_dict()
        gcs = candidate.get("gcs_trajectory")
        if not isinstance(gcs, dict):
            continue
        audit.append(
            {
                "scenario_id": scenario_id,
                "action_index": candidate["action_index"],
                "cell": candidate["cell"],
                "backend": gcs.get("backend"),
                "attempted": gcs.get("attempted"),
                "success": gcs.get("success"),
                "result_status": gcs.get("result_status"),
                "reason": gcs.get("reason"),
                "sample_count": _int_value(gcs.get("sample_count")),
                "collision_count": _int_value(gcs.get("collision_count")),
                "path_length": gcs.get("path_length"),
                "region_count": _int_value(gcs.get("region_count")),
            }
        )
    return audit


def _gcs_candidate_audit(evaluations, *, scenario_id: str) -> list[dict[str, Any]]:
    audit: list[dict[str, Any]] = []
    for item in evaluations:
        candidate = item.to_dict()
        gcs_candidate = candidate.get("gcs_candidate")
        if not isinstance(gcs_candidate, dict):
            continue
        audit.append(
            {
                "scenario_id": scenario_id,
                "action_index": candidate["action_index"],
                "cell": candidate["cell"],
                "attempted": gcs_candidate.get("attempted"),
                "available": gcs_candidate.get("available"),
                "selected": gcs_candidate.get("selected"),
                "selection_reason": gcs_candidate.get("selection_reason"),
                "fallback_reason": gcs_candidate.get("fallback_reason"),
                "collision_count": _int_value(gcs_candidate.get("collision_count")),
                "path_length": gcs_candidate.get("path_length"),
                "path_cost": gcs_candidate.get("path_cost"),
                "high_cost_exposure": gcs_candidate.get("high_cost_exposure"),
                "baseline_overlap_ratio": gcs_candidate.get("baseline_overlap_ratio"),
                "cost_delta_vs_baseline": gcs_candidate.get("cost_delta_vs_baseline"),
                "cost_delta_vs_postprocess": gcs_candidate.get("cost_delta_vs_postprocess"),
            }
        )
    return audit


def _gcs_control_point_candidate_audit(evaluations, *, scenario_id: str) -> list[dict[str, Any]]:
    audit: list[dict[str, Any]] = []
    for item in evaluations:
        candidate = item.to_dict()
        gcs = candidate.get("gcs_trajectory")
        if not isinstance(gcs, dict) or gcs.get("backend") != GCS_CONTROL_POINT_BACKEND:
            continue
        gcs_candidate = candidate.get("gcs_candidate")
        gcs_candidate = gcs_candidate if isinstance(gcs_candidate, dict) else {}
        trajectory_cost = gcs.get("cost_summary")
        trajectory_cost = trajectory_cost if isinstance(trajectory_cost, dict) else {}
        candidate_cost = gcs_candidate.get("cost_summary")
        candidate_cost = candidate_cost if isinstance(candidate_cost, dict) else {}
        audit.append(
            {
                "scenario_id": scenario_id,
                "action_index": candidate["action_index"],
                "cell": candidate["cell"],
                "backend": gcs.get("backend"),
                "attempted": gcs.get("attempted"),
                "success": gcs.get("success"),
                "reason": gcs.get("reason"),
                "candidate_selected": gcs_candidate.get("selected"),
                "candidate_fallback_reason": gcs_candidate.get("fallback_reason"),
                "terrain_objective_source": _first_present(
                    trajectory_cost.get("terrain_objective_source"),
                    candidate_cost.get("terrain_objective_source"),
                ),
                "terrain_objective_weight": _first_present(
                    trajectory_cost.get("terrain_objective_weight"),
                    candidate_cost.get("terrain_objective_weight"),
                ),
                "sampled_terrain_cost": _first_present(
                    trajectory_cost.get("sampled_terrain_cost"),
                    candidate_cost.get("sampled_terrain_cost"),
                ),
                "control_point_terrain_cost": _first_present(
                    trajectory_cost.get("control_point_terrain_cost"),
                    candidate_cost.get("control_point_terrain_cost"),
                ),
                "high_cost_exposure": _first_present(
                    candidate_cost.get("high_cost_exposure"),
                    trajectory_cost.get("high_cost_exposure"),
                    gcs_candidate.get("high_cost_exposure"),
                ),
                "baseline_high_cost_exposure": candidate_cost.get("baseline_high_cost_exposure"),
                "high_cost_exposure_delta_vs_baseline": candidate_cost.get(
                    "high_cost_exposure_delta_vs_baseline"
                ),
            }
        )
    return audit


def _gcs_motion_feasibility_audit(evaluations, *, scenario_id: str) -> list[dict[str, Any]]:
    audit: list[dict[str, Any]] = []
    for item in evaluations:
        candidate = item.to_dict()
        motion = candidate.get("gcs_motion_feasibility")
        if not isinstance(motion, dict):
            continue
        audit.append(
            {
                "scenario_id": scenario_id,
                "action_index": candidate["action_index"],
                "cell": candidate["cell"],
                "evaluated": motion.get("evaluated"),
                "trajectory_source": motion.get("trajectory_source"),
                "motion_model": motion.get("motion_model"),
                "feasibility_status": motion.get("feasibility_status"),
                "fallback_reason": motion.get("fallback_reason"),
                "min_turning_radius_m": motion.get("min_turning_radius_m"),
                "max_heading_change_deg": motion.get("max_heading_change_deg"),
                "curvature_violation_count": _int_value(motion.get("curvature_violation_count")),
                "heading_violation_count": _int_value(motion.get("heading_violation_count")),
                "violation_indices": (
                    motion.get("violation_indices")
                    if isinstance(motion.get("violation_indices"), list)
                    else []
                ),
                "sample_count": _int_value(motion.get("sample_count")),
                "path_length": motion.get("path_length"),
                "constraint_summary": (
                    motion.get("constraint_summary")
                    if isinstance(motion.get("constraint_summary"), dict)
                    else {}
                ),
            }
        )
    return audit


def _gcs_curvature_constrained_audit(evaluations, *, scenario_id: str) -> list[dict[str, Any]]:
    audit: list[dict[str, Any]] = []
    for item in evaluations:
        candidate = item.to_dict()
        constrained = candidate.get("gcs_curvature_constrained_candidate")
        if not isinstance(constrained, dict):
            continue
        audit.append(
            {
                "scenario_id": scenario_id,
                "action_index": candidate["action_index"],
                "cell": candidate["cell"],
                "attempted": constrained.get("attempted"),
                "available": constrained.get("available"),
                "selected": constrained.get("selected"),
                "repair_success": constrained.get("repair_success"),
                "source": constrained.get("source"),
                "repair_strategy": constrained.get("repair_strategy"),
                "status_before": constrained.get("status_before"),
                "status_after": constrained.get("status_after"),
                "fallback_reason": constrained.get("fallback_reason"),
                "curvature_violation_count_before": _int_value(
                    constrained.get("curvature_violation_count_before")
                ),
                "curvature_violation_count_after": _int_value(
                    constrained.get("curvature_violation_count_after")
                ),
                "heading_violation_count_before": _int_value(
                    constrained.get("heading_violation_count_before")
                ),
                "heading_violation_count_after": _int_value(
                    constrained.get("heading_violation_count_after")
                ),
                "violation_indices_before": (
                    constrained.get("violation_indices_before")
                    if isinstance(constrained.get("violation_indices_before"), list)
                    else []
                ),
                "violation_indices_after": (
                    constrained.get("violation_indices_after")
                    if isinstance(constrained.get("violation_indices_after"), list)
                    else []
                ),
                "collision_count": _int_value(constrained.get("collision_count")),
                "region_containment_violation_count": _int_value(
                    constrained.get("region_containment_violation_count")
                ),
                "path_length": constrained.get("path_length"),
                "path_cost": constrained.get("path_cost"),
                "cost_delta_vs_baseline": constrained.get("cost_delta_vs_baseline"),
                "constraint_summary": (
                    constrained.get("constraint_summary")
                    if isinstance(constrained.get("constraint_summary"), dict)
                    else {}
                ),
            }
        )
    return audit
__all__ = (
    "_sampled_region_path_candidate_audit",
    "_convex_region_candidate_audit",
    "_gcs_trajectory_candidate_audit",
    "_gcs_candidate_audit",
    "_gcs_control_point_candidate_audit",
    "_gcs_motion_feasibility_audit",
    "_gcs_curvature_constrained_audit",
)
