from __future__ import annotations

from pathlib import Path
from typing import Any

from .path_feedback_manifest import PathFeedbackManifest
from .path_feedback_compact_summary import compact_summary_payload
from .planning_anchor_projection import anchor_projection_candidate_config_from_mapping


PATH_FEEDBACK_SUMMARY_SCHEMA_VERSION = "path-feedback-summary/v1"
PATH_FEEDBACK_SUMMARY_ACCEPTANCE_METRICS = (
    "selection_changed_rate",
    "path_planning_failure_count",
    "replan_count",
    "tracking_safety_violation_count",
    "trajectory_optimization_fallback_count",
    "region_graph_disconnected_count",
    "coverage_per_path_cost",
)
PATH_FEEDBACK_SUMMARY_REQUIRED_KEYS = (
    "schema_version",
    "scenario_count",
    "scenario_set",
    "diagnostic_profile",
    "acceptance_gate",
    "top_k",
    "planner_extra_args",
    "candidate_count",
    "reachable_count",
    "path_planning_failure_count",
    "replan_count",
    "total_path_cost",
    "average_path_cost",
    "coverage_per_path_cost",
    "selection_changed_count",
    "selection_changed_rate",
    "tracking_safety_violation_count",
    "trajectory_optimization_fallback_count",
    "region_graph_disconnected_count",
    "open_grid_fallback_used",
    "failure_reasons",
    "iris_requested_count",
    "iris_report_count",
    "iris_status_counts",
    "iris_fallback_count",
    "iris_failure_count",
    "iris_region_count_total",
    "iris_fallback_reasons",
    "region_graph_source_counts",
    "region_graph_fallback_count",
    "region_graph_fallback_reasons",
    "region_graph_start_goal_disconnected_count",
    "open_grid_fallback_used_gate",
    "acceptance_metadata",
    "scenario_group_summary",
    "scenarios",
)


def validate_path_feedback_summary_contract(
    summary: dict[str, Any],
    *,
    require_sidecar_inputs: bool = True,
) -> dict[str, Any]:
    if not isinstance(summary, dict):
        raise ValueError("path-feedback-summary/v1 summary must be an object")
    schema_version = summary.get("schema_version")
    if schema_version != PATH_FEEDBACK_SUMMARY_SCHEMA_VERSION:
        raise ValueError(
            f"schema_version must be {PATH_FEEDBACK_SUMMARY_SCHEMA_VERSION}, got {schema_version!r}"
        )
    missing = [key for key in PATH_FEEDBACK_SUMMARY_REQUIRED_KEYS if key not in summary]
    if missing:
        raise ValueError("path-feedback-summary/v1 missing required keys: " + ", ".join(missing))
    if require_sidecar_inputs and summary.get("open_grid_fallback_used") is not False:
        raise ValueError("open_grid_fallback_used must be false for semi-real path feedback validation")
    return {
        "status": "valid",
        "schema_version": PATH_FEEDBACK_SUMMARY_SCHEMA_VERSION,
        "required_key_count": len(PATH_FEEDBACK_SUMMARY_REQUIRED_KEYS),
        "acceptance_metric_count": len(PATH_FEEDBACK_SUMMARY_ACCEPTANCE_METRICS),
    }


def _legacy_compact_path_feedback_summary_payload(
    summary: dict[str, Any],
    *,
    summary_output: Path | None = None,
    report_output: Path | None = None,
) -> dict[str, Any]:
    payload = {
        "status": "completed",
        "schema_version": summary.get("schema_version"),
        "scenario_count": summary.get("scenario_count"),
        "scenario_set": summary.get("scenario_set"),
        "diagnostic_profile": summary.get("diagnostic_profile"),
        "acceptance_gate": summary.get("acceptance_gate"),
        "top_k": summary.get("top_k"),
        "planner_extra_args": summary.get("planner_extra_args", []),
        "candidate_count": summary.get("candidate_count"),
        "reachable_count": summary.get("reachable_count"),
        "path_planning_failure_count": summary.get("path_planning_failure_count"),
        "replan_count": summary.get("replan_count"),
        "selection_changed_count": summary.get("selection_changed_count"),
        "selection_changed_rate": summary.get("selection_changed_rate"),
        "total_path_cost": summary.get("total_path_cost"),
        "average_path_cost": summary.get("average_path_cost"),
        "coverage_per_path_cost": summary.get("coverage_per_path_cost"),
        "tracking_safety_violation_count": summary.get("tracking_safety_violation_count"),
        "trajectory_optimization_fallback_count": summary.get("trajectory_optimization_fallback_count"),
        "region_graph_disconnected_count": summary.get("region_graph_disconnected_count"),
        "open_grid_fallback_used": summary.get("open_grid_fallback_used"),
        "open_grid_fallback_used_gate": summary.get("open_grid_fallback_used_gate", {}),
        "acceptance_metadata": summary.get("acceptance_metadata", {}),
        "failure_reasons": summary.get("failure_reasons", []),
        "iris_requested_count": summary.get("iris_requested_count"),
        "iris_report_count": summary.get("iris_report_count"),
        "iris_status_counts": summary.get("iris_status_counts", {}),
        "iris_fallback_count": summary.get("iris_fallback_count"),
        "iris_failure_count": summary.get("iris_failure_count"),
        "region_graph_source_counts": summary.get("region_graph_source_counts", {}),
        "region_graph_fallback_count": summary.get("region_graph_fallback_count"),
        "region_graph_start_goal_disconnected_count": summary.get("region_graph_start_goal_disconnected_count"),
        "convex_region_report_count": summary.get("convex_region_report_count"),
        "convex_region_count_total": summary.get("convex_region_count_total"),
        "convex_region_backend_counts": summary.get("convex_region_backend_counts", {}),
        "convex_region_fallback_used_count": summary.get("convex_region_fallback_used_count"),
        "convex_region_gcs_ready_count": summary.get("convex_region_gcs_ready_count"),
        "convex_region_blocked_cell_violation_count": summary.get(
            "convex_region_blocked_cell_violation_count"
        ),
        "convex_region_coverage_status_counts": summary.get("convex_region_coverage_status_counts", {}),
        "convex_region_gcs_ready_reason_counts": summary.get("convex_region_gcs_ready_reason_counts", {}),
        "convex_region_start_contained_count": summary.get("convex_region_start_contained_count"),
        "convex_region_goal_contained_count": summary.get("convex_region_goal_contained_count"),
        "convex_region_adjacent_overlap_count": summary.get("convex_region_adjacent_overlap_count"),
        "convex_region_portal_count": summary.get("convex_region_portal_count"),
        "convex_region_candidate_audit": summary.get("convex_region_candidate_audit", []),
        "gcs_trajectory_report_count": summary.get("gcs_trajectory_report_count"),
        "gcs_trajectory_attempted_count": summary.get("gcs_trajectory_attempted_count"),
        "gcs_trajectory_success_count": summary.get("gcs_trajectory_success_count"),
        "gcs_trajectory_collision_count": summary.get("gcs_trajectory_collision_count"),
        "gcs_trajectory_region_count_total": summary.get("gcs_trajectory_region_count_total"),
        "gcs_trajectory_sample_count_total": summary.get("gcs_trajectory_sample_count_total"),
        "gcs_trajectory_backend_counts": summary.get("gcs_trajectory_backend_counts", {}),
        "gcs_trajectory_reason_counts": summary.get("gcs_trajectory_reason_counts", {}),
        "gcs_trajectory_result_status_counts": summary.get("gcs_trajectory_result_status_counts", {}),
        "gcs_trajectory_candidate_audit": summary.get("gcs_trajectory_candidate_audit", []),
        "gcs_candidate_report_count": summary.get("gcs_candidate_report_count"),
        "gcs_candidate_attempted_count": summary.get("gcs_candidate_attempted_count"),
        "gcs_candidate_available_count": summary.get("gcs_candidate_available_count"),
        "gcs_candidate_selected_count": summary.get("gcs_candidate_selected_count"),
        "gcs_candidate_collision_count": summary.get("gcs_candidate_collision_count"),
        "gcs_candidate_fallback_reason_counts": summary.get("gcs_candidate_fallback_reason_counts", {}),
        "gcs_candidate_selection_reason_counts": summary.get("gcs_candidate_selection_reason_counts", {}),
        "gcs_candidate_cost_delta_vs_baseline_negative_count": summary.get(
            "gcs_candidate_cost_delta_vs_baseline_negative_count"
        ),
        "gcs_candidate_cost_delta_vs_baseline_positive_count": summary.get(
            "gcs_candidate_cost_delta_vs_baseline_positive_count"
        ),
        "gcs_candidate_cost_delta_vs_baseline_zero_count": summary.get(
            "gcs_candidate_cost_delta_vs_baseline_zero_count"
        ),
        "gcs_candidate_audit": summary.get("gcs_candidate_audit", []),
        "gcs_control_point_report_count": summary.get("gcs_control_point_report_count"),
        "gcs_control_point_attempted_count": summary.get("gcs_control_point_attempted_count"),
        "gcs_control_point_success_count": summary.get("gcs_control_point_success_count"),
        "gcs_control_point_backend_counts": summary.get("gcs_control_point_backend_counts", {}),
        "gcs_control_point_candidate_selected_count": summary.get(
            "gcs_control_point_candidate_selected_count"
        ),
        "gcs_control_point_candidate_fallback_reason_counts": summary.get(
            "gcs_control_point_candidate_fallback_reason_counts",
            {},
        ),
        "gcs_control_point_terrain_objective_source_counts": summary.get(
            "gcs_control_point_terrain_objective_source_counts",
            {},
        ),
        "gcs_control_point_sampled_terrain_cost_count": summary.get(
            "gcs_control_point_sampled_terrain_cost_count"
        ),
        "gcs_control_point_sampled_terrain_cost_min": summary.get(
            "gcs_control_point_sampled_terrain_cost_min"
        ),
        "gcs_control_point_sampled_terrain_cost_max": summary.get(
            "gcs_control_point_sampled_terrain_cost_max"
        ),
        "gcs_control_point_sampled_terrain_cost_mean": summary.get(
            "gcs_control_point_sampled_terrain_cost_mean"
        ),
        "gcs_control_point_high_cost_exposure_delta_count": summary.get(
            "gcs_control_point_high_cost_exposure_delta_count"
        ),
        "gcs_control_point_high_cost_exposure_delta_min": summary.get(
            "gcs_control_point_high_cost_exposure_delta_min"
        ),
        "gcs_control_point_high_cost_exposure_delta_max": summary.get(
            "gcs_control_point_high_cost_exposure_delta_max"
        ),
        "gcs_control_point_high_cost_exposure_delta_mean": summary.get(
            "gcs_control_point_high_cost_exposure_delta_mean"
        ),
        "gcs_control_point_candidate_audit": summary.get("gcs_control_point_candidate_audit", []),
        "gcs_control_point_candidate_triage": summary.get("gcs_control_point_candidate_triage", {}),
        "gcs_control_point_candidate_artifacts": summary.get("gcs_control_point_candidate_artifacts", {}),
        "gcs_motion_feasibility_report_count": summary.get("gcs_motion_feasibility_report_count"),
        "gcs_motion_feasibility_evaluated_count": summary.get("gcs_motion_feasibility_evaluated_count"),
        "gcs_motion_feasibility_feasible_count": summary.get("gcs_motion_feasibility_feasible_count"),
        "gcs_motion_feasibility_infeasible_count": summary.get("gcs_motion_feasibility_infeasible_count"),
        "gcs_motion_feasibility_diagnostic_only_count": summary.get(
            "gcs_motion_feasibility_diagnostic_only_count"
        ),
        "gcs_motion_feasibility_curvature_violation_count": summary.get(
            "gcs_motion_feasibility_curvature_violation_count"
        ),
        "gcs_motion_feasibility_heading_violation_count": summary.get(
            "gcs_motion_feasibility_heading_violation_count"
        ),
        "gcs_motion_feasibility_status_counts": summary.get("gcs_motion_feasibility_status_counts", {}),
        "gcs_motion_feasibility_fallback_reason_counts": summary.get(
            "gcs_motion_feasibility_fallback_reason_counts",
            {},
        ),
        "gcs_motion_feasibility_motion_model_counts": summary.get(
            "gcs_motion_feasibility_motion_model_counts",
            {},
        ),
        "gcs_motion_feasibility_audit": summary.get("gcs_motion_feasibility_audit", []),
        "gcs_curvature_constrained_report_count": summary.get("gcs_curvature_constrained_report_count"),
        "gcs_curvature_constrained_attempted_count": summary.get("gcs_curvature_constrained_attempted_count"),
        "gcs_curvature_constrained_available_count": summary.get("gcs_curvature_constrained_available_count"),
        "gcs_curvature_constrained_selected_count": summary.get("gcs_curvature_constrained_selected_count"),
        "gcs_curvature_constrained_repair_success_count": summary.get(
            "gcs_curvature_constrained_repair_success_count"
        ),
        "gcs_curvature_constrained_infeasible_count": summary.get("gcs_curvature_constrained_infeasible_count"),
        "gcs_curvature_constrained_diagnostic_only_count": summary.get(
            "gcs_curvature_constrained_diagnostic_only_count"
        ),
        "gcs_curvature_constrained_curvature_violation_count_before": summary.get(
            "gcs_curvature_constrained_curvature_violation_count_before"
        ),
        "gcs_curvature_constrained_curvature_violation_count_after": summary.get(
            "gcs_curvature_constrained_curvature_violation_count_after"
        ),
        "gcs_curvature_constrained_heading_violation_count_before": summary.get(
            "gcs_curvature_constrained_heading_violation_count_before"
        ),
        "gcs_curvature_constrained_heading_violation_count_after": summary.get(
            "gcs_curvature_constrained_heading_violation_count_after"
        ),
        "gcs_curvature_constrained_collision_count": summary.get("gcs_curvature_constrained_collision_count"),
        "gcs_curvature_constrained_region_containment_violation_count": summary.get(
            "gcs_curvature_constrained_region_containment_violation_count"
        ),
        "gcs_curvature_constrained_status_before_counts": summary.get(
            "gcs_curvature_constrained_status_before_counts",
            {},
        ),
        "gcs_curvature_constrained_status_after_counts": summary.get(
            "gcs_curvature_constrained_status_after_counts",
            {},
        ),
        "gcs_curvature_constrained_fallback_reason_counts": summary.get(
            "gcs_curvature_constrained_fallback_reason_counts",
            {},
        ),
        "gcs_curvature_constrained_repair_strategy_counts": summary.get(
            "gcs_curvature_constrained_repair_strategy_counts",
            {},
        ),
        "gcs_curvature_constrained_audit": summary.get("gcs_curvature_constrained_audit", []),
        "channel_aware_astar_report_count": summary.get("channel_aware_astar_report_count"),
        "channel_aware_astar_selected_count": summary.get("channel_aware_astar_selected_count"),
        "channel_aware_astar_fallback_count": summary.get("channel_aware_astar_fallback_count"),
        "channel_aware_astar_requested_backend_counts": summary.get(
            "channel_aware_astar_requested_backend_counts", {}
        ),
        "channel_aware_astar_selected_backend_counts": summary.get(
            "channel_aware_astar_selected_backend_counts", {}
        ),
        "channel_aware_astar_status_counts": summary.get("channel_aware_astar_status_counts", {}),
        "channel_aware_astar_fallback_reason_counts": summary.get(
            "channel_aware_astar_fallback_reason_counts", {}
        ),
        "channel_aware_astar_blocker_class_counts": summary.get(
            "channel_aware_astar_blocker_class_counts", {}
        ),
        "channel_aware_astar_platform_goal_feasibility_class_counts": summary.get(
            "channel_aware_astar_platform_goal_feasibility_class_counts", {}
        ),
        "channel_aware_astar_platform_goal_contract_mismatch_count": summary.get(
            "channel_aware_astar_platform_goal_contract_mismatch_count"
        ),
        "channel_aware_astar_platform_goal_anchor_available_count": summary.get(
            "channel_aware_astar_platform_goal_anchor_available_count"
        ),
        "channel_aware_astar_platform_goal_unresolved_count": summary.get(
            "channel_aware_astar_platform_goal_unresolved_count"
        ),
        "channel_aware_astar_path_changed_count": summary.get("channel_aware_astar_path_changed_count"),
        "channel_aware_astar_path_changed_rate": summary.get("channel_aware_astar_path_changed_rate"),
        "channel_aware_astar_path_cost_delta_count": summary.get("channel_aware_astar_path_cost_delta_count"),
        "channel_aware_astar_path_cost_delta_min": summary.get("channel_aware_astar_path_cost_delta_min"),
        "channel_aware_astar_path_cost_delta_max": summary.get("channel_aware_astar_path_cost_delta_max"),
        "channel_aware_astar_path_cost_delta_mean": summary.get("channel_aware_astar_path_cost_delta_mean"),
        "channel_aware_astar_channel_cost_delta_count": summary.get(
            "channel_aware_astar_channel_cost_delta_count"
        ),
        "channel_aware_astar_channel_cost_delta_min": summary.get(
            "channel_aware_astar_channel_cost_delta_min"
        ),
        "channel_aware_astar_channel_cost_delta_max": summary.get(
            "channel_aware_astar_channel_cost_delta_max"
        ),
        "channel_aware_astar_channel_cost_delta_mean": summary.get(
            "channel_aware_astar_channel_cost_delta_mean"
        ),
        "channel_aware_astar_high_cost_exposure_delta_count": summary.get(
            "channel_aware_astar_high_cost_exposure_delta_count"
        ),
        "channel_aware_astar_high_cost_exposure_delta_min": summary.get(
            "channel_aware_astar_high_cost_exposure_delta_min"
        ),
        "channel_aware_astar_high_cost_exposure_delta_max": summary.get(
            "channel_aware_astar_high_cost_exposure_delta_max"
        ),
        "channel_aware_astar_high_cost_exposure_delta_mean": summary.get(
            "channel_aware_astar_high_cost_exposure_delta_mean"
        ),
        "channel_aware_astar_candidate_audit": summary.get("channel_aware_astar_candidate_audit", []),
        "sampled_region_path_selected_count": summary.get("sampled_region_path_selected_count"),
        "sampled_region_path_fallback_count": summary.get("sampled_region_path_fallback_count"),
        "sampled_region_path_status_counts": summary.get("sampled_region_path_status_counts", {}),
        "sampled_region_path_source_counts": summary.get("sampled_region_path_source_counts", {}),
        "sampled_region_path_fallback_reasons": summary.get("sampled_region_path_fallback_reasons", {}),
        "sampled_region_path_sample_attempt_count": summary.get("sampled_region_path_sample_attempt_count"),
        "sampled_region_path_candidate_ranking_count": summary.get("sampled_region_path_candidate_ranking_count"),
        "sampled_region_path_anchor_region_added_count": summary.get(
            "sampled_region_path_anchor_region_added_count"
        ),
        "sampled_region_path_anchor_region_connected_count": summary.get(
            "sampled_region_path_anchor_region_connected_count"
        ),
        "sampled_region_path_anchor_closure_attempt_count": summary.get(
            "sampled_region_path_anchor_closure_attempt_count"
        ),
        "sampled_region_path_anchor_closure_connected_count": summary.get(
            "sampled_region_path_anchor_closure_connected_count"
        ),
        "sampled_region_path_anchor_closure_status_counts": summary.get(
            "sampled_region_path_anchor_closure_status_counts",
            {},
        ),
        "sampled_region_path_anchor_closure_reason_counts": summary.get(
            "sampled_region_path_anchor_closure_reason_counts",
            {},
        ),
        "sampled_region_path_anchor_closure_connection_kind_counts": summary.get(
            "sampled_region_path_anchor_closure_connection_kind_counts",
            {},
        ),
        "sampled_region_path_start_classification_counts": summary.get(
            "sampled_region_path_start_classification_counts",
            {},
        ),
        "sampled_region_path_goal_classification_counts": summary.get(
            "sampled_region_path_goal_classification_counts",
            {},
        ),
        "sampled_region_path_connector_attempt_count": summary.get(
            "sampled_region_path_connector_attempt_count"
        ),
        "sampled_region_path_connector_strategy_counts": summary.get(
            "sampled_region_path_connector_strategy_counts",
            {},
        ),
        "sampled_region_path_bridge_aware_connector_attempt_count": summary.get(
            "sampled_region_path_bridge_aware_connector_attempt_count"
        ),
        "sampled_region_path_bridge_aware_connector_available_count": summary.get(
            "sampled_region_path_bridge_aware_connector_available_count"
        ),
        "sampled_region_path_bridge_aware_connector_selected_count": summary.get(
            "sampled_region_path_bridge_aware_connector_selected_count"
        ),
        "sampled_region_path_bridge_aware_connector_rejected_count": summary.get(
            "sampled_region_path_bridge_aware_connector_rejected_count"
        ),
        "sampled_region_path_bridge_aware_connector_status_counts": summary.get(
            "sampled_region_path_bridge_aware_connector_status_counts",
            {},
        ),
        "sampled_region_path_bridge_aware_fallback_reasons": summary.get(
            "sampled_region_path_bridge_aware_fallback_reasons",
            {},
        ),
        "sampled_region_path_bridge_aware_bridge_cell_count": summary.get(
            "sampled_region_path_bridge_aware_bridge_cell_count"
        ),
        "sampled_region_path_bridge_aware_mask_added_cell_count": summary.get(
            "sampled_region_path_bridge_aware_mask_added_cell_count"
        ),
        "sampled_region_path_bridge_corridor_connector_attempt_count": summary.get(
            "sampled_region_path_bridge_corridor_connector_attempt_count"
        ),
        "sampled_region_path_bridge_corridor_connector_available_count": summary.get(
            "sampled_region_path_bridge_corridor_connector_available_count"
        ),
        "sampled_region_path_bridge_corridor_connector_selected_count": summary.get(
            "sampled_region_path_bridge_corridor_connector_selected_count"
        ),
        "sampled_region_path_bridge_corridor_connector_rejected_count": summary.get(
            "sampled_region_path_bridge_corridor_connector_rejected_count"
        ),
        "sampled_region_path_bridge_corridor_status_counts": summary.get(
            "sampled_region_path_bridge_corridor_status_counts",
            {},
        ),
        "sampled_region_path_bridge_corridor_fallback_reasons": summary.get(
            "sampled_region_path_bridge_corridor_fallback_reasons",
            {},
        ),
        "sampled_region_path_bridge_corridor_radius_counts": summary.get(
            "sampled_region_path_bridge_corridor_radius_counts",
            {},
        ),
        "sampled_region_path_bridge_corridor_added_cell_count": summary.get(
            "sampled_region_path_bridge_corridor_added_cell_count"
        ),
        "sampled_region_path_terminal_adjusted_count": summary.get(
            "sampled_region_path_terminal_adjusted_count"
        ),
        "sampled_region_path_terminal_adjustment_candidate_count": summary.get(
            "sampled_region_path_terminal_adjustment_candidate_count"
        ),
        "sampled_region_path_terminal_adjustment_status_counts": summary.get(
            "sampled_region_path_terminal_adjustment_status_counts",
            {},
        ),
        "sampled_region_path_terminal_adjustment_reason_counts": summary.get(
            "sampled_region_path_terminal_adjustment_reason_counts",
            {},
        ),
        "sampled_region_path_reachable_component_status_counts": summary.get(
            "sampled_region_path_reachable_component_status_counts",
            {},
        ),
        "sampled_region_path_reachable_component_reason_counts": summary.get(
            "sampled_region_path_reachable_component_reason_counts",
            {},
        ),
        "sampled_region_path_reachable_component_disconnected_count": summary.get(
            "sampled_region_path_reachable_component_disconnected_count"
        ),
        "sampled_region_path_reachable_component_replacement_selected_count": summary.get(
            "sampled_region_path_reachable_component_replacement_selected_count"
        ),
        "sampled_region_path_reachable_component_terminal_candidate_count": summary.get(
            "sampled_region_path_reachable_component_terminal_candidate_count"
        ),
        "sampled_region_path_reachable_terminal_rescue_count": summary.get(
            "sampled_region_path_reachable_terminal_rescue_count"
        ),
        "sampled_region_path_proxy_goal_anchor_selected_count": summary.get(
            "sampled_region_path_proxy_goal_anchor_selected_count"
        ),
        "sampled_region_path_goal_rescue_candidate_count": summary.get(
            "sampled_region_path_goal_rescue_candidate_count"
        ),
        "sampled_region_path_benefit_surface_present_count": summary.get(
            "sampled_region_path_benefit_surface_present_count"
        ),
        "sampled_region_path_path_duplicate_with_baseline_count": summary.get(
            "sampled_region_path_path_duplicate_with_baseline_count"
        ),
        "sampled_region_path_baseline_equivalent_count": summary.get(
            "sampled_region_path_baseline_equivalent_count"
        ),
        "sampled_region_path_no_quality_gain_count": summary.get(
            "sampled_region_path_no_quality_gain_count"
        ),
        "sampled_region_path_fixture_no_benefit_surface_count": summary.get(
            "sampled_region_path_fixture_no_benefit_surface_count"
        ),
        "sampled_region_path_candidate_missing_metrics_count": summary.get(
            "sampled_region_path_candidate_missing_metrics_count"
        ),
        "sampled_region_path_constrained_connector_failed_count": summary.get(
            "sampled_region_path_constrained_connector_failed_count"
        ),
        "sampled_region_path_complexity_reason_counts": summary.get(
            "sampled_region_path_complexity_reason_counts",
            {},
        ),
        "sampled_region_path_execution_tie_break_status_counts": summary.get(
            "sampled_region_path_execution_tie_break_status_counts",
            {},
        ),
        "sampled_region_path_execution_tie_break_reason_counts": summary.get(
            "sampled_region_path_execution_tie_break_reason_counts",
            {},
        ),
        "sampled_region_path_candidate_audit": summary.get("sampled_region_path_candidate_audit", []),
        "diagnostic_interpretation": summary.get("diagnostic_interpretation", {}),
    }
    if summary_output is not None:
        payload["summary_output"] = str(summary_output)
    if report_output is not None:
        payload["report_output"] = str(report_output)
    return payload


def compact_path_feedback_summary(
    summary: dict[str, Any],
    *,
    summary_output: Path | None = None,
    report_output: Path | None = None,
) -> dict[str, Any]:
    return compact_summary_payload(
        summary,
        summary_output=summary_output,
        report_output=report_output,
    )


def _acceptance_metadata(
    manifest: PathFeedbackManifest,
    *,
    open_grid_fallback_used: bool,
) -> dict[str, Any]:
    gate_status = "failed" if open_grid_fallback_used else "passed"
    reason_codes = ["open_grid_fallback_used"] if open_grid_fallback_used else ["open_grid_fallback_not_used"]
    return {
        "schema_version": "path-feedback-acceptance-metadata/v1",
        "scenario_set": manifest.scenario_set,
        "diagnostic_profile": manifest.diagnostic_profile,
        "acceptance_gate": manifest.acceptance_gate,
        "top_k": int(manifest.top_k),
        "python_executable": manifest.python_executable,
        "planner_extra_args": list(manifest.planner_extra_args),
        "anchor_projection_candidate_generation_enabled": bool(
            anchor_projection_candidate_config_from_mapping(
            manifest.planner_config.get("anchor_projection_candidate_generation")
        ).enabled
        ),
        "open_grid_fallback_used": bool(open_grid_fallback_used),
        "open_grid_fallback_used_gate": {
            "status": gate_status,
            "expected": False,
            "actual": bool(open_grid_fallback_used),
            "reason_codes": reason_codes,
        },
    }


acceptance_metadata = _acceptance_metadata

__all__ = [
    'PATH_FEEDBACK_SUMMARY_SCHEMA_VERSION',
    'PATH_FEEDBACK_SUMMARY_ACCEPTANCE_METRICS',
    'PATH_FEEDBACK_SUMMARY_REQUIRED_KEYS',
    'validate_path_feedback_summary_contract',
    'compact_path_feedback_summary',
    'acceptance_metadata',
    '_acceptance_metadata',
]
