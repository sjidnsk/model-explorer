from __future__ import annotations

from pathlib import Path
from typing import Any


_NO_DEFAULT = object()

_CORE_RUN_FIELDS = (
    ("schema_version", _NO_DEFAULT),
    ("scenario_count", _NO_DEFAULT),
    ("scenario_set", _NO_DEFAULT),
    ("diagnostic_profile", _NO_DEFAULT),
    ("acceptance_gate", _NO_DEFAULT),
    ("top_k", _NO_DEFAULT),
    ("planner_extra_args", []),
    ("candidate_count", _NO_DEFAULT),
    ("reachable_count", _NO_DEFAULT),
    ("path_planning_failure_count", _NO_DEFAULT),
    ("replan_count", _NO_DEFAULT),
    ("selection_changed_count", _NO_DEFAULT),
    ("selection_changed_rate", _NO_DEFAULT),
    ("total_path_cost", _NO_DEFAULT),
    ("average_path_cost", _NO_DEFAULT),
    ("coverage_per_path_cost", _NO_DEFAULT),
    ("tracking_safety_violation_count", _NO_DEFAULT),
    ("trajectory_optimization_fallback_count", _NO_DEFAULT),
    ("region_graph_disconnected_count", _NO_DEFAULT),
    ("open_grid_fallback_used", _NO_DEFAULT),
)

_ACCEPTANCE_METADATA_FIELDS = (
    ("open_grid_fallback_used_gate", {}),
    ("acceptance_metadata", {}),
    ("failure_reasons", []),
)

_IRIS_REGION_GRAPH_FIELDS = (
    ("iris_requested_count", _NO_DEFAULT),
    ("iris_report_count", _NO_DEFAULT),
    ("iris_status_counts", {}),
    ("iris_fallback_count", _NO_DEFAULT),
    ("iris_failure_count", _NO_DEFAULT),
    ("region_graph_source_counts", {}),
    ("region_graph_fallback_count", _NO_DEFAULT),
    ("region_graph_start_goal_disconnected_count", _NO_DEFAULT),
)

_CONVEX_REGION_FIELDS = (
    ("convex_region_report_count", _NO_DEFAULT),
    ("convex_region_count_total", _NO_DEFAULT),
    ("convex_region_backend_counts", {}),
    ("convex_region_fallback_used_count", _NO_DEFAULT),
    ("convex_region_gcs_ready_count", _NO_DEFAULT),
    ("convex_region_blocked_cell_violation_count", _NO_DEFAULT),
    ("convex_region_coverage_status_counts", {}),
    ("convex_region_gcs_ready_reason_counts", {}),
    ("convex_region_start_contained_count", _NO_DEFAULT),
    ("convex_region_goal_contained_count", _NO_DEFAULT),
    ("convex_region_adjacent_overlap_count", _NO_DEFAULT),
    ("convex_region_portal_count", _NO_DEFAULT),
    ("convex_region_candidate_audit", []),
)

_GCS_TRAJECTORY_FIELDS = (
    ("gcs_trajectory_report_count", _NO_DEFAULT),
    ("gcs_trajectory_attempted_count", _NO_DEFAULT),
    ("gcs_trajectory_success_count", _NO_DEFAULT),
    ("gcs_trajectory_collision_count", _NO_DEFAULT),
    ("gcs_trajectory_region_count_total", _NO_DEFAULT),
    ("gcs_trajectory_sample_count_total", _NO_DEFAULT),
    ("gcs_trajectory_backend_counts", {}),
    ("gcs_trajectory_reason_counts", {}),
    ("gcs_trajectory_result_status_counts", {}),
    ("gcs_trajectory_candidate_audit", []),
)

_GCS_CANDIDATE_FIELDS = (
    ("gcs_candidate_report_count", _NO_DEFAULT),
    ("gcs_candidate_attempted_count", _NO_DEFAULT),
    ("gcs_candidate_available_count", _NO_DEFAULT),
    ("gcs_candidate_selected_count", _NO_DEFAULT),
    ("gcs_candidate_collision_count", _NO_DEFAULT),
    ("gcs_candidate_fallback_reason_counts", {}),
    ("gcs_candidate_selection_reason_counts", {}),
    ("gcs_candidate_cost_delta_vs_baseline_negative_count", _NO_DEFAULT),
    ("gcs_candidate_cost_delta_vs_baseline_positive_count", _NO_DEFAULT),
    ("gcs_candidate_cost_delta_vs_baseline_zero_count", _NO_DEFAULT),
    ("gcs_candidate_audit", []),
)

_GCS_CONTROL_POINT_FIELDS = (
    ("gcs_control_point_report_count", _NO_DEFAULT),
    ("gcs_control_point_attempted_count", _NO_DEFAULT),
    ("gcs_control_point_success_count", _NO_DEFAULT),
    ("gcs_control_point_backend_counts", {}),
    ("gcs_control_point_candidate_selected_count", _NO_DEFAULT),
    ("gcs_control_point_candidate_fallback_reason_counts", {}),
    ("gcs_control_point_terrain_objective_source_counts", {}),
    ("gcs_control_point_sampled_terrain_cost_count", _NO_DEFAULT),
    ("gcs_control_point_sampled_terrain_cost_min", _NO_DEFAULT),
    ("gcs_control_point_sampled_terrain_cost_max", _NO_DEFAULT),
    ("gcs_control_point_sampled_terrain_cost_mean", _NO_DEFAULT),
    ("gcs_control_point_high_cost_exposure_delta_count", _NO_DEFAULT),
    ("gcs_control_point_high_cost_exposure_delta_min", _NO_DEFAULT),
    ("gcs_control_point_high_cost_exposure_delta_max", _NO_DEFAULT),
    ("gcs_control_point_high_cost_exposure_delta_mean", _NO_DEFAULT),
    ("gcs_control_point_candidate_audit", []),
    ("gcs_control_point_candidate_triage", {}),
    ("gcs_control_point_candidate_artifacts", {}),
)

_GCS_MOTION_FEASIBILITY_FIELDS = (
    ("gcs_motion_feasibility_report_count", _NO_DEFAULT),
    ("gcs_motion_feasibility_evaluated_count", _NO_DEFAULT),
    ("gcs_motion_feasibility_feasible_count", _NO_DEFAULT),
    ("gcs_motion_feasibility_infeasible_count", _NO_DEFAULT),
    ("gcs_motion_feasibility_diagnostic_only_count", _NO_DEFAULT),
    ("gcs_motion_feasibility_curvature_violation_count", _NO_DEFAULT),
    ("gcs_motion_feasibility_heading_violation_count", _NO_DEFAULT),
    ("gcs_motion_feasibility_status_counts", {}),
    ("gcs_motion_feasibility_fallback_reason_counts", {}),
    ("gcs_motion_feasibility_motion_model_counts", {}),
    ("gcs_motion_feasibility_audit", []),
)

_GCS_CURVATURE_CONSTRAINED_FIELDS = (
    ("gcs_curvature_constrained_report_count", _NO_DEFAULT),
    ("gcs_curvature_constrained_attempted_count", _NO_DEFAULT),
    ("gcs_curvature_constrained_available_count", _NO_DEFAULT),
    ("gcs_curvature_constrained_selected_count", _NO_DEFAULT),
    ("gcs_curvature_constrained_repair_success_count", _NO_DEFAULT),
    ("gcs_curvature_constrained_infeasible_count", _NO_DEFAULT),
    ("gcs_curvature_constrained_diagnostic_only_count", _NO_DEFAULT),
    ("gcs_curvature_constrained_curvature_violation_count_before", _NO_DEFAULT),
    ("gcs_curvature_constrained_curvature_violation_count_after", _NO_DEFAULT),
    ("gcs_curvature_constrained_heading_violation_count_before", _NO_DEFAULT),
    ("gcs_curvature_constrained_heading_violation_count_after", _NO_DEFAULT),
    ("gcs_curvature_constrained_collision_count", _NO_DEFAULT),
    ("gcs_curvature_constrained_region_containment_violation_count", _NO_DEFAULT),
    ("gcs_curvature_constrained_status_before_counts", {}),
    ("gcs_curvature_constrained_status_after_counts", {}),
    ("gcs_curvature_constrained_fallback_reason_counts", {}),
    ("gcs_curvature_constrained_repair_strategy_counts", {}),
    ("gcs_curvature_constrained_audit", []),
)

_CHANNEL_AWARE_ASTAR_FIELDS = (
    ("channel_aware_astar_report_count", _NO_DEFAULT),
    ("channel_aware_astar_selected_count", _NO_DEFAULT),
    ("channel_aware_astar_fallback_count", _NO_DEFAULT),
    ("channel_aware_astar_requested_backend_counts", {}),
    ("channel_aware_astar_selected_backend_counts", {}),
    ("channel_aware_astar_status_counts", {}),
    ("channel_aware_astar_fallback_reason_counts", {}),
    ("channel_aware_astar_blocker_class_counts", {}),
    ("channel_aware_astar_platform_goal_feasibility_class_counts", {}),
    ("channel_aware_astar_platform_goal_contract_mismatch_count", _NO_DEFAULT),
    ("channel_aware_astar_platform_goal_anchor_available_count", _NO_DEFAULT),
    ("channel_aware_astar_platform_goal_unresolved_count", _NO_DEFAULT),
    ("channel_aware_astar_path_changed_count", _NO_DEFAULT),
    ("channel_aware_astar_path_changed_rate", _NO_DEFAULT),
    ("channel_aware_astar_path_cost_delta_count", _NO_DEFAULT),
    ("channel_aware_astar_path_cost_delta_min", _NO_DEFAULT),
    ("channel_aware_astar_path_cost_delta_max", _NO_DEFAULT),
    ("channel_aware_astar_path_cost_delta_mean", _NO_DEFAULT),
    ("channel_aware_astar_channel_cost_delta_count", _NO_DEFAULT),
    ("channel_aware_astar_channel_cost_delta_min", _NO_DEFAULT),
    ("channel_aware_astar_channel_cost_delta_max", _NO_DEFAULT),
    ("channel_aware_astar_channel_cost_delta_mean", _NO_DEFAULT),
    ("channel_aware_astar_high_cost_exposure_delta_count", _NO_DEFAULT),
    ("channel_aware_astar_high_cost_exposure_delta_min", _NO_DEFAULT),
    ("channel_aware_astar_high_cost_exposure_delta_max", _NO_DEFAULT),
    ("channel_aware_astar_high_cost_exposure_delta_mean", _NO_DEFAULT),
    ("channel_aware_astar_candidate_audit", []),
)

_SAMPLED_REGION_FIELDS = (
    ("sampled_region_path_selected_count", _NO_DEFAULT),
    ("sampled_region_path_fallback_count", _NO_DEFAULT),
    ("sampled_region_path_status_counts", {}),
    ("sampled_region_path_source_counts", {}),
    ("sampled_region_path_fallback_reasons", {}),
    ("sampled_region_path_sample_attempt_count", _NO_DEFAULT),
    ("sampled_region_path_candidate_ranking_count", _NO_DEFAULT),
    ("sampled_region_path_anchor_region_added_count", _NO_DEFAULT),
    ("sampled_region_path_anchor_region_connected_count", _NO_DEFAULT),
    ("sampled_region_path_anchor_closure_attempt_count", _NO_DEFAULT),
    ("sampled_region_path_anchor_closure_connected_count", _NO_DEFAULT),
    ("sampled_region_path_anchor_closure_status_counts", {}),
    ("sampled_region_path_anchor_closure_reason_counts", {}),
    ("sampled_region_path_anchor_closure_connection_kind_counts", {}),
    ("sampled_region_path_start_classification_counts", {}),
    ("sampled_region_path_goal_classification_counts", {}),
    ("sampled_region_path_connector_attempt_count", _NO_DEFAULT),
    ("sampled_region_path_connector_strategy_counts", {}),
    ("sampled_region_path_bridge_aware_connector_attempt_count", _NO_DEFAULT),
    ("sampled_region_path_bridge_aware_connector_available_count", _NO_DEFAULT),
    ("sampled_region_path_bridge_aware_connector_selected_count", _NO_DEFAULT),
    ("sampled_region_path_bridge_aware_connector_rejected_count", _NO_DEFAULT),
    ("sampled_region_path_bridge_aware_connector_status_counts", {}),
    ("sampled_region_path_bridge_aware_fallback_reasons", {}),
    ("sampled_region_path_bridge_aware_bridge_cell_count", _NO_DEFAULT),
    ("sampled_region_path_bridge_aware_mask_added_cell_count", _NO_DEFAULT),
    ("sampled_region_path_bridge_corridor_connector_attempt_count", _NO_DEFAULT),
    ("sampled_region_path_bridge_corridor_connector_available_count", _NO_DEFAULT),
    ("sampled_region_path_bridge_corridor_connector_selected_count", _NO_DEFAULT),
    ("sampled_region_path_bridge_corridor_connector_rejected_count", _NO_DEFAULT),
    ("sampled_region_path_bridge_corridor_status_counts", {}),
    ("sampled_region_path_bridge_corridor_fallback_reasons", {}),
    ("sampled_region_path_bridge_corridor_radius_counts", {}),
    ("sampled_region_path_bridge_corridor_added_cell_count", _NO_DEFAULT),
    ("sampled_region_path_terminal_adjusted_count", _NO_DEFAULT),
    ("sampled_region_path_terminal_adjustment_candidate_count", _NO_DEFAULT),
    ("sampled_region_path_terminal_adjustment_status_counts", {}),
    ("sampled_region_path_terminal_adjustment_reason_counts", {}),
    ("sampled_region_path_reachable_component_status_counts", {}),
    ("sampled_region_path_reachable_component_reason_counts", {}),
    ("sampled_region_path_reachable_component_disconnected_count", _NO_DEFAULT),
    ("sampled_region_path_reachable_component_replacement_selected_count", _NO_DEFAULT),
    ("sampled_region_path_reachable_component_terminal_candidate_count", _NO_DEFAULT),
    ("sampled_region_path_reachable_terminal_rescue_count", _NO_DEFAULT),
    ("sampled_region_path_proxy_goal_anchor_selected_count", _NO_DEFAULT),
    ("sampled_region_path_goal_rescue_candidate_count", _NO_DEFAULT),
    ("sampled_region_path_benefit_surface_present_count", _NO_DEFAULT),
    ("sampled_region_path_path_duplicate_with_baseline_count", _NO_DEFAULT),
    ("sampled_region_path_baseline_equivalent_count", _NO_DEFAULT),
    ("sampled_region_path_no_quality_gain_count", _NO_DEFAULT),
    ("sampled_region_path_fixture_no_benefit_surface_count", _NO_DEFAULT),
    ("sampled_region_path_candidate_missing_metrics_count", _NO_DEFAULT),
    ("sampled_region_path_constrained_connector_failed_count", _NO_DEFAULT),
    ("sampled_region_path_complexity_reason_counts", {}),
    ("sampled_region_path_execution_tie_break_status_counts", {}),
    ("sampled_region_path_execution_tie_break_reason_counts", {}),
    ("sampled_region_path_candidate_audit", []),
)

_ARTIFACT_TRIAGE_FIELDS = ()

_DIAGNOSTIC_INTERPRETATION_FIELDS = (("diagnostic_interpretation", {}),)

_FIELD_GROUPS = (
    _CORE_RUN_FIELDS,
    _ACCEPTANCE_METADATA_FIELDS,
    _IRIS_REGION_GRAPH_FIELDS,
    _CONVEX_REGION_FIELDS,
    _GCS_TRAJECTORY_FIELDS,
    _GCS_CANDIDATE_FIELDS,
    _GCS_CONTROL_POINT_FIELDS,
    _GCS_MOTION_FEASIBILITY_FIELDS,
    _GCS_CURVATURE_CONSTRAINED_FIELDS,
    _CHANNEL_AWARE_ASTAR_FIELDS,
    _SAMPLED_REGION_FIELDS,
    _ARTIFACT_TRIAGE_FIELDS,
    _DIAGNOSTIC_INTERPRETATION_FIELDS,
)


def _copy_group(summary: dict[str, Any], fields: tuple[tuple[str, object], ...]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, default in fields:
        if default is _NO_DEFAULT:
            payload[key] = summary.get(key)
        elif isinstance(default, (dict, list)):
            payload[key] = summary.get(key, default.copy())
        else:
            payload[key] = summary.get(key, default)
    return payload


def compact_summary_payload(
    summary: dict[str, Any],
    *,
    summary_output: Path | None = None,
    report_output: Path | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"status": "completed"}
    for fields in _FIELD_GROUPS:
        payload.update(_copy_group(summary, fields))
    if summary_output is not None:
        payload["summary_output"] = str(summary_output)
    if report_output is not None:
        payload["report_output"] = str(report_output)
    return payload


__all__ = ("compact_summary_payload",)
