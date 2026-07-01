from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any


GCS_CONTROL_POINT_BACKEND = "pydrake_control_point_direction_cone_program"
PLATFORM_GOAL_CONTRACT_MISMATCH_CLASSES = frozenset(
    {
        "platform_inflated_goal_blocked",
        "original_goal_blocked",
        "out_of_bounds",
        "unknown_contract_mismatch",
    }
)


def _diagnostic_aggregate(scenarios: list[dict[str, Any]]) -> dict[str, Any]:
    iris_status_counts: Counter[str] = Counter()
    iris_fallback_reasons: Counter[str] = Counter()
    region_graph_source_counts: Counter[str] = Counter()
    region_graph_fallback_reasons: Counter[str] = Counter()
    convex_region_backend_counts: Counter[str] = Counter()
    convex_region_coverage_status_counts: Counter[str] = Counter()
    convex_region_gcs_ready_reason_counts: Counter[str] = Counter()
    gcs_trajectory_backend_counts: Counter[str] = Counter()
    gcs_trajectory_reason_counts: Counter[str] = Counter()
    gcs_trajectory_result_status_counts: Counter[str] = Counter()
    gcs_candidate_fallback_reason_counts: Counter[str] = Counter()
    gcs_candidate_selection_reason_counts: Counter[str] = Counter()
    gcs_control_point_backend_counts: Counter[str] = Counter()
    gcs_control_point_candidate_fallback_reason_counts: Counter[str] = Counter()
    gcs_control_point_terrain_objective_source_counts: Counter[str] = Counter()
    gcs_motion_status_counts: Counter[str] = Counter()
    gcs_motion_fallback_reason_counts: Counter[str] = Counter()
    gcs_motion_model_counts: Counter[str] = Counter()
    gcs_curvature_status_before_counts: Counter[str] = Counter()
    gcs_curvature_status_after_counts: Counter[str] = Counter()
    gcs_curvature_fallback_reason_counts: Counter[str] = Counter()
    gcs_curvature_repair_strategy_counts: Counter[str] = Counter()
    sampled_status_counts: Counter[str] = Counter()
    sampled_source_counts: Counter[str] = Counter()
    sampled_fallback_reasons: Counter[str] = Counter()
    sampled_start_classification_counts: Counter[str] = Counter()
    sampled_goal_classification_counts: Counter[str] = Counter()
    sampled_connector_strategy_counts: Counter[str] = Counter()
    sampled_bridge_aware_connector_status_counts: Counter[str] = Counter()
    sampled_bridge_aware_fallback_reasons: Counter[str] = Counter()
    sampled_bridge_corridor_status_counts: Counter[str] = Counter()
    sampled_bridge_corridor_fallback_reasons: Counter[str] = Counter()
    sampled_bridge_corridor_radius_counts: Counter[str] = Counter()
    sampled_anchor_closure_status_counts: Counter[str] = Counter()
    sampled_anchor_closure_reason_counts: Counter[str] = Counter()
    sampled_anchor_closure_connection_kind_counts: Counter[str] = Counter()
    sampled_terminal_adjustment_status_counts: Counter[str] = Counter()
    sampled_terminal_adjustment_reason_counts: Counter[str] = Counter()
    sampled_reachable_component_status_counts: Counter[str] = Counter()
    sampled_reachable_component_reason_counts: Counter[str] = Counter()
    sampled_execution_tie_break_status_counts: Counter[str] = Counter()
    sampled_execution_tie_break_reason_counts: Counter[str] = Counter()
    sampled_complexity_reason_counts: Counter[str] = Counter()
    sampled_candidate_audit: list[dict[str, Any]] = []
    channel_aware_astar_diagnostics: list[dict[str, Any]] = []
    group_summary: dict[str, dict[str, Any]] = defaultdict(_empty_group_summary)
    iris_report_count = 0
    iris_fallback_count = 0
    iris_failure_count = 0
    iris_region_count_total = 0
    region_graph_fallback_count = 0
    region_graph_start_goal_disconnected_count = 0
    convex_region_report_count = 0
    convex_region_count_total = 0
    convex_region_fallback_used_count = 0
    convex_region_gcs_ready_count = 0
    convex_region_blocked_cell_violation_count = 0
    convex_region_start_contained_count = 0
    convex_region_goal_contained_count = 0
    convex_region_adjacent_overlap_count = 0
    convex_region_portal_count = 0
    gcs_trajectory_report_count = 0
    gcs_trajectory_attempted_count = 0
    gcs_trajectory_success_count = 0
    gcs_trajectory_collision_count = 0
    gcs_trajectory_region_count_total = 0
    gcs_trajectory_sample_count_total = 0
    gcs_candidate_report_count = 0
    gcs_candidate_attempted_count = 0
    gcs_candidate_available_count = 0
    gcs_candidate_selected_count = 0
    gcs_candidate_collision_count = 0
    gcs_candidate_cost_delta_negative_count = 0
    gcs_candidate_cost_delta_positive_count = 0
    gcs_candidate_cost_delta_zero_count = 0
    gcs_control_point_report_count = 0
    gcs_control_point_attempted_count = 0
    gcs_control_point_success_count = 0
    gcs_control_point_candidate_selected_count = 0
    gcs_control_point_sampled_terrain_cost_stats: list[dict[str, Any]] = []
    gcs_control_point_high_cost_exposure_delta_stats: list[dict[str, Any]] = []
    gcs_motion_report_count = 0
    gcs_motion_evaluated_count = 0
    gcs_motion_feasible_count = 0
    gcs_motion_infeasible_count = 0
    gcs_motion_diagnostic_only_count = 0
    gcs_motion_curvature_violation_count = 0
    gcs_motion_heading_violation_count = 0
    gcs_curvature_report_count = 0
    gcs_curvature_attempted_count = 0
    gcs_curvature_available_count = 0
    gcs_curvature_selected_count = 0
    gcs_curvature_repair_success_count = 0
    gcs_curvature_infeasible_count = 0
    gcs_curvature_diagnostic_only_count = 0
    gcs_curvature_curvature_violation_count_before = 0
    gcs_curvature_curvature_violation_count_after = 0
    gcs_curvature_heading_violation_count_before = 0
    gcs_curvature_heading_violation_count_after = 0
    gcs_curvature_collision_count = 0
    gcs_curvature_region_containment_violation_count = 0
    sampled_selected_count = 0
    sampled_fallback_count = 0
    sampled_sample_attempt_count = 0
    sampled_candidate_ranking_count = 0
    sampled_anchor_region_added_count = 0
    sampled_anchor_region_connected_count = 0
    sampled_anchor_closure_attempt_count = 0
    sampled_anchor_closure_connected_count = 0
    sampled_connector_attempt_count = 0
    sampled_bridge_aware_connector_attempt_count = 0
    sampled_bridge_aware_connector_available_count = 0
    sampled_bridge_aware_connector_selected_count = 0
    sampled_bridge_aware_connector_rejected_count = 0
    sampled_bridge_aware_bridge_cell_count = 0
    sampled_bridge_aware_mask_added_cell_count = 0
    sampled_bridge_corridor_connector_attempt_count = 0
    sampled_bridge_corridor_connector_available_count = 0
    sampled_bridge_corridor_connector_selected_count = 0
    sampled_bridge_corridor_connector_rejected_count = 0
    sampled_bridge_corridor_added_cell_count = 0
    sampled_terminal_adjusted_count = 0
    sampled_terminal_adjustment_candidate_count = 0
    sampled_reachable_component_disconnected_count = 0
    sampled_reachable_component_replacement_selected_count = 0
    sampled_reachable_component_terminal_candidate_count = 0
    sampled_reachable_terminal_rescue_count = 0
    sampled_proxy_goal_anchor_selected_count = 0
    sampled_goal_rescue_candidate_count = 0
    sampled_benefit_surface_present_count = 0
    sampled_path_duplicate_with_baseline_count = 0
    sampled_baseline_equivalent_count = 0
    sampled_no_quality_gain_count = 0
    sampled_fixture_no_benefit_surface_count = 0
    sampled_candidate_missing_metrics_count = 0
    sampled_constrained_connector_failed_count = 0
    convex_region_candidate_audit: list[dict[str, Any]] = []
    gcs_trajectory_candidate_audit: list[dict[str, Any]] = []
    gcs_candidate_audit: list[dict[str, Any]] = []
    gcs_control_point_candidate_audit: list[dict[str, Any]] = []
    gcs_motion_feasibility_audit: list[dict[str, Any]] = []
    gcs_curvature_constrained_audit: list[dict[str, Any]] = []

    for scenario in scenarios:
        group = str(scenario.get("scenario_group") or "unknown")
        group_payload = group_summary[group]
        group_payload["scenario_count"] += 1
        group_payload["candidate_count"] += int(scenario["path_feedback"]["candidate_count"])
        group_payload["reachable_count"] += int(scenario["path_feedback"]["reachable_count"])
        group_payload["failure_count"] += int(scenario["path_feedback"]["failure_count"])
        group_payload["replan_count"] += int(scenario["path_feedback"]["replan_count"])
        group_payload["selection_changed_count"] += int(bool(scenario["selection_changed_by_path_feedback"]))

        iris = scenario["iris_diagnostics"]
        graph = scenario["region_graph_diagnostics"]
        convex = scenario["convex_region_diagnostics"]
        gcs = scenario["gcs_trajectory_diagnostics"]
        gcs_candidate = scenario["gcs_candidate_diagnostics"]
        gcs_control_point = scenario["gcs_control_point_diagnostics"]
        gcs_motion = scenario["gcs_motion_feasibility_diagnostics"]
        gcs_curvature = scenario["gcs_curvature_constrained_diagnostics"]
        sampled = scenario["sampled_region_path_diagnostics"]
        channel_aware = scenario.get("channel_aware_astar_diagnostics")
        channel_aware = channel_aware if isinstance(channel_aware, dict) else {}
        group_payload["iris_report_count"] += int(iris["report_count"])
        group_payload["iris_fallback_count"] += int(iris["fallback_count"])
        group_payload["region_graph_fallback_count"] += int(graph["fallback_count"])
        group_payload["region_graph_start_goal_disconnected_count"] += int(graph["start_goal_disconnected_count"])
        group_payload["convex_region_report_count"] += int(convex["report_count"])
        group_payload["convex_region_count_total"] += int(convex["region_count_total"])
        group_payload["convex_region_fallback_used_count"] += int(convex["fallback_used_count"])
        group_payload["convex_region_gcs_ready_count"] += int(convex["gcs_ready_count"])
        group_payload["convex_region_blocked_cell_violation_count"] += int(
            convex["blocked_cell_violation_count"]
        )
        group_payload["convex_region_start_contained_count"] += int(convex["start_contained_count"])
        group_payload["convex_region_goal_contained_count"] += int(convex["goal_contained_count"])
        group_payload["convex_region_adjacent_overlap_count"] += int(convex["adjacent_overlap_count"])
        group_payload["convex_region_portal_count"] += int(convex["portal_count"])
        group_payload["gcs_trajectory_report_count"] += int(gcs["report_count"])
        group_payload["gcs_trajectory_attempted_count"] += int(gcs["attempted_count"])
        group_payload["gcs_trajectory_success_count"] += int(gcs["success_count"])
        group_payload["gcs_trajectory_collision_count"] += int(gcs["collision_count"])
        group_payload["gcs_trajectory_region_count_total"] += int(gcs["region_count_total"])
        group_payload["gcs_trajectory_sample_count_total"] += int(gcs["sample_count_total"])
        group_payload["gcs_candidate_report_count"] += int(gcs_candidate["report_count"])
        group_payload["gcs_candidate_attempted_count"] += int(gcs_candidate["attempted_count"])
        group_payload["gcs_candidate_available_count"] += int(gcs_candidate["available_count"])
        group_payload["gcs_candidate_selected_count"] += int(gcs_candidate["selected_count"])
        group_payload["gcs_candidate_collision_count"] += int(gcs_candidate["collision_count"])
        group_payload["gcs_candidate_cost_delta_vs_baseline_negative_count"] += int(
            gcs_candidate["cost_delta_vs_baseline_negative_count"]
        )
        group_payload["gcs_candidate_cost_delta_vs_baseline_positive_count"] += int(
            gcs_candidate["cost_delta_vs_baseline_positive_count"]
        )
        group_payload["gcs_candidate_cost_delta_vs_baseline_zero_count"] += int(
            gcs_candidate["cost_delta_vs_baseline_zero_count"]
        )
        group_payload["gcs_control_point_report_count"] += int(gcs_control_point["report_count"])
        group_payload["gcs_control_point_attempted_count"] += int(gcs_control_point["attempted_count"])
        group_payload["gcs_control_point_success_count"] += int(gcs_control_point["success_count"])
        group_payload["gcs_control_point_candidate_selected_count"] += int(
            gcs_control_point["candidate_selected_count"]
        )
        group_payload["gcs_control_point_sampled_terrain_cost_count"] += int(
            gcs_control_point["sampled_terrain_cost_count"]
        )
        group_payload["gcs_control_point_high_cost_exposure_delta_count"] += int(
            gcs_control_point["high_cost_exposure_delta_count"]
        )
        group_payload["gcs_motion_feasibility_report_count"] += int(gcs_motion["report_count"])
        group_payload["gcs_motion_feasibility_evaluated_count"] += int(gcs_motion["evaluated_count"])
        group_payload["gcs_motion_feasibility_feasible_count"] += int(gcs_motion["feasible_count"])
        group_payload["gcs_motion_feasibility_infeasible_count"] += int(gcs_motion["infeasible_count"])
        group_payload["gcs_motion_feasibility_diagnostic_only_count"] += int(
            gcs_motion["diagnostic_only_count"]
        )
        group_payload["gcs_motion_feasibility_curvature_violation_count"] += int(
            gcs_motion["curvature_violation_count"]
        )
        group_payload["gcs_motion_feasibility_heading_violation_count"] += int(
            gcs_motion["heading_violation_count"]
        )
        group_payload["gcs_curvature_constrained_report_count"] += int(gcs_curvature["report_count"])
        group_payload["gcs_curvature_constrained_attempted_count"] += int(gcs_curvature["attempted_count"])
        group_payload["gcs_curvature_constrained_available_count"] += int(gcs_curvature["available_count"])
        group_payload["gcs_curvature_constrained_selected_count"] += int(gcs_curvature["selected_count"])
        group_payload["gcs_curvature_constrained_repair_success_count"] += int(
            gcs_curvature["repair_success_count"]
        )
        group_payload["gcs_curvature_constrained_infeasible_count"] += int(gcs_curvature["infeasible_count"])
        group_payload["gcs_curvature_constrained_diagnostic_only_count"] += int(
            gcs_curvature["diagnostic_only_count"]
        )
        group_payload["gcs_curvature_constrained_curvature_violation_count_before"] += int(
            gcs_curvature["curvature_violation_count_before"]
        )
        group_payload["gcs_curvature_constrained_curvature_violation_count_after"] += int(
            gcs_curvature["curvature_violation_count_after"]
        )
        group_payload["gcs_curvature_constrained_heading_violation_count_before"] += int(
            gcs_curvature["heading_violation_count_before"]
        )
        group_payload["gcs_curvature_constrained_heading_violation_count_after"] += int(
            gcs_curvature["heading_violation_count_after"]
        )
        group_payload["gcs_curvature_constrained_collision_count"] += int(gcs_curvature["collision_count"])
        group_payload["gcs_curvature_constrained_region_containment_violation_count"] += int(
            gcs_curvature["region_containment_violation_count"]
        )
        group_payload["sampled_region_path_selected_count"] += int(sampled["selected_count"])
        group_payload["sampled_region_path_fallback_count"] += int(sampled["fallback_count"])
        group_payload["sampled_region_path_sample_attempt_count"] += int(sampled["sample_attempt_count"])
        group_payload["sampled_region_path_candidate_ranking_count"] += int(sampled["candidate_ranking_count"])
        group_payload["sampled_region_path_anchor_region_added_count"] += int(sampled["anchor_region_added_count"])
        group_payload["sampled_region_path_anchor_region_connected_count"] += int(
            sampled["anchor_region_connected_count"]
        )
        group_payload["sampled_region_path_anchor_closure_attempt_count"] += int(
            sampled["anchor_closure_attempt_count"]
        )
        group_payload["sampled_region_path_anchor_closure_connected_count"] += int(
            sampled["anchor_closure_connected_count"]
        )
        group_payload["sampled_region_path_connector_attempt_count"] += int(sampled["connector_attempt_count"])
        group_payload["sampled_region_path_bridge_aware_connector_attempt_count"] += int(
            sampled["bridge_aware_connector_attempt_count"]
        )
        group_payload["sampled_region_path_bridge_aware_connector_available_count"] += int(
            sampled["bridge_aware_connector_available_count"]
        )
        group_payload["sampled_region_path_bridge_aware_connector_selected_count"] += int(
            sampled["bridge_aware_connector_selected_count"]
        )
        group_payload["sampled_region_path_bridge_aware_connector_rejected_count"] += int(
            sampled["bridge_aware_connector_rejected_count"]
        )
        group_payload["sampled_region_path_bridge_aware_bridge_cell_count"] += int(
            sampled["bridge_aware_bridge_cell_count"]
        )
        group_payload["sampled_region_path_bridge_aware_mask_added_cell_count"] += int(
            sampled["bridge_aware_mask_added_cell_count"]
        )
        group_payload["sampled_region_path_bridge_corridor_connector_attempt_count"] += int(
            sampled["bridge_corridor_connector_attempt_count"]
        )
        group_payload["sampled_region_path_bridge_corridor_connector_available_count"] += int(
            sampled["bridge_corridor_connector_available_count"]
        )
        group_payload["sampled_region_path_bridge_corridor_connector_selected_count"] += int(
            sampled["bridge_corridor_connector_selected_count"]
        )
        group_payload["sampled_region_path_bridge_corridor_connector_rejected_count"] += int(
            sampled["bridge_corridor_connector_rejected_count"]
        )
        group_payload["sampled_region_path_bridge_corridor_added_cell_count"] += int(
            sampled["bridge_corridor_added_cell_count"]
        )
        group_payload["sampled_region_path_terminal_adjusted_count"] += int(sampled["terminal_adjusted_count"])
        group_payload["sampled_region_path_terminal_adjustment_candidate_count"] += int(
            sampled["terminal_adjustment_candidate_count"]
        )
        group_payload["sampled_region_path_reachable_component_disconnected_count"] += int(
            sampled["reachable_component_disconnected_count"]
        )
        group_payload["sampled_region_path_reachable_component_replacement_selected_count"] += int(
            sampled["reachable_component_replacement_selected_count"]
        )
        group_payload["sampled_region_path_reachable_component_terminal_candidate_count"] += int(
            sampled["reachable_component_terminal_candidate_count"]
        )
        group_payload["sampled_region_path_reachable_terminal_rescue_count"] += int(
            sampled["reachable_terminal_rescue_count"]
        )
        group_payload["sampled_region_path_proxy_goal_anchor_selected_count"] += int(
            sampled["proxy_goal_anchor_selected_count"]
        )
        group_payload["sampled_region_path_goal_rescue_candidate_count"] += int(
            sampled["goal_rescue_candidate_count"]
        )
        group_payload["sampled_region_path_benefit_surface_present_count"] += int(
            sampled["benefit_surface_present_count"]
        )
        group_payload["sampled_region_path_path_duplicate_with_baseline_count"] += int(
            sampled["path_duplicate_with_baseline_count"]
        )
        group_payload["sampled_region_path_baseline_equivalent_count"] += int(
            sampled["baseline_equivalent_count"]
        )
        group_payload["sampled_region_path_no_quality_gain_count"] += int(sampled["no_quality_gain_count"])
        group_payload["sampled_region_path_fixture_no_benefit_surface_count"] += int(
            sampled["fixture_no_benefit_surface_count"]
        )
        group_payload["sampled_region_path_candidate_missing_metrics_count"] += int(
            sampled["candidate_missing_metrics_count"]
        )
        group_payload["sampled_region_path_constrained_connector_failed_count"] += int(
            sampled["constrained_connector_failed_count"]
        )
        group_payload["channel_aware_astar_report_count"] += _int_value(
            channel_aware.get("report_count")
        )
        group_payload["channel_aware_astar_selected_count"] += _int_value(
            channel_aware.get("selected_count")
        )
        group_payload["channel_aware_astar_fallback_count"] += _int_value(
            channel_aware.get("fallback_count")
        )
        group_payload["channel_aware_astar_path_changed_count"] += _int_value(
            channel_aware.get("path_changed_count")
        )

        iris_report_count += int(iris["report_count"])
        iris_fallback_count += int(iris["fallback_count"])
        iris_failure_count += int(iris["failure_count"])
        iris_region_count_total += int(iris["region_count_total"])
        region_graph_fallback_count += int(graph["fallback_count"])
        region_graph_start_goal_disconnected_count += int(graph["start_goal_disconnected_count"])
        convex_region_report_count += int(convex["report_count"])
        convex_region_count_total += int(convex["region_count_total"])
        convex_region_fallback_used_count += int(convex["fallback_used_count"])
        convex_region_gcs_ready_count += int(convex["gcs_ready_count"])
        convex_region_blocked_cell_violation_count += int(convex["blocked_cell_violation_count"])
        convex_region_start_contained_count += int(convex["start_contained_count"])
        convex_region_goal_contained_count += int(convex["goal_contained_count"])
        convex_region_adjacent_overlap_count += int(convex["adjacent_overlap_count"])
        convex_region_portal_count += int(convex["portal_count"])
        gcs_trajectory_report_count += int(gcs["report_count"])
        gcs_trajectory_attempted_count += int(gcs["attempted_count"])
        gcs_trajectory_success_count += int(gcs["success_count"])
        gcs_trajectory_collision_count += int(gcs["collision_count"])
        gcs_trajectory_region_count_total += int(gcs["region_count_total"])
        gcs_trajectory_sample_count_total += int(gcs["sample_count_total"])
        gcs_candidate_report_count += int(gcs_candidate["report_count"])
        gcs_candidate_attempted_count += int(gcs_candidate["attempted_count"])
        gcs_candidate_available_count += int(gcs_candidate["available_count"])
        gcs_candidate_selected_count += int(gcs_candidate["selected_count"])
        gcs_candidate_collision_count += int(gcs_candidate["collision_count"])
        gcs_candidate_cost_delta_negative_count += int(gcs_candidate["cost_delta_vs_baseline_negative_count"])
        gcs_candidate_cost_delta_positive_count += int(gcs_candidate["cost_delta_vs_baseline_positive_count"])
        gcs_candidate_cost_delta_zero_count += int(gcs_candidate["cost_delta_vs_baseline_zero_count"])
        gcs_control_point_report_count += int(gcs_control_point["report_count"])
        gcs_control_point_attempted_count += int(gcs_control_point["attempted_count"])
        gcs_control_point_success_count += int(gcs_control_point["success_count"])
        gcs_control_point_candidate_selected_count += int(gcs_control_point["candidate_selected_count"])
        gcs_control_point_sampled_terrain_cost_stats.append(
            {
                "count": gcs_control_point["sampled_terrain_cost_count"],
                "min": gcs_control_point["sampled_terrain_cost_min"],
                "max": gcs_control_point["sampled_terrain_cost_max"],
                "mean": gcs_control_point["sampled_terrain_cost_mean"],
            }
        )
        gcs_control_point_high_cost_exposure_delta_stats.append(
            {
                "count": gcs_control_point["high_cost_exposure_delta_count"],
                "min": gcs_control_point["high_cost_exposure_delta_min"],
                "max": gcs_control_point["high_cost_exposure_delta_max"],
                "mean": gcs_control_point["high_cost_exposure_delta_mean"],
            }
        )
        gcs_motion_report_count += int(gcs_motion["report_count"])
        gcs_motion_evaluated_count += int(gcs_motion["evaluated_count"])
        gcs_motion_feasible_count += int(gcs_motion["feasible_count"])
        gcs_motion_infeasible_count += int(gcs_motion["infeasible_count"])
        gcs_motion_diagnostic_only_count += int(gcs_motion["diagnostic_only_count"])
        gcs_motion_curvature_violation_count += int(gcs_motion["curvature_violation_count"])
        gcs_motion_heading_violation_count += int(gcs_motion["heading_violation_count"])
        gcs_curvature_report_count += int(gcs_curvature["report_count"])
        gcs_curvature_attempted_count += int(gcs_curvature["attempted_count"])
        gcs_curvature_available_count += int(gcs_curvature["available_count"])
        gcs_curvature_selected_count += int(gcs_curvature["selected_count"])
        gcs_curvature_repair_success_count += int(gcs_curvature["repair_success_count"])
        gcs_curvature_infeasible_count += int(gcs_curvature["infeasible_count"])
        gcs_curvature_diagnostic_only_count += int(gcs_curvature["diagnostic_only_count"])
        gcs_curvature_curvature_violation_count_before += int(
            gcs_curvature["curvature_violation_count_before"]
        )
        gcs_curvature_curvature_violation_count_after += int(
            gcs_curvature["curvature_violation_count_after"]
        )
        gcs_curvature_heading_violation_count_before += int(
            gcs_curvature["heading_violation_count_before"]
        )
        gcs_curvature_heading_violation_count_after += int(
            gcs_curvature["heading_violation_count_after"]
        )
        gcs_curvature_collision_count += int(gcs_curvature["collision_count"])
        gcs_curvature_region_containment_violation_count += int(
            gcs_curvature["region_containment_violation_count"]
        )
        sampled_selected_count += int(sampled["selected_count"])
        sampled_fallback_count += int(sampled["fallback_count"])
        sampled_sample_attempt_count += int(sampled["sample_attempt_count"])
        sampled_candidate_ranking_count += int(sampled["candidate_ranking_count"])
        sampled_anchor_region_added_count += int(sampled["anchor_region_added_count"])
        sampled_anchor_region_connected_count += int(sampled["anchor_region_connected_count"])
        sampled_anchor_closure_attempt_count += int(sampled["anchor_closure_attempt_count"])
        sampled_anchor_closure_connected_count += int(sampled["anchor_closure_connected_count"])
        sampled_connector_attempt_count += int(sampled["connector_attempt_count"])
        sampled_bridge_aware_connector_attempt_count += int(sampled["bridge_aware_connector_attempt_count"])
        sampled_bridge_aware_connector_available_count += int(sampled["bridge_aware_connector_available_count"])
        sampled_bridge_aware_connector_selected_count += int(sampled["bridge_aware_connector_selected_count"])
        sampled_bridge_aware_connector_rejected_count += int(sampled["bridge_aware_connector_rejected_count"])
        sampled_bridge_aware_bridge_cell_count += int(sampled["bridge_aware_bridge_cell_count"])
        sampled_bridge_aware_mask_added_cell_count += int(sampled["bridge_aware_mask_added_cell_count"])
        sampled_bridge_corridor_connector_attempt_count += int(sampled["bridge_corridor_connector_attempt_count"])
        sampled_bridge_corridor_connector_available_count += int(
            sampled["bridge_corridor_connector_available_count"]
        )
        sampled_bridge_corridor_connector_selected_count += int(sampled["bridge_corridor_connector_selected_count"])
        sampled_bridge_corridor_connector_rejected_count += int(sampled["bridge_corridor_connector_rejected_count"])
        sampled_bridge_corridor_added_cell_count += int(sampled["bridge_corridor_added_cell_count"])
        sampled_terminal_adjusted_count += int(sampled["terminal_adjusted_count"])
        sampled_terminal_adjustment_candidate_count += int(sampled["terminal_adjustment_candidate_count"])
        sampled_reachable_component_disconnected_count += int(sampled["reachable_component_disconnected_count"])
        sampled_reachable_component_replacement_selected_count += int(
            sampled["reachable_component_replacement_selected_count"]
        )
        sampled_reachable_component_terminal_candidate_count += int(
            sampled["reachable_component_terminal_candidate_count"]
        )
        sampled_reachable_terminal_rescue_count += int(sampled["reachable_terminal_rescue_count"])
        sampled_proxy_goal_anchor_selected_count += int(sampled["proxy_goal_anchor_selected_count"])
        sampled_goal_rescue_candidate_count += int(sampled["goal_rescue_candidate_count"])
        sampled_benefit_surface_present_count += int(sampled["benefit_surface_present_count"])
        sampled_path_duplicate_with_baseline_count += int(sampled["path_duplicate_with_baseline_count"])
        sampled_baseline_equivalent_count += int(sampled["baseline_equivalent_count"])
        sampled_no_quality_gain_count += int(sampled["no_quality_gain_count"])
        sampled_fixture_no_benefit_surface_count += int(sampled["fixture_no_benefit_surface_count"])
        sampled_candidate_missing_metrics_count += int(sampled["candidate_missing_metrics_count"])
        sampled_constrained_connector_failed_count += int(sampled["constrained_connector_failed_count"])
        sampled_candidate_audit.extend(scenario.get("sampled_region_path_candidate_audit", []))
        convex_region_candidate_audit.extend(scenario.get("convex_region_candidate_audit", []))
        gcs_trajectory_candidate_audit.extend(scenario.get("gcs_trajectory_candidate_audit", []))
        gcs_candidate_audit.extend(scenario.get("gcs_candidate_audit", []))
        gcs_control_point_candidate_audit.extend(scenario.get("gcs_control_point_candidate_audit", []))
        gcs_motion_feasibility_audit.extend(scenario.get("gcs_motion_feasibility_audit", []))
        gcs_curvature_constrained_audit.extend(scenario.get("gcs_curvature_constrained_audit", []))
        iris_status_counts.update(iris["status_counts"])
        iris_fallback_reasons.update(iris["fallback_reasons"])
        region_graph_source_counts.update(graph["source_counts"])
        region_graph_fallback_reasons.update(graph["fallback_reasons"])
        convex_region_backend_counts.update(convex["backend_counts"])
        convex_region_coverage_status_counts.update(convex["coverage_status_counts"])
        convex_region_gcs_ready_reason_counts.update(convex["gcs_ready_reason_counts"])
        gcs_trajectory_backend_counts.update(gcs["backend_counts"])
        gcs_trajectory_reason_counts.update(gcs["reason_counts"])
        gcs_trajectory_result_status_counts.update(gcs["result_status_counts"])
        gcs_candidate_fallback_reason_counts.update(gcs_candidate["fallback_reason_counts"])
        gcs_candidate_selection_reason_counts.update(gcs_candidate["selection_reason_counts"])
        gcs_control_point_backend_counts.update(gcs_control_point["backend_counts"])
        gcs_control_point_candidate_fallback_reason_counts.update(
            gcs_control_point["candidate_fallback_reason_counts"]
        )
        gcs_control_point_terrain_objective_source_counts.update(
            gcs_control_point["terrain_objective_source_counts"]
        )
        gcs_motion_status_counts.update(gcs_motion["status_counts"])
        gcs_motion_fallback_reason_counts.update(gcs_motion["fallback_reason_counts"])
        gcs_motion_model_counts.update(gcs_motion["motion_model_counts"])
        gcs_curvature_status_before_counts.update(gcs_curvature["status_before_counts"])
        gcs_curvature_status_after_counts.update(gcs_curvature["status_after_counts"])
        gcs_curvature_fallback_reason_counts.update(gcs_curvature["fallback_reason_counts"])
        gcs_curvature_repair_strategy_counts.update(gcs_curvature["repair_strategy_counts"])
        sampled_status_counts.update(sampled["status_counts"])
        sampled_source_counts.update(sampled["source_counts"])
        sampled_fallback_reasons.update(sampled["fallback_reasons"])
        sampled_start_classification_counts.update(sampled["start_classification_counts"])
        sampled_goal_classification_counts.update(sampled["goal_classification_counts"])
        sampled_anchor_closure_status_counts.update(sampled["anchor_closure_status_counts"])
        sampled_anchor_closure_reason_counts.update(sampled["anchor_closure_reason_counts"])
        sampled_anchor_closure_connection_kind_counts.update(sampled["anchor_closure_connection_kind_counts"])
        sampled_connector_strategy_counts.update(sampled["connector_strategy_counts"])
        sampled_bridge_aware_connector_status_counts.update(sampled["bridge_aware_connector_status_counts"])
        sampled_bridge_aware_fallback_reasons.update(sampled["bridge_aware_fallback_reasons"])
        sampled_bridge_corridor_status_counts.update(sampled["bridge_corridor_status_counts"])
        sampled_bridge_corridor_fallback_reasons.update(sampled["bridge_corridor_fallback_reasons"])
        sampled_bridge_corridor_radius_counts.update(sampled["bridge_corridor_radius_counts"])
        sampled_terminal_adjustment_status_counts.update(sampled["terminal_adjustment_status_counts"])
        sampled_terminal_adjustment_reason_counts.update(sampled["terminal_adjustment_reason_counts"])
        sampled_reachable_component_status_counts.update(sampled["reachable_component_status_counts"])
        sampled_reachable_component_reason_counts.update(sampled["reachable_component_reason_counts"])
        sampled_execution_tie_break_status_counts.update(sampled["execution_tie_break_status_counts"])
        sampled_execution_tie_break_reason_counts.update(sampled["execution_tie_break_reason_counts"])
        sampled_complexity_reason_counts.update(sampled["complexity_reason_counts"])
        channel_aware_astar_diagnostics.append(channel_aware)

    control_point_sampled_stats = _aggregate_metric_stats(gcs_control_point_sampled_terrain_cost_stats)
    control_point_exposure_stats = _aggregate_metric_stats(gcs_control_point_high_cost_exposure_delta_stats)
    channel_aware_astar_summary = _aggregate_channel_aware_astar_diagnostics(
        channel_aware_astar_diagnostics
    )
    return {
        "iris_requested_count": iris_report_count,
        "iris_report_count": iris_report_count,
        "iris_status_counts": dict(sorted(iris_status_counts.items())),
        "iris_fallback_count": iris_fallback_count,
        "iris_failure_count": iris_failure_count,
        "iris_region_count_total": iris_region_count_total,
        "iris_fallback_reasons": dict(sorted(iris_fallback_reasons.items())),
        "region_graph_source_counts": dict(sorted(region_graph_source_counts.items())),
        "region_graph_fallback_count": region_graph_fallback_count,
        "region_graph_fallback_reasons": dict(sorted(region_graph_fallback_reasons.items())),
        "region_graph_start_goal_disconnected_count": region_graph_start_goal_disconnected_count,
        "convex_region_report_count": convex_region_report_count,
        "convex_region_count_total": convex_region_count_total,
        "convex_region_backend_counts": dict(sorted(convex_region_backend_counts.items())),
        "convex_region_fallback_used_count": convex_region_fallback_used_count,
        "convex_region_gcs_ready_count": convex_region_gcs_ready_count,
        "convex_region_blocked_cell_violation_count": convex_region_blocked_cell_violation_count,
        "convex_region_coverage_status_counts": dict(sorted(convex_region_coverage_status_counts.items())),
        "convex_region_gcs_ready_reason_counts": dict(sorted(convex_region_gcs_ready_reason_counts.items())),
        "convex_region_start_contained_count": convex_region_start_contained_count,
        "convex_region_goal_contained_count": convex_region_goal_contained_count,
        "convex_region_adjacent_overlap_count": convex_region_adjacent_overlap_count,
        "convex_region_portal_count": convex_region_portal_count,
        "convex_region_candidate_audit": convex_region_candidate_audit,
        "gcs_trajectory_report_count": gcs_trajectory_report_count,
        "gcs_trajectory_attempted_count": gcs_trajectory_attempted_count,
        "gcs_trajectory_success_count": gcs_trajectory_success_count,
        "gcs_trajectory_collision_count": gcs_trajectory_collision_count,
        "gcs_trajectory_region_count_total": gcs_trajectory_region_count_total,
        "gcs_trajectory_sample_count_total": gcs_trajectory_sample_count_total,
        "gcs_trajectory_backend_counts": dict(sorted(gcs_trajectory_backend_counts.items())),
        "gcs_trajectory_reason_counts": dict(sorted(gcs_trajectory_reason_counts.items())),
        "gcs_trajectory_result_status_counts": dict(sorted(gcs_trajectory_result_status_counts.items())),
        "gcs_trajectory_candidate_audit": gcs_trajectory_candidate_audit,
        "gcs_candidate_report_count": gcs_candidate_report_count,
        "gcs_candidate_attempted_count": gcs_candidate_attempted_count,
        "gcs_candidate_available_count": gcs_candidate_available_count,
        "gcs_candidate_selected_count": gcs_candidate_selected_count,
        "gcs_candidate_collision_count": gcs_candidate_collision_count,
        "gcs_candidate_fallback_reason_counts": dict(sorted(gcs_candidate_fallback_reason_counts.items())),
        "gcs_candidate_selection_reason_counts": dict(sorted(gcs_candidate_selection_reason_counts.items())),
        "gcs_candidate_cost_delta_vs_baseline_negative_count": gcs_candidate_cost_delta_negative_count,
        "gcs_candidate_cost_delta_vs_baseline_positive_count": gcs_candidate_cost_delta_positive_count,
        "gcs_candidate_cost_delta_vs_baseline_zero_count": gcs_candidate_cost_delta_zero_count,
        "gcs_candidate_audit": gcs_candidate_audit,
        "gcs_control_point_report_count": gcs_control_point_report_count,
        "gcs_control_point_attempted_count": gcs_control_point_attempted_count,
        "gcs_control_point_success_count": gcs_control_point_success_count,
        "gcs_control_point_backend_counts": dict(sorted(gcs_control_point_backend_counts.items())),
        "gcs_control_point_candidate_selected_count": gcs_control_point_candidate_selected_count,
        "gcs_control_point_candidate_fallback_reason_counts": dict(
            sorted(gcs_control_point_candidate_fallback_reason_counts.items())
        ),
        "gcs_control_point_terrain_objective_source_counts": dict(
            sorted(gcs_control_point_terrain_objective_source_counts.items())
        ),
        "gcs_control_point_sampled_terrain_cost_count": control_point_sampled_stats["count"],
        "gcs_control_point_sampled_terrain_cost_min": control_point_sampled_stats["min"],
        "gcs_control_point_sampled_terrain_cost_max": control_point_sampled_stats["max"],
        "gcs_control_point_sampled_terrain_cost_mean": control_point_sampled_stats["mean"],
        "gcs_control_point_high_cost_exposure_delta_count": control_point_exposure_stats["count"],
        "gcs_control_point_high_cost_exposure_delta_min": control_point_exposure_stats["min"],
        "gcs_control_point_high_cost_exposure_delta_max": control_point_exposure_stats["max"],
        "gcs_control_point_high_cost_exposure_delta_mean": control_point_exposure_stats["mean"],
        "gcs_control_point_candidate_audit": gcs_control_point_candidate_audit,
        "gcs_motion_feasibility_report_count": gcs_motion_report_count,
        "gcs_motion_feasibility_evaluated_count": gcs_motion_evaluated_count,
        "gcs_motion_feasibility_feasible_count": gcs_motion_feasible_count,
        "gcs_motion_feasibility_infeasible_count": gcs_motion_infeasible_count,
        "gcs_motion_feasibility_diagnostic_only_count": gcs_motion_diagnostic_only_count,
        "gcs_motion_feasibility_curvature_violation_count": gcs_motion_curvature_violation_count,
        "gcs_motion_feasibility_heading_violation_count": gcs_motion_heading_violation_count,
        "gcs_motion_feasibility_status_counts": dict(sorted(gcs_motion_status_counts.items())),
        "gcs_motion_feasibility_fallback_reason_counts": dict(
            sorted(gcs_motion_fallback_reason_counts.items())
        ),
        "gcs_motion_feasibility_motion_model_counts": dict(sorted(gcs_motion_model_counts.items())),
        "gcs_motion_feasibility_audit": gcs_motion_feasibility_audit,
        "gcs_curvature_constrained_report_count": gcs_curvature_report_count,
        "gcs_curvature_constrained_attempted_count": gcs_curvature_attempted_count,
        "gcs_curvature_constrained_available_count": gcs_curvature_available_count,
        "gcs_curvature_constrained_selected_count": gcs_curvature_selected_count,
        "gcs_curvature_constrained_repair_success_count": gcs_curvature_repair_success_count,
        "gcs_curvature_constrained_infeasible_count": gcs_curvature_infeasible_count,
        "gcs_curvature_constrained_diagnostic_only_count": gcs_curvature_diagnostic_only_count,
        "gcs_curvature_constrained_curvature_violation_count_before": (
            gcs_curvature_curvature_violation_count_before
        ),
        "gcs_curvature_constrained_curvature_violation_count_after": (
            gcs_curvature_curvature_violation_count_after
        ),
        "gcs_curvature_constrained_heading_violation_count_before": (
            gcs_curvature_heading_violation_count_before
        ),
        "gcs_curvature_constrained_heading_violation_count_after": (
            gcs_curvature_heading_violation_count_after
        ),
        "gcs_curvature_constrained_collision_count": gcs_curvature_collision_count,
        "gcs_curvature_constrained_region_containment_violation_count": (
            gcs_curvature_region_containment_violation_count
        ),
        "gcs_curvature_constrained_status_before_counts": dict(
            sorted(gcs_curvature_status_before_counts.items())
        ),
        "gcs_curvature_constrained_status_after_counts": dict(
            sorted(gcs_curvature_status_after_counts.items())
        ),
        "gcs_curvature_constrained_fallback_reason_counts": dict(
            sorted(gcs_curvature_fallback_reason_counts.items())
        ),
        "gcs_curvature_constrained_repair_strategy_counts": dict(
            sorted(gcs_curvature_repair_strategy_counts.items())
        ),
        "gcs_curvature_constrained_audit": gcs_curvature_constrained_audit,
        **_channel_aware_astar_prefixed_fields(channel_aware_astar_summary),
        "sampled_region_path_selected_count": sampled_selected_count,
        "sampled_region_path_fallback_count": sampled_fallback_count,
        "sampled_region_path_status_counts": dict(sorted(sampled_status_counts.items())),
        "sampled_region_path_source_counts": dict(sorted(sampled_source_counts.items())),
        "sampled_region_path_fallback_reasons": dict(sorted(sampled_fallback_reasons.items())),
        "sampled_region_path_sample_attempt_count": sampled_sample_attempt_count,
        "sampled_region_path_candidate_ranking_count": sampled_candidate_ranking_count,
        "sampled_region_path_anchor_region_added_count": sampled_anchor_region_added_count,
        "sampled_region_path_anchor_region_connected_count": sampled_anchor_region_connected_count,
        "sampled_region_path_anchor_closure_attempt_count": sampled_anchor_closure_attempt_count,
        "sampled_region_path_anchor_closure_connected_count": sampled_anchor_closure_connected_count,
        "sampled_region_path_anchor_closure_status_counts": dict(
            sorted(sampled_anchor_closure_status_counts.items())
        ),
        "sampled_region_path_anchor_closure_reason_counts": dict(
            sorted(sampled_anchor_closure_reason_counts.items())
        ),
        "sampled_region_path_anchor_closure_connection_kind_counts": dict(
            sorted(sampled_anchor_closure_connection_kind_counts.items())
        ),
        "sampled_region_path_start_classification_counts": dict(sorted(sampled_start_classification_counts.items())),
        "sampled_region_path_goal_classification_counts": dict(sorted(sampled_goal_classification_counts.items())),
        "sampled_region_path_connector_attempt_count": sampled_connector_attempt_count,
        "sampled_region_path_connector_strategy_counts": dict(sorted(sampled_connector_strategy_counts.items())),
        "sampled_region_path_bridge_aware_connector_attempt_count": sampled_bridge_aware_connector_attempt_count,
        "sampled_region_path_bridge_aware_connector_available_count": sampled_bridge_aware_connector_available_count,
        "sampled_region_path_bridge_aware_connector_selected_count": sampled_bridge_aware_connector_selected_count,
        "sampled_region_path_bridge_aware_connector_rejected_count": sampled_bridge_aware_connector_rejected_count,
        "sampled_region_path_bridge_aware_connector_status_counts": dict(
            sorted(sampled_bridge_aware_connector_status_counts.items())
        ),
        "sampled_region_path_bridge_aware_fallback_reasons": dict(
            sorted(sampled_bridge_aware_fallback_reasons.items())
        ),
        "sampled_region_path_bridge_aware_bridge_cell_count": sampled_bridge_aware_bridge_cell_count,
        "sampled_region_path_bridge_aware_mask_added_cell_count": sampled_bridge_aware_mask_added_cell_count,
        "sampled_region_path_bridge_corridor_connector_attempt_count": (
            sampled_bridge_corridor_connector_attempt_count
        ),
        "sampled_region_path_bridge_corridor_connector_available_count": (
            sampled_bridge_corridor_connector_available_count
        ),
        "sampled_region_path_bridge_corridor_connector_selected_count": (
            sampled_bridge_corridor_connector_selected_count
        ),
        "sampled_region_path_bridge_corridor_connector_rejected_count": (
            sampled_bridge_corridor_connector_rejected_count
        ),
        "sampled_region_path_bridge_corridor_status_counts": dict(
            sorted(sampled_bridge_corridor_status_counts.items())
        ),
        "sampled_region_path_bridge_corridor_fallback_reasons": dict(
            sorted(sampled_bridge_corridor_fallback_reasons.items())
        ),
        "sampled_region_path_bridge_corridor_radius_counts": dict(
            sorted(sampled_bridge_corridor_radius_counts.items())
        ),
        "sampled_region_path_bridge_corridor_added_cell_count": sampled_bridge_corridor_added_cell_count,
        "sampled_region_path_terminal_adjusted_count": sampled_terminal_adjusted_count,
        "sampled_region_path_terminal_adjustment_candidate_count": sampled_terminal_adjustment_candidate_count,
        "sampled_region_path_terminal_adjustment_status_counts": dict(
            sorted(sampled_terminal_adjustment_status_counts.items())
        ),
        "sampled_region_path_terminal_adjustment_reason_counts": dict(
            sorted(sampled_terminal_adjustment_reason_counts.items())
        ),
        "sampled_region_path_reachable_component_status_counts": dict(
            sorted(sampled_reachable_component_status_counts.items())
        ),
        "sampled_region_path_reachable_component_reason_counts": dict(
            sorted(sampled_reachable_component_reason_counts.items())
        ),
        "sampled_region_path_reachable_component_disconnected_count": (
            sampled_reachable_component_disconnected_count
        ),
        "sampled_region_path_reachable_component_replacement_selected_count": (
            sampled_reachable_component_replacement_selected_count
        ),
        "sampled_region_path_reachable_component_terminal_candidate_count": (
            sampled_reachable_component_terminal_candidate_count
        ),
        "sampled_region_path_reachable_terminal_rescue_count": sampled_reachable_terminal_rescue_count,
        "sampled_region_path_proxy_goal_anchor_selected_count": sampled_proxy_goal_anchor_selected_count,
        "sampled_region_path_goal_rescue_candidate_count": sampled_goal_rescue_candidate_count,
        "sampled_region_path_benefit_surface_present_count": sampled_benefit_surface_present_count,
        "sampled_region_path_path_duplicate_with_baseline_count": sampled_path_duplicate_with_baseline_count,
        "sampled_region_path_baseline_equivalent_count": sampled_baseline_equivalent_count,
        "sampled_region_path_no_quality_gain_count": sampled_no_quality_gain_count,
        "sampled_region_path_fixture_no_benefit_surface_count": sampled_fixture_no_benefit_surface_count,
        "sampled_region_path_candidate_missing_metrics_count": sampled_candidate_missing_metrics_count,
        "sampled_region_path_constrained_connector_failed_count": sampled_constrained_connector_failed_count,
        "sampled_region_path_complexity_reason_counts": dict(sorted(sampled_complexity_reason_counts.items())),
        "sampled_region_path_execution_tie_break_status_counts": dict(
            sorted(sampled_execution_tie_break_status_counts.items())
        ),
        "sampled_region_path_execution_tie_break_reason_counts": dict(
            sorted(sampled_execution_tie_break_reason_counts.items())
        ),
        "sampled_region_path_candidate_audit": sampled_candidate_audit,
        "scenario_group_summary": {
            group: dict(payload)
            for group, payload in sorted(group_summary.items())
        },
    }


def _diagnostic_interpretation_summary(scenarios: list[dict[str, Any]]) -> dict[str, Any]:
    group_summary: dict[str, dict[str, Counter[str]]] = defaultdict(_empty_group_interpretation)
    for scenario in scenarios:
        group = str(scenario.get("scenario_group") or "unknown")
        interpretation = scenario["diagnostic_interpretation"]
        group_payload = group_summary[group]
        group_payload["target_replacement_reasons"].update(
            [str(interpretation["target_replacement_reason"])]
        )
        group_payload["iris_region_graph_signal_counts"].update(
            [str(interpretation["iris_region_graph_signal"])]
        )
        for source in interpretation["failure_sources"]:
            group_payload["failure_sources"].update([str(source)])

    return {
        "scenario_group_interpretation": {
            group: {
                "target_replacement_reasons": dict(sorted(payload["target_replacement_reasons"].items())),
                "failure_sources": dict(sorted(payload["failure_sources"].items())),
                "iris_region_graph_signal_counts": dict(
                    sorted(payload["iris_region_graph_signal_counts"].items())
                ),
            }
            for group, payload in sorted(group_summary.items())
        }
    }


def _empty_group_interpretation() -> dict[str, Counter[str]]:
    return {
        "target_replacement_reasons": Counter(),
        "failure_sources": Counter(),
        "iris_region_graph_signal_counts": Counter(),
    }


def _scenario_diagnostic_interpretation(scenario: dict[str, Any]) -> dict[str, Any]:
    candidates = scenario["path_feedback"]["candidates"]
    before_candidate = _candidate_by_cell(candidates, scenario["selected_cell_before_path_feedback"])
    after_candidate = _candidate_by_cell(candidates, scenario["selected_cell_after_path_feedback"])
    failure_sources = _scenario_failure_sources(scenario)
    return {
        "target_replacement_reason": _target_replacement_reason(
            scenario,
            before_candidate=before_candidate,
            after_candidate=after_candidate,
        ),
        "failure_sources": failure_sources,
        "primary_failure_reason": _primary_failure_reason(scenario),
        "iris_region_graph_signal": _iris_region_graph_signal(scenario),
        "selected_after_feasible": None if after_candidate is None else bool(after_candidate["reachable"]),
        "selected_after_replan_required": None
        if after_candidate is None
        else bool(after_candidate["replan_required"]),
        "open_grid_fallback_used": bool(scenario["open_grid_fallback_used"]),
    }


def _target_replacement_reason(
    scenario: dict[str, Any],
    *,
    before_candidate: dict[str, Any] | None,
    after_candidate: dict[str, Any] | None,
) -> str:
    if not scenario["selection_changed_by_path_feedback"]:
        return "unchanged"
    if after_candidate is None:
        return "no_feasible_candidate_after_path_feedback"
    before_flags = (
        before_candidate.get("diagnostic_interpretation", {}).get("diagnostic_flags", [])
        if before_candidate is not None
        else []
    )
    if "path_planning_failure" in before_flags:
        return "before_candidate_path_planning_failed"
    if "region_graph_disconnected" in before_flags:
        return "before_candidate_region_graph_disconnected"
    if "region_graph_fallback" in before_flags:
        return "before_candidate_region_graph_fallback"
    if "iris_fallback" in before_flags:
        return "before_candidate_iris_fallback"
    if "replan_required" in before_flags:
        return "before_candidate_replan_required"
    before_cost = scenario["selected_path_cost_before_feedback"]
    after_cost = scenario["selected_path_cost_after_feedback"]
    if before_cost is None:
        return "selected_before_not_evaluated_or_infeasible"
    if after_cost is not None and float(after_cost) < float(before_cost):
        return "lower_path_cost_candidate"
    return "path_feedback_tiebreak"


def _scenario_failure_sources(scenario: dict[str, Any]) -> list[str]:
    sources: list[str] = []
    feedback = scenario["path_feedback"]
    if int(feedback["failure_count"]) > 0:
        sources.append("path_planning_failure")
    if int(feedback["replan_count"]) > 0:
        sources.append("replan_required")
    if int(scenario["tracking_safety_violation_count"]) > 0:
        sources.append("tracking_safety_violation")
    if int(scenario["trajectory_optimization_fallback_count"]) > 0:
        sources.append("trajectory_optimization_fallback")
    if int(scenario["region_graph_disconnected_count"]) > 0:
        sources.append("region_graph_disconnected")
    if int(scenario["region_graph_diagnostics"]["fallback_count"]) > 0:
        sources.append("region_graph_fallback")
    if int(scenario["sampled_region_path_diagnostics"]["fallback_count"]) > 0:
        sources.append("sampled_region_path_fallback")
    if scenario["gcs_curvature_constrained_diagnostics"].get("fallback_reason_counts"):
        sources.append("gcs_curvature_constrained_fallback")
    if scenario["gcs_control_point_diagnostics"].get("candidate_fallback_reason_counts"):
        sources.append("gcs_control_point_fallback")
    if int(scenario["iris_diagnostics"]["fallback_count"]) > 0:
        sources.append("iris_fallback")
    if bool(scenario["open_grid_fallback_used"]):
        sources.append("open_grid_fallback")
    return sources or ["none"]


def _primary_failure_reason(scenario: dict[str, Any]) -> str | None:
    reasons = scenario["path_feedback"].get("failure_reasons", [])
    if not reasons:
        graph_reasons = scenario["region_graph_diagnostics"].get("fallback_reasons", {})
        if graph_reasons:
            return next(iter(graph_reasons))
        iris_reasons = scenario["iris_diagnostics"].get("fallback_reasons", {})
        if iris_reasons:
            return next(iter(iris_reasons))
        sampled_reasons = scenario["sampled_region_path_diagnostics"].get("fallback_reasons", {})
        if sampled_reasons:
            return next(iter(sampled_reasons))
        constrained_reasons = scenario["gcs_curvature_constrained_diagnostics"].get("fallback_reason_counts", {})
        if constrained_reasons:
            return next(iter(constrained_reasons))
        return None
    counts = Counter(str(reason) for reason in reasons)
    return counts.most_common(1)[0][0]


def _iris_region_graph_signal(scenario: dict[str, Any]) -> str:
    if (
        int(scenario["region_graph_disconnected_count"]) > 0
        or int(scenario["region_graph_diagnostics"]["fallback_count"]) > 0
        or int(scenario["iris_diagnostics"]["fallback_count"]) > 0
        or int(scenario["sampled_region_path_diagnostics"]["fallback_count"]) > 0
    ):
        return "diagnostic_explains_replan_or_failure"
    if scenario["region_graph_diagnostics"]["source_counts"] or int(scenario["iris_diagnostics"]["report_count"]) > 0:
        return "diagnostic_present"
    return "not_reported"


def _candidate_by_cell(candidates: list[dict[str, Any]], cell: list[int] | None) -> dict[str, Any] | None:
    if cell is None:
        return None
    for candidate in candidates:
        if candidate.get("cell") == cell:
            return candidate
    return None


def _list_text(values: Any) -> str:
    if not values:
        return "none"
    if isinstance(values, dict):
        return ", ".join(f"{key}:{value}" for key, value in values.items()) or "none"
    if isinstance(values, list | tuple | set):
        return ", ".join(str(value) for value in values) or "none"
    return str(values)


def _empty_group_summary() -> dict[str, int]:
    return {
        "scenario_count": 0,
        "candidate_count": 0,
        "reachable_count": 0,
        "failure_count": 0,
        "replan_count": 0,
        "selection_changed_count": 0,
        "iris_report_count": 0,
        "iris_fallback_count": 0,
        "region_graph_fallback_count": 0,
        "region_graph_start_goal_disconnected_count": 0,
        "convex_region_report_count": 0,
        "convex_region_count_total": 0,
        "convex_region_fallback_used_count": 0,
        "convex_region_gcs_ready_count": 0,
        "convex_region_blocked_cell_violation_count": 0,
        "convex_region_start_contained_count": 0,
        "convex_region_goal_contained_count": 0,
        "convex_region_adjacent_overlap_count": 0,
        "convex_region_portal_count": 0,
        "gcs_trajectory_report_count": 0,
        "gcs_trajectory_attempted_count": 0,
        "gcs_trajectory_success_count": 0,
        "gcs_trajectory_collision_count": 0,
        "gcs_trajectory_region_count_total": 0,
        "gcs_trajectory_sample_count_total": 0,
        "gcs_candidate_report_count": 0,
        "gcs_candidate_attempted_count": 0,
        "gcs_candidate_available_count": 0,
        "gcs_candidate_selected_count": 0,
        "gcs_candidate_collision_count": 0,
        "gcs_candidate_cost_delta_vs_baseline_negative_count": 0,
        "gcs_candidate_cost_delta_vs_baseline_positive_count": 0,
        "gcs_candidate_cost_delta_vs_baseline_zero_count": 0,
        "gcs_control_point_report_count": 0,
        "gcs_control_point_attempted_count": 0,
        "gcs_control_point_success_count": 0,
        "gcs_control_point_candidate_selected_count": 0,
        "gcs_control_point_sampled_terrain_cost_count": 0,
        "gcs_control_point_high_cost_exposure_delta_count": 0,
        "gcs_motion_feasibility_report_count": 0,
        "gcs_motion_feasibility_evaluated_count": 0,
        "gcs_motion_feasibility_feasible_count": 0,
        "gcs_motion_feasibility_infeasible_count": 0,
        "gcs_motion_feasibility_diagnostic_only_count": 0,
        "gcs_motion_feasibility_curvature_violation_count": 0,
        "gcs_motion_feasibility_heading_violation_count": 0,
        "gcs_curvature_constrained_report_count": 0,
        "gcs_curvature_constrained_attempted_count": 0,
        "gcs_curvature_constrained_available_count": 0,
        "gcs_curvature_constrained_selected_count": 0,
        "gcs_curvature_constrained_repair_success_count": 0,
        "gcs_curvature_constrained_infeasible_count": 0,
        "gcs_curvature_constrained_diagnostic_only_count": 0,
        "gcs_curvature_constrained_curvature_violation_count_before": 0,
        "gcs_curvature_constrained_curvature_violation_count_after": 0,
        "gcs_curvature_constrained_heading_violation_count_before": 0,
        "gcs_curvature_constrained_heading_violation_count_after": 0,
        "gcs_curvature_constrained_collision_count": 0,
        "gcs_curvature_constrained_region_containment_violation_count": 0,
        "sampled_region_path_selected_count": 0,
        "sampled_region_path_fallback_count": 0,
        "sampled_region_path_sample_attempt_count": 0,
        "sampled_region_path_candidate_ranking_count": 0,
        "sampled_region_path_anchor_region_added_count": 0,
        "sampled_region_path_anchor_region_connected_count": 0,
        "sampled_region_path_anchor_closure_attempt_count": 0,
        "sampled_region_path_anchor_closure_connected_count": 0,
        "sampled_region_path_connector_attempt_count": 0,
        "sampled_region_path_bridge_aware_connector_attempt_count": 0,
        "sampled_region_path_bridge_aware_connector_available_count": 0,
        "sampled_region_path_bridge_aware_connector_selected_count": 0,
        "sampled_region_path_bridge_aware_connector_rejected_count": 0,
        "sampled_region_path_bridge_aware_bridge_cell_count": 0,
        "sampled_region_path_bridge_aware_mask_added_cell_count": 0,
        "sampled_region_path_bridge_corridor_connector_attempt_count": 0,
        "sampled_region_path_bridge_corridor_connector_available_count": 0,
        "sampled_region_path_bridge_corridor_connector_selected_count": 0,
        "sampled_region_path_bridge_corridor_connector_rejected_count": 0,
        "sampled_region_path_bridge_corridor_added_cell_count": 0,
        "sampled_region_path_terminal_adjusted_count": 0,
        "sampled_region_path_terminal_adjustment_candidate_count": 0,
        "sampled_region_path_reachable_component_disconnected_count": 0,
        "sampled_region_path_reachable_component_replacement_selected_count": 0,
        "sampled_region_path_reachable_component_terminal_candidate_count": 0,
        "sampled_region_path_reachable_terminal_rescue_count": 0,
        "sampled_region_path_proxy_goal_anchor_selected_count": 0,
        "sampled_region_path_goal_rescue_candidate_count": 0,
        "sampled_region_path_benefit_surface_present_count": 0,
        "sampled_region_path_path_duplicate_with_baseline_count": 0,
        "sampled_region_path_baseline_equivalent_count": 0,
        "sampled_region_path_no_quality_gain_count": 0,
        "sampled_region_path_fixture_no_benefit_surface_count": 0,
        "sampled_region_path_candidate_missing_metrics_count": 0,
        "sampled_region_path_constrained_connector_failed_count": 0,
        "channel_aware_astar_report_count": 0,
        "channel_aware_astar_selected_count": 0,
        "channel_aware_astar_fallback_count": 0,
        "channel_aware_astar_path_changed_count": 0,
    }


def _iris_diagnostics(evaluations) -> dict[str, Any]:
    status_counts: Counter[str] = Counter()
    fallback_reasons: Counter[str] = Counter()
    report_count = 0
    fallback_count = 0
    failure_count = 0
    region_count_total = 0
    for item in evaluations:
        iris = item.to_dict().get("iris_region")
        if not isinstance(iris, dict):
            continue
        report_count += 1
        status = str(iris.get("status") or "unknown")
        status_counts[status] += 1
        region_count_total += _int_value(iris.get("region_count"))
        if bool(iris.get("fallback_used")):
            fallback_count += 1
        if status == "failed":
            failure_count += 1
        reason = iris.get("failure_reason")
        if reason:
            fallback_reasons[str(reason)] += 1
    return {
        "report_count": report_count,
        "status_counts": dict(sorted(status_counts.items())),
        "fallback_count": fallback_count,
        "failure_count": failure_count,
        "region_count_total": region_count_total,
        "fallback_reasons": dict(sorted(fallback_reasons.items())),
    }


def _region_graph_diagnostics(evaluations) -> dict[str, Any]:
    source_counts: Counter[str] = Counter()
    fallback_reasons: Counter[str] = Counter()
    fallback_count = 0
    start_goal_disconnected_count = 0
    for item in evaluations:
        graph = item.to_dict().get("region_graph")
        if not isinstance(graph, dict):
            continue
        source_counts[str(graph.get("graph_source") or graph.get("region_source") or "unknown")] += 1
        if bool(graph.get("fallback_used")):
            fallback_count += 1
        if graph.get("start_goal_connected") is False:
            start_goal_disconnected_count += 1
        reason = graph.get("fallback_reason")
        if reason:
            fallback_reasons[str(reason)] += 1
    return {
        "source_counts": dict(sorted(source_counts.items())),
        "fallback_count": fallback_count,
        "fallback_reasons": dict(sorted(fallback_reasons.items())),
        "start_goal_disconnected_count": start_goal_disconnected_count,
    }


def _convex_region_diagnostics(evaluations) -> dict[str, Any]:
    backend_counts: Counter[str] = Counter()
    coverage_status_counts: Counter[str] = Counter()
    gcs_ready_reason_counts: Counter[str] = Counter()
    report_count = 0
    region_count_total = 0
    fallback_used_count = 0
    gcs_ready_count = 0
    blocked_cell_violation_count = 0
    start_contained_count = 0
    goal_contained_count = 0
    adjacent_overlap_count = 0
    portal_count = 0
    for item in evaluations:
        convex = item.to_dict().get("convex_region")
        if not isinstance(convex, dict):
            continue
        report_count += 1
        backend = convex.get("backend")
        if backend:
            backend_counts[str(backend)] += 1
        coverage_status = convex.get("coverage_status")
        if coverage_status:
            coverage_status_counts[str(coverage_status)] += 1
        reason = convex.get("gcs_ready_reason")
        if reason:
            gcs_ready_reason_counts[str(reason)] += 1
        region_count_total += _int_value(convex.get("region_count"))
        if convex.get("fallback_used") is True:
            fallback_used_count += 1
        if convex.get("gcs_ready") is True:
            gcs_ready_count += 1
        if convex.get("start_contained") is True:
            start_contained_count += 1
        if convex.get("goal_contained") is True:
            goal_contained_count += 1
        blocked_cell_violation_count += _int_value(convex.get("blocked_cell_violation_count"))
        adjacent_overlap_count += _int_value(convex.get("adjacent_overlap_count"))
        portal_count += _int_value(convex.get("portal_count"))
    return {
        "report_count": report_count,
        "region_count_total": region_count_total,
        "backend_counts": dict(sorted(backend_counts.items())),
        "fallback_used_count": fallback_used_count,
        "gcs_ready_count": gcs_ready_count,
        "blocked_cell_violation_count": blocked_cell_violation_count,
        "coverage_status_counts": dict(sorted(coverage_status_counts.items())),
        "gcs_ready_reason_counts": dict(sorted(gcs_ready_reason_counts.items())),
        "start_contained_count": start_contained_count,
        "goal_contained_count": goal_contained_count,
        "adjacent_overlap_count": adjacent_overlap_count,
        "portal_count": portal_count,
    }


def _gcs_trajectory_diagnostics(evaluations) -> dict[str, Any]:
    backend_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    result_status_counts: Counter[str] = Counter()
    report_count = 0
    attempted_count = 0
    success_count = 0
    collision_count = 0
    region_count_total = 0
    sample_count_total = 0
    for item in evaluations:
        gcs = item.to_dict().get("gcs_trajectory")
        if not isinstance(gcs, dict):
            continue
        report_count += 1
        backend = gcs.get("backend")
        if backend:
            backend_counts[str(backend)] += 1
        reason = gcs.get("reason")
        if reason:
            reason_counts[str(reason)] += 1
        result_status = gcs.get("result_status")
        if result_status:
            result_status_counts[str(result_status)] += 1
        if gcs.get("attempted") is True:
            attempted_count += 1
        if gcs.get("success") is True:
            success_count += 1
        collision_count += _int_value(gcs.get("collision_count"))
        region_count_total += _int_value(gcs.get("region_count"))
        sample_count_total += _int_value(gcs.get("sample_count"))
    return {
        "report_count": report_count,
        "attempted_count": attempted_count,
        "success_count": success_count,
        "collision_count": collision_count,
        "region_count_total": region_count_total,
        "sample_count_total": sample_count_total,
        "backend_counts": dict(sorted(backend_counts.items())),
        "reason_counts": dict(sorted(reason_counts.items())),
        "result_status_counts": dict(sorted(result_status_counts.items())),
    }


def _gcs_candidate_diagnostics(evaluations) -> dict[str, Any]:
    fallback_reason_counts: Counter[str] = Counter()
    selection_reason_counts: Counter[str] = Counter()
    report_count = 0
    attempted_count = 0
    available_count = 0
    selected_count = 0
    collision_count = 0
    delta_negative_count = 0
    delta_positive_count = 0
    delta_zero_count = 0
    for item in evaluations:
        candidate = item.to_dict().get("gcs_candidate")
        if not isinstance(candidate, dict):
            continue
        report_count += 1
        if candidate.get("attempted") is True:
            attempted_count += 1
        if candidate.get("available") is True:
            available_count += 1
        if candidate.get("selected") is True:
            selected_count += 1
        collision_count += _int_value(candidate.get("collision_count"))
        fallback_reason = candidate.get("fallback_reason")
        if fallback_reason:
            fallback_reason_counts[str(fallback_reason)] += 1
        selection_reason = candidate.get("selection_reason")
        if selection_reason:
            selection_reason_counts[str(selection_reason)] += 1
        delta = _float_value(candidate.get("cost_delta_vs_baseline"))
        if delta is None:
            continue
        if delta < 0.0:
            delta_negative_count += 1
        elif delta > 0.0:
            delta_positive_count += 1
        else:
            delta_zero_count += 1
    return {
        "report_count": report_count,
        "attempted_count": attempted_count,
        "available_count": available_count,
        "selected_count": selected_count,
        "collision_count": collision_count,
        "fallback_reason_counts": dict(sorted(fallback_reason_counts.items())),
        "selection_reason_counts": dict(sorted(selection_reason_counts.items())),
        "cost_delta_vs_baseline_negative_count": delta_negative_count,
        "cost_delta_vs_baseline_positive_count": delta_positive_count,
        "cost_delta_vs_baseline_zero_count": delta_zero_count,
    }


def _gcs_control_point_diagnostics(evaluations) -> dict[str, Any]:
    backend_counts: Counter[str] = Counter()
    fallback_reason_counts: Counter[str] = Counter()
    terrain_objective_source_counts: Counter[str] = Counter()
    report_count = 0
    attempted_count = 0
    success_count = 0
    candidate_selected_count = 0
    sampled_terrain_costs: list[float] = []
    high_cost_exposure_deltas: list[float] = []
    for item in evaluations:
        payload = item.to_dict()
        gcs = payload.get("gcs_trajectory")
        if not isinstance(gcs, dict) or gcs.get("backend") != GCS_CONTROL_POINT_BACKEND:
            continue
        report_count += 1
        backend_counts[str(gcs.get("backend"))] += 1
        if gcs.get("attempted") is True:
            attempted_count += 1
        if gcs.get("success") is True:
            success_count += 1

        candidate = payload.get("gcs_candidate")
        candidate = candidate if isinstance(candidate, dict) else {}
        if candidate.get("selected") is True:
            candidate_selected_count += 1
        fallback_reason = candidate.get("fallback_reason")
        if fallback_reason:
            fallback_reason_counts[str(fallback_reason)] += 1

        trajectory_cost = gcs.get("cost_summary")
        trajectory_cost = trajectory_cost if isinstance(trajectory_cost, dict) else {}
        candidate_cost = candidate.get("cost_summary")
        candidate_cost = candidate_cost if isinstance(candidate_cost, dict) else {}
        terrain_source = _first_present(
            trajectory_cost.get("terrain_objective_source"),
            candidate_cost.get("terrain_objective_source"),
        )
        if terrain_source:
            terrain_objective_source_counts[str(terrain_source)] += 1

        sampled_terrain_cost = _first_float(
            trajectory_cost.get("sampled_terrain_cost"),
            candidate_cost.get("sampled_terrain_cost"),
        )
        if sampled_terrain_cost is not None:
            sampled_terrain_costs.append(sampled_terrain_cost)

        exposure_delta = _first_float(
            candidate_cost.get("high_cost_exposure_delta_vs_baseline"),
            trajectory_cost.get("high_cost_exposure_delta_vs_baseline"),
        )
        if exposure_delta is not None:
            high_cost_exposure_deltas.append(exposure_delta)

    sampled_stats = _numeric_metric_stats(sampled_terrain_costs)
    exposure_stats = _numeric_metric_stats(high_cost_exposure_deltas)
    return {
        "report_count": report_count,
        "attempted_count": attempted_count,
        "success_count": success_count,
        "backend_counts": dict(sorted(backend_counts.items())),
        "candidate_selected_count": candidate_selected_count,
        "candidate_fallback_reason_counts": dict(sorted(fallback_reason_counts.items())),
        "terrain_objective_source_counts": dict(sorted(terrain_objective_source_counts.items())),
        "sampled_terrain_cost_count": sampled_stats["count"],
        "sampled_terrain_cost_min": sampled_stats["min"],
        "sampled_terrain_cost_max": sampled_stats["max"],
        "sampled_terrain_cost_mean": sampled_stats["mean"],
        "high_cost_exposure_delta_count": exposure_stats["count"],
        "high_cost_exposure_delta_min": exposure_stats["min"],
        "high_cost_exposure_delta_max": exposure_stats["max"],
        "high_cost_exposure_delta_mean": exposure_stats["mean"],
    }


def _gcs_motion_feasibility_diagnostics(evaluations) -> dict[str, Any]:
    status_counts: Counter[str] = Counter()
    fallback_reason_counts: Counter[str] = Counter()
    motion_model_counts: Counter[str] = Counter()
    report_count = 0
    evaluated_count = 0
    feasible_count = 0
    infeasible_count = 0
    diagnostic_only_count = 0
    curvature_violation_count = 0
    heading_violation_count = 0
    for item in evaluations:
        motion = item.to_dict().get("gcs_motion_feasibility")
        if not isinstance(motion, dict):
            continue
        report_count += 1
        if motion.get("evaluated") is True:
            evaluated_count += 1
        status = motion.get("feasibility_status")
        if status:
            status_counts[str(status)] += 1
        if status == "feasible":
            feasible_count += 1
        elif status == "infeasible":
            infeasible_count += 1
        elif status == "diagnostic_only":
            diagnostic_only_count += 1
        fallback_reason = motion.get("fallback_reason")
        if fallback_reason:
            fallback_reason_counts[str(fallback_reason)] += 1
        motion_model = motion.get("motion_model")
        if motion_model:
            motion_model_counts[str(motion_model)] += 1
        curvature_violation_count += _int_value(motion.get("curvature_violation_count"))
        heading_violation_count += _int_value(motion.get("heading_violation_count"))
    return {
        "report_count": report_count,
        "evaluated_count": evaluated_count,
        "feasible_count": feasible_count,
        "infeasible_count": infeasible_count,
        "diagnostic_only_count": diagnostic_only_count,
        "curvature_violation_count": curvature_violation_count,
        "heading_violation_count": heading_violation_count,
        "status_counts": dict(sorted(status_counts.items())),
        "fallback_reason_counts": dict(sorted(fallback_reason_counts.items())),
        "motion_model_counts": dict(sorted(motion_model_counts.items())),
    }


def _gcs_curvature_constrained_diagnostics(evaluations) -> dict[str, Any]:
    status_before_counts: Counter[str] = Counter()
    status_after_counts: Counter[str] = Counter()
    fallback_reason_counts: Counter[str] = Counter()
    repair_strategy_counts: Counter[str] = Counter()
    report_count = 0
    attempted_count = 0
    available_count = 0
    selected_count = 0
    repair_success_count = 0
    infeasible_count = 0
    diagnostic_only_count = 0
    curvature_violation_count_before = 0
    curvature_violation_count_after = 0
    heading_violation_count_before = 0
    heading_violation_count_after = 0
    collision_count = 0
    region_containment_violation_count = 0
    for item in evaluations:
        candidate = item.to_dict().get("gcs_curvature_constrained_candidate")
        if not isinstance(candidate, dict):
            continue
        report_count += 1
        if candidate.get("attempted") is True:
            attempted_count += 1
        if candidate.get("available") is True:
            available_count += 1
        if candidate.get("selected") is True:
            selected_count += 1
        if candidate.get("repair_success") is True:
            repair_success_count += 1
        status_before = candidate.get("status_before")
        if status_before:
            status_before_counts[str(status_before)] += 1
        status_after = candidate.get("status_after")
        if status_after:
            status_after_counts[str(status_after)] += 1
        if status_after == "infeasible":
            infeasible_count += 1
        elif status_after == "diagnostic_only":
            diagnostic_only_count += 1
        repair_strategy = candidate.get("repair_strategy")
        if repair_strategy:
            repair_strategy_counts[str(repair_strategy)] += 1
        fallback_reason = candidate.get("fallback_reason")
        if fallback_reason:
            fallback_reason_counts[str(fallback_reason)] += 1
        curvature_violation_count_before += _int_value(candidate.get("curvature_violation_count_before"))
        curvature_violation_count_after += _int_value(candidate.get("curvature_violation_count_after"))
        heading_violation_count_before += _int_value(candidate.get("heading_violation_count_before"))
        heading_violation_count_after += _int_value(candidate.get("heading_violation_count_after"))
        collision_count += _int_value(candidate.get("collision_count"))
        region_containment_violation_count += _int_value(
            candidate.get("region_containment_violation_count")
        )
    return {
        "report_count": report_count,
        "attempted_count": attempted_count,
        "available_count": available_count,
        "selected_count": selected_count,
        "repair_success_count": repair_success_count,
        "infeasible_count": infeasible_count,
        "diagnostic_only_count": diagnostic_only_count,
        "curvature_violation_count_before": curvature_violation_count_before,
        "curvature_violation_count_after": curvature_violation_count_after,
        "heading_violation_count_before": heading_violation_count_before,
        "heading_violation_count_after": heading_violation_count_after,
        "collision_count": collision_count,
        "region_containment_violation_count": region_containment_violation_count,
        "status_before_counts": dict(sorted(status_before_counts.items())),
        "status_after_counts": dict(sorted(status_after_counts.items())),
        "fallback_reason_counts": dict(sorted(fallback_reason_counts.items())),
        "repair_strategy_counts": dict(sorted(repair_strategy_counts.items())),
    }


def _channel_aware_astar_diagnostics(evaluations, *, scenario_id: str) -> dict[str, Any]:
    requested_backend_counts: Counter[str] = Counter()
    selected_backend_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    fallback_reason_counts: Counter[str] = Counter()
    blocker_class_counts: Counter[str] = Counter()
    platform_goal_class_counts: Counter[str] = Counter()
    path_cost_deltas: list[float] = []
    channel_cost_deltas: list[float] = []
    high_cost_exposure_deltas: list[float] = []
    candidate_audit: list[dict[str, Any]] = []
    report_count = 0
    selected_count = 0
    fallback_count = 0
    path_changed_count = 0
    platform_goal_contract_mismatch_count = 0
    platform_goal_anchor_available_count = 0
    platform_goal_unresolved_count = 0

    for item in evaluations:
        candidate = item.to_dict()
        planning_backend = candidate.get("planning_backend")
        if not isinstance(planning_backend, dict):
            continue
        requested_backend = planning_backend.get("requested_backend")
        selected_backend = planning_backend.get("selected_backend")
        if requested_backend != "channel_aware_astar" and selected_backend != "channel_aware_astar":
            continue

        report_count += 1
        requested_key = str(requested_backend or "unknown")
        selected_key = str(selected_backend or "unknown")
        status = str(planning_backend.get("status") or "unknown")
        fallback_reason = planning_backend.get("fallback_reason")
        fallback_reason_text = str(fallback_reason) if fallback_reason else None
        requested_backend_counts[requested_key] += 1
        selected_backend_counts[selected_key] += 1
        status_counts[status] += 1

        selected = status == "selected" or selected_backend == "channel_aware_astar"
        fallback = status == "fallback" or bool(fallback_reason_text) or not selected
        if selected:
            selected_count += 1
        if fallback:
            fallback_count += 1
        if fallback_reason_text:
            fallback_reason_counts[fallback_reason_text] += 1

        platform_goal_feasibility = candidate.get("platform_goal_feasibility")
        platform_goal_feasibility = (
            platform_goal_feasibility
            if isinstance(platform_goal_feasibility, dict)
            else {}
        )
        platform_goal_class = str(platform_goal_feasibility.get("classification") or "unavailable")
        platform_goal_class_counts[platform_goal_class] += 1
        if _platform_goal_contract_mismatch(platform_goal_feasibility):
            platform_goal_contract_mismatch_count += 1
        if platform_goal_feasibility.get("nearest_inflated_passable_anchor") is not None:
            platform_goal_anchor_available_count += 1
        if platform_goal_class == "unknown_contract_mismatch":
            platform_goal_unresolved_count += 1

        blocker_class = _channel_aware_astar_blocker_class(
            status=status,
            selected_backend=selected_key,
            fallback_reason=fallback_reason_text,
            platform_goal_feasibility=platform_goal_feasibility,
        )
        failure_taxonomy = _channel_aware_astar_failure_taxonomy(
            blocker_class=blocker_class,
            fallback_reason=fallback_reason_text,
        )
        blocker_class_counts[blocker_class] += 1

        comparison = planning_backend.get("comparison")
        comparison = comparison if isinstance(comparison, dict) else {}
        if comparison.get("path_changed") is True:
            path_changed_count += 1
        path_cost_delta = _float_value(comparison.get("path_cost_delta"))
        channel_cost_delta = _float_value(comparison.get("channel_cost_delta"))
        high_cost_delta = _float_value(comparison.get("high_cost_exposure_delta"))
        if path_cost_delta is not None:
            path_cost_deltas.append(path_cost_delta)
        if channel_cost_delta is not None:
            channel_cost_deltas.append(channel_cost_delta)
        if high_cost_delta is not None:
            high_cost_exposure_deltas.append(high_cost_delta)

        candidate_audit.append(
            {
                "scenario_id": scenario_id,
                "action_index": candidate.get("action_index"),
                "cell": candidate.get("cell"),
                "requested_backend": requested_key,
                "selected_backend": selected_key,
                "status": status,
                "fallback_reason": fallback_reason_text,
                "blocker_class": blocker_class,
                "failure_taxonomy": failure_taxonomy,
                "platform_goal_feasibility": platform_goal_feasibility,
                "comparison": {
                    "path_changed": comparison.get("path_changed"),
                    "path_cost_delta": path_cost_delta,
                    "channel_cost_delta": channel_cost_delta,
                    "high_cost_exposure_delta": high_cost_delta,
                },
            }
        )

    return {
        "report_count": report_count,
        "selected_count": selected_count,
        "fallback_count": fallback_count,
        "requested_backend_counts": dict(sorted(requested_backend_counts.items())),
        "selected_backend_counts": dict(sorted(selected_backend_counts.items())),
        "status_counts": dict(sorted(status_counts.items())),
        "fallback_reason_counts": dict(sorted(fallback_reason_counts.items())),
        "blocker_class_counts": dict(sorted(blocker_class_counts.items())),
        "platform_goal_feasibility_class_counts": dict(sorted(platform_goal_class_counts.items())),
        "platform_goal_contract_mismatch_count": platform_goal_contract_mismatch_count,
        "platform_goal_anchor_available_count": platform_goal_anchor_available_count,
        "platform_goal_unresolved_count": platform_goal_unresolved_count,
        "path_changed_count": path_changed_count,
        "path_changed_rate": (path_changed_count / report_count if report_count else 0.0),
        "path_cost_delta": _numeric_metric_stats(path_cost_deltas),
        "channel_cost_delta": _numeric_metric_stats(channel_cost_deltas),
        "high_cost_exposure_delta": _numeric_metric_stats(high_cost_exposure_deltas),
        "candidate_audit": candidate_audit,
    }


def _channel_aware_astar_blocker_class(
    *,
    status: str,
    selected_backend: str,
    fallback_reason: str | None,
    platform_goal_feasibility: dict[str, Any] | None = None,
) -> str:
    if status == "selected" or selected_backend == "channel_aware_astar":
        return "selected"
    reason = fallback_reason or ""
    if "goal_blocked" in reason:
        platform_class = _platform_goal_failure_class(platform_goal_feasibility)
        if platform_class is not None:
            return platform_class
        return "goal_blocked"
    if "same_as_baseline" in reason:
        return "same_as_baseline"
    if "not_lower_risk" in reason:
        return "not_lower_risk"
    if reason.startswith("channel_search_failed"):
        return "search_failed"
    if reason:
        return reason
    return "fallback_unspecified"


def _channel_aware_astar_failure_taxonomy(
    *,
    blocker_class: str,
    fallback_reason: str | None,
) -> str:
    if blocker_class in PLATFORM_GOAL_CONTRACT_MISMATCH_CLASSES:
        return blocker_class
    if blocker_class in {"selected", "same_as_baseline", "not_lower_risk", "goal_blocked"}:
        return blocker_class
    reason = fallback_reason or ""
    if "goal_blocked" in reason:
        return "goal_blocked"
    if reason.startswith("channel_search_failed"):
        return "search_failed"
    return blocker_class


def _platform_goal_failure_class(feasibility: dict[str, Any] | None) -> str | None:
    if not isinstance(feasibility, dict):
        return None
    classification = str(feasibility.get("classification") or "")
    if classification in PLATFORM_GOAL_CONTRACT_MISMATCH_CLASSES:
        return classification
    return None


def _platform_goal_contract_mismatch(feasibility: dict[str, Any]) -> bool:
    return bool(feasibility.get("contract_reachable")) and (
        str(feasibility.get("classification") or "") in PLATFORM_GOAL_CONTRACT_MISMATCH_CLASSES
    )


def _aggregate_channel_aware_astar_diagnostics(items: list[dict[str, Any]]) -> dict[str, Any]:
    requested_backend_counts: Counter[str] = Counter()
    selected_backend_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    fallback_reason_counts: Counter[str] = Counter()
    blocker_class_counts: Counter[str] = Counter()
    platform_goal_class_counts: Counter[str] = Counter()
    path_cost_stats: list[dict[str, Any]] = []
    channel_cost_stats: list[dict[str, Any]] = []
    high_cost_stats: list[dict[str, Any]] = []
    candidate_audit: list[dict[str, Any]] = []
    report_count = 0
    selected_count = 0
    fallback_count = 0
    path_changed_count = 0
    platform_goal_contract_mismatch_count = 0
    platform_goal_anchor_available_count = 0
    platform_goal_unresolved_count = 0

    for item in items:
        if not isinstance(item, dict):
            continue
        report_count += _int_value(item.get("report_count"))
        selected_count += _int_value(item.get("selected_count"))
        fallback_count += _int_value(item.get("fallback_count"))
        path_changed_count += _int_value(item.get("path_changed_count"))
        requested_backend_counts.update(_counter_dict(item.get("requested_backend_counts")))
        selected_backend_counts.update(_counter_dict(item.get("selected_backend_counts")))
        status_counts.update(_counter_dict(item.get("status_counts")))
        fallback_reason_counts.update(_counter_dict(item.get("fallback_reason_counts")))
        blocker_class_counts.update(_counter_dict(item.get("blocker_class_counts")))
        platform_goal_class_counts.update(
            _counter_dict(item.get("platform_goal_feasibility_class_counts"))
        )
        platform_goal_contract_mismatch_count += _int_value(
            item.get("platform_goal_contract_mismatch_count")
        )
        platform_goal_anchor_available_count += _int_value(
            item.get("platform_goal_anchor_available_count")
        )
        platform_goal_unresolved_count += _int_value(item.get("platform_goal_unresolved_count"))
        if isinstance(item.get("path_cost_delta"), dict):
            path_cost_stats.append(item["path_cost_delta"])
        if isinstance(item.get("channel_cost_delta"), dict):
            channel_cost_stats.append(item["channel_cost_delta"])
        if isinstance(item.get("high_cost_exposure_delta"), dict):
            high_cost_stats.append(item["high_cost_exposure_delta"])
        audit = item.get("candidate_audit")
        if isinstance(audit, list):
            candidate_audit.extend(entry for entry in audit if isinstance(entry, dict))

    return {
        "report_count": report_count,
        "selected_count": selected_count,
        "fallback_count": fallback_count,
        "requested_backend_counts": dict(sorted(requested_backend_counts.items())),
        "selected_backend_counts": dict(sorted(selected_backend_counts.items())),
        "status_counts": dict(sorted(status_counts.items())),
        "fallback_reason_counts": dict(sorted(fallback_reason_counts.items())),
        "blocker_class_counts": dict(sorted(blocker_class_counts.items())),
        "platform_goal_feasibility_class_counts": dict(sorted(platform_goal_class_counts.items())),
        "platform_goal_contract_mismatch_count": platform_goal_contract_mismatch_count,
        "platform_goal_anchor_available_count": platform_goal_anchor_available_count,
        "platform_goal_unresolved_count": platform_goal_unresolved_count,
        "path_changed_count": path_changed_count,
        "path_changed_rate": (path_changed_count / report_count if report_count else 0.0),
        "path_cost_delta": _aggregate_metric_stats(path_cost_stats),
        "channel_cost_delta": _aggregate_metric_stats(channel_cost_stats),
        "high_cost_exposure_delta": _aggregate_metric_stats(high_cost_stats),
        "candidate_audit": candidate_audit,
    }


def _counter_dict(value: Any) -> Counter[str]:
    counts: Counter[str] = Counter()
    if not isinstance(value, dict):
        return counts
    for key, count in value.items():
        counts[str(key)] += _int_value(count)
    return counts


def _channel_aware_astar_prefixed_fields(summary: dict[str, Any]) -> dict[str, Any]:
    path_cost = summary.get("path_cost_delta") if isinstance(summary.get("path_cost_delta"), dict) else {}
    channel_cost = summary.get("channel_cost_delta") if isinstance(summary.get("channel_cost_delta"), dict) else {}
    high_cost = (
        summary.get("high_cost_exposure_delta")
        if isinstance(summary.get("high_cost_exposure_delta"), dict)
        else {}
    )
    return {
        "channel_aware_astar_report_count": _int_value(summary.get("report_count")),
        "channel_aware_astar_selected_count": _int_value(summary.get("selected_count")),
        "channel_aware_astar_fallback_count": _int_value(summary.get("fallback_count")),
        "channel_aware_astar_requested_backend_counts": dict(summary.get("requested_backend_counts", {})),
        "channel_aware_astar_selected_backend_counts": dict(summary.get("selected_backend_counts", {})),
        "channel_aware_astar_status_counts": dict(summary.get("status_counts", {})),
        "channel_aware_astar_fallback_reason_counts": dict(summary.get("fallback_reason_counts", {})),
        "channel_aware_astar_blocker_class_counts": dict(summary.get("blocker_class_counts", {})),
        "channel_aware_astar_platform_goal_feasibility_class_counts": dict(
            summary.get("platform_goal_feasibility_class_counts", {})
        ),
        "channel_aware_astar_platform_goal_contract_mismatch_count": _int_value(
            summary.get("platform_goal_contract_mismatch_count")
        ),
        "channel_aware_astar_platform_goal_anchor_available_count": _int_value(
            summary.get("platform_goal_anchor_available_count")
        ),
        "channel_aware_astar_platform_goal_unresolved_count": _int_value(
            summary.get("platform_goal_unresolved_count")
        ),
        "channel_aware_astar_path_changed_count": _int_value(summary.get("path_changed_count")),
        "channel_aware_astar_path_changed_rate": float(summary.get("path_changed_rate") or 0.0),
        "channel_aware_astar_path_cost_delta_count": _int_value(path_cost.get("count")),
        "channel_aware_astar_path_cost_delta_min": path_cost.get("min"),
        "channel_aware_astar_path_cost_delta_max": path_cost.get("max"),
        "channel_aware_astar_path_cost_delta_mean": path_cost.get("mean"),
        "channel_aware_astar_channel_cost_delta_count": _int_value(channel_cost.get("count")),
        "channel_aware_astar_channel_cost_delta_min": channel_cost.get("min"),
        "channel_aware_astar_channel_cost_delta_max": channel_cost.get("max"),
        "channel_aware_astar_channel_cost_delta_mean": channel_cost.get("mean"),
        "channel_aware_astar_high_cost_exposure_delta_count": _int_value(high_cost.get("count")),
        "channel_aware_astar_high_cost_exposure_delta_min": high_cost.get("min"),
        "channel_aware_astar_high_cost_exposure_delta_max": high_cost.get("max"),
        "channel_aware_astar_high_cost_exposure_delta_mean": high_cost.get("mean"),
        "channel_aware_astar_candidate_audit": list(summary.get("candidate_audit", [])),
    }


def _sampled_region_path_diagnostics(evaluations) -> dict[str, Any]:
    status_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    fallback_reasons: Counter[str] = Counter()
    start_classification_counts: Counter[str] = Counter()
    goal_classification_counts: Counter[str] = Counter()
    connector_strategy_counts: Counter[str] = Counter()
    bridge_aware_connector_status_counts: Counter[str] = Counter()
    bridge_aware_fallback_reasons: Counter[str] = Counter()
    bridge_corridor_status_counts: Counter[str] = Counter()
    bridge_corridor_fallback_reasons: Counter[str] = Counter()
    bridge_corridor_radius_counts: Counter[str] = Counter()
    reachable_component_status_counts: Counter[str] = Counter()
    reachable_component_reason_counts: Counter[str] = Counter()
    anchor_closure_status_counts: Counter[str] = Counter()
    anchor_closure_reason_counts: Counter[str] = Counter()
    anchor_closure_connection_kind_counts: Counter[str] = Counter()
    terminal_adjustment_status_counts: Counter[str] = Counter()
    terminal_adjustment_reason_counts: Counter[str] = Counter()
    execution_tie_break_status_counts: Counter[str] = Counter()
    execution_tie_break_reason_counts: Counter[str] = Counter()
    complexity_reason_counts: Counter[str] = Counter()
    selected_count = 0
    fallback_count = 0
    sample_attempt_count = 0
    candidate_ranking_count = 0
    anchor_region_added_count = 0
    anchor_region_connected_count = 0
    anchor_closure_attempt_count = 0
    anchor_closure_connected_count = 0
    connector_attempt_count = 0
    bridge_aware_connector_attempt_count = 0
    bridge_aware_connector_available_count = 0
    bridge_aware_connector_selected_count = 0
    bridge_aware_connector_rejected_count = 0
    bridge_aware_bridge_cell_count = 0
    bridge_aware_mask_added_cell_count = 0
    bridge_corridor_connector_attempt_count = 0
    bridge_corridor_connector_available_count = 0
    bridge_corridor_connector_selected_count = 0
    bridge_corridor_connector_rejected_count = 0
    bridge_corridor_added_cell_count = 0
    terminal_adjusted_count = 0
    terminal_adjustment_candidate_count = 0
    reachable_component_disconnected_count = 0
    reachable_component_replacement_selected_count = 0
    reachable_component_terminal_candidate_count = 0
    reachable_terminal_rescue_count = 0
    proxy_goal_anchor_selected_count = 0
    goal_rescue_candidate_count = 0
    benefit_surface_present_count = 0
    path_duplicate_with_baseline_count = 0
    baseline_equivalent_count = 0
    no_quality_gain_count = 0
    fixture_no_benefit_surface_count = 0
    candidate_missing_metrics_count = 0
    constrained_connector_failed_count = 0
    for item in evaluations:
        candidate = item.to_dict()
        planning_backend = candidate.get("planning_backend")
        if not isinstance(planning_backend, dict):
            continue
        sampled = planning_backend.get("sampled_region_path")
        if not isinstance(sampled, dict) or not sampled:
            continue
        status = str(sampled.get("status") or planning_backend.get("status") or "unknown")
        status_counts[status] += 1
        if status == "selected" or planning_backend.get("selected_backend") == "sampled_region_path":
            selected_count += 1
        if status == "fallback" or sampled.get("fallback_reason"):
            fallback_count += 1
        reason = sampled.get("fallback_reason")
        if reason:
            fallback_reasons[str(reason)] += 1
        constrained_connector_failed_seen = reason == "constrained_connector_failed"
        comparison = sampled.get("candidate_comparison")
        comparison = comparison if isinstance(comparison, dict) else {}
        complexity_reason = comparison.get("complexity_reason")
        if not complexity_reason and reason in {
            "candidate_missing_metrics",
            "constrained_connector_failed",
            "fixture_no_benefit_surface",
            "sampled_candidate_baseline_equivalent",
            "sampled_candidate_no_quality_gain",
            "sampled_candidate_path_duplicate",
        }:
            complexity_reason = reason
        if complexity_reason:
            complexity_reason_value = str(complexity_reason)
            complexity_reason_counts[complexity_reason_value] += 1
            if complexity_reason_value == "sampled_candidate_baseline_equivalent":
                baseline_equivalent_count += 1
            if complexity_reason_value == "sampled_candidate_path_duplicate":
                path_duplicate_with_baseline_count += 1
            if complexity_reason_value == "sampled_candidate_no_quality_gain":
                no_quality_gain_count += 1
            if complexity_reason_value == "fixture_no_benefit_surface":
                fixture_no_benefit_surface_count += 1
            if complexity_reason_value == "candidate_missing_metrics":
                candidate_missing_metrics_count += 1
            if complexity_reason_value == "constrained_connector_failed":
                constrained_connector_failed_seen = True
        if comparison.get("benefit_surface_present") is True:
            benefit_surface_present_count += 1
        if (
            comparison.get("path_duplicate_with_baseline") is True
            and complexity_reason != "sampled_candidate_path_duplicate"
        ):
            path_duplicate_with_baseline_count += 1
        sample_attempt_count += _int_value(sampled.get("sample_attempt_count"))
        anchoring = sampled.get("start_goal_anchoring")
        anchoring = anchoring if isinstance(anchoring, dict) else {}
        start_classification = anchoring.get("start_classification")
        goal_classification = anchoring.get("goal_classification")
        if start_classification:
            start_classification_counts[str(start_classification)] += 1
        if goal_classification:
            goal_classification_counts[str(goal_classification)] += 1
        for endpoint in ("start", "goal"):
            if anchoring.get(f"{endpoint}_anchor_region_added") is True:
                anchor_region_added_count += 1
            if anchoring.get(f"{endpoint}_anchor_region_connected") is True:
                anchor_region_connected_count += 1
        closure = anchoring.get("anchor_connectivity_closure")
        closure = closure if isinstance(closure, dict) else {}
        anchor_closure_attempt_count += _int_value(closure.get("attempt_count"))
        anchor_closure_connected_count += _int_value(closure.get("connected_count"))
        closure_status_counts = closure.get("status_counts")
        if isinstance(closure_status_counts, dict):
            anchor_closure_status_counts.update(
                {str(key): _int_value(value) for key, value in closure_status_counts.items()}
            )
        closure_reason_counts = closure.get("reason_counts")
        if isinstance(closure_reason_counts, dict):
            anchor_closure_reason_counts.update(
                {str(key): _int_value(value) for key, value in closure_reason_counts.items()}
            )
        closure_kind_counts = closure.get("connection_kind_counts")
        if isinstance(closure_kind_counts, dict):
            anchor_closure_connection_kind_counts.update(
                {str(key): _int_value(value) for key, value in closure_kind_counts.items()}
            )
        sample_attempts = sampled.get("sample_attempts")
        bridge_aware_report_fallback_reasons: set[str] = set()
        bridge_corridor_report_fallback_reasons: set[str] = set()
        if isinstance(sample_attempts, list):
            for attempt in sample_attempts:
                if not isinstance(attempt, dict):
                    continue
                if attempt.get("kind") != "connector_attempt":
                    continue
                connector_attempt_count += 1
                strategy = attempt.get("strategy")
                if strategy:
                    connector_strategy_counts[str(strategy)] += 1
                if strategy == "bridge_aware_constrained_astar":
                    bridge_aware_connector_attempt_count += 1
                    status_value = str(attempt.get("status") or "unknown")
                    bridge_aware_connector_status_counts[status_value] += 1
                    if status_value == "available":
                        bridge_aware_connector_available_count += 1
                    bridge_aware_bridge_cell_count += _int_value(attempt.get("bridge_cell_count"))
                    bridge_aware_mask_added_cell_count += _int_value(attempt.get("bridge_mask_added_cell_count"))
                    bridge_reason = attempt.get("fallback_reason")
                    if bridge_reason:
                        bridge_aware_report_fallback_reasons.add(str(bridge_reason))
                if strategy == "bridge_corridor_constrained_astar":
                    bridge_corridor_connector_attempt_count += 1
                    status_value = str(attempt.get("status") or "unknown")
                    bridge_corridor_status_counts[status_value] += 1
                    if status_value == "available":
                        bridge_corridor_connector_available_count += 1
                    bridge_corridor_added_cell_count += _int_value(
                        attempt.get("bridge_corridor_added_cell_count")
                    )
                    radius = attempt.get("bridge_corridor_radius_cells")
                    if radius is not None:
                        bridge_corridor_radius_counts[str(radius)] += 1
                    corridor_reason = attempt.get("fallback_reason") or attempt.get("bridge_corridor_failure_reason")
                    if corridor_reason:
                        bridge_corridor_report_fallback_reasons.add(str(corridor_reason))
        rankings = sampled.get("candidate_rankings")
        if isinstance(rankings, list):
            candidate_ranking_count += len(rankings)
            for ranking in rankings:
                if not isinstance(ranking, dict):
                    continue
                strategy = ranking.get("strategy")
                ranking_status = str(ranking.get("status") or "unknown")
                if strategy == "bridge_corridor_constrained_astar":
                    if ranking_status == "selected":
                        bridge_corridor_connector_selected_count += 1
                    if ranking_status == "rejected":
                        bridge_corridor_connector_rejected_count += 1
                    corridor_reason = ranking.get("fallback_reason") or ranking.get(
                        "bridge_corridor_failure_reason"
                    )
                    if corridor_reason:
                        bridge_corridor_report_fallback_reasons.add(str(corridor_reason))
                    if corridor_reason == "constrained_connector_failed":
                        constrained_connector_failed_seen = True
                    continue
                if strategy != "bridge_aware_constrained_astar":
                    if ranking.get("fallback_reason") == "constrained_connector_failed":
                        constrained_connector_failed_seen = True
                    continue
                if ranking_status == "selected":
                    bridge_aware_connector_selected_count += 1
                if ranking_status == "rejected":
                    bridge_aware_connector_rejected_count += 1
                bridge_reason = ranking.get("fallback_reason")
                if bridge_reason:
                    bridge_aware_report_fallback_reasons.add(str(bridge_reason))
                if bridge_reason == "constrained_connector_failed":
                    constrained_connector_failed_seen = True
        if constrained_connector_failed_seen:
            constrained_connector_failed_count += 1
        bridge_aware_fallback_reasons.update(bridge_aware_report_fallback_reasons)
        bridge_corridor_fallback_reasons.update(bridge_corridor_report_fallback_reasons)
        terminal = sampled.get("terminal_adjustment_report")
        terminal = terminal if isinstance(terminal, dict) else {}
        terminal_status = terminal.get("status")
        if terminal_status:
            terminal_adjustment_status_counts[str(terminal_status)] += 1
        terminal_reason = terminal.get("reason_code") or terminal.get("reason")
        if terminal_reason:
            terminal_adjustment_reason_counts[str(terminal_reason)] += 1
        if terminal.get("target_adjusted") is True:
            terminal_adjusted_count += 1
        terminal_adjustment_candidate_count += _int_value(terminal.get("candidate_count"))
        reachable_component_terminal_candidate_count += _int_value(terminal.get("reachable_candidate_count"))
        goal_rescue_candidate_count += _int_value(terminal.get("rescue_candidate_count"))
        if (
            terminal.get("reachable_terminal_rescue_used") is True
            or terminal_reason == "reachable_terminal_selected_by_component_projection"
        ):
            reachable_terminal_rescue_count += 1
        if terminal.get("proxy_goal_anchor_selected") is True or terminal_reason == "proxy_goal_anchor_selected":
            proxy_goal_anchor_selected_count += 1
        component = terminal.get("reachable_component_report")
        if not isinstance(component, dict):
            component = anchoring.get("reachable_component_report")
        component = component if isinstance(component, dict) else {}
        component_status = component.get("status")
        if component_status:
            reachable_component_status_counts[str(component_status)] += 1
        component_reason = component.get("reason")
        if component_reason:
            reachable_component_reason_counts[str(component_reason)] += 1
        if component_status == "disconnected" or component_reason == "target_component_disconnected":
            reachable_component_disconnected_count += 1
        if (
            terminal.get("reachable_component_replacement_selected") is True
            or component_reason == "reachable_component_replacement_selected"
        ):
            reachable_component_replacement_selected_count += 1
        tie_break = sampled.get("execution_tie_break")
        tie_break = tie_break if isinstance(tie_break, dict) else {}
        tie_break_status = tie_break.get("status")
        if tie_break_status:
            execution_tie_break_status_counts[str(tie_break_status)] += 1
        tie_break_reason = tie_break.get("reason")
        if tie_break_reason:
            execution_tie_break_reason_counts[str(tie_break_reason)] += 1
        graph = candidate.get("region_graph")
        if isinstance(graph, dict):
            source_counts[str(graph.get("graph_source") or graph.get("region_source") or "unknown")] += 1
        else:
            source_counts["unknown"] += 1
    return {
        "selected_count": selected_count,
        "fallback_count": fallback_count,
        "status_counts": dict(sorted(status_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "fallback_reasons": dict(sorted(fallback_reasons.items())),
        "sample_attempt_count": sample_attempt_count,
        "candidate_ranking_count": candidate_ranking_count,
        "anchor_region_added_count": anchor_region_added_count,
        "anchor_region_connected_count": anchor_region_connected_count,
        "anchor_closure_attempt_count": anchor_closure_attempt_count,
        "anchor_closure_connected_count": anchor_closure_connected_count,
        "anchor_closure_status_counts": dict(sorted(anchor_closure_status_counts.items())),
        "anchor_closure_reason_counts": dict(sorted(anchor_closure_reason_counts.items())),
        "anchor_closure_connection_kind_counts": dict(sorted(anchor_closure_connection_kind_counts.items())),
        "start_classification_counts": dict(sorted(start_classification_counts.items())),
        "goal_classification_counts": dict(sorted(goal_classification_counts.items())),
        "connector_attempt_count": connector_attempt_count,
        "connector_strategy_counts": dict(sorted(connector_strategy_counts.items())),
        "bridge_aware_connector_attempt_count": bridge_aware_connector_attempt_count,
        "bridge_aware_connector_available_count": bridge_aware_connector_available_count,
        "bridge_aware_connector_selected_count": bridge_aware_connector_selected_count,
        "bridge_aware_connector_rejected_count": bridge_aware_connector_rejected_count,
        "bridge_aware_connector_status_counts": dict(sorted(bridge_aware_connector_status_counts.items())),
        "bridge_aware_fallback_reasons": dict(sorted(bridge_aware_fallback_reasons.items())),
        "bridge_aware_bridge_cell_count": bridge_aware_bridge_cell_count,
        "bridge_aware_mask_added_cell_count": bridge_aware_mask_added_cell_count,
        "bridge_corridor_connector_attempt_count": bridge_corridor_connector_attempt_count,
        "bridge_corridor_connector_available_count": bridge_corridor_connector_available_count,
        "bridge_corridor_connector_selected_count": bridge_corridor_connector_selected_count,
        "bridge_corridor_connector_rejected_count": bridge_corridor_connector_rejected_count,
        "bridge_corridor_status_counts": dict(sorted(bridge_corridor_status_counts.items())),
        "bridge_corridor_fallback_reasons": dict(sorted(bridge_corridor_fallback_reasons.items())),
        "bridge_corridor_radius_counts": dict(sorted(bridge_corridor_radius_counts.items())),
        "bridge_corridor_added_cell_count": bridge_corridor_added_cell_count,
        "terminal_adjusted_count": terminal_adjusted_count,
        "terminal_adjustment_candidate_count": terminal_adjustment_candidate_count,
        "terminal_adjustment_status_counts": dict(sorted(terminal_adjustment_status_counts.items())),
        "terminal_adjustment_reason_counts": dict(sorted(terminal_adjustment_reason_counts.items())),
        "reachable_component_status_counts": dict(sorted(reachable_component_status_counts.items())),
        "reachable_component_reason_counts": dict(sorted(reachable_component_reason_counts.items())),
        "reachable_component_disconnected_count": reachable_component_disconnected_count,
        "reachable_component_replacement_selected_count": reachable_component_replacement_selected_count,
        "reachable_component_terminal_candidate_count": reachable_component_terminal_candidate_count,
        "reachable_terminal_rescue_count": reachable_terminal_rescue_count,
        "proxy_goal_anchor_selected_count": proxy_goal_anchor_selected_count,
        "goal_rescue_candidate_count": goal_rescue_candidate_count,
        "benefit_surface_present_count": benefit_surface_present_count,
        "path_duplicate_with_baseline_count": path_duplicate_with_baseline_count,
        "baseline_equivalent_count": baseline_equivalent_count,
        "no_quality_gain_count": no_quality_gain_count,
        "fixture_no_benefit_surface_count": fixture_no_benefit_surface_count,
        "candidate_missing_metrics_count": candidate_missing_metrics_count,
        "constrained_connector_failed_count": constrained_connector_failed_count,
        "complexity_reason_counts": dict(sorted(complexity_reason_counts.items())),
        "execution_tie_break_status_counts": dict(sorted(execution_tie_break_status_counts.items())),
        "execution_tie_break_reason_counts": dict(sorted(execution_tie_break_reason_counts.items())),
    }


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


def _int_value(value: Any) -> int:
    if isinstance(value, bool) or value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _float_value(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in {float("inf"), float("-inf")}:
        return None
    return number


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _first_float(*values: Any) -> float | None:
    for value in values:
        number = _float_value(value)
        if number is not None:
            return number
    return None


def _numeric_metric_stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None}
    return {
        "count": len(values),
        "min": float(min(values)),
        "max": float(max(values)),
        "mean": float(sum(values) / len(values)),
    }


def _aggregate_metric_stats(stats: list[dict[str, Any]]) -> dict[str, float | int | None]:
    count = 0
    weighted_total = 0.0
    minimum: float | None = None
    maximum: float | None = None
    for item in stats:
        item_count = _int_value(item.get("count"))
        if item_count <= 0:
            continue
        item_mean = _float_value(item.get("mean"))
        if item_mean is None:
            continue
        count += item_count
        weighted_total += item_mean * item_count
        item_min = _float_value(item.get("min"))
        item_max = _float_value(item.get("max"))
        if item_min is not None:
            minimum = item_min if minimum is None else min(minimum, item_min)
        if item_max is not None:
            maximum = item_max if maximum is None else max(maximum, item_max)
    if count == 0:
        return {"count": 0, "min": None, "max": None, "mean": None}
    return {
        "count": count,
        "min": minimum,
        "max": maximum,
        "mean": float(weighted_total / count),
    }


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if not isinstance(value, list | tuple):
        raise ValueError("planner_extra_args must be a list")
    return tuple(str(item) for item in value)


def _open_grid_fallback_used(evaluations) -> bool:
    for item in evaluations:
        request_metadata = item.result.metadata.get("request_payload", {}).get("metadata", {})
        if request_metadata.get("cost_source") == "open_grid_fallback":
            return True
        if request_metadata.get("passable_mask_source") == "open_grid_fallback":
            return True
    return False


def _tracking_safety_violation_count(evaluations) -> int:
    total = 0
    for item in evaluations:
        postprocess = item.to_dict().get("postprocess")
        if isinstance(postprocess, dict):
            total += int(postprocess.get("tracking_safety_violation_count") or 0)
    return total


def _trajectory_optimization_fallback_count(evaluations) -> int:
    total = 0
    for item in evaluations:
        optimization = item.to_dict().get("trajectory_optimization")
        if isinstance(optimization, dict):
            fallback = optimization.get("fallback_status")
            if isinstance(fallback, str) and fallback not in {"ok", "not_needed", "none"}:
                total += 1
    return total


def _region_graph_disconnected_count(evaluations) -> int:
    total = 0
    for item in evaluations:
        region_graph = item.to_dict().get("region_graph")
        if isinstance(region_graph, dict) and region_graph.get("start_goal_connected") is False:
            total += 1
    return total


diagnostic_aggregate = _diagnostic_aggregate
diagnostic_interpretation_summary = _diagnostic_interpretation_summary
scenario_diagnostic_interpretation = _scenario_diagnostic_interpretation
iris_diagnostics = _iris_diagnostics
region_graph_diagnostics = _region_graph_diagnostics
convex_region_diagnostics = _convex_region_diagnostics
gcs_trajectory_diagnostics = _gcs_trajectory_diagnostics
gcs_candidate_diagnostics = _gcs_candidate_diagnostics
gcs_control_point_diagnostics = _gcs_control_point_diagnostics
gcs_motion_feasibility_diagnostics = _gcs_motion_feasibility_diagnostics
gcs_curvature_constrained_diagnostics = _gcs_curvature_constrained_diagnostics
channel_aware_astar_diagnostics = _channel_aware_astar_diagnostics
sampled_region_path_diagnostics = _sampled_region_path_diagnostics
sampled_region_path_candidate_audit = _sampled_region_path_candidate_audit
convex_region_candidate_audit = _convex_region_candidate_audit
gcs_trajectory_candidate_audit = _gcs_trajectory_candidate_audit
gcs_candidate_audit = _gcs_candidate_audit
gcs_control_point_candidate_audit = _gcs_control_point_candidate_audit
gcs_motion_feasibility_audit = _gcs_motion_feasibility_audit
gcs_curvature_constrained_audit = _gcs_curvature_constrained_audit
open_grid_fallback_used = _open_grid_fallback_used
tracking_safety_violation_count = _tracking_safety_violation_count
trajectory_optimization_fallback_count = _trajectory_optimization_fallback_count
region_graph_disconnected_count = _region_graph_disconnected_count

__all__ = [name for name in globals() if not name.startswith('__')]
