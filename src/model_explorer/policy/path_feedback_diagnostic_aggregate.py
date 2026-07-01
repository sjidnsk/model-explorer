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

_PATH_FEEDBACK_GROUP_FIELDS = (
    ("candidate_count", "candidate_count"), ("reachable_count", "reachable_count"),
    ("failure_count", "failure_count"), ("replan_count", "replan_count"),
)
_DIAGNOSTIC_KEYS = {
    "iris": "iris_diagnostics", "graph": "region_graph_diagnostics",
    "convex": "convex_region_diagnostics", "gcs": "gcs_trajectory_diagnostics",
    "gcs_candidate": "gcs_candidate_diagnostics",
    "gcs_control_point": "gcs_control_point_diagnostics",
    "gcs_motion": "gcs_motion_feasibility_diagnostics",
    "gcs_curvature": "gcs_curvature_constrained_diagnostics",
    "sampled": "sampled_region_path_diagnostics",
}
_GROUP_SCALAR_FIELDS = (
    ("iris_report_count", "iris", "report_count"), ("iris_fallback_count", "iris", "fallback_count"),
    ("region_graph_fallback_count", "graph", "fallback_count"),
    ("region_graph_start_goal_disconnected_count", "graph", "start_goal_disconnected_count"),
    ("convex_region_report_count", "convex", "report_count"), ("convex_region_count_total", "convex", "region_count_total"),
    ("convex_region_fallback_used_count", "convex", "fallback_used_count"), ("convex_region_gcs_ready_count", "convex", "gcs_ready_count"),
    ("convex_region_blocked_cell_violation_count", "convex", "blocked_cell_violation_count"),
    ("convex_region_start_contained_count", "convex", "start_contained_count"), ("convex_region_goal_contained_count", "convex", "goal_contained_count"),
    ("convex_region_adjacent_overlap_count", "convex", "adjacent_overlap_count"), ("convex_region_portal_count", "convex", "portal_count"),
    ("gcs_trajectory_report_count", "gcs", "report_count"), ("gcs_trajectory_attempted_count", "gcs", "attempted_count"),
    ("gcs_trajectory_success_count", "gcs", "success_count"), ("gcs_trajectory_collision_count", "gcs", "collision_count"),
    ("gcs_trajectory_region_count_total", "gcs", "region_count_total"), ("gcs_trajectory_sample_count_total", "gcs", "sample_count_total"),
    ("gcs_candidate_report_count", "gcs_candidate", "report_count"), ("gcs_candidate_attempted_count", "gcs_candidate", "attempted_count"),
    ("gcs_candidate_available_count", "gcs_candidate", "available_count"), ("gcs_candidate_selected_count", "gcs_candidate", "selected_count"),
    ("gcs_candidate_collision_count", "gcs_candidate", "collision_count"),
    ("gcs_candidate_cost_delta_vs_baseline_negative_count", "gcs_candidate", "cost_delta_vs_baseline_negative_count"),
    ("gcs_candidate_cost_delta_vs_baseline_positive_count", "gcs_candidate", "cost_delta_vs_baseline_positive_count"),
    ("gcs_candidate_cost_delta_vs_baseline_zero_count", "gcs_candidate", "cost_delta_vs_baseline_zero_count"),
    ("gcs_control_point_report_count", "gcs_control_point", "report_count"), ("gcs_control_point_attempted_count", "gcs_control_point", "attempted_count"),
    ("gcs_control_point_success_count", "gcs_control_point", "success_count"),
    ("gcs_control_point_candidate_selected_count", "gcs_control_point", "candidate_selected_count"),
    ("gcs_control_point_sampled_terrain_cost_count", "gcs_control_point", "sampled_terrain_cost_count"),
    ("gcs_control_point_high_cost_exposure_delta_count", "gcs_control_point", "high_cost_exposure_delta_count"),
    ("gcs_motion_feasibility_report_count", "gcs_motion", "report_count"), ("gcs_motion_feasibility_evaluated_count", "gcs_motion", "evaluated_count"),
    ("gcs_motion_feasibility_feasible_count", "gcs_motion", "feasible_count"), ("gcs_motion_feasibility_infeasible_count", "gcs_motion", "infeasible_count"),
    ("gcs_motion_feasibility_diagnostic_only_count", "gcs_motion", "diagnostic_only_count"),
    ("gcs_motion_feasibility_curvature_violation_count", "gcs_motion", "curvature_violation_count"),
    ("gcs_motion_feasibility_heading_violation_count", "gcs_motion", "heading_violation_count"),
    ("gcs_curvature_constrained_report_count", "gcs_curvature", "report_count"), ("gcs_curvature_constrained_attempted_count", "gcs_curvature", "attempted_count"),
    ("gcs_curvature_constrained_available_count", "gcs_curvature", "available_count"), ("gcs_curvature_constrained_selected_count", "gcs_curvature", "selected_count"),
    ("gcs_curvature_constrained_repair_success_count", "gcs_curvature", "repair_success_count"),
    ("gcs_curvature_constrained_infeasible_count", "gcs_curvature", "infeasible_count"),
    ("gcs_curvature_constrained_diagnostic_only_count", "gcs_curvature", "diagnostic_only_count"),
    ("gcs_curvature_constrained_curvature_violation_count_before", "gcs_curvature", "curvature_violation_count_before"),
    ("gcs_curvature_constrained_curvature_violation_count_after", "gcs_curvature", "curvature_violation_count_after"),
    ("gcs_curvature_constrained_heading_violation_count_before", "gcs_curvature", "heading_violation_count_before"),
    ("gcs_curvature_constrained_heading_violation_count_after", "gcs_curvature", "heading_violation_count_after"),
    ("gcs_curvature_constrained_collision_count", "gcs_curvature", "collision_count"),
    ("gcs_curvature_constrained_region_containment_violation_count", "gcs_curvature", "region_containment_violation_count"),
    ("sampled_region_path_selected_count", "sampled", "selected_count"), ("sampled_region_path_fallback_count", "sampled", "fallback_count"),
    ("sampled_region_path_sample_attempt_count", "sampled", "sample_attempt_count"), ("sampled_region_path_candidate_ranking_count", "sampled", "candidate_ranking_count"),
    ("sampled_region_path_anchor_region_added_count", "sampled", "anchor_region_added_count"),
    ("sampled_region_path_anchor_region_connected_count", "sampled", "anchor_region_connected_count"),
    ("sampled_region_path_anchor_closure_attempt_count", "sampled", "anchor_closure_attempt_count"),
    ("sampled_region_path_anchor_closure_connected_count", "sampled", "anchor_closure_connected_count"),
    ("sampled_region_path_connector_attempt_count", "sampled", "connector_attempt_count"),
    ("sampled_region_path_bridge_aware_connector_attempt_count", "sampled", "bridge_aware_connector_attempt_count"),
    ("sampled_region_path_bridge_aware_connector_available_count", "sampled", "bridge_aware_connector_available_count"),
    ("sampled_region_path_bridge_aware_connector_selected_count", "sampled", "bridge_aware_connector_selected_count"),
    ("sampled_region_path_bridge_aware_connector_rejected_count", "sampled", "bridge_aware_connector_rejected_count"),
    ("sampled_region_path_bridge_aware_bridge_cell_count", "sampled", "bridge_aware_bridge_cell_count"),
    ("sampled_region_path_bridge_aware_mask_added_cell_count", "sampled", "bridge_aware_mask_added_cell_count"),
    ("sampled_region_path_bridge_corridor_connector_attempt_count", "sampled", "bridge_corridor_connector_attempt_count"),
    ("sampled_region_path_bridge_corridor_connector_available_count", "sampled", "bridge_corridor_connector_available_count"),
    ("sampled_region_path_bridge_corridor_connector_selected_count", "sampled", "bridge_corridor_connector_selected_count"),
    ("sampled_region_path_bridge_corridor_connector_rejected_count", "sampled", "bridge_corridor_connector_rejected_count"),
    ("sampled_region_path_bridge_corridor_added_cell_count", "sampled", "bridge_corridor_added_cell_count"),
    ("sampled_region_path_terminal_adjusted_count", "sampled", "terminal_adjusted_count"),
    ("sampled_region_path_terminal_adjustment_candidate_count", "sampled", "terminal_adjustment_candidate_count"),
    ("sampled_region_path_reachable_component_disconnected_count", "sampled", "reachable_component_disconnected_count"),
    ("sampled_region_path_reachable_component_replacement_selected_count", "sampled", "reachable_component_replacement_selected_count"),
    ("sampled_region_path_reachable_component_terminal_candidate_count", "sampled", "reachable_component_terminal_candidate_count"),
    ("sampled_region_path_reachable_terminal_rescue_count", "sampled", "reachable_terminal_rescue_count"),
    ("sampled_region_path_proxy_goal_anchor_selected_count", "sampled", "proxy_goal_anchor_selected_count"),
    ("sampled_region_path_goal_rescue_candidate_count", "sampled", "goal_rescue_candidate_count"),
    ("sampled_region_path_benefit_surface_present_count", "sampled", "benefit_surface_present_count"),
    ("sampled_region_path_path_duplicate_with_baseline_count", "sampled", "path_duplicate_with_baseline_count"),
    ("sampled_region_path_baseline_equivalent_count", "sampled", "baseline_equivalent_count"),
    ("sampled_region_path_no_quality_gain_count", "sampled", "no_quality_gain_count"),
    ("sampled_region_path_fixture_no_benefit_surface_count", "sampled", "fixture_no_benefit_surface_count"),
    ("sampled_region_path_candidate_missing_metrics_count", "sampled", "candidate_missing_metrics_count"),
    ("sampled_region_path_constrained_connector_failed_count", "sampled", "constrained_connector_failed_count"),
)
_CHANNEL_GROUP_SCALAR_FIELDS = (
    ("channel_aware_astar_report_count", "report_count"), ("channel_aware_astar_selected_count", "selected_count"),
    ("channel_aware_astar_fallback_count", "fallback_count"), ("channel_aware_astar_path_changed_count", "path_changed_count"),
)
_TOTAL_SCALAR_FIELDS = (
    ("iris_report_count", "iris", "report_count"), ("iris_fallback_count", "iris", "fallback_count"),
    ("iris_failure_count", "iris", "failure_count"), ("iris_region_count_total", "iris", "region_count_total"),
    *(_GROUP_SCALAR_FIELDS[2:]),
)
_COUNTER_FIELDS = (
    ("iris_status_counts", "iris", "status_counts"), ("iris_fallback_reasons", "iris", "fallback_reasons"),
    ("region_graph_source_counts", "graph", "source_counts"), ("region_graph_fallback_reasons", "graph", "fallback_reasons"),
    ("convex_region_backend_counts", "convex", "backend_counts"), ("convex_region_coverage_status_counts", "convex", "coverage_status_counts"),
    ("convex_region_gcs_ready_reason_counts", "convex", "gcs_ready_reason_counts"),
    ("gcs_trajectory_backend_counts", "gcs", "backend_counts"), ("gcs_trajectory_reason_counts", "gcs", "reason_counts"),
    ("gcs_trajectory_result_status_counts", "gcs", "result_status_counts"),
    ("gcs_candidate_fallback_reason_counts", "gcs_candidate", "fallback_reason_counts"),
    ("gcs_candidate_selection_reason_counts", "gcs_candidate", "selection_reason_counts"),
    ("gcs_control_point_backend_counts", "gcs_control_point", "backend_counts"),
    ("gcs_control_point_candidate_fallback_reason_counts", "gcs_control_point", "candidate_fallback_reason_counts"),
    ("gcs_control_point_terrain_objective_source_counts", "gcs_control_point", "terrain_objective_source_counts"),
    ("gcs_motion_feasibility_status_counts", "gcs_motion", "status_counts"),
    ("gcs_motion_feasibility_fallback_reason_counts", "gcs_motion", "fallback_reason_counts"),
    ("gcs_motion_feasibility_motion_model_counts", "gcs_motion", "motion_model_counts"),
    ("gcs_curvature_constrained_status_before_counts", "gcs_curvature", "status_before_counts"),
    ("gcs_curvature_constrained_status_after_counts", "gcs_curvature", "status_after_counts"),
    ("gcs_curvature_constrained_fallback_reason_counts", "gcs_curvature", "fallback_reason_counts"),
    ("gcs_curvature_constrained_repair_strategy_counts", "gcs_curvature", "repair_strategy_counts"),
    ("sampled_region_path_status_counts", "sampled", "status_counts"), ("sampled_region_path_source_counts", "sampled", "source_counts"),
    ("sampled_region_path_fallback_reasons", "sampled", "fallback_reasons"),
    ("sampled_region_path_start_classification_counts", "sampled", "start_classification_counts"),
    ("sampled_region_path_goal_classification_counts", "sampled", "goal_classification_counts"),
    ("sampled_region_path_anchor_closure_status_counts", "sampled", "anchor_closure_status_counts"),
    ("sampled_region_path_anchor_closure_reason_counts", "sampled", "anchor_closure_reason_counts"),
    ("sampled_region_path_anchor_closure_connection_kind_counts", "sampled", "anchor_closure_connection_kind_counts"),
    ("sampled_region_path_connector_strategy_counts", "sampled", "connector_strategy_counts"),
    ("sampled_region_path_bridge_aware_connector_status_counts", "sampled", "bridge_aware_connector_status_counts"),
    ("sampled_region_path_bridge_aware_fallback_reasons", "sampled", "bridge_aware_fallback_reasons"),
    ("sampled_region_path_bridge_corridor_status_counts", "sampled", "bridge_corridor_status_counts"),
    ("sampled_region_path_bridge_corridor_fallback_reasons", "sampled", "bridge_corridor_fallback_reasons"),
    ("sampled_region_path_bridge_corridor_radius_counts", "sampled", "bridge_corridor_radius_counts"),
    ("sampled_region_path_terminal_adjustment_status_counts", "sampled", "terminal_adjustment_status_counts"),
    ("sampled_region_path_terminal_adjustment_reason_counts", "sampled", "terminal_adjustment_reason_counts"),
    ("sampled_region_path_reachable_component_status_counts", "sampled", "reachable_component_status_counts"),
    ("sampled_region_path_reachable_component_reason_counts", "sampled", "reachable_component_reason_counts"),
    ("sampled_region_path_execution_tie_break_status_counts", "sampled", "execution_tie_break_status_counts"),
    ("sampled_region_path_execution_tie_break_reason_counts", "sampled", "execution_tie_break_reason_counts"),
    ("sampled_region_path_complexity_reason_counts", "sampled", "complexity_reason_counts"),
)
_AUDIT_FIELDS = (
    ("convex_region_candidate_audit", "convex_region_candidate_audit"),
    ("gcs_trajectory_candidate_audit", "gcs_trajectory_candidate_audit"),
    ("gcs_candidate_audit", "gcs_candidate_audit"),
    ("gcs_control_point_candidate_audit", "gcs_control_point_candidate_audit"),
    ("gcs_motion_feasibility_audit", "gcs_motion_feasibility_audit"),
    ("gcs_curvature_constrained_audit", "gcs_curvature_constrained_audit"),
    ("sampled_region_path_candidate_audit", "sampled_region_path_candidate_audit"),
)
_CONTROL_POINT_METRIC_FIELDS = (
    ("gcs_control_point_sampled_terrain_cost", "sampled_terrain_cost"),
    ("gcs_control_point_high_cost_exposure_delta", "high_cost_exposure_delta"),
)
_GROUP_SUMMARY_KEYS = (
    "scenario_count", "candidate_count", "reachable_count", "failure_count", "replan_count", "selection_changed_count",
    *(key for key, _, _ in _GROUP_SCALAR_FIELDS), *(key for key, _ in _CHANNEL_GROUP_SCALAR_FIELDS),
)


def _diagnostic_aggregate(scenarios: list[dict[str, Any]]) -> dict[str, Any]:
    from .path_feedback_backend_diagnostics import (
        _aggregate_channel_aware_astar_diagnostics,
        _channel_aware_astar_prefixed_fields,
    )

    totals: Counter[str] = Counter()
    counters = {key: Counter() for key, _, _ in _COUNTER_FIELDS}
    audits: dict[str, list[dict[str, Any]]] = {key: [] for key, _ in _AUDIT_FIELDS}
    control_point_stats: dict[str, list[dict[str, Any]]] = {key: [] for key, _ in _CONTROL_POINT_METRIC_FIELDS}
    channel_aware_astar_diagnostics: list[dict[str, Any]] = []
    group_summary: dict[str, dict[str, Any]] = defaultdict(_empty_group_summary)

    for scenario in scenarios:
        diagnostics = {name: scenario[key] for name, key in _DIAGNOSTIC_KEYS.items()}
        channel_aware = scenario.get("channel_aware_astar_diagnostics")
        channel_aware = channel_aware if isinstance(channel_aware, dict) else {}
        group_payload = group_summary[str(scenario.get("scenario_group") or "unknown")]
        group_payload["scenario_count"] += 1
        feedback = scenario["path_feedback"]
        for group_key, feedback_key in _PATH_FEEDBACK_GROUP_FIELDS:
            group_payload[group_key] += int(feedback[feedback_key])
        group_payload["selection_changed_count"] += int(bool(scenario["selection_changed_by_path_feedback"]))
        for group_key, source_name, source_key in _GROUP_SCALAR_FIELDS:
            group_payload[group_key] += int(diagnostics[source_name][source_key])
        for group_key, source_key in _CHANNEL_GROUP_SCALAR_FIELDS:
            group_payload[group_key] += _int_value(channel_aware.get(source_key))
        for output_key, source_name, source_key in _TOTAL_SCALAR_FIELDS:
            totals[output_key] += int(diagnostics[source_name][source_key])
        for output_key, source_name, source_key in _COUNTER_FIELDS:
            counters[output_key].update(diagnostics[source_name][source_key])
        for output_key, scenario_key in _AUDIT_FIELDS:
            audits[output_key].extend(scenario.get(scenario_key, []))
        for output_key, source_prefix in _CONTROL_POINT_METRIC_FIELDS:
            source = diagnostics["gcs_control_point"]
            control_point_stats[output_key].append(
                {"count": source[f"{source_prefix}_count"], "min": source[f"{source_prefix}_min"],
                 "max": source[f"{source_prefix}_max"], "mean": source[f"{source_prefix}_mean"]}
            )
        channel_aware_astar_diagnostics.append(channel_aware)

    result: dict[str, Any] = {"iris_requested_count": totals["iris_report_count"]}
    result.update({key: totals[key] for key, _, _ in _TOTAL_SCALAR_FIELDS})
    result.update({key: dict(sorted(value.items())) for key, value in counters.items()})
    result.update({key: value for key, value in audits.items() if not key.startswith("sampled_region_path")})
    for output_key, _ in _CONTROL_POINT_METRIC_FIELDS:
        stats = _aggregate_metric_stats(control_point_stats[output_key])
        result.update({f"{output_key}_count": stats["count"], f"{output_key}_min": stats["min"],
                       f"{output_key}_max": stats["max"], f"{output_key}_mean": stats["mean"]})
    result.update(_channel_aware_astar_prefixed_fields(_aggregate_channel_aware_astar_diagnostics(channel_aware_astar_diagnostics)))
    result.update({key: totals[key] for key, _, _ in _GROUP_SCALAR_FIELDS if key.startswith("sampled_region_path")})
    result.update({key: dict(sorted(counters[key].items())) for key, _, _ in _COUNTER_FIELDS if key.startswith("sampled_region_path")})
    result["sampled_region_path_candidate_audit"] = audits["sampled_region_path_candidate_audit"]
    result["scenario_group_summary"] = {group: dict(payload) for group, payload in sorted(group_summary.items())}
    return result


def _empty_group_summary() -> dict[str, int]:
    return dict.fromkeys(_GROUP_SUMMARY_KEYS, 0)

def _counter_dict(value: Any) -> Counter[str]:
    counts: Counter[str] = Counter()
    if not isinstance(value, dict):
        return counts
    for key, count in value.items():
        counts[str(key)] += _int_value(count)
    return counts

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
__all__ = (
    "GCS_CONTROL_POINT_BACKEND",
    "PLATFORM_GOAL_CONTRACT_MISMATCH_CLASSES",
    "_diagnostic_aggregate",
    "_empty_group_summary",
    "_counter_dict",
    "_numeric_metric_stats",
    "_aggregate_metric_stats",
    "_int_value",
    "_float_value",
    "_first_present",
    "_first_float",
    "_optional_str",
    "_string_tuple",
    "_open_grid_fallback_used",
    "_tracking_safety_violation_count",
    "_trajectory_optimization_fallback_count",
    "_region_graph_disconnected_count",
)
