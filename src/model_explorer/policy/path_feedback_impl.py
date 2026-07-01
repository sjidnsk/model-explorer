"""Compatibility re-export for migrated path-feedback symbols."""

from importlib import import_module as _import_module

_PUBLIC_MODULES = (
    "path_feedback_manifest",
    "path_feedback_summary",
    "path_feedback_reports",
    "path_feedback_diagnostics",
    "path_feedback_artifacts",
    "feedback_selection",
    "path_feedback_runner",
)

_TEMPORARY_EXPORT_NAMES = {
    "Any",
    "Counter",
    "Path",
    "annotations",
    "dataclass",
    "defaultdict",
    "field",
    "isfinite",
    "json",
    "load_scenario",
}

_LEGACY_PRIVATE_EXPORTS = {
    "_RouteFixturePlanner": "path_feedback_runner",
    "_acceptance_metadata": "path_feedback_summary",
    "_aggregate_channel_aware_astar_diagnostics": "path_feedback_diagnostics",
    "_aggregate_metric_stats": "path_feedback_diagnostics",
    "_aggregate_row_counter": "path_feedback_artifacts",
    "_anchor_projection_adjusted_path_cost": "feedback_selection",
    "_anchor_projection_candidate_config": "path_feedback_runner",
    "_anchor_projection_candidate_generation_summary": "feedback_selection",
    "_anchor_projection_source_selection_quality": "feedback_selection",
    "_annotate_policy_context_ids": "path_feedback_runner",
    "_artifact_index_payload": "path_feedback_artifacts",
    "_best_source_selection_alternative": "feedback_selection",
    "_candidate_by_cell": "path_feedback_diagnostics",
    "_candidate_float": "feedback_selection",
    "_candidate_generation_for_selection": "feedback_selection",
    "_candidate_path_cost_for_cell": "feedback_selection",
    "_cell": "path_feedback_manifest",
    "_cell_to_list": "path_feedback_manifest",
    "_cell_tuple": "path_feedback_manifest",
    "_channel_aware_astar_blocker_class": "path_feedback_diagnostics",
    "_channel_aware_astar_diagnostics": "path_feedback_diagnostics",
    "_channel_aware_astar_failure_taxonomy": "path_feedback_diagnostics",
    "_channel_aware_astar_prefixed_fields": "path_feedback_diagnostics",
    "_channel_aware_blocker_reason": "feedback_selection",
    "_channel_aware_evidence_by_action_index": "feedback_selection",
    "_channel_aware_score_adjustment": "feedback_selection",
    "_contract_aware_preferred_evaluation": "feedback_selection",
    "_contract_aware_preferred_selection": "feedback_selection",
    "_contract_safe_trainable_candidate": "feedback_selection",
    "_contract_safe_trainable_evaluation": "feedback_selection",
    "_control_point_blocker_class": "path_feedback_artifacts",
    "_control_point_calibration_sweep": "path_feedback_artifacts",
    "_control_point_triage_interpretation": "path_feedback_artifacts",
    "_convex_region_candidate_audit": "path_feedback_diagnostics",
    "_convex_region_diagnostics": "path_feedback_diagnostics",
    "_counter_dict": "path_feedback_diagnostics",
    "_dedupe": "feedback_selection",
    "_diagnostic_aggregate": "path_feedback_diagnostics",
    "_diagnostic_interpretation_summary": "path_feedback_diagnostics",
    "_empty_group_interpretation": "path_feedback_diagnostics",
    "_empty_group_summary": "path_feedback_diagnostics",
    "_fallback_status_is_problem": "feedback_selection",
    "_finite_float": "feedback_selection",
    "_finite_float_optional": "feedback_selection",
    "_first_float": "path_feedback_diagnostics",
    "_first_present": "path_feedback_diagnostics",
    "_float_value": "path_feedback_diagnostics",
    "_gcs_candidate_audit": "path_feedback_diagnostics",
    "_gcs_candidate_diagnostics": "path_feedback_diagnostics",
    "_gcs_control_point_candidate_artifact_index": "path_feedback_artifacts",
    "_gcs_control_point_candidate_artifacts": "path_feedback_artifacts",
    "_gcs_control_point_candidate_audit": "path_feedback_diagnostics",
    "_gcs_control_point_candidate_triage_summary": "path_feedback_artifacts",
    "_gcs_control_point_diagnostics": "path_feedback_diagnostics",
    "_gcs_control_point_triage_row": "path_feedback_artifacts",
    "_gcs_curvature_constrained_audit": "path_feedback_diagnostics",
    "_gcs_curvature_constrained_diagnostics": "path_feedback_diagnostics",
    "_gcs_motion_feasibility_audit": "path_feedback_diagnostics",
    "_gcs_motion_feasibility_diagnostics": "path_feedback_diagnostics",
    "_gcs_trajectory_candidate_audit": "path_feedback_diagnostics",
    "_gcs_trajectory_diagnostics": "path_feedback_diagnostics",
    "_goals_by_evaluation_action_index": "feedback_selection",
    "_int_value": "path_feedback_diagnostics",
    "_iris_diagnostics": "path_feedback_diagnostics",
    "_iris_region_graph_signal": "path_feedback_diagnostics",
    "_list_cell": "feedback_selection",
    "_list_text": "path_feedback_diagnostics",
    "_min_max_normalize": "feedback_selection",
    "_non_positive_number": "path_feedback_artifacts",
    "_normalization_features": "feedback_selection",
    "_normalized_features": "feedback_selection",
    "_numeric_experimental": "feedback_selection",
    "_numeric_experimental_optional": "feedback_selection",
    "_numeric_metric_stats": "path_feedback_diagnostics",
    "_numeric_observation": "path_feedback_runner",
    "_objective_term_weight": "path_feedback_artifacts",
    "_open_grid_fallback_used": "path_feedback_diagnostics",
    "_optional_path": "path_feedback_manifest",
    "_optional_str": "path_feedback_manifest",
    "_optional_string": "path_feedback_manifest",
    "_path_cost_delta": "feedback_selection",
    "_path_cost_for_selection": "feedback_selection",
    "_path_feedback_penalty": "feedback_selection",
    "_planner_for_scenario": "path_feedback_runner",
    "_planner_validated_distance_exception_candidate": "feedback_selection",
    "_planner_validated_distance_exception_evaluation": "feedback_selection",
    "_platform_goal_contract_mismatch": "path_feedback_diagnostics",
    "_platform_goal_failure_class": "path_feedback_diagnostics",
    "_policy_context_planning_backend": "path_feedback_runner",
    "_preferred_trainable_candidate": "feedback_selection",
    "_preferred_trainable_evaluation": "feedback_selection",
    "_primary_failure_reason": "path_feedback_diagnostics",
    "_ranking_key": "feedback_selection",
    "_region_graph_diagnostics": "path_feedback_diagnostics",
    "_region_graph_disconnected_count": "path_feedback_diagnostics",
    "_required_path": "path_feedback_manifest",
    "_resolve_path": "path_feedback_manifest",
    "_resolve_planner_config": "path_feedback_manifest",
    "_route_fixtures": "path_feedback_manifest",
    "_route_payload_from_metadata": "path_feedback_artifacts",
    "_run_feedback_scenario": "path_feedback_runner",
    "_safe_int": "feedback_selection",
    "_safe_path_component": "path_feedback_artifacts",
    "_sampled_region_path_candidate_audit": "path_feedback_diagnostics",
    "_sampled_region_path_diagnostics": "path_feedback_diagnostics",
    "_scenario_diagnostic_interpretation": "path_feedback_diagnostics",
    "_scenario_failure_sources": "path_feedback_diagnostics",
    "_scenario_from_payload": "path_feedback_manifest",
    "_scenario_seed": "path_feedback_manifest",
    "_score_evaluations": "feedback_selection",
    "_selected_after_feedback": "feedback_selection",
    "_selected_before_feedback": "feedback_selection",
    "_selection_payload": "feedback_selection",
    "_source_selection_key": "feedback_selection",
    "_source_selection_quality_regression": "feedback_selection",
    "_string_tuple": "path_feedback_manifest",
    "_target_replacement_reason": "path_feedback_diagnostics",
    "_tracking_safety_violation_count": "path_feedback_diagnostics",
    "_trajectory_optimization_fallback_count": "path_feedback_diagnostics",
    "_unique_numeric_values": "path_feedback_artifacts",
    "_updated_trainability_gate": "feedback_selection",
    "_write_json": "path_feedback_artifacts",
}

_exports = {}
for _module_name in _PUBLIC_MODULES:
    _module = _import_module(f"{__package__}.{_module_name}")
    for _name in getattr(_module, "__all__", ()):
        if _name.startswith("_") or _name in _TEMPORARY_EXPORT_NAMES:
            continue
        if hasattr(_module, _name):
            _exports.setdefault(_name, getattr(_module, _name))

for _name, _module_name in _LEGACY_PRIVATE_EXPORTS.items():
    _module = _import_module(f"{__package__}.{_module_name}")
    if hasattr(_module, _name):
        _exports[_name] = getattr(_module, _name)

globals().update(_exports)
__all__ = tuple(sorted(_exports))

del _exports
del _import_module
del _LEGACY_PRIVATE_EXPORTS
del _module
del _module_name
del _name
del _PUBLIC_MODULES
del _TEMPORARY_EXPORT_NAMES
