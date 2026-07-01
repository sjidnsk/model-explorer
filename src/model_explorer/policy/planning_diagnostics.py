"""Compatibility facade for planning diagnostic helpers."""

from .planning_backend_summaries import (
    _convex_region_route_report,
    _convex_region_summary,
    _gcs_candidate_route_report,
    _gcs_candidate_summary,
    _gcs_curvature_constrained_candidate_route_report,
    _gcs_curvature_constrained_candidate_summary,
    _gcs_motion_feasibility_route_report,
    _gcs_motion_feasibility_summary,
    _gcs_trajectory_route_report,
    _gcs_trajectory_summary,
    _iris_region_summary,
    _optimization_summary,
    _planning_backend_summary,
    _postprocess_summary,
    _region_graph_summary,
    path_feedback_summary,
)
from .planning_diagnostic_interpretation import (
    _candidate_diagnostic_interpretation,
    _dedupe,
    _input_source_summary,
    _primary_diagnostic_source,
    _report_present,
)
from .planning_platform_feasibility import (
    _platform_goal_classification,
    _platform_goal_contract_mismatch,
    _platform_goal_feasibility,
    _platform_goal_feasibility_payload,
    _with_projected_anchor_feasibility,
)


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
    "_platform_goal_feasibility",
    "_platform_goal_feasibility_payload",
    "_with_projected_anchor_feasibility",
    "_platform_goal_classification",
    "_platform_goal_contract_mismatch",
    "_input_source_summary",
    "_candidate_diagnostic_interpretation",
    "_primary_diagnostic_source",
    "_dedupe",
    "_report_present",
)
