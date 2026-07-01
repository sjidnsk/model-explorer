from __future__ import annotations

import importlib


DIAGNOSTIC_MODULES = (
    "model_explorer.policy.planning_backend_summaries",
    "model_explorer.policy.planning_platform_feasibility",
    "model_explorer.policy.planning_diagnostic_interpretation",
)

DIAGNOSTIC_EXPORTS = (
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

REPRESENTATIVE_DIAGNOSTIC_SYMBOLS = (
    "path_feedback_summary",
    "_platform_goal_feasibility",
    "_candidate_diagnostic_interpretation",
)

FORBIDDEN_TEMP_NAMES = {
    "annotations",
    "Any",
    "Counter",
    "Sequence",
    "deque",
    "ceil",
    "hypot",
}


def test_diagnostic_split_modules_import_and_export_expected_symbols() -> None:
    modules = {name: importlib.import_module(name) for name in DIAGNOSTIC_MODULES}

    assert modules["model_explorer.policy.planning_backend_summaries"].path_feedback_summary
    assert modules["model_explorer.policy.planning_platform_feasibility"]._platform_goal_feasibility
    assert modules["model_explorer.policy.planning_diagnostic_interpretation"]._candidate_diagnostic_interpretation

    for module in modules.values():
        assert FORBIDDEN_TEMP_NAMES.isdisjoint(set(module.__all__))


def test_planning_diagnostics_facade_preserves_core_exports_without_temp_names() -> None:
    facade = importlib.import_module("model_explorer.policy.planning_diagnostics")

    assert set(DIAGNOSTIC_EXPORTS).issubset(set(facade.__all__))
    assert FORBIDDEN_TEMP_NAMES.isdisjoint(set(facade.__all__))
    for name in FORBIDDEN_TEMP_NAMES:
        assert not hasattr(facade, name)


def test_planning_impl_and_facade_preserve_diagnostic_symbol_identity() -> None:
    planning = importlib.import_module("model_explorer.policy.planning")
    planning_impl = importlib.import_module("model_explorer.policy.planning_impl")
    diagnostics_facade = importlib.import_module("model_explorer.policy.planning_diagnostics")

    for name in REPRESENTATIVE_DIAGNOSTIC_SYMBOLS:
        diagnostic_symbol = getattr(diagnostics_facade, name)
        assert getattr(planning_impl, name) is diagnostic_symbol
        assert getattr(planning, name) is diagnostic_symbol
