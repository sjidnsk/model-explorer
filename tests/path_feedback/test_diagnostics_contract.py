from __future__ import annotations

import importlib

import pytest


DIAGNOSTIC_MODULES = {
    "aggregate": "model_explorer.policy.path_feedback_diagnostic_aggregate",
    "interpretation": "model_explorer.policy.path_feedback_diagnostic_interpretation",
    "backend": "model_explorer.policy.path_feedback_backend_diagnostics",
    "candidate_audits": "model_explorer.policy.path_feedback_candidate_audits",
}

SPLIT_PRIVATE_SYMBOLS = {
    "aggregate": (
        "_aggregate_metric_stats",
        "_counter_dict",
        "_diagnostic_aggregate",
        "_empty_group_summary",
        "_first_float",
        "_first_present",
        "_float_value",
        "_int_value",
        "_numeric_metric_stats",
        "_open_grid_fallback_used",
        "_region_graph_disconnected_count",
        "_tracking_safety_violation_count",
        "_trajectory_optimization_fallback_count",
    ),
    "interpretation": (
        "_candidate_by_cell",
        "_diagnostic_interpretation_summary",
        "_empty_group_interpretation",
        "_iris_region_graph_signal",
        "_list_text",
        "_primary_failure_reason",
        "_scenario_diagnostic_interpretation",
        "_scenario_failure_sources",
        "_target_replacement_reason",
    ),
    "backend": (
        "_aggregate_channel_aware_astar_diagnostics",
        "_channel_aware_astar_blocker_class",
        "_channel_aware_astar_diagnostics",
        "_channel_aware_astar_failure_taxonomy",
        "_channel_aware_astar_prefixed_fields",
        "_convex_region_diagnostics",
        "_gcs_candidate_diagnostics",
        "_gcs_control_point_diagnostics",
        "_gcs_curvature_constrained_diagnostics",
        "_gcs_motion_feasibility_diagnostics",
        "_gcs_trajectory_diagnostics",
        "_iris_diagnostics",
        "_platform_goal_contract_mismatch",
        "_platform_goal_failure_class",
        "_region_graph_diagnostics",
        "_sampled_region_path_diagnostics",
    ),
    "candidate_audits": (
        "_convex_region_candidate_audit",
        "_gcs_candidate_audit",
        "_gcs_control_point_candidate_audit",
        "_gcs_curvature_constrained_audit",
        "_gcs_motion_feasibility_audit",
        "_gcs_trajectory_candidate_audit",
        "_sampled_region_path_candidate_audit",
    ),
}

CORE_LEGACY_SYMBOLS = (
    "_diagnostic_aggregate",
    "_diagnostic_interpretation_summary",
    "_iris_diagnostics",
    "_gcs_candidate_audit",
    "diagnostic_aggregate",
    "diagnostic_interpretation_summary",
    "iris_diagnostics",
    "gcs_candidate_audit",
)


def test_path_feedback_diagnostic_split_modules_import() -> None:
    for module_name in DIAGNOSTIC_MODULES.values():
        module = importlib.import_module(module_name)
        assert isinstance(module.__all__, tuple)


def test_path_feedback_diagnostics_facade_keeps_core_legacy_symbols() -> None:
    facade = importlib.import_module("model_explorer.policy.path_feedback_diagnostics")

    for symbol in CORE_LEGACY_SYMBOLS:
        assert hasattr(facade, symbol)
        assert symbol in facade.__all__


@pytest.mark.parametrize(
    ("module_key", "symbol"),
    [
        (module_key, symbol)
        for module_key, symbols in SPLIT_PRIVATE_SYMBOLS.items()
        for symbol in symbols
    ],
)
def test_path_feedback_impl_and_facade_diagnostic_symbols_point_to_split_modules(
    module_key: str,
    symbol: str,
) -> None:
    split_module = importlib.import_module(DIAGNOSTIC_MODULES[module_key])
    facade = importlib.import_module("model_explorer.policy.path_feedback_diagnostics")
    impl = importlib.import_module("model_explorer.policy.path_feedback_impl")

    assert getattr(impl, symbol) is getattr(split_module, symbol)
    assert getattr(facade, symbol) is getattr(split_module, symbol)
    assert symbol in facade.__all__


def test_path_feedback_impl_manifest_helpers_keep_legacy_identity() -> None:
    impl = importlib.import_module("model_explorer.policy.path_feedback_impl")
    manifest = importlib.import_module("model_explorer.policy.path_feedback_manifest")

    assert impl._optional_str is manifest._optional_str
    assert impl._string_tuple is manifest._string_tuple


def test_path_feedback_diagnostics_all_does_not_leak_temporary_names() -> None:
    facade = importlib.import_module("model_explorer.policy.path_feedback_diagnostics")
    leaked_names = {"Any", "Counter", "defaultdict", "annotations"}

    assert leaked_names.isdisjoint(set(facade.__all__))
