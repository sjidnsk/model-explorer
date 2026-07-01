from __future__ import annotations

import importlib


ANCHOR_MODULES = (
    "model_explorer.policy.planning_anchor_evaluation",
    "model_explorer.policy.planning_anchor_projection",
    "model_explorer.policy.planning_anchor_grid",
)

ANCHOR_EXPORTS = (
    "evaluate_candidate_paths",
    "_same_action_anchor_projection_candidate_evaluation",
    "_projected_anchor_candidate_evaluation",
    "anchor_projection_candidate_config_from_mapping",
    "_anchor_projection_candidate_generation_payload",
    "_trainability_distance_reject_reasons",
    "_planner_validated_distance_exception_safe",
    "_anchor_projection_analysis",
    "_anchor_projection_reject_reason",
    "_proxy_anchor_route_comparison",
    "_proxy_route_unavailable",
    "_connected_component_labels",
    "_best_reachable_anchor_in_component",
    "_grid_distance_map",
    "_component_id_at",
    "_component_size",
    "_cell_list",
    "_cell_manhattan_or_none",
    "_cell_distance_m_or_none",
    "_inflated_passable_mask",
    "_nearest_inflated_passable_anchor",
    "_grid_path",
    "_mask_neighbors",
    "_path_cost_for_payload",
    "_bool_grid",
    "_mask_value",
    "_grid_value",
)

REPRESENTATIVE_ANCHOR_SYMBOLS = (
    "evaluate_candidate_paths",
    "anchor_projection_candidate_config_from_mapping",
    "_anchor_projection_analysis",
    "_grid_path",
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


def test_anchor_split_modules_import_and_export_expected_symbols() -> None:
    modules = {name: importlib.import_module(name) for name in ANCHOR_MODULES}

    assert modules["model_explorer.policy.planning_anchor_evaluation"].evaluate_candidate_paths
    assert modules["model_explorer.policy.planning_anchor_projection"].anchor_projection_candidate_config_from_mapping
    assert modules["model_explorer.policy.planning_anchor_grid"]._grid_path

    for module in modules.values():
        assert FORBIDDEN_TEMP_NAMES.isdisjoint(set(module.__all__))


def test_planning_anchor_facade_preserves_core_exports_without_temp_names() -> None:
    facade = importlib.import_module("model_explorer.policy.planning_anchor")

    assert set(ANCHOR_EXPORTS).issubset(set(facade.__all__))
    assert FORBIDDEN_TEMP_NAMES.isdisjoint(set(facade.__all__))
    for name in FORBIDDEN_TEMP_NAMES:
        assert not hasattr(facade, name)


def test_planning_impl_and_facade_preserve_anchor_symbol_identity() -> None:
    planning = importlib.import_module("model_explorer.policy.planning")
    planning_impl = importlib.import_module("model_explorer.policy.planning_impl")
    anchor_facade = importlib.import_module("model_explorer.policy.planning_anchor")

    for name in REPRESENTATIVE_ANCHOR_SYMBOLS:
        anchor_symbol = getattr(anchor_facade, name)
        assert getattr(planning_impl, name) is anchor_symbol
        assert getattr(planning, name) is anchor_symbol
