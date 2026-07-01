from __future__ import annotations

import importlib


SPLIT_MODULES = (
    "feedback_selection_types",
    "feedback_selection_channel",
    "feedback_selection_scoring",
    "feedback_selection_trainability",
    "feedback_selection_anchor",
    "feedback_selection_sources",
)


PRIVATE_COMPATIBILITY_SYMBOLS = {
    "_channel_aware_evidence_by_action_index": "feedback_selection_channel",
    "_channel_aware_score_adjustment": "feedback_selection_channel",
    "_channel_aware_blocker_reason": "feedback_selection_channel",
    "_goals_by_evaluation_action_index": "feedback_selection_scoring",
    "_score_evaluations": "feedback_selection_scoring",
    "_normalized_features": "feedback_selection_scoring",
    "_path_feedback_penalty": "feedback_selection_scoring",
    "_ranking_key": "feedback_selection_scoring",
    "_coverage_value": "feedback_selection_scoring",
    "_numeric_experimental": "feedback_selection_scoring",
    "_numeric_experimental_optional": "feedback_selection_scoring",
    "_min_max_normalize": "feedback_selection_scoring",
    "_finite_float": "feedback_selection_scoring",
    "_finite_float_optional": "feedback_selection_scoring",
    "_fallback_status_is_problem": "feedback_selection_scoring",
    "_safe_int": "feedback_selection_scoring",
    "_dedupe": "feedback_selection_scoring",
    "_contract_aware_preferred_evaluation": "feedback_selection_trainability",
    "_preferred_trainable_evaluation": "feedback_selection_trainability",
    "_contract_safe_trainable_evaluation": "feedback_selection_trainability",
    "_planner_validated_distance_exception_evaluation": "feedback_selection_trainability",
    "_contract_aware_preferred_selection": "feedback_selection_trainability",
    "_preferred_trainable_candidate": "feedback_selection_trainability",
    "_contract_safe_trainable_candidate": "feedback_selection_trainability",
    "_planner_validated_distance_exception_candidate": "feedback_selection_trainability",
    "_updated_trainability_gate": "feedback_selection_trainability",
    "_anchor_projection_adjusted_path_cost": "feedback_selection_anchor",
    "_anchor_projection_candidate_generation_summary": "feedback_selection_anchor",
    "_anchor_projection_source_selection_quality": "feedback_selection_anchor",
    "_candidate_generation_for_selection": "feedback_selection_anchor",
    "_selected_before_feedback": "feedback_selection_sources",
    "_selected_after_feedback": "feedback_selection_sources",
    "_source_selection_key": "feedback_selection_sources",
    "_selection_payload": "feedback_selection_sources",
    "_best_source_selection_alternative": "feedback_selection_sources",
    "_source_selection_quality_regression": "feedback_selection_sources",
    "_candidate_float": "feedback_selection_sources",
    "_list_cell": "feedback_selection_sources",
    "_path_cost_for_selection": "feedback_selection_sources",
    "_candidate_path_cost_for_cell": "feedback_selection_sources",
    "_path_cost_delta": "feedback_selection_sources",
}


def test_feedback_selection_split_modules_and_facade_contract() -> None:
    modules = {
        name: importlib.import_module(f"model_explorer.policy.{name}")
        for name in SPLIT_MODULES
    }
    facade = importlib.import_module("model_explorer.policy.feedback_selection")

    assert facade.select_goal_with_path_feedback is modules[
        "feedback_selection_scoring"
    ].select_goal_with_path_feedback
    assert facade.FeedbackAwareSelectionConfig is modules[
        "feedback_selection_types"
    ].FeedbackAwareSelectionConfig
    assert facade.selected_after_feedback is modules[
        "feedback_selection_sources"
    ]._selected_after_feedback
    assert facade._selected_after_feedback is modules[
        "feedback_selection_sources"
    ]._selected_after_feedback

    temporary_names = {"Any", "Counter", "dataclass", "field", "isfinite", "annotations"}
    assert temporary_names.isdisjoint(set(facade.__all__))
    assert temporary_names.isdisjoint(vars(facade))


def test_path_feedback_impl_private_compatibility_points_to_split_modules() -> None:
    impl = importlib.import_module("model_explorer.policy.path_feedback_impl")

    for symbol, module_name in PRIVATE_COMPATIBILITY_SYMBOLS.items():
        module = importlib.import_module(f"model_explorer.policy.{module_name}")
        assert getattr(impl, symbol) is getattr(module, symbol)
