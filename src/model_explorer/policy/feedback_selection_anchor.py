from __future__ import annotations

from collections import Counter
from typing import Any

from .path_feedback_manifest import cell_tuple as _cell_tuple
from .planning_anchor_projection import anchor_projection_candidate_config_from_mapping
from .planning_types import AnchorProjectionCandidateConfig


SOURCE_SELECTION_BEST_ALTERNATIVE_SCOPE = "reachable_non_replan_candidates_including_policy_and_projected_targets"


def annotate_source_selected_anchor_projection(
    feedback: dict[str, Any],
    *,
    selected_evaluation: Any | None,
    anchor_projection_candidate_config: AnchorProjectionCandidateConfig | dict[str, Any] | None = None,
) -> dict[str, Any]:
    from .feedback_selection_sources import _best_source_selection_alternative
    from .feedback_selection_trainability import (
        _contract_safe_trainable_candidate,
        _planner_validated_distance_exception_candidate,
        _updated_trainability_gate,
    )

    if not isinstance(feedback, dict):
        return feedback
    candidates = feedback.get("candidates")
    if not isinstance(candidates, list):
        return feedback
    projection_config = anchor_projection_candidate_config_from_mapping(anchor_projection_candidate_config)
    selected_action_index = None if selected_evaluation is None else getattr(selected_evaluation, "action_index", None)
    selected_cell = None if selected_evaluation is None else getattr(selected_evaluation, "cell", None)
    best_alternative = _best_source_selection_alternative(
        candidates,
        selected_action_index=selected_action_index,
        selected_cell=selected_cell,
    )
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        generation = candidate.get("candidate_generation")
        if not isinstance(generation, dict):
            continue
        if generation.get("candidate_role") != "projected_execution_target":
            continue
        adjusted_path_cost = _anchor_projection_adjusted_path_cost(
            candidate,
            config=projection_config,
        )
        candidate_cell = _cell_tuple(candidate.get("cell"))
        source_selected = (
            selected_action_index is not None
            and candidate.get("action_index") == selected_action_index
            and selected_cell is not None
            and candidate_cell == selected_cell
        )
        quality_regression, quality_update = _anchor_projection_source_selection_quality(
            candidate,
            alternative=best_alternative,
            config=projection_config,
        )
        contract_aware_mode = projection_config.contract_aware_trainable_target_generation
        ppo_consumable_trainable = _contract_safe_trainable_candidate(candidate, config=projection_config)
        planner_validated_exception = _planner_validated_distance_exception_candidate(
            candidate,
            config=projection_config,
        )
        if source_selected and candidate.get("reachable") is True and not bool(candidate.get("replan_required")):
            if quality_regression:
                update = {
                    "training_use": "not_positive_evidence",
                    "sample_weight": 0.0,
                    "reject_reason": "source_selection_quality_regression",
                    "source_selection_status": "source_selected_quality_regression",
                    "comparison_scope": "projected_target_anchor_contrast",
                    "scope": "projected_target_anchor_contrast",
                    "evidence_boundary": "source_selected_projected_target_quality_regression_not_positive_evidence",
                    "audit_proxy_positive_evidence": False,
                    "source_selection_path_cost_bonus": projection_config.source_selection_path_cost_bonus,
                    "source_selection_adjusted_path_cost": adjusted_path_cost,
                }
            elif contract_aware_mode and not ppo_consumable_trainable and not planner_validated_exception:
                update = {
                    "training_use": "not_positive_evidence",
                    "sample_weight": 0.0,
                    "reject_reason": "not_ppo_consumable_action",
                    "source_selection_status": "source_selected_not_ppo_consumable",
                    "comparison_scope": "projected_target_anchor_contrast",
                    "scope": "projected_target_anchor_contrast",
                    "evidence_boundary": "source_selected_projected_target_not_ppo_consumable",
                    "audit_proxy_positive_evidence": False,
                    "source_selection_path_cost_bonus": projection_config.source_selection_path_cost_bonus,
                    "source_selection_adjusted_path_cost": adjusted_path_cost,
                }
            elif contract_aware_mode and planner_validated_exception and not ppo_consumable_trainable:
                update = {
                    "training_use": "not_positive_evidence",
                    "sample_weight": 0.0,
                    "reject_reason": "planner_validated_distance_exception_pending_mining",
                    "source_selection_status": "source_selected",
                    "comparison_scope": "projected_target_anchor_contrast",
                    "scope": "projected_target_anchor_contrast",
                    "evidence_boundary": "source_selected_planner_validated_distance_exception_pending_mining",
                    "audit_proxy_positive_evidence": False,
                    "source_selection_path_cost_bonus": projection_config.source_selection_path_cost_bonus,
                    "source_selection_adjusted_path_cost": adjusted_path_cost,
                    "planner_validated_mining_decision": (
                        "selected_planner_validated_distance_exception"
                    ),
                    "planner_validated_trainable_target_mining": True,
                }
            else:
                update = {
                    "training_use": "trainable_anchor_projection_contrast",
                    "sample_weight": 1.0,
                    "reject_reason": None,
                    "source_selection_status": "source_selected",
                    "comparison_scope": "projected_target_anchor_contrast",
                    "scope": "projected_target_anchor_contrast",
                    "evidence_boundary": "source_selected_projected_target_candidate",
                    "audit_proxy_positive_evidence": False,
                    "source_selection_path_cost_bonus": projection_config.source_selection_path_cost_bonus,
                    "source_selection_adjusted_path_cost": adjusted_path_cost,
                }
        else:
            update = {
                "training_use": "not_positive_evidence",
                "sample_weight": 0.0,
                "reject_reason": (
                    "projected_candidate_replan_required"
                    if bool(candidate.get("replan_required"))
                    else "source_candidate_not_selected"
                ),
                "source_selection_status": "not_source_selected",
                "comparison_scope": "projected_target_anchor_contrast",
                "scope": "projected_target_anchor_contrast",
                "evidence_boundary": "source_candidate_not_selected_not_positive_evidence",
                "audit_proxy_positive_evidence": False,
                "source_selection_path_cost_bonus": projection_config.source_selection_path_cost_bonus,
                "source_selection_adjusted_path_cost": adjusted_path_cost,
            }
            quality_update["source_selection_quality_regression"] = False
        update["trainability_gate"] = _updated_trainability_gate(
            generation.get("trainability_gate"),
            update=update,
            ppo_consumable_trainable=ppo_consumable_trainable,
            contract_aware_mode=contract_aware_mode,
        )
        update.update(quality_update)
        generation.update(update)
        feasibility = candidate.get("platform_goal_feasibility")
        feasibility = feasibility if isinstance(feasibility, dict) else {}
        projection = feasibility.get("anchor_projection")
        if isinstance(projection, dict):
            projection.update(update)
            projection["same_cell_positive_evidence"] = False
    return feedback


def _anchor_projection_adjusted_path_cost(
    evaluation_or_candidate: Any,
    *,
    config: AnchorProjectionCandidateConfig,
) -> float:
    from .feedback_selection_sources import _path_cost_for_selection
    from .feedback_selection_trainability import (
        _contract_safe_trainable_candidate,
        _planner_validated_distance_exception_candidate,
    )

    path_cost = _path_cost_for_selection(evaluation_or_candidate)
    if not config.enabled:
        return path_cost
    if config.source_selection_path_cost_bonus <= 0.0:
        return path_cost
    candidate_generation = _candidate_generation_for_selection(evaluation_or_candidate)
    if candidate_generation.get("candidate_role") != "projected_execution_target":
        return path_cost
    if candidate_generation.get("comparison_scope") != "projected_target_anchor_contrast":
        return path_cost
    if candidate_generation.get("anchor_reachable") is not True:
        return path_cost
    if config.prefer_contract_safe_trainable_targets and not _contract_safe_trainable_candidate(
        evaluation_or_candidate,
        config=config,
    ) and not _planner_validated_distance_exception_candidate(evaluation_or_candidate, config=config):
        return path_cost
    return path_cost - float(config.source_selection_path_cost_bonus)


def _anchor_projection_candidate_generation_summary(
    scenarios: list[dict[str, Any]],
) -> dict[str, Any]:
    generated_count = 0
    source_selected_count = 0
    trainable_count = 0
    nontrainable_count = 0
    ppo_consumable_trainable_count = 0
    positive_audit_proxy_count = 0
    reject_reason_counts: Counter[str] = Counter()
    for scenario in scenarios:
        feedback = scenario.get("path_feedback")
        feedback = feedback if isinstance(feedback, dict) else {}
        candidates = feedback.get("candidates")
        if not isinstance(candidates, list):
            continue
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            generation = candidate.get("candidate_generation")
            if not isinstance(generation, dict):
                continue
            if generation.get("candidate_role") != "projected_execution_target":
                continue
            generated_count += 1
            if generation.get("source_selection_status") == "source_selected":
                source_selected_count += 1
            if generation.get("training_use") == "trainable_anchor_projection_contrast":
                trainable_count += 1
                if generation.get("ppo_consumable_action") is True:
                    ppo_consumable_trainable_count += 1
                if generation.get("comparison_scope") == "audit_proxy_anchor_not_same_cell":
                    positive_audit_proxy_count += 1
            else:
                nontrainable_count += 1
            reject_reason = generation.get("reject_reason")
            if reject_reason:
                reject_reason_counts[str(reject_reason)] += 1
    return {
        "anchor_projection_candidate_generated_count": generated_count,
        "anchor_projection_source_selected_count": source_selected_count,
        "trainable_anchor_projection_count": trainable_count,
        "ppo_consumable_trainable_target_count": ppo_consumable_trainable_count,
        "nontrainable_anchor_projection_count": nontrainable_count,
        "positive_training_evidence_contains_audit_proxy_anchor_count": positive_audit_proxy_count,
        "anchor_projection_candidate_reject_reason_counts": dict(sorted(reject_reason_counts.items())),
    }


def _anchor_projection_source_selection_quality(
    candidate: dict[str, Any],
    *,
    alternative: dict[str, Any] | None,
    config: AnchorProjectionCandidateConfig,
) -> tuple[bool, dict[str, Any]]:
    from .feedback_selection_sources import _candidate_float, _list_cell

    update: dict[str, Any] = {}
    if alternative is None:
        return False, update
    path_margin = _candidate_float(candidate, "path_cost", 0.0) - _candidate_float(
        alternative,
        "path_cost",
        0.0,
    )
    risk_margin = _candidate_float(candidate, "risk", 0.0) - _candidate_float(alternative, "risk", 0.0)
    update["source_selection_best_alternative_action_index"] = alternative.get("action_index")
    update["source_selection_best_alternative_cell"] = _list_cell(_cell_tuple(alternative.get("cell")))
    update["source_selection_best_alternative_scope"] = SOURCE_SELECTION_BEST_ALTERNATIVE_SCOPE
    update["source_selection_best_alternative_candidate_role"] = (
        alternative.get("candidate_role")
        or _candidate_generation_for_selection(alternative).get("candidate_role")
    )
    update["source_selection_path_cost_margin_vs_best_alternative"] = float(path_margin)
    update["source_selection_risk_margin_vs_best_alternative"] = float(risk_margin)
    path_regressed = (
        config.max_source_selection_path_cost_regression is not None
        and path_margin > float(config.max_source_selection_path_cost_regression)
    )
    risk_regressed = (
        config.max_source_selection_risk_regression is not None
        and risk_margin > float(config.max_source_selection_risk_regression)
    )
    update["source_selection_quality_regression"] = bool(path_regressed or risk_regressed)
    return bool(path_regressed or risk_regressed), update


def _candidate_generation_for_selection(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        generation = value.get("candidate_generation")
    else:
        generation = getattr(value, "candidate_generation", None)
    return generation if isinstance(generation, dict) else {}


__all__ = (
    "SOURCE_SELECTION_BEST_ALTERNATIVE_SCOPE",
    "annotate_source_selected_anchor_projection",
    "_anchor_projection_adjusted_path_cost",
    "_anchor_projection_candidate_generation_summary",
    "_anchor_projection_source_selection_quality",
    "_candidate_generation_for_selection",
)
