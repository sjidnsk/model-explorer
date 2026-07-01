from __future__ import annotations

from typing import Any

from .feedback_selection_anchor import (
    _anchor_projection_source_selection_quality,
    _candidate_generation_for_selection,
)
from .feedback_selection_scoring import _finite_float, _ranking_key
from .feedback_selection_sources import (
    _best_source_selection_alternative,
    _candidate_float,
    _selection_payload,
    _source_selection_key,
    _source_selection_quality_regression,
)
from .path_feedback_manifest import cell_tuple as _cell_tuple
from .planning_types import AnchorProjectionCandidateConfig, PathCandidateEvaluation


def _contract_aware_preferred_evaluation(
    evaluations: tuple[PathCandidateEvaluation, ...],
    *,
    goals_by_action_index: dict[int, GoalCandidate],
    scores: dict[int, float],
    config: AnchorProjectionCandidateConfig,
) -> PathCandidateEvaluation | None:
    if not config.enabled or not config.prefer_contract_safe_trainable_targets:
        return None
    eligible = [
        evaluation
        for evaluation in evaluations
        if _preferred_trainable_evaluation(evaluation, config=config)
        and not _source_selection_quality_regression(evaluation, evaluations, config=config)
    ]
    if not eligible:
        return None
    return min(eligible, key=lambda evaluation: _ranking_key(evaluation, goals_by_action_index, scores))


def _preferred_trainable_evaluation(
    evaluation: PathCandidateEvaluation,
    *,
    config: AnchorProjectionCandidateConfig,
) -> bool:
    return _contract_safe_trainable_evaluation(
        evaluation,
        config=config,
    ) or _planner_validated_distance_exception_evaluation(evaluation, config=config)


def _contract_safe_trainable_evaluation(
    evaluation: PathCandidateEvaluation,
    *,
    config: AnchorProjectionCandidateConfig,
) -> bool:
    generation = evaluation.candidate_generation if isinstance(evaluation.candidate_generation, dict) else {}
    if generation.get("candidate_role") != "projected_execution_target":
        return False
    if generation.get("target_binding_mode") != "same_action_execution_substitute":
        return False
    if generation.get("ppo_consumable_action") is not True:
        return False
    if generation.get("contract_safe") is not True:
        return False
    if generation.get("anchor_reachable") is not True:
        return False
    distance_cells = _finite_float(generation.get("projection_distance_cells"))
    distance_m = _finite_float(generation.get("projection_distance_m"))
    return (
        evaluation.result.feasible
        and not evaluation.result.replan_required
        and distance_cells <= float(config.max_trainable_projection_distance_cells)
        and distance_m <= float(config.max_trainable_projection_distance_m)
    )


def _planner_validated_distance_exception_evaluation(
    evaluation: PathCandidateEvaluation,
    *,
    config: AnchorProjectionCandidateConfig,
) -> bool:
    if (
        not config.planner_validated_trainable_target_mining
        or not config.allow_planner_validated_distance_exception
    ):
        return False
    generation = evaluation.candidate_generation if isinstance(evaluation.candidate_generation, dict) else {}
    if generation.get("candidate_role") != "projected_execution_target":
        return False
    if generation.get("target_binding_mode") != "same_action_execution_substitute":
        return False
    if generation.get("ppo_consumable_action") is not True:
        return False
    if generation.get("anchor_reachable") is not True:
        return False
    if generation.get("planner_validated_exception_safe") is not True:
        return False
    distance_cells = _finite_float(generation.get("projection_distance_cells"))
    distance_m = _finite_float(generation.get("projection_distance_m"))
    return (
        evaluation.result.feasible
        and not evaluation.result.replan_required
        and distance_cells <= float(config.max_planner_validated_distance_cells)
        and distance_m <= float(config.max_planner_validated_distance_m)
    )


def _contract_aware_preferred_selection(
    evaluations: list[Any],
    *,
    config: AnchorProjectionCandidateConfig,
) -> Any | None:
    if not config.enabled or not config.prefer_contract_safe_trainable_targets:
        return None
    candidate_payloads = [_selection_payload(item) for item in evaluations]
    eligible: list[Any] = []
    for evaluation, payload in zip(evaluations, candidate_payloads):
        if not _preferred_trainable_candidate(payload, config=config):
            continue
        alternative = _best_source_selection_alternative(
            candidate_payloads,
            selected_action_index=payload.get("action_index"),
            selected_cell=_cell_tuple(payload.get("cell")),
        )
        quality_regression, _ = _anchor_projection_source_selection_quality(
            payload,
            alternative=alternative,
            config=config,
        )
        if not quality_regression:
            eligible.append(evaluation)
    if not eligible:
        return None
    return min(eligible, key=lambda item: _source_selection_key(item, config=config))


def _preferred_trainable_candidate(
    evaluation_or_candidate: Any,
    *,
    config: AnchorProjectionCandidateConfig,
) -> bool:
    return _contract_safe_trainable_candidate(
        evaluation_or_candidate,
        config=config,
    ) or _planner_validated_distance_exception_candidate(evaluation_or_candidate, config=config)


def _contract_safe_trainable_candidate(
    evaluation_or_candidate: Any,
    *,
    config: AnchorProjectionCandidateConfig,
) -> bool:
    generation = _candidate_generation_for_selection(evaluation_or_candidate)
    if generation.get("candidate_role") != "projected_execution_target":
        return False
    if generation.get("target_binding_mode") != "same_action_execution_substitute":
        return False
    if generation.get("ppo_consumable_action") is not True:
        return False
    if generation.get("contract_safe") is not True:
        return False
    if generation.get("anchor_reachable") is not True:
        return False
    distance_cells = _candidate_float(generation, "projection_distance_cells", float("inf"))
    distance_m = _candidate_float(generation, "projection_distance_m", float("inf"))
    return (
        distance_cells <= float(config.max_trainable_projection_distance_cells)
        and distance_m <= float(config.max_trainable_projection_distance_m)
    )


def _planner_validated_distance_exception_candidate(
    evaluation_or_candidate: Any,
    *,
    config: AnchorProjectionCandidateConfig,
) -> bool:
    if (
        not config.planner_validated_trainable_target_mining
        or not config.allow_planner_validated_distance_exception
    ):
        return False
    generation = _candidate_generation_for_selection(evaluation_or_candidate)
    if generation.get("candidate_role") != "projected_execution_target":
        return False
    if generation.get("target_binding_mode") != "same_action_execution_substitute":
        return False
    if generation.get("ppo_consumable_action") is not True:
        return False
    if generation.get("anchor_reachable") is not True:
        return False
    if generation.get("planner_validated_exception_safe") is not True:
        return False
    distance_cells = _candidate_float(generation, "projection_distance_cells", float("inf"))
    distance_m = _candidate_float(generation, "projection_distance_m", float("inf"))
    return (
        distance_cells <= float(config.max_planner_validated_distance_cells)
        and distance_m <= float(config.max_planner_validated_distance_m)
    )


def _updated_trainability_gate(
    value: Any,
    *,
    update: dict[str, Any],
    ppo_consumable_trainable: bool,
    contract_aware_mode: bool,
) -> dict[str, Any]:
    gate = dict(value) if isinstance(value, dict) else {}
    reason_codes = [
        str(item)
        for item in gate.get("reason_codes", [])
        if item is not None
    ] if isinstance(gate.get("reason_codes"), list) else []
    reject_reason = update.get("reject_reason")
    if reject_reason and reject_reason not in reason_codes:
        reason_codes.append(str(reject_reason))
    if update.get("training_use") == "trainable_anchor_projection_contrast":
        status = "selected_trainable"
        reason_codes = []
    elif update.get("source_selection_status") == "pending_source_selection":
        status = "eligible_if_source_selected"
    else:
        status = "rejected"
    gate.update(
        {
            "status": status,
            "reason_codes": reason_codes,
            "ppo_consumable_action": bool(gate.get("ppo_consumable_action", ppo_consumable_trainable)),
            "contract_safe": bool(gate.get("contract_safe", ppo_consumable_trainable or not contract_aware_mode)),
        }
    )
    return gate


__all__ = (
    "_contract_aware_preferred_evaluation",
    "_preferred_trainable_evaluation",
    "_contract_safe_trainable_evaluation",
    "_planner_validated_distance_exception_evaluation",
    "_contract_aware_preferred_selection",
    "_preferred_trainable_candidate",
    "_contract_safe_trainable_candidate",
    "_planner_validated_distance_exception_candidate",
    "_updated_trainability_gate",
)
