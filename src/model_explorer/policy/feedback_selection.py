from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from math import isfinite
from typing import Any

from ..core.interfaces import ExplorerDecision, GoalCandidate, ModelExplorerContract
from .planning_anchor import (
    anchor_projection_candidate_config_from_mapping,
    evaluate_candidate_paths,
)
from .planning_types import (
    AnchorProjectionCandidateConfig,
    PathCandidateEvaluation,
    PathPlanningAdapter,
)
from .path_feedback_manifest import cell_tuple as _cell_tuple


@dataclass(frozen=True)
class FeedbackAwareSelectionConfig:
    coverage_weight: float = 0.35
    information_gain_weight: float = 0.20
    confidence_gain_weight: float = 0.20
    value_weight: float = 0.15
    path_cost_weight: float = 0.10
    risk_weight: float = 0.15
    failure_penalty: float = 1.0
    replan_penalty: float = 0.5
    tracking_safety_penalty: float = 0.25
    postprocess_fallback_penalty: float = 0.15
    trajectory_optimization_fallback_penalty: float = 0.20
    region_graph_disconnected_penalty: float = 0.20
    region_graph_fallback_penalty: float = 0.10
    iris_fallback_penalty: float = 0.05
    open_grid_fallback_penalty: float = 0.05
    channel_aware_quality_bonus: float = 0.05


@dataclass(frozen=True)
class FeedbackAwareSelection:
    decision: ExplorerDecision
    evaluations: tuple[PathCandidateEvaluation, ...]
    scores_by_action_index: dict[int, float]
    selected_evaluation: PathCandidateEvaluation | None = None
    selected_action_index: int | None = None
    selected_score: float | None = None
    runner_up_action_index: int | None = None
    runner_up_score: float | None = None
    score_margin: float | None = None
    ranked_action_indices: tuple[int, ...] = ()
    channel_aware_evidence_by_action_index: dict[int, dict[str, Any]] = field(default_factory=dict)
    channel_aware_score_adjustments_by_action_index: dict[int, float] = field(default_factory=dict)


def select_goal_with_path_feedback(
    contract: ModelExplorerContract,
    *,
    planner: PathPlanningAdapter,
    current_cell: tuple[int, int] = (0, 0),
    step_index: int = 0,
    top_k: int | None = None,
    config: FeedbackAwareSelectionConfig | None = None,
    anchor_projection_candidate_config: AnchorProjectionCandidateConfig | dict[str, Any] | None = None,
) -> FeedbackAwareSelection:
    selection_config = config or FeedbackAwareSelectionConfig()
    projection_config = anchor_projection_candidate_config_from_mapping(anchor_projection_candidate_config)
    candidate_limit = len(contract.top_goals) if top_k is None else max(0, int(top_k))
    evaluations = evaluate_candidate_paths(
        contract,
        current_cell=current_cell,
        top_k=candidate_limit,
        planner=planner,
        step_index=step_index,
        anchor_projection_candidate_config=projection_config,
    )
    if not evaluations:
        decision = ExplorerDecision(status="no_reachable_goal", selected_goal=None, ranked_goals=())
        return FeedbackAwareSelection(
            decision=decision,
            evaluations=evaluations,
            scores_by_action_index={},
            selected_evaluation=None,
        )

    goals_by_action_index = _goals_by_evaluation_action_index(evaluations, contract)
    scores = _score_evaluations(evaluations, goals_by_action_index, selection_config)
    channel_aware_evidence_by_action_index = _channel_aware_evidence_by_action_index(evaluations)
    channel_aware_score_adjustments_by_action_index = {
        evaluation.action_index: _channel_aware_score_adjustment(
            evaluation.result.metadata,
            config=selection_config,
        )
        for evaluation in evaluations
        if evaluation.action_index in channel_aware_evidence_by_action_index
    }
    preferred_evaluation = _contract_aware_preferred_evaluation(
        evaluations,
        goals_by_action_index=goals_by_action_index,
        scores=scores,
        config=projection_config,
    )
    ranked_evaluations = tuple(
        sorted(
            evaluations,
            key=lambda evaluation: _ranking_key(evaluation, goals_by_action_index, scores),
        )
    )
    if preferred_evaluation is not None:
        ranked_evaluations = (preferred_evaluation,) + tuple(
            evaluation for evaluation in ranked_evaluations if evaluation is not preferred_evaluation
        )
    ranked_goals = tuple(
        goals_by_action_index[evaluation.action_index]
        for evaluation in ranked_evaluations
        if evaluation.action_index in goals_by_action_index
    )
    ranked_action_indices = tuple(
        evaluation.action_index
        for evaluation in ranked_evaluations
        if evaluation.action_index in goals_by_action_index
    )
    selected_evaluation = ranked_evaluations[0]
    selected_action_index = selected_evaluation.action_index
    selected_goal = goals_by_action_index[selected_action_index]
    selected_score = scores.get(selected_action_index)
    runner_up_action_index = ranked_action_indices[1] if len(ranked_action_indices) > 1 else None
    runner_up_score = None if runner_up_action_index is None else scores.get(runner_up_action_index)
    score_margin = (
        None
        if selected_score is None or runner_up_score is None
        else _finite_float(selected_score - runner_up_score)
    )
    decision = ExplorerDecision(status="selected", selected_goal=selected_goal, ranked_goals=ranked_goals)
    return FeedbackAwareSelection(
        decision=decision,
        evaluations=evaluations,
        scores_by_action_index=scores,
        selected_evaluation=selected_evaluation,
        selected_action_index=selected_action_index,
        selected_score=selected_score,
        runner_up_action_index=runner_up_action_index,
        runner_up_score=runner_up_score,
        score_margin=score_margin,
        ranked_action_indices=ranked_action_indices,
        channel_aware_evidence_by_action_index=channel_aware_evidence_by_action_index,
        channel_aware_score_adjustments_by_action_index=channel_aware_score_adjustments_by_action_index,
    )


def _goals_by_evaluation_action_index(
    evaluations: tuple[PathCandidateEvaluation, ...],
    contract: ModelExplorerContract,
) -> dict[int, GoalCandidate]:
    goals: dict[int, GoalCandidate] = {}
    for evaluation in evaluations:
        if evaluation.selection_goal is not None:
            goals[evaluation.action_index] = evaluation.selection_goal
            continue
        if 0 <= evaluation.action_index < len(contract.top_goals):
            goal = contract.top_goals[evaluation.action_index]
            if goal.reachable:
                goals[evaluation.action_index] = goal
            continue
        goals[evaluation.action_index] = GoalCandidate(
            cell=evaluation.cell,
            utility=evaluation.utility,
            reachable=bool(evaluation.result.feasible),
        )
    return goals


def _score_evaluations(
    evaluations: tuple[PathCandidateEvaluation, ...],
    goals_by_action_index: dict[int, GoalCandidate],
    config: FeedbackAwareSelectionConfig,
) -> dict[int, float]:
    normalized = _normalized_features(evaluations, goals_by_action_index)
    scores: dict[int, float] = {}
    for index, evaluation in enumerate(evaluations):
        result = evaluation.result
        score = (
            config.coverage_weight * normalized["coverage"][index]
            + config.information_gain_weight * normalized["information_gain"][index]
            + config.confidence_gain_weight * normalized["confidence_gain"][index]
            + config.value_weight * normalized["value"][index]
            - config.path_cost_weight * normalized["path_cost"][index]
            - config.risk_weight * normalized["risk"][index]
            - _path_feedback_penalty(result=result, config=config)
            + _channel_aware_score_adjustment(result.metadata, config=config)
        )
        scores[evaluation.action_index] = _finite_float(score)
    return scores


def _normalized_features(
    evaluations: tuple[PathCandidateEvaluation, ...],
    goals_by_action_index: dict[int, GoalCandidate],
) -> dict[str, tuple[float, ...]]:
    raw_values = {
        "coverage": tuple(
            _coverage_value(goals_by_action_index[evaluation.action_index]) for evaluation in evaluations
        ),
        "information_gain": tuple(
            _numeric_experimental(goals_by_action_index[evaluation.action_index], "information_gain")
            for evaluation in evaluations
        ),
        "confidence_gain": tuple(
            _numeric_experimental(goals_by_action_index[evaluation.action_index], "confidence_gain")
            for evaluation in evaluations
        ),
        "value": tuple(
            _numeric_experimental(goals_by_action_index[evaluation.action_index], "value")
            for evaluation in evaluations
        ),
        "path_cost": tuple(_finite_float(evaluation.result.path_cost) for evaluation in evaluations),
        "risk": tuple(_finite_float(evaluation.result.risk) for evaluation in evaluations),
    }
    return {field: _min_max_normalize(values) for field, values in raw_values.items()}


def _path_feedback_penalty(
    *,
    result,
    config: FeedbackAwareSelectionConfig,
) -> float:
    penalty = 0.0
    if not result.feasible or result.failure_reason is not None:
        penalty += config.failure_penalty
    if result.replan_required:
        penalty += config.replan_penalty

    metadata = result.metadata if isinstance(result.metadata, dict) else {}
    postprocess = metadata.get("postprocess")
    if isinstance(postprocess, dict):
        if _fallback_status_is_problem(postprocess.get("fallback_status")):
            penalty += config.postprocess_fallback_penalty
        if _tracking_safety_violation_count(postprocess) > 0:
            penalty += config.tracking_safety_penalty

    optimization = metadata.get("trajectory_optimization_report")
    if isinstance(optimization, dict) and _fallback_status_is_problem(optimization.get("fallback_status")):
        penalty += config.trajectory_optimization_fallback_penalty

    region_graph = metadata.get("region_graph_report")
    if isinstance(region_graph, dict):
        quality = region_graph.get("quality_metrics")
        quality = quality if isinstance(quality, dict) else {}
        if quality.get("start_goal_connected") is False:
            penalty += config.region_graph_disconnected_penalty
        if bool(region_graph.get("fallback_used")):
            penalty += config.region_graph_fallback_penalty

    iris_region = metadata.get("iris_region_report")
    if isinstance(iris_region, dict) and bool(iris_region.get("fallback_used")):
        penalty += config.iris_fallback_penalty

    request_payload = metadata.get("request_payload")
    request_metadata = request_payload.get("metadata") if isinstance(request_payload, dict) else {}
    request_metadata = request_metadata if isinstance(request_metadata, dict) else {}
    if (
        request_metadata.get("cost_source") == "open_grid_fallback"
        or request_metadata.get("passable_mask_source") == "open_grid_fallback"
    ):
        penalty += config.open_grid_fallback_penalty

    return penalty


def classify_channel_aware_feedback_evidence(report: Any) -> dict[str, Any]:
    report = report if isinstance(report, dict) else {}
    requested_backend = str(report.get("requested_backend", ""))
    selected_backend = str(report.get("selected_backend", ""))
    status = str(report.get("status", ""))
    present = requested_backend == "channel_aware_astar" or selected_backend == "channel_aware_astar"
    comparison = report.get("comparison") if isinstance(report.get("comparison"), dict) else {}
    path_cost_delta = _finite_float_optional(comparison.get("path_cost_delta"))
    channel_cost_delta = _finite_float_optional(comparison.get("channel_cost_delta"))
    high_cost_exposure_delta = _finite_float_optional(comparison.get("high_cost_exposure_delta"))
    risk_delta = _finite_float_optional(comparison.get("risk_delta"))
    selected = status == "selected" or selected_backend == "channel_aware_astar"
    quality_improvement = bool(
        selected
        and high_cost_exposure_delta is not None
        and high_cost_exposure_delta < 0.0
        and channel_cost_delta is not None
        and channel_cost_delta < 0.0
    )
    path_cost_tradeoff = bool(selected and path_cost_delta is not None and path_cost_delta > 0.0)
    risk_or_high_cost_improvement = bool(
        (high_cost_exposure_delta is not None and high_cost_exposure_delta < 0.0)
        or (risk_delta is not None and risk_delta < 0.0)
    )
    blocker = _channel_aware_blocker_reason(report)
    reason_codes: list[str] = []
    if not present:
        reason_codes.append("channel_aware_evidence_missing")
    if quality_improvement:
        reason_codes.append("channel_aware_quality_improved")
    elif selected:
        reason_codes.append("channel_aware_quality_not_improved")
    if path_cost_tradeoff:
        reason_codes.append("path_cost_tradeoff")
    if blocker not in (None, "selected"):
        reason_codes.append(blocker)

    if not present:
        recommendation = "needs_more_evidence"
    elif quality_improvement:
        recommendation = "keep"
    elif blocker in {"goal_blocked", "not_lower_risk"}:
        recommendation = "reject"
    elif blocker == "same_as_baseline" or selected:
        recommendation = "downweight"
    else:
        recommendation = "needs_more_evidence"

    return {
        "schema_version": "channel-aware-feedback-evidence/v1",
        "present": present,
        "requested_backend": requested_backend or None,
        "selected_backend": selected_backend or None,
        "status": status or None,
        "selected": selected,
        "quality_improvement": quality_improvement,
        "risk_or_high_cost_improvement": risk_or_high_cost_improvement,
        "path_cost_tradeoff": path_cost_tradeoff,
        "blocker_reason": blocker,
        "recommendation": recommendation,
        "reason_codes": _dedupe(reason_codes),
        "comparison": {
            "path_changed": bool(comparison.get("path_changed", False)),
            "path_cost_delta": path_cost_delta,
            "channel_cost_delta": channel_cost_delta,
            "high_cost_exposure_delta": high_cost_exposure_delta,
            "risk_delta": risk_delta,
        },
    }


def _channel_aware_evidence_by_action_index(
    evaluations: tuple[PathCandidateEvaluation, ...],
) -> dict[int, dict[str, Any]]:
    evidence: dict[int, dict[str, Any]] = {}
    for evaluation in evaluations:
        metadata = evaluation.result.metadata if isinstance(evaluation.result.metadata, dict) else {}
        report = metadata.get("planning_backend_report")
        audit = classify_channel_aware_feedback_evidence(report)
        if audit["present"]:
            evidence[evaluation.action_index] = audit
    return evidence


def _channel_aware_score_adjustment(
    metadata: Any,
    *,
    config: FeedbackAwareSelectionConfig,
) -> float:
    metadata = metadata if isinstance(metadata, dict) else {}
    audit = classify_channel_aware_feedback_evidence(metadata.get("planning_backend_report"))
    if audit["quality_improvement"]:
        return max(0.0, float(config.channel_aware_quality_bonus))
    return 0.0


def _channel_aware_blocker_reason(report: dict[str, Any]) -> str | None:
    blocker = report.get("blocker_class")
    fallback = report.get("fallback_reason")
    for value in (blocker, fallback):
        if value is None:
            continue
        text = str(value)
        if "goal_blocked" in text:
            return "goal_blocked"
        if "same_as_baseline" in text:
            return "same_as_baseline"
        if "not_lower_risk" in text:
            return "not_lower_risk"
    status = str(report.get("status", ""))
    selected_backend = str(report.get("selected_backend", ""))
    if status == "selected" or selected_backend == "channel_aware_astar":
        return "selected"
    return None


def _ranking_key(
    evaluation: PathCandidateEvaluation,
    goals_by_action_index: dict[int, GoalCandidate],
    scores: dict[int, float],
) -> tuple[float, int, int, float, int, int]:
    goal = goals_by_action_index[evaluation.action_index]
    return (
        -scores[evaluation.action_index],
        0 if evaluation.result.feasible else 1,
        1 if evaluation.result.replan_required else 0,
        -goal.utility,
        goal.cell[0],
        goal.cell[1],
    )


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


def _source_selection_quality_regression(
    evaluation: PathCandidateEvaluation,
    evaluations: tuple[PathCandidateEvaluation, ...],
    *,
    config: AnchorProjectionCandidateConfig,
) -> bool:
    alternatives = [
        item
        for item in evaluations
        if item is not evaluation
        and item.result.feasible
        and not item.result.replan_required
    ]
    if not alternatives:
        return False
    alternative = min(
        alternatives,
        key=lambda item: (
            float(item.result.path_cost),
            float(item.result.risk),
            -float(item.utility),
            item.cell[0],
            item.cell[1],
        ),
    )
    path_margin = float(evaluation.result.path_cost) - float(alternative.result.path_cost)
    risk_margin = float(evaluation.result.risk) - float(alternative.result.risk)
    return (
        config.max_source_selection_path_cost_regression is not None
        and path_margin > float(config.max_source_selection_path_cost_regression)
    ) or (
        config.max_source_selection_risk_regression is not None
        and risk_margin > float(config.max_source_selection_risk_regression)
    )


def _coverage_value(goal: GoalCandidate) -> float:
    explicit_rate_delta = _numeric_experimental_optional(goal, "expected_coverage_rate_delta")
    if explicit_rate_delta is not None:
        return explicit_rate_delta
    return _numeric_experimental(goal, "expected_new_coverage_area")


def _numeric_experimental(goal: GoalCandidate, field: str) -> float:
    return _finite_float(goal.experimental.get(field))


def _numeric_experimental_optional(goal: GoalCandidate, field: str) -> float | None:
    value = goal.experimental.get(field)
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if isfinite(number) else None


def _min_max_normalize(values: tuple[float, ...]) -> tuple[float, ...]:
    if not values:
        return ()
    min_value = min(values)
    max_value = max(values)
    if max_value == min_value:
        return tuple(0.0 for _ in values)
    return tuple((value - min_value) / (max_value - min_value) for value in values)


def _finite_float(value: Any) -> float:
    if isinstance(value, bool) or value is None:
        return 0.0
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if isfinite(number) else 0.0


def _finite_float_optional(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if isfinite(number) else None


def _fallback_status_is_problem(value: Any) -> bool:
    return isinstance(value, str) and value not in {"ok", "not_needed", "none"}


def _tracking_safety_violation_count(postprocess: dict[str, Any]) -> int:
    tracking_safety = postprocess.get("tracking_safety_report")
    if isinstance(tracking_safety, dict):
        return _safe_int(tracking_safety.get("violation_count"))
    return _safe_int(postprocess.get("tracking_safety_violation_count"))


def _safe_int(value: Any) -> int:
    if isinstance(value, bool) or value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


SOURCE_SELECTION_BEST_ALTERNATIVE_SCOPE = "reachable_non_replan_candidates_including_policy_and_projected_targets"


def annotate_source_selected_anchor_projection(
    feedback: dict[str, Any],
    *,
    selected_evaluation: Any | None,
    anchor_projection_candidate_config: AnchorProjectionCandidateConfig | dict[str, Any] | None = None,
) -> dict[str, Any]:
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


def _selected_before_feedback(contract: ModelExplorerContract) -> GoalCandidate | None:
    for goal in contract.top_goals:
        if goal.reachable:
            return goal
    return None


def _selected_after_feedback(
    evaluations,
    *,
    anchor_projection_candidate_config: AnchorProjectionCandidateConfig | dict[str, Any] | None = None,
) -> Any | None:
    projection_config = anchor_projection_candidate_config_from_mapping(anchor_projection_candidate_config)
    feasible = [item for item in evaluations if item.result.feasible and not item.result.replan_required]
    if not feasible:
        feasible = [item for item in evaluations if item.result.feasible]
    if not feasible:
        return None
    preferred = _contract_aware_preferred_selection(feasible, config=projection_config)
    if preferred is not None:
        return preferred
    return min(
        feasible,
        key=lambda item: _source_selection_key(item, config=projection_config),
    )


def _source_selection_key(
    evaluation: Any,
    *,
    config: AnchorProjectionCandidateConfig,
) -> tuple[float, float, float, int, int]:
    return (
        _anchor_projection_adjusted_path_cost(evaluation, config=config),
        float(evaluation.result.risk),
        -float(evaluation.utility),
        evaluation.cell[0],
        evaluation.cell[1],
    )


def _anchor_projection_adjusted_path_cost(
    evaluation_or_candidate: Any,
    *,
    config: AnchorProjectionCandidateConfig,
) -> float:
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


def _selection_payload(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {
        "action_index": getattr(value, "action_index", None),
        "source_action_index": getattr(value, "source_action_index", None),
        "cell": _list_cell(getattr(value, "cell", None)),
        "candidate_role": _candidate_generation_for_selection(value).get("candidate_role", "policy_target"),
        "reachable": bool(getattr(value.result, "feasible", False)),
        "replan_required": bool(getattr(value.result, "replan_required", False)),
        "path_cost": float(getattr(value.result, "path_cost", 0.0)),
        "risk": float(getattr(value.result, "risk", 0.0)),
        "utility": float(getattr(value, "utility", 0.0)),
        "candidate_generation": dict(_candidate_generation_for_selection(value)),
    }


def _best_source_selection_alternative(
    candidates: list[Any],
    *,
    selected_action_index: Any,
    selected_cell: tuple[int, int] | None,
) -> dict[str, Any] | None:
    alternatives = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        # This mirrors the source-selection pool: policy targets and projected
        # targets are both valid contrasts if they are reachable and non-replan.
        if candidate.get("reachable") is not True or bool(candidate.get("replan_required")):
            continue
        if candidate.get("action_index") == selected_action_index and _cell_tuple(candidate.get("cell")) == selected_cell:
            continue
        alternatives.append(candidate)
    if not alternatives:
        return None
    return min(
        alternatives,
        key=lambda candidate: (
            _candidate_float(candidate, "path_cost", float("inf")),
            _candidate_float(candidate, "risk", float("inf")),
            -_candidate_float(candidate, "utility", 0.0),
            _cell_tuple(candidate.get("cell")) or (10**9, 10**9),
        ),
    )


def _anchor_projection_source_selection_quality(
    candidate: dict[str, Any],
    *,
    alternative: dict[str, Any] | None,
    config: AnchorProjectionCandidateConfig,
) -> tuple[bool, dict[str, Any]]:
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


def _candidate_float(candidate: dict[str, Any], field: str, default: float) -> float:
    try:
        return float(candidate.get(field, default))
    except (TypeError, ValueError):
        return default


def _list_cell(cell: tuple[int, int] | None) -> list[int] | None:
    if cell is None:
        return None
    return [cell[0], cell[1]]


def _path_cost_for_selection(value: Any) -> float:
    if isinstance(value, dict):
        try:
            return float(value.get("path_cost", 0.0))
        except (TypeError, ValueError):
            return 0.0
    return float(value.result.path_cost)


def _candidate_generation_for_selection(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        generation = value.get("candidate_generation")
    else:
        generation = getattr(value, "candidate_generation", None)
    return generation if isinstance(generation, dict) else {}


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


def _candidate_path_cost_for_cell(evaluations, cell: tuple[int, int] | None) -> float | None:
    if cell is None:
        return None
    for item in evaluations:
        if item.cell == cell:
            return float(item.result.path_cost) if item.result.feasible else None
    return None


def _path_cost_delta(before: float | None, after: float | None) -> float | None:
    if before is None or after is None:
        return None
    return float(after - before)


selected_before_feedback = _selected_before_feedback
selected_after_feedback = _selected_after_feedback
source_selection_key = _source_selection_key
anchor_projection_adjusted_path_cost = _anchor_projection_adjusted_path_cost
contract_aware_preferred_selection = _contract_aware_preferred_selection
preferred_trainable_candidate = _preferred_trainable_candidate
contract_safe_trainable_candidate = _contract_safe_trainable_candidate
planner_validated_distance_exception_candidate = _planner_validated_distance_exception_candidate
selection_payload = _selection_payload
best_source_selection_alternative = _best_source_selection_alternative
anchor_projection_source_selection_quality = _anchor_projection_source_selection_quality
anchor_projection_candidate_generation_summary = _anchor_projection_candidate_generation_summary
candidate_path_cost_for_cell = _candidate_path_cost_for_cell
path_cost_delta = _path_cost_delta

__all__ = [name for name in globals() if not name.startswith('__')]
