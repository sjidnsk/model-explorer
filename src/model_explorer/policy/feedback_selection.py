from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Any

from ..core.interfaces import ExplorerDecision, GoalCandidate, ModelExplorerContract
from .planning import (
    AnchorProjectionCandidateConfig,
    PathCandidateEvaluation,
    PathPlanningAdapter,
    anchor_projection_candidate_config_from_mapping,
    evaluate_candidate_paths,
)


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
        if _contract_safe_trainable_evaluation(evaluation, config=config)
        and not _source_selection_quality_regression(evaluation, evaluations, config=config)
    ]
    if not eligible:
        return None
    return min(eligible, key=lambda evaluation: _ranking_key(evaluation, goals_by_action_index, scores))


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
