from __future__ import annotations

from typing import Any

from .feedback_selection_scoring import _dedupe, _finite_float_optional
from .feedback_selection_types import FeedbackAwareSelectionConfig
from .planning_types import PathCandidateEvaluation


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


__all__ = (
    "classify_channel_aware_feedback_evidence",
    "_channel_aware_evidence_by_action_index",
    "_channel_aware_score_adjustment",
    "_channel_aware_blocker_reason",
)
