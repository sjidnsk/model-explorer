from __future__ import annotations

from typing import Any

from ..core.interfaces import GoalCandidate, ModelExplorerContract
from .feedback_selection_anchor import _candidate_generation_for_selection
from .path_feedback_manifest import cell_tuple as _cell_tuple
from .planning_anchor_projection import anchor_projection_candidate_config_from_mapping
from .planning_types import AnchorProjectionCandidateConfig, PathCandidateEvaluation


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
    from .feedback_selection_trainability import _contract_aware_preferred_selection

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
    from .feedback_selection_anchor import _anchor_projection_adjusted_path_cost

    return (
        _anchor_projection_adjusted_path_cost(evaluation, config=config),
        float(evaluation.result.risk),
        -float(evaluation.utility),
        evaluation.cell[0],
        evaluation.cell[1],
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


selected_after_feedback = _selected_after_feedback

__all__ = (
    "selected_after_feedback",
    "_selected_before_feedback",
    "_selected_after_feedback",
    "_source_selection_key",
    "_selection_payload",
    "_best_source_selection_alternative",
    "_source_selection_quality_regression",
    "_candidate_float",
    "_list_cell",
    "_path_cost_for_selection",
    "_candidate_path_cost_for_cell",
    "_path_cost_delta",
)
