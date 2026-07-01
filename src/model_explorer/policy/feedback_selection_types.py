from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..core.interfaces import ExplorerDecision
from .planning_types import PathCandidateEvaluation


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


__all__ = (
    "FeedbackAwareSelectionConfig",
    "FeedbackAwareSelection",
)
