"""Planning data types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from ..core.interfaces import GoalCandidate, ModelExplorerContract


@dataclass(frozen=True)
class PathPlanRequest:
    contract: ModelExplorerContract
    step_index: int
    action_index: int
    selected_goal: GoalCandidate
    current_cell: tuple[int, int] = (0, 0)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def selected_cell(self) -> tuple[int, int]:
        return self.selected_goal.cell

    @property
    def selected_world(self) -> tuple[float, float]:
        return self.contract.cell_to_world(self.selected_goal.cell)


@dataclass(frozen=True)
class PathPlanResult:
    feasible: bool
    path_cost: float = 0.0
    path_length: float = 0.0
    risk: float = 0.0
    failure_reason: str | None = None
    replan_required: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AnchorProjectionCandidateConfig:
    enabled: bool = False
    max_projection_distance_cells: int | None = None
    max_projection_distance_m: float | None = None
    require_anchor_reachable: bool = True
    source_selection_path_cost_bonus: float = 0.0
    max_source_selection_path_cost_regression: float | None = None
    max_source_selection_risk_regression: float | None = None
    contract_aware_trainable_target_generation: bool = False
    prefer_contract_safe_trainable_targets: bool = False
    max_trainable_projection_distance_cells: int = 2
    max_trainable_projection_distance_m: float = 1.0
    planner_validated_trainable_target_mining: bool = False
    allow_planner_validated_distance_exception: bool = False
    max_planner_validated_distance_cells: int = 3
    max_planner_validated_distance_m: float = 1.5


@dataclass(frozen=True)
class PathCandidateEvaluation:
    action_index: int
    cell: tuple[int, int]
    utility: float
    result: PathPlanResult
    selection_goal: GoalCandidate | None = None
    source_action_index: int | None = None
    candidate_generation: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        from .planning_backend_summaries import (
            _convex_region_summary,
            _gcs_candidate_summary,
            _gcs_curvature_constrained_candidate_summary,
            _gcs_motion_feasibility_summary,
            _gcs_trajectory_summary,
            _iris_region_summary,
            _optimization_summary,
            _planning_backend_summary,
            _postprocess_summary,
            _region_graph_summary,
        )
        from .planning_diagnostic_interpretation import _input_source_summary, _report_present
        from .planning_platform_feasibility import (
            _platform_goal_feasibility,
            _with_projected_anchor_feasibility,
        )
        from .planning_utils import _cell_pair

        input_sources = _input_source_summary(self.result.metadata.get("request_payload"))
        generation = dict(self.candidate_generation or {})
        policy_target_cell = _cell_pair(generation.get("policy_target_cell")) or self.cell
        execution_goal_cell = _cell_pair(generation.get("execution_goal_cell")) or self.cell
        platform_goal_feasibility = _platform_goal_feasibility(
            cell=policy_target_cell,
            result=self.result,
        )
        if generation:
            platform_goal_feasibility = _with_projected_anchor_feasibility(
                platform_goal_feasibility,
                candidate_generation=generation,
            )
        payload = {
            "action_index": self.action_index,
            "source_action_index": self.source_action_index,
            "cell": [self.cell[0], self.cell[1]],
            "candidate_role": str(generation.get("candidate_role", "policy_target")),
            "policy_target_cell": [policy_target_cell[0], policy_target_cell[1]],
            "execution_goal_cell": [execution_goal_cell[0], execution_goal_cell[1]],
            "utility": float(self.utility),
            "reachable": bool(self.result.feasible),
            "path_cost": float(self.result.path_cost),
            "path_length": float(self.result.path_length),
            "risk": float(self.result.risk),
            "failure_reason": self.result.failure_reason,
            "replan_required": bool(self.result.replan_required),
            "diagnostics": self.result.metadata.get("diagnostics"),
            "postprocess": _postprocess_summary(self.result.metadata.get("postprocess")),
            "tracking_simulation": _report_present(self.result.metadata.get("tracking_simulation_report")),
            "trajectory_optimization": _optimization_summary(
                self.result.metadata.get("trajectory_optimization_report")
            ),
            "planning_backend": _planning_backend_summary(self.result.metadata.get("planning_backend_report")),
            "region_graph": _region_graph_summary(self.result.metadata.get("region_graph_report")),
            "iris_region": _iris_region_summary(self.result.metadata.get("iris_region_report")),
            "convex_region": _convex_region_summary(self.result.metadata.get("convex_region_report")),
            "gcs_trajectory": _gcs_trajectory_summary(self.result.metadata.get("gcs_trajectory_report")),
            "gcs_candidate": _gcs_candidate_summary(self.result.metadata.get("gcs_candidate_report")),
            "gcs_motion_feasibility": _gcs_motion_feasibility_summary(
                self.result.metadata.get("gcs_motion_feasibility_report")
            ),
            "gcs_curvature_constrained_candidate": _gcs_curvature_constrained_candidate_summary(
                self.result.metadata.get("gcs_curvature_constrained_candidate_report")
            ),
            "input_sources": input_sources,
            "open_grid_fallback_used": bool(input_sources["open_grid_fallback_used"]),
            "platform_goal_feasibility": platform_goal_feasibility,
        }
        if generation:
            payload["candidate_generation"] = generation
        return payload


class PathPlanningAdapter(Protocol):
    def plan(self, request: PathPlanRequest) -> PathPlanResult:
        ...


__all__ = (
    "GoalCandidate",
    "ModelExplorerContract",
    "PathPlanRequest",
    "PathPlanResult",
    "AnchorProjectionCandidateConfig",
    "PathCandidateEvaluation",
    "PathPlanningAdapter",
)
