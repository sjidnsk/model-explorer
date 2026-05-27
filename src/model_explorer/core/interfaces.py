from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


MODEL_EXPLORER_SCHEMA_VERSION = "model-explorer-contract/v1"


class ContractValidationError(ValueError):
    """Raised when a model-explorer contract is missing required stable fields."""


@dataclass(frozen=True)
class GridSummary:
    width: int
    height: int
    resolution: float
    frame_id: str
    origin: tuple[float, float]
    layers: tuple[str, ...]


@dataclass(frozen=True)
class ConstraintSummary:
    violation_count: int
    passable_ratio: float
    reason_counts: dict[str, int]


@dataclass(frozen=True)
class GoalCandidate:
    cell: tuple[int, int]
    utility: float
    reachable: bool
    experimental: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GoalSequence:
    cells: tuple[tuple[int, int], ...]
    utility: float
    coverage_area: float
    experimental: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelExplorerContract:
    schema_version: str
    grid: GridSummary
    constraints: ConstraintSummary
    top_goals: tuple[GoalCandidate, ...]
    top_sequences: tuple[GoalSequence, ...]
    observation_update: dict[str, Any]
    stable_fields: tuple[str, ...] = ()
    experimental_fields: tuple[str, ...] = ()

    def cell_to_world(self, cell: tuple[int, int]) -> tuple[float, float]:
        x, y = cell
        return (
            self.grid.origin[0] + x * self.grid.resolution,
            self.grid.origin[1] + y * self.grid.resolution,
        )


@dataclass(frozen=True)
class ExplorerDecision:
    status: str
    selected_goal: GoalCandidate | None
    ranked_goals: tuple[GoalCandidate, ...] = ()


@dataclass(frozen=True)
class ExplorerStepResult:
    step_index: int
    decision: ExplorerDecision
    observation_update: dict[str, Any]
    replan_reasons: tuple[str, ...]
