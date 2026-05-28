from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from heapq import heappop, heappush
from math import hypot
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


class PathPlanningAdapter(Protocol):
    def plan(self, request: PathPlanRequest) -> PathPlanResult:
        ...


class ContractCostPlanner:
    def plan(self, request: PathPlanRequest) -> PathPlanResult:
        return PathPlanResult(
            feasible=bool(request.selected_goal.reachable),
            path_cost=_numeric_experimental(request.selected_goal, "path_cost"),
            path_length=_numeric_experimental(request.selected_goal, "path_length"),
            risk=_numeric_experimental(request.selected_goal, "risk"),
            failure_reason=None if request.selected_goal.reachable else "unreachable_goal",
            replan_required=not request.selected_goal.reachable,
            metadata={"planner": "contract_cost"},
        )


class StraightLineProxyPlanner:
    def __init__(self, *, cost_per_cell: float = 1.0) -> None:
        self._cost_per_cell = float(cost_per_cell)

    def plan(self, request: PathPlanRequest) -> PathPlanResult:
        distance = _grid_distance(request.current_cell, request.selected_goal.cell)
        return PathPlanResult(
            feasible=bool(request.selected_goal.reachable),
            path_cost=distance * self._cost_per_cell,
            path_length=distance,
            risk=_numeric_experimental(request.selected_goal, "risk"),
            failure_reason=None if request.selected_goal.reachable else "unreachable_goal",
            replan_required=not request.selected_goal.reachable,
            metadata={"planner": "straight_line"},
        )


class GridAStarPlanner:
    def __init__(self, *, passable_grid: Sequence[Sequence[bool]], cost_per_step: float = 1.0) -> None:
        self._passable_grid = tuple(tuple(bool(value) for value in row) for row in passable_grid)
        self._cost_per_step = float(cost_per_step)
        if not self._passable_grid or not self._passable_grid[0]:
            raise ValueError("passable_grid must contain at least one row and one column")
        width = len(self._passable_grid[0])
        if any(len(row) != width for row in self._passable_grid):
            raise ValueError("passable_grid rows must have equal length")

    def plan(self, request: PathPlanRequest) -> PathPlanResult:
        start = request.current_cell
        goal = request.selected_goal.cell
        if not request.selected_goal.reachable:
            return _blocked_result("unreachable_goal")
        if not self._is_passable(start) or not self._is_passable(goal):
            return _blocked_result("path_blocked")

        path = self._find_path(start, goal)
        if path is None:
            return _blocked_result("path_blocked")

        path_length = float(max(len(path) - 1, 0))
        return PathPlanResult(
            feasible=True,
            path_cost=path_length * self._cost_per_step,
            path_length=path_length,
            risk=_numeric_experimental(request.selected_goal, "risk"),
            metadata={
                "planner": "grid_astar",
                "path_cells": [[cell[0], cell[1]] for cell in path],
            },
        )

    def _find_path(self, start: tuple[int, int], goal: tuple[int, int]) -> tuple[tuple[int, int], ...] | None:
        frontier: list[tuple[float, int, tuple[int, int]]] = []
        heappush(frontier, (_grid_distance(start, goal), 0, start))
        came_from: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
        cost_so_far: dict[tuple[int, int], int] = {start: 0}

        while frontier:
            _, current_cost, current = heappop(frontier)
            if current == goal:
                return _reconstruct_path(came_from, current)

            for neighbor in self._neighbors(current):
                new_cost = current_cost + 1
                if neighbor in cost_so_far and new_cost >= cost_so_far[neighbor]:
                    continue
                cost_so_far[neighbor] = new_cost
                came_from[neighbor] = current
                priority = new_cost + _manhattan_distance(neighbor, goal)
                heappush(frontier, (float(priority), new_cost, neighbor))

        return None

    def _neighbors(self, cell: tuple[int, int]) -> tuple[tuple[int, int], ...]:
        x, y = cell
        candidates = ((x + 1, y), (x, y + 1), (x - 1, y), (x, y - 1))
        return tuple(candidate for candidate in candidates if self._is_passable(candidate))

    def _is_passable(self, cell: tuple[int, int]) -> bool:
        x, y = cell
        return 0 <= y < len(self._passable_grid) and 0 <= x < len(self._passable_grid[y]) and self._passable_grid[y][x]


class FutureGcsPlannerAdapter:
    def plan(self, request: PathPlanRequest) -> PathPlanResult:
        return PathPlanResult(
            feasible=False,
            failure_reason="gcs_adapter_unavailable",
            replan_required=True,
            metadata={"planner": "future_gcs", "enabled": False},
        )


def planner_from_config(config: dict[str, Any] | None) -> PathPlanningAdapter:
    payload = dict(config or {})
    backend = str(payload.get("backend", "contract_cost"))
    if backend == "contract_cost":
        return ContractCostPlanner()
    if backend == "straight_line":
        return StraightLineProxyPlanner(cost_per_cell=float(payload.get("cost_per_cell", 1.0)))
    if backend == "grid_astar":
        return GridAStarPlanner(
            passable_grid=payload["passable_grid"],
            cost_per_step=float(payload.get("cost_per_step", 1.0)),
        )
    if backend == "future_gcs":
        return FutureGcsPlannerAdapter()
    raise ValueError(f"unknown planner backend: {backend}")


def _numeric_experimental(goal: GoalCandidate, field: str) -> float:
    value = goal.experimental.get(field)
    if isinstance(value, bool) or value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _grid_distance(first: tuple[int, int], second: tuple[int, int]) -> float:
    return float(hypot(second[0] - first[0], second[1] - first[1]))


def _manhattan_distance(first: tuple[int, int], second: tuple[int, int]) -> int:
    return abs(second[0] - first[0]) + abs(second[1] - first[1])


def _blocked_result(reason: str) -> PathPlanResult:
    return PathPlanResult(feasible=False, failure_reason=reason, replan_required=True)


def _reconstruct_path(
    came_from: dict[tuple[int, int], tuple[int, int] | None],
    current: tuple[int, int],
) -> tuple[tuple[int, int], ...]:
    path = [current]
    while came_from[current] is not None:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return tuple(path)
