"""Planning adapter implementations."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from heapq import heappop, heappush
from pathlib import Path
from typing import Any

from .planning_routes import (
    _ExistingDirectory,
    _default_path_planner_root,
    _path_planner_env,
    _read_json,
    _tail,
    build_path_planner_request_dict,
    load_path_planner_sidecar,
    path_plan_result_from_route_dict,
)
from .planning_types import PathPlanningAdapter, PathPlanRequest, PathPlanResult
from .planning_utils import (
    _grid_distance,
    _manhattan_distance,
    _metadata_or_config,
    _numeric_experimental,
    _reconstruct_path,
)


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


class PathPlannerRouteAdapter:
    """Run path-planner through its JSON/CLI boundary and map route feedback."""

    def __init__(
        self,
        *,
        path_planner_root: str | Path | None = None,
        python_executable: str | None = None,
        output_dir: str | Path | None = None,
        platform: str | None = None,
        platform_config: str | Path | None = None,
        cost: Sequence[Sequence[float]] | None = None,
        passable_mask: Sequence[Sequence[bool]] | None = None,
        sidecar: dict[str, Any] | None = None,
        extra_args: Sequence[str] = (),
        route_json: str | Path | None = None,
    ) -> None:
        sidecar_payload = dict(sidecar or {})
        self._path_planner_root = _default_path_planner_root(path_planner_root)
        self._python_executable = python_executable or sys.executable
        self._output_dir = None if output_dir is None else Path(output_dir)
        self._platform = platform
        self._platform_config = None if platform_config is None else Path(platform_config)
        self._cost = cost if cost is not None else sidecar_payload.get("cost")
        self._passable_mask = passable_mask if passable_mask is not None else sidecar_payload.get("passable_mask")
        self._sidecar_metadata = sidecar_payload.get("metadata", {})
        self._extra_args = tuple(str(arg) for arg in extra_args)
        self._route_json = None if route_json is None else Path(route_json)

    def plan(self, request: PathPlanRequest) -> PathPlanResult:
        plan_payload = build_path_planner_request_dict(
            request,
            cost=_metadata_or_config(request, "path_planner_cost", self._cost),
            passable_mask=_metadata_or_config(request, "path_planner_passable_mask", self._passable_mask),
            sidecar_metadata=self._sidecar_metadata,
        )
        if self._route_json is not None:
            route_payload = _read_json(self._route_json)
            result = path_plan_result_from_route_dict(route_payload, request=request)
            return _with_adapter_metadata(
                result,
                request_payload=plan_payload,
                route_json=str(self._route_json),
                mode="route_json",
            )

        try:
            route_payload, run_metadata = self._run_cli(plan_payload)
        except Exception as exc:
            return PathPlanResult(
                feasible=False,
                risk=_numeric_experimental(request.selected_goal, "risk"),
                failure_reason="path_planner_adapter_failed",
                replan_required=True,
                metadata={
                    "planner": "path_planner_route",
                    "mode": "cli",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                    "request": plan_payload,
                },
            )
        result = path_plan_result_from_route_dict(route_payload, request=request)
        return _with_adapter_metadata(result, request_payload=plan_payload, route_payload=route_payload, **run_metadata)

    def _run_cli(self, plan_payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        output_context = (
            tempfile.TemporaryDirectory(prefix="model-explorer-path-planner-")
            if self._output_dir is None
            else _ExistingDirectory(self._output_dir)
        )
        with output_context as output_dir_raw:
            output_dir = Path(output_dir_raw)
            output_dir.mkdir(parents=True, exist_ok=True)
            input_path = output_dir / "path-planner-request.json"
            route_path = output_dir / "path-planner-route.json"
            diagnostics_dir = output_dir / "diagnostics"
            input_path.write_text(json.dumps(plan_payload, ensure_ascii=False, indent=2), encoding="utf-8")
            command = [
                self._python_executable,
                "-m",
                "path_planner.cli",
                "--input",
                str(input_path),
                "--output-json",
                str(route_path),
                "--output-dir",
                str(diagnostics_dir),
            ]
            if self._platform is not None:
                command.extend(["--platform", self._platform])
            if self._platform_config is not None:
                command.extend(["--platform-config", str(self._platform_config)])
            command.extend(self._extra_args)
            completed = subprocess.run(
                command,
                cwd=self._path_planner_root,
                env=_path_planner_env(self._path_planner_root),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    "path-planner CLI failed "
                    f"with return code {completed.returncode}: {_tail(completed.stderr or completed.stdout)}"
                )
            route_payload = _read_json(route_path)
            return route_payload, {
                "planner": "path_planner_route",
                "mode": "cli",
                "command": command,
                "stdout_tail": _tail(completed.stdout),
                "stderr_tail": _tail(completed.stderr),
                "output_dir": str(output_dir),
            }


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
    if backend == "path_planner_route":
        sidecar = payload.get("sidecar")
        sidecar_path = payload.get("path_planner_sidecar")
        if sidecar is None and sidecar_path is not None:
            sidecar = load_path_planner_sidecar(sidecar_path)
        return PathPlannerRouteAdapter(
            path_planner_root=payload.get("path_planner_root"),
            python_executable=payload.get("python_executable"),
            output_dir=payload.get("output_dir"),
            platform=payload.get("platform"),
            platform_config=payload.get("platform_config"),
            cost=payload.get("cost"),
            passable_mask=payload.get("passable_mask"),
            sidecar=sidecar,
            extra_args=payload.get("extra_args", ()),
            route_json=payload.get("route_json"),
        )
    raise ValueError(f"unknown planner backend: {backend}")


def _blocked_result(reason: str) -> PathPlanResult:
    return PathPlanResult(feasible=False, failure_reason=reason, replan_required=True)


def _with_adapter_metadata(result: PathPlanResult, **metadata: Any) -> PathPlanResult:
    merged = dict(result.metadata)
    merged.update(metadata)
    return PathPlanResult(
        feasible=result.feasible,
        path_cost=result.path_cost,
        path_length=result.path_length,
        risk=result.risk,
        failure_reason=result.failure_reason,
        replan_required=result.replan_required,
        metadata=merged,
    )


__all__ = [name for name in globals() if not name.startswith("__")]
