from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from heapq import heappop, heappush
from math import hypot
from pathlib import Path
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
class PathCandidateEvaluation:
    action_index: int
    cell: tuple[int, int]
    utility: float
    result: PathPlanResult

    def to_dict(self) -> dict[str, Any]:
        input_sources = _input_source_summary(self.result.metadata.get("request_payload"))
        payload = {
            "action_index": self.action_index,
            "cell": [self.cell[0], self.cell[1]],
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
        }
        return payload


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


def build_path_planner_request_dict(
    request: PathPlanRequest,
    *,
    cost: Sequence[Sequence[float]] | None = None,
    passable_mask: Sequence[Sequence[bool]] | None = None,
    sidecar_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    grid = request.contract.grid
    cost_rows, cost_source = _grid_rows(
        cost,
        width=grid.width,
        height=grid.height,
        default_value=1.0,
        field_name="cost",
    )
    mask_rows, mask_source = _grid_rows(
        passable_mask,
        width=grid.width,
        height=grid.height,
        default_value=True,
        field_name="passable_mask",
    )
    return {
        "schema_version": "path-planner-request/v1",
        "grid": {
            "width": grid.width,
            "height": grid.height,
            "resolution": grid.resolution,
            "origin": [grid.origin[0], grid.origin[1]],
            "frame_id": grid.frame_id,
        },
        "cost": cost_rows,
        "passable_mask": mask_rows,
        "start": [request.current_cell[0], request.current_cell[1]],
        "goal": [request.selected_goal.cell[0], request.selected_goal.cell[1]],
        "max_iterations": int(request.metadata.get("path_planner_max_iterations", 100_000)),
        "metadata": {
            "source": "model-explorer",
            "step_index": request.step_index,
            "action_index": request.action_index,
            "selected_goal_utility": request.selected_goal.utility,
            "cost_source": cost_source,
            "passable_mask_source": mask_source,
            "sidecar": dict(sidecar_metadata or {}),
        },
    }


def load_path_planner_sidecar(path: str | Path) -> dict[str, Any]:
    payload = _read_json(path)
    if payload.get("schema_version") != "path-planner-sidecar/v1":
        raise ValueError("path planner sidecar schema_version must be path-planner-sidecar/v1")
    if "cost" not in payload or "passable_mask" not in payload:
        raise ValueError("path planner sidecar must contain cost and passable_mask")
    return payload


def path_plan_result_from_route_dict(
    route: dict[str, Any],
    *,
    request: PathPlanRequest | None = None,
) -> PathPlanResult:
    if route.get("schema_version") != "path-planner-route/v1":
        raise ValueError("route schema_version must be path-planner-route/v1")
    feasible = bool(route.get("reachable"))
    path_cost = _finite_number(route.get("path_cost"), default=0.0)
    diagnostics = route.get("diagnostics") if isinstance(route.get("diagnostics"), dict) else {}
    path_length = _finite_number(diagnostics.get("path_length_m"), default=None)
    if path_length is None:
        path_length = _path_length_from_route(route)
    route_failure = route.get("failure_reason")
    failure_reason = None if feasible else str(route_failure or "path_planner_unreachable")
    replan_required = not feasible or _route_has_safety_or_fallback_issue(route)
    risk = 0.0 if request is None else _numeric_experimental(request.selected_goal, "risk")
    return PathPlanResult(
        feasible=feasible,
        path_cost=path_cost,
        path_length=path_length,
        risk=risk,
        failure_reason=failure_reason,
        replan_required=replan_required,
        metadata={
            "planner": "path_planner_route",
            "route_schema_version": route.get("schema_version"),
            "diagnostics": diagnostics,
            "postprocess": route.get("postprocess"),
            "tracking_simulation_report": route.get("tracking_simulation_report"),
            "trajectory_optimization_report": route.get("trajectory_optimization_report"),
            "planning_backend_report": route.get("planning_backend_report"),
            "region_graph_report": route.get("region_graph_report"),
            "iris_region_report": route.get("iris_region_report"),
            "convex_region_report": _convex_region_route_report(route),
            "gcs_trajectory_report": _gcs_trajectory_route_report(route),
            "gcs_candidate_report": _gcs_candidate_route_report(route),
            "gcs_motion_feasibility_report": _gcs_motion_feasibility_route_report(route),
            "gcs_curvature_constrained_candidate_report": _gcs_curvature_constrained_candidate_route_report(route),
        },
    )


def evaluate_candidate_paths(
    contract: ModelExplorerContract,
    *,
    current_cell: tuple[int, int],
    top_k: int,
    planner: PathPlanningAdapter,
    step_index: int = 0,
) -> tuple[PathCandidateEvaluation, ...]:
    evaluations: list[PathCandidateEvaluation] = []
    for action_index, goal in enumerate(contract.top_goals):
        if len(evaluations) >= top_k:
            break
        if not goal.reachable:
            continue
        result = planner.plan(
            PathPlanRequest(
                contract=contract,
                step_index=step_index,
                action_index=action_index,
                selected_goal=goal,
                current_cell=current_cell,
            )
        )
        evaluations.append(
            PathCandidateEvaluation(
                action_index=action_index,
                cell=goal.cell,
                utility=goal.utility,
                result=result,
            )
        )
    return tuple(evaluations)


def path_feedback_summary(evaluations: Sequence[PathCandidateEvaluation]) -> dict[str, Any]:
    items = []
    for evaluation in evaluations:
        item = evaluation.to_dict()
        item["diagnostic_interpretation"] = _candidate_diagnostic_interpretation(item)
        items.append(item)
    failure_reasons = [
        item["failure_reason"] for item in items if item.get("failure_reason") is not None
    ]
    replan_count = sum(1 for item in items if item["replan_required"])
    feasible_items = [item for item in items if item["reachable"]]
    return {
        "candidate_count": len(items),
        "reachable_count": len(feasible_items),
        "failure_count": len(items) - len(feasible_items),
        "replan_count": replan_count,
        "failure_reasons": failure_reasons,
        "best_by_path_cost": min(
            feasible_items,
            key=lambda item: (float(item["path_cost"]), -float(item["utility"])),
            default=None,
        ),
        "candidates": items,
    }


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


def _metadata_or_config(
    request: PathPlanRequest,
    metadata_key: str,
    config_value: Sequence[Sequence[Any]] | None,
) -> Sequence[Sequence[Any]] | None:
    value = request.metadata.get(metadata_key)
    return value if value is not None else config_value


def _grid_rows(
    values: Sequence[Sequence[Any]] | None,
    *,
    width: int,
    height: int,
    default_value: Any,
    field_name: str,
) -> tuple[list[list[Any]], str]:
    if values is None:
        return [[default_value for _ in range(width)] for _ in range(height)], "open_grid_fallback"
    rows = [list(row) for row in values]
    if len(rows) != height or any(len(row) != width for row in rows):
        raise ValueError(f"{field_name} must have shape height={height}, width={width}")
    if field_name == "passable_mask":
        return [[bool(value) for value in row] for row in rows], "configured"
    return [[float(value) for value in row] for row in rows], "configured"


def _finite_number(value: Any, *, default: float | None) -> float | None:
    if isinstance(value, bool) or value is None:
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number != number or number in {float("inf"), float("-inf")}:
        return default
    return number


def _path_length_from_route(route: dict[str, Any]) -> float:
    geometric_path = route.get("geometric_path")
    if not isinstance(geometric_path, dict):
        return 0.0
    world_points = geometric_path.get("world")
    if isinstance(world_points, list) and len(world_points) >= 2:
        total = 0.0
        previous: tuple[float, float] | None = None
        for item in world_points:
            if not isinstance(item, list | tuple) or len(item) < 2:
                return 0.0
            current = (float(item[0]), float(item[1]))
            if previous is not None:
                total += hypot(current[0] - previous[0], current[1] - previous[1])
            previous = current
        return total
    cells = geometric_path.get("cells")
    if isinstance(cells, list):
        return float(max(len(cells) - 1, 0))
    return 0.0


def _route_has_safety_or_fallback_issue(route: dict[str, Any]) -> bool:
    postprocess = route.get("postprocess")
    if isinstance(postprocess, dict):
        fallback = postprocess.get("fallback_status")
        if isinstance(fallback, str) and fallback not in {"ok", "not_needed", "none"}:
            return True
        tracking_safety = postprocess.get("tracking_safety_report")
        if isinstance(tracking_safety, dict) and int(tracking_safety.get("violation_count", 0) or 0) > 0:
            return True
    optimization = route.get("trajectory_optimization_report")
    if isinstance(optimization, dict):
        fallback = optimization.get("fallback_status")
        if isinstance(fallback, str) and fallback not in {"ok", "not_needed", "none"}:
            return True
    region_graph = route.get("region_graph_report")
    if isinstance(region_graph, dict):
        quality = region_graph.get("quality_metrics")
        if isinstance(quality, dict) and quality.get("start_goal_connected") is False:
            return True
    return False


def _postprocess_summary(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    tracking_safety = value.get("tracking_safety_report")
    return {
        "fallback_status": value.get("fallback_status"),
        "has_smoothed_path": bool(value.get("smoothed_path")),
        "tracking_safety_violation_count": (
            None
            if not isinstance(tracking_safety, dict)
            else int(tracking_safety.get("violation_count", 0) or 0)
        ),
    }


def _optimization_summary(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    return {
        "solver_status": value.get("solver_status"),
        "fallback_status": value.get("fallback_status"),
        "has_optimized_path": bool(value.get("optimized_path")),
    }


def _planning_backend_summary(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    sampled = value.get("sampled_region_path_report")
    return {
        "requested_backend": value.get("requested_backend"),
        "selected_backend": value.get("selected_backend"),
        "status": value.get("status"),
        "fallback_reason": value.get("fallback_reason"),
        "segment_count": value.get("segment_count"),
        "comparison": value.get("comparison") if isinstance(value.get("comparison"), dict) else {},
        "region_graph_candidate": (
            value.get("region_graph_candidate") if isinstance(value.get("region_graph_candidate"), dict) else {}
        ),
        "sampled_region_path": sampled if isinstance(sampled, dict) else {},
    }


def _region_graph_summary(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    quality = value.get("quality_metrics")
    quality = quality if isinstance(quality, dict) else {}
    quality_summary = {
        "requested_region_source": quality.get("requested_region_source"),
        "graph_source": quality.get("graph_source", value.get("region_source")),
        "fallback_ratio": quality.get("fallback_ratio"),
        "connected_component_count": quality.get("connected_component_count"),
        "fallback_reason": quality.get("fallback_reason") or value.get("failure_reason"),
        "start_goal_connected": quality.get("start_goal_connected"),
    }
    return {
        "status": value.get("status"),
        "region_source": value.get("region_source"),
        "vertex_count": value.get("vertex_count"),
        "edge_count": value.get("edge_count"),
        **quality_summary,
        "quality_metrics": quality_summary,
        "fallback_used": value.get("fallback_used"),
    }


def _iris_region_summary(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    return {
        "backend": value.get("backend"),
        "status": value.get("status"),
        "region_count": value.get("region_count"),
        "fallback_used": value.get("fallback_used"),
        "failure_status": value.get("failure_status"),
        "failure_reason": value.get("failure_reason"),
    }


def _convex_region_route_report(route: dict[str, Any]) -> dict[str, Any] | None:
    if "convex_region_count" not in route and "gcs_ready" not in route:
        return None
    return {
        "schema_version": route.get("convex_region_sequence_schema_version"),
        "region_count": route.get("convex_region_count"),
        "backend": route.get("convex_region_backend"),
        "fallback_used": route.get("convex_region_fallback_used"),
        "coverage_status": route.get("convex_region_coverage_status"),
        "start_contained": route.get("convex_region_start_contained"),
        "goal_contained": route.get("convex_region_goal_contained"),
        "adjacent_overlap_count": route.get("convex_region_adjacent_overlap_count"),
        "portal_count": route.get("convex_region_portal_count"),
        "blocked_cell_violation_count": route.get("convex_region_blocked_cell_violation_count"),
        "pydrake_available": route.get("convex_region_pydrake_available"),
        "gcs_ready": route.get("gcs_ready"),
        "gcs_ready_reason": route.get("gcs_ready_reason"),
        "sequence": route.get("convex_region_sequence") if isinstance(route.get("convex_region_sequence"), list) else [],
    }


def _convex_region_summary(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    return {
        "schema_version": value.get("schema_version"),
        "region_count": value.get("region_count"),
        "backend": value.get("backend"),
        "fallback_used": value.get("fallback_used"),
        "coverage_status": value.get("coverage_status"),
        "start_contained": value.get("start_contained"),
        "goal_contained": value.get("goal_contained"),
        "adjacent_overlap_count": value.get("adjacent_overlap_count"),
        "portal_count": value.get("portal_count"),
        "blocked_cell_violation_count": value.get("blocked_cell_violation_count"),
        "pydrake_available": value.get("pydrake_available"),
        "gcs_ready": value.get("gcs_ready"),
        "gcs_ready_reason": value.get("gcs_ready_reason"),
    }


def _gcs_trajectory_route_report(route: dict[str, Any]) -> dict[str, Any] | None:
    if "gcs_trajectory_report_schema_version" not in route:
        return None
    return {
        "schema_version": route.get("gcs_trajectory_report_schema_version"),
        "attempted": route.get("gcs_trajectory_attempted"),
        "success": route.get("gcs_trajectory_success"),
        "backend": route.get("gcs_trajectory_backend"),
        "result_status": route.get("gcs_trajectory_result_status"),
        "reason": route.get("gcs_trajectory_reason"),
        "sample_count": route.get("gcs_trajectory_sample_count"),
        "collision_count": route.get("gcs_trajectory_collision_count"),
        "path_length": route.get("gcs_trajectory_path_length"),
        "region_count": route.get("gcs_trajectory_region_count"),
        "sampled_points": (
            route.get("gcs_trajectory_sampled_points")
            if isinstance(route.get("gcs_trajectory_sampled_points"), list)
            else []
        ),
        "cost_summary": (
            route.get("gcs_trajectory_cost_summary")
            if isinstance(route.get("gcs_trajectory_cost_summary"), dict)
            else {}
        ),
    }


def _gcs_trajectory_summary(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    return {
        "schema_version": value.get("schema_version"),
        "attempted": value.get("attempted"),
        "success": value.get("success"),
        "backend": value.get("backend"),
        "result_status": value.get("result_status"),
        "reason": value.get("reason"),
        "sample_count": value.get("sample_count"),
        "collision_count": value.get("collision_count"),
        "path_length": value.get("path_length"),
        "region_count": value.get("region_count"),
        "cost_summary": value.get("cost_summary") if isinstance(value.get("cost_summary"), dict) else {},
    }


def _gcs_candidate_route_report(route: dict[str, Any]) -> dict[str, Any] | None:
    if "gcs_candidate_report_schema_version" not in route:
        return None
    return {
        "schema_version": route.get("gcs_candidate_report_schema_version"),
        "attempted": route.get("gcs_candidate_attempted"),
        "available": route.get("gcs_candidate_available"),
        "selected": route.get("gcs_candidate_selected"),
        "selection_reason": route.get("gcs_candidate_selection_reason"),
        "fallback_reason": route.get("gcs_candidate_fallback_reason"),
        "path_length": route.get("gcs_candidate_path_length"),
        "path_cost": route.get("gcs_candidate_path_cost"),
        "collision_count": route.get("gcs_candidate_collision_count"),
        "high_cost_exposure": route.get("gcs_candidate_high_cost_exposure"),
        "baseline_overlap_ratio": route.get("gcs_candidate_baseline_overlap_ratio"),
        "cost_delta_vs_baseline": route.get("gcs_candidate_cost_delta_vs_baseline"),
        "cost_delta_vs_postprocess": route.get("gcs_candidate_cost_delta_vs_postprocess"),
        "cost_summary": (
            route.get("gcs_candidate_cost_summary")
            if isinstance(route.get("gcs_candidate_cost_summary"), dict)
            else {}
        ),
    }


def _gcs_candidate_summary(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    return {
        "schema_version": value.get("schema_version"),
        "attempted": value.get("attempted"),
        "available": value.get("available"),
        "selected": value.get("selected"),
        "selection_reason": value.get("selection_reason"),
        "fallback_reason": value.get("fallback_reason"),
        "path_length": value.get("path_length"),
        "path_cost": value.get("path_cost"),
        "collision_count": value.get("collision_count"),
        "high_cost_exposure": value.get("high_cost_exposure"),
        "baseline_overlap_ratio": value.get("baseline_overlap_ratio"),
        "cost_delta_vs_baseline": value.get("cost_delta_vs_baseline"),
        "cost_delta_vs_postprocess": value.get("cost_delta_vs_postprocess"),
        "cost_summary": value.get("cost_summary") if isinstance(value.get("cost_summary"), dict) else {},
    }


def _gcs_motion_feasibility_route_report(route: dict[str, Any]) -> dict[str, Any] | None:
    if "gcs_motion_feasibility_report_schema_version" not in route:
        return None
    return {
        "schema_version": route.get("gcs_motion_feasibility_report_schema_version"),
        "evaluated": route.get("gcs_motion_feasibility_evaluated"),
        "trajectory_source": route.get("gcs_motion_feasibility_trajectory_source"),
        "motion_model": route.get("gcs_motion_feasibility_motion_model"),
        "feasibility_status": route.get("gcs_motion_feasibility_feasibility_status"),
        "fallback_reason": route.get("gcs_motion_feasibility_fallback_reason"),
        "min_turning_radius_m": route.get("gcs_motion_feasibility_min_turning_radius_m"),
        "max_heading_change_deg": route.get("gcs_motion_feasibility_max_heading_change_deg"),
        "curvature_violation_count": route.get("gcs_motion_feasibility_curvature_violation_count"),
        "heading_violation_count": route.get("gcs_motion_feasibility_heading_violation_count"),
        "violation_indices": (
            route.get("gcs_motion_feasibility_violation_indices")
            if isinstance(route.get("gcs_motion_feasibility_violation_indices"), list)
            else []
        ),
        "sample_count": route.get("gcs_motion_feasibility_sample_count"),
        "path_length": route.get("gcs_motion_feasibility_path_length"),
        "constraint_summary": (
            route.get("gcs_motion_feasibility_constraint_summary")
            if isinstance(route.get("gcs_motion_feasibility_constraint_summary"), dict)
            else {}
        ),
    }


def _gcs_motion_feasibility_summary(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    return {
        "schema_version": value.get("schema_version"),
        "evaluated": value.get("evaluated"),
        "trajectory_source": value.get("trajectory_source"),
        "motion_model": value.get("motion_model"),
        "feasibility_status": value.get("feasibility_status"),
        "fallback_reason": value.get("fallback_reason"),
        "min_turning_radius_m": value.get("min_turning_radius_m"),
        "max_heading_change_deg": value.get("max_heading_change_deg"),
        "curvature_violation_count": value.get("curvature_violation_count"),
        "heading_violation_count": value.get("heading_violation_count"),
        "violation_indices": value.get("violation_indices") if isinstance(value.get("violation_indices"), list) else [],
        "sample_count": value.get("sample_count"),
        "path_length": value.get("path_length"),
        "constraint_summary": value.get("constraint_summary") if isinstance(value.get("constraint_summary"), dict) else {},
    }


def _gcs_curvature_constrained_candidate_route_report(route: dict[str, Any]) -> dict[str, Any] | None:
    if "gcs_curvature_constrained_report_schema_version" not in route:
        return None
    return {
        "schema_version": route.get("gcs_curvature_constrained_report_schema_version"),
        "attempted": route.get("gcs_curvature_constrained_attempted"),
        "available": route.get("gcs_curvature_constrained_available"),
        "selected": route.get("gcs_curvature_constrained_selected"),
        "repair_success": route.get("gcs_curvature_constrained_repair_success"),
        "source": route.get("gcs_curvature_constrained_source"),
        "repair_strategy": route.get("gcs_curvature_constrained_repair_strategy"),
        "status_before": route.get("gcs_curvature_constrained_status_before"),
        "status_after": route.get("gcs_curvature_constrained_status_after"),
        "fallback_reason": route.get("gcs_curvature_constrained_fallback_reason"),
        "curvature_violation_count_before": route.get(
            "gcs_curvature_constrained_curvature_violation_count_before"
        ),
        "curvature_violation_count_after": route.get(
            "gcs_curvature_constrained_curvature_violation_count_after"
        ),
        "heading_violation_count_before": route.get(
            "gcs_curvature_constrained_heading_violation_count_before"
        ),
        "heading_violation_count_after": route.get(
            "gcs_curvature_constrained_heading_violation_count_after"
        ),
        "violation_indices_before": (
            route.get("gcs_curvature_constrained_violation_indices_before")
            if isinstance(route.get("gcs_curvature_constrained_violation_indices_before"), list)
            else []
        ),
        "violation_indices_after": (
            route.get("gcs_curvature_constrained_violation_indices_after")
            if isinstance(route.get("gcs_curvature_constrained_violation_indices_after"), list)
            else []
        ),
        "region_containment_violation_count": route.get(
            "gcs_curvature_constrained_region_containment_violation_count"
        ),
        "collision_count": route.get("gcs_curvature_constrained_collision_count"),
        "path_length": route.get("gcs_curvature_constrained_path_length"),
        "path_cost": route.get("gcs_curvature_constrained_path_cost"),
        "cost_delta_vs_baseline": route.get("gcs_curvature_constrained_cost_delta_vs_baseline"),
        "constraint_summary": (
            route.get("gcs_curvature_constrained_constraint_summary")
            if isinstance(route.get("gcs_curvature_constrained_constraint_summary"), dict)
            else {}
        ),
    }


def _gcs_curvature_constrained_candidate_summary(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    return {
        "schema_version": value.get("schema_version"),
        "attempted": value.get("attempted"),
        "available": value.get("available"),
        "selected": value.get("selected"),
        "repair_success": value.get("repair_success"),
        "source": value.get("source"),
        "repair_strategy": value.get("repair_strategy"),
        "status_before": value.get("status_before"),
        "status_after": value.get("status_after"),
        "fallback_reason": value.get("fallback_reason"),
        "curvature_violation_count_before": value.get("curvature_violation_count_before"),
        "curvature_violation_count_after": value.get("curvature_violation_count_after"),
        "heading_violation_count_before": value.get("heading_violation_count_before"),
        "heading_violation_count_after": value.get("heading_violation_count_after"),
        "violation_indices_before": (
            value.get("violation_indices_before")
            if isinstance(value.get("violation_indices_before"), list)
            else []
        ),
        "violation_indices_after": (
            value.get("violation_indices_after")
            if isinstance(value.get("violation_indices_after"), list)
            else []
        ),
        "region_containment_violation_count": value.get("region_containment_violation_count"),
        "collision_count": value.get("collision_count"),
        "path_length": value.get("path_length"),
        "path_cost": value.get("path_cost"),
        "cost_delta_vs_baseline": value.get("cost_delta_vs_baseline"),
        "constraint_summary": value.get("constraint_summary") if isinstance(value.get("constraint_summary"), dict) else {},
    }


def _input_source_summary(value: Any) -> dict[str, Any]:
    metadata = value.get("metadata") if isinstance(value, dict) else {}
    metadata = metadata if isinstance(metadata, dict) else {}
    cost_source = metadata.get("cost_source")
    passable_mask_source = metadata.get("passable_mask_source")
    return {
        "cost_source": cost_source,
        "passable_mask_source": passable_mask_source,
        "open_grid_fallback_used": (
            cost_source == "open_grid_fallback"
            or passable_mask_source == "open_grid_fallback"
        ),
    }


def _candidate_diagnostic_interpretation(item: dict[str, Any]) -> dict[str, Any]:
    flags: list[str] = []
    if item.get("failure_reason") is not None or not bool(item.get("reachable")):
        flags.append("path_planning_failure")
    if bool(item.get("replan_required")):
        flags.append("replan_required")
    if bool(item.get("open_grid_fallback_used")):
        flags.append("open_grid_fallback")

    optimization = item.get("trajectory_optimization")
    if isinstance(optimization, dict):
        fallback = optimization.get("fallback_status")
        if isinstance(fallback, str) and fallback not in {"ok", "not_needed", "none"}:
            flags.append("trajectory_optimization_fallback")

    postprocess = item.get("postprocess")
    if isinstance(postprocess, dict) and int(postprocess.get("tracking_safety_violation_count") or 0) > 0:
        flags.append("tracking_safety_violation")

    iris = item.get("iris_region")
    iris_status = None
    iris_fallback_used = False
    if isinstance(iris, dict):
        iris_status = iris.get("status")
        iris_fallback_used = bool(iris.get("fallback_used"))
        if iris_fallback_used:
            flags.append("iris_fallback")
        if iris_status == "failed":
            flags.append("iris_failure")

    graph = item.get("region_graph")
    graph_source = None
    graph_fallback_used = False
    graph_connected = None
    if isinstance(graph, dict):
        graph_source = graph.get("graph_source") or graph.get("region_source")
        graph_fallback_used = bool(graph.get("fallback_used"))
        graph_connected = graph.get("start_goal_connected")
        if graph_fallback_used:
            flags.append("region_graph_fallback")
        if graph_connected is False:
            flags.append("region_graph_disconnected")

    sampled_status = None
    sampled_fallback_reason = None
    planning_backend = item.get("planning_backend")
    if isinstance(planning_backend, dict):
        sampled = planning_backend.get("sampled_region_path")
        if isinstance(sampled, dict) and sampled:
            sampled_status = sampled.get("status")
            sampled_fallback_reason = sampled.get("fallback_reason")
            if sampled_status == "fallback" or sampled_fallback_reason:
                flags.append("sampled_region_path_fallback")

    ordered_flags = _dedupe(flags)
    return {
        "primary_source": _primary_diagnostic_source(ordered_flags),
        "diagnostic_flags": ordered_flags,
        "iris_status": iris_status,
        "iris_fallback_used": iris_fallback_used,
        "region_graph_source": graph_source,
        "region_graph_fallback_used": graph_fallback_used,
        "region_graph_start_goal_connected": graph_connected,
        "sampled_region_path_status": sampled_status,
        "sampled_region_path_fallback_reason": sampled_fallback_reason,
        "open_grid_fallback_used": bool(item.get("open_grid_fallback_used")),
    }


def _primary_diagnostic_source(flags: Sequence[str]) -> str:
    priority = (
        "path_planning_failure",
        "region_graph_disconnected",
        "region_graph_fallback",
        "iris_failure",
        "iris_fallback",
        "sampled_region_path_fallback",
        "tracking_safety_violation",
        "trajectory_optimization_fallback",
        "open_grid_fallback",
        "replan_required",
    )
    for flag in priority:
        if flag in flags:
            return flag
    return "none"


def _dedupe(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _report_present(value: Any) -> bool:
    return isinstance(value, dict)


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


def _read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _default_path_planner_root(path_planner_root: str | Path | None) -> Path:
    if path_planner_root is not None:
        return Path(path_planner_root)
    return Path(__file__).resolve().parents[4] / "path-planner"


def _path_planner_env(path_planner_root: Path) -> dict[str, str]:
    env = dict(os.environ)
    src_path = str(path_planner_root / "src")
    current = env.get("PYTHONPATH")
    env["PYTHONPATH"] = src_path if not current else f"{src_path}{os.pathsep}{current}"
    return env


def _tail(text: str, *, max_chars: int = 2000) -> str:
    return text[-max_chars:]


class _ExistingDirectory:
    def __init__(self, path: Path) -> None:
        self._path = path

    def __enter__(self) -> str:
        self._path.mkdir(parents=True, exist_ok=True)
        return str(self._path)

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        return None
