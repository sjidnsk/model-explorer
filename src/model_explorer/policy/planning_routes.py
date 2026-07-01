"""Path-planner request, sidecar, route, and CLI boundary helpers."""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .planning_backend_summaries import (
    _convex_region_route_report,
    _gcs_candidate_route_report,
    _gcs_curvature_constrained_candidate_route_report,
    _gcs_motion_feasibility_route_report,
    _gcs_trajectory_route_report,
)
from .planning_types import PathPlanRequest, PathPlanResult
from .planning_utils import (
    _finite_number,
    _grid_rows,
    _numeric_experimental,
    _path_length_from_route,
    _route_has_safety_or_fallback_issue,
)


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



def planner_from_config(config: dict[str, Any] | None) -> Any:
    from .planning_adapters import planner_from_config as _planner_from_config

    return _planner_from_config(config)


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


__all__ = [name for name in globals() if not name.startswith("__")]
