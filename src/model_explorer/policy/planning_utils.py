"""Shared planning helper functions."""

from __future__ import annotations

from collections.abc import Sequence
from math import hypot
from typing import Any

from ..core.interfaces import GoalCandidate


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

def _cell_pair(value: Any) -> tuple[int, int] | None:
    if not isinstance(value, list | tuple) or len(value) != 2:
        return None
    try:
        return (int(value[0]), int(value[1]))
    except (TypeError, ValueError):
        return None


def _positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _positive_float(value: Any, *, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0.0 else default


def _optional_positive_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0.0 else None


def _optional_nonnegative_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _optional_nonnegative_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0.0 else None


__all__ = (
    "GoalCandidate",
    "_numeric_experimental",
    "_grid_distance",
    "_manhattan_distance",
    "_reconstruct_path",
    "_metadata_or_config",
    "_grid_rows",
    "_finite_number",
    "_path_length_from_route",
    "_route_has_safety_or_fallback_issue",
    "_cell_pair",
    "_positive_int",
    "_positive_float",
    "_optional_positive_float",
    "_optional_nonnegative_int",
    "_optional_nonnegative_float",
)
