"""Grid helpers for anchor projection diagnostics."""

from __future__ import annotations

from collections import deque
from math import ceil, hypot
from typing import Any

from .planning_utils import _manhattan_distance, _reconstruct_path


def _connected_component_labels(
    mask: tuple[tuple[bool, ...], ...],
) -> tuple[tuple[tuple[int | None, ...], ...], dict[int, int]]:
    height = len(mask)
    width = len(mask[0]) if height else 0
    labels: list[list[int | None]] = [[None for _ in range(width)] for _ in range(height)]
    component_sizes: dict[int, int] = {}
    next_component_id = 0
    for y, row in enumerate(mask):
        for x, passable in enumerate(row):
            if not passable or labels[y][x] is not None:
                continue
            component_id = next_component_id
            next_component_id += 1
            frontier: deque[tuple[int, int]] = deque([(x, y)])
            labels[y][x] = component_id
            size = 0
            while frontier:
                current = frontier.popleft()
                size += 1
                for neighbor in _mask_neighbors(mask, current):
                    nx, ny = neighbor
                    if labels[ny][nx] is not None:
                        continue
                    labels[ny][nx] = component_id
                    frontier.append(neighbor)
            component_sizes[component_id] = size
    return tuple(tuple(row) for row in labels), component_sizes


def _best_reachable_anchor_in_component(
    mask: tuple[tuple[bool, ...], ...],
    *,
    target_cell: tuple[int, int],
    start: tuple[int, int] | None,
    start_component_id: int | None,
    component_labels: tuple[tuple[int | None, ...], ...],
) -> tuple[tuple[int, int] | None, int]:
    if start is None or start_component_id is None:
        return None, 0
    distances = _grid_distance_map(mask, start=start)
    candidates: list[tuple[int, float, int, int, int]] = []
    for y, row in enumerate(mask):
        for x, passable in enumerate(row):
            if not passable or _component_id_at(component_labels, (x, y)) != start_component_id:
                continue
            start_distance = distances.get((x, y))
            if start_distance is None:
                continue
            manhattan = abs(x - target_cell[0]) + abs(y - target_cell[1])
            euclidean = hypot(x - target_cell[0], y - target_cell[1])
            candidates.append((manhattan, euclidean, start_distance, y, x))
    if not candidates:
        return None, 0
    _, _, _, y, x = min(candidates)
    return (x, y), len(candidates)


def _grid_distance_map(
    mask: tuple[tuple[bool, ...], ...],
    *,
    start: tuple[int, int],
) -> dict[tuple[int, int], int]:
    if _mask_value(mask, start) is not True:
        return {}
    frontier: deque[tuple[int, int]] = deque([start])
    distances: dict[tuple[int, int], int] = {start: 0}
    while frontier:
        current = frontier.popleft()
        for neighbor in _mask_neighbors(mask, current):
            if neighbor in distances:
                continue
            distances[neighbor] = distances[current] + 1
            frontier.append(neighbor)
    return distances


def _component_id_at(
    labels: tuple[tuple[int | None, ...], ...],
    cell: tuple[int, int] | None,
) -> int | None:
    if cell is None:
        return None
    x, y = cell
    if y < 0 or y >= len(labels):
        return None
    if x < 0 or x >= len(labels[y]):
        return None
    return labels[y][x]


def _component_size(component_sizes: dict[int, int], component_id: int | None) -> int | None:
    if component_id is None:
        return None
    return component_sizes.get(component_id)


def _cell_list(cell: tuple[int, int] | None) -> list[int] | None:
    if cell is None:
        return None
    return [cell[0], cell[1]]


def _cell_manhattan_or_none(
    origin: tuple[int, int],
    cell: tuple[int, int] | None,
) -> int | None:
    if cell is None:
        return None
    return _manhattan_distance(origin, cell)


def _cell_distance_m_or_none(
    origin: tuple[int, int],
    cell: tuple[int, int] | None,
    resolution: float,
) -> float | None:
    if cell is None:
        return None
    return float(hypot(cell[0] - origin[0], cell[1] - origin[1]) * resolution)


def _inflated_passable_mask(
    mask: tuple[tuple[bool, ...], ...],
    *,
    resolution: float,
    footprint_radius_m: float | None,
) -> tuple[tuple[bool, ...], ...]:
    if footprint_radius_m is None or footprint_radius_m <= 0.0:
        return tuple(tuple(row) for row in mask)
    height = len(mask)
    width = len(mask[0]) if height else 0
    safe = [[bool(value) for value in row] for row in mask]
    radius_cells = int(ceil(footprint_radius_m / max(resolution, 1.0e-12)))
    for blocked_y, row in enumerate(mask):
        for blocked_x, passable in enumerate(row):
            if passable:
                continue
            min_y = max(0, blocked_y - radius_cells)
            max_y = min(height - 1, blocked_y + radius_cells)
            min_x = max(0, blocked_x - radius_cells)
            max_x = min(width - 1, blocked_x + radius_cells)
            for y in range(min_y, max_y + 1):
                for x in range(min_x, max_x + 1):
                    distance_m = hypot((x - blocked_x) * resolution, (y - blocked_y) * resolution)
                    if distance_m <= footprint_radius_m:
                        safe[y][x] = False
    return tuple(tuple(row) for row in safe)


def _nearest_inflated_passable_anchor(
    mask: tuple[tuple[bool, ...], ...],
    cell: tuple[int, int],
) -> tuple[int, int] | None:
    candidates: list[tuple[int, float, int, int]] = []
    for y, row in enumerate(mask):
        for x, passable in enumerate(row):
            if passable:
                manhattan = abs(x - cell[0]) + abs(y - cell[1])
                euclidean = hypot(x - cell[0], y - cell[1])
                candidates.append((manhattan, euclidean, y, x))
    if not candidates:
        return None
    _, _, y, x = min(candidates)
    return (x, y)


def _grid_path(
    mask: tuple[tuple[bool, ...], ...],
    *,
    start: tuple[int, int],
    goal: tuple[int, int],
) -> tuple[tuple[int, int], ...] | None:
    if _mask_value(mask, start) is not True or _mask_value(mask, goal) is not True:
        return None
    frontier: deque[tuple[int, int]] = deque([start])
    came_from: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    while frontier:
        current = frontier.popleft()
        if current == goal:
            return _reconstruct_path(came_from, current)
        for neighbor in _mask_neighbors(mask, current):
            if neighbor in came_from:
                continue
            came_from[neighbor] = current
            frontier.append(neighbor)
    return None


def _mask_neighbors(
    mask: tuple[tuple[bool, ...], ...],
    cell: tuple[int, int],
) -> tuple[tuple[int, int], ...]:
    x, y = cell
    candidates = ((x + 1, y), (x, y + 1), (x - 1, y), (x, y - 1))
    return tuple(candidate for candidate in candidates if _mask_value(mask, candidate) is True)


def _path_cost_for_payload(cost_payload: Any, path: tuple[tuple[int, int], ...]) -> float | None:
    if len(path) < 2:
        return 0.0
    total = 0.0
    for _, cell in zip(path[:-1], path[1:]):
        value = _grid_value(cost_payload, cell)
        if value is None:
            return None
        total += float(value)
    return float(total)


def _bool_grid(value: Any, *, width: int | None, height: int | None) -> tuple[tuple[bool, ...], ...] | None:
    if width is None or height is None or not isinstance(value, list):
        return None
    rows: list[tuple[bool, ...]] = []
    if len(value) != height:
        return None
    for row in value:
        if not isinstance(row, list) or len(row) != width:
            return None
        rows.append(tuple(bool(item) for item in row))
    return tuple(rows)


def _mask_value(mask: tuple[tuple[bool, ...], ...], cell: tuple[int, int]) -> bool | None:
    x, y = cell
    if y < 0 or y >= len(mask):
        return None
    if x < 0 or x >= len(mask[y]):
        return None
    return bool(mask[y][x])


def _grid_value(value: Any, cell: tuple[int, int]) -> float | None:
    x, y = cell
    if not isinstance(value, list) or y < 0 or y >= len(value):
        return None
    row = value[y]
    if not isinstance(row, list) or x < 0 or x >= len(row):
        return None
    try:
        return float(row[x])
    except (TypeError, ValueError):
        return None


__all__ = (
    "_connected_component_labels",
    "_best_reachable_anchor_in_component",
    "_grid_distance_map",
    "_component_id_at",
    "_component_size",
    "_cell_list",
    "_cell_manhattan_or_none",
    "_cell_distance_m_or_none",
    "_inflated_passable_mask",
    "_nearest_inflated_passable_anchor",
    "_grid_path",
    "_mask_neighbors",
    "_path_cost_for_payload",
    "_bool_grid",
    "_mask_value",
    "_grid_value",
)
