from __future__ import annotations

from dataclasses import dataclass
from math import hypot

from ..core.interfaces import GoalCandidate, ModelExplorerContract


CANDIDATE_FEATURE_NAMES = (
    "cell_x",
    "cell_y",
    "relative_dx",
    "relative_dy",
    "relative_distance",
    "utility",
    "reachable",
    "expected_coverage_rate_delta",
    "expected_new_coverage_area",
    "information_gain",
    "confidence_gain",
    "value",
    "risk",
    "path_cost",
    "energy_cost",
)

GLOBAL_FEATURE_NAMES = (
    "grid_width",
    "grid_height",
    "grid_resolution",
    "passable_ratio",
    "violation_count",
    "coverage_rate",
    "step_index",
    "remaining_steps",
)

_BENEFIT_FIELDS = (
    "expected_coverage_rate_delta",
    "expected_new_coverage_area",
    "information_gain",
    "confidence_gain",
    "value",
)

_COST_FIELDS = ("risk", "path_cost", "energy_cost")


@dataclass(frozen=True)
class PolicyObservation:
    candidate_feature_names: tuple[str, ...]
    candidate_features: tuple[tuple[float, ...], ...]
    global_feature_names: tuple[str, ...]
    global_features: tuple[float, ...]
    action_mask: tuple[bool, ...]
    candidate_cells: tuple[tuple[int, int] | None, ...]


def extract_policy_observation(
    contract: ModelExplorerContract,
    *,
    current_cell: tuple[int, int] = (0, 0),
    step_index: int = 0,
    remaining_steps: int = 0,
    max_candidates: int | None = None,
) -> PolicyObservation:
    goals = contract.top_goals if max_candidates is None else contract.top_goals[:max_candidates]
    cost_defaults = _cost_defaults(goals)

    candidate_features = tuple(
        _candidate_features(contract, goal, current_cell=current_cell, cost_defaults=cost_defaults) for goal in goals
    )
    action_mask = tuple(goal.reachable for goal in goals)
    candidate_cells: tuple[tuple[int, int] | None, ...] = tuple(goal.cell for goal in goals)

    if max_candidates is not None and len(goals) < max_candidates:
        padding_count = max_candidates - len(goals)
        candidate_features += tuple(_zero_candidate_features() for _ in range(padding_count))
        action_mask += tuple(False for _ in range(padding_count))
        candidate_cells += tuple(None for _ in range(padding_count))

    return PolicyObservation(
        candidate_feature_names=CANDIDATE_FEATURE_NAMES,
        candidate_features=candidate_features,
        global_feature_names=GLOBAL_FEATURE_NAMES,
        global_features=_global_features(contract, step_index=step_index, remaining_steps=remaining_steps),
        action_mask=action_mask,
        candidate_cells=candidate_cells,
    )


def _candidate_features(
    contract: ModelExplorerContract,
    goal: GoalCandidate,
    *,
    current_cell: tuple[int, int],
    cost_defaults: dict[str, float],
) -> tuple[float, ...]:
    x, y = goal.cell
    current_x, current_y = current_cell
    dx = x - current_x
    dy = y - current_y
    diagonal = hypot(contract.grid.width, contract.grid.height) or 1.0

    values = {
        "cell_x": x / contract.grid.width,
        "cell_y": y / contract.grid.height,
        "relative_dx": dx / contract.grid.width,
        "relative_dy": dy / contract.grid.height,
        "relative_distance": hypot(dx, dy) / diagonal,
        "utility": goal.utility,
        "reachable": 1.0 if goal.reachable else 0.0,
    }
    for field in _BENEFIT_FIELDS:
        values[field] = _numeric_experimental(goal, field) or 0.0
    for field in _COST_FIELDS:
        values[field] = _numeric_experimental(goal, field)
        if values[field] is None:
            values[field] = cost_defaults[field]

    return tuple(float(values[name]) for name in CANDIDATE_FEATURE_NAMES)


def _global_features(contract: ModelExplorerContract, *, step_index: int, remaining_steps: int) -> tuple[float, ...]:
    values = {
        "grid_width": float(contract.grid.width),
        "grid_height": float(contract.grid.height),
        "grid_resolution": float(contract.grid.resolution),
        "passable_ratio": float(contract.constraints.passable_ratio),
        "violation_count": float(contract.constraints.violation_count),
        "coverage_rate": _numeric_mapping_value(contract.observation_update, "coverage_rate") or 0.0,
        "step_index": float(step_index),
        "remaining_steps": float(remaining_steps),
    }
    return tuple(values[name] for name in GLOBAL_FEATURE_NAMES)


def _cost_defaults(goals: tuple[GoalCandidate, ...]) -> dict[str, float]:
    defaults: dict[str, float] = {}
    for field in _COST_FIELDS:
        values = tuple(value for goal in goals for value in [_numeric_experimental(goal, field)] if value is not None)
        defaults[field] = max(values) if values else 0.0
    return defaults


def _zero_candidate_features() -> tuple[float, ...]:
    return tuple(0.0 for _ in CANDIDATE_FEATURE_NAMES)


def _numeric_experimental(goal: GoalCandidate, field: str) -> float | None:
    return _numeric_mapping_value(goal.experimental, field)


def _numeric_mapping_value(mapping: dict, field: str) -> float | None:
    value = mapping.get(field)
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
