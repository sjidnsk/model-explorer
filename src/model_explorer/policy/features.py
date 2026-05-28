from __future__ import annotations

from dataclasses import dataclass
from math import hypot, isfinite, log1p

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

EXPERIMENTAL_CANDIDATE_FIELDS = _BENEFIT_FIELDS + _COST_FIELDS
MISSING_INDICATOR_NAMES = tuple(f"{field}_missing" for field in EXPERIMENTAL_CANDIDATE_FIELDS)

_GRID_SIZE_NORMALIZER = 1000.0


@dataclass(frozen=True)
class PolicyObservation:
    candidate_feature_names: tuple[str, ...]
    candidate_features: tuple[tuple[float, ...], ...]
    global_feature_names: tuple[str, ...]
    global_features: tuple[float, ...]
    action_mask: tuple[bool, ...]
    candidate_cells: tuple[tuple[int, int] | None, ...]
    candidate_missing_feature_names: tuple[tuple[str, ...], ...] = ()
    candidate_missing_indicator_names: tuple[str, ...] = MISSING_INDICATOR_NAMES
    candidate_missing_indicators: tuple[tuple[float, ...], ...] = ()


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
    missing_feature_names = tuple(_candidate_missing_feature_names(goal) for goal in goals)
    missing_indicators = tuple(_candidate_missing_indicators(goal) for goal in goals)
    action_mask = tuple(goal.reachable for goal in goals)
    candidate_cells: tuple[tuple[int, int] | None, ...] = tuple(goal.cell for goal in goals)

    if max_candidates is not None and len(goals) < max_candidates:
        padding_count = max_candidates - len(goals)
        candidate_features += tuple(_zero_candidate_features() for _ in range(padding_count))
        missing_feature_names += tuple(() for _ in range(padding_count))
        missing_indicators += tuple(_zero_missing_indicators() for _ in range(padding_count))
        action_mask += tuple(False for _ in range(padding_count))
        candidate_cells += tuple(None for _ in range(padding_count))

    return PolicyObservation(
        candidate_feature_names=CANDIDATE_FEATURE_NAMES,
        candidate_features=candidate_features,
        global_feature_names=GLOBAL_FEATURE_NAMES,
        global_features=_global_features(contract, step_index=step_index, remaining_steps=remaining_steps),
        action_mask=action_mask,
        candidate_cells=candidate_cells,
        candidate_missing_feature_names=missing_feature_names,
        candidate_missing_indicator_names=MISSING_INDICATOR_NAMES,
        candidate_missing_indicators=missing_indicators,
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
    grid_width = _positive_float(contract.grid.width, fallback=1.0)
    grid_height = _positive_float(contract.grid.height, fallback=1.0)
    diagonal = hypot(grid_width, grid_height) or 1.0
    grid_area = max(grid_width * grid_height, 1.0)

    values = {
        "cell_x": _clip_unit(x / grid_width),
        "cell_y": _clip_unit(y / grid_height),
        "relative_dx": _clip_signed_unit(dx / grid_width),
        "relative_dy": _clip_signed_unit(dy / grid_height),
        "relative_distance": _clip_unit(hypot(dx, dy) / diagonal),
        "utility": _clip_unit(goal.utility),
        "reachable": 1.0 if goal.reachable else 0.0,
    }
    for field in _BENEFIT_FIELDS:
        raw_value = _numeric_experimental(goal, field) or 0.0
        if field == "expected_new_coverage_area":
            values[field] = _clip_unit(raw_value / grid_area)
        else:
            values[field] = _clip_unit(raw_value)
    for field in _COST_FIELDS:
        raw_value = _numeric_experimental(goal, field)
        if raw_value is None:
            raw_value = cost_defaults[field]
        if field == "risk":
            values[field] = _clip_unit(raw_value)
        else:
            values[field] = _normalize_positive(raw_value, cost_defaults[field])

    return tuple(float(values[name]) for name in CANDIDATE_FEATURE_NAMES)


def _global_features(contract: ModelExplorerContract, *, step_index: int, remaining_steps: int) -> tuple[float, ...]:
    grid_width = _positive_float(contract.grid.width, fallback=1.0)
    grid_height = _positive_float(contract.grid.height, fallback=1.0)
    grid_area = max(grid_width * grid_height, 1.0)
    horizon = max(float(step_index + remaining_steps), 1.0)
    values = {
        "grid_width": _log_scale(grid_width, normalizer=_GRID_SIZE_NORMALIZER),
        "grid_height": _log_scale(grid_height, normalizer=_GRID_SIZE_NORMALIZER),
        "grid_resolution": _clip_unit(contract.grid.resolution),
        "passable_ratio": _clip_unit(contract.constraints.passable_ratio),
        "violation_count": _log_scale(contract.constraints.violation_count, normalizer=grid_area),
        "coverage_rate": _clip_unit(_numeric_mapping_value(contract.observation_update, "coverage_rate") or 0.0),
        "step_index": _clip_unit(float(step_index) / horizon),
        "remaining_steps": _clip_unit(float(remaining_steps) / horizon),
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


def _zero_missing_indicators() -> tuple[float, ...]:
    return tuple(0.0 for _ in MISSING_INDICATOR_NAMES)


def _candidate_missing_feature_names(goal: GoalCandidate) -> tuple[str, ...]:
    return tuple(
        field
        for field in EXPERIMENTAL_CANDIDATE_FIELDS
        if _numeric_experimental(goal, field) is None
    )


def _candidate_missing_indicators(goal: GoalCandidate) -> tuple[float, ...]:
    missing = set(_candidate_missing_feature_names(goal))
    return tuple(1.0 if field in missing else 0.0 for field in EXPERIMENTAL_CANDIDATE_FIELDS)


def _numeric_experimental(goal: GoalCandidate, field: str) -> float | None:
    return _numeric_mapping_value(goal.experimental, field)


def _numeric_mapping_value(mapping: dict, field: str) -> float | None:
    value = mapping.get(field)
    if isinstance(value, bool) or value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if isfinite(numeric) else None


def _clip_unit(value: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not isfinite(numeric):
        return 0.0
    return min(max(numeric, 0.0), 1.0)


def _clip_signed_unit(value: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not isfinite(numeric):
        return 0.0
    return min(max(numeric, -1.0), 1.0)


def _normalize_positive(value: float, scale: float) -> float:
    numeric = _positive_float(value, fallback=0.0)
    denominator = _positive_float(scale, fallback=0.0)
    if denominator <= 0.0:
        return 0.0
    return _clip_unit(numeric / denominator)


def _positive_float(value: float, *, fallback: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return fallback
    if not isfinite(numeric):
        return fallback
    return max(numeric, 0.0)


def _log_scale(value: float, *, normalizer: float) -> float:
    numeric = _positive_float(value, fallback=0.0)
    scale = max(_positive_float(normalizer, fallback=1.0), 1.0)
    return _clip_unit(log1p(numeric) / log1p(scale))
