from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core.interfaces import (
    MODEL_EXPLORER_SCHEMA_VERSION,
    ContractValidationError,
    ConstraintSummary,
    GoalCandidate,
    GoalSequence,
    GridSummary,
    ModelExplorerContract,
)


@dataclass(frozen=True)
class Scenario:
    snapshots: tuple[ModelExplorerContract, ...]
    metadata: dict[str, Any] = field(default_factory=dict)


def load_scenario(path: str | Path) -> Scenario:
    scenario_path = Path(path)
    payload = json.loads(scenario_path.read_text(encoding="utf-8"))

    if not isinstance(payload, dict):
        raise ContractValidationError("scenario root must be a JSON object")

    has_scenario_wrapper = "snapshots" in payload
    if has_scenario_wrapper:
        raw_snapshots = payload["snapshots"]
        if not isinstance(raw_snapshots, list):
            raise ContractValidationError("snapshots must be a list")
    else:
        raw_snapshots = [payload]

    if not raw_snapshots:
        raise ContractValidationError("snapshots must contain at least one contract")

    metadata = payload.get("metadata", {}) if has_scenario_wrapper else {}
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, dict):
        raise ContractValidationError("metadata must be an object when present")

    return Scenario(
        tuple(_parse_contract(item, f"snapshots[{index}]") for index, item in enumerate(raw_snapshots)),
        metadata=dict(metadata),
    )


def _parse_contract(payload: Any, prefix: str) -> ModelExplorerContract:
    if not isinstance(payload, dict):
        raise ContractValidationError(f"{prefix} must be a JSON object")

    schema_version = _required(payload, "schema_version", prefix)
    if schema_version != MODEL_EXPLORER_SCHEMA_VERSION:
        raise ContractValidationError(
            f"{prefix}.schema_version must be {MODEL_EXPLORER_SCHEMA_VERSION!r}, got {schema_version!r}"
        )

    grid = _parse_grid(_required(payload, "grid", prefix), f"{prefix}.grid")
    constraints = _parse_constraints(_required(payload, "constraints", prefix), f"{prefix}.constraints")
    top_goals = _parse_goals(_required(payload, "top_goals", prefix), f"{prefix}.top_goals")
    top_sequences = _parse_sequences(_required(payload, "top_sequences", prefix), f"{prefix}.top_sequences")
    observation_update = _required(payload, "observation_update", prefix)
    if not isinstance(observation_update, dict):
        raise ContractValidationError(f"{prefix}.observation_update must be an object")

    return ModelExplorerContract(
        schema_version=str(schema_version),
        grid=grid,
        constraints=constraints,
        top_goals=top_goals,
        top_sequences=top_sequences,
        observation_update=dict(observation_update),
        stable_fields=tuple(payload.get("stable_fields", ())),
        experimental_fields=tuple(payload.get("experimental_fields", ())),
    )


def _parse_grid(payload: Any, prefix: str) -> GridSummary:
    if not isinstance(payload, dict):
        raise ContractValidationError(f"{prefix} must be an object")

    origin = _cell_like(_required(payload, "origin", prefix), f"{prefix}.origin", item_type=float)
    layers = _required(payload, "layers", prefix)
    if not isinstance(layers, list) or not all(isinstance(layer, str) for layer in layers):
        raise ContractValidationError(f"{prefix}.layers must be a list of strings")

    width = int(_required(payload, "width", prefix))
    height = int(_required(payload, "height", prefix))
    resolution = float(_required(payload, "resolution", prefix))
    if width <= 0 or height <= 0:
        raise ContractValidationError(f"{prefix}.width and {prefix}.height must be positive")
    if resolution <= 0.0:
        raise ContractValidationError(f"{prefix}.resolution must be positive")

    return GridSummary(
        width=width,
        height=height,
        resolution=resolution,
        frame_id=str(_required(payload, "frame_id", prefix)),
        origin=(origin[0], origin[1]),
        layers=tuple(layers),
    )


def _parse_constraints(payload: Any, prefix: str) -> ConstraintSummary:
    if not isinstance(payload, dict):
        raise ContractValidationError(f"{prefix} must be an object")

    reason_counts = _required(payload, "reason_counts", prefix)
    if not isinstance(reason_counts, dict):
        raise ContractValidationError(f"{prefix}.reason_counts must be an object")

    return ConstraintSummary(
        violation_count=int(_required(payload, "violation_count", prefix)),
        passable_ratio=float(_required(payload, "passable_ratio", prefix)),
        reason_counts={str(key): int(value) for key, value in reason_counts.items()},
    )


def _parse_goals(payload: Any, prefix: str) -> tuple[GoalCandidate, ...]:
    if not isinstance(payload, list):
        raise ContractValidationError(f"{prefix} must be a list")

    goals: list[GoalCandidate] = []
    stable_keys = {"cell", "utility", "reachable"}
    for index, item in enumerate(payload):
        item_prefix = f"{prefix}[{index}]"
        if not isinstance(item, dict):
            raise ContractValidationError(f"{item_prefix} must be an object")
        goals.append(
            GoalCandidate(
                cell=_cell_like(_required(item, "cell", item_prefix), f"{item_prefix}.cell"),
                utility=float(_required(item, "utility", item_prefix)),
                reachable=bool(_required(item, "reachable", item_prefix)),
                experimental={key: value for key, value in item.items() if key not in stable_keys},
            )
        )
    return tuple(goals)


def _parse_sequences(payload: Any, prefix: str) -> tuple[GoalSequence, ...]:
    if not isinstance(payload, list):
        raise ContractValidationError(f"{prefix} must be a list")

    sequences: list[GoalSequence] = []
    stable_keys = {"cells", "utility", "coverage_area"}
    for index, item in enumerate(payload):
        item_prefix = f"{prefix}[{index}]"
        if not isinstance(item, dict):
            raise ContractValidationError(f"{item_prefix} must be an object")
        raw_cells = _required(item, "cells", item_prefix)
        if not isinstance(raw_cells, list):
            raise ContractValidationError(f"{item_prefix}.cells must be a list")
        cells = tuple(_cell_like(cell, f"{item_prefix}.cells[{cell_index}]") for cell_index, cell in enumerate(raw_cells))
        sequences.append(
            GoalSequence(
                cells=cells,
                utility=float(_required(item, "utility", item_prefix)),
                coverage_area=float(_required(item, "coverage_area", item_prefix)),
                experimental={key: value for key, value in item.items() if key not in stable_keys},
            )
        )
    return tuple(sequences)


def _required(payload: dict[str, Any], key: str, prefix: str) -> Any:
    if key not in payload:
        raise ContractValidationError(f"{prefix}.{key} is required")
    return payload[key]


def _cell_like(value: Any, prefix: str, *, item_type: type = int) -> tuple:
    if not isinstance(value, list) or len(value) != 2:
        raise ContractValidationError(f"{prefix} must be a list with 2 values")
    try:
        return tuple(item_type(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise ContractValidationError(f"{prefix} contains invalid values") from exc
