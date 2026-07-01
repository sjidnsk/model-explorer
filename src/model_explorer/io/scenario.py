from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..contracts.parsing import cell_like, required, strict_bool, strict_float, strict_int, strict_string_list
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

    schema_version = required(payload, "schema_version", prefix)
    if schema_version != MODEL_EXPLORER_SCHEMA_VERSION:
        raise ContractValidationError(
            f"{prefix}.schema_version must be {MODEL_EXPLORER_SCHEMA_VERSION!r}, got {schema_version!r}"
        )

    grid = _parse_grid(required(payload, "grid", prefix), f"{prefix}.grid")
    constraints = _parse_constraints(required(payload, "constraints", prefix), f"{prefix}.constraints")
    top_goals = _parse_goals(required(payload, "top_goals", prefix), f"{prefix}.top_goals")
    top_sequences = _parse_sequences(required(payload, "top_sequences", prefix), f"{prefix}.top_sequences")
    observation_update = required(payload, "observation_update", prefix)
    if not isinstance(observation_update, dict):
        raise ContractValidationError(f"{prefix}.observation_update must be an object")

    return ModelExplorerContract(
        schema_version=str(schema_version),
        grid=grid,
        constraints=constraints,
        top_goals=top_goals,
        top_sequences=top_sequences,
        observation_update=dict(observation_update),
        stable_fields=strict_string_list(payload.get("stable_fields"), f"{prefix}.stable_fields"),
        experimental_fields=strict_string_list(payload.get("experimental_fields"), f"{prefix}.experimental_fields"),
    )


def _parse_grid(payload: Any, prefix: str) -> GridSummary:
    if not isinstance(payload, dict):
        raise ContractValidationError(f"{prefix} must be an object")

    origin = cell_like(required(payload, "origin", prefix), f"{prefix}.origin", item_type=float)
    layers = required(payload, "layers", prefix)
    if not isinstance(layers, list) or not all(isinstance(layer, str) for layer in layers):
        raise ContractValidationError(f"{prefix}.layers must be a list of strings")

    width = strict_int(required(payload, "width", prefix), f"{prefix}.width")
    height = strict_int(required(payload, "height", prefix), f"{prefix}.height")
    resolution = strict_float(required(payload, "resolution", prefix), f"{prefix}.resolution")
    if width <= 0 or height <= 0:
        raise ContractValidationError(f"{prefix}.width and {prefix}.height must be positive")
    if resolution <= 0.0:
        raise ContractValidationError(f"{prefix}.resolution must be positive")

    return GridSummary(
        width=width,
        height=height,
        resolution=resolution,
        frame_id=str(required(payload, "frame_id", prefix)),
        origin=(origin[0], origin[1]),
        layers=tuple(layers),
    )


def _parse_constraints(payload: Any, prefix: str) -> ConstraintSummary:
    if not isinstance(payload, dict):
        raise ContractValidationError(f"{prefix} must be an object")

    reason_counts = required(payload, "reason_counts", prefix)
    if not isinstance(reason_counts, dict):
        raise ContractValidationError(f"{prefix}.reason_counts must be an object")

    return ConstraintSummary(
        violation_count=strict_int(required(payload, "violation_count", prefix), f"{prefix}.violation_count"),
        passable_ratio=strict_float(required(payload, "passable_ratio", prefix), f"{prefix}.passable_ratio"),
        reason_counts={str(key): strict_int(value, f"{prefix}.reason_counts.{key}") for key, value in reason_counts.items()},
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
                cell=cell_like(required(item, "cell", item_prefix), f"{item_prefix}.cell"),
                utility=strict_float(required(item, "utility", item_prefix), f"{item_prefix}.utility"),
                reachable=strict_bool(required(item, "reachable", item_prefix), f"{item_prefix}.reachable"),
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
        raw_cells = required(item, "cells", item_prefix)
        if not isinstance(raw_cells, list):
            raise ContractValidationError(f"{item_prefix}.cells must be a list")
        cells = tuple(cell_like(cell, f"{item_prefix}.cells[{cell_index}]") for cell_index, cell in enumerate(raw_cells))
        sequences.append(
            GoalSequence(
                cells=cells,
                utility=strict_float(required(item, "utility", item_prefix), f"{item_prefix}.utility"),
                coverage_area=strict_float(required(item, "coverage_area", item_prefix), f"{item_prefix}.coverage_area"),
                experimental={key: value for key, value in item.items() if key not in stable_keys},
            )
        )
    return tuple(sequences)
