from __future__ import annotations

import random
import json
from dataclasses import dataclass
from math import hypot, isfinite
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..core.interfaces import (
    MODEL_EXPLORER_SCHEMA_VERSION,
    ConstraintSummary,
    GoalCandidate,
    GoalSequence,
    GridSummary,
    ModelExplorerContract,
)
from ..io.scenario import Scenario

if TYPE_CHECKING:
    from ..policy.rollout import RolloutEpisode


GENERATOR_VERSION = "lola-south-pole-rollout/v1"


@dataclass(frozen=True)
class LolaSouthPoleRoiConfig:
    roi_x: int
    roi_y: int
    roi_width: int
    roi_height: int
    candidate_count: int = 8
    episode_count: int = 1
    seed: int = 0
    start_cell: tuple[int, int] = (0, 0)


def generate_lola_south_pole_scenarios(
    dem_values,
    observation_count_values,
    *,
    dataset_id: str,
    data_class: str,
    region: str,
    resolution: float,
    config: LolaSouthPoleRoiConfig,
    source_config: LolaSouthPoleRoiConfig | None = None,
    metadata_extra: dict[str, Any] | None = None,
) -> tuple[Scenario, ...]:
    dem = _coerce_grid(dem_values, name="dem_values")
    counts = _coerce_grid(observation_count_values, name="observation_count_values")
    if len(dem) != len(counts) or len(dem[0]) != len(counts[0]):
        raise ValueError("dem_values and observation_count_values must have the same shape")
    if config.candidate_count <= 0:
        raise ValueError("candidate_count must be positive")
    if config.episode_count <= 0:
        raise ValueError("episode_count must be positive")

    scenarios: list[Scenario] = []
    provenance_config = config if source_config is None else source_config
    for episode_index in range(config.episode_count):
        seed = int(config.seed) + episode_index
        provenance = _provenance(
            dataset_id=dataset_id,
            data_class=data_class,
            region=region,
            resolution=resolution,
            config=provenance_config,
            seed=seed,
            extra=metadata_extra,
        )
        contract = _contract_from_roi(
            dem,
            counts,
            dataset_id=dataset_id,
            data_class=data_class,
            region=region,
            resolution=resolution,
            config=config,
            source_config=provenance_config,
            seed=seed,
        )
        scenarios.append(Scenario(snapshots=(contract,), metadata=provenance))
    return tuple(scenarios)


def generate_lola_south_pole_rollout_episodes(
    dem_values,
    observation_count_values,
    *,
    dataset_id: str,
    data_class: str,
    region: str,
    resolution: float,
    config: LolaSouthPoleRoiConfig,
    source_config: LolaSouthPoleRoiConfig | None = None,
    metadata_extra: dict[str, Any] | None = None,
) -> tuple[RolloutEpisode, ...]:
    from ..experiments.lola_rollouts import generate_lola_south_pole_rollout_episodes as _generate

    return _generate(
        dem_values,
        observation_count_values,
        dataset_id=dataset_id,
        data_class=data_class,
        region=region,
        resolution=resolution,
        config=config,
        source_config=source_config,
        metadata_extra=metadata_extra,
    )


def write_lola_south_pole_rollouts_jsonl(
    path: str | Path,
    dem_values,
    observation_count_values,
    *,
    dataset_id: str,
    data_class: str,
    region: str,
    resolution: float,
    config: LolaSouthPoleRoiConfig,
) -> tuple[RolloutEpisode, ...]:
    from ..experiments.lola_rollouts import write_lola_south_pole_rollouts_jsonl as _write

    return _write(
        path,
        dem_values,
        observation_count_values,
        dataset_id=dataset_id,
        data_class=data_class,
        region=region,
        resolution=resolution,
        config=config,
    )


def write_lola_south_pole_scenarios_json(
    output_dir: str | Path,
    dem_values,
    observation_count_values,
    *,
    dataset_id: str,
    data_class: str,
    region: str,
    resolution: float,
    config: LolaSouthPoleRoiConfig,
    source_config: LolaSouthPoleRoiConfig | None = None,
    metadata_extra: dict[str, Any] | None = None,
) -> tuple[Path, ...]:
    scenarios = generate_lola_south_pole_scenarios(
        dem_values,
        observation_count_values,
        dataset_id=dataset_id,
        data_class=data_class,
        region=region,
        resolution=resolution,
        config=config,
        source_config=source_config,
        metadata_extra=metadata_extra,
    )
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for index, scenario in enumerate(scenarios):
        path = output_root / f"lola-south-pole-{index:03d}.json"
        path.write_text(json.dumps(_scenario_to_dict(scenario), indent=2, ensure_ascii=False), encoding="utf-8")
        paths.append(path)
    return tuple(paths)


def write_lola_south_pole_rollouts_from_manifest_jsonl(
    manifest_path: str | Path,
    output_path: str | Path,
    *,
    config: LolaSouthPoleRoiConfig,
) -> tuple[RolloutEpisode, ...]:
    from ..experiments.lola_rollouts import write_lola_south_pole_rollouts_from_manifest_jsonl as _write

    return _write(manifest_path, output_path, config=config)


def _contract_from_roi(
    dem: tuple[tuple[float, ...], ...],
    counts: tuple[tuple[float, ...], ...],
    *,
    dataset_id: str,
    data_class: str,
    region: str,
    resolution: float,
    config: LolaSouthPoleRoiConfig,
    source_config: LolaSouthPoleRoiConfig,
    seed: int,
) -> ModelExplorerContract:
    dem_roi = _slice_roi(dem, config=config, name="dem_values")
    count_roi = _slice_roi(counts, config=config, name="observation_count_values")
    terrain = _terrain_candidates(dem_roi, count_roi, resolution=resolution, start_cell=config.start_cell)
    goals = _select_candidate_goals(terrain, candidate_count=config.candidate_count, seed=seed)
    reachable_count = sum(1 for candidate in terrain if candidate["reachable"])
    total_count = max(len(terrain), 1)
    selected_sequence = _sequence_from_goals(goals)
    observation_delta = _mean(tuple(goal.experimental["expected_coverage_rate_delta"] for goal in goals if goal.reachable))

    return ModelExplorerContract(
        schema_version=MODEL_EXPLORER_SCHEMA_VERSION,
        grid=GridSummary(
            width=config.roi_width,
            height=config.roi_height,
            resolution=float(resolution),
            frame_id="moon_south_pole_polar_stereographic",
            origin=(float(source_config.roi_x) * float(resolution), float(source_config.roi_y) * float(resolution)),
            layers=(
                "lola_dem",
                "lola_observation_count",
                "terrain_slope_proxy",
                "terrain_roughness_proxy",
            ),
        ),
        constraints=ConstraintSummary(
            violation_count=total_count - reachable_count,
            passable_ratio=reachable_count / total_count,
            reason_counts={
                "terrain_risk": sum(1 for candidate in terrain if candidate["risk"] >= 0.88),
                "low_observation_count": sum(1 for candidate in terrain if candidate["confidence"] <= 0.0),
            },
        ),
        top_goals=goals,
        top_sequences=selected_sequence,
        observation_update={
            "coverage_rate": _clip_unit(observation_delta),
            "coverage_rate_delta": _clip_unit(observation_delta),
            "value_coverage": _mean(tuple(goal.experimental["value"] for goal in goals if goal.reachable)),
            "data_class": data_class,
            "dataset_id": dataset_id,
            "region": region,
            "generator_version": GENERATOR_VERSION,
        },
        stable_fields=("schema_version", "grid", "constraints", "top_goals", "top_sequences", "observation_update"),
        experimental_fields=(
            "expected_coverage_rate_delta",
            "expected_new_coverage_area",
            "information_gain",
            "confidence_gain",
            "value",
            "risk",
            "path_cost",
            "energy_cost",
            "terrain_relative_elevation",
            "terrain_slope_proxy",
            "terrain_roughness_proxy",
            "observation_count",
            "data_confidence",
        ),
    )


def _terrain_candidates(
    dem: tuple[tuple[float, ...], ...],
    counts: tuple[tuple[float, ...], ...],
    *,
    resolution: float,
    start_cell: tuple[int, int],
) -> tuple[dict[str, Any], ...]:
    height = len(dem)
    width = len(dem[0])
    elevations = tuple(value for row in dem for value in row)
    min_elevation = min(elevations)
    max_elevation = max(elevations)
    elevation_span = max(max_elevation - min_elevation, 1.0)
    max_count = max((value for row in counts for value in row), default=0.0)
    slope_raw = {}
    roughness_raw = {}
    for y in range(height):
        for x in range(width):
            neighbors = _neighbor_values(dem, x=x, y=y)
            deltas = tuple(abs(dem[y][x] - value) for value in neighbors)
            slope_raw[(x, y)] = max(deltas) if deltas else 0.0
            roughness_raw[(x, y)] = _mean(deltas)
    max_slope = max(max(slope_raw.values(), default=0.0), 1.0)
    max_roughness = max(max(roughness_raw.values(), default=0.0), 1.0)

    candidates: list[dict[str, Any]] = []
    for y in range(height):
        for x in range(width):
            if (x, y) == start_cell:
                continue
            confidence = _clip_unit(counts[y][x] / max_count) if max_count > 0.0 else 0.0
            relative_elevation = _clip_unit((dem[y][x] - min_elevation) / elevation_span)
            slope = _clip_unit(slope_raw[(x, y)] / max_slope)
            roughness = _clip_unit(roughness_raw[(x, y)] / max_roughness)
            information_gain = _clip_unit(1.0 - confidence)
            risk = _clip_unit(0.45 * slope + 0.35 * roughness + 0.20 * (1.0 - confidence))
            distance = hypot(x - start_cell[0], y - start_cell[1])
            path_cost = distance * float(resolution) * (1.0 + risk)
            energy_cost = path_cost * (1.0 + slope)
            reachable = confidence > 0.0 and risk < 0.88
            value = _clip_unit(0.45 * information_gain + 0.35 * confidence + 0.20 * (1.0 - risk))
            coverage_delta = _clip_unit(0.02 + 0.05 * information_gain + 0.03 * (1.0 - risk))
            utility = _clip_unit(0.35 * value + 0.30 * coverage_delta + 0.20 * confidence + 0.15 * (1.0 - risk))
            candidates.append(
                {
                    "cell": (x, y),
                    "utility": utility,
                    "reachable": reachable,
                    "risk": risk,
                    "confidence": confidence,
                    "experimental": {
                        "expected_coverage_rate_delta": coverage_delta,
                        "expected_new_coverage_area": float(resolution) ** 2 * (1.0 + information_gain),
                        "information_gain": information_gain,
                        "confidence_gain": confidence,
                        "value": value,
                        "risk": risk,
                        "path_cost": path_cost,
                        "energy_cost": energy_cost,
                        "terrain_relative_elevation": relative_elevation,
                        "terrain_slope_proxy": slope,
                        "terrain_roughness_proxy": roughness,
                        "observation_count": counts[y][x],
                        "data_confidence": confidence,
                    },
                }
            )
    return tuple(candidates)


def _product_file_name(manifest: dict[str, Any], *, role: str) -> str:
    products = manifest.get("products", [])
    if not isinstance(products, list):
        raise ValueError("manifest.products must be a list")
    for product in products:
        if not isinstance(product, dict) or product.get("role") != role:
            continue
        files = product.get("files", [])
        if isinstance(files, list):
            for file_info in files:
                if isinstance(file_info, dict) and str(file_info.get("name", "")).lower().endswith(".jp2"):
                    return str(file_info["name"])
            for file_info in files:
                if isinstance(file_info, dict) and file_info.get("name"):
                    return str(file_info["name"])
    raise ValueError(f"manifest is missing product file for role {role!r}")


def _manifest_resolution(manifest: dict[str, Any]) -> float:
    projection = manifest.get("projection", {})
    if isinstance(projection, dict) and projection.get("map_scale_meters_per_pixel") is not None:
        return float(projection["map_scale_meters_per_pixel"])
    return 1.0


def _select_candidate_goals(
    terrain: tuple[dict[str, Any], ...],
    *,
    candidate_count: int,
    seed: int,
) -> tuple[GoalCandidate, ...]:
    rng = random.Random(seed)
    ranked = sorted(
        terrain,
        key=lambda item: (-float(item["utility"]) - rng.random() * 1.0e-9, item["cell"][0], item["cell"][1]),
    )
    selected = list(ranked[:candidate_count])
    if selected and not any(not item["reachable"] for item in selected):
        unreachable = [item for item in ranked if not item["reachable"]]
        if unreachable and candidate_count > 1:
            selected[-1] = unreachable[0]
    if selected and not any(item["reachable"] for item in selected):
        reachable = [item for item in ranked if item["reachable"]]
        if reachable:
            selected[0] = reachable[0]
    selected = selected[:candidate_count]
    return tuple(
        GoalCandidate(
            cell=item["cell"],
            utility=float(item["utility"]),
            reachable=bool(item["reachable"]),
            experimental=dict(item["experimental"]),
        )
        for item in selected
    )


def _sequence_from_goals(goals: tuple[GoalCandidate, ...]) -> tuple[GoalSequence, ...]:
    reachable = tuple(goal for goal in goals if goal.reachable)
    if not reachable:
        return ()
    first = reachable[0]
    return (
        GoalSequence(
            cells=(first.cell,),
            utility=first.utility,
            coverage_area=float(first.experimental.get("expected_new_coverage_area", 0.0)),
            experimental={
                "data_confidence": first.experimental.get("data_confidence", 0.0),
                "risk": first.experimental.get("risk", 0.0),
            },
        ),
    )


def _scenario_to_dict(scenario: Scenario) -> dict[str, Any]:
    return {
        "metadata": dict(scenario.metadata),
        "snapshots": [_contract_to_dict(contract) for contract in scenario.snapshots],
    }


def _contract_to_dict(contract: ModelExplorerContract) -> dict[str, Any]:
    return {
        "schema_version": contract.schema_version,
        "grid": {
            "width": contract.grid.width,
            "height": contract.grid.height,
            "resolution": contract.grid.resolution,
            "frame_id": contract.grid.frame_id,
            "origin": [contract.grid.origin[0], contract.grid.origin[1]],
            "layers": list(contract.grid.layers),
        },
        "constraints": {
            "violation_count": contract.constraints.violation_count,
            "passable_ratio": contract.constraints.passable_ratio,
            "reason_counts": dict(contract.constraints.reason_counts),
        },
        "top_goals": [_goal_to_dict(goal) for goal in contract.top_goals],
        "top_sequences": [_sequence_to_dict(sequence) for sequence in contract.top_sequences],
        "observation_update": dict(contract.observation_update),
        "stable_fields": list(contract.stable_fields),
        "experimental_fields": list(contract.experimental_fields),
    }


def _goal_to_dict(goal: GoalCandidate) -> dict[str, Any]:
    payload = {
        "cell": [goal.cell[0], goal.cell[1]],
        "utility": goal.utility,
        "reachable": goal.reachable,
    }
    payload.update(goal.experimental)
    return payload


def _sequence_to_dict(sequence: GoalSequence) -> dict[str, Any]:
    payload = {
        "cells": [[cell[0], cell[1]] for cell in sequence.cells],
        "utility": sequence.utility,
        "coverage_area": sequence.coverage_area,
    }
    payload.update(sequence.experimental)
    return payload


def _provenance(
    *,
    dataset_id: str,
    data_class: str,
    region: str,
    resolution: float,
    config: LolaSouthPoleRoiConfig,
    seed: int,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    provenance = {
        "dataset_id": str(dataset_id),
        "data_class": str(data_class),
        "region": str(region),
        "resolution": float(resolution),
        "roi": {
            "x": int(config.roi_x),
            "y": int(config.roi_y),
            "width": int(config.roi_width),
            "height": int(config.roi_height),
        },
        "seed": int(seed),
        "generator_version": GENERATOR_VERSION,
    }
    if extra:
        provenance.update(dict(extra))
    return provenance


def _slice_roi(
    values: tuple[tuple[float, ...], ...],
    *,
    config: LolaSouthPoleRoiConfig,
    name: str,
) -> tuple[tuple[float, ...], ...]:
    if config.roi_width <= 0 or config.roi_height <= 0:
        raise ValueError("roi_width and roi_height must be positive")
    if config.roi_x < 0 or config.roi_y < 0:
        raise ValueError("roi_x and roi_y must be non-negative")
    if config.roi_y + config.roi_height > len(values) or config.roi_x + config.roi_width > len(values[0]):
        raise ValueError(f"{name} ROI is outside input bounds")
    return tuple(
        tuple(row[config.roi_x : config.roi_x + config.roi_width])
        for row in values[config.roi_y : config.roi_y + config.roi_height]
    )


def _coerce_grid(values, *, name: str) -> tuple[tuple[float, ...], ...]:
    rows = tuple(tuple(_finite(value) for value in row) for row in values)
    if not rows or not rows[0]:
        raise ValueError(f"{name} must contain at least one row and one column")
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError(f"{name} rows must have equal length")
    return rows


def _neighbor_values(values: tuple[tuple[float, ...], ...], *, x: int, y: int) -> tuple[float, ...]:
    result: list[float] = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            neighbor_x = x + dx
            neighbor_y = y + dy
            if 0 <= neighbor_y < len(values) and 0 <= neighbor_x < len(values[neighbor_y]):
                result.append(values[neighbor_y][neighbor_x])
    return tuple(result)


def _finite(value) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    return numeric if isfinite(numeric) else 0.0


def _clip_unit(value: float) -> float:
    numeric = _finite(value)
    return min(max(numeric, 0.0), 1.0)


def _mean(values: tuple[float, ...]) -> float:
    if not values:
        return 0.0
    return sum(float(value) for value in values) / len(values)
