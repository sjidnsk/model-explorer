from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ...data.lola_south_pole import (
    LolaSouthPoleRoiConfig,
    write_lola_south_pole_scenarios_json,
)
from ...data.raster import read_raster_window
from .manifest import (
    QuasiRealEvaluationManifest,
    VALID_SPLITS,
    _mask_stress_metadata,
    _normalize_mask_stress_config,
    mask_stress_enabled,
    mask_stress_metadata,
    manifest_resolution,
    product_file_name,
)


def _experiment_manifest_payload(
    manifest: QuasiRealEvaluationManifest,
    split_groups: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    splits = {split: groups for split, groups in split_groups.items() if groups}
    return {
        "schema_version": "model-explorer-experiment/v1",
        "name": manifest.name,
        "run_id": manifest.run_id,
        "splits": splits,
        "max_candidates": max(roi.candidate_count for roi in manifest.rois),
        "planner": dict(manifest.planner_config),
        "reward": dict(manifest.reward_config),
        "dataset_validation": dict(manifest.dataset_validation),
        "train": dict(manifest.train_config),
        "outputs": {"root": "out"},
    }


def _apply_mask_stress_to_scenario_json(path: Path, config: dict[str, Any]) -> None:
    normalized = _normalize_mask_stress_config(config)
    if not bool(normalized["enabled"]):
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("scenario JSON root must be an object")
    metadata = payload.setdefault("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("scenario metadata must be an object")
    metadata.update(_mask_stress_metadata(normalized))
    snapshots = payload.get("snapshots", [])
    if not isinstance(snapshots, list):
        raise ValueError("scenario snapshots must be a list")
    for snapshot in snapshots:
        if not isinstance(snapshot, dict):
            continue
        _apply_mask_stress_to_contract_payload(snapshot, normalized)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _apply_mask_stress_to_contract_payload(contract: dict[str, Any], config: dict[str, Any]) -> None:
    goals = contract.get("top_goals", [])
    if not isinstance(goals, list) or not goals:
        return
    missing_fields = tuple(config.get("missing_experimental_fields", ()))
    for index, goal in enumerate(goals):
        if not isinstance(goal, dict):
            continue
        removed_fields = []
        for field in missing_fields:
            if field in goal:
                goal.pop(field)
                removed_fields.append(field)
        if removed_fields:
            goal["mask_stress_missing_fields"] = removed_fields
        goal["mask_stress_augmented"] = True

    reachable_indices = [
        index
        for index, goal in enumerate(goals)
        if isinstance(goal, dict) and bool(goal.get("reachable", False))
    ]
    target_unreachable_count = int(config.get("unreachable_candidate_count", 0))
    changed = 0
    for index in reversed(reachable_indices):
        if changed >= target_unreachable_count:
            break
        if len(reachable_indices) - changed <= 1:
            break
        goal = goals[index]
        if not isinstance(goal, dict):
            continue
        goal["reachable"] = False
        goal["mask_stress_reason"] = "deterministic_unreachable"
        changed += 1

    observation_update = contract.setdefault("observation_update", {})
    if isinstance(observation_update, dict):
        observation_update.update(_mask_stress_metadata(config))
    experimental_fields = contract.get("experimental_fields", [])
    if isinstance(experimental_fields, list):
        for field in ("mask_stress_augmented", "mask_stress_reason", "mask_stress_missing_fields"):
            if field not in experimental_fields:
                experimental_fields.append(field)



def generate_quasi_real_scenarios(
    manifest: QuasiRealEvaluationManifest,
    *,
    data_manifest: dict[str, Any],
    raw_dir: Path,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    dem_path = raw_dir / product_file_name(data_manifest, role="shape_map_radius")
    count_path = raw_dir / product_file_name(data_manifest, role="observation_count")
    resolution = manifest_resolution(data_manifest)
    split_groups: dict[str, list[dict[str, Any]]] = {split: [] for split in VALID_SPLITS}
    generated_scenarios: list[dict[str, Any]] = []
    for roi in manifest.rois:
        dem_window = read_raster_window(
            dem_path,
            x=roi.roi_x,
            y=roi.roi_y,
            width=roi.roi_width,
            height=roi.roi_height,
        )
        count_window = read_raster_window(
            count_path,
            x=roi.roi_x,
            y=roi.roi_y,
            width=roi.roi_width,
            height=roi.roi_height,
        )
        scenario_dir = manifest.output_root / "scenarios" / roi.split / roi.name
        source_config = LolaSouthPoleRoiConfig(
            roi_x=roi.roi_x,
            roi_y=roi.roi_y,
            roi_width=roi.roi_width,
            roi_height=roi.roi_height,
            candidate_count=roi.candidate_count,
            episode_count=roi.episode_count,
            seed=roi.seed,
            start_cell=roi.start_cell,
        )
        metadata_extra = {
            "roi_name": roi.name,
            "split": roi.split,
            "roi": roi.bounds,
            "start_cell": [roi.start_cell[0], roi.start_cell[1]],
        }
        if mask_stress_enabled(manifest.mask_stress_config):
            metadata_extra.update(mask_stress_metadata(manifest.mask_stress_config))
        scenario_paths = write_lola_south_pole_scenarios_json(
            scenario_dir,
            dem_window.values,
            count_window.values,
            dataset_id=str(data_manifest.get("dataset_id", "unknown")),
            data_class=str(data_manifest.get("data_class", "unknown")),
            region=str(data_manifest.get("region", "unknown")),
            resolution=resolution,
            config=LolaSouthPoleRoiConfig(
                roi_x=0,
                roi_y=0,
                roi_width=roi.roi_width,
                roi_height=roi.roi_height,
                candidate_count=roi.candidate_count,
                episode_count=roi.episode_count,
                seed=roi.seed,
                start_cell=roi.start_cell,
            ),
            source_config=source_config,
            metadata_extra=metadata_extra,
        )
        if mask_stress_enabled(manifest.mask_stress_config):
            for scenario_path in scenario_paths:
                apply_mask_stress_to_scenario_json(scenario_path, manifest.mask_stress_config)
        relative_paths = [str(scenario_path.relative_to(manifest.output_root)) for scenario_path in scenario_paths]
        split_groups[roi.split].append({"name": roi.name, "scenarios": relative_paths})
        generated_scenarios.append(
            {
                "name": roi.name,
                "split": roi.split,
                "bounds": roi.bounds,
                "scenario_count": len(scenario_paths),
                "scenarios": relative_paths,
            }
        )
    return split_groups, generated_scenarios


experiment_manifest_payload = _experiment_manifest_payload
apply_mask_stress_to_scenario_json = _apply_mask_stress_to_scenario_json
apply_mask_stress_to_contract_payload = _apply_mask_stress_to_contract_payload

_PUBLIC_EXPORTS = [
    "generate_quasi_real_scenarios",
    "experiment_manifest_payload",
    "apply_mask_stress_to_scenario_json",
    "apply_mask_stress_to_contract_payload",
]
_PRIVATE_COMPAT_EXPORTS = [
    "_experiment_manifest_payload",
    "_apply_mask_stress_to_scenario_json",
    "_apply_mask_stress_to_contract_payload",
]
__all__ = [*_PUBLIC_EXPORTS, *_PRIVATE_COMPAT_EXPORTS]
