"""LOLA South Pole rollout orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..data.lola_south_pole import LolaSouthPoleRoiConfig, generate_lola_south_pole_scenarios
from ..data.manifest import load_data_manifest, validate_data_manifest
from ..data.raster import read_raster_window
from ..policy.collector import collect_rollout_episode
from ..policy.rollout import RolloutEpisode
from ..policy.rollout_io import write_rollout_episodes_jsonl


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
    return tuple(
        collect_rollout_episode(scenario, max_candidates=config.candidate_count)
        for scenario in scenarios
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
    episodes = generate_lola_south_pole_rollout_episodes(
        dem_values,
        observation_count_values,
        dataset_id=dataset_id,
        data_class=data_class,
        region=region,
        resolution=resolution,
        config=config,
    )
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_rollout_episodes_jsonl(output_path, episodes)
    return episodes


def write_lola_south_pole_rollouts_from_manifest_jsonl(
    manifest_path: str | Path,
    output_path: str | Path,
    *,
    config: LolaSouthPoleRoiConfig,
) -> tuple[RolloutEpisode, ...]:
    validation = validate_data_manifest(manifest_path)
    validation.require_valid()
    manifest = load_data_manifest(manifest_path)
    raw_dir = Path(validation.raw_dir)
    dem_file = _product_file_name(manifest, role="shape_map_radius")
    count_file = _product_file_name(manifest, role="observation_count")
    resolution = _manifest_resolution(manifest)
    dem_window = read_raster_window(
        raw_dir / dem_file,
        x=config.roi_x,
        y=config.roi_y,
        width=config.roi_width,
        height=config.roi_height,
    )
    count_window = read_raster_window(
        raw_dir / count_file,
        x=config.roi_x,
        y=config.roi_y,
        width=config.roi_width,
        height=config.roi_height,
    )
    window_config = LolaSouthPoleRoiConfig(
        roi_x=0,
        roi_y=0,
        roi_width=config.roi_width,
        roi_height=config.roi_height,
        candidate_count=config.candidate_count,
        episode_count=config.episode_count,
        seed=config.seed,
        start_cell=config.start_cell,
    )
    episodes = generate_lola_south_pole_rollout_episodes(
        dem_window.values,
        count_window.values,
        dataset_id=str(manifest.get("dataset_id", "unknown")),
        data_class=str(manifest.get("data_class", "unknown")),
        region=str(manifest.get("region", "unknown")),
        resolution=resolution,
        config=window_config,
        source_config=config,
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_rollout_episodes_jsonl(output, episodes)
    return episodes


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


__all__ = [
    "generate_lola_south_pole_rollout_episodes",
    "write_lola_south_pole_rollouts_from_manifest_jsonl",
    "write_lola_south_pole_rollouts_jsonl",
]
