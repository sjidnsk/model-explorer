from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


PATH_FEEDBACK_SCHEMA_VERSION = "path-feedback-manifest/v1"


@dataclass(frozen=True)
class PathFeedbackScenario:
    scenario_id: str
    contract_path: Path
    sidecar_path: Path
    scenario_group: str = "unknown"
    scenario_seed: int | str | None = None
    scenario_variant_id: str | None = None
    current_cell: tuple[int, int] = (0, 0)
    route_fixtures: dict[int, Path] = field(default_factory=dict)


@dataclass(frozen=True)
class PathFeedbackManifest:
    schema_version: str
    scenarios: tuple[PathFeedbackScenario, ...]
    planner_config: dict[str, Any]
    top_k: int
    scenario_set: str | None = None
    diagnostic_profile: str | None = None
    acceptance_gate: str | None = None
    python_executable: str | None = None
    planner_extra_args: tuple[str, ...] = ()
    summary_output: Path | None = None
    report_output: Path | None = None
    gcs_control_point_candidate_artifact_output: Path | None = None


def load_path_feedback_manifest(path: str | Path) -> PathFeedbackManifest:
    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("path feedback manifest must be a JSON object")
    schema_version = str(payload.get("schema_version", PATH_FEEDBACK_SCHEMA_VERSION))
    if schema_version != PATH_FEEDBACK_SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {PATH_FEEDBACK_SCHEMA_VERSION}")
    raw_scenarios = payload.get("scenarios")
    if not isinstance(raw_scenarios, list) or not raw_scenarios:
        raise ValueError("scenarios must be a non-empty list")
    scenarios = tuple(_scenario_from_payload(item, base_dir=manifest_path.parent) for item in raw_scenarios)
    outputs = payload.get("outputs") if isinstance(payload.get("outputs"), dict) else {}
    planner_config = _resolve_planner_config(payload.get("planner", {}), base_dir=manifest_path.parent)
    validation_parameters = payload.get("validation_parameters")
    validation_parameters = validation_parameters if isinstance(validation_parameters, dict) else {}
    planner_extra_args = payload.get("planner_extra_args", planner_config.get("extra_args", ()))
    summary_output = _optional_path(outputs.get("summary"), base_dir=manifest_path.parent)
    report_output = _optional_path(outputs.get("report"), base_dir=manifest_path.parent)
    gcs_artifact_output = _optional_path(
        outputs.get("gcs_control_point_candidate_artifacts"),
        base_dir=manifest_path.parent,
    )
    if gcs_artifact_output is None and summary_output is not None:
        gcs_artifact_output = summary_output.parent / "gcs_control_point_candidate_artifacts"
    return PathFeedbackManifest(
        schema_version=schema_version,
        scenarios=scenarios,
        planner_config=planner_config,
        top_k=int(payload.get("top_k", 3)),
        scenario_set=_optional_str(payload.get("scenario_set", validation_parameters.get("scenario_set"))),
        diagnostic_profile=_optional_str(
            payload.get("diagnostic_profile", validation_parameters.get("diagnostic_profile"))
        ),
        acceptance_gate=_optional_str(payload.get("acceptance_gate", validation_parameters.get("acceptance_gate"))),
        python_executable=_optional_str(planner_config.get("python_executable")),
        planner_extra_args=_string_tuple(planner_extra_args),
        summary_output=summary_output,
        report_output=report_output,
        gcs_control_point_candidate_artifact_output=gcs_artifact_output,
    )


def validate_path_feedback_manifest(path: str | Path) -> dict[str, Any]:
    from .planning_routes import load_path_planner_sidecar

    manifest = load_path_feedback_manifest(path)
    for scenario in manifest.scenarios:
        if not scenario.contract_path.exists():
            raise ValueError(f"contract does not exist: {scenario.contract_path}")
        if not scenario.sidecar_path.exists():
            raise ValueError(f"sidecar does not exist: {scenario.sidecar_path}")
        load_path_planner_sidecar(scenario.sidecar_path)
        for fixture in scenario.route_fixtures.values():
            if not fixture.exists():
                raise ValueError(f"route fixture does not exist: {fixture}")
    return {
        "status": "valid",
        "schema_version": manifest.schema_version,
        "scenario_count": len(manifest.scenarios),
        "top_k": manifest.top_k,
    }


def _scenario_from_payload(payload: Any, *, base_dir: Path) -> PathFeedbackScenario:
    if not isinstance(payload, dict):
        raise ValueError("scenario entries must be objects")
    scenario_id = str(payload.get("scenario_id") or payload.get("id") or "")
    if not scenario_id:
        raise ValueError("scenario_id is required")
    return PathFeedbackScenario(
        scenario_id=scenario_id,
        contract_path=_required_path(payload, "contract", base_dir=base_dir),
        sidecar_path=_required_path(payload, "sidecar", base_dir=base_dir),
        scenario_group=str(payload.get("scenario_group") or payload.get("group") or "unknown"),
        scenario_seed=_scenario_seed(payload.get("scenario_seed")),
        scenario_variant_id=_optional_string(payload.get("scenario_variant_id")),
        current_cell=_cell(payload.get("current_cell", [0, 0])),
        route_fixtures=_route_fixtures(payload.get("route_fixtures", {}), base_dir=base_dir),
    )


def _scenario_seed(value: Any) -> int | str | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    try:
        return int(value)
    except (TypeError, ValueError):
        parsed = str(value)
        return parsed if parsed else None


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    parsed = str(value)
    return parsed if parsed else None


def _resolve_planner_config(config: Any, *, base_dir: Path) -> dict[str, Any]:
    if config is None:
        return {"backend": "path_planner_route"}
    if not isinstance(config, dict):
        raise ValueError("planner must be an object")
    resolved = dict(config)
    for key in ("path_planner_root", "platform_config", "output_dir"):
        if key in resolved and resolved[key] is not None:
            resolved[key] = str(_resolve_path(base_dir, resolved[key]))
    return resolved


def _route_fixtures(payload: Any, *, base_dir: Path) -> dict[int, Path]:
    if payload is None:
        return {}
    if isinstance(payload, list):
        return {index: _resolve_path(base_dir, value) for index, value in enumerate(payload)}
    if isinstance(payload, dict):
        return {int(key): _resolve_path(base_dir, value) for key, value in payload.items()}
    raise ValueError("route_fixtures must be a list or object")


def _required_path(payload: dict[str, Any], key: str, *, base_dir: Path) -> Path:
    value = payload.get(key)
    if value is None:
        raise ValueError(f"{key} is required")
    return _resolve_path(base_dir, value)


def _optional_path(value: Any, *, base_dir: Path) -> Path | None:
    if value is None:
        return None
    return _resolve_path(base_dir, value)


def _resolve_path(base_dir: Path, value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else base_dir / path


def _cell(value: Any) -> tuple[int, int]:
    if not isinstance(value, list | tuple) or len(value) != 2:
        raise ValueError("cell must be [x, y]")
    return (int(value[0]), int(value[1]))


def _cell_to_list(cell: tuple[int, int] | None) -> list[int] | None:
    return None if cell is None else [cell[0], cell[1]]


def _cell_tuple(value: Any) -> tuple[int, int] | None:
    if not isinstance(value, list | tuple) or len(value) != 2:
        return None
    try:
        return (int(value[0]), int(value[1]))
    except (TypeError, ValueError):
        return None


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if not isinstance(value, list | tuple):
        raise ValueError("planner_extra_args must be a list")
    return tuple(str(item) for item in value)


resolve_path = _resolve_path
required_path = _required_path
optional_path = _optional_path
route_fixtures = _route_fixtures
scenario_from_payload = _scenario_from_payload
cell_to_list = _cell_to_list
cell_tuple = _cell_tuple

__all__ = [
    'PATH_FEEDBACK_SCHEMA_VERSION',
    'PathFeedbackManifest',
    'PathFeedbackScenario',
    'load_path_feedback_manifest',
    'validate_path_feedback_manifest',
    'resolve_path',
    'required_path',
    'optional_path',
    'route_fixtures',
    'scenario_from_payload',
    'cell_to_list',
    'cell_tuple',
]
