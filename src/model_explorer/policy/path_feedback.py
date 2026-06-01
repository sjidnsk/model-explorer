from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core.interfaces import GoalCandidate, ModelExplorerContract
from ..io.scenario import load_scenario
from .planning import (
    PathPlanRequest,
    PathPlanResult,
    PathPlanningAdapter,
    evaluate_candidate_paths,
    load_path_planner_sidecar,
    path_feedback_summary,
    planner_from_config,
    PathPlannerRouteAdapter,
)


PATH_FEEDBACK_SCHEMA_VERSION = "path-feedback-manifest/v1"


@dataclass(frozen=True)
class PathFeedbackScenario:
    scenario_id: str
    contract_path: Path
    sidecar_path: Path
    current_cell: tuple[int, int] = (0, 0)
    route_fixtures: dict[int, Path] = field(default_factory=dict)


@dataclass(frozen=True)
class PathFeedbackManifest:
    schema_version: str
    scenarios: tuple[PathFeedbackScenario, ...]
    planner_config: dict[str, Any]
    top_k: int
    summary_output: Path | None = None
    report_output: Path | None = None


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
    return PathFeedbackManifest(
        schema_version=schema_version,
        scenarios=scenarios,
        planner_config=_resolve_planner_config(payload.get("planner", {}), base_dir=manifest_path.parent),
        top_k=int(payload.get("top_k", 3)),
        summary_output=_optional_path(outputs.get("summary"), base_dir=manifest_path.parent),
        report_output=_optional_path(outputs.get("report"), base_dir=manifest_path.parent),
    )


def dry_run_path_feedback_manifest(path: str | Path) -> dict[str, Any]:
    manifest = load_path_feedback_manifest(path)
    return {
        "status": "dry_run",
        "schema_version": manifest.schema_version,
        "scenario_count": len(manifest.scenarios),
        "top_k": manifest.top_k,
        "planner": str(manifest.planner_config.get("backend", "path_planner_route")),
        "scenarios": [
            {
                "scenario_id": scenario.scenario_id,
                "contract": str(scenario.contract_path),
                "sidecar": str(scenario.sidecar_path),
                "route_fixture_count": len(scenario.route_fixtures),
            }
            for scenario in manifest.scenarios
        ],
        "would_write": [
            str(path)
            for path in (manifest.summary_output, manifest.report_output)
            if path is not None
        ],
    }


def validate_path_feedback_manifest(path: str | Path) -> dict[str, Any]:
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


def run_path_feedback_manifest(path: str | Path) -> dict[str, Any]:
    manifest = load_path_feedback_manifest(path)
    summary = run_path_feedback(manifest)
    if manifest.summary_output is not None:
        manifest.summary_output.parent.mkdir(parents=True, exist_ok=True)
        manifest.summary_output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    if manifest.report_output is not None:
        manifest.report_output.parent.mkdir(parents=True, exist_ok=True)
        manifest.report_output.write_text(render_path_feedback_markdown(summary), encoding="utf-8")
    return summary


def run_path_feedback(manifest: PathFeedbackManifest) -> dict[str, Any]:
    scenario_summaries = [
        _run_feedback_scenario(scenario, manifest=manifest)
        for scenario in manifest.scenarios
    ]
    total_path_cost = sum(float(item["selected_path_cost_after_feedback"] or 0.0) for item in scenario_summaries)
    total_coverage_delta = sum(float(item["coverage_rate_delta"]) for item in scenario_summaries)
    selected_path_costs = [
        float(item["selected_path_cost_after_feedback"])
        for item in scenario_summaries
        if item["selected_path_cost_after_feedback"] is not None
    ]
    return {
        "schema_version": "path-feedback-summary/v1",
        "scenario_count": len(scenario_summaries),
        "top_k": manifest.top_k,
        "candidate_count": sum(int(item["path_feedback"]["candidate_count"]) for item in scenario_summaries),
        "reachable_count": sum(int(item["path_feedback"]["reachable_count"]) for item in scenario_summaries),
        "path_planning_failure_count": sum(int(item["path_feedback"]["failure_count"]) for item in scenario_summaries),
        "replan_count": sum(int(item["path_feedback"]["replan_count"]) for item in scenario_summaries),
        "total_path_cost": total_path_cost,
        "average_path_cost": (
            total_path_cost / len(selected_path_costs)
            if selected_path_costs
            else 0.0
        ),
        "coverage_per_path_cost": (
            total_coverage_delta / total_path_cost
            if total_path_cost > 0.0
            else 0.0
        ),
        "tracking_safety_violation_count": sum(
            int(item["tracking_safety_violation_count"]) for item in scenario_summaries
        ),
        "trajectory_optimization_fallback_count": sum(
            int(item["trajectory_optimization_fallback_count"]) for item in scenario_summaries
        ),
        "region_graph_disconnected_count": sum(
            int(item["region_graph_disconnected_count"]) for item in scenario_summaries
        ),
        "open_grid_fallback_used": any(bool(item["open_grid_fallback_used"]) for item in scenario_summaries),
        "failure_reasons": [
            reason
            for item in scenario_summaries
            for reason in item["path_feedback"]["failure_reasons"]
        ],
        "scenarios": scenario_summaries,
    }


def render_path_feedback_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Path Feedback Summary",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| scenario_count | {summary['scenario_count']} |",
        f"| candidate_count | {summary['candidate_count']} |",
        f"| reachable_count | {summary['reachable_count']} |",
        f"| path_planning_failure_count | {summary['path_planning_failure_count']} |",
        f"| replan_count | {summary['replan_count']} |",
        f"| total_path_cost | {summary['total_path_cost']} |",
        f"| coverage_per_path_cost | {summary['coverage_per_path_cost']} |",
        f"| open_grid_fallback_used | {summary['open_grid_fallback_used']} |",
        "",
        "## Scenarios",
        "",
        "| scenario | before | after | changed | reachable | failures | path_cost |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for item in summary["scenarios"]:
        lines.append(
            "| {scenario_id} | {before} | {after} | {changed} | {reachable} | {failures} | {cost} |".format(
                scenario_id=item["scenario_id"],
                before=item["selected_cell_before_path_feedback"],
                after=item["selected_cell_after_path_feedback"],
                changed=item["selection_changed_by_path_feedback"],
                reachable=item["path_feedback"]["reachable_count"],
                failures=item["path_feedback"]["failure_count"],
                cost=item["selected_path_cost_after_feedback"],
            )
        )
    lines.extend(
        [
            "",
            "IRIS / region graph fields are diagnostic features only. They should not replace the current reliable fallback chain until they consistently explain path failure or high-risk exposure.",
            "",
        ]
    )
    return "\n".join(lines)


def _run_feedback_scenario(
    scenario: PathFeedbackScenario,
    *,
    manifest: PathFeedbackManifest,
) -> dict[str, Any]:
    contract = load_scenario(scenario.contract_path).snapshots[0]
    planner = _planner_for_scenario(scenario, manifest=manifest)
    evaluations = evaluate_candidate_paths(
        contract,
        current_cell=scenario.current_cell,
        top_k=manifest.top_k,
        planner=planner,
    )
    feedback = path_feedback_summary(evaluations)
    selected_before = _selected_before_feedback(contract)
    selected_after = _selected_after_feedback(evaluations)
    selected_after_cost = None if selected_after is None else selected_after.result.path_cost
    return {
        "scenario_id": scenario.scenario_id,
        "selected_cell_before_path_feedback": _cell_to_list(
            selected_before.cell if selected_before is not None else None
        ),
        "selected_cell_after_path_feedback": _cell_to_list(
            selected_after.cell if selected_after is not None else None
        ),
        "selection_changed_by_path_feedback": (
            _cell_to_list(selected_before.cell if selected_before is not None else None)
            != _cell_to_list(selected_after.cell if selected_after is not None else None)
        ),
        "selected_path_cost_after_feedback": selected_after_cost,
        "coverage_rate_delta": _numeric_observation(contract, "coverage_rate_delta"),
        "open_grid_fallback_used": _open_grid_fallback_used(evaluations),
        "tracking_safety_violation_count": _tracking_safety_violation_count(evaluations),
        "trajectory_optimization_fallback_count": _trajectory_optimization_fallback_count(evaluations),
        "region_graph_disconnected_count": _region_graph_disconnected_count(evaluations),
        "path_feedback": feedback,
    }


def _planner_for_scenario(
    scenario: PathFeedbackScenario,
    *,
    manifest: PathFeedbackManifest,
) -> PathPlanningAdapter:
    if scenario.route_fixtures:
        return _RouteFixturePlanner(scenario.sidecar_path, scenario.route_fixtures)
    config = dict(manifest.planner_config)
    config.setdefault("backend", "path_planner_route")
    config["path_planner_sidecar"] = str(scenario.sidecar_path)
    return planner_from_config(config)


def _selected_before_feedback(contract: ModelExplorerContract) -> GoalCandidate | None:
    for goal in contract.top_goals:
        if goal.reachable:
            return goal
    return None


def _selected_after_feedback(evaluations) -> Any | None:
    feasible = [item for item in evaluations if item.result.feasible and not item.result.replan_required]
    if not feasible:
        feasible = [item for item in evaluations if item.result.feasible]
    if not feasible:
        return None
    return min(
        feasible,
        key=lambda item: (
            float(item.result.path_cost),
            float(item.result.risk),
            -float(item.utility),
            item.cell[0],
            item.cell[1],
        ),
    )


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
        current_cell=_cell(payload.get("current_cell", [0, 0])),
        route_fixtures=_route_fixtures(payload.get("route_fixtures", {}), base_dir=base_dir),
    )


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


def _numeric_observation(contract: ModelExplorerContract, field: str) -> float:
    value = contract.observation_update.get(field)
    if isinstance(value, bool) or value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _open_grid_fallback_used(evaluations) -> bool:
    for item in evaluations:
        request_metadata = item.result.metadata.get("request_payload", {}).get("metadata", {})
        if request_metadata.get("cost_source") == "open_grid_fallback":
            return True
        if request_metadata.get("passable_mask_source") == "open_grid_fallback":
            return True
    return False


def _tracking_safety_violation_count(evaluations) -> int:
    total = 0
    for item in evaluations:
        postprocess = item.to_dict().get("postprocess")
        if isinstance(postprocess, dict):
            total += int(postprocess.get("tracking_safety_violation_count") or 0)
    return total


def _trajectory_optimization_fallback_count(evaluations) -> int:
    total = 0
    for item in evaluations:
        optimization = item.to_dict().get("trajectory_optimization")
        if isinstance(optimization, dict):
            fallback = optimization.get("fallback_status")
            if isinstance(fallback, str) and fallback not in {"ok", "not_needed", "none"}:
                total += 1
    return total


def _region_graph_disconnected_count(evaluations) -> int:
    total = 0
    for item in evaluations:
        region_graph = item.to_dict().get("region_graph")
        if isinstance(region_graph, dict) and region_graph.get("start_goal_connected") is False:
            total += 1
    return total


class _RouteFixturePlanner:
    def __init__(self, sidecar_path: Path, route_fixtures: dict[int, Path]) -> None:
        self._sidecar = load_path_planner_sidecar(sidecar_path)
        self._route_fixtures = dict(route_fixtures)

    def plan(self, request: PathPlanRequest) -> PathPlanResult:
        route_path = self._route_fixtures.get(request.action_index)
        if route_path is None:
            raise ValueError(f"missing route fixture for action_index={request.action_index}")
        return PathPlannerRouteAdapter(sidecar=self._sidecar, route_json=route_path).plan(request)
