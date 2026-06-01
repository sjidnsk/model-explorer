from __future__ import annotations

import json
from collections import Counter, defaultdict
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
PATH_FEEDBACK_SUMMARY_SCHEMA_VERSION = "path-feedback-summary/v1"
PATH_FEEDBACK_SUMMARY_ACCEPTANCE_METRICS = (
    "selection_changed_rate",
    "path_planning_failure_count",
    "replan_count",
    "tracking_safety_violation_count",
    "trajectory_optimization_fallback_count",
    "region_graph_disconnected_count",
    "coverage_per_path_cost",
)
PATH_FEEDBACK_SUMMARY_REQUIRED_KEYS = (
    "schema_version",
    "scenario_count",
    "top_k",
    "candidate_count",
    "reachable_count",
    "path_planning_failure_count",
    "replan_count",
    "total_path_cost",
    "average_path_cost",
    "coverage_per_path_cost",
    "selection_changed_count",
    "selection_changed_rate",
    "tracking_safety_violation_count",
    "trajectory_optimization_fallback_count",
    "region_graph_disconnected_count",
    "open_grid_fallback_used",
    "failure_reasons",
    "iris_requested_count",
    "iris_report_count",
    "iris_status_counts",
    "iris_fallback_count",
    "iris_failure_count",
    "iris_region_count_total",
    "iris_fallback_reasons",
    "region_graph_source_counts",
    "region_graph_fallback_count",
    "region_graph_fallback_reasons",
    "region_graph_start_goal_disconnected_count",
    "scenario_group_summary",
    "scenarios",
)


@dataclass(frozen=True)
class PathFeedbackScenario:
    scenario_id: str
    contract_path: Path
    sidecar_path: Path
    scenario_group: str = "unknown"
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
                "scenario_group": scenario.scenario_group,
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
    validate_path_feedback_summary_contract(summary)
    if manifest.summary_output is not None:
        manifest.summary_output.parent.mkdir(parents=True, exist_ok=True)
        manifest.summary_output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    if manifest.report_output is not None:
        manifest.report_output.parent.mkdir(parents=True, exist_ok=True)
        manifest.report_output.write_text(render_path_feedback_markdown(summary), encoding="utf-8")
    return summary


def validate_path_feedback_summary_contract(
    summary: dict[str, Any],
    *,
    require_sidecar_inputs: bool = True,
) -> dict[str, Any]:
    if not isinstance(summary, dict):
        raise ValueError("path-feedback-summary/v1 summary must be an object")
    schema_version = summary.get("schema_version")
    if schema_version != PATH_FEEDBACK_SUMMARY_SCHEMA_VERSION:
        raise ValueError(
            f"schema_version must be {PATH_FEEDBACK_SUMMARY_SCHEMA_VERSION}, got {schema_version!r}"
        )
    missing = [key for key in PATH_FEEDBACK_SUMMARY_REQUIRED_KEYS if key not in summary]
    if missing:
        raise ValueError("path-feedback-summary/v1 missing required keys: " + ", ".join(missing))
    if require_sidecar_inputs and summary.get("open_grid_fallback_used") is not False:
        raise ValueError("open_grid_fallback_used must be false for semi-real path feedback validation")
    return {
        "status": "valid",
        "schema_version": PATH_FEEDBACK_SUMMARY_SCHEMA_VERSION,
        "required_key_count": len(PATH_FEEDBACK_SUMMARY_REQUIRED_KEYS),
        "acceptance_metric_count": len(PATH_FEEDBACK_SUMMARY_ACCEPTANCE_METRICS),
    }


def compact_path_feedback_summary(
    summary: dict[str, Any],
    *,
    summary_output: Path | None = None,
    report_output: Path | None = None,
) -> dict[str, Any]:
    payload = {
        "status": "completed",
        "schema_version": summary.get("schema_version"),
        "scenario_count": summary.get("scenario_count"),
        "top_k": summary.get("top_k"),
        "candidate_count": summary.get("candidate_count"),
        "reachable_count": summary.get("reachable_count"),
        "path_planning_failure_count": summary.get("path_planning_failure_count"),
        "replan_count": summary.get("replan_count"),
        "selection_changed_count": summary.get("selection_changed_count"),
        "selection_changed_rate": summary.get("selection_changed_rate"),
        "total_path_cost": summary.get("total_path_cost"),
        "average_path_cost": summary.get("average_path_cost"),
        "coverage_per_path_cost": summary.get("coverage_per_path_cost"),
        "tracking_safety_violation_count": summary.get("tracking_safety_violation_count"),
        "trajectory_optimization_fallback_count": summary.get("trajectory_optimization_fallback_count"),
        "region_graph_disconnected_count": summary.get("region_graph_disconnected_count"),
        "open_grid_fallback_used": summary.get("open_grid_fallback_used"),
        "failure_reasons": summary.get("failure_reasons", []),
        "iris_requested_count": summary.get("iris_requested_count"),
        "iris_report_count": summary.get("iris_report_count"),
        "iris_status_counts": summary.get("iris_status_counts", {}),
        "iris_fallback_count": summary.get("iris_fallback_count"),
        "iris_failure_count": summary.get("iris_failure_count"),
        "region_graph_source_counts": summary.get("region_graph_source_counts", {}),
        "region_graph_fallback_count": summary.get("region_graph_fallback_count"),
        "region_graph_start_goal_disconnected_count": summary.get("region_graph_start_goal_disconnected_count"),
        "diagnostic_interpretation": summary.get("diagnostic_interpretation", {}),
    }
    if summary_output is not None:
        payload["summary_output"] = str(summary_output)
    if report_output is not None:
        payload["report_output"] = str(report_output)
    return payload


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
    selection_changed_count = sum(
        1 for item in scenario_summaries if item["selection_changed_by_path_feedback"]
    )
    diagnostic_summary = _diagnostic_aggregate(scenario_summaries)
    diagnostic_interpretation = _diagnostic_interpretation_summary(scenario_summaries)
    return {
        "schema_version": PATH_FEEDBACK_SUMMARY_SCHEMA_VERSION,
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
        "selection_changed_count": selection_changed_count,
        "selection_changed_rate": (
            selection_changed_count / len(scenario_summaries)
            if scenario_summaries
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
        **diagnostic_summary,
        "diagnostic_interpretation": diagnostic_interpretation,
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
        f"| selection_changed_count | {summary['selection_changed_count']} |",
        f"| selection_changed_rate | {summary['selection_changed_rate']} |",
        f"| total_path_cost | {summary['total_path_cost']} |",
        f"| coverage_per_path_cost | {summary['coverage_per_path_cost']} |",
        f"| open_grid_fallback_used | {summary['open_grid_fallback_used']} |",
        "",
        "## Baseline vs Feedback",
        "",
        "| scenario | group | before | after | changed | before_path_cost | after_path_cost | delta | coverage_delta | reachable | failures | replans |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summary["scenarios"]:
        lines.append(
            "| {scenario_id} | {group} | {before} | {after} | {changed} | {before_cost} | {after_cost} | {delta} | {coverage_delta} | {reachable} | {failures} | {replans} |".format(
                scenario_id=item["scenario_id"],
                group=item["scenario_group"],
                before=item["selected_cell_before_path_feedback"],
                after=item["selected_cell_after_path_feedback"],
                changed=item["selection_changed_by_path_feedback"],
                before_cost=item["selected_path_cost_before_feedback"],
                after_cost=item["selected_path_cost_after_feedback"],
                delta=item["path_cost_delta_after_feedback"],
                coverage_delta=item["coverage_rate_delta"],
                reachable=item["path_feedback"]["reachable_count"],
                failures=item["path_feedback"]["failure_count"],
                replans=item["path_feedback"]["replan_count"],
            )
        )
    lines.extend(
        [
            "",
            "## Candidate Paths",
            "",
            "| scenario | action | cell | reachable | path_cost | risk | utility | replan | failure |",
            "|---|---:|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for item in summary["scenarios"]:
        for candidate in item["path_feedback"]["candidates"]:
            lines.append(
                "| {scenario_id} | {action} | {cell} | {reachable} | {path_cost} | {risk} | {utility} | {replan} | {failure} |".format(
                    scenario_id=item["scenario_id"],
                    action=candidate["action_index"],
                    cell=candidate["cell"],
                    reachable=candidate["reachable"],
                    path_cost=candidate["path_cost"],
                    risk=candidate["risk"],
                    utility=candidate["utility"],
                    replan=candidate["replan_required"],
                    failure=candidate["failure_reason"],
                )
            )
    lines.extend(
        [
            "",
            "## Diagnostic Interpretation",
            "",
            "| scenario | group | replacement_reason | failure_sources | primary_failure_reason | iris_region_graph_signal | open_grid_fallback |",
            "|---|---|---|---|---|---|---:|",
        ]
    )
    for item in summary["scenarios"]:
        interpretation = item["diagnostic_interpretation"]
        lines.append(
            "| {scenario_id} | {group} | {reason} | {sources} | {primary} | {signal} | {open_grid} |".format(
                scenario_id=item["scenario_id"],
                group=item["scenario_group"],
                reason=interpretation["target_replacement_reason"],
                sources=_list_text(interpretation["failure_sources"]),
                primary=interpretation["primary_failure_reason"],
                signal=interpretation["iris_region_graph_signal"],
                open_grid=interpretation["open_grid_fallback_used"],
            )
        )
    lines.extend(
        [
            "",
            "## Candidate Diagnostics",
            "",
            "| scenario | action | cell | reachable | replan | failure | flags | iris_status | iris_fallback | graph_source | graph_fallback | graph_connected | open_grid |",
            "|---|---:|---|---:|---:|---|---|---|---:|---|---:|---|---:|",
        ]
    )
    for item in summary["scenarios"]:
        for candidate in item["path_feedback"]["candidates"]:
            interpretation = candidate["diagnostic_interpretation"]
            lines.append(
                "| {scenario_id} | {action} | {cell} | {reachable} | {replan} | {failure} | {flags} | {iris_status} | {iris_fallback} | {graph_source} | {graph_fallback} | {graph_connected} | {open_grid} |".format(
                    scenario_id=item["scenario_id"],
                    action=candidate["action_index"],
                    cell=candidate["cell"],
                    reachable=candidate["reachable"],
                    replan=candidate["replan_required"],
                    failure=candidate["failure_reason"],
                    flags=_list_text(interpretation["diagnostic_flags"]),
                    iris_status=interpretation["iris_status"],
                    iris_fallback=interpretation["iris_fallback_used"],
                    graph_source=interpretation["region_graph_source"],
                    graph_fallback=interpretation["region_graph_fallback_used"],
                    graph_connected=interpretation["region_graph_start_goal_connected"],
                    open_grid=interpretation["open_grid_fallback_used"],
                )
            )
    lines.extend(
        [
            "",
            "## IRIS Diagnostics",
            "",
            "| scenario | group | before | after | failures | replans | iris_status_counts | iris_fallback_reasons | iris_region_count |",
            "|---|---|---|---|---:|---:|---|---|---:|",
        ]
    )
    for item in summary["scenarios"]:
        iris = item["iris_diagnostics"]
        lines.append(
            "| {scenario_id} | {group} | {before} | {after} | {failures} | {replans} | {statuses} | {reasons} | {count} |".format(
                scenario_id=item["scenario_id"],
                group=item["scenario_group"],
                before=item["selected_cell_before_path_feedback"],
                after=item["selected_cell_after_path_feedback"],
                failures=item["path_feedback"]["failure_count"],
                replans=item["path_feedback"]["replan_count"],
                statuses=iris["status_counts"],
                reasons=iris["fallback_reasons"],
                count=iris["region_count_total"],
            )
        )
    lines.extend(
        [
            "",
            "## Region Graph Diagnostics",
            "",
            "| scenario | group | before | after | failures | replans | graph_source_counts | fallback_reasons | disconnected |",
            "|---|---|---|---|---:|---:|---|---|---:|",
        ]
    )
    for item in summary["scenarios"]:
        graph = item["region_graph_diagnostics"]
        lines.append(
            "| {scenario_id} | {group} | {before} | {after} | {failures} | {replans} | {sources} | {reasons} | {disconnected} |".format(
                scenario_id=item["scenario_id"],
                group=item["scenario_group"],
                before=item["selected_cell_before_path_feedback"],
                after=item["selected_cell_after_path_feedback"],
                failures=item["path_feedback"]["failure_count"],
                replans=item["path_feedback"]["replan_count"],
                sources=graph["source_counts"],
                reasons=graph["fallback_reasons"],
                disconnected=graph["start_goal_disconnected_count"],
            )
        )
    lines.extend(
        [
            "",
            "## Scenario Groups",
            "",
            "| group | scenarios | candidates | reachable | failures | replans | changed | iris_reports | graph_fallbacks | disconnected |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for group, payload in summary["scenario_group_summary"].items():
        lines.append(
            "| {group} | {scenario_count} | {candidate_count} | {reachable_count} | {failure_count} | {replan_count} | {selection_changed_count} | {iris_report_count} | {region_graph_fallback_count} | {region_graph_start_goal_disconnected_count} |".format(
                group=group,
                **payload,
            )
        )
    lines.extend(
        [
            "",
            "IRIS / region graph fields are diagnostic features only. They should not replace the current reliable fallback chain until they consistently explain path failure or high-risk exposure.",
            "They are not a GCS trajectory or an Ackermann/skid-steer feasibility proof, and IRIS fallback must not change a successful A* route to unreachable.",
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
    selected_before_cost = _candidate_path_cost_for_cell(
        evaluations,
        None if selected_before is None else selected_before.cell,
    )
    path_cost_delta = _path_cost_delta(selected_before_cost, selected_after_cost)
    before_cell = _cell_to_list(selected_before.cell if selected_before is not None else None)
    after_cell = _cell_to_list(selected_after.cell if selected_after is not None else None)
    summary = {
        "scenario_id": scenario.scenario_id,
        "scenario_group": scenario.scenario_group,
        "selected_cell_before_path_feedback": before_cell,
        "selected_cell_after_path_feedback": after_cell,
        "selection_changed_by_path_feedback": before_cell != after_cell,
        "selected_path_cost_before_feedback": selected_before_cost,
        "selected_path_cost_after_feedback": selected_after_cost,
        "path_cost_delta_after_feedback": path_cost_delta,
        "coverage_rate_delta": _numeric_observation(contract, "coverage_rate_delta"),
        "open_grid_fallback_used": _open_grid_fallback_used(evaluations),
        "tracking_safety_violation_count": _tracking_safety_violation_count(evaluations),
        "trajectory_optimization_fallback_count": _trajectory_optimization_fallback_count(evaluations),
        "region_graph_disconnected_count": _region_graph_disconnected_count(evaluations),
        "iris_diagnostics": _iris_diagnostics(evaluations),
        "region_graph_diagnostics": _region_graph_diagnostics(evaluations),
        "baseline_vs_feedback": {
            "before_cell": before_cell,
            "after_cell": after_cell,
            "selection_changed": before_cell != after_cell,
            "selected_path_cost_before_feedback": selected_before_cost,
            "selected_path_cost_after_feedback": selected_after_cost,
            "path_cost_delta_after_feedback": path_cost_delta,
            "coverage_rate_delta": _numeric_observation(contract, "coverage_rate_delta"),
        },
        "path_feedback": feedback,
    }
    summary["diagnostic_interpretation"] = _scenario_diagnostic_interpretation(summary)
    return summary


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


def _candidate_path_cost_for_cell(evaluations, cell: tuple[int, int] | None) -> float | None:
    if cell is None:
        return None
    for item in evaluations:
        if item.cell == cell:
            return float(item.result.path_cost) if item.result.feasible else None
    return None


def _path_cost_delta(before: float | None, after: float | None) -> float | None:
    if before is None or after is None:
        return None
    return float(after - before)


def _diagnostic_aggregate(scenarios: list[dict[str, Any]]) -> dict[str, Any]:
    iris_status_counts: Counter[str] = Counter()
    iris_fallback_reasons: Counter[str] = Counter()
    region_graph_source_counts: Counter[str] = Counter()
    region_graph_fallback_reasons: Counter[str] = Counter()
    group_summary: dict[str, dict[str, Any]] = defaultdict(_empty_group_summary)
    iris_report_count = 0
    iris_fallback_count = 0
    iris_failure_count = 0
    iris_region_count_total = 0
    region_graph_fallback_count = 0
    region_graph_start_goal_disconnected_count = 0

    for scenario in scenarios:
        group = str(scenario.get("scenario_group") or "unknown")
        group_payload = group_summary[group]
        group_payload["scenario_count"] += 1
        group_payload["candidate_count"] += int(scenario["path_feedback"]["candidate_count"])
        group_payload["reachable_count"] += int(scenario["path_feedback"]["reachable_count"])
        group_payload["failure_count"] += int(scenario["path_feedback"]["failure_count"])
        group_payload["replan_count"] += int(scenario["path_feedback"]["replan_count"])
        group_payload["selection_changed_count"] += int(bool(scenario["selection_changed_by_path_feedback"]))

        iris = scenario["iris_diagnostics"]
        graph = scenario["region_graph_diagnostics"]
        group_payload["iris_report_count"] += int(iris["report_count"])
        group_payload["iris_fallback_count"] += int(iris["fallback_count"])
        group_payload["region_graph_fallback_count"] += int(graph["fallback_count"])
        group_payload["region_graph_start_goal_disconnected_count"] += int(graph["start_goal_disconnected_count"])

        iris_report_count += int(iris["report_count"])
        iris_fallback_count += int(iris["fallback_count"])
        iris_failure_count += int(iris["failure_count"])
        iris_region_count_total += int(iris["region_count_total"])
        region_graph_fallback_count += int(graph["fallback_count"])
        region_graph_start_goal_disconnected_count += int(graph["start_goal_disconnected_count"])
        iris_status_counts.update(iris["status_counts"])
        iris_fallback_reasons.update(iris["fallback_reasons"])
        region_graph_source_counts.update(graph["source_counts"])
        region_graph_fallback_reasons.update(graph["fallback_reasons"])

    return {
        "iris_requested_count": iris_report_count,
        "iris_report_count": iris_report_count,
        "iris_status_counts": dict(sorted(iris_status_counts.items())),
        "iris_fallback_count": iris_fallback_count,
        "iris_failure_count": iris_failure_count,
        "iris_region_count_total": iris_region_count_total,
        "iris_fallback_reasons": dict(sorted(iris_fallback_reasons.items())),
        "region_graph_source_counts": dict(sorted(region_graph_source_counts.items())),
        "region_graph_fallback_count": region_graph_fallback_count,
        "region_graph_fallback_reasons": dict(sorted(region_graph_fallback_reasons.items())),
        "region_graph_start_goal_disconnected_count": region_graph_start_goal_disconnected_count,
        "scenario_group_summary": {
            group: dict(payload)
            for group, payload in sorted(group_summary.items())
        },
    }


def _diagnostic_interpretation_summary(scenarios: list[dict[str, Any]]) -> dict[str, Any]:
    group_summary: dict[str, dict[str, Counter[str]]] = defaultdict(_empty_group_interpretation)
    for scenario in scenarios:
        group = str(scenario.get("scenario_group") or "unknown")
        interpretation = scenario["diagnostic_interpretation"]
        group_payload = group_summary[group]
        group_payload["target_replacement_reasons"].update(
            [str(interpretation["target_replacement_reason"])]
        )
        group_payload["iris_region_graph_signal_counts"].update(
            [str(interpretation["iris_region_graph_signal"])]
        )
        for source in interpretation["failure_sources"]:
            group_payload["failure_sources"].update([str(source)])

    return {
        "scenario_group_interpretation": {
            group: {
                "target_replacement_reasons": dict(sorted(payload["target_replacement_reasons"].items())),
                "failure_sources": dict(sorted(payload["failure_sources"].items())),
                "iris_region_graph_signal_counts": dict(
                    sorted(payload["iris_region_graph_signal_counts"].items())
                ),
            }
            for group, payload in sorted(group_summary.items())
        }
    }


def _empty_group_interpretation() -> dict[str, Counter[str]]:
    return {
        "target_replacement_reasons": Counter(),
        "failure_sources": Counter(),
        "iris_region_graph_signal_counts": Counter(),
    }


def _scenario_diagnostic_interpretation(scenario: dict[str, Any]) -> dict[str, Any]:
    candidates = scenario["path_feedback"]["candidates"]
    before_candidate = _candidate_by_cell(candidates, scenario["selected_cell_before_path_feedback"])
    after_candidate = _candidate_by_cell(candidates, scenario["selected_cell_after_path_feedback"])
    failure_sources = _scenario_failure_sources(scenario)
    return {
        "target_replacement_reason": _target_replacement_reason(
            scenario,
            before_candidate=before_candidate,
            after_candidate=after_candidate,
        ),
        "failure_sources": failure_sources,
        "primary_failure_reason": _primary_failure_reason(scenario),
        "iris_region_graph_signal": _iris_region_graph_signal(scenario),
        "selected_after_feasible": None if after_candidate is None else bool(after_candidate["reachable"]),
        "selected_after_replan_required": None
        if after_candidate is None
        else bool(after_candidate["replan_required"]),
        "open_grid_fallback_used": bool(scenario["open_grid_fallback_used"]),
    }


def _target_replacement_reason(
    scenario: dict[str, Any],
    *,
    before_candidate: dict[str, Any] | None,
    after_candidate: dict[str, Any] | None,
) -> str:
    if not scenario["selection_changed_by_path_feedback"]:
        return "unchanged"
    if after_candidate is None:
        return "no_feasible_candidate_after_path_feedback"
    before_flags = (
        before_candidate.get("diagnostic_interpretation", {}).get("diagnostic_flags", [])
        if before_candidate is not None
        else []
    )
    if "path_planning_failure" in before_flags:
        return "before_candidate_path_planning_failed"
    if "region_graph_disconnected" in before_flags:
        return "before_candidate_region_graph_disconnected"
    if "region_graph_fallback" in before_flags:
        return "before_candidate_region_graph_fallback"
    if "iris_fallback" in before_flags:
        return "before_candidate_iris_fallback"
    if "replan_required" in before_flags:
        return "before_candidate_replan_required"
    before_cost = scenario["selected_path_cost_before_feedback"]
    after_cost = scenario["selected_path_cost_after_feedback"]
    if before_cost is None:
        return "selected_before_not_evaluated_or_infeasible"
    if after_cost is not None and float(after_cost) < float(before_cost):
        return "lower_path_cost_candidate"
    return "path_feedback_tiebreak"


def _scenario_failure_sources(scenario: dict[str, Any]) -> list[str]:
    sources: list[str] = []
    feedback = scenario["path_feedback"]
    if int(feedback["failure_count"]) > 0:
        sources.append("path_planning_failure")
    if int(feedback["replan_count"]) > 0:
        sources.append("replan_required")
    if int(scenario["tracking_safety_violation_count"]) > 0:
        sources.append("tracking_safety_violation")
    if int(scenario["trajectory_optimization_fallback_count"]) > 0:
        sources.append("trajectory_optimization_fallback")
    if int(scenario["region_graph_disconnected_count"]) > 0:
        sources.append("region_graph_disconnected")
    if int(scenario["region_graph_diagnostics"]["fallback_count"]) > 0:
        sources.append("region_graph_fallback")
    if int(scenario["iris_diagnostics"]["fallback_count"]) > 0:
        sources.append("iris_fallback")
    if bool(scenario["open_grid_fallback_used"]):
        sources.append("open_grid_fallback")
    return sources or ["none"]


def _primary_failure_reason(scenario: dict[str, Any]) -> str | None:
    reasons = scenario["path_feedback"].get("failure_reasons", [])
    if not reasons:
        graph_reasons = scenario["region_graph_diagnostics"].get("fallback_reasons", {})
        if graph_reasons:
            return next(iter(graph_reasons))
        iris_reasons = scenario["iris_diagnostics"].get("fallback_reasons", {})
        if iris_reasons:
            return next(iter(iris_reasons))
        return None
    counts = Counter(str(reason) for reason in reasons)
    return counts.most_common(1)[0][0]


def _iris_region_graph_signal(scenario: dict[str, Any]) -> str:
    if (
        int(scenario["region_graph_disconnected_count"]) > 0
        or int(scenario["region_graph_diagnostics"]["fallback_count"]) > 0
        or int(scenario["iris_diagnostics"]["fallback_count"]) > 0
    ):
        return "diagnostic_explains_replan_or_failure"
    if scenario["region_graph_diagnostics"]["source_counts"] or int(scenario["iris_diagnostics"]["report_count"]) > 0:
        return "diagnostic_present"
    return "not_reported"


def _candidate_by_cell(candidates: list[dict[str, Any]], cell: list[int] | None) -> dict[str, Any] | None:
    if cell is None:
        return None
    for candidate in candidates:
        if candidate.get("cell") == cell:
            return candidate
    return None


def _list_text(values: Any) -> str:
    if not values:
        return "none"
    if isinstance(values, dict):
        return ", ".join(f"{key}:{value}" for key, value in values.items()) or "none"
    if isinstance(values, list | tuple | set):
        return ", ".join(str(value) for value in values) or "none"
    return str(values)


def _empty_group_summary() -> dict[str, int]:
    return {
        "scenario_count": 0,
        "candidate_count": 0,
        "reachable_count": 0,
        "failure_count": 0,
        "replan_count": 0,
        "selection_changed_count": 0,
        "iris_report_count": 0,
        "iris_fallback_count": 0,
        "region_graph_fallback_count": 0,
        "region_graph_start_goal_disconnected_count": 0,
    }


def _iris_diagnostics(evaluations) -> dict[str, Any]:
    status_counts: Counter[str] = Counter()
    fallback_reasons: Counter[str] = Counter()
    report_count = 0
    fallback_count = 0
    failure_count = 0
    region_count_total = 0
    for item in evaluations:
        iris = item.to_dict().get("iris_region")
        if not isinstance(iris, dict):
            continue
        report_count += 1
        status = str(iris.get("status") or "unknown")
        status_counts[status] += 1
        region_count_total += _int_value(iris.get("region_count"))
        if bool(iris.get("fallback_used")):
            fallback_count += 1
        if status == "failed":
            failure_count += 1
        reason = iris.get("failure_reason")
        if reason:
            fallback_reasons[str(reason)] += 1
    return {
        "report_count": report_count,
        "status_counts": dict(sorted(status_counts.items())),
        "fallback_count": fallback_count,
        "failure_count": failure_count,
        "region_count_total": region_count_total,
        "fallback_reasons": dict(sorted(fallback_reasons.items())),
    }


def _region_graph_diagnostics(evaluations) -> dict[str, Any]:
    source_counts: Counter[str] = Counter()
    fallback_reasons: Counter[str] = Counter()
    fallback_count = 0
    start_goal_disconnected_count = 0
    for item in evaluations:
        graph = item.to_dict().get("region_graph")
        if not isinstance(graph, dict):
            continue
        source_counts[str(graph.get("graph_source") or graph.get("region_source") or "unknown")] += 1
        if bool(graph.get("fallback_used")):
            fallback_count += 1
        if graph.get("start_goal_connected") is False:
            start_goal_disconnected_count += 1
        reason = graph.get("fallback_reason")
        if reason:
            fallback_reasons[str(reason)] += 1
    return {
        "source_counts": dict(sorted(source_counts.items())),
        "fallback_count": fallback_count,
        "fallback_reasons": dict(sorted(fallback_reasons.items())),
        "start_goal_disconnected_count": start_goal_disconnected_count,
    }


def _int_value(value: Any) -> int:
    if isinstance(value, bool) or value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


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
