from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..core.interfaces import ModelExplorerContract
from ..io.scenario import load_scenario
from .context_id import policy_context_id_metadata
from .feedback_selection_anchor import (
    _anchor_projection_candidate_generation_summary,
    annotate_source_selected_anchor_projection,
)
from .feedback_selection_sources import (
    _candidate_path_cost_for_cell,
    _path_cost_delta,
    _selected_after_feedback,
    _selected_before_feedback,
)
from .path_feedback_artifacts import (
    _gcs_control_point_candidate_artifact_index,
    _gcs_control_point_candidate_artifacts,
    _gcs_control_point_candidate_triage_summary,
)
from .path_feedback_backend_diagnostics import (
    _channel_aware_astar_diagnostics,
    _convex_region_diagnostics,
    _gcs_candidate_diagnostics,
    _gcs_control_point_diagnostics,
    _gcs_curvature_constrained_diagnostics,
    _gcs_motion_feasibility_diagnostics,
    _gcs_trajectory_diagnostics,
    _iris_diagnostics,
    _region_graph_diagnostics,
    _sampled_region_path_diagnostics,
)
from .path_feedback_candidate_audits import (
    _convex_region_candidate_audit,
    _gcs_candidate_audit,
    _gcs_control_point_candidate_audit,
    _gcs_curvature_constrained_audit,
    _gcs_motion_feasibility_audit,
    _gcs_trajectory_candidate_audit,
    _sampled_region_path_candidate_audit,
)
from .path_feedback_diagnostic_aggregate import (
    _diagnostic_aggregate,
    _open_grid_fallback_used,
    _region_graph_disconnected_count,
    _tracking_safety_violation_count,
    _trajectory_optimization_fallback_count,
)
from .path_feedback_diagnostic_interpretation import (
    _diagnostic_interpretation_summary,
    _scenario_diagnostic_interpretation,
)
from .path_feedback_manifest import (
    PATH_FEEDBACK_SCHEMA_VERSION,
    PathFeedbackManifest,
    PathFeedbackScenario,
    cell_to_list as _cell_to_list,
    load_path_feedback_manifest,
)
from .path_feedback_reports import render_path_feedback_markdown
from .path_feedback_summary import (
    PATH_FEEDBACK_SUMMARY_SCHEMA_VERSION,
    PATH_FEEDBACK_SUMMARY_ACCEPTANCE_METRICS,
    PATH_FEEDBACK_SUMMARY_REQUIRED_KEYS,
    _acceptance_metadata,
    compact_path_feedback_summary,
    validate_path_feedback_summary_contract,
)
from .planning_adapters import PathPlannerRouteAdapter, planner_from_config
from .planning_anchor_evaluation import evaluate_candidate_paths
from .planning_anchor_projection import anchor_projection_candidate_config_from_mapping
from .planning_backend_summaries import path_feedback_summary
from .planning_routes import load_path_planner_sidecar
from .planning_types import (
    AnchorProjectionCandidateConfig,
    PathPlanRequest,
    PathPlanResult,
    PathPlanningAdapter,
)


def dry_run_path_feedback_manifest(path: str | Path) -> dict[str, Any]:
    manifest = load_path_feedback_manifest(path)
    return {
        "status": "dry_run",
        "schema_version": manifest.schema_version,
        "scenario_count": len(manifest.scenarios),
        "scenario_set": manifest.scenario_set,
        "diagnostic_profile": manifest.diagnostic_profile,
        "acceptance_gate": manifest.acceptance_gate,
        "top_k": manifest.top_k,
        "python_executable": manifest.python_executable,
        "planner_extra_args": list(manifest.planner_extra_args),
        "planner": str(manifest.planner_config.get("backend", "path_planner_route")),
        "scenarios": [
            {
                "scenario_id": scenario.scenario_id,
                "scenario_group": scenario.scenario_group,
                "scenario_seed": scenario.scenario_seed,
                "scenario_variant_id": scenario.scenario_variant_id,
                "contract": str(scenario.contract_path),
                "sidecar": str(scenario.sidecar_path),
                "route_fixture_count": len(scenario.route_fixtures),
            }
            for scenario in manifest.scenarios
        ],
        "would_write": [
            str(path)
            for path in (
                manifest.summary_output,
                manifest.report_output,
                manifest.gcs_control_point_candidate_artifact_output,
            )
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
    gcs_control_point_artifacts = _gcs_control_point_candidate_artifact_index(
        scenario_summaries,
        artifact_root=manifest.gcs_control_point_candidate_artifact_output,
    )
    gcs_control_point_triage = _gcs_control_point_candidate_triage_summary(
        scenario_summaries,
        artifact_index=gcs_control_point_artifacts,
    )
    open_grid_fallback_used = any(bool(item["open_grid_fallback_used"]) for item in scenario_summaries)
    acceptance_metadata = _acceptance_metadata(
        manifest,
        open_grid_fallback_used=open_grid_fallback_used,
    )
    anchor_projection_candidate_generation = _anchor_projection_candidate_generation_summary(
        scenario_summaries
    )
    return {
        "schema_version": PATH_FEEDBACK_SUMMARY_SCHEMA_VERSION,
        "scenario_count": len(scenario_summaries),
        "scenario_set": manifest.scenario_set,
        "diagnostic_profile": manifest.diagnostic_profile,
        "acceptance_gate": manifest.acceptance_gate,
        "top_k": manifest.top_k,
        "planner_extra_args": list(manifest.planner_extra_args),
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
        "open_grid_fallback_used": open_grid_fallback_used,
        "open_grid_fallback_used_gate": acceptance_metadata["open_grid_fallback_used_gate"],
        "acceptance_metadata": acceptance_metadata,
        **anchor_projection_candidate_generation,
        "failure_reasons": [
            reason
            for item in scenario_summaries
            for reason in item["path_feedback"]["failure_reasons"]
        ],
        **diagnostic_summary,
        "gcs_control_point_candidate_artifacts": gcs_control_point_artifacts,
        "gcs_control_point_candidate_triage": gcs_control_point_triage,
        "diagnostic_interpretation": diagnostic_interpretation,
        "scenarios": scenario_summaries,
    }


def _run_feedback_scenario(
    scenario: PathFeedbackScenario,
    *,
    manifest: PathFeedbackManifest,
) -> dict[str, Any]:
    contract = load_scenario(scenario.contract_path).snapshots[0]
    planner = _planner_for_scenario(scenario, manifest=manifest)
    anchor_projection_candidate_config = _anchor_projection_candidate_config(manifest)
    evaluations = evaluate_candidate_paths(
        contract,
        current_cell=scenario.current_cell,
        top_k=manifest.top_k,
        planner=planner,
        anchor_projection_candidate_config=anchor_projection_candidate_config,
    )
    selected_before = _selected_before_feedback(contract)
    selected_after = _selected_after_feedback(
        evaluations,
        anchor_projection_candidate_config=anchor_projection_candidate_config,
    )
    feedback = annotate_source_selected_anchor_projection(
        path_feedback_summary(evaluations),
        selected_evaluation=selected_after,
        anchor_projection_candidate_config=anchor_projection_candidate_config,
    )
    _annotate_policy_context_ids(feedback, scenario=scenario, manifest=manifest)
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
        "scenario_seed": scenario.scenario_seed,
        "scenario_variant_id": scenario.scenario_variant_id,
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
        "convex_region_diagnostics": _convex_region_diagnostics(evaluations),
        "convex_region_candidate_audit": _convex_region_candidate_audit(
            evaluations,
            scenario_id=scenario.scenario_id,
        ),
        "gcs_trajectory_diagnostics": _gcs_trajectory_diagnostics(evaluations),
        "gcs_trajectory_candidate_audit": _gcs_trajectory_candidate_audit(
            evaluations,
            scenario_id=scenario.scenario_id,
        ),
        "gcs_candidate_diagnostics": _gcs_candidate_diagnostics(evaluations),
        "gcs_candidate_audit": _gcs_candidate_audit(
            evaluations,
            scenario_id=scenario.scenario_id,
        ),
        "gcs_control_point_diagnostics": _gcs_control_point_diagnostics(evaluations),
        "gcs_control_point_candidate_audit": _gcs_control_point_candidate_audit(
            evaluations,
            scenario_id=scenario.scenario_id,
        ),
        "gcs_control_point_candidate_artifacts": _gcs_control_point_candidate_artifacts(
            evaluations,
            scenario=scenario,
            artifact_root=manifest.gcs_control_point_candidate_artifact_output,
        ),
        "gcs_motion_feasibility_diagnostics": _gcs_motion_feasibility_diagnostics(evaluations),
        "gcs_motion_feasibility_audit": _gcs_motion_feasibility_audit(
            evaluations,
            scenario_id=scenario.scenario_id,
        ),
        "gcs_curvature_constrained_diagnostics": _gcs_curvature_constrained_diagnostics(evaluations),
        "gcs_curvature_constrained_audit": _gcs_curvature_constrained_audit(
            evaluations,
            scenario_id=scenario.scenario_id,
        ),
        "channel_aware_astar_diagnostics": _channel_aware_astar_diagnostics(
            evaluations,
            scenario_id=scenario.scenario_id,
        ),
        "sampled_region_path_diagnostics": _sampled_region_path_diagnostics(evaluations),
        "sampled_region_path_candidate_audit": _sampled_region_path_candidate_audit(
            evaluations,
            scenario_id=scenario.scenario_id,
        ),
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


def _annotate_policy_context_ids(
    feedback: dict[str, Any],
    *,
    scenario: PathFeedbackScenario,
    manifest: PathFeedbackManifest,
) -> None:
    candidates = feedback.get("candidates")
    if not isinstance(candidates, list):
        return
    planning_backend = _policy_context_planning_backend(manifest)
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        source_action_index = candidate.get("source_action_index")
        if source_action_index is None:
            source_action_index = candidate.get("action_index")
        candidate_role = str(candidate.get("candidate_role") or "policy_target")
        fields = {
            "scenario_id": scenario.scenario_id,
            "scenario_group": scenario.scenario_group,
            "scenario_seed": scenario.scenario_seed,
            "scenario_variant_id": scenario.scenario_variant_id,
            "diagnostic_profile": manifest.diagnostic_profile,
            "planning_backend": planning_backend,
            "top_k": manifest.top_k,
            "sample_type": "path_feedback_candidate",
            "candidate_role": candidate_role,
            "source_action_index": source_action_index,
            "policy_target_cell": candidate.get("policy_target_cell") or candidate.get("cell"),
            "execution_goal_cell": candidate.get("execution_goal_cell") or candidate.get("cell"),
            "target_binding_mode": candidate.get("target_binding_mode") or candidate_role,
        }
        candidate.update(policy_context_id_metadata(fields))


def _policy_context_planning_backend(manifest: PathFeedbackManifest) -> str:
    extra_args = list(manifest.planner_extra_args)
    for index, value in enumerate(extra_args):
        if value == "--planning-backend" and index + 1 < len(extra_args):
            return str(extra_args[index + 1])
    return str(manifest.planner_config.get("backend") or "path_planner_route")


def _anchor_projection_candidate_config(manifest: PathFeedbackManifest) -> AnchorProjectionCandidateConfig:
    return anchor_projection_candidate_config_from_mapping(
        manifest.planner_config.get("anchor_projection_candidate_generation")
    )


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


def _numeric_observation(contract: ModelExplorerContract, field: str) -> float:
    value = contract.observation_update.get(field)
    if isinstance(value, bool) or value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


class _RouteFixturePlanner:
    def __init__(self, sidecar_path: Path, route_fixtures: dict[int, Path]) -> None:
        self._sidecar = load_path_planner_sidecar(sidecar_path)
        self._route_fixtures = dict(route_fixtures)

    def plan(self, request: PathPlanRequest) -> PathPlanResult:
        route_path = self._route_fixtures.get(request.action_index)
        if route_path is None:
            raise ValueError(f"missing route fixture for action_index={request.action_index}")
        return PathPlannerRouteAdapter(sidecar=self._sidecar, route_json=route_path).plan(request)


__all__ = (
    "ModelExplorerContract",
    "policy_context_id_metadata",
    "_anchor_projection_candidate_generation_summary",
    "annotate_source_selected_anchor_projection",
    "_candidate_path_cost_for_cell",
    "_path_cost_delta",
    "_selected_after_feedback",
    "_selected_before_feedback",
    "_gcs_control_point_candidate_artifact_index",
    "_gcs_control_point_candidate_artifacts",
    "_gcs_control_point_candidate_triage_summary",
    "_channel_aware_astar_diagnostics",
    "_convex_region_diagnostics",
    "_gcs_candidate_diagnostics",
    "_gcs_control_point_diagnostics",
    "_gcs_curvature_constrained_diagnostics",
    "_gcs_motion_feasibility_diagnostics",
    "_gcs_trajectory_diagnostics",
    "_iris_diagnostics",
    "_region_graph_diagnostics",
    "_sampled_region_path_diagnostics",
    "_convex_region_candidate_audit",
    "_gcs_candidate_audit",
    "_gcs_control_point_candidate_audit",
    "_gcs_curvature_constrained_audit",
    "_gcs_motion_feasibility_audit",
    "_gcs_trajectory_candidate_audit",
    "_sampled_region_path_candidate_audit",
    "_diagnostic_aggregate",
    "_open_grid_fallback_used",
    "_region_graph_disconnected_count",
    "_tracking_safety_violation_count",
    "_trajectory_optimization_fallback_count",
    "_diagnostic_interpretation_summary",
    "_scenario_diagnostic_interpretation",
    "PATH_FEEDBACK_SCHEMA_VERSION",
    "PathFeedbackManifest",
    "PathFeedbackScenario",
    "_cell_to_list",
    "load_path_feedback_manifest",
    "render_path_feedback_markdown",
    "PATH_FEEDBACK_SUMMARY_SCHEMA_VERSION",
    "PATH_FEEDBACK_SUMMARY_ACCEPTANCE_METRICS",
    "PATH_FEEDBACK_SUMMARY_REQUIRED_KEYS",
    "_acceptance_metadata",
    "compact_path_feedback_summary",
    "validate_path_feedback_summary_contract",
    "PathPlannerRouteAdapter",
    "planner_from_config",
    "evaluate_candidate_paths",
    "anchor_projection_candidate_config_from_mapping",
    "path_feedback_summary",
    "load_path_planner_sidecar",
    "AnchorProjectionCandidateConfig",
    "PathPlanRequest",
    "PathPlanResult",
    "PathPlanningAdapter",
    "dry_run_path_feedback_manifest",
    "validate_path_feedback_manifest",
    "run_path_feedback_manifest",
    "run_path_feedback",
    "_run_feedback_scenario",
    "_annotate_policy_context_ids",
    "_policy_context_planning_backend",
    "_anchor_projection_candidate_config",
    "_planner_for_scenario",
    "_numeric_observation",
    "_RouteFixturePlanner",
)
