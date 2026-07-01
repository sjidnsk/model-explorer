from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from .path_feedback_manifest import PathFeedbackScenario


GCS_CONTROL_POINT_CANDIDATE_TRIAGE_SCHEMA_VERSION = "gcs-control-point-candidate-triage-summary/v1"
GCS_CONTROL_POINT_CANDIDATE_ARTIFACT_INDEX_SCHEMA_VERSION = (
    "gcs-control-point-candidate-artifact-index/v1"
)
GCS_CONTROL_POINT_CANDIDATE_CALIBRATION_SWEEP_SCHEMA_VERSION = (
    "gcs-control-point-candidate-calibration-sweep/v1"
)
GCS_CONTROL_POINT_BACKEND = "pydrake_control_point_direction_cone_program"


def _int_value(value: Any) -> int:
    if isinstance(value, bool) or value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _float_value(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in {float("inf"), float("-inf")}:
        return None
    return number


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _numeric_metric_stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None}
    return {
        "count": len(values),
        "min": float(min(values)),
        "max": float(max(values)),
        "mean": float(sum(values) / len(values)),
    }


def _gcs_control_point_candidate_artifacts(
    evaluations,
    *,
    scenario: PathFeedbackScenario,
    artifact_root: Path | None,
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for item in evaluations:
        candidate = item.to_dict()
        gcs = candidate.get("gcs_trajectory")
        if not isinstance(gcs, dict) or gcs.get("backend") != GCS_CONTROL_POINT_BACKEND:
            continue
        metadata = item.result.metadata if isinstance(item.result.metadata, dict) else {}
        route_artifact: Path | None = None
        request_artifact: Path | None = None
        if artifact_root is not None:
            candidate_dir = (
                artifact_root
                / _safe_path_component(scenario.scenario_id)
                / f"action-{int(candidate['action_index']):03d}"
            )
            route_payload = _route_payload_from_metadata(metadata)
            if route_payload is not None:
                route_artifact = candidate_dir / "path-planner-route.json"
                _write_json(route_artifact, route_payload)
            request_payload = metadata.get("request_payload")
            if isinstance(request_payload, dict):
                request_artifact = candidate_dir / "path-planner-request.json"
                _write_json(request_artifact, request_payload)
        gcs_candidate = candidate.get("gcs_candidate")
        gcs_candidate = gcs_candidate if isinstance(gcs_candidate, dict) else {}
        entries.append(
            {
                "scenario_id": scenario.scenario_id,
                "scenario_group": scenario.scenario_group,
                "action_index": candidate["action_index"],
                "cell": candidate["cell"],
                "contract_json": str(scenario.contract_path),
                "sidecar_json": str(scenario.sidecar_path),
                "source_route_json": metadata.get("route_json"),
                "route_artifact": None if route_artifact is None else str(route_artifact),
                "request_artifact": None if request_artifact is None else str(request_artifact),
                "backend": gcs.get("backend"),
                "candidate_selected": gcs_candidate.get("selected"),
                "candidate_fallback_reason": gcs_candidate.get("fallback_reason"),
            }
        )
    return _artifact_index_payload(entries, artifact_root=artifact_root)


def _gcs_control_point_candidate_artifact_index(
    scenarios: list[dict[str, Any]],
    *,
    artifact_root: Path | None,
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for scenario in scenarios:
        artifact_payload = scenario.get("gcs_control_point_candidate_artifacts")
        artifact_payload = artifact_payload if isinstance(artifact_payload, dict) else {}
        scenario_entries = artifact_payload.get("entries")
        if isinstance(scenario_entries, list):
            entries.extend(entry for entry in scenario_entries if isinstance(entry, dict))
    return _artifact_index_payload(entries, artifact_root=artifact_root)


def _artifact_index_payload(entries: list[dict[str, Any]], *, artifact_root: Path | None) -> dict[str, Any]:
    return {
        "schema_version": GCS_CONTROL_POINT_CANDIDATE_ARTIFACT_INDEX_SCHEMA_VERSION,
        "artifact_root": None if artifact_root is None else str(artifact_root),
        "candidate_count": len(entries),
        "route_artifact_count": sum(1 for entry in entries if entry.get("route_artifact")),
        "entries": entries,
    }


def _gcs_control_point_candidate_triage_summary(
    scenarios: list[dict[str, Any]],
    *,
    artifact_index: dict[str, Any],
) -> dict[str, Any]:
    artifact_by_candidate = {
        (entry.get("scenario_id"), entry.get("action_index")): entry
        for entry in artifact_index.get("entries", [])
        if isinstance(entry, dict)
    }
    fallback_reason_counts: Counter[str] = Counter()
    terrain_objective_source_counts: Counter[str] = Counter()
    blocker_class_counts: Counter[str] = Counter()
    sampled_terrain_costs: list[float] = []
    high_cost_exposure_deltas: list[float] = []
    rows: list[dict[str, Any]] = []
    attempted_count = 0
    success_count = 0
    selected_count = 0
    for scenario in scenarios:
        feedback = scenario.get("path_feedback")
        feedback = feedback if isinstance(feedback, dict) else {}
        candidates = feedback.get("candidates")
        if not isinstance(candidates, list):
            continue
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            gcs = candidate.get("gcs_trajectory")
            if not isinstance(gcs, dict) or gcs.get("backend") != GCS_CONTROL_POINT_BACKEND:
                continue
            gcs_candidate = candidate.get("gcs_candidate")
            gcs_candidate = gcs_candidate if isinstance(gcs_candidate, dict) else {}
            motion = candidate.get("gcs_motion_feasibility")
            motion = motion if isinstance(motion, dict) else {}
            trajectory_cost = gcs.get("cost_summary")
            trajectory_cost = trajectory_cost if isinstance(trajectory_cost, dict) else {}
            candidate_cost = gcs_candidate.get("cost_summary")
            candidate_cost = candidate_cost if isinstance(candidate_cost, dict) else {}
            direction_cone = gcs.get("constraint_summary")
            direction_cone = direction_cone if isinstance(direction_cone, dict) else {}
            artifact = artifact_by_candidate.get((scenario.get("scenario_id"), candidate.get("action_index")), {})
            row = _gcs_control_point_triage_row(
                scenario,
                candidate,
                gcs,
                gcs_candidate,
                motion,
                trajectory_cost,
                candidate_cost,
                direction_cone,
                artifact if isinstance(artifact, dict) else {},
            )
            rows.append(row)
            if row["attempted"] is True:
                attempted_count += 1
            if row["success"] is True:
                success_count += 1
            if row["candidate_selected"] is True:
                selected_count += 1
            fallback_reason = row["candidate_fallback_reason"]
            if fallback_reason:
                fallback_reason_counts[str(fallback_reason)] += 1
            terrain_source = row["terrain_objective_source"]
            if terrain_source:
                terrain_objective_source_counts[str(terrain_source)] += 1
            blocker_class_counts[_control_point_blocker_class(row)] += 1
            sampled_terrain_cost = _float_value(row["sampled_terrain_cost"])
            if sampled_terrain_cost is not None:
                sampled_terrain_costs.append(sampled_terrain_cost)
            exposure_delta = _float_value(row["high_cost_exposure_delta_vs_baseline"])
            if exposure_delta is not None:
                high_cost_exposure_deltas.append(exposure_delta)
    sampled_stats = _numeric_metric_stats(sampled_terrain_costs)
    exposure_stats = _numeric_metric_stats(high_cost_exposure_deltas)
    return {
        "schema_version": GCS_CONTROL_POINT_CANDIDATE_TRIAGE_SCHEMA_VERSION,
        "candidate_count": len(rows),
        "attempted_count": attempted_count,
        "success_count": success_count,
        "selected_count": selected_count,
        "route_artifact_count": int(artifact_index.get("route_artifact_count") or 0),
        "fallback_reason_counts": dict(sorted(fallback_reason_counts.items())),
        "terrain_objective_source_counts": dict(sorted(terrain_objective_source_counts.items())),
        "blocker_class_counts": dict(sorted(blocker_class_counts.items())),
        "sampled_terrain_cost": sampled_stats,
        "high_cost_exposure_delta_vs_baseline": exposure_stats,
        "calibration_sweep": _control_point_calibration_sweep(
            rows,
            fallback_reason_counts=fallback_reason_counts,
        ),
        "interpretation": _control_point_triage_interpretation(
            candidate_count=len(rows),
            success_count=success_count,
            selected_count=selected_count,
            fallback_reason_counts=fallback_reason_counts,
        ),
        "candidates": rows,
    }


def _gcs_control_point_triage_row(
    scenario: dict[str, Any],
    candidate: dict[str, Any],
    gcs: dict[str, Any],
    gcs_candidate: dict[str, Any],
    motion: dict[str, Any],
    trajectory_cost: dict[str, Any],
    candidate_cost: dict[str, Any],
    direction_cone: dict[str, Any],
    artifact: dict[str, Any],
) -> dict[str, Any]:
    return {
        "scenario_id": scenario.get("scenario_id"),
        "scenario_group": scenario.get("scenario_group"),
        "action_index": candidate.get("action_index"),
        "cell": candidate.get("cell"),
        "backend": gcs.get("backend"),
        "attempted": gcs.get("attempted"),
        "success": gcs.get("success"),
        "reason": gcs.get("reason"),
        "candidate_selected": gcs_candidate.get("selected"),
        "candidate_fallback_reason": gcs_candidate.get("fallback_reason"),
        "selection_reason": gcs_candidate.get("selection_reason"),
        "cost_delta_vs_baseline": gcs_candidate.get("cost_delta_vs_baseline"),
        "cost_delta_vs_postprocess": gcs_candidate.get("cost_delta_vs_postprocess"),
        "baseline_overlap_ratio": gcs_candidate.get("baseline_overlap_ratio"),
        "terrain_objective_source": _first_present(
            trajectory_cost.get("terrain_objective_source"),
            candidate_cost.get("terrain_objective_source"),
        ),
        "terrain_objective_weight": _first_present(
            trajectory_cost.get("terrain_objective_weight"),
            candidate_cost.get("terrain_objective_weight"),
        ),
        "sampled_terrain_cost": _first_present(
            trajectory_cost.get("sampled_terrain_cost"),
            candidate_cost.get("sampled_terrain_cost"),
        ),
        "control_point_terrain_cost": _first_present(
            trajectory_cost.get("control_point_terrain_cost"),
            candidate_cost.get("control_point_terrain_cost"),
        ),
        "high_cost_exposure": _first_present(
            candidate_cost.get("high_cost_exposure"),
            trajectory_cost.get("high_cost_exposure"),
            gcs_candidate.get("high_cost_exposure"),
        ),
        "baseline_high_cost_exposure": candidate_cost.get("baseline_high_cost_exposure"),
        "high_cost_exposure_delta_vs_baseline": candidate_cost.get(
            "high_cost_exposure_delta_vs_baseline"
        ),
        "direction_cone_evaluated": direction_cone.get("evaluated"),
        "direction_cone_backend_enforced": direction_cone.get("backend_enforced"),
        "direction_cone_violation_count": _int_value(direction_cone.get("violation_count")),
        "direction_cone_eta": _float_value(direction_cone.get("eta")),
        "direction_cone_rho_min": _float_value(direction_cone.get("rho_min")),
        "direction_cone_tolerance_deg": _float_value(
            direction_cone.get("max_allowed_direction_error_deg")
        ),
        "direction_cone_constraint_tightness_min": _float_value(
            direction_cone.get("constraint_tightness_min")
        ),
        "direction_cone_risk_flags": (
            direction_cone.get("risk_flags") if isinstance(direction_cone.get("risk_flags"), list) else []
        ),
        "direction_cone_rho_source_counts": (
            direction_cone.get("rho_source_counts")
            if isinstance(direction_cone.get("rho_source_counts"), dict)
            else {}
        ),
        "second_difference_weight": _objective_term_weight(
            direction_cone,
            "control_point_second_difference_quadratic",
        ),
        "motion_feasibility_status": motion.get("feasibility_status"),
        "motion_feasibility_fallback_reason": motion.get("fallback_reason"),
        "motion_feasibility_curvature_violation_count": _int_value(
            motion.get("curvature_violation_count")
        ),
        "motion_feasibility_heading_violation_count": _int_value(
            motion.get("heading_violation_count")
        ),
        "route_artifact": artifact.get("route_artifact"),
        "request_artifact": artifact.get("request_artifact"),
        "source_route_json": artifact.get("source_route_json"),
    }


def _objective_term_weight(summary: dict[str, Any], name: str) -> float | None:
    weights = summary.get("objective_term_weights")
    if not isinstance(weights, dict):
        return None
    return _float_value(weights.get(name))


def _control_point_calibration_sweep(
    rows: list[dict[str, Any]],
    *,
    fallback_reason_counts: Counter[str],
) -> dict[str, Any]:
    quality_blocked = [row for row in rows if row.get("candidate_fallback_reason") == "cost_dominated"]
    direction_blocked = [
        row
        for row in rows
        if row.get("candidate_fallback_reason") == "direction_cone_constraint_violation"
        or _int_value(row.get("direction_cone_violation_count")) > 0
        or bool(row.get("direction_cone_risk_flags"))
    ]
    unsupported = [
        row for row in rows if row.get("candidate_fallback_reason") == "unsupported_route_replacement"
    ]
    conservative_gate_candidates = [
        row
        for row in quality_blocked
        if _non_positive_number(row.get("cost_delta_vs_baseline"))
        and _non_positive_number(row.get("high_cost_exposure_delta_vs_baseline"))
        and row.get("motion_feasibility_status") in {None, "feasible"}
        and _int_value(row.get("direction_cone_violation_count")) == 0
        and not row.get("direction_cone_risk_flags")
    ]
    default_change_reason = "no_control_point_candidates_reported"
    if rows:
        default_change_reason = "requires_solver_rerun_and_no_safety_diagnostic_degradation"
        if quality_blocked or direction_blocked:
            default_change_reason = "recorded_candidates_remain_blocked_by_quality_or_direction_cone_gate"
    return {
        "schema_version": GCS_CONTROL_POINT_CANDIDATE_CALIBRATION_SWEEP_SCHEMA_VERSION,
        "mode": "recorded_candidate_gate_diagnostics",
        "solver_rerun_required": True,
        "default_change_recommended": False,
        "default_change_reason": default_change_reason,
        "sweep_dimensions": [
            "terrain_objective_weight",
            "control_point_second_difference_quadratic_weight",
            "direction_cone_rho_eta_tolerance",
            "quality_gate_thresholds",
        ],
        "observed_current_values": {
            "terrain_objective_weight": _unique_numeric_values(rows, "terrain_objective_weight"),
            "second_difference_weight": _unique_numeric_values(rows, "second_difference_weight"),
            "direction_cone_eta": _unique_numeric_values(rows, "direction_cone_eta"),
            "direction_cone_rho_min": _unique_numeric_values(rows, "direction_cone_rho_min"),
            "direction_cone_tolerance_deg": _unique_numeric_values(rows, "direction_cone_tolerance_deg"),
            "direction_cone_rho_source_counts": _aggregate_row_counter(
                rows,
                "direction_cone_rho_source_counts",
            ),
        },
        "candidate_gate_outcomes": {
            "quality_gate_blocked_count": len(quality_blocked),
            "direction_cone_blocked_count": len(direction_blocked),
            "expected_not_evaluated_count": len(unsupported),
            "fallback_reason_counts": dict(sorted(fallback_reason_counts.items())),
            "conservative_gate_relaxation_candidate_count": len(conservative_gate_candidates),
            "unsafe_or_unproven_quality_relaxation_count": max(
                0,
                len(quality_blocked) - len(conservative_gate_candidates),
            ),
        },
        "safety_regression_guard": {
            "terrain_cost_degradation_allowed": False,
            "high_cost_exposure_degradation_allowed": False,
            "collision_degradation_allowed": False,
            "direction_cone_degradation_allowed": False,
            "motion_diagnostic_degradation_allowed": False,
            "default_gate_relaxation_allowed_without_evidence": False,
        },
        "next_solver_rerun_matrix": [
            {
                "dimension": "terrain_objective_weight",
                "target_blocker": "cost_dominated",
                "acceptance": "lower_sampled_terrain_cost_and_high_cost_exposure_without_collision_or_direction_cone_regression",
            },
            {
                "dimension": "control_point_second_difference_quadratic_weight",
                "target_blocker": "cost_dominated_or_motion_diagnostic_regression",
                "acceptance": "smoother_control_points_without_region_or_motion_feasibility_regression",
            },
            {
                "dimension": "direction_cone_rho_eta_tolerance",
                "target_blocker": "direction_cone_constraint_violation",
                "acceptance": "fewer_direction_cone_violations_without_motion_or_collision_regression",
            },
            {
                "dimension": "quality_gate_thresholds",
                "target_blocker": "candidate_selected_count_zero",
                "acceptance": "selected_count_can_increase_only_when_cost_and_safety_metrics_do_not_degrade",
            },
        ],
    }


def _non_positive_number(value: Any) -> bool:
    number = _float_value(value)
    return number is not None and number <= 0.0


def _unique_numeric_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    values = {_float_value(row.get(key)) for row in rows}
    return sorted(value for value in values if value is not None)


def _aggregate_row_counter(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        payload = row.get(key)
        if not isinstance(payload, dict):
            continue
        for item_key, value in payload.items():
            if isinstance(value, int) and not isinstance(value, bool):
                counts[str(item_key)] += value
    return dict(sorted(counts.items()))


def _control_point_blocker_class(row: dict[str, Any]) -> str:
    if row.get("candidate_selected") is True:
        return "selected"
    fallback_reason = row.get("candidate_fallback_reason")
    if fallback_reason == "unsupported_route_replacement":
        return "expected_not_evaluated_unreachable_or_unsupported"
    if fallback_reason == "cost_dominated":
        return "quality_gate_cost_or_high_cost_exposure"
    if fallback_reason == "direction_cone_constraint_violation":
        return "direction_cone_constraint"
    if row.get("success") is not True:
        return "solver_or_region_sequence_not_successful"
    if fallback_reason:
        return str(fallback_reason)
    return "not_selected_without_reason"


def _control_point_triage_interpretation(
    *,
    candidate_count: int,
    success_count: int,
    selected_count: int,
    fallback_reason_counts: Counter[str],
) -> str:
    if candidate_count == 0:
        return "no_control_point_candidates_reported"
    if selected_count == 0 and success_count > 0:
        if fallback_reason_counts:
            return "solved_candidates_blocked_by_quality_or_direction_cone_gate"
        return "solved_candidates_not_selected_without_fallback_reason"
    if selected_count > 0:
        return "some_control_point_candidates_selected"
    return "control_point_candidates_not_solved_or_not_evaluated"


def _route_payload_from_metadata(metadata: dict[str, Any]) -> dict[str, Any] | None:
    route_payload = metadata.get("route_payload")
    if isinstance(route_payload, dict):
        return route_payload
    route_json = metadata.get("route_json")
    if route_json is None:
        return None
    path = Path(str(route_json))
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _safe_path_component(value: Any) -> str:
    text = str(value)
    safe = "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in text)
    return safe or "unknown"


gcs_control_point_candidate_artifacts = _gcs_control_point_candidate_artifacts
gcs_control_point_candidate_artifact_index = _gcs_control_point_candidate_artifact_index
artifact_index_payload = _artifact_index_payload
gcs_control_point_candidate_triage_summary = _gcs_control_point_candidate_triage_summary
control_point_calibration_sweep = _control_point_calibration_sweep

__all__ = [
    'GCS_CONTROL_POINT_CANDIDATE_TRIAGE_SCHEMA_VERSION',
    'GCS_CONTROL_POINT_CANDIDATE_ARTIFACT_INDEX_SCHEMA_VERSION',
    'GCS_CONTROL_POINT_CANDIDATE_CALIBRATION_SWEEP_SCHEMA_VERSION',
    'GCS_CONTROL_POINT_BACKEND',
    'gcs_control_point_candidate_artifacts',
    'gcs_control_point_candidate_artifact_index',
    'artifact_index_payload',
    'gcs_control_point_candidate_triage_summary',
    'control_point_calibration_sweep',
    '_gcs_control_point_candidate_artifacts',
    '_gcs_control_point_candidate_artifact_index',
    '_artifact_index_payload',
    '_gcs_control_point_candidate_triage_summary',
    '_control_point_calibration_sweep',
]
