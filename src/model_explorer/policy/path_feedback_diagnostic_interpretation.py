from __future__ import annotations
from collections import Counter, defaultdict
from typing import Any
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
    if int(scenario["sampled_region_path_diagnostics"]["fallback_count"]) > 0:
        sources.append("sampled_region_path_fallback")
    if scenario["gcs_curvature_constrained_diagnostics"].get("fallback_reason_counts"):
        sources.append("gcs_curvature_constrained_fallback")
    if scenario["gcs_control_point_diagnostics"].get("candidate_fallback_reason_counts"):
        sources.append("gcs_control_point_fallback")
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
        sampled_reasons = scenario["sampled_region_path_diagnostics"].get("fallback_reasons", {})
        if sampled_reasons:
            return next(iter(sampled_reasons))
        constrained_reasons = scenario["gcs_curvature_constrained_diagnostics"].get("fallback_reason_counts", {})
        if constrained_reasons:
            return next(iter(constrained_reasons))
        return None
    counts = Counter(str(reason) for reason in reasons)
    return counts.most_common(1)[0][0]


def _iris_region_graph_signal(scenario: dict[str, Any]) -> str:
    if (
        int(scenario["region_graph_disconnected_count"]) > 0
        or int(scenario["region_graph_diagnostics"]["fallback_count"]) > 0
        or int(scenario["iris_diagnostics"]["fallback_count"]) > 0
        or int(scenario["sampled_region_path_diagnostics"]["fallback_count"]) > 0
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
__all__ = (
    "_diagnostic_interpretation_summary",
    "_empty_group_interpretation",
    "_scenario_diagnostic_interpretation",
    "_target_replacement_reason",
    "_scenario_failure_sources",
    "_primary_failure_reason",
    "_iris_region_graph_signal",
    "_candidate_by_cell",
    "_list_text",
)
