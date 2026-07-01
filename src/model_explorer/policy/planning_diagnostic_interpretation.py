"""Interpret planning diagnostic summaries into ordered flags."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def _input_source_summary(value: Any) -> dict[str, Any]:
    metadata = value.get("metadata") if isinstance(value, dict) else {}
    metadata = metadata if isinstance(metadata, dict) else {}
    cost_source = metadata.get("cost_source")
    passable_mask_source = metadata.get("passable_mask_source")
    return {
        "cost_source": cost_source,
        "passable_mask_source": passable_mask_source,
        "open_grid_fallback_used": (
            cost_source == "open_grid_fallback"
            or passable_mask_source == "open_grid_fallback"
        ),
    }


def _candidate_diagnostic_interpretation(item: dict[str, Any]) -> dict[str, Any]:
    flags: list[str] = []
    if item.get("failure_reason") is not None or not bool(item.get("reachable")):
        flags.append("path_planning_failure")
    if bool(item.get("replan_required")):
        flags.append("replan_required")
    if bool(item.get("open_grid_fallback_used")):
        flags.append("open_grid_fallback")

    optimization = item.get("trajectory_optimization")
    if isinstance(optimization, dict):
        fallback = optimization.get("fallback_status")
        if isinstance(fallback, str) and fallback not in {"ok", "not_needed", "none"}:
            flags.append("trajectory_optimization_fallback")

    postprocess = item.get("postprocess")
    if isinstance(postprocess, dict) and int(postprocess.get("tracking_safety_violation_count") or 0) > 0:
        flags.append("tracking_safety_violation")

    iris = item.get("iris_region")
    iris_status = None
    iris_fallback_used = False
    if isinstance(iris, dict):
        iris_status = iris.get("status")
        iris_fallback_used = bool(iris.get("fallback_used"))
        if iris_fallback_used:
            flags.append("iris_fallback")
        if iris_status == "failed":
            flags.append("iris_failure")

    graph = item.get("region_graph")
    graph_source = None
    graph_fallback_used = False
    graph_connected = None
    if isinstance(graph, dict):
        graph_source = graph.get("graph_source") or graph.get("region_source")
        graph_fallback_used = bool(graph.get("fallback_used"))
        graph_connected = graph.get("start_goal_connected")
        if graph_fallback_used:
            flags.append("region_graph_fallback")
        if graph_connected is False:
            flags.append("region_graph_disconnected")

    sampled_status = None
    sampled_fallback_reason = None
    planning_backend = item.get("planning_backend")
    if isinstance(planning_backend, dict):
        sampled = planning_backend.get("sampled_region_path")
        if isinstance(sampled, dict) and sampled:
            sampled_status = sampled.get("status")
            sampled_fallback_reason = sampled.get("fallback_reason")
            if sampled_status == "fallback" or sampled_fallback_reason:
                flags.append("sampled_region_path_fallback")

    ordered_flags = _dedupe(flags)
    return {
        "primary_source": _primary_diagnostic_source(ordered_flags),
        "diagnostic_flags": ordered_flags,
        "iris_status": iris_status,
        "iris_fallback_used": iris_fallback_used,
        "region_graph_source": graph_source,
        "region_graph_fallback_used": graph_fallback_used,
        "region_graph_start_goal_connected": graph_connected,
        "sampled_region_path_status": sampled_status,
        "sampled_region_path_fallback_reason": sampled_fallback_reason,
        "open_grid_fallback_used": bool(item.get("open_grid_fallback_used")),
    }


def _primary_diagnostic_source(flags: Sequence[str]) -> str:
    priority = (
        "path_planning_failure",
        "region_graph_disconnected",
        "region_graph_fallback",
        "iris_failure",
        "iris_fallback",
        "sampled_region_path_fallback",
        "tracking_safety_violation",
        "trajectory_optimization_fallback",
        "open_grid_fallback",
        "replan_required",
    )
    for flag in priority:
        if flag in flags:
            return flag
    return "none"


def _dedupe(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _report_present(value: Any) -> bool:
    return isinstance(value, dict)


__all__ = (
    "_input_source_summary",
    "_candidate_diagnostic_interpretation",
    "_primary_diagnostic_source",
    "_dedupe",
    "_report_present",
)
