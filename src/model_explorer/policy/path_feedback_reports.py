from __future__ import annotations

from typing import Any


def _list_text(value: Any) -> str:
    if isinstance(value, list | tuple):
        return ','.join(str(item) for item in value)
    if value is None:
        return ''
    return str(value)


def _legacy_render_path_feedback_markdown(summary: dict[str, Any]) -> str:
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
    triage = summary.get("gcs_control_point_candidate_triage")
    if isinstance(triage, dict) and int(triage.get("candidate_count") or 0) > 0:
        lines.extend(
            [
                "",
                "## GCS Control-Point Candidate Triage",
                "",
                f"schema_version: {triage.get('schema_version')}",
                "",
                "| scenario | action | selected | fallback | cost_delta | exposure_delta | direction_violations | direction_risks | motion_status | route_artifact |",
                "|---|---:|---:|---|---:|---:|---:|---|---|---|",
            ]
        )
        for row in triage.get("candidates", []):
            if not isinstance(row, dict):
                continue
            lines.append(
                "| {scenario_id} | {action} | {selected} | {fallback} | {cost_delta} | {exposure_delta} | {violations} | {risks} | {motion} | {route_artifact} |".format(
                    scenario_id=row.get("scenario_id"),
                    action=row.get("action_index"),
                    selected=row.get("candidate_selected"),
                    fallback=row.get("candidate_fallback_reason"),
                    cost_delta=row.get("cost_delta_vs_baseline"),
                    exposure_delta=row.get("high_cost_exposure_delta_vs_baseline"),
                    violations=row.get("direction_cone_violation_count"),
                    risks=_list_text(row.get("direction_cone_risk_flags")),
                    motion=row.get("motion_feasibility_status"),
                    route_artifact=row.get("route_artifact"),
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
            "## Sampled Region Path Diagnostics",
            "",
            "| scenario | group | selected | fallback | status_counts | source_counts | fallback_reasons |",
            "|---|---|---:|---:|---|---|---|",
        ]
    )
    for item in summary["scenarios"]:
        sampled = item["sampled_region_path_diagnostics"]
        lines.append(
            "| {scenario_id} | {group} | {selected} | {fallback} | {statuses} | {sources} | {reasons} |".format(
                scenario_id=item["scenario_id"],
                group=item["scenario_group"],
                selected=sampled["selected_count"],
                fallback=sampled["fallback_count"],
                statuses=sampled["status_counts"],
                sources=sampled["source_counts"],
                reasons=sampled["fallback_reasons"],
            )
        )
    lines.extend(
        [
            "",
            "## Sampled Region Path Candidate Audit",
            "",
            "| scenario | action | source | status | fallback | sequence | attempts | rankings | edge_transitions | cost_delta |",
            "|---|---:|---|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for item in summary["scenarios"]:
        for audit in item.get("sampled_region_path_candidate_audit", []):
            metrics = audit.get("candidate_metrics", {})
            lines.append(
                "| {scenario_id} | {action} | {source} | {status} | {fallback} | {sequence} | {attempts} | {rankings} | {edges} | {delta} |".format(
                    scenario_id=audit["scenario_id"],
                    action=audit["action_index"],
                    source=audit["region_source"],
                    status=audit["status"],
                    fallback=audit["fallback_reason"],
                    sequence=audit["region_sequence"],
                    attempts=audit["sample_attempt_count"],
                    rankings=audit["candidate_ranking_count"],
                    edges=audit["edge_transition_count"],
                    delta=metrics.get("candidate_cost_delta"),
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


def render_path_feedback_markdown(summary: dict[str, Any]) -> str:
    from .path_feedback_report_sections import render_markdown_sections

    return "\n".join(render_markdown_sections(summary))


__all__ = ['render_path_feedback_markdown']
