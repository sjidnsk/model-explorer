from __future__ import annotations


def _report_summary() -> dict:
    candidate = {
        "action_index": 0,
        "cell": [1, 1],
        "reachable": True,
        "path_cost": 2.5,
        "risk": 0.1,
        "utility": 0.9,
        "replan_required": False,
        "failure_reason": None,
        "diagnostic_interpretation": {
            "diagnostic_flags": ["iris_ok"],
            "iris_status": "success",
            "iris_fallback_used": False,
            "region_graph_source": "iris",
            "region_graph_fallback_used": False,
            "region_graph_start_goal_connected": True,
            "open_grid_fallback_used": False,
        },
    }
    scenario = {
        "scenario_id": "scenario-a",
        "scenario_group": "group-a",
        "selected_cell_before_path_feedback": [0, 0],
        "selected_cell_after_path_feedback": [1, 1],
        "selection_changed_by_path_feedback": True,
        "selected_path_cost_before_feedback": 4.0,
        "selected_path_cost_after_feedback": 2.5,
        "path_cost_delta_after_feedback": -1.5,
        "coverage_rate_delta": 0.2,
        "path_feedback": {
            "reachable_count": 1,
            "failure_count": 0,
            "replan_count": 1,
            "candidates": [candidate],
        },
        "diagnostic_interpretation": {
            "target_replacement_reason": "lower_cost",
            "failure_sources": ["none"],
            "primary_failure_reason": "none",
            "iris_region_graph_signal": "connected",
            "open_grid_fallback_used": False,
        },
        "iris_diagnostics": {
            "status_counts": {"success": 1},
            "fallback_reasons": {},
            "region_count_total": 2,
        },
        "region_graph_diagnostics": {
            "source_counts": {"iris": 1},
            "fallback_reasons": {},
            "start_goal_disconnected_count": 0,
        },
        "sampled_region_path_diagnostics": {
            "selected_count": 1,
            "fallback_count": 0,
            "status_counts": {"selected": 1},
            "source_counts": {"sampled": 1},
            "fallback_reasons": {},
        },
        "sampled_region_path_candidate_audit": [
            {
                "scenario_id": "scenario-a",
                "action_index": 0,
                "region_source": "sampled",
                "status": "selected",
                "fallback_reason": None,
                "region_sequence": [0, 1],
                "sample_attempt_count": 3,
                "candidate_ranking_count": 2,
                "edge_transition_count": 1,
                "candidate_metrics": {"candidate_cost_delta": -1.5},
            }
        ],
    }
    return {
        "scenario_count": 1,
        "candidate_count": 1,
        "reachable_count": 1,
        "path_planning_failure_count": 0,
        "replan_count": 1,
        "selection_changed_count": 1,
        "selection_changed_rate": 1.0,
        "total_path_cost": 2.5,
        "coverage_per_path_cost": 0.4,
        "open_grid_fallback_used": False,
        "scenarios": [scenario],
        "gcs_control_point_candidate_triage": {
            "schema_version": "gcs-control-point-triage/v1",
            "candidate_count": 1,
            "candidates": [
                {
                    "scenario_id": "scenario-a",
                    "action_index": 0,
                    "candidate_selected": True,
                    "candidate_fallback_reason": None,
                    "cost_delta_vs_baseline": -1.5,
                    "high_cost_exposure_delta_vs_baseline": -0.1,
                    "direction_cone_violation_count": 0,
                    "direction_cone_risk_flags": [],
                    "motion_feasibility_status": "diagnostic_only",
                    "route_artifact": "route.json",
                }
            ],
        },
        "scenario_group_summary": {
            "group-a": {
                "scenario_count": 1,
                "candidate_count": 1,
                "reachable_count": 1,
                "failure_count": 0,
                "replan_count": 1,
                "selection_changed_count": 1,
                "iris_report_count": 1,
                "region_graph_fallback_count": 0,
                "region_graph_start_goal_disconnected_count": 0,
            }
        },
    }


def test_render_path_feedback_markdown_preserves_key_sections_and_order() -> None:
    from model_explorer.policy.path_feedback_report_sections import render_markdown_sections
    from model_explorer.policy.path_feedback_reports import render_path_feedback_markdown

    report = render_path_feedback_markdown(_report_summary())

    assert report == "\n".join(render_markdown_sections(_report_summary()))
    headings = [
        "# Path Feedback Summary",
        "## Baseline vs Feedback",
        "## Candidate Paths",
        "## Diagnostic Interpretation",
        "## Candidate Diagnostics",
        "## GCS Control-Point Candidate Triage",
        "## IRIS Diagnostics",
        "## Region Graph Diagnostics",
        "## Sampled Region Path Diagnostics",
        "## Sampled Region Path Candidate Audit",
        "## Scenario Groups",
    ]
    positions = [report.index(heading) for heading in headings]
    assert positions == sorted(positions)
    assert "| metric | value |" in report
    assert "| scenario | action | cell | reachable | path_cost | risk | utility | replan | failure |" in report
    assert "IRIS / region graph fields are diagnostic features only." in report
