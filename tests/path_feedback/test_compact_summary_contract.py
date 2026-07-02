from __future__ import annotations

from pathlib import Path


def _summary_with_compact_fields() -> dict:
    return {
        "schema_version": "path-feedback-summary/v1",
        "scenario_count": 1,
        "scenario_set": "contract",
        "diagnostic_profile": "diagnostic",
        "acceptance_gate": "semi_real",
        "top_k": 3,
        "planner_extra_args": ["--planner"],
        "candidate_count": 4,
        "reachable_count": 2,
        "path_planning_failure_count": 1,
        "replan_count": 1,
        "total_path_cost": 12.5,
        "average_path_cost": 6.25,
        "coverage_per_path_cost": 0.4,
        "selection_changed_count": 1,
        "selection_changed_rate": 1.0,
        "tracking_safety_violation_count": 0,
        "trajectory_optimization_fallback_count": 0,
        "region_graph_disconnected_count": 0,
        "open_grid_fallback_used": False,
        "failure_reasons": ["blocked"],
        "iris_requested_count": 4,
        "iris_report_count": 4,
        "iris_status_counts": {"ok": 4},
        "iris_fallback_count": 0,
        "iris_failure_count": 0,
        "iris_region_count_total": 8,
        "iris_fallback_reasons": {},
        "region_graph_source_counts": {"iris": 4},
        "region_graph_fallback_count": 0,
        "region_graph_fallback_reasons": {},
        "region_graph_start_goal_disconnected_count": 0,
        "open_grid_fallback_used_gate": {"status": "passed"},
        "acceptance_metadata": {"schema_version": "path-feedback-acceptance-metadata/v1"},
        "scenario_group_summary": {},
        "scenarios": [],
        "convex_region_report_count": 1,
        "convex_region_backend_counts": {"iris": 1},
        "gcs_candidate_available_count": 1,
        "gcs_control_point_candidate_triage": {"candidate_count": 1},
        "channel_aware_astar_selected_backend_counts": {"channel": 1},
        "sampled_region_path_bridge_corridor_status_counts": {"connected": 1},
        "diagnostic_interpretation": {"primary": "path_feedback"},
    }


def test_compact_path_feedback_summary_preserves_required_and_group_fields(tmp_path: Path) -> None:
    from model_explorer.policy.path_feedback_compact_summary import compact_summary_payload
    from model_explorer.policy.path_feedback_summary import (
        PATH_FEEDBACK_SUMMARY_ACCEPTANCE_METRICS,
        PATH_FEEDBACK_SUMMARY_REQUIRED_KEYS,
        compact_path_feedback_summary,
        validate_path_feedback_summary_contract,
    )

    summary = _summary_with_compact_fields()
    validate_path_feedback_summary_contract(summary)

    payload = compact_path_feedback_summary(
        summary,
        summary_output=tmp_path / "summary.json",
        report_output=tmp_path / "summary.md",
    )

    assert payload == compact_summary_payload(
        summary,
        summary_output=tmp_path / "summary.json",
        report_output=tmp_path / "summary.md",
    )
    assert "scenario_group_summary" in PATH_FEEDBACK_SUMMARY_REQUIRED_KEYS
    assert "scenarios" in PATH_FEEDBACK_SUMMARY_REQUIRED_KEYS
    assert set(PATH_FEEDBACK_SUMMARY_ACCEPTANCE_METRICS).issubset(payload)
    assert payload["status"] == "completed"
    assert payload["convex_region_backend_counts"] == {"iris": 1}
    assert payload["gcs_candidate_available_count"] == 1
    assert payload["gcs_control_point_candidate_triage"] == {"candidate_count": 1}
    assert payload["channel_aware_astar_selected_backend_counts"] == {"channel": 1}
    assert payload["sampled_region_path_bridge_corridor_status_counts"] == {"connected": 1}
    assert payload["diagnostic_interpretation"] == {"primary": "path_feedback"}
    assert payload["summary_output"] == str(tmp_path / "summary.json")
    assert payload["report_output"] == str(tmp_path / "summary.md")


def test_compact_summary_defaults_match_public_contract() -> None:
    from model_explorer.policy.path_feedback_summary import compact_path_feedback_summary

    payload = compact_path_feedback_summary(_summary_with_compact_fields())

    assert payload["planner_extra_args"] == ["--planner"]
    assert payload["gcs_candidate_audit"] == []
    assert payload["gcs_motion_feasibility_status_counts"] == {}
    assert "summary_output" not in payload
    assert "report_output" not in payload


def test_compact_summary_default_containers_are_not_shared_between_calls() -> None:
    from model_explorer.policy.path_feedback_summary import compact_path_feedback_summary

    first = compact_path_feedback_summary(_summary_with_compact_fields())
    first["gcs_candidate_audit"].append({"candidate": "mutated"})
    first["gcs_motion_feasibility_status_counts"]["mutated"] = 1

    second = compact_path_feedback_summary(_summary_with_compact_fields())

    assert second["gcs_candidate_audit"] == []
    assert second["gcs_motion_feasibility_status_counts"] == {}
