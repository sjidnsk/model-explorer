from __future__ import annotations

import json


def _minimal_contract() -> dict:
    return {
        "schema_version": "model-explorer-contract/v1",
        "grid": {
            "width": 6,
            "height": 5,
            "resolution": 0.5,
            "frame_id": "moon_local",
            "origin": [0.0, 0.0],
            "layers": ["confidence", "cost"],
        },
        "constraints": {
            "violation_count": 0,
            "passable_ratio": 0.9,
            "reason_counts": {"obstacle": 0},
        },
        "top_goals": [
            {
                "cell": [2, 1],
                "utility": 0.5,
                "reachable": True,
                "expected_coverage_rate_delta": 0.2,
                "risk": 0.1,
                "path_cost": 1.5,
            }
        ],
        "top_sequences": [{"cells": [[2, 1]], "utility": 0.5, "coverage_area": 1.0}],
        "observation_update": {"coverage_rate": 0.2, "coverage_rate_delta": 0.2},
    }


def test_path_feedback_run_keeps_json_and_markdown_contract(tmp_path) -> None:
    from model_explorer.policy.path_feedback_manifest import validate_path_feedback_manifest
    from model_explorer.policy.path_feedback_runner import (
        dry_run_path_feedback_manifest,
        run_path_feedback_manifest,
    )
    from model_explorer.policy.path_feedback_summary import PATH_FEEDBACK_SUMMARY_REQUIRED_KEYS

    contract_path = tmp_path / "scenario.json"
    sidecar_path = tmp_path / "sidecar.json"
    route_path = tmp_path / "route.json"
    manifest_path = tmp_path / "path-feedback.json"
    summary_path = tmp_path / "summary.json"
    report_path = tmp_path / "summary.md"

    contract_path.write_text(json.dumps(_minimal_contract()), encoding="utf-8")
    sidecar_path.write_text(
        json.dumps(
            {
                "schema_version": "path-planner-sidecar/v1",
                "grid": {
                    "width": 6,
                    "height": 5,
                    "resolution": 0.5,
                    "frame_id": "moon_local",
                    "origin": [0.0, 0.0],
                },
                "cost": [[1.0 for _ in range(6)] for _ in range(5)],
                "passable_mask": [[True for _ in range(6)] for _ in range(5)],
                "metadata": {"scenario_id": "contract-path-feedback"},
            }
        ),
        encoding="utf-8",
    )
    route_path.write_text(
        json.dumps(
            {
                "schema_version": "path-planner-route/v1",
                "reachable": True,
                "path_cost": 2.0,
                "failure_reason": None,
                "geometric_path": {"cells": [[0, 0], [2, 1]], "world": [[0.0, 0.0], [1.0, 0.5]]},
                "diagnostics": {"path_length_m": 1.2, "search_mode": "platform_aware_astar"},
            }
        ),
        encoding="utf-8",
    )
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "path-feedback-manifest/v1",
                "top_k": 1,
                "scenarios": [
                    {
                        "scenario_id": "contract-path-feedback",
                        "contract": "scenario.json",
                        "sidecar": "sidecar.json",
                        "route_fixtures": {"0": "route.json"},
                    }
                ],
                "outputs": {"summary": str(summary_path), "report": str(report_path)},
            }
        ),
        encoding="utf-8",
    )

    validation = validate_path_feedback_manifest(manifest_path)
    dry_run = dry_run_path_feedback_manifest(manifest_path)
    summary = run_path_feedback_manifest(manifest_path)
    report = report_path.read_text(encoding="utf-8")

    assert validation["status"] == "valid"
    assert dry_run["status"] == "dry_run"
    assert set(PATH_FEEDBACK_SUMMARY_REQUIRED_KEYS).issubset(summary)
    assert summary["schema_version"] == "path-feedback-summary/v1"
    assert summary["scenario_count"] == 1
    assert summary["selection_changed_count"] == 0
    assert "scenarios" in json.loads(summary_path.read_text(encoding="utf-8"))
    assert "# Path Feedback Summary" in report
    assert "## Baseline vs Feedback" in report
    assert "## Candidate Paths" in report
