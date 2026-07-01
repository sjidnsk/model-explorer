from __future__ import annotations

import json


def _minimal_contract() -> dict:
    return {
        "schema_version": "model-explorer-contract/v1",
        "grid": {
            "width": 4,
            "height": 3,
            "resolution": 0.5,
            "frame_id": "moon_local",
            "origin": [1.0, 2.0],
            "layers": ["confidence", "cost"],
        },
        "constraints": {
            "violation_count": 0,
            "passable_ratio": 1.0,
            "reason_counts": {"obstacle": 0},
        },
        "top_goals": [
            {
                "cell": [2, 1],
                "utility": 0.42,
                "reachable": True,
                "path_cost": 2.0,
            }
        ],
        "top_sequences": [{"cells": [[2, 1]], "utility": 0.42, "coverage_area": 1.0}],
        "observation_update": {
            "coverage_rate": 0.25,
            "coverage_rate_delta": 0.1,
            "value_coverage": 0.3,
        },
    }


def test_experiment_run_keeps_summary_and_report_contract(tmp_path) -> None:
    from model_explorer.experiments.runner import (
        dry_run_experiment_manifest,
        run_experiment_manifest,
        validate_experiment_manifest,
    )

    scenario_path = tmp_path / "scenario.json"
    manifest_path = tmp_path / "experiment.json"
    rollout_path = tmp_path / "rollouts.jsonl"
    evaluation_path = tmp_path / "evaluation.json"
    report_path = tmp_path / "report.md"

    scenario_path.write_text(json.dumps(_minimal_contract()), encoding="utf-8")
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "model-explorer-experiment/v1",
                "scenarios": [str(scenario_path)],
                "planner": {"backend": "contract_cost"},
                "outputs": {
                    "rollouts": str(rollout_path),
                    "evaluation": str(evaluation_path),
                    "report": str(report_path),
                },
            }
        ),
        encoding="utf-8",
    )

    validation = validate_experiment_manifest(manifest_path)
    dry_run = dry_run_experiment_manifest(manifest_path)
    summary = run_experiment_manifest(manifest_path)
    report = report_path.read_text(encoding="utf-8")

    assert validation["status"] == "valid"
    assert dry_run["status"] == "dry_run"
    assert summary["schema_version"] == "model-explorer-experiment/v1"
    assert summary["scenario_count"] == 1
    assert summary["transition_count"] == 1
    assert summary["report_output"] == str(report_path)
    assert "policy_ranking" in summary
    assert "baseline_deltas" in summary
    assert "# Model Explorer Experiment Report" in report
    assert "## Policy Ranking" in report
    assert "## Baselines" in report
    assert "| utility |" in report
    assert "| coverage_heuristic |" in report
