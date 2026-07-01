from __future__ import annotations


def test_quasi_real_report_and_metrics_contract_outputs() -> None:
    from model_explorer.experiments.quasi_real_matrix.metrics import (
        architecture_nested_metric_summary,
        evaluation_metric,
        numeric_summary,
        per_group_action_outcomes,
    )
    from model_explorer.experiments.quasi_real_matrix.reports import render_quasi_real_matrix_markdown

    runs = [
        {
            "architecture": "mlp_v1",
            "validation_evaluation": {
                "torch_policy": {
                    "final_coverage_rate": 0.4,
                    "action_sensitive_metrics": {"selected_expected_coverage_delta": 0.2},
                    "oracle_regret": {"coverage_regret": 0.1},
                }
            },
        }
    ]
    summary = {
        "status": "completed",
        "schema_version": "model-explorer-quasi-real-evaluation/v1",
        "evaluation_scope": "quasi-real evaluation; not real-world generalization benchmark",
        "name": "contract-quasi-real",
        "run_id": "run-001",
        "dataset_id": "fixture_matrix_lola",
        "data_class": "quasi_real",
        "region": "lunar_south_pole",
        "roi_count": 1,
        "rois": [
            {
                "split": "validation",
                "name": "smooth_high_confidence",
                "bounds": {"x": 0, "y": 0, "width": 4, "height": 4},
                "scenario_count": 1,
            }
        ],
        "splits": {"validation": 1},
        "mask_stress": {"enabled": False},
        "quality_gates": {"min_trainable_transition_count": 1},
        "experiment": {
            "dataset_summary": {"data_class": "quasi_real", "dataset_id": "fixture_matrix_lola"},
            "training": {"architectures": ["mlp_v1"]},
            "policy_ranking": [{"rank": 1, "policy": "torch_policy", "final_coverage_rate": 0.4}],
        },
        "architecture_selection": {
            "enabled": True,
            "status": "selected",
            "recommended_architecture": "mlp_v1",
            "selection_metric": "torch_policy.final_coverage_rate",
            "quality_gates": {"status": "passed", "violations": []},
            "architectures": {"mlp_v1": {"selection_metric": numeric_summary((0.4,))}},
            "per_group_action_outcomes": {},
        },
        "stability_summary": {"architectures": {"mlp_v1": {"final_coverage_rate": numeric_summary((0.4,))}}},
        "coverage_warnings": [],
    }

    nested = architecture_nested_metric_summary(
        runs,
        architectures=["mlp_v1"],
        section="action_sensitive_metrics",
    )
    per_group = per_group_action_outcomes(
        runs,
        architectures=["mlp_v1"],
        uncertainty_multiplier=1.0,
    )
    report = render_quasi_real_matrix_markdown(summary)

    assert evaluation_metric(runs[0]["validation_evaluation"], "torch_policy.final_coverage_rate") == 0.4
    assert evaluation_metric(summary["experiment"], "torch_policy.final_coverage_rate") is None
    assert nested["mlp_v1"]["selected_expected_coverage_delta"]["count"] == 1
    assert per_group["aggregate"]["architectures"]["mlp_v1"]["coverage_regret"]["count"] == 1
    assert "# Quasi-real South Pole Evaluation Matrix" in report
    assert "quasi-real evaluation; not real-world generalization benchmark" in report
    assert "## Quality Gates" in report
    assert "## Architecture Selection" in report
