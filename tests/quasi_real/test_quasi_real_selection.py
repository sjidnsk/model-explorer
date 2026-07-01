from __future__ import annotations


def test_quasi_real_selection_api_reports_inconclusive_seed_variance() -> None:
    from model_explorer.experiments.quasi_real_matrix.selection import selection_decision

    decision = selection_decision(
        {
            "mlp_v1": {"count": 3, "mean": 0.50, "std": 0.10, "min": 0.40, "max": 0.60},
            "mlp_missing_v1": {"count": 3, "mean": 0.54, "std": 0.08, "min": 0.46, "max": 0.62},
        },
        metric="torch_policy.final_coverage_rate",
        mode="max",
        uncertainty_multiplier=1.0,
    )

    assert decision["status"] == "inconclusive"
    assert decision["recommended_architecture"] is None
