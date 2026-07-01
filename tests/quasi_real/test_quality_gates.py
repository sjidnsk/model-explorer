from __future__ import annotations


def test_selection_quality_gates_report_explicit_minimum_violations() -> None:
    from model_explorer.experiments.quasi_real_matrix import quality_gates

    result = quality_gates._selection_quality_gates(
        {
            "min_seed_count": 3,
            "min_architecture_count": 2,
            "min_roi_group_count": 4,
            "min_unreachable_candidate_count": 1,
            "min_mask_stress_sample_count": 1,
        },
        architectures=["mlp_v1"],
        seeds=[7],
        runs=[{"architecture": "mlp_v1", "loss": 0.2}],
        metric_stats={"mlp_v1": {"count": 1}},
        loss_stats={"mlp_v1": {"count": 1}},
        exception_counts={"mlp_v1": 0},
        dataset_summary={
            "roi_count": 1,
            "unreachable_candidate_count": 0,
            "mask_stress_sample_count": 0,
        },
    )

    assert result["status"] == "failed"
    assert {
        violation["gate"] for violation in result["violations"]
    } >= {
        "min_seed_count",
        "min_architecture_count",
        "min_roi_group_count",
        "min_unreachable_candidate_count",
        "min_mask_stress_sample_count",
    }


def test_mask_stress_coverage_coerces_missing_and_non_numeric_counts() -> None:
    from model_explorer.experiments.quasi_real_matrix import quality_gates

    coverage = quality_gates._mask_stress_coverage(
        {
            "unreachable_candidate_count": "3",
            "padding_candidate_count": None,
            "missing_experimental_feature_candidate_count": "not-a-number",
            "mask_stress_sample_count": 5.7,
            "mask_stress_augmented": True,
        }
    )

    assert coverage == {
        "unreachable_candidate_count": 3,
        "padding_candidate_count": 0,
        "missing_experimental_feature_candidate_count": 0,
        "mask_stress_sample_count": 5,
        "mask_stress_augmented": True,
    }
