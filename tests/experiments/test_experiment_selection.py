from __future__ import annotations


def test_experiment_selection_api_prefers_best_validation_metric() -> None:
    from model_explorer.experiments.selection import select_best_training_run

    runs = [
        {"seed": 1, "validation_evaluation": {"torch_policy": {"final_coverage_rate": 0.2}}},
        {"seed": 2, "validation_evaluation": {"torch_policy": {"final_coverage_rate": 0.4}}},
    ]

    best = select_best_training_run(runs, policy="torch_policy", metric="final_coverage_rate")

    assert best["seed"] == 2
