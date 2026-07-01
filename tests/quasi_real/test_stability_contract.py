from __future__ import annotations

from pathlib import Path


def test_stability_summary_groups_runs_by_architecture_and_baseline_delta() -> None:
    from model_explorer.experiments.quasi_real_matrix import stability

    summary = stability._stability_summary(
        {
            "training": {
                "runs": [
                    {
                        "architecture": "mlp_v1",
                        "loss": 0.2,
                        "policy_loss": 0.1,
                        "dataset_summary": {"mask_stress_sample_count": 2},
                        "validation_evaluation": {
                            "aggregate": {
                                "torch_policy": {
                                    "final_coverage_rate": 0.6,
                                    "total_path_cost": 12.0,
                                    "average_risk": 0.3,
                                    "failure_count": 0,
                                }
                            }
                        },
                        "baseline_deltas": {
                            "coverage_heuristic": {"final_coverage_rate": 0.05}
                        },
                    },
                    {"architecture": "mlp_v1", "loss": 0.4},
                    {"architecture": "candidate_attention_v1", "loss": 0.1},
                ]
            }
        }
    )

    assert summary["architectures"]["mlp_v1"]["run_count"] == 2
    assert summary["architectures"]["mlp_v1"]["loss"]["count"] == 2
    assert summary["architectures"]["mlp_v1"]["torch_policy.final_coverage_rate"]["mean"] == 0.6
    assert summary["architectures"]["mlp_v1"]["dataset.mask_stress_sample_count"]["mean"] == 2.0
    assert (
        summary["baseline_deltas"]["mlp_v1"]["coverage_heuristic.final_coverage_rate"]["mean"]
        == 0.05
    )
    assert summary["loss_distribution"]["loss"]["count"] == 3


def test_manifest_architecture_seed_helpers_preserve_manifest_defaults() -> None:
    from model_explorer.experiments.quasi_real_matrix.manifest import QuasiRealEvaluationManifest
    from model_explorer.experiments.quasi_real_matrix import stability

    manifest = QuasiRealEvaluationManifest(
        path=Path("manifest.json"),
        name="fixture",
        run_id="run",
        dataset_manifest=Path("dataset.json"),
        output_root=Path("out"),
        rois=(),
        mask_stress_config={},
        selection_config={},
        dataset_validation={},
        train_config={"architecture": "", "seed": "9"},
        planner_config={},
        reward_config={},
    )

    assert stability._manifest_architectures(manifest) == ["mlp_v1"]
    assert stability._manifest_seeds(manifest) == [9]
    assert stability._architecture_run_count(
        [{"architecture": "mlp_v1"}, {"architecture": "candidate_attention_v1"}],
        "mlp_v1",
    ) == 1
