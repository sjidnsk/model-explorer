from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


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


class QuasiRealStabilityManifestTemplateTests(unittest.TestCase):
    def test_stability_manifest_template_is_tracked_and_expands_roi_and_seed_coverage(self):
        from model_explorer.experiments.quasi_real_matrix.manifest import load_quasi_real_evaluation_manifest

        manifest_path = ROOT / "data" / "manifests" / "lunar_south_pole_lro_lola_stability_matrix_v1.json"

        manifest = load_quasi_real_evaluation_manifest(manifest_path)
        roi_counts: dict[str, int] = {}
        for roi in manifest.rois:
            roi_counts[roi.name] = roi_counts.get(roi.name, 0) + 1

        self.assertTrue(manifest_path.exists())
        self.assertEqual(manifest.dataset_manifest.name, "lunar_south_pole_lro_lola_gdr_875s_20m.json")
        self.assertEqual(manifest.output_root.name, "qreal_stability_v1")
        self.assertEqual(manifest.output_root.parent.name, "processed")
        self.assertEqual(manifest.output_root.parent.parent.name, "data")
        self.assertEqual(set(manifest.train_config["architectures"]), {"mlp_v1", "mlp_missing_v1", "candidate_attention_v1"})
        self.assertGreaterEqual(len(manifest.train_config["seeds"]), 3)
        self.assertGreaterEqual(set(roi.split for roi in manifest.rois), {"train", "validation", "test"})
        for roi_name in ("smooth_high_confidence", "rim_or_steep_slope", "low_observation_count", "mixed_risk"):
            self.assertGreaterEqual(roi_counts.get(roi_name, 0), 2)
