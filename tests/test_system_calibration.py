import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


def _run(
    checkpoint: str,
    metric: float,
    *,
    seed: int,
    source: str = "feedback_aware",
    teacher_weight: float = 0.1,
    curriculum_profile: str = "soft_all_valid",
    teacher_status: str = "passed",
) -> dict:
    return {
        "checkpoint": checkpoint,
        "seed": seed,
        "training_data_selection_strategy": source,
        "teacher_imitation_weight": teacher_weight,
        "teacher_margin_curriculum_profile": curriculum_profile,
        "teacher_quality_gates": {
            "status": teacher_status,
            "reason_codes": [] if teacher_status == "passed" else ["missing_teacher_signal_rate_exceeded"],
        },
        "dataset_summary": {
            "dataset_id": "fixture-quasi-real",
            "data_class": "quasi_real",
            "mask_stress_augmented": True,
            "mask_stress_sample_count": 2,
        },
        "validation_evaluation": {
            "torch_policy": {
                "final_coverage_rate": metric,
                "feedback_aware_confidence_calibration": {"teacher_label_nll": 0.2},
            }
        },
    }


def _path_feedback_summary(
    *,
    open_grid_fallback_used: bool = False,
    failure_count: int = 0,
    replan_count: int = 0,
    reachable_count: int = 6,
    candidate_count: int = 6,
    iris_fallback_count: int = 0,
    region_graph_fallback_count: int = 0,
    region_graph_disconnected_count: int = 0,
) -> dict:
    stress_failure = max(1, failure_count)
    stress_replan = max(1, replan_count)
    return {
        "schema_version": "path-feedback-summary/v1",
        "scenario_count": 3,
        "top_k": 3,
        "candidate_count": candidate_count,
        "reachable_count": reachable_count,
        "path_planning_failure_count": failure_count,
        "replan_count": replan_count,
        "open_grid_fallback_used": open_grid_fallback_used,
        "iris_fallback_count": iris_fallback_count,
        "region_graph_fallback_count": region_graph_fallback_count,
        "region_graph_disconnected_count": region_graph_disconnected_count,
        "region_graph_start_goal_disconnected_count": region_graph_disconnected_count,
        "failure_reasons": ["path_blocked"] if failure_count else [],
        "scenario_group_summary": {
            "stress": {
                "scenario_count": 1,
                "candidate_count": 3,
                "reachable_count": 0,
                "failure_count": stress_failure,
                "replan_count": stress_replan,
                "selection_changed_count": 1,
                "iris_report_count": 3,
                "iris_fallback_count": iris_fallback_count,
                "region_graph_fallback_count": region_graph_fallback_count,
                "region_graph_start_goal_disconnected_count": region_graph_disconnected_count,
            },
            "mixed_stress": {
                "scenario_count": 1,
                "candidate_count": 3,
                "reachable_count": 1,
                "failure_count": 1,
                "replan_count": 1,
                "selection_changed_count": 1,
                "iris_report_count": 3,
                "iris_fallback_count": iris_fallback_count,
                "region_graph_fallback_count": region_graph_fallback_count,
                "region_graph_start_goal_disconnected_count": region_graph_disconnected_count,
            },
        },
        "diagnostic_interpretation": {
            "scenario_group_interpretation": {
                "stress": {
                    "failure_sources": {
                        "path_planning_failure": stress_failure,
                        "replan_required": stress_replan,
                    }
                },
                "mixed_stress": {
                    "failure_sources": {
                        "path_planning_failure": 1,
                        "replan_required": 1,
                    }
                },
            }
        },
    }


class SystemCalibrationSummaryTests(unittest.TestCase):
    def test_path_feedback_gate_failed_run_is_not_recommended_by_default(self) -> None:
        from model_explorer.policy.system_calibration import build_system_calibration_summary

        high_metric_failed = _run("failed.pt", 0.95, seed=1)
        lower_metric_passed = _run("passed.pt", 0.75, seed=2)

        summary = build_system_calibration_summary(
            {
                "runs": [high_metric_failed, lower_metric_passed],
                "calibration_recommendation": {"recommended_checkpoint": "failed.pt"},
            },
            path_feedback_summaries=[
                {
                    "source_selection_strategy": "feedback_aware",
                    "teacher_imitation_weight": 0.1,
                    "teacher_margin_curriculum_profile": "soft_all_valid",
                    "seed": 1,
                    "summary": _path_feedback_summary(open_grid_fallback_used=True),
                },
                {
                    "source_selection_strategy": "feedback_aware",
                    "teacher_imitation_weight": 0.1,
                    "teacher_margin_curriculum_profile": "soft_all_valid",
                    "seed": 2,
                    "summary": _path_feedback_summary(),
                },
            ],
            config={
                "data_class": "quasi_real",
                "benchmark_scope": "not real-world generalization benchmark",
                "mask_stress_augmented": True,
                "path_feedback_gate": {"require_open_grid_fallback_used_false": True},
            },
            policy="torch_policy",
            metric="final_coverage_rate",
        )

        self.assertEqual(summary["selection"]["recommended_checkpoint"], "passed.pt")
        self.assertEqual(summary["selection"]["status"], "selected")
        excluded = {item["checkpoint"]: item for item in summary["selection"]["excluded_runs"]}
        self.assertEqual(excluded["failed.pt"]["path_feedback_gate"]["status"], "failed")
        self.assertIn("open_grid_fallback_used", excluded["failed.pt"]["reason_codes"])

    def test_no_system_gate_config_keeps_legacy_best_metric_selection(self) -> None:
        from model_explorer.policy.system_calibration import select_system_best_run

        high_metric = _run("legacy-high.pt", 0.95, seed=1)
        lower_metric = _run("legacy-low.pt", 0.75, seed=2)

        selection = select_system_best_run(
            [high_metric, lower_metric],
            policy="torch_policy",
            metric="final_coverage_rate",
            system_gate_configured=False,
        )

        self.assertEqual(selection["run"]["checkpoint"], "legacy-high.pt")
        self.assertEqual(selection["mode"], "legacy_best_metric")
        self.assertEqual(selection["excluded_runs"], [])

    def test_open_grid_fallback_is_machine_readable_path_feedback_exclusion_reason(self) -> None:
        from model_explorer.policy.system_calibration import evaluate_path_feedback_gate

        gate = evaluate_path_feedback_gate(
            _path_feedback_summary(open_grid_fallback_used=True),
            {"require_open_grid_fallback_used_false": True},
        )

        self.assertEqual(gate["status"], "failed")
        self.assertEqual(gate["reason_codes"], ["open_grid_fallback_used"])
        self.assertTrue(all(isinstance(reason, str) for reason in gate["reason_codes"]))

    def test_system_summary_preserves_stress_mixed_stress_diagnostics_and_joint_gate_rates(self) -> None:
        from model_explorer.policy.system_calibration import build_system_calibration_summary

        runs = [
            _run("teacher-and-path-pass.pt", 0.7, seed=1),
            _run("teacher-fail.pt", 0.8, seed=2, teacher_status="failed"),
            _run("path-fail.pt", 0.9, seed=3),
        ]

        summary = build_system_calibration_summary(
            {
                "runs": runs,
                "calibration_recommendation": {"recommended_checkpoint": "path-fail.pt"},
            },
            path_feedback_summaries=[
                {"seed": 1, "summary": _path_feedback_summary(failure_count=2, replan_count=2, reachable_count=4)},
                {"seed": 2, "summary": _path_feedback_summary()},
                {
                    "seed": 3,
                    "summary": _path_feedback_summary(
                        failure_count=3,
                        replan_count=2,
                        reachable_count=2,
                        iris_fallback_count=2,
                        region_graph_fallback_count=2,
                        region_graph_disconnected_count=2,
                    ),
                },
            ],
            config={
                "data_class": "quasi_real",
                "benchmark_scope": "not real-world generalization benchmark",
                "mask_stress_augmented": True,
                "path_feedback_gate": {
                    "require_open_grid_fallback_used_false": True,
                    "max_path_planning_failure_rate": 0.8,
                    "max_replan_rate": 0.8,
                    "max_region_graph_disconnected_rate": 0.1,
                },
            },
            policy="torch_policy",
            metric="final_coverage_rate",
        )

        self.assertEqual(summary["schema_version"], "system-calibration-summary/v1")
        self.assertEqual(summary["data_class"], "quasi_real")
        self.assertTrue(summary["mask_stress_augmented"])
        self.assertEqual(summary["benchmark_scope"], "not real-world generalization benchmark")
        self.assertEqual(summary["evaluation_scope"], "calibration evidence; not real-world generalization benchmark")
        self.assertEqual(summary["gate_summary"]["teacher_quality_gate_pass_count"], 2)
        self.assertEqual(summary["gate_summary"]["path_feedback_gate_pass_count"], 2)
        self.assertEqual(summary["gate_summary"]["joint_gate_pass_count"], 1)
        self.assertAlmostEqual(summary["gate_summary"]["joint_gate_pass_rate"], 1 / 3)
        self.assertEqual(summary["selection"]["recommended_checkpoint"], "teacher-and-path-pass.pt")

        stress = summary["path_feedback_diagnostics"]["stress"]
        mixed = summary["path_feedback_diagnostics"]["mixed_stress"]
        self.assertGreaterEqual(stress["failure_count"], 1)
        self.assertGreaterEqual(stress["replan_count"], 1)
        self.assertGreaterEqual(mixed["reachable_count"], 1)
        self.assertGreaterEqual(mixed["failure_count"], 1)
        self.assertGreaterEqual(mixed["replan_count"], 1)

        path_fail = next(item for item in summary["runs"] if item["checkpoint"] == "path-fail.pt")
        self.assertEqual(path_fail["selection_decision"]["status"], "excluded")
        self.assertIn("region_graph_disconnected_rate_exceeded", path_fail["selection_decision"]["reason_codes"])
        self.assertEqual(path_fail["quality_signal_use"], "calibration_only")
        self.assertTrue(path_fail["not_real_world_performance_claim"])

    def test_training_system_calibration_config_controls_best_checkpoint_and_writes_json_summary(self) -> None:
        from model_explorer.policy.experiment import _run_training

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            failed_summary_path = root / "failed-path-feedback.json"
            passed_summary_path = root / "passed-path-feedback.json"
            failed_summary_path.write_text(
                json.dumps(_path_feedback_summary(open_grid_fallback_used=True)),
                encoding="utf-8",
            )
            passed_summary_path.write_text(
                json.dumps(_path_feedback_summary()),
                encoding="utf-8",
            )

            def training_result(*args, **kwargs):
                seed = kwargs["seed"]
                return _run(
                    f"raw-{seed}.pt",
                    0.95 if seed == 1 else 0.75,
                    seed=seed,
                )

            with patch("model_explorer.policy.training.train_policy_on_episodes", side_effect=training_result):
                summary = _run_training(
                    (),
                    {
                        "seeds": [1, 2],
                        "checkpoint": str(root / "checkpoint-{seed}.pt"),
                        "evaluate_trained_policy": False,
                        "system_calibration": {
                            "summary_output": str(root / "system-calibration-summary.json"),
                            "data_class": "quasi_real",
                            "benchmark_scope": "not real-world generalization benchmark",
                            "mask_stress_augmented": True,
                            "path_feedback_gate": {"require_open_grid_fallback_used_false": True},
                            "path_feedback_summaries": [
                                {"seed": 1, "path": str(failed_summary_path)},
                                {"seed": 2, "path": str(passed_summary_path)},
                            ],
                        },
                    },
                    base_dir=root,
                )

            system_summary_path = root / "system-calibration-summary.json"
            persisted = json.loads(system_summary_path.read_text(encoding="utf-8"))

        self.assertEqual(summary["best_seed"], 2)
        self.assertTrue(summary["best_checkpoint"].endswith("checkpoint-2.pt"))
        self.assertEqual(summary["calibration_recommendation"]["recommended_seed"], 2)
        self.assertEqual(summary["best_selection"]["mode"], "system_gate_best_metric")
        self.assertEqual(summary["best_selection"]["excluded_run_count"], 1)
        self.assertIn("open_grid_fallback_used", summary["best_selection"]["excluded_runs"][0]["reason_codes"])
        self.assertIn("system_calibration_summary", summary)
        self.assertEqual(
            summary["system_calibration_summary"]["selection"]["recommended_checkpoint"],
            summary["best_checkpoint"],
        )
        self.assertEqual(persisted["schema_version"], "system-calibration-summary/v1")
        self.assertEqual(persisted["selection"]["recommended_seed"], 2)


if __name__ == "__main__":
    unittest.main()
