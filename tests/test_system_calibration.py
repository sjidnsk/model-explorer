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


def _episode_with_scenario_id(scenario_id: str):
    from model_explorer.policy.features import PolicyObservation
    from model_explorer.policy.rollout import EpisodeMetrics, RolloutEpisode, RolloutInfo, RolloutTransition

    observation = PolicyObservation(
        candidate_feature_names=("utility",),
        candidate_features=((1.0,),),
        global_feature_names=("coverage_rate",),
        global_features=(0.0,),
        action_mask=(True,),
        candidate_cells=((1, 1),),
        candidate_missing_feature_names=((),),
        candidate_missing_indicator_names=(),
        candidate_missing_indicators=((),),
    )
    transition = RolloutTransition(
        observation=observation,
        action_index=0,
        log_prob=None,
        value=None,
        reward=1.0,
        next_observation=None,
        done=True,
        info=RolloutInfo(
            selected_cell=(1, 1),
            coverage_rate_delta=0.1,
            path_cost=1.0,
            risk=0.0,
            final_coverage_rate=0.1,
            extra={
                "scenario_id": scenario_id,
                "provenance": {
                    "scenario_id": scenario_id,
                    "data_class": "quasi_real",
                    "mask_stress_augmented": True,
                    "mask_stress_label": "mask_stress_augmented",
                },
            },
        ),
    )
    return RolloutEpisode(transitions=(transition,), metrics=EpisodeMetrics(final_coverage_rate=0.1))


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
    scenario_set: str = "all",
    diagnostic_profile: str = "all",
    acceptance_gate: str = "semi-real-closed-loop",
) -> dict:
    stress_failure = max(1, failure_count)
    stress_replan = max(1, replan_count)
    return {
        "schema_version": "path-feedback-summary/v1",
        "scenario_count": 3,
        "scenario_set": scenario_set,
        "diagnostic_profile": diagnostic_profile,
        "acceptance_gate": acceptance_gate,
        "top_k": 3,
        "planner_extra_args": ["--simulate-tracking", "--optimize-trajectory", "--drake-iris-regions"],
        "candidate_count": candidate_count,
        "reachable_count": reachable_count,
        "path_planning_failure_count": failure_count,
        "replan_count": replan_count,
        "open_grid_fallback_used": open_grid_fallback_used,
        "acceptance_metadata": {
            "scenario_set": scenario_set,
            "diagnostic_profile": diagnostic_profile,
            "acceptance_gate": acceptance_gate,
            "top_k": 3,
            "planner_extra_args": ["--simulate-tracking", "--optimize-trajectory", "--drake-iris-regions"],
            "open_grid_fallback_used": open_grid_fallback_used,
            "open_grid_fallback_used_gate": {
                "status": "failed" if open_grid_fallback_used else "passed",
                "expected": False,
                "actual": open_grid_fallback_used,
                "reason_codes": ["open_grid_fallback_used"] if open_grid_fallback_used else ["open_grid_fallback_not_used"],
            },
        },
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
        "scenarios": [
            {
                "scenario_id": "stress-a",
                "scenario_group": "stress",
                "selection_changed_by_path_feedback": True,
                "open_grid_fallback_used": open_grid_fallback_used,
                "path_feedback": {
                    "candidate_count": 3,
                    "reachable_count": 1,
                    "failure_count": failure_count,
                    "replan_count": replan_count,
                    "candidates": [
                        {
                            "action_index": 0,
                            "cell": [1, 1],
                            "reachable": failure_count == 0,
                            "replan_required": replan_count > 0,
                            "failure_reason": "path_blocked" if failure_count else None,
                            "diagnostic_interpretation": {
                                "diagnostic_flags": [
                                    *("path_planning_failure" for _ in range(1 if failure_count else 0)),
                                    *("replan_required" for _ in range(1 if replan_count else 0)),
                                    *("iris_fallback" for _ in range(1 if iris_fallback_count else 0)),
                                    *("region_graph_fallback" for _ in range(1 if region_graph_fallback_count else 0)),
                                    *("region_graph_disconnected" for _ in range(1 if region_graph_disconnected_count else 0)),
                                    *("open_grid_fallback" for _ in range(1 if open_grid_fallback_used else 0)),
                                ],
                                "open_grid_fallback_used": open_grid_fallback_used,
                            },
                        }
                    ],
                },
                "diagnostic_interpretation": {
                    "target_replacement_reason": "before_candidate_path_planning_failed"
                    if failure_count
                    else "unchanged",
                    "failure_sources": [
                        *("path_planning_failure" for _ in range(1 if failure_count else 0)),
                        *("replan_required" for _ in range(1 if replan_count else 0)),
                        *("iris_fallback" for _ in range(1 if iris_fallback_count else 0)),
                        *("region_graph_fallback" for _ in range(1 if region_graph_fallback_count else 0)),
                        *("region_graph_disconnected" for _ in range(1 if region_graph_disconnected_count else 0)),
                        *("open_grid_fallback" for _ in range(1 if open_grid_fallback_used else 0)),
                    ],
                    "open_grid_fallback_used": open_grid_fallback_used,
                },
            }
        ],
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

    def test_acceptance_metadata_missing_or_mismatch_is_machine_readable(self) -> None:
        from model_explorer.policy.system_calibration import evaluate_path_feedback_gate

        expected_gate = {
            "scenario_set": "all",
            "diagnostic_profile": "all",
            "top_k": 3,
            "require_open_grid_fallback_used": False,
        }
        missing_metadata = _path_feedback_summary()
        del missing_metadata["acceptance_metadata"]

        missing_gate = evaluate_path_feedback_gate(
            missing_metadata,
            {
                "require_open_grid_fallback_used_false": True,
                "acceptance_gate": expected_gate,
            },
        )
        self.assertEqual(missing_gate["status"], "failed")
        self.assertIn("acceptance_metadata_missing", missing_gate["reason_codes"])
        self.assertIn("acceptance_metadata_missing", missing_gate["warning_reason_codes"])
        self.assertTrue(all(isinstance(reason, str) for reason in missing_gate["reason_codes"]))

        mismatch_gate = evaluate_path_feedback_gate(
            _path_feedback_summary(scenario_set="stress"),
            {
                "require_open_grid_fallback_used_false": True,
                "acceptance_gate": expected_gate,
            },
        )
        self.assertEqual(mismatch_gate["status"], "failed")
        self.assertIn("acceptance_metadata_mismatch", mismatch_gate["reason_codes"])
        self.assertEqual(
            mismatch_gate["acceptance_metadata"]["mismatches"][0]["field"],
            "scenario_set",
        )

    def test_sample_quality_summary_maps_path_feedback_diagnostics_without_performance_claims(self) -> None:
        from model_explorer.policy.system_calibration import build_sample_quality_summary

        summary = build_sample_quality_summary(
            [_path_feedback_summary(
                open_grid_fallback_used=True,
                failure_count=1,
                replan_count=1,
                iris_fallback_count=1,
                region_graph_fallback_count=1,
                region_graph_disconnected_count=1,
            )],
            {
                "enabled": True,
                "exclude_reason_codes": ["open_grid_fallback"],
                "downweight_reason_codes": [
                    "path_planning_failure",
                    "replan_required",
                    "iris_fallback",
                    "region_graph_fallback",
                    "region_graph_disconnected",
                ],
            },
        )

        self.assertEqual(summary["quality_signal_use"], "calibration_only")
        self.assertTrue(summary["not_real_world_performance_claim"])
        record = summary["records"][0]
        for reason in (
            "path_planning_failure",
            "replan_required",
            "iris_fallback",
            "region_graph_fallback",
            "region_graph_disconnected",
            "open_grid_fallback",
        ):
            self.assertIn(reason, record["reason_codes"])
        self.assertEqual(record["decision"], "exclude")
        self.assertTrue(all(isinstance(reason, str) for reason in record["reason_codes"]))

    def test_training_sample_quality_is_explicit_and_preserves_legacy_default(self) -> None:
        from model_explorer.experiments.training_matrix import run_training as _run_training

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            def training_result(episodes, *args, **kwargs):
                episode_count = len(tuple(episodes))
                return _run(
                    f"raw-{kwargs['seed']}.pt",
                    float(episode_count),
                    seed=kwargs["seed"],
                ) | {"sample_count": episode_count, "dataset_summary": {"data_class": "quasi_real", "mask_stress_augmented": True}}

            base_episode = _episode_with_scenario_id("stress-a")
            clean_episode = _episode_with_scenario_id("clean-a")
            feedback_path = root / "path-feedback.json"
            feedback_path.write_text(
                json.dumps(_path_feedback_summary(open_grid_fallback_used=True, failure_count=1, replan_count=1)),
                encoding="utf-8",
            )

            with patch("model_explorer.policy.training.train_policy_on_episodes", side_effect=training_result):
                legacy = _run_training(
                    (base_episode, clean_episode),
                    {
                        "seed": 1,
                        "checkpoint": str(root / "legacy.pt"),
                        "evaluate_trained_policy": False,
                    },
                    base_dir=root,
                )

            with patch("model_explorer.policy.training.train_policy_on_episodes", side_effect=training_result):
                implicit = _run_training(
                    (base_episode, clean_episode),
                    {
                        "seed": 1,
                        "checkpoint": str(root / "implicit.pt"),
                        "evaluate_trained_policy": False,
                        "system_calibration": {
                            "path_feedback_summaries": [{"path": str(feedback_path)}],
                            "sample_quality": {
                                "match_key": "scenario_id",
                                "exclude_reason_codes": ["open_grid_fallback"],
                                "downweight_reason_codes": ["path_planning_failure", "replan_required"],
                            },
                        },
                    },
                    base_dir=root,
                )

            with patch("model_explorer.policy.training.train_policy_on_episodes", side_effect=training_result):
                gated = _run_training(
                    (base_episode, clean_episode),
                    {
                        "seed": 1,
                        "checkpoint": str(root / "gated.pt"),
                        "evaluate_trained_policy": False,
                        "system_calibration": {
                            "path_feedback_summaries": [{"path": str(feedback_path)}],
                            "sample_quality": {
                                "enabled": True,
                                "match_key": "scenario_id",
                                "exclude_reason_codes": ["open_grid_fallback"],
                                "downweight_reason_codes": ["path_planning_failure", "replan_required"],
                            },
                        },
                    },
                    base_dir=root,
                )

        self.assertNotIn("sample_quality_summary", legacy)
        self.assertEqual(legacy["sample_count"], 2)
        self.assertNotIn("sample_quality_summary", implicit)
        self.assertEqual(implicit["sample_count"], 2)
        self.assertIn("sample_quality_summary", gated)
        self.assertEqual(gated["sample_count"], 1)
        self.assertEqual(gated["sample_quality_summary"]["excluded_sample_count"], 1)
        self.assertIn("open_grid_fallback", gated["sample_quality_summary"]["records"][0]["reason_codes"])

    def test_sample_quality_audit_summary_aggregates_cross_summary_metadata_and_actions(self) -> None:
        from model_explorer.policy.system_calibration import build_sample_quality_audit_summary

        hard_exclusion_summary = _path_feedback_summary(
            open_grid_fallback_used=True,
            failure_count=1,
            replan_count=1,
        )
        hard_exclusion_summary["scenarios"][0]["scenario_id"] = "roi-open-grid"
        hard_exclusion_summary["scenarios"][0]["roi_group"] = "south-pole-rim"
        hard_exclusion_summary["scenarios"][0]["data_class"] = "quasi_real"
        hard_exclusion_summary["scenarios"][0]["mask_stress_augmented"] = True
        hard_exclusion_summary["scenarios"][0]["benchmark_scope"] = "not real-world generalization benchmark"

        downweight_summary = _path_feedback_summary(
            open_grid_fallback_used=False,
            failure_count=1,
            replan_count=1,
            iris_fallback_count=1,
            region_graph_fallback_count=1,
            region_graph_disconnected_count=1,
            scenario_set="stress",
            diagnostic_profile="iris",
        )
        downweight_summary["scenarios"][0]["scenario_id"] = "roi-region-diagnostic"
        downweight_summary["scenarios"][0]["roi_group"] = "shadowed-crater"
        downweight_summary["scenarios"][0]["data_class"] = "quasi_real"
        downweight_summary["scenarios"][0]["mask_stress_augmented"] = True
        downweight_summary["scenarios"][0]["benchmark_scope"] = "not real-world generalization benchmark"

        audit = build_sample_quality_audit_summary(
            [
                {"path": "outputs/run-a/path-feedback-summary.json", "summary": hard_exclusion_summary},
                {"path": "outputs/run-b/path-feedback-summary.json", "summary": downweight_summary},
            ],
            {
                "enabled": True,
                "data_class": "quasi_real",
                "mask_stress_augmented": True,
                "benchmark_scope": "not real-world generalization benchmark",
                "downweight_factor": 0.25,
            },
        )

        self.assertEqual(audit["schema_version"], "sample-quality-audit-summary/v1")
        self.assertEqual(audit["data_class"], "quasi_real")
        self.assertTrue(audit["mask_stress_augmented"])
        self.assertEqual(audit["benchmark_scope"], "not real-world generalization benchmark")
        self.assertEqual(audit["by_action"]["exclude"]["record_count"], 1)
        self.assertEqual(audit["by_action"]["downweight"]["record_count"], 1)
        self.assertEqual(audit["by_source_summary_path"]["outputs/run-a/path-feedback-summary.json"]["record_count"], 1)
        self.assertEqual(audit["by_scenario_set"]["all"]["record_count"], 1)
        self.assertEqual(audit["by_scenario_set"]["stress"]["record_count"], 1)
        self.assertEqual(audit["by_diagnostic_profile"]["iris"]["record_count"], 1)
        self.assertEqual(audit["by_top_k"]["3"]["record_count"], 2)
        self.assertEqual(audit["by_roi_group"]["shadowed-crater"]["record_count"], 1)
        self.assertEqual(audit["by_reason_code"]["open_grid_fallback"]["action_counts"]["exclude"], 1)
        self.assertEqual(audit["by_reason_code"]["region_graph_disconnected"]["action_counts"]["downweight"], 1)
        self.assertEqual(
            audit["records"][0]["acceptance_metadata"]["open_grid_fallback_used_gate"]["status"],
            "failed",
        )
        self.assertEqual(audit["records"][1]["sample_weight"], 0.25)
        self.assertTrue(
            all(
                isinstance(reason, str)
                for record in audit["records"]
                for reason in record["reason_codes"]
            )
        )

    def test_system_summary_exposes_sample_quality_audit_for_enabled_dataset_application(self) -> None:
        from model_explorer.policy.system_calibration import build_system_calibration_summary

        runs = [_run("selected.pt", 0.7, seed=1)]
        path_summary = _path_feedback_summary(
            failure_count=1,
            replan_count=1,
            iris_fallback_count=1,
            region_graph_fallback_count=1,
            region_graph_disconnected_count=1,
        )
        path_summary["scenarios"][0]["roi_group"] = "shadowed-crater"
        path_summary["scenarios"][0]["data_class"] = "quasi_real"
        path_summary["scenarios"][0]["mask_stress_augmented"] = True
        path_summary["scenarios"][0]["benchmark_scope"] = "not real-world generalization benchmark"

        summary = build_system_calibration_summary(
            {
                "runs": runs,
                "calibration_recommendation": {"recommended_checkpoint": "selected.pt"},
            },
            path_feedback_summaries=[
                {"seed": 1, "path": "outputs/run/path-feedback-summary.json", "summary": path_summary},
            ],
            config={
                "data_class": "quasi_real",
                "benchmark_scope": "not real-world generalization benchmark",
                "mask_stress_augmented": True,
                "sample_quality": {
                    "enabled": True,
                    "downweight_factor": 0.5,
                },
            },
            policy="torch_policy",
            metric="final_coverage_rate",
        )

        self.assertIn("sample_quality_summary", summary)
        self.assertIn("sample_quality_audit_summary", summary)
        self.assertEqual(summary["sample_quality_audit_summary"]["by_action"]["downweight"]["record_count"], 1)
        self.assertEqual(
            summary["sample_quality_audit_summary"]["records"][0]["source_summary_path"],
            "outputs/run/path-feedback-summary.json",
        )
        self.assertEqual(summary["sample_quality_audit_summary"]["records"][0]["data_class"], "quasi_real")
        self.assertTrue(summary["sample_quality_audit_summary"]["records"][0]["mask_stress_augmented"])

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
        from model_explorer.experiments.training_matrix import run_training as _run_training

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
