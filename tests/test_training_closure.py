import json
import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


def minimal_contract(
    *,
    goals=None,
    sequences=None,
    observation_update=None,
    schema_version="model-explorer-contract/v1",
):
    return {
        "schema_version": schema_version,
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
        "top_goals": goals
        if goals is not None
        else [{"cell": [2, 1], "utility": 0.42, "reachable": True}],
        "top_sequences": sequences
        if sequences is not None
        else [{"cells": [[2, 1]], "utility": 0.42, "coverage_area": 1.0}],
        "observation_update": observation_update
        if observation_update is not None
        else {"delta_c": 0.25, "visible_cell_count": 3, "updated_cell_count": 3},
    }


def load_contract_from_dict(payload):
    from model_explorer.io.scenario import load_scenario

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "contract.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return load_scenario(path).snapshots[0]


class RolloutDatasetSummaryTests(unittest.TestCase):
    def test_dataset_summary_counts_multi_episode_jsonl_and_missing_fields_are_finite(self):
        from model_explorer.policy.collector import collect_rollout_episode
        from model_explorer.policy.dataset import summarize_rollout_dataset
        from model_explorer.policy.rollout_io import read_rollout_episodes, write_rollout_episodes_jsonl

        successful_episode = collect_rollout_episode(
            [
                load_contract_from_dict(
                    minimal_contract(
                        goals=[
                            {"cell": [1, 1], "utility": 0.5, "reachable": True, "risk": 0.2},
                            {"cell": [2, 1], "utility": 0.4, "reachable": False},
                        ],
                        observation_update={"coverage_rate": 0.2, "coverage_rate_delta": 0.2},
                    )
                )
            ],
            max_candidates=3,
        )
        failure_episode = collect_rollout_episode(
            [
                load_contract_from_dict(
                    minimal_contract(
                        goals=[
                            {"cell": [0, 0], "utility": 0.5, "reachable": False},
                            {"cell": [1, 0], "utility": 0.4, "reachable": False},
                        ],
                        observation_update={},
                    )
                )
            ],
            max_candidates=3,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "rollouts.jsonl"
            write_rollout_episodes_jsonl(path, [successful_episode, failure_episode])
            loaded = read_rollout_episodes(path)
            summary = summarize_rollout_dataset(loaded)

        self.assertEqual(summary["episode_count"], 2)
        self.assertEqual(summary["transition_count"], 2)
        self.assertEqual(summary["trainable_transition_count"], 1)
        self.assertEqual(summary["no_op_transition_count"], 1)
        self.assertEqual(summary["failure_transition_count"], 1)
        self.assertEqual(summary["reachable_action_count_distribution"]["counts"], [1, 0])
        self.assertEqual(summary["empty_action_mask_count"], 1)
        self.assertEqual(summary["invalid_action_mask_count"], 0)
        self.assertEqual(summary["reward"]["min"], -1.0)
        self.assertAlmostEqual(summary["reward"]["max"], 0.16)
        self.assertEqual(summary["failure_count"], 1)
        self.assertEqual(summary["coverage_delta_total"], 0.2)
        self.assertEqual(summary["total_path_cost"], 0.0)
        self.assertEqual(summary["average_risk"], 0.1)

    def test_dataset_validation_reports_empty_mask_and_no_trainable_transitions(self):
        from model_explorer.policy.collector import collect_rollout_episode
        from model_explorer.policy.dataset import summarize_rollout_dataset, validate_rollout_dataset

        episode = collect_rollout_episode(
            [
                load_contract_from_dict(
                    minimal_contract(
                        goals=[{"cell": [0, 0], "utility": 0.5, "reachable": False}],
                        observation_update={},
                    )
                )
            ],
            max_candidates=2,
        )

        summary = summarize_rollout_dataset([episode])

        self.assertIn("no_trainable_transitions", summary["warnings"])
        self.assertIn("empty_action_mask", summary["warnings"])
        with self.assertRaisesRegex(ValueError, "no trainable transitions"):
            validate_rollout_dataset([episode])

    def test_dataset_validation_gates_report_named_threshold_failures(self):
        from model_explorer.policy.collector import collect_rollout_episode
        from model_explorer.policy.dataset import validate_rollout_dataset

        episode = collect_rollout_episode(
            [
                load_contract_from_dict(
                    minimal_contract(
                        goals=[{"cell": [0, 0], "utility": 0.5, "reachable": False}],
                        observation_update={},
                    )
                )
            ],
            max_candidates=2,
        )

        gates = {
            "min_episode_count": 1,
            "min_transition_count": 1,
            "min_trainable_transition_count": 1,
            "max_empty_action_mask_count": 0,
            "max_failure_rate": 0.0,
            "require_finite_reward": True,
        }

        with self.assertRaisesRegex(
            ValueError,
            "min_trainable_transition_count.*max_empty_action_mask_count.*max_failure_rate",
        ):
            validate_rollout_dataset([episode], gates=gates)

    def test_enhanced_dataset_gates_measure_missing_features_masks_and_reward_std(self):
        from model_explorer.policy.collector import collect_rollout_episode
        from model_explorer.policy.dataset import summarize_rollout_dataset, validate_rollout_dataset

        episode = collect_rollout_episode(
            [
                load_contract_from_dict(
                    minimal_contract(
                        goals=[
                            {"cell": [0, 0], "utility": 0.6, "reachable": True},
                            {"cell": [1, 1], "utility": 0.5, "reachable": False},
                        ],
                        observation_update={"coverage_rate": 0.1, "coverage_rate_delta": 0.1},
                    )
                )
            ],
            max_candidates=2,
        )

        summary = summarize_rollout_dataset([episode])

        self.assertGreater(summary["missing_experimental_feature_rate"], 0.0)
        self.assertEqual(summary["action_mask_valid_mean"], 0.5)
        self.assertEqual(summary["unreachable_candidate_rate"], 0.5)
        self.assertEqual(summary["reward"]["std"], 0.0)

        gates = {
            "max_missing_experimental_feature_rate": 0.0,
            "min_action_mask_valid_mean": 0.75,
            "max_unreachable_candidate_rate": 0.25,
            "min_reward_std": 0.01,
        }
        with self.assertRaisesRegex(
            ValueError,
            "max_missing_experimental_feature_rate.*min_action_mask_valid_mean.*"
            "max_unreachable_candidate_rate.*min_reward_std",
        ):
            validate_rollout_dataset([episode], gates=gates)


class ReturnAdvantageTests(unittest.TestCase):
    def test_discounted_return_helper_respects_done_boundaries(self):
        from model_explorer.policy.training import compute_returns_and_advantages

        rewards = [1.0, 2.0, 3.0, 4.0]
        dones = [False, True, False, True]

        result = compute_returns_and_advantages(
            rewards=rewards,
            dones=dones,
            discount_factor=0.5,
            mode="discounted",
        )

        self.assertEqual(result.returns, (2.0, 2.0, 5.0, 4.0))
        self.assertEqual(result.advantages, (2.0, 2.0, 5.0, 4.0))


class ExperimentManifestValidationTests(unittest.TestCase):
    def test_manifest_output_root_derives_reproducible_run_paths(self):
        from model_explorer.policy.experiment import run_experiment_manifest

        scenario_payload = minimal_contract(
            goals=[{"cell": [1, 1], "utility": 0.5, "reachable": True}],
            observation_update={"coverage_rate": 0.1, "coverage_rate_delta": 0.1},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            scenario_path = Path(tmpdir) / "scenario.json"
            output_root = Path(tmpdir) / "out"
            manifest_path = Path(tmpdir) / "experiment.json"
            scenario_path.write_text(json.dumps(scenario_payload), encoding="utf-8")
            manifest_path.write_text(
                json.dumps(
                    {
                        "name": "formal-smoke",
                        "run_id": "run-a",
                        "scenarios": [str(scenario_path)],
                        "outputs": {"root": str(output_root)},
                    }
                ),
                encoding="utf-8",
            )

            summary = run_experiment_manifest(manifest_path)

            run_dir = output_root / "formal-smoke" / "run-a"
            dataset_summary = json.loads((run_dir / "dataset-summary.json").read_text(encoding="utf-8"))
            report = (run_dir / "report.md").read_text(encoding="utf-8")

        self.assertEqual(summary["experiment_name"], "formal-smoke")
        self.assertEqual(summary["run_id"], "run-a")
        self.assertEqual(summary["output_layout"]["run_dir"], str(run_dir))
        self.assertEqual(summary["rollout_output"], str(run_dir / "rollouts.jsonl"))
        self.assertEqual(summary["evaluation_output"], str(run_dir / "evaluation.json"))
        self.assertEqual(summary["dataset_summary_output"], str(run_dir / "dataset-summary.json"))
        self.assertEqual(summary["report_output"], str(run_dir / "report.md"))
        self.assertEqual(dataset_summary["episode_count"], 1)
        self.assertIn("## Dataset Summary", report)

    def test_manifest_dataset_validation_gates_fail_before_training(self):
        from model_explorer.policy.experiment import run_experiment_manifest

        scenario_payload = minimal_contract(
            goals=[{"cell": [0, 0], "utility": 0.5, "reachable": False}],
            observation_update={},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            scenario_path = Path(tmpdir) / "scenario.json"
            manifest_path = Path(tmpdir) / "experiment.json"
            scenario_path.write_text(json.dumps(scenario_payload), encoding="utf-8")
            manifest_path.write_text(
                json.dumps(
                    {
                        "scenarios": [str(scenario_path)],
                        "max_candidates": 2,
                        "outputs": {
                            "rollouts": str(Path(tmpdir) / "rollouts.jsonl"),
                            "evaluation": str(Path(tmpdir) / "evaluation.json"),
                        },
                        "dataset_validation": {
                            "min_trainable_transition_count": 1,
                            "max_empty_action_mask_count": 0,
                            "max_failure_rate": 0.0,
                        },
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "max_empty_action_mask_count"):
                run_experiment_manifest(manifest_path)

    def test_explicit_splits_write_resolved_manifest_and_environment_metadata(self):
        from model_explorer.policy.experiment import run_experiment_manifest

        train_payload = minimal_contract(
            goals=[{"cell": [1, 1], "utility": 0.5, "reachable": True, "expected_coverage_rate_delta": 0.1}],
            observation_update={"coverage_rate": 0.1, "coverage_rate_delta": 0.1},
        )
        validation_payload = minimal_contract(
            goals=[{"cell": [2, 1], "utility": 0.4, "reachable": True, "expected_coverage_rate_delta": 0.2}],
            observation_update={"coverage_rate": 0.2, "coverage_rate_delta": 0.1},
        )
        test_payload = minimal_contract(
            goals=[{"cell": [3, 1], "utility": 0.3, "reachable": True, "expected_coverage_rate_delta": 0.3}],
            observation_update={"coverage_rate": 0.3, "coverage_rate_delta": 0.1},
        )
        benchmark_payload = minimal_contract(
            goals=[{"cell": [1, 2], "utility": 0.2, "reachable": True, "expected_coverage_rate_delta": 0.4}],
            observation_update={"coverage_rate": 0.4, "coverage_rate_delta": 0.1},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = {}
            for name, payload in (
                ("train", train_payload),
                ("validation", validation_payload),
                ("test", test_payload),
                ("benchmark", benchmark_payload),
            ):
                path = Path(tmpdir) / f"{name}.json"
                path.write_text(json.dumps(payload), encoding="utf-8")
                paths[name] = path

            output_root = Path(tmpdir) / "out"
            manifest_path = Path(tmpdir) / "experiment.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "name": "split-smoke",
                        "run_id": "run-xyz",
                        "splits": {
                            "train": [str(paths["train"])],
                            "validation": [str(paths["validation"])],
                            "test": [str(paths["test"])],
                            "benchmark": {"coverage_dominant": [str(paths["benchmark"])]},
                        },
                        "outputs": {"root": str(output_root)},
                    }
                ),
                encoding="utf-8",
            )

            summary = run_experiment_manifest(manifest_path)
            run_dir = output_root / "split-smoke" / "run-xyz"
            resolved = json.loads((run_dir / "manifest.resolved.json").read_text(encoding="utf-8"))
            evaluation = json.loads((run_dir / "evaluation.json").read_text(encoding="utf-8"))
            report = (run_dir / "report.md").read_text(encoding="utf-8")

        self.assertEqual(summary["split_summaries"]["train"]["scenario_count"], 1)
        self.assertEqual(summary["split_summaries"]["validation"]["transition_count"], 1)
        self.assertEqual(summary["split_summaries"]["test"]["episode_count"], 1)
        self.assertEqual(summary["split_summaries"]["benchmark"]["groups"]["coverage_dominant"]["scenario_count"], 1)
        self.assertEqual(summary["resolved_manifest_output"], str(run_dir / "manifest.resolved.json"))
        self.assertEqual(resolved["experiment_name"], "split-smoke")
        self.assertTrue(Path(resolved["splits"]["train"][0]).is_absolute())
        self.assertIn("python_version", summary["environment"])
        self.assertIn("available", summary["environment"]["torch"])
        self.assertIn("commit", summary["environment"]["git"])
        self.assertIn("dirty", summary["environment"]["git"])
        self.assertIn("coverage_dominant", evaluation["groups"])
        self.assertIn("## Benchmark Groups", report)


class BestCheckpointSelectionTests(unittest.TestCase):
    def test_best_checkpoint_selection_prefers_highest_validation_metric(self):
        from model_explorer.policy.experiment import _select_best_training_run

        runs = [
            {
                "seed": 11,
                "checkpoint": "seed-11/checkpoint.pt",
                "validation_evaluation": {"torch_policy": {"final_coverage_rate": 0.2}},
            },
            {
                "seed": 13,
                "checkpoint": "seed-13/checkpoint.pt",
                "validation_evaluation": {"torch_policy": {"final_coverage_rate": 0.35}},
            },
        ]

        best = _select_best_training_run(runs, policy="torch_policy", metric="final_coverage_rate")

        self.assertEqual(best["seed"], 13)
        self.assertEqual(best["checkpoint"], "seed-13/checkpoint.pt")


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not available")
class TrainingClosureTests(unittest.TestCase):
    def test_checkpoint_metadata_contains_training_configuration_and_old_checkpoints_load(self):
        import torch

        from model_explorer.policy.collector import collect_rollout_episode
        from model_explorer.policy.torch_policy import MaskedCandidatePolicyNetwork
        from model_explorer.policy.training import load_policy_checkpoint, train_policy_on_episodes

        episode = collect_rollout_episode(
            [
                load_contract_from_dict(
                    minimal_contract(
                        goals=[{"cell": [1, 1], "utility": 0.5, "reachable": True}],
                        observation_update={"coverage_rate_delta": 0.1},
                    )
                )
            ],
            max_candidates=2,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "policy.pt"
            result = train_policy_on_episodes(
                [episode],
                checkpoint_path=checkpoint_path,
                seed=23,
                hidden_size=16,
                learning_rate=5.0e-4,
                epochs=2,
            )
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            metadata = checkpoint["metadata"]

            old_network = MaskedCandidatePolicyNetwork(
                candidate_feature_count=len(checkpoint["candidate_feature_names"]),
                global_feature_count=len(checkpoint["global_feature_names"]),
                hidden_size=16,
            )
            old_checkpoint_path = Path(tmpdir) / "old-policy.pt"
            torch.save(
                {
                    "state_dict": old_network.state_dict(),
                    "hidden_size": 16,
                    "candidate_feature_names": checkpoint["candidate_feature_names"],
                    "global_feature_names": checkpoint["global_feature_names"],
                    "metadata": {
                        "format": "model-explorer-masked-policy/v1",
                        "action_count": 2,
                        "seed": 1,
                        "sample_count": 1,
                    },
                },
                old_checkpoint_path,
            )
            old_scorer = load_policy_checkpoint(old_checkpoint_path)

        self.assertEqual(result["epochs"], 2)
        self.assertEqual(result.get("architecture"), "mlp_v1")
        self.assertEqual(result["architecture_config"]["hidden_dim"], 16)
        self.assertEqual(result["architecture_config"]["dropout"], 0.0)
        self.assertEqual(result["architecture_diagnostics"]["architecture"], "mlp_v1")
        self.assertEqual(result["architecture_diagnostics"]["observation_schema_version"], "policy-observation/v1.1")
        self.assertEqual(result["architecture_diagnostics"]["candidate_feature_dim"], len(checkpoint["candidate_feature_names"]))
        self.assertEqual(result["architecture_diagnostics"]["global_feature_dim"], len(checkpoint["global_feature_names"]))
        self.assertEqual(
            result["architecture_diagnostics"]["missing_indicator_dim"],
            len(checkpoint["candidate_missing_indicator_names"]),
        )
        self.assertEqual(
            result["architecture_diagnostics"]["mask_valid_action_count"],
            result["dataset_summary"]["reachable_action_count_distribution"],
        )
        self.assertEqual(metadata["format"], "model-explorer-masked-policy")
        self.assertEqual(metadata["version"], 2)
        self.assertEqual(metadata.get("architecture"), "mlp_v1")
        self.assertEqual(metadata["architecture_config"], result["architecture_config"])
        self.assertEqual(metadata["candidate_feature_names"], checkpoint["candidate_feature_names"])
        self.assertEqual(metadata["global_feature_names"], checkpoint["global_feature_names"])
        self.assertEqual(metadata["observation_schema_version"], "policy-observation/v1.1")
        self.assertEqual(
            metadata["candidate_missing_indicator_names"],
            checkpoint["candidate_missing_indicator_names"],
        )
        self.assertEqual(metadata["action_count"], 2)
        self.assertEqual(metadata["seed"], 23)
        self.assertEqual(metadata["sample_count"], 1)
        self.assertEqual(metadata["epoch_count"], 2)
        self.assertEqual(metadata["hidden_size"], 16)
        self.assertEqual(metadata["learning_rate"], 5.0e-4)
        self.assertIsNotNone(old_scorer)
        self.assertEqual(old_scorer.network.architecture_name, "mlp_v1")
        self.assertEqual(old_scorer.network.architecture_config["hidden_dim"], 16)

    def test_mlp_missing_architecture_trains_saves_and_loads_from_checkpoint(self):
        import torch

        from model_explorer.policy.collector import collect_rollout_episode
        from model_explorer.policy.training import load_policy_checkpoint, train_policy_on_episodes

        episode = collect_rollout_episode(
            [
                load_contract_from_dict(
                    minimal_contract(
                        goals=[
                            {"cell": [1, 1], "utility": 0.5, "reachable": True},
                            {"cell": [2, 1], "utility": 0.4, "reachable": True, "risk": 0.0},
                        ],
                        observation_update={"coverage_rate_delta": 0.1},
                    )
                )
            ],
            max_candidates=3,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "missing-policy.pt"
            result = train_policy_on_episodes(
                [episode],
                checkpoint_path=checkpoint_path,
                seed=31,
                hidden_size=16,
                epochs=1,
                architecture="mlp_missing_v1",
            )
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            scorer = load_policy_checkpoint(checkpoint_path)

        self.assertEqual(result["architecture"], "mlp_missing_v1")
        self.assertEqual(checkpoint["metadata"]["architecture"], "mlp_missing_v1")
        self.assertIn("candidate_missing_indicator_names", checkpoint["metadata"])
        self.assertEqual(scorer.network.architecture_name, "mlp_missing_v1")
        self.assertTrue(torch.isfinite(torch.tensor(result["total_loss"])))

    def test_training_rejects_unknown_architecture_config_fields(self):
        from model_explorer.policy.collector import collect_rollout_episode
        from model_explorer.policy.training import train_policy_on_episodes

        episode = collect_rollout_episode(
            [
                load_contract_from_dict(
                    minimal_contract(
                        goals=[{"cell": [1, 1], "utility": 0.5, "reachable": True}],
                        observation_update={"coverage_rate_delta": 0.1},
                    )
                )
            ],
            max_candidates=2,
        )

        with self.assertRaisesRegex(ValueError, "unknown architecture config field.*unexpected_knob"):
            train_policy_on_episodes(
                [episode],
                architecture="mlp_v1",
                architecture_config={"unexpected_knob": 1},
            )

    def test_unknown_training_architecture_returns_readable_error(self):
        from model_explorer.policy.collector import collect_rollout_episode
        from model_explorer.policy.training import train_policy_on_episodes

        episode = collect_rollout_episode(
            [
                load_contract_from_dict(
                    minimal_contract(
                        goals=[{"cell": [1, 1], "utility": 0.5, "reachable": True}],
                        observation_update={"coverage_rate_delta": 0.1},
                    )
                )
            ],
            max_candidates=2,
        )

        with self.assertRaisesRegex(ValueError, "unknown architecture.*not_a_model.*mlp_v1"):
            train_policy_on_episodes([episode], architecture="not_a_model")

    def test_candidate_attention_architecture_trains_saves_and_loads_from_checkpoint(self):
        import torch

        from model_explorer.policy.collector import collect_rollout_episode
        from model_explorer.policy.training import load_policy_checkpoint, train_policy_on_episodes

        episode = collect_rollout_episode(
            [
                load_contract_from_dict(
                    minimal_contract(
                        goals=[
                            {"cell": [1, 1], "utility": 0.5, "reachable": True, "risk": 0.0},
                            {"cell": [2, 1], "utility": 0.4, "reachable": True, "risk": 0.2},
                            {"cell": [3, 1], "utility": 9.0, "reachable": False, "risk": 0.1},
                        ],
                        observation_update={"coverage_rate_delta": 0.1},
                    )
                )
            ],
            max_candidates=4,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "attention-policy.pt"
            result = train_policy_on_episodes(
                [episode],
                checkpoint_path=checkpoint_path,
                seed=37,
                epochs=1,
                architecture="candidate_attention_v1",
                architecture_config={"hidden_dim": 16, "attention_heads": 2, "dropout": 0.0},
            )
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            scorer = load_policy_checkpoint(checkpoint_path)

        self.assertEqual(result["architecture"], "candidate_attention_v1")
        self.assertEqual(result["architecture_config"]["attention_heads"], 2)
        self.assertEqual(checkpoint["metadata"]["architecture"], "candidate_attention_v1")
        self.assertEqual(checkpoint["metadata"]["architecture_config"]["attention_heads"], 2)
        self.assertEqual(scorer.network.architecture_name, "candidate_attention_v1")
        self.assertEqual(scorer.network.architecture_config["attention_heads"], 2)
        self.assertTrue(torch.isfinite(torch.tensor(result["total_loss"])))

    def test_experiment_train_block_evaluates_trained_policy_and_reports_training_section(self):
        scenario_payload = minimal_contract(
            goals=[
                {"cell": [1, 1], "utility": 0.5, "reachable": True, "path_cost": 1.0},
                {"cell": [2, 1], "utility": 10.0, "reachable": False, "path_cost": 1.0},
            ],
            observation_update={
                "coverage_rate": 0.1,
                "coverage_rate_delta": 0.1,
                "value_coverage": 0.2,
            },
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            scenario_path = Path(tmpdir) / "scenario-a.json"
            second_scenario_path = Path(tmpdir) / "scenario-b.json"
            manifest_path = Path(tmpdir) / "experiment.json"
            rollout_path = Path(tmpdir) / "rollouts.jsonl"
            evaluation_path = Path(tmpdir) / "evaluation.json"
            report_path = Path(tmpdir) / "report.md"
            checkpoint_path = Path(tmpdir) / "policy.pt"
            loss_log_path = Path(tmpdir) / "losses.jsonl"
            scenario_path.write_text(json.dumps(scenario_payload), encoding="utf-8")
            second_scenario_path.write_text(json.dumps(scenario_payload), encoding="utf-8")
            manifest_path.write_text(
                json.dumps(
                    {
                        "scenarios": [str(scenario_path), str(second_scenario_path)],
                        "max_candidates": 2,
                        "planner": {"backend": "contract_cost"},
                        "outputs": {
                            "rollouts": str(rollout_path),
                            "evaluation": str(evaluation_path),
                            "report": str(report_path),
                        },
                        "train": {
                            "checkpoint": str(checkpoint_path),
                            "loss_log": str(loss_log_path),
                            "architecture": "mlp_missing_v1",
                            "architecture_config": {"hidden_dim": 16, "dropout": 0.0},
                            "seed": 29,
                            "epochs": 1,
                            "validation_fraction": 0.5,
                            "evaluate_trained_policy": True,
                        },
                    }
                ),
                encoding="utf-8",
            )

            completed = subprocess.run(
                [sys.executable, str(ROOT / "scripts" / "run_experiment.py"), str(manifest_path)],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
            summary = json.loads(completed.stdout)
            evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
            report = report_path.read_text(encoding="utf-8")

        self.assertIn("dataset_summary", summary)
        self.assertIn("dataset_summary", summary["training"])
        self.assertEqual(summary["training"]["architecture"], "mlp_missing_v1")
        self.assertEqual(summary["training"]["architecture_config"]["hidden_dim"], 16)
        self.assertEqual(summary["training"]["architecture_diagnostics"]["architecture"], "mlp_missing_v1")
        self.assertEqual(
            summary["training"]["architecture_diagnostics"]["observation_schema_version"],
            "policy-observation/v1.1",
        )
        self.assertIn("mask_valid_action_count", summary["training"]["architecture_diagnostics"])
        self.assertIn("mlp_missing_v1", summary.get("architecture_deltas", {}))
        self.assertIn("torch_policy", evaluation)
        self.assertIn("torch_policy", summary["training"]["baseline_evaluation"])
        self.assertIn("## Training", report)
        self.assertIn("| architecture | mlp_missing_v1 |", report)
        self.assertIn("## Architecture Diagnostics", report)
        self.assertIn("| observation_schema_version | policy-observation/v1.1 |", report)
        self.assertIn("| hidden_dim | 16 |", report)
        self.assertIn("## Architecture Deltas", report)
        self.assertIn("checkpoint", report)
        self.assertIn("dataset_summary", report)
        self.assertIn("| torch_policy |", report)
        for metric_name in (
            "final_coverage_rate",
            "cumulative_coverage_rate_delta",
            "total_path_cost",
            "average_risk",
            "failure_count",
            "replan_count",
            "value_coverage",
        ):
            self.assertIn(metric_name, summary["training"]["baseline_evaluation"]["torch_policy"])

    def test_experiment_train_block_supports_architecture_matrix_outputs(self):
        from model_explorer.policy.experiment import run_experiment_manifest

        architectures = ["mlp_v1", "mlp_missing_v1", "candidate_attention_v1"]
        scenario_payload = minimal_contract(
            goals=[
                {
                    "cell": [1, 1],
                    "utility": 0.5,
                    "reachable": True,
                    "expected_coverage_rate_delta": 0.2,
                    "risk": 0.1,
                    "path_cost": 1.0,
                },
                {
                    "cell": [2, 1],
                    "utility": 0.4,
                    "reachable": True,
                    "expected_coverage_rate_delta": 0.1,
                    "risk": 0.2,
                    "path_cost": 2.0,
                },
            ],
            observation_update={
                "coverage_rate": 0.2,
                "coverage_rate_delta": 0.1,
                "value_coverage": 0.2,
            },
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            first_scenario_path = Path(tmpdir) / "scenario-a.json"
            second_scenario_path = Path(tmpdir) / "scenario-b.json"
            output_root = Path(tmpdir) / "out"
            manifest_path = Path(tmpdir) / "experiment.json"
            first_scenario_path.write_text(json.dumps(scenario_payload), encoding="utf-8")
            second_scenario_path.write_text(json.dumps(scenario_payload), encoding="utf-8")
            manifest_path.write_text(
                json.dumps(
                    {
                        "name": "architecture-matrix-smoke",
                        "run_id": "run-001",
                        "scenarios": [str(first_scenario_path), str(second_scenario_path)],
                        "max_candidates": 2,
                        "outputs": {"root": str(output_root)},
                        "train": {
                            "seed": 17,
                            "architectures": architectures,
                            "architecture_configs": {
                                "mlp_v1": {"hidden_dim": 16, "dropout": 0.0},
                                "mlp_missing_v1": {"hidden_dim": 16, "dropout": 0.0},
                                "candidate_attention_v1": {
                                    "hidden_dim": 16,
                                    "attention_heads": 2,
                                    "dropout": 0.0,
                                },
                            },
                            "epochs": 1,
                            "validation_fraction": 0.5,
                            "evaluate_trained_policy": True,
                        },
                    }
                ),
                encoding="utf-8",
            )

            summary = run_experiment_manifest(manifest_path)
            report = (output_root / "architecture-matrix-smoke" / "run-001" / "report.md").read_text(
                encoding="utf-8"
            )
            checkpoint_records = [
                {
                    "architecture": run["architecture"],
                    "seed": run["seed"],
                    "parts": Path(run["checkpoint"]).parts,
                    "exists": Path(run["checkpoint"]).exists(),
                    "baseline_deltas": run.get("baseline_deltas", {}),
                }
                for run in summary["training"]["runs"]
            ]

        training = summary["training"]
        self.assertEqual(training["architectures"], architectures)
        self.assertEqual(training["run_count"], 3)
        self.assertEqual({run["architecture"] for run in training["runs"]}, set(architectures))
        self.assertEqual(set(summary["architecture_deltas"]), set(architectures))
        for record in checkpoint_records:
            self.assertTrue(record["exists"])
            self.assertIn(record["architecture"], record["parts"])
            self.assertIn("seed-17", record["parts"])
            self.assertEqual(record["seed"], 17)
            self.assertIn("utility", record["baseline_deltas"])
            self.assertIn("coverage_heuristic", record["baseline_deltas"])
        for run in training["runs"]:
            self.assertIn("architecture_config", run)
            self.assertIn("architecture_diagnostics", run)
            self.assertEqual(run["architecture_diagnostics"]["architecture"], run["architecture"])
        self.assertIn("synthetic smoke / regression suite", report)
        self.assertIn("## Architecture Diagnostics", report)
        for architecture in architectures:
            self.assertIn(architecture, report)

    def test_experiment_train_block_supports_multi_seed_outputs_and_summary(self):
        from model_explorer.policy.experiment import run_experiment_manifest

        first_payload = minimal_contract(
            goals=[
                {"cell": [1, 1], "utility": 0.5, "reachable": True, "path_cost": 1.0},
                {"cell": [2, 1], "utility": 9.0, "reachable": False, "path_cost": 1.0},
            ],
            observation_update={"coverage_rate": 0.1, "coverage_rate_delta": 0.1},
        )
        second_payload = minimal_contract(
            goals=[
                {"cell": [1, 2], "utility": 0.4, "reachable": True, "path_cost": 1.0},
                {"cell": [3, 1], "utility": 8.0, "reachable": False, "path_cost": 1.0},
            ],
            observation_update={"coverage_rate": 0.2, "coverage_rate_delta": 0.2},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            first_scenario_path = Path(tmpdir) / "scenario-a.json"
            second_scenario_path = Path(tmpdir) / "scenario-b.json"
            output_root = Path(tmpdir) / "out"
            manifest_path = Path(tmpdir) / "experiment.json"
            first_scenario_path.write_text(json.dumps(first_payload), encoding="utf-8")
            second_scenario_path.write_text(json.dumps(second_payload), encoding="utf-8")
            manifest_path.write_text(
                json.dumps(
                    {
                        "name": "multi-seed-smoke",
                        "run_id": "run-001",
                        "scenarios": [str(first_scenario_path), str(second_scenario_path)],
                        "max_candidates": 2,
                        "planner": {"backend": "contract_cost"},
                        "outputs": {"root": str(output_root)},
                        "dataset_validation": {
                            "min_episode_count": 2,
                            "min_transition_count": 2,
                            "min_trainable_transition_count": 2,
                            "max_empty_action_mask_count": 0,
                            "max_invalid_action_mask_count": 0,
                            "max_failure_rate": 0.0,
                            "require_finite_reward": True,
                        },
                        "train": {
                            "seeds": [11, 13],
                            "hidden_size": 16,
                            "epochs": 1,
                            "validation_fraction": 0.5,
                        },
                    }
                ),
                encoding="utf-8",
            )

            summary = run_experiment_manifest(manifest_path)
            run_dir = output_root / "multi-seed-smoke" / "run-001"
            checkpoint_exists = {
                seed: (run_dir / f"seed-{seed}" / "checkpoint.pt").exists()
                for seed in (11, 13)
            }
            loss_log_exists = {
                seed: (run_dir / f"seed-{seed}" / "losses.jsonl").exists()
                for seed in (11, 13)
            }
            report = (run_dir / "report.md").read_text(encoding="utf-8")

        training = summary["training"]
        self.assertEqual(training["seeds"], [11, 13])
        self.assertEqual(len(training["runs"]), 2)
        self.assertEqual(training["run_count"], 2)
        self.assertEqual(training["best_seed"], 11)
        self.assertEqual(training["best_selection"]["metric"], "final_coverage_rate")
        self.assertEqual(training["best_checkpoint_path"], str(run_dir / "seed-11" / "checkpoint.pt"))
        self.assertEqual(training["last_checkpoint_path"], str(run_dir / "seed-13" / "checkpoint.pt"))
        for seed in (11, 13):
            self.assertTrue(checkpoint_exists[seed])
            self.assertTrue(loss_log_exists[seed])

        utility_stats = training["multi_seed_summary"]["utility"]["final_coverage_rate"]
        self.assertEqual(utility_stats["mean"], 0.2)
        self.assertEqual(utility_stats["std"], 0.0)
        self.assertEqual(utility_stats["min"], 0.2)
        self.assertEqual(utility_stats["max"], 0.2)
        self.assertIn("torch_policy", training["multi_seed_summary"])
        self.assertIn("## Dataset Validation Gates", report)
        self.assertIn("## Multi-Seed Summary", report)
        self.assertIn("## Best Checkpoint", report)
        self.assertIn("best_checkpoint_path", report)

    def test_explicit_split_training_writes_per_epoch_logs_and_baseline_deltas(self):
        from model_explorer.policy.experiment import run_experiment_manifest

        train_payload = minimal_contract(
            goals=[
                {"cell": [1, 1], "utility": 0.5, "reachable": True, "expected_coverage_rate_delta": 0.2},
                {"cell": [2, 1], "utility": 9.0, "reachable": False, "expected_coverage_rate_delta": 0.9},
            ],
            observation_update={"coverage_rate": 0.2, "coverage_rate_delta": 0.2},
        )
        validation_payload = minimal_contract(
            goals=[{"cell": [1, 2], "utility": 0.5, "reachable": True, "expected_coverage_rate_delta": 0.3}],
            observation_update={"coverage_rate": 0.3, "coverage_rate_delta": 0.1},
        )
        benchmark_payload = minimal_contract(
            goals=[{"cell": [3, 1], "utility": 0.5, "reachable": True, "expected_coverage_rate_delta": 0.4}],
            observation_update={"coverage_rate": 0.4, "coverage_rate_delta": 0.1},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            train_path = Path(tmpdir) / "train.json"
            validation_path = Path(tmpdir) / "validation.json"
            benchmark_path = Path(tmpdir) / "benchmark.json"
            train_path.write_text(json.dumps(train_payload), encoding="utf-8")
            validation_path.write_text(json.dumps(validation_payload), encoding="utf-8")
            benchmark_path.write_text(json.dumps(benchmark_payload), encoding="utf-8")
            output_root = Path(tmpdir) / "out"
            manifest_path = Path(tmpdir) / "experiment.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "name": "explicit-training",
                        "run_id": "run-001",
                        "splits": {
                            "train": [str(train_path)],
                            "validation": [str(validation_path)],
                            "benchmark": {"coverage_dominant": [str(benchmark_path)]},
                        },
                        "max_candidates": 2,
                        "outputs": {"root": str(output_root)},
                        "train": {
                            "seed": 41,
                            "hidden_size": 16,
                            "epochs": 2,
                            "evaluate_trained_policy": True,
                        },
                    }
                ),
                encoding="utf-8",
            )

            summary = run_experiment_manifest(manifest_path)
            run_dir = output_root / "explicit-training" / "run-001"
            seed_dir = run_dir / "seed-41"
            loss_records = [
                json.loads(line)
                for line in (seed_dir / "losses.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            training_summary = json.loads((seed_dir / "training-summary.json").read_text(encoding="utf-8"))
            report = (run_dir / "report.md").read_text(encoding="utf-8")

        self.assertEqual(summary["training"]["train_episode_count"], 1)
        self.assertEqual(summary["training"]["validation_episode_count"], 1)
        self.assertEqual([record["epoch"] for record in loss_records], [1, 2])
        self.assertIn("loss", loss_records[0])
        self.assertIn("validation_evaluation", training_summary)
        self.assertIn("baseline_deltas", summary)
        self.assertIn("torch_policy", summary["baseline_deltas"])
        self.assertIn("utility", summary["baseline_deltas"]["torch_policy"])
        self.assertIn("multi_seed_loss_summary", summary["training"])
        self.assertIn("warnings", summary["training"]["runs"][0])
        self.assertIn("## Baseline Comparison", report)
        self.assertIn("## Training Quality", report)


if __name__ == "__main__":
    unittest.main()
