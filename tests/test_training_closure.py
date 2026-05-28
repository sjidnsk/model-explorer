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
        self.assertEqual(metadata["format"], "model-explorer-masked-policy")
        self.assertEqual(metadata["version"], 2)
        self.assertEqual(metadata["candidate_feature_names"], checkpoint["candidate_feature_names"])
        self.assertEqual(metadata["global_feature_names"], checkpoint["global_feature_names"])
        self.assertEqual(metadata["action_count"], 2)
        self.assertEqual(metadata["seed"], 23)
        self.assertEqual(metadata["sample_count"], 1)
        self.assertEqual(metadata["epoch_count"], 2)
        self.assertEqual(metadata["hidden_size"], 16)
        self.assertEqual(metadata["learning_rate"], 5.0e-4)
        self.assertIsNotNone(old_scorer)

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
                            "seed": 29,
                            "hidden_size": 16,
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
        self.assertIn("torch_policy", evaluation)
        self.assertIn("torch_policy", summary["training"]["baseline_evaluation"])
        self.assertIn("## Training", report)
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


if __name__ == "__main__":
    unittest.main()
