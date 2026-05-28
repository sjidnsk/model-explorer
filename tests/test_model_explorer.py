import json
import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

DEV_PLATFORM_CONTRACT_EXAMPLE = (
    ROOT.parent / "dev-platform-constraints" / "docs" / "model-explorer-contract-example.json"
)

from model_explorer.core.interfaces import ContractValidationError
from model_explorer.decision.selector import select_goal
from model_explorer.io.scenario import load_scenario
from model_explorer.orchestration.loop import run_exploration_loop


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
        else [
            {
                "cell": [2, 1],
                "utility": 0.42,
                "reachable": True,
            }
        ],
        "top_sequences": sequences
        if sequences is not None
        else [
            {
                "cells": [[2, 1]],
                "utility": 0.42,
                "coverage_area": 1.0,
            }
        ],
        "observation_update": observation_update
        if observation_update is not None
        else {"delta_c": 0.25, "visible_cell_count": 3, "updated_cell_count": 3},
    }


class ScenarioLoadingTests(unittest.TestCase):
    def test_load_single_contract_and_convert_cell_to_world(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "scenario.json"
            path.write_text(json.dumps(minimal_contract()), encoding="utf-8")

            scenario = load_scenario(path)

        self.assertEqual(len(scenario.snapshots), 1)
        contract = scenario.snapshots[0]
        self.assertEqual(contract.grid.width, 4)
        self.assertEqual(contract.top_goals[0].cell, (2, 1))
        self.assertEqual(contract.cell_to_world((2, 1)), (2.0, 2.5))

    def test_load_multi_snapshot_scenario(self):
        payload = {
            "snapshots": [
                minimal_contract(goals=[{"cell": [1, 1], "utility": 0.3, "reachable": True}]),
                minimal_contract(goals=[{"cell": [2, 1], "utility": 0.5, "reachable": True}]),
            ]
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "scenario.json"
            path.write_text(json.dumps(payload), encoding="utf-8")

            scenario = load_scenario(path)

        self.assertEqual(len(scenario.snapshots), 2)
        self.assertEqual(scenario.snapshots[1].top_goals[0].cell, (2, 1))

    def test_rejects_wrong_schema_version(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "scenario.json"
            path.write_text(
                json.dumps(minimal_contract(schema_version="model-explorer-contract/v2")),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ContractValidationError, "schema_version"):
                load_scenario(path)

    def test_rejects_missing_core_field(self):
        payload = minimal_contract()
        del payload["top_goals"]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "scenario.json"
            path.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(ContractValidationError, "top_goals"):
                load_scenario(path)


class GoalSelectionTests(unittest.TestCase):
    def test_selects_only_reachable_goal_with_highest_utility(self):
        contract = load_contract_from_dict(
            minimal_contract(
                goals=[
                    {"cell": [0, 0], "utility": 10.0, "reachable": False},
                    {"cell": [3, 1], "utility": 0.7, "reachable": True},
                    {"cell": [1, 2], "utility": 0.4, "reachable": True},
                ]
            )
        )

        decision = select_goal(contract)

        self.assertEqual(decision.selected_goal.cell, (3, 1))
        self.assertEqual(decision.status, "selected")

    def test_utility_ties_are_broken_by_cell_order(self):
        contract = load_contract_from_dict(
            minimal_contract(
                goals=[
                    {"cell": [2, 1], "utility": 0.5, "reachable": True},
                    {"cell": [1, 2], "utility": 0.5, "reachable": True},
                    {"cell": [1, 1], "utility": 0.5, "reachable": True},
                ]
            )
        )

        decision = select_goal(contract)

        self.assertEqual(decision.selected_goal.cell, (1, 1))

    def test_returns_no_reachable_goal_when_all_candidates_are_unreachable(self):
        contract = load_contract_from_dict(
            minimal_contract(
                goals=[
                    {"cell": [0, 0], "utility": 0.8, "reachable": False},
                    {"cell": [1, 1], "utility": 0.7, "reachable": False},
                ]
            )
        )

        decision = select_goal(contract)

        self.assertIsNone(decision.selected_goal)
        self.assertEqual(decision.status, "no_reachable_goal")

    def test_missing_experimental_fields_do_not_affect_selection(self):
        contract = load_contract_from_dict(
            minimal_contract(goals=[{"cell": [2, 1], "utility": 0.42, "reachable": True}])
        )

        decision = select_goal(contract)

        self.assertEqual(decision.selected_goal.cell, (2, 1))
        self.assertEqual(decision.selected_goal.experimental, {})

    def test_coverage_priority_can_beat_utility_when_experimental_fields_exist(self):
        contract = load_contract_from_dict(
            minimal_contract(
                goals=[
                    {
                        "cell": [0, 0],
                        "utility": 1.0,
                        "reachable": True,
                        "expected_coverage_rate_delta": 0.05,
                    },
                    {
                        "cell": [1, 1],
                        "utility": 0.2,
                        "reachable": True,
                        "expected_coverage_rate_delta": 0.5,
                    },
                    {
                        "cell": [2, 2],
                        "utility": 0.1,
                        "reachable": False,
                        "expected_coverage_rate_delta": 0.9,
                    },
                ]
            )
        )

        decision = select_goal(contract)

        self.assertEqual(decision.selected_goal.cell, (1, 1))
        self.assertNotIn((2, 2), [goal.cell for goal in decision.ranked_goals])


class LoopTests(unittest.TestCase):
    def test_goal_change_triggers_replan_reason(self):
        scenario = [
            load_contract_from_dict(
                minimal_contract(
                    goals=[{"cell": [1, 1], "utility": 0.5, "reachable": True}],
                    observation_update={"delta_c": 0.0},
                )
            ),
            load_contract_from_dict(
                minimal_contract(
                    goals=[{"cell": [2, 1], "utility": 0.6, "reachable": True}],
                    observation_update={"delta_c": 0.0},
                )
            ),
        ]

        results = run_exploration_loop(scenario)

        self.assertEqual(results[1].decision.selected_goal.cell, (2, 1))
        self.assertIn("goal_changed", results[1].replan_reasons)

    def test_observation_delta_at_threshold_triggers_replan_reason(self):
        scenario = [
            load_contract_from_dict(
                minimal_contract(
                    goals=[{"cell": [1, 1], "utility": 0.5, "reachable": True}],
                    observation_update={"delta_c": 0.1},
                )
            )
        ]

        results = run_exploration_loop(scenario)

        self.assertIn("observation_delta", results[0].replan_reasons)

    def test_no_reachable_goal_triggers_replan_reason(self):
        scenario = [
            load_contract_from_dict(
                minimal_contract(goals=[{"cell": [1, 1], "utility": 0.5, "reachable": False}])
            )
        ]

        results = run_exploration_loop(scenario)

        self.assertIn("no_reachable_goal", results[0].replan_reasons)


class PolicyFeatureExtractionTests(unittest.TestCase):
    def test_extracts_candidate_features_and_action_mask_from_top_goals(self):
        from model_explorer.policy.features import extract_policy_observation

        contract = load_contract_from_dict(
            minimal_contract(
                goals=[
                    {
                        "cell": [1, 1],
                        "utility": 0.7,
                        "reachable": True,
                        "expected_coverage_rate_delta": 0.3,
                        "risk": 0.4,
                    },
                    {
                        "cell": [3, 2],
                        "utility": 0.9,
                        "reachable": False,
                        "expected_new_coverage_area": 8.0,
                        "path_cost": 4.0,
                    },
                ],
                observation_update={"coverage_rate": 0.25},
            )
        )

        observation = extract_policy_observation(
            contract,
            current_cell=(0, 1),
            step_index=2,
            remaining_steps=5,
            max_candidates=3,
        )

        feature_names = observation.candidate_feature_names
        first_features = dict(zip(feature_names, observation.candidate_features[0]))
        second_features = dict(zip(feature_names, observation.candidate_features[1]))

        self.assertEqual(observation.action_mask, (True, False, False))
        self.assertEqual(observation.candidate_cells, ((1, 1), (3, 2), None))
        self.assertAlmostEqual(first_features["cell_x"], 0.25)
        self.assertAlmostEqual(first_features["cell_y"], 1.0 / 3.0)
        self.assertAlmostEqual(first_features["relative_dx"], 0.25)
        self.assertAlmostEqual(first_features["relative_dy"], 0.0)
        self.assertAlmostEqual(first_features["relative_distance"], 0.2)
        self.assertEqual(first_features["utility"], 0.7)
        self.assertEqual(first_features["reachable"], 1.0)
        self.assertEqual(first_features["expected_coverage_rate_delta"], 0.3)
        self.assertEqual(first_features["path_cost"], 4.0)
        self.assertEqual(second_features["reachable"], 0.0)

        global_features = dict(zip(observation.global_feature_names, observation.global_features))
        self.assertEqual(global_features["grid_width"], 4.0)
        self.assertEqual(global_features["grid_height"], 3.0)
        self.assertEqual(global_features["coverage_rate"], 0.25)
        self.assertEqual(global_features["step_index"], 2.0)
        self.assertEqual(global_features["remaining_steps"], 5.0)

    def test_missing_experimental_features_use_compatibility_defaults(self):
        from model_explorer.policy.features import extract_policy_observation

        contract = load_contract_from_dict(
            minimal_contract(
                goals=[
                    {"cell": [2, 1], "utility": 0.42, "reachable": True},
                ],
                observation_update={},
            )
        )

        observation = extract_policy_observation(contract)

        feature_names = observation.candidate_feature_names
        features = dict(zip(feature_names, observation.candidate_features[0]))
        global_features = dict(zip(observation.global_feature_names, observation.global_features))

        self.assertEqual(observation.action_mask, (True,))
        self.assertEqual(features["expected_coverage_rate_delta"], 0.0)
        self.assertEqual(features["risk"], 0.0)
        self.assertEqual(features["path_cost"], 0.0)
        self.assertEqual(global_features["coverage_rate"], 0.0)


class RolloutLoggingTests(unittest.TestCase):
    def test_transition_serializes_observation_action_reward_and_info(self):
        from model_explorer.policy.features import extract_policy_observation
        from model_explorer.policy.rollout import RolloutInfo, RolloutTransition

        contract = load_contract_from_dict(
            minimal_contract(
                goals=[
                    {
                        "cell": [1, 1],
                        "utility": 0.7,
                        "reachable": True,
                        "risk": 0.2,
                        "path_cost": 3.5,
                    },
                    {"cell": [2, 2], "utility": 0.9, "reachable": False},
                ],
                observation_update={"coverage_rate": 0.2},
            )
        )
        next_contract = load_contract_from_dict(
            minimal_contract(
                goals=[{"cell": [3, 1], "utility": 0.8, "reachable": True}],
                observation_update={"coverage_rate": 0.32, "coverage_rate_delta": 0.12},
            )
        )
        observation = extract_policy_observation(contract, max_candidates=2)
        next_observation = extract_policy_observation(next_contract, max_candidates=2)

        transition = RolloutTransition(
            observation=observation,
            action_index=0,
            log_prob=-0.5,
            value=0.25,
            reward=0.12,
            next_observation=next_observation,
            done=False,
            info=RolloutInfo(
                selected_cell=(1, 1),
                coverage_rate_delta=0.12,
                path_cost=3.5,
                risk=0.2,
                failure_reason=None,
                final_coverage_rate=None,
                total_cost=3.5,
                failure_count=0,
                replan_count=1,
            ),
        )

        payload = transition.to_dict()

        self.assertEqual(payload["action_index"], 0)
        self.assertEqual(payload["action_mask"], [True, False])
        self.assertEqual(payload["reward"], 0.12)
        self.assertEqual(payload["log_prob"], -0.5)
        self.assertEqual(payload["value"], 0.25)
        self.assertFalse(payload["done"])
        self.assertEqual(payload["observation"]["candidate_cells"], [[1, 1], [2, 2]])
        self.assertEqual(payload["next_observation"]["candidate_cells"], [[3, 1], None])
        self.assertEqual(payload["info"]["selected_cell"], [1, 1])
        self.assertEqual(payload["info"]["coverage_rate_delta"], 0.12)
        self.assertEqual(payload["info"]["path_cost"], 3.5)
        self.assertEqual(payload["info"]["replan_count"], 1)

    def test_transition_rejects_masked_action_index(self):
        from model_explorer.policy.features import extract_policy_observation
        from model_explorer.policy.rollout import RolloutInfo, RolloutTransition

        contract = load_contract_from_dict(
            minimal_contract(
                goals=[
                    {"cell": [1, 1], "utility": 0.7, "reachable": True},
                    {"cell": [2, 2], "utility": 0.9, "reachable": False},
                ]
            )
        )
        observation = extract_policy_observation(contract, max_candidates=2)

        with self.assertRaisesRegex(ValueError, "masked action"):
            RolloutTransition(
                observation=observation,
                action_index=1,
                log_prob=None,
                value=None,
                reward=0.0,
                next_observation=None,
                done=True,
                info=RolloutInfo(selected_cell=(2, 2), failure_reason="unreachable"),
            )

    def test_episode_serializes_transitions_and_episode_metrics(self):
        from model_explorer.policy.features import extract_policy_observation
        from model_explorer.policy.rollout import EpisodeMetrics, RolloutEpisode, RolloutInfo, RolloutTransition

        contract = load_contract_from_dict(
            minimal_contract(goals=[{"cell": [1, 1], "utility": 0.7, "reachable": True}])
        )
        observation = extract_policy_observation(contract)
        transition = RolloutTransition(
            observation=observation,
            action_index=0,
            log_prob=None,
            value=None,
            reward=0.2,
            next_observation=None,
            done=True,
            info=RolloutInfo(selected_cell=(1, 1), coverage_rate_delta=0.2),
        )

        episode = RolloutEpisode(
            transitions=(transition,),
            metrics=EpisodeMetrics(
                final_coverage_rate=0.45,
                cumulative_coverage_rate_delta=0.2,
                total_path_cost=3.5,
                average_risk=0.1,
                failure_count=1,
                replan_count=2,
                value_coverage=0.3,
            ),
        )

        payload = episode.to_dict()

        self.assertEqual(len(payload["transitions"]), 1)
        self.assertEqual(payload["metrics"]["final_coverage_rate"], 0.45)
        self.assertEqual(payload["metrics"]["total_path_cost"], 3.5)
        self.assertEqual(payload["metrics"]["failure_count"], 1)
        self.assertEqual(payload["metrics"]["replan_count"], 2)


class PolicySelectionInterfaceTests(unittest.TestCase):
    def test_policy_scores_can_rank_reachable_candidates(self):
        class StubPolicy:
            def __init__(self):
                self.seen_action_mask = None

            def score(self, observation):
                self.seen_action_mask = observation.action_mask
                return (0.1, 0.9, 99.0)

        policy = StubPolicy()
        contract = load_contract_from_dict(
            minimal_contract(
                goals=[
                    {"cell": [0, 0], "utility": 1.0, "reachable": True},
                    {"cell": [1, 1], "utility": 0.2, "reachable": True},
                    {"cell": [2, 2], "utility": 0.1, "reachable": False},
                ]
            )
        )

        decision = select_goal(contract, policy=policy)

        self.assertEqual(policy.seen_action_mask, (True, True, False))
        self.assertEqual(decision.selected_goal.cell, (1, 1))
        self.assertNotIn((2, 2), [goal.cell for goal in decision.ranked_goals])

    def test_invalid_policy_scores_fall_back_to_heuristic_selection(self):
        class InvalidPolicy:
            def score(self, observation):
                return (0.1,)

        contract = load_contract_from_dict(
            minimal_contract(
                goals=[
                    {"cell": [0, 0], "utility": 1.0, "reachable": True},
                    {"cell": [1, 1], "utility": 0.2, "reachable": True},
                ]
            )
        )

        decision = select_goal(contract, policy=InvalidPolicy())

        self.assertEqual(decision.selected_goal.cell, (0, 0))


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not available")
class TorchPolicyNetworkTests(unittest.TestCase):
    def test_masked_network_outputs_zero_probability_for_invalid_actions(self):
        import torch

        from model_explorer.policy.features import extract_policy_observation
        from model_explorer.policy.torch_policy import MaskedCandidatePolicyNetwork, observation_to_tensors

        contract = load_contract_from_dict(
            minimal_contract(
                goals=[
                    {"cell": [0, 0], "utility": 0.5, "reachable": True},
                    {"cell": [1, 1], "utility": 0.9, "reachable": False},
                    {"cell": [2, 1], "utility": 0.4, "reachable": True},
                ]
            )
        )
        observation = extract_policy_observation(contract, max_candidates=4)
        tensors = observation_to_tensors(observation)
        network = MaskedCandidatePolicyNetwork(
            candidate_feature_count=len(observation.candidate_feature_names),
            global_feature_count=len(observation.global_feature_names),
            hidden_size=16,
        )

        output = network(**tensors)

        self.assertEqual(tuple(output.masked_logits.shape), (1, 4))
        self.assertEqual(tuple(output.action_probs.shape), (1, 4))
        self.assertEqual(tuple(output.value.shape), (1,))
        self.assertTrue(torch.isfinite(output.value).all())
        action_probs = output.action_probs.detach()
        self.assertAlmostEqual(float(action_probs[0, 1]), 0.0)
        self.assertAlmostEqual(float(action_probs[0, 3]), 0.0)
        self.assertAlmostEqual(float(action_probs.sum()), 1.0, places=6)

    def test_torch_policy_scorer_plugs_into_selector_without_selecting_masked_candidate(self):
        from model_explorer.policy.features import extract_policy_observation
        from model_explorer.policy.torch_policy import MaskedCandidatePolicyNetwork, TorchPolicyScorer

        contract = load_contract_from_dict(
            minimal_contract(
                goals=[
                    {"cell": [0, 0], "utility": 0.5, "reachable": True},
                    {"cell": [1, 1], "utility": 5.0, "reachable": False},
                    {"cell": [2, 1], "utility": 0.4, "reachable": True},
                ]
            )
        )
        observation = extract_policy_observation(contract)
        network = MaskedCandidatePolicyNetwork(
            candidate_feature_count=len(observation.candidate_feature_names),
            global_feature_count=len(observation.global_feature_names),
            hidden_size=16,
        )

        decision = select_goal(contract, policy=TorchPolicyScorer(network))

        self.assertIsNotNone(decision.selected_goal)
        self.assertNotEqual(decision.selected_goal.cell, (1, 1))


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not available")
class MaskedPpoTests(unittest.TestCase):
    def test_masked_ppo_loss_is_finite_and_backpropagates(self):
        import torch

        from model_explorer.policy.features import extract_policy_observation
        from model_explorer.policy.ppo import compute_masked_ppo_loss
        from model_explorer.policy.torch_policy import MaskedCandidatePolicyNetwork, observation_to_tensors

        contract = load_contract_from_dict(
            minimal_contract(
                goals=[
                    {"cell": [0, 0], "utility": 0.5, "reachable": True},
                    {"cell": [1, 1], "utility": 0.9, "reachable": False},
                    {"cell": [2, 1], "utility": 0.4, "reachable": True},
                ]
            )
        )
        observation = extract_policy_observation(contract, max_candidates=3)
        tensors = observation_to_tensors(observation)
        network = MaskedCandidatePolicyNetwork(
            candidate_feature_count=len(observation.candidate_feature_names),
            global_feature_count=len(observation.global_feature_names),
            hidden_size=16,
        )

        losses = compute_masked_ppo_loss(
            network,
            **tensors,
            actions=torch.tensor([0]),
            old_log_probs=torch.tensor([-0.7]),
            returns=torch.tensor([0.3]),
            advantages=torch.tensor([0.2]),
        )
        losses.total_loss.backward()
        grad_norm = sum(
            float(parameter.grad.abs().sum())
            for parameter in network.parameters()
            if parameter.grad is not None
        )

        self.assertTrue(torch.isfinite(losses.total_loss))
        self.assertTrue(torch.isfinite(losses.policy_loss))
        self.assertTrue(torch.isfinite(losses.value_loss))
        self.assertTrue(torch.isfinite(losses.entropy))
        self.assertGreater(grad_norm, 0.0)


@unittest.skipUnless(DEV_PLATFORM_CONTRACT_EXAMPLE.exists(), "dev-platform-constraints contract example is not present")
class DevPlatformContractIntegrationTests(unittest.TestCase):
    def test_documented_contract_example_runs_through_loop_and_cli(self):
        scenario = load_scenario(DEV_PLATFORM_CONTRACT_EXAMPLE)

        results = run_exploration_loop(scenario)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].decision.selected_goal.cell, (24, 10))
        self.assertEqual(scenario.snapshots[0].cell_to_world((24, 10)), (12.0, 5.0))
        self.assertIn("observation_delta", results[0].replan_reasons)

        env = os.environ.copy()
        env["PYTHONPATH"] = str(SRC)
        completed = subprocess.run(
            [sys.executable, "-m", "model_explorer", str(DEV_PLATFORM_CONTRACT_EXAMPLE)],
            cwd=ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )

        output = json.loads(completed.stdout)
        self.assertEqual(output[0]["status"], "selected")
        self.assertEqual(output[0]["selected_cell"], [24, 10])
        self.assertEqual(output[0]["selected_world"], [12.0, 5.0])
        self.assertEqual(output[0]["observation_update"]["coverage_rate_delta"], 0.071875)


class RolloutCollectorTests(unittest.TestCase):
    def test_collect_rollout_episode_records_transitions_for_multi_snapshot_scenario(self):
        from model_explorer.policy.collector import collect_rollout_episode

        scenario = [
            load_contract_from_dict(
                minimal_contract(
                    goals=[
                        {
                            "cell": [1, 1],
                            "utility": 0.4,
                            "reachable": True,
                            "path_cost": 2.0,
                            "risk": 0.1,
                        }
                    ],
                    observation_update={"coverage_rate": 0.1, "coverage_rate_delta": 0.05},
                )
            ),
            load_contract_from_dict(
                minimal_contract(
                    goals=[
                        {
                            "cell": [2, 1],
                            "utility": 0.5,
                            "reachable": True,
                            "path_cost": 3.0,
                            "risk": 0.2,
                        }
                    ],
                    observation_update={"coverage_rate": 0.18, "coverage_rate_delta": 0.08},
                )
            ),
        ]

        episode = collect_rollout_episode(scenario, max_candidates=2)

        self.assertEqual(len(episode.transitions), 2)
        self.assertEqual(episode.transitions[0].action_index, 0)
        self.assertEqual(episode.transitions[0].next_observation.candidate_cells, ((2, 1), None))
        self.assertTrue(episode.transitions[-1].done)
        self.assertAlmostEqual(episode.metrics.final_coverage_rate, 0.18)
        self.assertAlmostEqual(episode.metrics.cumulative_coverage_rate_delta, 0.13)
        self.assertAlmostEqual(episode.metrics.total_path_cost, 5.0)

    def test_reward_uses_compatibility_defaults_when_experimental_fields_are_missing(self):
        from model_explorer.policy.reward import compute_step_reward

        goal = load_contract_from_dict(
            minimal_contract(goals=[{"cell": [2, 1], "utility": 0.42, "reachable": True}])
        ).top_goals[0]

        reward_info = compute_step_reward(goal, {})

        self.assertEqual(reward_info.reward, 0.0)
        self.assertEqual(reward_info.coverage_rate_delta, 0.0)
        self.assertEqual(reward_info.path_cost, 0.0)
        self.assertEqual(reward_info.risk, 0.0)


class RolloutIoTests(unittest.TestCase):
    def test_rollout_episode_round_trips_through_json_file(self):
        from model_explorer.policy.collector import collect_rollout_episode
        from model_explorer.policy.rollout_io import read_rollout_episode, write_rollout_episode

        scenario = [
            load_contract_from_dict(
                minimal_contract(
                    goals=[{"cell": [1, 1], "utility": 0.4, "reachable": True}],
                    observation_update={"coverage_rate": 0.1, "coverage_rate_delta": 0.05},
                )
            )
        ]
        episode = collect_rollout_episode(scenario, max_candidates=2)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "rollout.json"
            write_rollout_episode(path, episode)
            loaded = read_rollout_episode(path)

        self.assertEqual(loaded.transitions[0].action_index, episode.transitions[0].action_index)
        self.assertEqual(loaded.transitions[0].observation.action_mask, (True, False))
        self.assertEqual(loaded.transitions[0].reward, episode.transitions[0].reward)
        self.assertEqual(loaded.metrics.final_coverage_rate, episode.metrics.final_coverage_rate)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not available")
class PolicyTrainingTests(unittest.TestCase):
    def test_training_step_produces_finite_losses_and_checkpoint_policy_is_mask_safe(self):
        import torch

        from model_explorer.policy.collector import collect_rollout_episode
        from model_explorer.policy.training import load_policy_checkpoint, train_policy_on_episode

        scenario = [
            load_contract_from_dict(
                minimal_contract(
                    goals=[
                        {"cell": [0, 0], "utility": 0.5, "reachable": True},
                        {"cell": [1, 1], "utility": 9.0, "reachable": False},
                        {"cell": [2, 1], "utility": 0.4, "reachable": True},
                    ],
                    observation_update={"coverage_rate": 0.1, "coverage_rate_delta": 0.05},
                )
            )
        ]
        episode = collect_rollout_episode(scenario, max_candidates=3)

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "policy.pt"
            result = train_policy_on_episode(
                episode,
                checkpoint_path=checkpoint_path,
                seed=7,
                hidden_size=16,
            )
            self.assertTrue(checkpoint_path.exists())
            scorer = load_policy_checkpoint(checkpoint_path)

        self.assertTrue(torch.isfinite(torch.tensor(result["total_loss"])))
        self.assertIn("entropy", result)

        decision = select_goal(scenario[0], policy=scorer)

        self.assertIsNotNone(decision.selected_goal)
        self.assertNotEqual(decision.selected_goal.cell, (1, 1))


class BaselineEvaluationTests(unittest.TestCase):
    def test_baseline_evaluation_outputs_core_metrics(self):
        from model_explorer.policy.evaluation import evaluate_policy_baselines

        scenario = [
            load_contract_from_dict(
                minimal_contract(
                    goals=[
                        {
                            "cell": [0, 0],
                            "utility": 0.9,
                            "reachable": True,
                            "path_cost": 2.0,
                            "risk": 0.1,
                            "expected_coverage_rate_delta": 0.01,
                        },
                        {
                            "cell": [1, 1],
                            "utility": 0.2,
                            "reachable": True,
                            "path_cost": 3.0,
                            "risk": 0.2,
                            "expected_coverage_rate_delta": 0.5,
                        },
                    ],
                    observation_update={"coverage_rate": 0.1, "coverage_rate_delta": 0.05},
                )
            )
        ]

        report = evaluate_policy_baselines(scenario)

        self.assertIn("utility", report)
        self.assertIn("coverage_heuristic", report)
        self.assertEqual(report["utility"]["selected_cells"], [[0, 0]])
        self.assertEqual(report["coverage_heuristic"]["selected_cells"], [[1, 1]])
        for metrics in report.values():
            self.assertIn("final_coverage_rate", metrics)
            self.assertIn("cumulative_coverage_rate_delta", metrics)
            self.assertIn("total_path_cost", metrics)
            self.assertIn("average_risk", metrics)
            self.assertIn("failure_count", metrics)
            self.assertIn("replan_count", metrics)


class PolicyScriptTests(unittest.TestCase):
    def test_collect_rollout_script_writes_episode_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            scenario_path = Path(tmpdir) / "scenario.json"
            rollout_path = Path(tmpdir) / "rollout.json"
            scenario_path.write_text(json.dumps(minimal_contract()), encoding="utf-8")

            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "collect_rollout.py"),
                    str(scenario_path),
                    str(rollout_path),
                ],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )

            payload = json.loads(rollout_path.read_text(encoding="utf-8"))

        summary = json.loads(completed.stdout)
        self.assertEqual(summary["transition_count"], 1)
        self.assertEqual(len(payload["transitions"]), 1)

    def test_evaluate_baselines_script_outputs_metrics_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            scenario_path = Path(tmpdir) / "scenario.json"
            scenario_path.write_text(json.dumps(minimal_contract()), encoding="utf-8")

            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "evaluate_baselines.py"),
                    str(scenario_path),
                ],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )

        report = json.loads(completed.stdout)
        self.assertIn("utility", report)
        self.assertIn("coverage_heuristic", report)

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not available")
    def test_train_masked_policy_script_saves_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            scenario_path = Path(tmpdir) / "scenario.json"
            rollout_path = Path(tmpdir) / "rollout.json"
            checkpoint_path = Path(tmpdir) / "policy.pt"
            scenario_path.write_text(json.dumps(minimal_contract()), encoding="utf-8")
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "collect_rollout.py"),
                    str(scenario_path),
                    str(rollout_path),
                ],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "train_masked_policy.py"),
                    str(rollout_path),
                    str(checkpoint_path),
                    "--hidden-size",
                    "16",
                ],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )

            result = json.loads(completed.stdout)
            checkpoint_exists = checkpoint_path.exists()

        self.assertTrue(checkpoint_exists)
        self.assertIn("total_loss", result)


class CliTests(unittest.TestCase):
    def test_module_cli_outputs_selected_goal_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "scenario.json"
            path.write_text(json.dumps(minimal_contract()), encoding="utf-8")

            env = os.environ.copy()
            env["PYTHONPATH"] = str(SRC)
            completed = subprocess.run(
                [sys.executable, "-m", "model_explorer", str(path)],
                cwd=ROOT,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )

        output = json.loads(completed.stdout)
        self.assertEqual(output[0]["status"], "selected")
        self.assertEqual(output[0]["selected_cell"], [2, 1])

    def test_script_cli_outputs_selected_goal_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "scenario.json"
            path.write_text(json.dumps(minimal_contract()), encoding="utf-8")

            completed = subprocess.run(
                [sys.executable, str(ROOT / "scripts" / "run_model_explorer.py"), str(path)],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )

        output = json.loads(completed.stdout)
        self.assertEqual(output[0]["status"], "selected")
        self.assertEqual(output[0]["selected_cell"], [2, 1])


class StructureTests(unittest.TestCase):
    def test_checkout_uses_src_scripts_tests_layout(self):
        self.assertTrue((ROOT / "src" / "model_explorer" / "__init__.py").exists())
        self.assertTrue((ROOT / "scripts" / "run_model_explorer.py").exists())
        self.assertTrue((ROOT / "tests" / "test_model_explorer.py").exists())
        self.assertFalse((ROOT / "model_explorer").exists())

    def test_core_code_is_grouped_by_responsibility(self):
        expected_files = [
            ROOT / "src" / "model_explorer" / "core" / "interfaces.py",
            ROOT / "src" / "model_explorer" / "io" / "scenario.py",
            ROOT / "src" / "model_explorer" / "decision" / "selector.py",
            ROOT / "src" / "model_explorer" / "orchestration" / "loop.py",
        ]
        for path in expected_files:
            with self.subTest(path=path):
                self.assertTrue(path.exists())

    def test_legacy_compatibility_modules_are_removed(self):
        legacy_modules = [
            ROOT / "src" / "model_explorer" / "interfaces.py",
            ROOT / "src" / "model_explorer" / "scenario.py",
            ROOT / "src" / "model_explorer" / "selector.py",
            ROOT / "src" / "model_explorer" / "loop.py",
        ]
        for path in legacy_modules:
            with self.subTest(path=path):
                self.assertFalse(path.exists())


def load_contract_from_dict(payload):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "contract.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return load_scenario(path).snapshots[0]


if __name__ == "__main__":
    unittest.main()
