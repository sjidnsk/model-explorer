import json
import importlib.util
import math
import os
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

DEV_PLATFORM_CONTRACT_EXAMPLE = (
    ROOT.parent / "dev-platform-constraints" / "docs" / "model-explorer-contract-example.json"
)
DEV_PLATFORM_ROOT = ROOT.parent / "dev-platform-constraints"
SYNTHETIC_EXPERIMENT_FIXTURE = ROOT / "tests" / "fixtures" / "synthetic_experiment"

from model_explorer.core.interfaces import ContractValidationError
from model_explorer.core.interfaces import GridSummary
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


def replace_contract_grid(contract, *, width, height, resolution):
    return replace(
        contract,
        grid=GridSummary(
            width=width,
            height=height,
            resolution=resolution,
            frame_id=contract.grid.frame_id,
            origin=contract.grid.origin,
            layers=contract.grid.layers,
        ),
    )


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
        self.assertEqual(first_features["path_cost"], 1.0)
        self.assertEqual(second_features["reachable"], 0.0)

        global_features = dict(zip(observation.global_feature_names, observation.global_features))
        self.assertGreater(global_features["grid_width"], 0.0)
        self.assertLessEqual(global_features["grid_width"], 1.0)
        self.assertGreater(global_features["grid_height"], 0.0)
        self.assertLessEqual(global_features["grid_height"], 1.0)
        self.assertEqual(global_features["coverage_rate"], 0.25)
        self.assertAlmostEqual(global_features["step_index"], 2.0 / 7.0)
        self.assertAlmostEqual(global_features["remaining_steps"], 5.0 / 7.0)

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

    def test_missing_indicators_distinguish_real_zero_from_fallback_zero(self):
        from model_explorer.policy.features import MISSING_INDICATOR_NAMES, extract_policy_observation

        explicit_zero_goal = {
            "cell": [1, 1],
            "utility": 0.0,
            "reachable": True,
            "expected_coverage_rate_delta": 0.0,
            "expected_new_coverage_area": 0.0,
            "information_gain": 0.0,
            "confidence_gain": 0.0,
            "value": 0.0,
            "risk": 0.0,
            "path_cost": 0.0,
            "energy_cost": 0.0,
        }
        missing_goal = {"cell": [2, 1], "utility": 0.0, "reachable": True}
        contract = load_contract_from_dict(minimal_contract(goals=[explicit_zero_goal, missing_goal]))

        observation = extract_policy_observation(contract, max_candidates=3)

        self.assertEqual(observation.candidate_missing_indicator_names, MISSING_INDICATOR_NAMES)
        self.assertEqual(observation.candidate_missing_indicators[0], tuple(0.0 for _ in MISSING_INDICATOR_NAMES))
        self.assertEqual(observation.candidate_missing_indicators[1], tuple(1.0 for _ in MISSING_INDICATOR_NAMES))
        self.assertEqual(observation.candidate_missing_indicators[2], tuple(0.0 for _ in MISSING_INDICATOR_NAMES))
        self.assertEqual(observation.candidate_missing_feature_names[0], ())
        self.assertEqual(set(observation.candidate_missing_feature_names[1]), {
            name.removesuffix("_missing") for name in MISSING_INDICATOR_NAMES
        })

    def test_feature_normalization_keeps_extreme_observation_tensors_finite(self):
        import math

        from model_explorer.policy.features import extract_policy_observation

        payload = minimal_contract(
            goals=[
                {
                    "cell": [999999, 999998],
                    "utility": 10.0,
                    "reachable": True,
                    "expected_coverage_rate_delta": 9.0,
                    "expected_new_coverage_area": 1.0e9,
                    "information_gain": 5.0,
                    "confidence_gain": 4.0,
                    "value": 3.0,
                    "risk": 7.5,
                    "path_cost": 1.0e12,
                    "energy_cost": 5.0e11,
                },
                {
                    "cell": [2, 1],
                    "utility": 0.5,
                    "reachable": True,
                    "path_cost": 2.0e12,
                    "energy_cost": 1.0e12,
                },
            ],
            observation_update={"coverage_rate": 5.0},
        )
        payload["grid"]["width"] = 1000000
        payload["grid"]["height"] = 1000000
        payload["constraints"]["violation_count"] = 1000000
        contract = load_contract_from_dict(payload)

        observation = extract_policy_observation(
            contract,
            current_cell=(0, 0),
            step_index=1000000,
            remaining_steps=1000000,
            max_candidates=3,
        )

        flattened = [
            *observation.global_features,
            *(value for row in observation.candidate_features for value in row),
            *(value for row in observation.candidate_missing_indicators for value in row),
        ]
        self.assertTrue(all(math.isfinite(value) for value in flattened))
        self.assertTrue(all(0.0 <= value <= 1.0 for value in observation.global_features))
        feature_names = observation.candidate_feature_names
        for row in observation.candidate_features[:2]:
            features = dict(zip(feature_names, row))
            for name in (
                "expected_coverage_rate_delta",
                "expected_new_coverage_area",
                "information_gain",
                "confidence_gain",
                "value",
                "risk",
                "path_cost",
                "energy_cost",
            ):
                self.assertGreaterEqual(features[name], 0.0)
                self.assertLessEqual(features[name], 1.0)


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
    def test_architecture_registry_builds_mlp_variants_and_rejects_unknown_names(self):
        from model_explorer.policy.architectures import SUPPORTED_ARCHITECTURES, build_policy_network
        from model_explorer.policy.features import extract_policy_observation

        contract = load_contract_from_dict(
            minimal_contract(
                goals=[
                    {"cell": [0, 0], "utility": 0.5, "reachable": True},
                    {"cell": [1, 1], "utility": 0.9, "reachable": False},
                ]
            )
        )
        observation = extract_policy_observation(contract, max_candidates=3)

        mlp = build_policy_network(
            "mlp_v1",
            observation=observation,
            hidden_size=16,
            architecture_config={"hidden_dim": 16, "dropout": 0.0},
        )
        missing = build_policy_network("mlp_missing_v1", observation=observation, hidden_size=16)
        attention = build_policy_network(
            "candidate_attention_v1",
            observation=observation,
            architecture_config={"hidden_dim": 16, "attention_heads": 2, "dropout": 0.0},
        )

        self.assertIn("mlp_v1", SUPPORTED_ARCHITECTURES)
        self.assertIn("mlp_missing_v1", SUPPORTED_ARCHITECTURES)
        self.assertIn("candidate_attention_v1", SUPPORTED_ARCHITECTURES)
        self.assertEqual(mlp.architecture_name, "mlp_v1")
        self.assertEqual(missing.architecture_name, "mlp_missing_v1")
        self.assertEqual(attention.architecture_name, "candidate_attention_v1")
        self.assertEqual(mlp.architecture_config["hidden_dim"], 16)
        self.assertEqual(mlp.architecture_config["dropout"], 0.0)
        self.assertEqual(attention.architecture_config["attention_heads"], 2)
        self.assertEqual(attention.candidate_attention.num_heads, 2)
        self.assertEqual(
            missing.candidate_encoder[0].in_features,
            len(observation.candidate_feature_names) + len(observation.candidate_missing_indicator_names),
        )
        with self.assertRaisesRegex(ValueError, "unknown architecture.*does_not_exist.*mlp_v1"):
            build_policy_network("does_not_exist", observation=observation, hidden_size=16)
        with self.assertRaisesRegex(ValueError, "unknown architecture config field.*unexpected_knob"):
            build_policy_network(
                "mlp_v1",
                observation=observation,
                architecture_config={"hidden_dim": 16, "unexpected_knob": 1},
            )
        with self.assertRaisesRegex(ValueError, "attention_heads.*divide hidden_dim"):
            build_policy_network(
                "candidate_attention_v1",
                observation=observation,
                architecture_config={"hidden_dim": 18, "attention_heads": 4},
            )

    def test_mlp_missing_network_is_mask_safe_with_missing_indicator_inputs(self):
        import torch

        from model_explorer.policy.architectures import build_policy_network
        from model_explorer.policy.features import extract_policy_observation
        from model_explorer.policy.torch_policy import observation_to_tensors

        contract = load_contract_from_dict(
            minimal_contract(
                goals=[
                    {"cell": [0, 0], "utility": 0.5, "reachable": True},
                    {"cell": [1, 1], "utility": 0.9, "reachable": False},
                    {"cell": [2, 1], "utility": 0.4, "reachable": True, "risk": 0.0},
                ]
            )
        )
        observation = extract_policy_observation(contract, max_candidates=4)
        network = build_policy_network("mlp_missing_v1", observation=observation, hidden_size=16)

        output = network(**observation_to_tensors(observation))

        self.assertEqual(tuple(output.masked_logits.shape), (1, 4))
        self.assertEqual(tuple(output.value.shape), (1,))
        self.assertTrue(torch.isfinite(output.value).all())
        action_probs = output.action_probs.detach()
        self.assertAlmostEqual(float(action_probs[0, 1]), 0.0)
        self.assertAlmostEqual(float(action_probs[0, 3]), 0.0)

    def test_candidate_attention_masks_padding_candidates_from_valid_probabilities(self):
        import torch

        from model_explorer.policy.architectures import build_policy_network
        from model_explorer.policy.features import extract_policy_observation
        from model_explorer.policy.torch_policy import observation_to_tensors

        contract = load_contract_from_dict(
            minimal_contract(
                goals=[
                    {"cell": [0, 0], "utility": 0.5, "reachable": True, "risk": 0.0},
                    {"cell": [2, 1], "utility": 0.4, "reachable": True, "risk": 0.2},
                ]
            )
        )
        observation = extract_policy_observation(contract, max_candidates=2)
        tensors = observation_to_tensors(observation)
        network = build_policy_network("candidate_attention_v1", observation=observation, hidden_size=16)
        network.eval()

        padded_features = torch.cat(
            (
                tensors["candidate_features"],
                torch.full((1, 2, tensors["candidate_features"].shape[-1]), 123.0),
            ),
            dim=1,
        )
        padded_missing = torch.cat(
            (
                tensors["candidate_missing_indicators"],
                torch.ones((1, 2, tensors["candidate_missing_indicators"].shape[-1])),
            ),
            dim=1,
        )
        padded_mask = torch.tensor([[True, True, False, False]], dtype=torch.bool)

        with torch.no_grad():
            base_output = network(**tensors)
            padded_output = network(
                candidate_features=padded_features,
                global_features=tensors["global_features"],
                action_mask=padded_mask,
                candidate_missing_indicators=padded_missing,
            )

        self.assertTrue(torch.allclose(base_output.action_probs[0], padded_output.action_probs[0, :2], atol=1.0e-6))
        self.assertAlmostEqual(float(padded_output.action_probs[0, 2]), 0.0)
        self.assertAlmostEqual(float(padded_output.action_probs[0, 3]), 0.0)

    def test_candidate_attention_valid_outputs_ignore_masked_candidate_mutation_and_reordering(self):
        import torch

        from model_explorer.policy.architectures import build_policy_network
        from model_explorer.policy.features import extract_policy_observation
        from model_explorer.policy.torch_policy import observation_to_tensors

        contract = load_contract_from_dict(
            minimal_contract(
                goals=[
                    {"cell": [0, 0], "utility": 0.5, "reachable": True, "risk": 0.0},
                    {"cell": [1, 0], "utility": 9.9, "reachable": False, "risk": 0.9},
                    {"cell": [2, 1], "utility": 0.4, "reachable": True, "risk": 0.2},
                ]
            )
        )
        observation = extract_policy_observation(contract, max_candidates=4)
        tensors = observation_to_tensors(observation)
        network = build_policy_network(
            "candidate_attention_v1",
            observation=observation,
            architecture_config={"hidden_dim": 16, "attention_heads": 2, "dropout": 0.0},
        )
        network.eval()

        mutated_tensors = dict(tensors)
        mutated_tensors["candidate_features"] = tensors["candidate_features"].clone()
        mutated_tensors["candidate_missing_indicators"] = tensors["candidate_missing_indicators"].clone()
        mutated_tensors["candidate_features"][0, 1, :] = 999.0
        mutated_tensors["candidate_features"][0, 3, :] = -999.0
        mutated_tensors["candidate_missing_indicators"][0, 1, :] = 1.0
        mutated_tensors["candidate_missing_indicators"][0, 3, :] = 1.0

        permutation = torch.tensor([2, 0, 1, 3], dtype=torch.long)
        permuted_tensors = {
            "candidate_features": tensors["candidate_features"].index_select(1, permutation),
            "global_features": tensors["global_features"],
            "action_mask": tensors["action_mask"].index_select(1, permutation),
            "candidate_missing_indicators": tensors["candidate_missing_indicators"].index_select(1, permutation),
        }

        with torch.no_grad():
            base_output = network(**tensors)
            mutated_output = network(**mutated_tensors)
            permuted_output = network(**permuted_tensors)

        valid_indices = torch.tensor([0, 2], dtype=torch.long)
        self.assertTrue(
            torch.allclose(
                base_output.logits[0].index_select(0, valid_indices),
                mutated_output.logits[0].index_select(0, valid_indices),
                atol=1.0e-6,
            )
        )
        self.assertTrue(
            torch.allclose(
                base_output.action_probs[0].index_select(0, valid_indices),
                mutated_output.action_probs[0].index_select(0, valid_indices),
                atol=1.0e-6,
            )
        )
        restored_permuted_probs = torch.empty_like(base_output.action_probs[0])
        restored_permuted_probs[permutation] = permuted_output.action_probs[0]
        self.assertTrue(torch.allclose(base_output.action_probs[0], restored_permuted_probs, atol=1.0e-6))

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

    def test_dynamic_rollout_provider_records_replan_and_next_observation(self):
        from model_explorer.policy.collector import collect_dynamic_rollout_episode
        from model_explorer.policy.provider import SequenceContractProvider

        snapshots = [
            load_contract_from_dict(
                minimal_contract(
                    goals=[{"cell": [1, 1], "utility": 0.4, "reachable": True}],
                    observation_update={"coverage_rate": 0.1, "coverage_rate_delta": 0.05},
                )
            ),
            load_contract_from_dict(
                minimal_contract(
                    goals=[{"cell": [2, 1], "utility": 0.5, "reachable": True}],
                    observation_update={"coverage_rate": 0.2, "coverage_rate_delta": 0.1},
                )
            ),
            load_contract_from_dict(
                minimal_contract(
                    goals=[{"cell": [2, 1], "utility": 0.6, "reachable": True}],
                    observation_update={"coverage_rate": 0.25, "coverage_rate_delta": 0.05},
                )
            ),
        ]
        provider = SequenceContractProvider(
            snapshots,
            replan_reasons_by_step={
                0: ("observation_delta",),
                1: ("goal_changed",),
            },
        )

        episode = collect_dynamic_rollout_episode(provider, max_steps=3, max_candidates=2)

        self.assertEqual(len(episode.transitions), 3)
        self.assertEqual(episode.transitions[0].next_observation.candidate_cells, ((2, 1), None))
        self.assertEqual(episode.transitions[0].info.extra["replan_reasons"], ["observation_delta"])
        self.assertEqual(episode.metrics.replan_count, 2)
        self.assertAlmostEqual(episode.metrics.final_coverage_rate, 0.25)

    def test_collect_rollout_episode_records_failure_transition_when_no_goal_is_reachable(self):
        from model_explorer.policy.collector import collect_rollout_episode

        scenario = [
            load_contract_from_dict(
                minimal_contract(
                    goals=[
                        {"cell": [1, 1], "utility": 0.7, "reachable": False},
                        {"cell": [2, 1], "utility": 0.6, "reachable": False},
                    ],
                    observation_update={},
                )
            )
        ]

        episode = collect_rollout_episode(scenario, max_candidates=2)

        self.assertEqual(len(episode.transitions), 1)
        self.assertEqual(episode.transitions[0].action_index, -1)
        self.assertEqual(episode.transitions[0].info.failure_reason, "no_reachable_goal")
        self.assertAlmostEqual(episode.transitions[0].reward, -1.0)
        self.assertEqual(episode.metrics.failure_count, 1)
        self.assertEqual(episode.metrics.replan_count, 1)

    def test_rollout_selection_strategy_modes_record_effective_training_source(self):
        from model_explorer.policy.collector import collect_rollout_episode
        from model_explorer.policy.planning import PathPlanResult

        class FixedPlanner:
            def plan(self, request):
                if request.action_index == 1:
                    return PathPlanResult(
                        feasible=False,
                        path_cost=0.0,
                        path_length=0.0,
                        risk=0.0,
                        failure_reason="path_blocked",
                        replan_required=True,
                    )
                return PathPlanResult(feasible=True, path_cost=1.0, path_length=1.0, risk=0.1)

        scenario = [
            load_contract_from_dict(
                minimal_contract(
                    goals=[
                        {
                            "cell": [1, 1],
                            "utility": 0.9,
                            "reachable": True,
                            "expected_coverage_rate_delta": 0.1,
                        },
                        {
                            "cell": [2, 1],
                            "utility": 0.4,
                            "reachable": True,
                            "expected_coverage_rate_delta": 0.9,
                        },
                        {
                            "cell": [3, 1],
                            "utility": 9.0,
                            "reachable": False,
                            "expected_coverage_rate_delta": 2.0,
                        },
                    ],
                    observation_update={"coverage_rate": 0.1, "coverage_rate_delta": 0.1},
                )
            )
        ]

        cases = (
            ("auto", None, 1, "coverage_heuristic"),
            ("coverage_heuristic", FixedPlanner(), 1, "coverage_heuristic"),
            ("utility", FixedPlanner(), 0, "utility"),
            ("feedback_aware", FixedPlanner(), 0, "feedback_aware"),
            ("auto", FixedPlanner(), 0, "feedback_aware"),
        )
        for requested_strategy, planner, expected_action, expected_source in cases:
            with self.subTest(selection_strategy=requested_strategy, planner=planner is not None):
                episode = collect_rollout_episode(
                    scenario,
                    max_candidates=3,
                    planning_adapter=planner,
                    selection_strategy=requested_strategy,
                )
                transition = episode.transitions[0]

                self.assertEqual(transition.action_index, expected_action)
                self.assertEqual(transition.info.extra["requested_selection_strategy"], requested_strategy)
                self.assertEqual(transition.info.extra["selection_strategy"], expected_source)
                self.assertEqual(transition.info.extra["teacher_action_index"], expected_action)
                if expected_source == "feedback_aware":
                    self.assertIn("selection_score", transition.info.extra)

    def test_fake_execution_adapter_failure_is_recorded_as_failure_and_replan(self):
        from model_explorer.policy.collector import collect_dynamic_rollout_episode
        from model_explorer.policy.execution import FakeExecutionFeasibilityAdapter
        from model_explorer.policy.provider import SequenceContractProvider

        contract = load_contract_from_dict(
            minimal_contract(
                goals=[{"cell": [1, 1], "utility": 0.4, "reachable": True}],
                observation_update={"coverage_rate": 0.1, "coverage_rate_delta": 0.05},
            )
        )
        provider = SequenceContractProvider([contract])
        execution_adapter = FakeExecutionFeasibilityAdapter(
            failing_cells={(1, 1): "local_trajectory_infeasible"}
        )

        episode = collect_dynamic_rollout_episode(
            provider,
            execution_adapter=execution_adapter,
            max_steps=1,
        )

        self.assertEqual(len(episode.transitions), 1)
        self.assertEqual(episode.transitions[0].action_index, 0)
        self.assertEqual(episode.transitions[0].info.failure_reason, "local_trajectory_infeasible")
        self.assertFalse(episode.transitions[0].info.extra["execution_feasible"])
        self.assertEqual(episode.metrics.failure_count, 1)
        self.assertEqual(episode.metrics.replan_count, 1)

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


class PathPlanningAdapterTests(unittest.TestCase):
    def test_feedback_aware_selection_prefers_feasible_alternative_over_blocked_high_coverage_goal(self):
        from model_explorer.policy.feedback_selection import select_goal_with_path_feedback
        from model_explorer.policy.planning import PathPlanResult

        class FixedPlanner:
            def __init__(self):
                self.action_indices = []

            def plan(self, request):
                self.action_indices.append(request.action_index)
                if request.action_index == 0:
                    return PathPlanResult(
                        feasible=False,
                        path_cost=0.0,
                        path_length=0.0,
                        risk=0.0,
                        failure_reason="goal_blocked",
                        replan_required=True,
                    )
                return PathPlanResult(feasible=True, path_cost=5.0, path_length=5.0, risk=0.2)

        contract = load_contract_from_dict(
            minimal_contract(
                goals=[
                    {
                        "cell": [1, 1],
                        "utility": 0.9,
                        "reachable": True,
                        "expected_coverage_rate_delta": 0.9,
                        "value": 0.8,
                    },
                    {
                        "cell": [2, 1],
                        "utility": 0.2,
                        "reachable": True,
                        "expected_coverage_rate_delta": 0.2,
                        "value": 0.1,
                    },
                ]
            )
        )
        planner = FixedPlanner()

        selection = select_goal_with_path_feedback(
            contract,
            planner=planner,
            current_cell=(0, 0),
            top_k=2,
        )

        self.assertEqual(selection.decision.selected_goal.cell, (2, 1))
        self.assertEqual(selection.decision.status, "selected")
        self.assertEqual(planner.action_indices, [0, 1])
        self.assertLess(selection.scores_by_action_index[0], selection.scores_by_action_index[1])

    def test_feedback_aware_selection_records_top_two_teacher_scores_and_margin(self):
        from model_explorer.policy.feedback_selection import select_goal_with_path_feedback
        from model_explorer.policy.planning import PathPlanResult

        class FixedPlanner:
            def __init__(self):
                self.action_indices = []

            def plan(self, request):
                self.action_indices.append(request.action_index)
                return PathPlanResult(feasible=True, path_cost=1.0, path_length=1.0, risk=0.1)

        contract = load_contract_from_dict(
            minimal_contract(
                goals=[
                    {
                        "cell": [1, 1],
                        "utility": 0.8,
                        "reachable": True,
                        "expected_coverage_rate_delta": 0.9,
                    },
                    {
                        "cell": [2, 1],
                        "utility": 0.7,
                        "reachable": True,
                        "expected_coverage_rate_delta": 0.4,
                    },
                    {
                        "cell": [3, 1],
                        "utility": 99.0,
                        "reachable": False,
                        "expected_coverage_rate_delta": 1.0,
                    },
                ]
            )
        )

        selection = select_goal_with_path_feedback(
            contract,
            planner=FixedPlanner(),
            current_cell=(0, 0),
            top_k=3,
        )

        self.assertEqual(selection.selected_action_index, 0)
        self.assertEqual(selection.runner_up_action_index, 1)
        self.assertEqual(selection.ranked_action_indices, (0, 1))
        self.assertNotIn(2, selection.ranked_action_indices)
        self.assertAlmostEqual(selection.selected_score, selection.scores_by_action_index[0])
        self.assertAlmostEqual(selection.runner_up_score, selection.scores_by_action_index[1])
        self.assertAlmostEqual(
            selection.score_margin,
            selection.scores_by_action_index[0] - selection.scores_by_action_index[1],
        )
        self.assertGreater(selection.score_margin, 0.0)

    def test_feedback_aware_selection_penalizes_path_cost_risk_and_replan(self):
        from model_explorer.policy.feedback_selection import select_goal_with_path_feedback
        from model_explorer.policy.planning import PathPlanResult

        class FixedPlanner:
            def plan(self, request):
                results = {
                    0: PathPlanResult(
                        feasible=True,
                        path_cost=1.0,
                        path_length=1.0,
                        risk=0.05,
                        replan_required=True,
                    ),
                    1: PathPlanResult(feasible=True, path_cost=4.0, path_length=4.0, risk=0.2),
                    2: PathPlanResult(feasible=True, path_cost=80.0, path_length=80.0, risk=0.9),
                }
                return results[request.action_index]

        contract = load_contract_from_dict(
            minimal_contract(
                goals=[
                    {
                        "cell": [1, 1],
                        "utility": 0.9,
                        "reachable": True,
                        "expected_coverage_rate_delta": 0.6,
                    },
                    {
                        "cell": [2, 1],
                        "utility": 0.4,
                        "reachable": True,
                        "expected_coverage_rate_delta": 0.55,
                    },
                    {
                        "cell": [3, 1],
                        "utility": 0.8,
                        "reachable": True,
                        "expected_coverage_rate_delta": 0.55,
                    },
                ]
            )
        )

        selection = select_goal_with_path_feedback(
            contract,
            planner=FixedPlanner(),
            current_cell=(0, 0),
            top_k=3,
        )

        self.assertEqual(selection.decision.selected_goal.cell, (2, 1))
        self.assertLess(selection.scores_by_action_index[0], selection.scores_by_action_index[1])
        self.assertLess(selection.scores_by_action_index[2], selection.scores_by_action_index[1])

    def test_feedback_aware_selection_never_evaluates_or_selects_contract_unreachable_candidates(self):
        from model_explorer.policy.feedback_selection import select_goal_with_path_feedback
        from model_explorer.policy.planning import PathPlanResult

        class RecordingPlanner:
            def __init__(self):
                self.action_indices = []

            def plan(self, request):
                self.action_indices.append(request.action_index)
                return PathPlanResult(feasible=True, path_cost=1.0, path_length=1.0, risk=0.0)

        contract = load_contract_from_dict(
            minimal_contract(
                goals=[
                    {
                        "cell": [1, 1],
                        "utility": 99.0,
                        "reachable": False,
                        "expected_coverage_rate_delta": 1.0,
                    },
                    {
                        "cell": [2, 1],
                        "utility": 0.1,
                        "reachable": True,
                        "expected_coverage_rate_delta": 0.1,
                    },
                ]
            )
        )
        planner = RecordingPlanner()

        selection = select_goal_with_path_feedback(
            contract,
            planner=planner,
            current_cell=(0, 0),
            top_k=2,
        )

        self.assertEqual(selection.decision.selected_goal.cell, (2, 1))
        self.assertEqual(planner.action_indices, [1])
        self.assertNotIn(0, selection.scores_by_action_index)

    def test_path_feedback_summary_contract_lists_required_acceptance_metrics(self):
        from model_explorer.policy.path_feedback import (
            PATH_FEEDBACK_SUMMARY_ACCEPTANCE_METRICS,
            PATH_FEEDBACK_SUMMARY_REQUIRED_KEYS,
            validate_path_feedback_summary_contract,
        )

        acceptance_metrics = {
            "selection_changed_rate",
            "path_planning_failure_count",
            "replan_count",
            "tracking_safety_violation_count",
            "trajectory_optimization_fallback_count",
            "region_graph_disconnected_count",
            "coverage_per_path_cost",
        }

        self.assertTrue(acceptance_metrics.issubset(set(PATH_FEEDBACK_SUMMARY_ACCEPTANCE_METRICS)))
        self.assertTrue(set(PATH_FEEDBACK_SUMMARY_ACCEPTANCE_METRICS).issubset(set(PATH_FEEDBACK_SUMMARY_REQUIRED_KEYS)))

        summary = {key: 0 for key in PATH_FEEDBACK_SUMMARY_REQUIRED_KEYS}
        summary.update(
            {
                "schema_version": "path-feedback-summary/v1",
                "scenario_count": 1,
                "top_k": 3,
                "candidate_count": 3,
                "open_grid_fallback_used": False,
                "failure_reasons": [],
                "iris_status_counts": {},
                "region_graph_source_counts": {},
                "scenario_group_summary": {},
                "scenarios": [],
            }
        )

        validation = validate_path_feedback_summary_contract(summary)

        self.assertEqual(validation["status"], "valid")
        self.assertEqual(validation["schema_version"], "path-feedback-summary/v1")

        missing_metric = dict(summary)
        del missing_metric["coverage_per_path_cost"]
        with self.assertRaisesRegex(ValueError, "coverage_per_path_cost"):
            validate_path_feedback_summary_contract(missing_metric)

        fallback_summary = dict(summary)
        fallback_summary["open_grid_fallback_used"] = True
        with self.assertRaisesRegex(ValueError, "open_grid_fallback_used"):
            validate_path_feedback_summary_contract(fallback_summary)

    def test_contract_cost_planner_uses_finite_defaults_when_fields_are_missing(self):
        from model_explorer.policy.planning import ContractCostPlanner, PathPlanRequest

        contract = load_contract_from_dict(
            minimal_contract(goals=[{"cell": [2, 1], "utility": 0.42, "reachable": True}])
        )
        selected_goal = contract.top_goals[0]

        result = ContractCostPlanner().plan(
            PathPlanRequest(
                contract=contract,
                step_index=0,
                action_index=0,
                selected_goal=selected_goal,
                current_cell=(0, 0),
            )
        )

        self.assertTrue(result.feasible)
        self.assertEqual(result.path_cost, 0.0)
        self.assertEqual(result.risk, 0.0)
        self.assertIsNone(result.failure_reason)

    def test_straight_line_proxy_planner_computes_deterministic_grid_distance(self):
        from model_explorer.policy.planning import PathPlanRequest, StraightLineProxyPlanner

        contract = load_contract_from_dict(
            minimal_contract(goals=[{"cell": [3, 4], "utility": 0.4, "reachable": True, "risk": 0.25}])
        )
        selected_goal = contract.top_goals[0]

        result = StraightLineProxyPlanner().plan(
            PathPlanRequest(
                contract=contract,
                step_index=0,
                action_index=0,
                selected_goal=selected_goal,
                current_cell=(0, 0),
            )
        )

        self.assertTrue(result.feasible)
        self.assertAlmostEqual(result.path_length, 5.0)
        self.assertAlmostEqual(result.path_cost, 5.0)
        self.assertAlmostEqual(result.risk, 0.25)

    def test_grid_astar_planner_avoids_obstacles_and_reports_blocked_paths(self):
        from model_explorer.policy.planning import GridAStarPlanner, PathPlanRequest

        contract = load_contract_from_dict(
            minimal_contract(goals=[{"cell": [2, 2], "utility": 0.4, "reachable": True}])
        )
        selected_goal = contract.top_goals[0]
        planner = GridAStarPlanner(
            passable_grid=[
                [True, True, True],
                [False, False, True],
                [True, True, True],
            ]
        )

        result = planner.plan(
            PathPlanRequest(
                contract=contract,
                step_index=0,
                action_index=0,
                selected_goal=selected_goal,
                current_cell=(0, 0),
            )
        )

        self.assertTrue(result.feasible)
        self.assertEqual(result.path_length, 4.0)
        self.assertEqual(result.metadata["path_cells"], [[0, 0], [1, 0], [2, 0], [2, 1], [2, 2]])

        blocked_result = GridAStarPlanner(
            passable_grid=[
                [True, False, True],
                [False, False, True],
                [True, True, True],
            ]
        ).plan(
            PathPlanRequest(
                contract=contract,
                step_index=0,
                action_index=0,
                selected_goal=selected_goal,
                current_cell=(0, 0),
            )
        )

        self.assertFalse(blocked_result.feasible)
        self.assertEqual(blocked_result.failure_reason, "path_blocked")

    def test_path_planner_route_adapter_is_explicitly_unavailable_without_direct_imports(self):
        from model_explorer.policy.planning import (
            PathCandidateEvaluation,
            PathPlanRequest,
            build_path_planner_request_dict,
            evaluate_candidate_paths,
            path_plan_result_from_route_dict,
            path_feedback_summary,
            planner_from_config,
        )

        contract = load_contract_from_dict(
            minimal_contract(goals=[{"cell": [1, 1], "utility": 0.4, "reachable": True, "risk": 0.2}])
        )
        selected_goal = contract.top_goals[0]
        request = PathPlanRequest(
            contract=contract,
            step_index=0,
            action_index=0,
            selected_goal=selected_goal,
            current_cell=(0, 0),
        )

        planner_request = build_path_planner_request_dict(request)

        self.assertEqual(planner_request["schema_version"], "path-planner-request/v1")
        self.assertEqual(planner_request["start"], [0, 0])
        self.assertEqual(planner_request["goal"], [1, 1])
        self.assertEqual(planner_request["metadata"]["cost_source"], "open_grid_fallback")

        route_payload = {
            "schema_version": "path-planner-route/v1",
            "reachable": True,
            "path_cost": 2.5,
            "failure_reason": None,
            "geometric_path": {
                "cells": [[0, 0], [1, 1]],
                "world": [[1.0, 2.0], [1.5, 2.5]],
            },
            "diagnostics": {"path_length_m": 0.75, "search_mode": "platform_aware_astar"},
            "planning_backend_report": {
                "requested_backend": "region_graph_guided",
                "selected_backend": "sampled_region_path",
                "status": "selected",
                "fallback_reason": None,
                "segment_count": 2,
                "comparison": {"path_changed": True},
                "region_graph_candidate": {"status": "selected"},
                "sampled_region_path_report": {
                    "schema_version": "sampled_region_path_report/v1",
                    "status": "selected",
                    "fallback_reason": None,
                    "region_sequence": [0, 1],
                    "sample_count": 2,
                    "safety_checks": {"collision_free": True},
                    "candidate_comparison": {"candidate_cost_delta": -1.0},
                },
            },
            "postprocess": {"fallback_status": "ok"},
        }
        route_payload.update(
            _gcs_candidate_route_fields(
                available=True,
                selected=True,
                selection_reason="gcs_candidate_quality_improved",
                fallback_reason=None,
                collision_count=0,
                cost_delta=-0.75,
                overlap_ratio=0.25,
            )
        )
        route_payload.update(
            _gcs_motion_feasibility_route_fields(
                evaluated=True,
                feasibility_status="feasible",
                fallback_reason=None,
            )
        )
        route_payload.update(
            _gcs_curvature_constrained_candidate_route_fields(
                available=True,
                selected=True,
                repair_success=False,
                repair_strategy="none_required",
                status_before="feasible",
                status_after="feasible",
                fallback_reason=None,
            )
        )
        mapped = path_plan_result_from_route_dict(route_payload, request=request)

        self.assertTrue(mapped.feasible)
        self.assertEqual(mapped.path_cost, 2.5)
        self.assertEqual(mapped.path_length, 0.75)
        self.assertEqual(mapped.risk, 0.2)
        self.assertEqual(mapped.metadata["diagnostics"]["search_mode"], "platform_aware_astar")
        self.assertEqual(
            mapped.metadata["planning_backend_report"]["selected_backend"],
            "sampled_region_path",
        )
        self.assertEqual(mapped.metadata["gcs_candidate_report"]["selected"], True)
        self.assertEqual(mapped.metadata["gcs_candidate_report"]["selection_reason"], "gcs_candidate_quality_improved")
        self.assertEqual(mapped.metadata["gcs_motion_feasibility_report"]["feasibility_status"], "feasible")
        self.assertEqual(mapped.metadata["gcs_curvature_constrained_candidate_report"]["selected"], True)
        self.assertEqual(
            mapped.metadata["gcs_curvature_constrained_candidate_report"]["repair_strategy"],
            "none_required",
        )
        summary = path_feedback_summary(
            [
                PathCandidateEvaluation(
                    action_index=0,
                    cell=selected_goal.cell,
                    utility=selected_goal.utility,
                    result=mapped,
                )
            ]
        )
        self.assertEqual(
            summary["candidates"][0]["planning_backend"]["sampled_region_path"]["status"],
            "selected",
        )
        self.assertEqual(
            summary["candidates"][0]["diagnostic_interpretation"]["sampled_region_path_status"],
            "selected",
        )
        self.assertEqual(summary["candidates"][0]["gcs_candidate"]["selected"], True)
        self.assertEqual(
            summary["candidates"][0]["gcs_candidate"]["cost_delta_vs_baseline"],
            -0.75,
        )
        self.assertEqual(summary["candidates"][0]["gcs_motion_feasibility"]["feasibility_status"], "feasible")
        self.assertEqual(summary["candidates"][0]["gcs_curvature_constrained_candidate"]["selected"], True)

        disconnected = dict(route_payload)
        disconnected["region_graph_report"] = {
            "status": "ok",
            "quality_metrics": {"start_goal_connected": False},
        }
        disconnected_result = path_plan_result_from_route_dict(disconnected, request=request)
        self.assertTrue(disconnected_result.replan_required)
        self.assertFalse(disconnected_result.metadata["region_graph_report"]["quality_metrics"]["start_goal_connected"])

        with tempfile.TemporaryDirectory() as tmpdir:
            route_path = Path(tmpdir) / "route.json"
            route_path.write_text(json.dumps(route_payload), encoding="utf-8")
            sidecar_path = Path(tmpdir) / "sidecar.json"
            sidecar_path.write_text(
                json.dumps(
                    {
                        "schema_version": "path-planner-sidecar/v1",
                        "grid": planner_request["grid"],
                        "cost": [[1.0, 2.0, 1.0, 1.0] for _ in range(3)],
                        "passable_mask": [[True, True, True, True] for _ in range(3)],
                        "metadata": {"scenario_id": "unit-sidecar"},
                    }
                ),
                encoding="utf-8",
            )
            result = planner_from_config(
                {
                    "backend": "path_planner_route",
                    "route_json": str(route_path),
                    "path_planner_sidecar": str(sidecar_path),
                }
            ).plan(request)

        self.assertTrue(result.feasible)
        self.assertEqual(result.path_cost, 2.5)
        self.assertEqual(result.metadata["planner"], "path_planner_route")
        self.assertEqual(result.metadata["mode"], "route_json")
        self.assertEqual(result.metadata["request_payload"]["schema_version"], "path-planner-request/v1")
        self.assertEqual(result.metadata["request_payload"]["metadata"]["cost_source"], "configured")
        self.assertEqual(result.metadata["request_payload"]["metadata"]["sidecar"]["scenario_id"], "unit-sidecar")
        self.assertNotIn("a_gcs_ws", sys.modules)

        batch_contract = load_contract_from_dict(
            minimal_contract(
                goals=[
                    {"cell": [1, 1], "utility": 0.4, "reachable": True},
                    {"cell": [2, 1], "utility": 0.3, "reachable": True},
                    {"cell": [3, 1], "utility": 9.0, "reachable": False},
                ]
            )
        )

        class FixedPlanner:
            def plan(self, plan_request):
                route = (
                    {
                        "schema_version": "path-planner-route/v1",
                        "reachable": False,
                        "path_cost": None,
                        "failure_reason": "goal_blocked",
                        "geometric_path": {"cells": [], "world": []},
                        "diagnostics": {"search_mode": "platform_aware_astar"},
                    }
                    if plan_request.action_index == 0
                    else {
                        "schema_version": "path-planner-route/v1",
                        "reachable": True,
                        "path_cost": 1.5,
                        "failure_reason": None,
                        "geometric_path": {"cells": [[0, 0], [2, 1]], "world": [[0.0, 0.0], [2.0, 1.0]]},
                        "diagnostics": {"path_length_m": 2.2, "search_mode": "platform_aware_astar"},
                        "iris_region_report": {
                            "backend": "workspace_iris",
                            "status": "ok",
                            "region_count": 2,
                            "fallback_used": False,
                            "failure_status": "none",
                            "failure_reason": None,
                        },
                        "region_graph_report": {
                            "status": "ok",
                            "region_source": "iris",
                            "fallback_used": False,
                            "quality_metrics": {
                                "requested_region_source": "iris",
                                "graph_source": "iris",
                                "fallback_ratio": 0.0,
                                "connected_component_count": 1,
                                "start_goal_connected": True,
                                "fallback_reason": None,
                            },
                        },
                    }
                )
                return path_plan_result_from_route_dict(route, request=plan_request)

        evaluations = evaluate_candidate_paths(batch_contract, current_cell=(0, 0), top_k=2, planner=FixedPlanner())
        summary = path_feedback_summary(evaluations)

        self.assertEqual(summary["candidate_count"], 2)
        self.assertEqual(summary["reachable_count"], 1)
        self.assertEqual(summary["failure_reasons"], ["goal_blocked"])
        self.assertEqual(summary["candidates"][0]["diagnostics"]["search_mode"], "platform_aware_astar")
        self.assertEqual(summary["best_by_path_cost"]["cell"], [2, 1])
        blocked_candidate = summary["candidates"][0]
        self.assertEqual(
            blocked_candidate["diagnostic_interpretation"]["primary_source"],
            "path_planning_failure",
        )
        self.assertIn(
            "path_planning_failure",
            blocked_candidate["diagnostic_interpretation"]["diagnostic_flags"],
        )
        self.assertFalse(blocked_candidate["diagnostic_interpretation"]["open_grid_fallback_used"])
        iris_candidate = summary["candidates"][1]
        self.assertEqual(iris_candidate["iris_region"]["status"], "ok")
        self.assertEqual(iris_candidate["iris_region"]["backend"], "workspace_iris")
        self.assertEqual(iris_candidate["iris_region"]["region_count"], 2)
        self.assertFalse(iris_candidate["iris_region"]["fallback_used"])
        self.assertEqual(iris_candidate["region_graph"]["graph_source"], "iris")
        self.assertEqual(iris_candidate["region_graph"]["requested_region_source"], "iris")
        self.assertEqual(iris_candidate["region_graph"]["fallback_ratio"], 0.0)
        self.assertEqual(iris_candidate["region_graph"]["connected_component_count"], 1)
        self.assertEqual(iris_candidate["region_graph"]["quality_metrics"]["graph_source"], "iris")
        self.assertEqual(
            iris_candidate["region_graph"]["quality_metrics"]["requested_region_source"],
            "iris",
        )

    def test_path_planner_sidecar_validation_and_route_replan_signals(self):
        from model_explorer.policy.planning import (
            PathPlanRequest,
            load_path_planner_sidecar,
            path_plan_result_from_route_dict,
            planner_from_config,
        )

        contract = load_contract_from_dict(
            minimal_contract(goals=[{"cell": [1, 1], "utility": 0.4, "reachable": True}])
        )
        request = PathPlanRequest(
            contract=contract,
            step_index=0,
            action_index=0,
            selected_goal=contract.top_goals[0],
            current_cell=(0, 0),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            bad_schema = root / "bad-schema.json"
            bad_schema.write_text(json.dumps({"schema_version": "wrong", "cost": [], "passable_mask": []}), encoding="utf-8")
            missing_mask = root / "missing-mask.json"
            missing_mask.write_text(json.dumps({"schema_version": "path-planner-sidecar/v1", "cost": []}), encoding="utf-8")
            mismatched = root / "mismatched.json"
            mismatched.write_text(
                json.dumps(
                    {
                        "schema_version": "path-planner-sidecar/v1",
                        "cost": [[1.0]],
                        "passable_mask": [[True]],
                    }
                ),
                encoding="utf-8",
            )
            route_path = root / "route.json"
            route_path.write_text(
                json.dumps(
                    {
                        "schema_version": "path-planner-route/v1",
                        "reachable": True,
                        "path_cost": 1.0,
                        "failure_reason": None,
                        "geometric_path": {"cells": [[0, 0], [1, 1]], "world": [[0.0, 0.0], [1.0, 1.0]]},
                        "diagnostics": {"path_length_m": 1.4},
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "schema_version"):
                load_path_planner_sidecar(bad_schema)
            with self.assertRaisesRegex(ValueError, "cost and passable_mask"):
                load_path_planner_sidecar(missing_mask)
            with self.assertRaisesRegex(ValueError, "height=3, width=4"):
                planner_from_config(
                    {
                        "backend": "path_planner_route",
                        "path_planner_sidecar": str(mismatched),
                        "route_json": str(route_path),
                    }
                ).plan(request)

        base_route = {
            "schema_version": "path-planner-route/v1",
            "reachable": True,
            "path_cost": 1.0,
            "failure_reason": None,
            "geometric_path": {"cells": [[0, 0], [1, 1]], "world": [[0.0, 0.0], [1.0, 1.0]]},
            "diagnostics": {"path_length_m": 1.4},
        }
        postprocess_fallback = dict(base_route, postprocess={"fallback_status": "smoothed_path_failed"})
        tracking_violation = dict(
            base_route,
            postprocess={"fallback_status": "ok", "tracking_safety_report": {"violation_count": 2}},
        )
        optimization_fallback = dict(base_route, trajectory_optimization_report={"fallback_status": "solver_failed"})

        self.assertTrue(path_plan_result_from_route_dict(postprocess_fallback, request=request).replan_required)
        self.assertTrue(path_plan_result_from_route_dict(tracking_violation, request=request).replan_required)
        self.assertTrue(path_plan_result_from_route_dict(optimization_fallback, request=request).replan_required)

    def test_path_feedback_manifest_summarizes_three_generated_npz_scenarios(self):
        from model_explorer.policy.path_feedback import run_path_feedback_manifest

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            scenario_config = root / "npz_validation_scenarios.json"
            maps_dir = root / "maps"
            exports_dir = root / "exports"
            generator = subprocess.run(
                [
                    sys.executable,
                    str(DEV_PLATFORM_ROOT / "scripts" / "generate_npz_validation_maps.py"),
                    "--output-dir",
                    str(maps_dir),
                    "--scenario-config",
                    str(scenario_config),
                ],
                cwd=DEV_PLATFORM_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(generator.returncode, 0, generator.stdout + generator.stderr)
            exporter = subprocess.run(
                [
                    sys.executable,
                    str(DEV_PLATFORM_ROOT / "scripts" / "export_path_planner_sidecars.py"),
                    "--scenario-config",
                    str(scenario_config),
                    "--output-dir",
                    str(exports_dir),
                ],
                cwd=DEV_PLATFORM_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(exporter.returncode, 0, exporter.stdout + exporter.stderr)

            manifest_scenarios = []
            for index, scenario_id in enumerate(
                ("npz_shadow_corridor", "npz_rock_field_multi_pose", "npz_low_confidence_risk_band")
            ):
                route_0 = root / f"{scenario_id}-route-0.json"
                route_1 = root / f"{scenario_id}-route-1.json"
                route_0.write_text(json.dumps(_route_fixture(index, action_index=0)), encoding="utf-8")
                route_1.write_text(json.dumps(_route_fixture(index, action_index=1)), encoding="utf-8")
                manifest_scenarios.append(
                    {
                        "scenario_id": scenario_id,
                        "scenario_group": "smoke",
                        "contract": str(exports_dir / f"{scenario_id}.contract.json"),
                        "sidecar": str(exports_dir / f"{scenario_id}.path-planner-sidecar.json"),
                        "route_fixtures": {"0": str(route_0), "1": str(route_1)},
                    }
                )
            manifest_path = root / "path-feedback.json"
            summary_path = root / "summary.json"
            report_path = root / "summary.md"
            manifest_path.write_text(
                json.dumps(
                    {
                        "schema_version": "path-feedback-manifest/v1",
                        "top_k": 2,
                        "scenario_set": "all",
                        "diagnostic_profile": "all",
                        "acceptance_gate": "semi-real-closed-loop",
                        "planner": {"backend": "path_planner_route", "python_executable": sys.executable},
                        "scenarios": manifest_scenarios,
                        "outputs": {
                            "summary": str(summary_path),
                            "report": str(report_path),
                            "gcs_control_point_candidate_artifacts": str(
                                root / "gcs-control-point-candidate-artifacts"
                            ),
                        },
                    }
                ),
                encoding="utf-8",
            )

            summary = run_path_feedback_manifest(manifest_path)
            self.assertTrue(summary_path.exists())
            self.assertTrue(report_path.exists())
            report = report_path.read_text(encoding="utf-8")
            artifact_entry = (
                summary.get("gcs_control_point_candidate_artifacts", {}).get("entries", [{}])[0]
            )
            route_artifact_path = Path(str(artifact_entry.get("route_artifact", "")))
            route_artifact_exists = route_artifact_path.exists()
            persisted_route = (
                json.loads(route_artifact_path.read_text(encoding="utf-8"))
                if route_artifact_exists
                else {}
            )

        self.assertEqual(summary["schema_version"], "path-feedback-summary/v1")
        self.assertEqual(summary["scenario_count"], 3)
        self.assertEqual(summary["scenario_set"], "all")
        self.assertEqual(summary["diagnostic_profile"], "all")
        self.assertEqual(summary["acceptance_gate"], "semi-real-closed-loop")
        self.assertEqual(summary["planner_extra_args"], [])
        self.assertEqual(summary["acceptance_metadata"]["scenario_set"], "all")
        self.assertEqual(summary["acceptance_metadata"]["diagnostic_profile"], "all")
        self.assertEqual(summary["acceptance_metadata"]["top_k"], 2)
        self.assertEqual(summary["acceptance_metadata"]["python_executable"], sys.executable)
        self.assertEqual(summary["acceptance_metadata"]["planner_extra_args"], [])
        self.assertFalse(summary["acceptance_metadata"]["open_grid_fallback_used"])
        self.assertEqual(
            summary["acceptance_metadata"]["open_grid_fallback_used_gate"]["status"],
            "passed",
        )
        self.assertGreaterEqual(summary["candidate_count"], 6)
        self.assertFalse(summary["open_grid_fallback_used"])
        self.assertIn("total_path_cost", summary)
        self.assertIn("path_planning_failure_count", summary)
        self.assertIn("coverage_per_path_cost", summary)
        self.assertIn("selection_changed_count", summary)
        self.assertIn("selection_changed_rate", summary)
        self.assertIn("iris_requested_count", summary)
        self.assertIn("iris_status_counts", summary)
        self.assertIn("region_graph_source_counts", summary)
        self.assertIn("sampled_region_path_selected_count", summary)
        self.assertIn("sampled_region_path_fallback_count", summary)
        self.assertIn("sampled_region_path_source_counts", summary)
        self.assertIn("scenario_group_summary", summary)
        self.assertIn("diagnostic_interpretation", summary)
        self.assertIn("scenario_group_interpretation", summary["diagnostic_interpretation"])
        self.assertGreaterEqual(summary["iris_requested_count"], 1)
        self.assertEqual(summary["iris_status_counts"]["ok"], 1)
        self.assertEqual(summary["region_graph_source_counts"]["iris"], 1)
        self.assertEqual(summary["convex_region_report_count"], 5)
        self.assertEqual(summary["convex_region_count_total"], 10)
        self.assertEqual(summary["convex_region_backend_counts"]["fallback_box"], 4)
        self.assertEqual(summary["convex_region_backend_counts"]["workspace_iris"], 1)
        self.assertEqual(summary["convex_region_fallback_used_count"], 4)
        self.assertEqual(summary["convex_region_gcs_ready_count"], 5)
        self.assertEqual(summary["convex_region_blocked_cell_violation_count"], 0)
        self.assertEqual(summary["convex_region_coverage_status_counts"]["covered"], 5)
        self.assertEqual(summary["convex_region_gcs_ready_reason_counts"]["convex_region_sequence_ready"], 5)
        self.assertEqual(len(summary["convex_region_candidate_audit"]), 5)
        self.assertEqual(summary["gcs_trajectory_report_count"], 5)
        self.assertEqual(summary["gcs_trajectory_attempted_count"], 5)
        self.assertEqual(summary["gcs_trajectory_success_count"], 4)
        self.assertEqual(summary["gcs_trajectory_collision_count"], 1)
        self.assertEqual(summary["gcs_trajectory_region_count_total"], 10)
        self.assertEqual(summary["gcs_trajectory_sample_count_total"], 25)
        self.assertEqual(summary["gcs_trajectory_backend_counts"]["pydrake_gcs"], 4)
        self.assertEqual(
            summary["gcs_trajectory_backend_counts"]["pydrake_control_point_direction_cone_program"],
            1,
        )
        self.assertEqual(summary["gcs_trajectory_reason_counts"]["gcs_trajectory_solution_found"], 3)
        self.assertEqual(
            summary["gcs_trajectory_reason_counts"]["control_point_direction_cone_solution_found"],
            1,
        )
        self.assertEqual(summary["gcs_trajectory_reason_counts"]["sampled_trajectory_collision"], 1)
        self.assertEqual(len(summary["gcs_trajectory_candidate_audit"]), 5)
        self.assertEqual(summary["gcs_candidate_report_count"], 5)
        self.assertEqual(summary["gcs_candidate_attempted_count"], 5)
        self.assertEqual(summary["gcs_candidate_available_count"], 4)
        self.assertEqual(summary["gcs_candidate_selected_count"], 2)
        self.assertEqual(summary["gcs_candidate_collision_count"], 1)
        self.assertEqual(summary["gcs_candidate_fallback_reason_counts"]["cost_dominated"], 1)
        self.assertEqual(summary["gcs_candidate_fallback_reason_counts"]["path_duplicate_with_baseline"], 1)
        self.assertEqual(summary["gcs_candidate_fallback_reason_counts"]["sampled_trajectory_collision"], 1)
        self.assertEqual(summary["gcs_candidate_selection_reason_counts"]["gcs_candidate_quality_improved"], 2)
        self.assertEqual(summary["gcs_candidate_cost_delta_vs_baseline_negative_count"], 2)
        self.assertEqual(summary["gcs_candidate_cost_delta_vs_baseline_positive_count"], 1)
        self.assertEqual(summary["gcs_candidate_cost_delta_vs_baseline_zero_count"], 1)
        self.assertEqual(len(summary["gcs_candidate_audit"]), 5)
        self.assertEqual(summary["gcs_motion_feasibility_report_count"], 5)
        self.assertEqual(summary["gcs_motion_feasibility_evaluated_count"], 4)
        self.assertEqual(summary["gcs_motion_feasibility_feasible_count"], 3)
        self.assertEqual(summary["gcs_motion_feasibility_infeasible_count"], 1)
        self.assertEqual(summary["gcs_motion_feasibility_diagnostic_only_count"], 1)
        self.assertEqual(summary["gcs_motion_feasibility_curvature_violation_count"], 1)
        self.assertEqual(summary["gcs_motion_feasibility_heading_violation_count"], 1)
        self.assertEqual(summary["gcs_motion_feasibility_status_counts"]["feasible"], 3)
        self.assertEqual(summary["gcs_motion_feasibility_status_counts"]["infeasible"], 1)
        self.assertEqual(summary["gcs_motion_feasibility_status_counts"]["diagnostic_only"], 1)
        self.assertEqual(summary["gcs_motion_feasibility_fallback_reason_counts"]["motion_constraint_violation"], 1)
        self.assertEqual(summary["gcs_motion_feasibility_fallback_reason_counts"]["gcs_trajectory_failed"], 1)
        self.assertEqual(summary["gcs_motion_feasibility_motion_model_counts"]["curvature_bounded"], 5)
        self.assertEqual(len(summary["gcs_motion_feasibility_audit"]), 5)
        self.assertEqual(summary["gcs_curvature_constrained_report_count"], 5)
        self.assertEqual(summary["gcs_curvature_constrained_attempted_count"], 5)
        self.assertEqual(summary["gcs_curvature_constrained_available_count"], 4)
        self.assertEqual(summary["gcs_curvature_constrained_selected_count"], 4)
        self.assertEqual(summary["gcs_curvature_constrained_repair_success_count"], 1)
        self.assertEqual(summary["gcs_curvature_constrained_infeasible_count"], 0)
        self.assertEqual(summary["gcs_curvature_constrained_curvature_violation_count_before"], 1)
        self.assertEqual(summary["gcs_curvature_constrained_curvature_violation_count_after"], 0)
        self.assertEqual(summary["gcs_curvature_constrained_heading_violation_count_before"], 1)
        self.assertEqual(summary["gcs_curvature_constrained_heading_violation_count_after"], 0)
        self.assertEqual(summary["gcs_curvature_constrained_status_after_counts"]["feasible"], 4)
        self.assertEqual(summary["gcs_curvature_constrained_status_after_counts"]["diagnostic_only"], 1)
        self.assertEqual(summary["gcs_curvature_constrained_repair_strategy_counts"]["none_required"], 3)
        self.assertEqual(summary["gcs_curvature_constrained_repair_strategy_counts"]["moving_average_smoothing"], 1)
        self.assertEqual(summary["gcs_curvature_constrained_repair_strategy_counts"]["not_attempted"], 1)
        self.assertEqual(summary["gcs_curvature_constrained_fallback_reason_counts"]["gcs_trajectory_failed"], 1)
        self.assertEqual(len(summary["gcs_curvature_constrained_audit"]), 5)
        self.assertEqual(summary["gcs_control_point_report_count"], 1)
        self.assertEqual(summary["gcs_control_point_attempted_count"], 1)
        self.assertEqual(summary["gcs_control_point_success_count"], 1)
        self.assertEqual(
            summary["gcs_control_point_backend_counts"]["pydrake_control_point_direction_cone_program"],
            1,
        )
        self.assertEqual(summary["gcs_control_point_candidate_selected_count"], 1)
        self.assertEqual(summary["gcs_control_point_candidate_fallback_reason_counts"], {})
        self.assertEqual(
            summary["gcs_control_point_terrain_objective_source_counts"][
                "region_inverse_cost_weighted_passable_cell_centroid"
            ],
            1,
        )
        self.assertEqual(summary["gcs_control_point_sampled_terrain_cost_count"], 1)
        self.assertEqual(summary["gcs_control_point_sampled_terrain_cost_min"], 6.0)
        self.assertEqual(summary["gcs_control_point_sampled_terrain_cost_max"], 6.0)
        self.assertEqual(summary["gcs_control_point_sampled_terrain_cost_mean"], 6.0)
        self.assertEqual(summary["gcs_control_point_high_cost_exposure_delta_count"], 1)
        self.assertEqual(summary["gcs_control_point_high_cost_exposure_delta_min"], -1.0)
        self.assertEqual(summary["gcs_control_point_high_cost_exposure_delta_max"], -1.0)
        self.assertEqual(summary["gcs_control_point_high_cost_exposure_delta_mean"], -1.0)
        self.assertEqual(len(summary["gcs_control_point_candidate_audit"]), 1)
        self.assertEqual(
            summary["gcs_control_point_candidate_audit"][0]["terrain_objective_source"],
            "region_inverse_cost_weighted_passable_cell_centroid",
        )
        triage = summary["gcs_control_point_candidate_triage"]
        self.assertEqual(triage["schema_version"], "gcs-control-point-candidate-triage-summary/v1")
        self.assertEqual(triage["candidate_count"], 1)
        self.assertEqual(triage["attempted_count"], 1)
        self.assertEqual(triage["success_count"], 1)
        self.assertEqual(triage["selected_count"], 1)
        self.assertEqual(triage["route_artifact_count"], 1)
        self.assertEqual(triage["fallback_reason_counts"], {})
        triage_row = triage["candidates"][0]
        self.assertEqual(triage_row["scenario_id"], "npz_shadow_corridor")
        self.assertEqual(triage_row["action_index"], 1)
        self.assertEqual(triage_row["direction_cone_violation_count"], 0)
        self.assertEqual(triage_row["direction_cone_risk_flags"], [])
        self.assertTrue(triage_row["direction_cone_backend_enforced"])
        self.assertEqual(triage_row["direction_cone_eta"], 1.0)
        self.assertEqual(triage_row["direction_cone_rho_min"], 0.025)
        self.assertEqual(triage_row["direction_cone_tolerance_deg"], 45.0)
        self.assertEqual(triage_row["second_difference_weight"], 0.2)
        self.assertEqual(triage_row["motion_feasibility_status"], "feasible")
        self.assertEqual(triage_row["terrain_objective_weight"], 0.05)
        self.assertEqual(triage_row["cost_delta_vs_baseline"], -1.0)
        self.assertEqual(triage_row["high_cost_exposure_delta_vs_baseline"], -1.0)
        sweep = triage["calibration_sweep"]
        self.assertEqual(
            sweep["schema_version"],
            "gcs-control-point-candidate-calibration-sweep/v1",
        )
        self.assertFalse(sweep["default_change_recommended"])
        self.assertTrue(sweep["solver_rerun_required"])
        self.assertIn("terrain_objective_weight", sweep["sweep_dimensions"])
        self.assertEqual(sweep["observed_current_values"]["terrain_objective_weight"], [0.05])
        self.assertEqual(sweep["observed_current_values"]["second_difference_weight"], [0.2])
        self.assertFalse(sweep["safety_regression_guard"]["direction_cone_degradation_allowed"])
        artifacts = summary["gcs_control_point_candidate_artifacts"]
        self.assertEqual(artifacts["schema_version"], "gcs-control-point-candidate-artifact-index/v1")
        self.assertEqual(artifacts["candidate_count"], 1)
        self.assertEqual(artifacts["route_artifact_count"], 1)
        artifact_entry = artifacts["entries"][0]
        self.assertEqual(artifact_entry["scenario_id"], "npz_shadow_corridor")
        self.assertEqual(artifact_entry["action_index"], 1)
        self.assertTrue(route_artifact_exists)
        self.assertEqual(persisted_route["schema_version"], "path-planner-route/v1")
        self.assertEqual(
            persisted_route["gcs_trajectory_backend"],
            "pydrake_control_point_direction_cone_program",
        )
        self.assertIn("## GCS Control-Point Candidate Triage", report)
        self.assertIn("gcs-control-point-candidate-triage-summary/v1", report)
        self.assertEqual(summary["sampled_region_path_selected_count"], 1)
        self.assertEqual(summary["sampled_region_path_fallback_count"], 1)
        self.assertEqual(summary["sampled_region_path_source_counts"]["iris"], 1)
        self.assertEqual(summary["sampled_region_path_fallback_reasons"]["target_component_disconnected"], 1)
        self.assertEqual(summary["sampled_region_path_sample_attempt_count"], 10)
        self.assertEqual(summary["sampled_region_path_candidate_ranking_count"], 3)
        self.assertEqual(summary["sampled_region_path_anchor_region_added_count"], 1)
        self.assertEqual(summary["sampled_region_path_anchor_region_connected_count"], 1)
        self.assertEqual(summary["sampled_region_path_anchor_closure_attempt_count"], 3)
        self.assertEqual(summary["sampled_region_path_anchor_closure_connected_count"], 1)
        self.assertEqual(summary["sampled_region_path_anchor_closure_status_counts"]["connected"], 1)
        self.assertEqual(summary["sampled_region_path_anchor_closure_status_counts"]["unavailable"], 2)
        self.assertEqual(summary["sampled_region_path_anchor_closure_reason_counts"]["safe_bridge_found"], 1)
        self.assertEqual(
            summary["sampled_region_path_anchor_closure_reason_counts"]["safe_bridge_path_unavailable"],
            2,
        )
        self.assertEqual(
            summary["sampled_region_path_anchor_closure_connection_kind_counts"]["anchor_region_safe_bridge"],
            1,
        )
        self.assertEqual(
            summary["sampled_region_path_goal_classification_counts"]["goal_outside_region_coverage"],
            1,
        )
        self.assertEqual(summary["sampled_region_path_goal_classification_counts"]["covered"], 1)
        self.assertEqual(summary["sampled_region_path_connector_attempt_count"], 4)
        self.assertEqual(
            summary["sampled_region_path_connector_strategy_counts"]["cost_aware_constrained_astar"],
            2,
        )
        self.assertEqual(
            summary["sampled_region_path_connector_strategy_counts"]["bridge_aware_constrained_astar"],
            1,
        )
        self.assertEqual(
            summary["sampled_region_path_connector_strategy_counts"]["bridge_corridor_constrained_astar"],
            1,
        )
        self.assertEqual(summary["sampled_region_path_bridge_aware_connector_attempt_count"], 1)
        self.assertEqual(summary["sampled_region_path_bridge_aware_connector_available_count"], 0)
        self.assertEqual(summary["sampled_region_path_bridge_aware_connector_selected_count"], 0)
        self.assertEqual(summary["sampled_region_path_bridge_aware_connector_rejected_count"], 1)
        self.assertEqual(summary["sampled_region_path_bridge_aware_connector_status_counts"]["unavailable"], 1)
        self.assertEqual(
            summary["sampled_region_path_bridge_aware_fallback_reasons"][
                "bridge_aware_connector_path_unavailable"
            ],
            1,
        )
        self.assertEqual(summary["sampled_region_path_bridge_aware_bridge_cell_count"], 3)
        self.assertEqual(summary["sampled_region_path_bridge_aware_mask_added_cell_count"], 1)
        self.assertEqual(summary["sampled_region_path_bridge_corridor_connector_attempt_count"], 1)
        self.assertEqual(summary["sampled_region_path_bridge_corridor_connector_available_count"], 1)
        self.assertEqual(summary["sampled_region_path_bridge_corridor_connector_selected_count"], 1)
        self.assertEqual(summary["sampled_region_path_bridge_corridor_connector_rejected_count"], 0)
        self.assertEqual(summary["sampled_region_path_bridge_corridor_status_counts"]["available"], 1)
        self.assertEqual(summary["sampled_region_path_bridge_corridor_added_cell_count"], 4)
        self.assertEqual(summary["sampled_region_path_bridge_corridor_radius_counts"]["1"], 1)
        self.assertEqual(summary["sampled_region_path_terminal_adjusted_count"], 1)
        self.assertEqual(summary["sampled_region_path_terminal_adjustment_candidate_count"], 3)
        self.assertEqual(summary["sampled_region_path_terminal_adjustment_status_counts"]["selected"], 1)
        self.assertEqual(summary["sampled_region_path_reachable_component_status_counts"]["adjusted_connected"], 1)
        self.assertEqual(summary["sampled_region_path_reachable_component_status_counts"]["disconnected"], 1)
        self.assertEqual(
            summary["sampled_region_path_reachable_component_reason_counts"][
                "reachable_component_replacement_selected"
            ],
            1,
        )
        self.assertEqual(
            summary["sampled_region_path_reachable_component_reason_counts"]["target_component_disconnected"],
            1,
        )
        self.assertEqual(summary["sampled_region_path_reachable_component_disconnected_count"], 1)
        self.assertEqual(summary["sampled_region_path_reachable_component_replacement_selected_count"], 1)
        self.assertEqual(summary["sampled_region_path_reachable_component_terminal_candidate_count"], 1)
        self.assertEqual(summary["sampled_region_path_reachable_terminal_rescue_count"], 1)
        self.assertEqual(summary["sampled_region_path_proxy_goal_anchor_selected_count"], 0)
        self.assertEqual(summary["sampled_region_path_goal_rescue_candidate_count"], 2)
        self.assertEqual(summary["sampled_region_path_benefit_surface_present_count"], 1)
        self.assertEqual(summary["sampled_region_path_path_duplicate_with_baseline_count"], 0)
        self.assertEqual(summary["sampled_region_path_candidate_missing_metrics_count"], 1)
        self.assertEqual(
            summary["sampled_region_path_complexity_reason_counts"]["sampled_candidate_has_quality_gain"],
            1,
        )
        self.assertEqual(
            summary["sampled_region_path_complexity_reason_counts"]["candidate_missing_metrics"],
            1,
        )
        self.assertEqual(
            summary["sampled_region_path_terminal_adjustment_reason_counts"][
                "reachable_terminal_selected_by_component_projection"
            ],
            1,
        )
        self.assertEqual(
            summary["sampled_region_path_execution_tie_break_reason_counts"]["execution_tie_break_improved"],
            1,
        )
        self.assertEqual(
            summary["sampled_region_path_execution_tie_break_reason_counts"]["execution_tie_break_no_alternative"],
            1,
        )
        self.assertEqual(len(summary["sampled_region_path_candidate_audit"]), 2)
        fallback_audit = next(
            item
            for item in summary["sampled_region_path_candidate_audit"]
            if item["fallback_reason"] == "target_component_disconnected"
        )
        self.assertEqual(fallback_audit["scenario_id"], "npz_low_confidence_risk_band")
        self.assertEqual(fallback_audit["action_index"], 0)
        self.assertEqual(fallback_audit["region_source"], "grid_box")
        self.assertEqual(fallback_audit["region_sequence"], [0, 1])
        self.assertEqual(fallback_audit["start_goal_anchoring"]["start_region_id"], 0)
        self.assertEqual(fallback_audit["sample_attempt_count"], 3)
        self.assertEqual(fallback_audit["candidate_ranking_count"], 1)
        self.assertIn("candidate_metrics", fallback_audit)
        self.assertEqual(
            fallback_audit["execution_tie_break"]["reason"],
            "execution_tie_break_no_alternative",
        )
        selected_audit = next(
            item
            for item in summary["sampled_region_path_candidate_audit"]
            if item["status"] == "selected"
        )
        self.assertEqual(
            selected_audit["terminal_adjustment_report"]["reason_code"],
            "reachable_terminal_selected_by_component_projection",
        )
        self.assertEqual(selected_audit["execution_tie_break"]["reason"], "execution_tie_break_improved")
        self.assertIn("smoke", summary["scenario_group_summary"])
        self.assertIn("smoke", summary["diagnostic_interpretation"]["scenario_group_interpretation"])
        self.assertGreaterEqual(summary["selection_changed_count"], 1)
        self.assertGreater(summary["selection_changed_rate"], 0.0)
        self.assertGreaterEqual(summary["replan_count"], 1)
        self.assertGreaterEqual(summary["region_graph_disconnected_count"], 1)
        self.assertTrue(any(item["selection_changed_by_path_feedback"] for item in summary["scenarios"]))
        for item in summary["scenarios"]:
            self.assertIn("selected_path_cost_before_feedback", item)
            self.assertIn("path_cost_delta_after_feedback", item)
            self.assertIn("baseline_vs_feedback", item)
            self.assertIn("sampled_region_path_candidate_audit", item)
            self.assertIn("diagnostic_interpretation", item)
            self.assertIn("target_replacement_reason", item["diagnostic_interpretation"])
            self.assertIn("failure_sources", item["diagnostic_interpretation"])
            self.assertEqual(
                item["baseline_vs_feedback"]["path_cost_delta_after_feedback"],
                item["path_cost_delta_after_feedback"],
            )
        self.assertTrue(
            any(
                "region_graph_disconnected" in item["diagnostic_interpretation"]["failure_sources"]
                for item in summary["scenarios"]
            )
        )
        self.assertIn("npz_shadow_corridor", report)
        self.assertIn("npz_rock_field_multi_pose", report)
        self.assertIn("npz_low_confidence_risk_band", report)
        self.assertIn("## Baseline vs Feedback", report)
        self.assertIn("## Candidate Paths", report)
        self.assertIn("## Diagnostic Interpretation", report)
        self.assertIn("## Candidate Diagnostics", report)
        self.assertIn("## IRIS Diagnostics", report)
        self.assertIn("## Region Graph Diagnostics", report)
        self.assertIn("## Sampled Region Path Diagnostics", report)
        self.assertIn("## Sampled Region Path Candidate Audit", report)
        self.assertIn("## Scenario Groups", report)
        self.assertIn(
            "| scenario | group | replacement_reason | failure_sources | primary_failure_reason | iris_region_graph_signal | open_grid_fallback |",
            report,
        )
        self.assertIn(
            "| scenario | action | cell | reachable | replan | failure | flags | iris_status | iris_fallback | graph_source | graph_fallback | graph_connected | open_grid |",
            report,
        )
        self.assertIn(
            "| scenario | group | before | after | changed | before_path_cost | after_path_cost | delta | coverage_delta | reachable | failures | replans |",
            report,
        )
        self.assertIn(
            "| scenario | group | before | after | failures | replans | iris_status_counts | iris_fallback_reasons | iris_region_count |",
            report,
        )
        self.assertIn(
            "| scenario | group | before | after | failures | replans | graph_source_counts | fallback_reasons | disconnected |",
            report,
        )
        self.assertIn(
            "| scenario | group | selected | fallback | status_counts | source_counts | fallback_reasons |",
            report,
        )
        self.assertIn(
            "| scenario | action | source | status | fallback | sequence | attempts | rankings | edge_transitions | cost_delta |",
            report,
        )
        self.assertIn("| scenario | action | cell | reachable | path_cost | risk | utility | replan | failure |", report)
        json.dumps(summary)

    def test_path_planner_route_adapter_runs_cli_without_importing_path_planner(self):
        from model_explorer.policy.planning import PathPlannerRouteAdapter, PathPlanRequest

        contract = load_contract_from_dict(
            minimal_contract(
                goals=[{"cell": [5, 5], "utility": 0.4, "reachable": True}],
            )
        )
        contract = replace_contract_grid(contract, width=6, height=6, resolution=1.0)
        selected_goal = contract.top_goals[0]

        with tempfile.TemporaryDirectory() as tmpdir:
            result = PathPlannerRouteAdapter(
                path_planner_root=ROOT.parent / "path-planner",
                output_dir=Path(tmpdir) / "planner-output",
            ).plan(
                PathPlanRequest(
                    contract=contract,
                    step_index=0,
                    action_index=0,
                    selected_goal=selected_goal,
                    current_cell=(0, 0),
                )
            )

        self.assertTrue(result.feasible, result.metadata)
        self.assertGreater(result.path_cost, 0.0)
        self.assertEqual(result.metadata["mode"], "cli")
        self.assertIn("platform_aware_astar", result.metadata["diagnostics"]["search_mode"])
        self.assertNotIn("path_planner", sys.modules)

    def test_dev_platform_contract_example_can_emit_path_planner_request(self):
        from model_explorer.policy.planning import PathPlanRequest, build_path_planner_request_dict

        contract = load_contract_from_dict(json.loads(DEV_PLATFORM_CONTRACT_EXAMPLE.read_text(encoding="utf-8")))
        decision = select_goal(contract)

        planner_request = build_path_planner_request_dict(
            PathPlanRequest(
                contract=contract,
                step_index=0,
                action_index=0,
                selected_goal=decision.selected_goal,
                current_cell=(0, 0),
            )
        )

        self.assertEqual(planner_request["schema_version"], "path-planner-request/v1")
        self.assertEqual(planner_request["grid"]["width"], 32)
        self.assertEqual(planner_request["goal"], [24, 10])
        self.assertEqual(len(planner_request["cost"]), 20)
        self.assertEqual(planner_request["metadata"]["passable_mask_source"], "open_grid_fallback")

    def test_collector_uses_planning_result_for_reward_and_failure_metrics(self):
        from model_explorer.policy.collector import collect_dynamic_rollout_episode
        from model_explorer.policy.planning import PathPlanResult
        from model_explorer.policy.provider import SequenceContractProvider

        class FixedPlanner:
            def __init__(self, result):
                self.result = result
                self.requests = []

            def plan(self, request):
                self.requests.append(request)
                return self.result

        contract = load_contract_from_dict(
            minimal_contract(
                goals=[{"cell": [1, 1], "utility": 0.4, "reachable": True, "path_cost": 100.0, "risk": 0.9}],
                observation_update={"coverage_rate": 0.1, "coverage_rate_delta": 0.2},
            )
        )

        successful_episode = collect_dynamic_rollout_episode(
            SequenceContractProvider([contract]),
            planning_adapter=FixedPlanner(PathPlanResult(feasible=True, path_cost=4.0, path_length=4.0, risk=0.25)),
            max_steps=1,
        )

        self.assertEqual(successful_episode.transitions[0].info.path_cost, 4.0)
        self.assertEqual(successful_episode.transitions[0].info.risk, 0.25)
        self.assertAlmostEqual(successful_episode.transitions[0].reward, 0.2 - 0.1 * 0.04 - 0.2 * 0.25)
        self.assertEqual(successful_episode.metrics.total_path_cost, 4.0)
        self.assertEqual(successful_episode.metrics.failure_count, 0)

        failed_episode = collect_dynamic_rollout_episode(
            SequenceContractProvider([contract]),
            planning_adapter=FixedPlanner(
                PathPlanResult(
                    feasible=False,
                    path_cost=7.0,
                    path_length=7.0,
                    risk=0.5,
                    failure_reason="path_blocked",
                    replan_required=True,
                )
            ),
            max_steps=1,
        )

        self.assertEqual(failed_episode.transitions[0].action_index, 0)
        self.assertEqual(failed_episode.transitions[0].info.failure_reason, "path_blocked")
        self.assertEqual(failed_episode.transitions[0].info.path_cost, 7.0)
        self.assertEqual(failed_episode.metrics.failure_count, 1)
        self.assertEqual(failed_episode.metrics.replan_count, 1)

    def test_collector_applies_reward_config_to_planning_path_cost(self):
        from model_explorer.policy.collector import collect_dynamic_rollout_episode
        from model_explorer.policy.planning import PathPlanResult
        from model_explorer.policy.provider import SequenceContractProvider

        class FixedPlanner:
            def plan(self, request):
                return PathPlanResult(feasible=True, path_cost=10.0, path_length=10.0, risk=0.0)

        contract = load_contract_from_dict(
            minimal_contract(
                goals=[{"cell": [1, 1], "utility": 0.4, "reachable": True}],
                observation_update={"coverage_rate": 0.1, "coverage_rate_delta": 1.0},
            )
        )

        episode = collect_dynamic_rollout_episode(
            SequenceContractProvider([contract]),
            planning_adapter=FixedPlanner(),
            reward_config={"path_cost_weight": 0.5, "path_cost_normalizer": 10.0},
            max_steps=1,
        )

        self.assertAlmostEqual(episode.transitions[0].reward, 0.5)


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

    def test_old_rollout_json_without_missing_indicators_uses_compatible_defaults(self):
        from model_explorer.policy.collector import collect_rollout_episode
        from model_explorer.policy.rollout_io import read_rollout_episode

        scenario = [
            load_contract_from_dict(
                minimal_contract(
                    goals=[{"cell": [1, 1], "utility": 0.4, "reachable": True}],
                    observation_update={"coverage_rate": 0.1, "coverage_rate_delta": 0.05},
                )
            )
        ]
        episode = collect_rollout_episode(scenario, max_candidates=2)
        payload = episode.to_dict()
        payload["transitions"][0]["observation"].pop("candidate_missing_indicator_names", None)
        payload["transitions"][0]["observation"].pop("candidate_missing_indicators", None)
        if payload["transitions"][0]["next_observation"] is not None:
            payload["transitions"][0]["next_observation"].pop("candidate_missing_indicator_names", None)
            payload["transitions"][0]["next_observation"].pop("candidate_missing_indicators", None)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "old-rollout.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            loaded = read_rollout_episode(path)

        observation = loaded.transitions[0].observation
        self.assertEqual(len(observation.candidate_missing_indicators), len(observation.candidate_cells))
        self.assertTrue(all(value == 0.0 for row in observation.candidate_missing_indicators for value in row))

    def test_rollout_episodes_round_trip_through_jsonl_file(self):
        from model_explorer.policy.collector import collect_rollout_episode
        from model_explorer.policy.rollout_io import (
            read_rollout_episodes,
            write_rollout_episodes_jsonl,
        )

        successful_episode = collect_rollout_episode(
            [
                load_contract_from_dict(
                    minimal_contract(
                        goals=[{"cell": [1, 1], "utility": 0.4, "reachable": True}],
                        observation_update={"coverage_rate": 0.1, "coverage_rate_delta": 0.05},
                    )
                )
            ],
            max_candidates=2,
        )
        failure_episode = collect_rollout_episode(
            [
                load_contract_from_dict(
                    minimal_contract(
                        goals=[{"cell": [2, 1], "utility": 0.5, "reachable": False}],
                        observation_update={},
                    )
                )
            ],
            max_candidates=2,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "rollouts.jsonl"
            write_rollout_episodes_jsonl(path, [successful_episode, failure_episode])
            loaded = read_rollout_episodes(path)

        self.assertEqual(len(loaded), 2)
        self.assertEqual(loaded[0].transitions[0].observation.action_mask, (True, False))
        self.assertEqual(loaded[1].transitions[0].action_index, -1)
        self.assertEqual(loaded[1].transitions[0].info.failure_reason, "no_reachable_goal")


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

    def test_training_on_multiple_episodes_saves_checkpoint_metadata(self):
        import torch

        from model_explorer.policy.collector import collect_rollout_episode
        from model_explorer.policy.training import load_policy_checkpoint, train_policy_on_episodes

        episodes = []
        for cell_x, coverage_delta in ((0, 0.05), (2, 0.08)):
            scenario = [
                load_contract_from_dict(
                    minimal_contract(
                        goals=[
                            {"cell": [cell_x, 1], "utility": 0.5, "reachable": True},
                            {"cell": [1, 1], "utility": 9.0, "reachable": False},
                        ],
                        observation_update={
                            "coverage_rate": coverage_delta,
                            "coverage_rate_delta": coverage_delta,
                        },
                    )
                )
            ]
            episodes.append(collect_rollout_episode(scenario, max_candidates=2))

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "policy.pt"
            result = train_policy_on_episodes(
                episodes,
                checkpoint_path=checkpoint_path,
                seed=11,
                hidden_size=16,
            )
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            scorer = load_policy_checkpoint(checkpoint_path)

        self.assertEqual(result["sample_count"], 2)
        self.assertEqual(checkpoint["metadata"]["sample_count"], 2)
        self.assertEqual(checkpoint["metadata"]["seed"], 11)
        self.assertTrue(torch.isfinite(torch.tensor(result["total_loss"])))

        decision = select_goal(scenario[0], policy=scorer)

        self.assertIsNotNone(decision.selected_goal)
        self.assertNotEqual(decision.selected_goal.cell, (1, 1))

    def test_training_on_variable_candidate_counts_pads_action_masks(self):
        import torch

        from model_explorer.policy.collector import collect_rollout_episode
        from model_explorer.policy.training import train_policy_on_episodes

        one_candidate_episode = collect_rollout_episode(
            [
                load_contract_from_dict(
                    minimal_contract(
                        goals=[{"cell": [0, 1], "utility": 0.5, "reachable": True}],
                        observation_update={"coverage_rate": 0.05, "coverage_rate_delta": 0.05},
                    )
                )
            ]
        )
        two_candidate_episode = collect_rollout_episode(
            [
                load_contract_from_dict(
                    minimal_contract(
                        goals=[
                            {"cell": [1, 1], "utility": 0.6, "reachable": True},
                            {"cell": [2, 1], "utility": 0.4, "reachable": True},
                        ],
                        observation_update={"coverage_rate": 0.1, "coverage_rate_delta": 0.05},
                    )
                )
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "policy.pt"
            result = train_policy_on_episodes(
                [one_candidate_episode, two_candidate_episode],
                checkpoint_path=checkpoint_path,
                seed=13,
                hidden_size=16,
            )
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        self.assertEqual(result["sample_count"], 2)
        self.assertEqual(checkpoint["metadata"]["action_count"], 2)
        self.assertTrue(torch.isfinite(torch.tensor(result["total_loss"])))


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
            self.assertIn("value_coverage", metrics)

    def test_policy_action_diagnostics_keep_masked_actions_at_zero_probability(self):
        from model_explorer.policy.evaluation import evaluate_policy_baselines

        class InvalidPreferringPolicy:
            def score(self, observation):
                return (0.1, 100.0, 0.2)

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
                            "utility": 0.1,
                            "reachable": False,
                            "path_cost": 1.0,
                            "risk": 0.0,
                            "expected_coverage_rate_delta": 0.9,
                        },
                        {
                            "cell": [2, 2],
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

        report = evaluate_policy_baselines(scenario, torch_policy=InvalidPreferringPolicy())
        diagnostic = report["torch_policy"]["action_diagnostics"][0]

        self.assertEqual(diagnostic["selected_index"], 2)
        self.assertEqual(diagnostic["selected_cell"], [2, 2])
        self.assertTrue(diagnostic["selected_action_mask_valid"])
        self.assertEqual(diagnostic["valid_action_count"], 2)
        self.assertEqual(diagnostic["max_masked_action_probability"], 0.0)
        self.assertGreater(diagnostic["selected_action_probability"], 0.0)
        self.assertTrue(math.isfinite(diagnostic["entropy"]))
        self.assertFalse(diagnostic["agrees_with_utility"])
        self.assertTrue(diagnostic["agrees_with_coverage_heuristic"])

    def test_action_sensitive_metrics_and_oracle_regret_ignore_unreachable_candidates(self):
        from model_explorer.policy.evaluation import evaluate_policy_baselines

        class ReachableLowCoveragePolicy:
            def score(self, observation):
                return (10.0, 100.0, 1.0)

        scenario = [
            load_contract_from_dict(
                minimal_contract(
                    goals=[
                        {
                            "cell": [0, 0],
                            "utility": 0.8,
                            "reachable": True,
                            "expected_coverage_rate_delta": 0.1,
                            "value": 0.3,
                        },
                        {
                            "cell": [1, 1],
                            "utility": 0.1,
                            "reachable": False,
                            "expected_coverage_rate_delta": 0.99,
                            "risk": 0.0,
                            "path_cost": 0.0,
                            "value": 1.0,
                        },
                        {
                            "cell": [2, 2],
                            "utility": 0.6,
                            "reachable": True,
                            "expected_coverage_rate_delta": 0.5,
                            "risk": 0.4,
                            "path_cost": 4.0,
                            "value": 0.7,
                        },
                    ],
                    observation_update={"coverage_rate": 0.2, "coverage_rate_delta": 0.1},
                )
            )
        ]

        report = evaluate_policy_baselines(scenario, torch_policy=ReachableLowCoveragePolicy())
        metrics = report["torch_policy"]

        self.assertEqual(metrics["selected_cells"], [[0, 0]])
        self.assertEqual(metrics["oracle_actions"]["coverage_oracle_cell"], [2, 2])
        self.assertEqual(metrics["oracle_actions"]["composite_oracle_cell"], [2, 2])
        self.assertNotEqual(metrics["oracle_actions"]["coverage_oracle_cell"], [1, 1])
        self.assertAlmostEqual(
            metrics["action_sensitive_metrics"]["selected_expected_coverage_delta"],
            0.1,
        )
        self.assertGreater(metrics["oracle_regret"]["coverage_regret"], 0.0)
        self.assertGreaterEqual(metrics["oracle_regret"]["composite_regret"], 0.0)
        self.assertIn("candidate_coverage_spread", metrics["sample_discriminativeness"])
        for section in (
            "action_sensitive_metrics",
            "oracle_metrics",
            "oracle_regret",
            "sample_discriminativeness",
        ):
            self.assertTrue(
                all(
                    math.isfinite(float(value))
                    for value in metrics[section].values()
                    if isinstance(value, (int, float))
                )
            )

    def test_baseline_evaluation_can_aggregate_multiple_scenarios(self):
        from model_explorer.policy.evaluation import evaluate_policy_baseline_scenarios

        first = [
            load_contract_from_dict(
                minimal_contract(
                    goals=[{"cell": [0, 0], "utility": 0.9, "reachable": True}],
                    observation_update={
                        "coverage_rate": 0.1,
                        "coverage_rate_delta": 0.1,
                        "value_coverage": 0.2,
                    },
                )
            )
        ]
        second = [
            load_contract_from_dict(
                minimal_contract(
                    goals=[{"cell": [1, 1], "utility": 0.8, "reachable": True}],
                    observation_update={
                        "coverage_rate": 0.3,
                        "coverage_rate_delta": 0.2,
                        "value_coverage": 0.4,
                    },
                )
            )
        ]

        report = evaluate_policy_baseline_scenarios([first, second])

        self.assertEqual(report["utility"]["episode_count"], 2)
        self.assertAlmostEqual(report["utility"]["average_final_coverage_rate"], 0.2)
        self.assertAlmostEqual(report["utility"]["value_coverage"], 0.6)

    def test_baseline_evaluation_uses_planning_feedback_for_path_metrics(self):
        from model_explorer.policy.evaluation import evaluate_policy_baselines
        from model_explorer.policy.planning import PathPlanResult

        class FixedPlanner:
            def plan(self, request):
                return PathPlanResult(
                    feasible=False,
                    path_cost=8.0,
                    path_length=8.0,
                    risk=0.4,
                    failure_reason="path_blocked",
                    replan_required=True,
                )

        scenario = [
            load_contract_from_dict(
                minimal_contract(
                    goals=[{"cell": [1, 1], "utility": 0.4, "reachable": True, "path_cost": 100.0}],
                    observation_update={"coverage_rate": 0.1, "coverage_rate_delta": 0.05},
                )
            )
        ]

        report = evaluate_policy_baselines(scenario, planning_adapter=FixedPlanner())

        self.assertEqual(report["utility"]["total_path_cost"], 8.0)
        self.assertEqual(report["utility"]["average_risk"], 0.4)
        self.assertEqual(report["utility"]["failure_count"], 1)
        self.assertEqual(report["utility"]["replan_count"], 1)

    def test_baseline_evaluation_includes_feedback_aware_strategy_when_planner_is_available(self):
        from model_explorer.policy.evaluation import evaluate_policy_baselines
        from model_explorer.policy.planning import PathPlanResult

        class FixedPlanner:
            def plan(self, request):
                if request.action_index == 0:
                    return PathPlanResult(
                        feasible=False,
                        path_cost=0.0,
                        path_length=0.0,
                        risk=0.0,
                        failure_reason="goal_blocked",
                        replan_required=True,
                    )
                return PathPlanResult(feasible=True, path_cost=3.0, path_length=3.0, risk=0.1)

        scenario = [
            load_contract_from_dict(
                minimal_contract(
                    goals=[
                        {
                            "cell": [1, 1],
                            "utility": 0.9,
                            "reachable": True,
                            "expected_coverage_rate_delta": 0.9,
                        },
                        {
                            "cell": [2, 1],
                            "utility": 0.2,
                            "reachable": True,
                            "expected_coverage_rate_delta": 0.2,
                        },
                    ],
                    observation_update={"coverage_rate": 0.1, "coverage_rate_delta": 0.05},
                )
            )
        ]

        report = evaluate_policy_baselines(scenario, planning_adapter=FixedPlanner())

        self.assertIn("feedback_aware", report)
        self.assertEqual(report["coverage_heuristic"]["selected_cells"], [[1, 1]])
        self.assertEqual(report["feedback_aware"]["selected_cells"], [[2, 1]])
        self.assertEqual(report["feedback_aware"]["failure_count"], 0)

    def test_torch_policy_reports_feedback_aware_agreement_and_baseline_delta(self):
        from model_explorer.policy.evaluation import evaluate_policy_baselines
        from model_explorer.policy.experiment import _baseline_deltas
        from model_explorer.policy.planning import PathPlanResult

        class FixedPlanner:
            def plan(self, request):
                if request.action_index == 0:
                    return PathPlanResult(
                        feasible=False,
                        path_cost=0.0,
                        path_length=0.0,
                        risk=0.0,
                        failure_reason="goal_blocked",
                        replan_required=True,
                    )
                return PathPlanResult(feasible=True, path_cost=3.0, path_length=3.0, risk=0.1)

        class FirstActionPolicy:
            def score(self, observation):
                return [10.0, 0.0]

        scenario = [
            load_contract_from_dict(
                minimal_contract(
                    goals=[
                        {
                            "cell": [1, 1],
                            "utility": 0.9,
                            "reachable": True,
                            "expected_coverage_rate_delta": 0.9,
                        },
                        {
                            "cell": [2, 1],
                            "utility": 0.2,
                            "reachable": True,
                            "expected_coverage_rate_delta": 0.2,
                        },
                    ],
                    observation_update={"coverage_rate": 0.1, "coverage_rate_delta": 0.05},
                )
            )
        ]

        report = evaluate_policy_baselines(
            scenario,
            torch_policy=FirstActionPolicy(),
            planning_adapter=FixedPlanner(),
        )

        torch_metrics = report["torch_policy"]
        self.assertEqual(report["feedback_aware"]["selected_action_indices"], [1])
        self.assertEqual(torch_metrics["selected_action_indices"], [0])
        self.assertEqual(torch_metrics["feedback_aware_action_agreement_rate"], 0.0)
        self.assertEqual(torch_metrics["feedback_aware_selected_cell_agreement_rate"], 0.0)
        self.assertFalse(torch_metrics["action_diagnostics"][0]["agrees_with_feedback_aware"])
        self.assertIn("feedback_aware", _baseline_deltas(report)["torch_policy"])

    def test_torch_policy_reports_teacher_rank_topk_and_margin_bucket_agreement(self):
        from model_explorer.policy.evaluation import evaluate_policy_baselines
        from model_explorer.policy.planning import PathPlanResult

        class FixedPlanner:
            def plan(self, request):
                if request.action_index == 0:
                    return PathPlanResult(
                        feasible=False,
                        path_cost=0.0,
                        path_length=0.0,
                        risk=0.0,
                        failure_reason="goal_blocked",
                        replan_required=True,
                    )
                return PathPlanResult(feasible=True, path_cost=2.0, path_length=2.0, risk=0.1)

        class RunnerUpPolicy:
            def score(self, observation):
                return [10.0, 0.0]

        scenario = [
            load_contract_from_dict(
                minimal_contract(
                    goals=[
                        {"cell": [1, 1], "utility": 0.9, "reachable": True, "expected_coverage_rate_delta": 0.9},
                        {"cell": [2, 1], "utility": 0.2, "reachable": True, "expected_coverage_rate_delta": 0.2},
                    ],
                    observation_update={"coverage_rate": 0.1, "coverage_rate_delta": 0.05},
                )
            )
        ]

        report = evaluate_policy_baselines(
            scenario,
            torch_policy=RunnerUpPolicy(),
            planning_adapter=FixedPlanner(),
        )

        feedback_metrics = report["feedback_aware"]
        torch_metrics = report["torch_policy"]
        diagnostic = torch_metrics["action_diagnostics"][0]
        bucket = torch_metrics["feedback_aware_margin_bucket_agreement"]["high"]

        self.assertEqual(feedback_metrics["teacher_ranked_action_indices"], [[1, 0]])
        self.assertEqual(torch_metrics["feedback_aware_action_agreement_rate"], 0.0)
        self.assertEqual(torch_metrics["feedback_aware_top2_action_agreement_rate"], 1.0)
        self.assertEqual(torch_metrics["feedback_aware_topk_action_agreement_rate"], 1.0)
        self.assertEqual(torch_metrics["feedback_aware_teacher_rank_mean"], 2.0)
        self.assertEqual(diagnostic["feedback_aware_teacher_rank"], 2)
        self.assertTrue(diagnostic["agrees_with_feedback_aware_top2"])
        self.assertGreater(diagnostic["feedback_aware_teacher_score_margin"], 0.0)
        self.assertEqual(bucket["comparison_count"], 1)
        self.assertEqual(bucket["action_agreement_count"], 0)
        self.assertEqual(bucket["top2_action_agreement_count"], 1)

    def test_rollout_with_planning_adapter_uses_feedback_aware_selection(self):
        from model_explorer.policy.collector import collect_dynamic_rollout_episode
        from model_explorer.policy.planning import PathPlanResult
        from model_explorer.policy.provider import SequenceContractProvider

        class FixedPlanner:
            def plan(self, request):
                if request.action_index == 0:
                    return PathPlanResult(
                        feasible=False,
                        path_cost=0.0,
                        path_length=0.0,
                        risk=0.0,
                        failure_reason="goal_blocked",
                        replan_required=True,
                    )
                return PathPlanResult(feasible=True, path_cost=3.0, path_length=3.0, risk=0.1)

        contract = load_contract_from_dict(
            minimal_contract(
                goals=[
                    {
                        "cell": [1, 1],
                        "utility": 0.9,
                        "reachable": True,
                        "expected_coverage_rate_delta": 0.9,
                    },
                    {
                        "cell": [2, 1],
                        "utility": 0.2,
                        "reachable": True,
                        "expected_coverage_rate_delta": 0.2,
                    },
                ],
                observation_update={"coverage_rate": 0.1, "coverage_rate_delta": 0.2},
            )
        )

        episode = collect_dynamic_rollout_episode(
            SequenceContractProvider([contract]),
            planning_adapter=FixedPlanner(),
            max_steps=1,
            max_candidates=2,
        )

        transition = episode.transitions[0]
        self.assertEqual(transition.action_index, 1)
        self.assertEqual(transition.info.selected_cell, (2, 1))
        self.assertEqual(transition.info.extra["selection_strategy"], "feedback_aware")
        self.assertIsNone(transition.info.failure_reason)
        self.assertEqual(transition.info.path_cost, 3.0)
        self.assertEqual(episode.metrics.failure_count, 0)

    def test_feedback_aware_rollout_records_teacher_runner_up_score_margin(self):
        from model_explorer.policy.collector import collect_rollout_episode
        from model_explorer.policy.planning import PathPlanResult

        class FixedPlanner:
            def plan(self, request):
                if request.action_index == 0:
                    return PathPlanResult(
                        feasible=False,
                        failure_reason="path_blocked",
                        replan_required=True,
                    )
                return PathPlanResult(feasible=True, path_cost=2.0, path_length=2.0, risk=0.1)

        contract = load_contract_from_dict(
            minimal_contract(
                goals=[
                    {"cell": [1, 1], "utility": 0.9, "reachable": True, "expected_coverage_rate_delta": 0.9},
                    {"cell": [2, 1], "utility": 0.2, "reachable": True, "expected_coverage_rate_delta": 0.2},
                ],
                observation_update={"coverage_rate": 0.1, "coverage_rate_delta": 0.2},
            )
        )

        episode = collect_rollout_episode(
            [contract],
            planning_adapter=FixedPlanner(),
            selection_strategy="feedback_aware",
            max_candidates=2,
        )
        extra = episode.transitions[0].info.extra

        self.assertEqual(extra["teacher_action_index"], 1)
        self.assertEqual(extra["teacher_runner_up_action_index"], 0)
        self.assertEqual(extra["teacher_ranked_action_indices"], [1, 0])
        self.assertAlmostEqual(extra["teacher_score"], extra["selection_score"])
        self.assertLess(extra["teacher_runner_up_score"], extra["teacher_score"])
        self.assertAlmostEqual(
            extra["teacher_score_margin"],
            extra["teacher_score"] - extra["teacher_runner_up_score"],
        )
        self.assertGreater(extra["teacher_score_margin"], 0.0)

    def test_feedback_aware_rollout_dataset_summary_counts_selection_strategy(self):
        from model_explorer.policy.collector import collect_rollout_episode
        from model_explorer.policy.dataset import summarize_rollout_dataset
        from model_explorer.policy.planning import PathPlanResult

        class FixedPlanner:
            def plan(self, request):
                if request.action_index == 0:
                    return PathPlanResult(
                        feasible=False,
                        failure_reason="path_blocked",
                        replan_required=True,
                    )
                return PathPlanResult(feasible=True, path_cost=3.0, path_length=3.0, risk=0.1)

        contract = load_contract_from_dict(
            minimal_contract(
                goals=[
                    {
                        "cell": [1, 1],
                        "utility": 0.9,
                        "reachable": True,
                        "expected_coverage_rate_delta": 0.9,
                    },
                    {
                        "cell": [2, 1],
                        "utility": 0.2,
                        "reachable": True,
                        "expected_coverage_rate_delta": 0.2,
                    },
                ],
                observation_update={"coverage_rate": 0.1, "coverage_rate_delta": 0.2},
            )
        )

        episode = collect_rollout_episode(
            [contract],
            planning_adapter=FixedPlanner(),
            selection_strategy="feedback_aware",
            max_candidates=2,
        )
        summary = summarize_rollout_dataset([episode])

        self.assertEqual(summary["selection_strategy_counts"], {"feedback_aware": 1})
        self.assertEqual(summary["selection_strategy_fractions"]["feedback_aware"], 1.0)
        self.assertEqual(summary["primary_selection_strategy"], "feedback_aware")

    def test_dataset_summary_records_teacher_margin_distribution_and_buckets(self):
        from model_explorer.policy.collector import collect_rollout_episode
        from model_explorer.policy.dataset import summarize_rollout_dataset
        from model_explorer.policy.rollout import RolloutEpisode

        base_episode = collect_rollout_episode(
            [
                load_contract_from_dict(
                    minimal_contract(
                        goals=[
                            {"cell": [1, 1], "utility": 0.5, "reachable": True},
                            {"cell": [2, 1], "utility": 0.4, "reachable": True},
                        ],
                        observation_update={"coverage_rate": 0.1, "coverage_rate_delta": 0.1},
                    )
                )
            ],
            max_candidates=2,
        )
        base_transition = base_episode.transitions[0]
        margins = (0.03, 0.5)
        transitions = tuple(
            replace(
                base_transition,
                info=replace(
                    base_transition.info,
                    extra={
                        **base_transition.info.extra,
                        "selection_strategy": "feedback_aware",
                        "requested_selection_strategy": "feedback_aware",
                        "teacher_score_margin": margin,
                    },
                ),
            )
            for margin in margins
        )
        episode = RolloutEpisode(transitions=transitions, metrics=base_episode.metrics)

        summary = summarize_rollout_dataset([episode])

        self.assertEqual(summary["teacher_margin_sample_count"], 2)
        self.assertEqual(summary["teacher_low_margin_sample_count"], 1)
        self.assertEqual(summary["teacher_margin_bucket_counts"], {"high": 1, "low": 1, "medium": 0, "missing": 0})
        self.assertAlmostEqual(summary["teacher_score_margin"]["min"], 0.03)
        self.assertAlmostEqual(summary["teacher_score_margin"]["max"], 0.5)
        self.assertAlmostEqual(summary["teacher_score_margin"]["mean"], 0.265)
        self.assertAlmostEqual(summary["teacher_score_margin"]["std"], 0.235)


class ExperimentManifestTests(unittest.TestCase):
    def test_manifest_accepts_optional_v1_schema_and_rejects_unknown_schema(self):
        from model_explorer.policy.experiment import load_experiment_manifest

        with tempfile.TemporaryDirectory() as tmpdir:
            scenario_path = Path(tmpdir) / "scenario.json"
            rollout_path = Path(tmpdir) / "rollouts.jsonl"
            evaluation_path = Path(tmpdir) / "evaluation.json"
            scenario_path.write_text(json.dumps(minimal_contract()), encoding="utf-8")
            valid_manifest_path = Path(tmpdir) / "valid.json"
            valid_manifest_path.write_text(
                json.dumps(
                    {
                        "schema_version": "model-explorer-experiment/v1",
                        "scenarios": [str(scenario_path)],
                        "planner": {"backend": "contract_cost"},
                        "outputs": {
                            "rollouts": str(rollout_path),
                            "evaluation": str(evaluation_path),
                        },
                    }
                ),
                encoding="utf-8",
            )
            legacy_manifest_path = Path(tmpdir) / "legacy.json"
            legacy_manifest_path.write_text(
                json.dumps(
                    {
                        "scenarios": [str(scenario_path)],
                        "outputs": {
                            "rollouts": str(rollout_path),
                            "evaluation": str(evaluation_path),
                        },
                    }
                ),
                encoding="utf-8",
            )
            invalid_manifest_path = Path(tmpdir) / "invalid.json"
            invalid_manifest_path.write_text(
                json.dumps(
                    {
                        "schema_version": "model-explorer-experiment/v2",
                        "scenarios": [str(scenario_path)],
                        "outputs": {
                            "rollouts": str(rollout_path),
                            "evaluation": str(evaluation_path),
                        },
                    }
                ),
                encoding="utf-8",
            )

            valid_manifest = load_experiment_manifest(valid_manifest_path)
            legacy_manifest = load_experiment_manifest(legacy_manifest_path)

            with self.assertRaisesRegex(ValueError, "schema_version"):
                load_experiment_manifest(invalid_manifest_path)

        self.assertEqual(valid_manifest.schema_version, "model-explorer-experiment/v1")
        self.assertEqual(legacy_manifest.schema_version, "model-explorer-experiment/v1")

    def test_manifest_reports_missing_required_fields(self):
        from model_explorer.policy.experiment import load_experiment_manifest

        with tempfile.TemporaryDirectory() as tmpdir:
            missing_scenarios_path = Path(tmpdir) / "missing-scenarios.json"
            missing_scenarios_path.write_text(
                json.dumps({"outputs": {"rollouts": "rollouts.jsonl", "evaluation": "evaluation.json"}}),
                encoding="utf-8",
            )
            missing_rollouts_path = Path(tmpdir) / "missing-rollouts.json"
            missing_rollouts_path.write_text(
                json.dumps({"scenarios": ["scenario.json"], "outputs": {"evaluation": "evaluation.json"}}),
                encoding="utf-8",
            )
            missing_evaluation_path = Path(tmpdir) / "missing-evaluation.json"
            missing_evaluation_path.write_text(
                json.dumps({"scenarios": ["scenario.json"], "outputs": {"rollouts": "rollouts.jsonl"}}),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "scenarios"):
                load_experiment_manifest(missing_scenarios_path)
            with self.assertRaisesRegex(ValueError, "outputs.rollouts"):
                load_experiment_manifest(missing_rollouts_path)
            with self.assertRaisesRegex(ValueError, "outputs.evaluation"):
                load_experiment_manifest(missing_evaluation_path)

    def test_experiment_fixture_assets_are_reproducible_and_runnable(self):
        from model_explorer.policy.experiment import run_experiment_manifest

        manifest_path = SYNTHETIC_EXPERIMENT_FIXTURE / "experiment.json"
        sidecar_path = SYNTHETIC_EXPERIMENT_FIXTURE / "sidecar-grid.json"
        contract_cost_manifest = SYNTHETIC_EXPERIMENT_FIXTURE / "contract-cost-experiment.json"
        straight_line_manifest = SYNTHETIC_EXPERIMENT_FIXTURE / "straight-line-experiment.json"
        formal_training_manifest = SYNTHETIC_EXPERIMENT_FIXTURE / "formal-training-experiment.json"
        architecture_smoke_manifest = SYNTHETIC_EXPERIMENT_FIXTURE / "architecture-smoke-experiment.json"

        self.assertTrue(manifest_path.exists())
        self.assertTrue(sidecar_path.exists())
        self.assertTrue(contract_cost_manifest.exists())
        self.assertTrue(straight_line_manifest.exists())
        self.assertTrue(formal_training_manifest.exists())
        self.assertTrue(architecture_smoke_manifest.exists())
        formal_payload = json.loads(formal_training_manifest.read_text(encoding="utf-8"))
        self.assertIn("root", formal_payload["outputs"])
        self.assertIn("dataset_validation", formal_payload)
        self.assertEqual(formal_payload["train"]["seeds"], [11, 13])
        architecture_payload = json.loads(architecture_smoke_manifest.read_text(encoding="utf-8"))
        self.assertEqual(architecture_payload["train"]["seed"], 17)
        self.assertEqual(
            architecture_payload["train"]["architectures"],
            ["mlp_v1", "mlp_missing_v1", "candidate_attention_v1"],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            temp_manifest_path = Path(tmpdir) / "experiment.json"
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload["scenarios"] = [
                str((SYNTHETIC_EXPERIMENT_FIXTURE / path).resolve())
                for path in payload["scenarios"]
            ]
            payload["planner"]["sidecar_grid"] = str(
                (SYNTHETIC_EXPERIMENT_FIXTURE / payload["planner"]["sidecar_grid"]).resolve()
            )
            payload["outputs"] = {
                "rollouts": str(Path(tmpdir) / "rollouts.jsonl"),
                "evaluation": str(Path(tmpdir) / "evaluation.json"),
                "report": str(Path(tmpdir) / "report.md"),
            }
            temp_manifest_path.write_text(json.dumps(payload), encoding="utf-8")

            summary = run_experiment_manifest(temp_manifest_path)

        self.assertEqual(summary["planner"], "grid_astar")
        self.assertEqual(summary["scenario_count"], 2)
        self.assertGreater(summary["transition_count"], 0)

    def test_synthetic_benchmark_fixture_runs_with_quality_gates(self):
        from model_explorer.policy.experiment import run_experiment_manifest

        manifest_path = SYNTHETIC_EXPERIMENT_FIXTURE / "synthetic-benchmark-experiment.json"

        self.assertTrue(manifest_path.exists())

        with tempfile.TemporaryDirectory() as tmpdir:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload["splits"]["benchmark"] = {
                group_name: [
                    str((SYNTHETIC_EXPERIMENT_FIXTURE / path).resolve())
                    for path in scenario_paths
                ]
                for group_name, scenario_paths in payload["splits"]["benchmark"].items()
            }
            payload["outputs"] = {
                "rollouts": str(Path(tmpdir) / "benchmark-rollouts.jsonl"),
                "evaluation": str(Path(tmpdir) / "benchmark-evaluation.json"),
                "dataset_summary": str(Path(tmpdir) / "benchmark-dataset-summary.json"),
                "report": str(Path(tmpdir) / "benchmark-report.md"),
            }
            temp_manifest_path = Path(tmpdir) / "benchmark.json"
            temp_manifest_path.write_text(json.dumps(payload), encoding="utf-8")

            summary = run_experiment_manifest(temp_manifest_path)
            dataset_summary = json.loads(
                (Path(tmpdir) / "benchmark-dataset-summary.json").read_text(encoding="utf-8")
            )
            evaluation = json.loads((Path(tmpdir) / "benchmark-evaluation.json").read_text(encoding="utf-8"))
            report = (Path(tmpdir) / "benchmark-report.md").read_text(encoding="utf-8")

        self.assertEqual(summary["scenario_count"], 6)
        self.assertGreaterEqual(summary["dataset_summary"]["empty_action_mask_count"], 1)
        self.assertGreaterEqual(dataset_summary["trainable_transition_count"], 5)
        self.assertEqual(dataset_summary["validation_gates"]["status"], "passed")
        expected_groups = {
            "coverage_dominant",
            "risk_dominant",
            "path_cost_dominant",
            "sparse_candidates",
            "high_unreachable_rate",
            "missing_experimental_fields",
        }
        self.assertEqual(set(evaluation["groups"]), expected_groups)
        self.assertEqual(set(summary["split_summaries"]["benchmark"]["groups"]), expected_groups)
        self.assertIn("## Benchmark Groups", report)
        self.assertIn("missing_experimental_fields", report)


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

    def test_collect_rollout_script_writes_jsonl_for_multiple_scenarios(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            first_scenario_path = Path(tmpdir) / "scenario-a.json"
            second_scenario_path = Path(tmpdir) / "scenario-b.json"
            rollout_path = Path(tmpdir) / "rollouts.jsonl"
            first_scenario_path.write_text(json.dumps(minimal_contract()), encoding="utf-8")
            second_scenario_path.write_text(
                json.dumps(
                    minimal_contract(
                        goals=[{"cell": [1, 1], "utility": 0.5, "reachable": True}],
                        observation_update={"coverage_rate": 0.2, "coverage_rate_delta": 0.1},
                    )
                ),
                encoding="utf-8",
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "collect_rollout.py"),
                    str(first_scenario_path),
                    str(second_scenario_path),
                    str(rollout_path),
                ],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )

            lines = rollout_path.read_text(encoding="utf-8").splitlines()

        summary = json.loads(completed.stdout)
        self.assertEqual(summary["episode_count"], 2)
        self.assertEqual(summary["transition_count"], 2)
        self.assertEqual(len(lines), 2)

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

    def test_evaluate_baselines_script_aggregates_multiple_scenarios(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            first_scenario_path = Path(tmpdir) / "scenario-a.json"
            second_scenario_path = Path(tmpdir) / "scenario-b.json"
            first_scenario_path.write_text(
                json.dumps(
                    minimal_contract(
                        observation_update={
                            "coverage_rate": 0.1,
                            "coverage_rate_delta": 0.1,
                            "value_coverage": 0.2,
                        }
                    )
                ),
                encoding="utf-8",
            )
            second_scenario_path.write_text(
                json.dumps(
                    minimal_contract(
                        observation_update={
                            "coverage_rate": 0.3,
                            "coverage_rate_delta": 0.2,
                            "value_coverage": 0.4,
                        }
                    )
                ),
                encoding="utf-8",
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "evaluate_baselines.py"),
                    str(first_scenario_path),
                    str(second_scenario_path),
                ],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )

        report = json.loads(completed.stdout)
        self.assertEqual(report["utility"]["episode_count"], 2)
        self.assertAlmostEqual(report["utility"]["average_final_coverage_rate"], 0.2)
        self.assertAlmostEqual(report["utility"]["value_coverage"], 0.6)

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

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not available")
    def test_train_masked_policy_script_reads_jsonl_dataset(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            first_scenario_path = Path(tmpdir) / "scenario-a.json"
            second_scenario_path = Path(tmpdir) / "scenario-b.json"
            rollout_path = Path(tmpdir) / "rollouts.jsonl"
            checkpoint_path = Path(tmpdir) / "policy.pt"
            first_scenario_path.write_text(json.dumps(minimal_contract()), encoding="utf-8")
            second_scenario_path.write_text(
                json.dumps(
                    minimal_contract(
                        goals=[{"cell": [1, 1], "utility": 0.5, "reachable": True}],
                        observation_update={"coverage_rate": 0.2, "coverage_rate_delta": 0.1},
                    )
                ),
                encoding="utf-8",
            )
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "collect_rollout.py"),
                    str(first_scenario_path),
                    str(second_scenario_path),
                    str(rollout_path),
                    "--max-candidates",
                    "2",
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
        self.assertEqual(result["sample_count"], 2)
        self.assertIn("total_loss", result)

    def test_experiment_runner_manifest_writes_rollouts_and_evaluation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            first_scenario_path = Path(tmpdir) / "scenario-a.json"
            second_scenario_path = Path(tmpdir) / "scenario-b.json"
            manifest_path = Path(tmpdir) / "experiment.json"
            rollout_path = Path(tmpdir) / "rollouts.jsonl"
            evaluation_path = Path(tmpdir) / "evaluation.json"
            first_scenario_path.write_text(
                json.dumps(
                    minimal_contract(
                        goals=[{"cell": [3, 1], "utility": 0.5, "reachable": True}],
                        observation_update={"coverage_rate": 0.1, "coverage_rate_delta": 0.05},
                    )
                ),
                encoding="utf-8",
            )
            second_scenario_path.write_text(
                json.dumps(
                    minimal_contract(
                        goals=[{"cell": [1, 2], "utility": 0.4, "reachable": True}],
                        observation_update={"coverage_rate": 0.2, "coverage_rate_delta": 0.1},
                    )
                ),
                encoding="utf-8",
            )
            manifest_path.write_text(
                json.dumps(
                    {
                        "scenarios": [str(first_scenario_path), str(second_scenario_path)],
                        "planner": {"backend": "straight_line"},
                        "outputs": {
                            "rollouts": str(rollout_path),
                            "evaluation": str(evaluation_path),
                        },
                    }
                ),
                encoding="utf-8",
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "run_experiment.py"),
                    str(manifest_path),
                ],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )

            summary = json.loads(completed.stdout)
            rollout_lines = rollout_path.read_text(encoding="utf-8").splitlines()
            evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))

        self.assertEqual(summary["scenario_count"], 2)
        self.assertEqual(summary["planner"], "straight_line")
        self.assertEqual(len(rollout_lines), 2)
        self.assertIn("utility", evaluation)
        self.assertIn("coverage_heuristic", evaluation)

    def test_experiment_runner_writes_markdown_report_with_core_metrics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            scenario_path = Path(tmpdir) / "scenario.json"
            manifest_path = Path(tmpdir) / "experiment.json"
            rollout_path = Path(tmpdir) / "rollouts.jsonl"
            evaluation_path = Path(tmpdir) / "evaluation.json"
            report_path = Path(tmpdir) / "report.md"
            scenario_path.write_text(
                json.dumps(
                    minimal_contract(
                        goals=[{"cell": [2, 1], "utility": 0.4, "reachable": True, "path_cost": 2.0}],
                        observation_update={
                            "coverage_rate": 0.25,
                            "coverage_rate_delta": 0.1,
                            "value_coverage": 0.3,
                        },
                    )
                ),
                encoding="utf-8",
            )
            manifest_path.write_text(
                json.dumps(
                    {
                        "schema_version": "model-explorer-experiment/v1",
                        "scenarios": [str(scenario_path)],
                        "planner": {"backend": "contract_cost"},
                        "outputs": {
                            "rollouts": str(rollout_path),
                            "evaluation": str(evaluation_path),
                            "report": str(report_path),
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
            report = report_path.read_text(encoding="utf-8")

        self.assertEqual(summary["report_output"], str(report_path))
        for metric_name in (
            "final_coverage_rate",
            "cumulative_coverage_rate_delta",
            "total_path_cost",
            "average_risk",
            "failure_count",
            "replan_count",
            "value_coverage",
        ):
            self.assertIn(metric_name, report)
        self.assertIn("| utility |", report)
        self.assertIn("| coverage_heuristic |", report)

    def test_experiment_runner_supports_scenario_groups_and_per_scenario_metrics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            first_scenario_path = Path(tmpdir) / "scenario-a.json"
            second_scenario_path = Path(tmpdir) / "scenario-b.json"
            manifest_path = Path(tmpdir) / "experiment.json"
            evaluation_path = Path(tmpdir) / "evaluation.json"
            first_scenario_path.write_text(
                json.dumps(
                    minimal_contract(
                        goals=[{"cell": [1, 1], "utility": 0.5, "reachable": True}],
                        observation_update={"coverage_rate": 0.1, "coverage_rate_delta": 0.1},
                    )
                ),
                encoding="utf-8",
            )
            second_scenario_path.write_text(
                json.dumps(
                    minimal_contract(
                        goals=[{"cell": [3, 2], "utility": 0.4, "reachable": True}],
                        observation_update={"coverage_rate": 0.3, "coverage_rate_delta": 0.2},
                    )
                ),
                encoding="utf-8",
            )
            manifest_path.write_text(
                json.dumps(
                    {
                        "scenario_groups": [
                            {"name": "alpha", "scenarios": [str(first_scenario_path)]},
                            {"name": "beta", "scenarios": [str(second_scenario_path)]},
                        ],
                        "planner": {"backend": "straight_line"},
                        "outputs": {
                            "rollouts": str(Path(tmpdir) / "rollouts.jsonl"),
                            "evaluation": str(evaluation_path),
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

        self.assertEqual(summary["scenario_count"], 2)
        self.assertEqual(summary["group_count"], 2)
        self.assertEqual([item["name"] for item in summary["groups"]], ["alpha", "beta"])
        self.assertIn("aggregate", evaluation)
        self.assertEqual(len(evaluation["per_scenario"]), 2)
        self.assertIn("alpha", evaluation["groups"])
        self.assertIn("beta", evaluation["groups"])

    def test_experiment_runner_outputs_reward_ablations(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            scenario_path = Path(tmpdir) / "scenario.json"
            manifest_path = Path(tmpdir) / "experiment.json"
            scenario_path.write_text(
                json.dumps(
                    minimal_contract(
                        goals=[{"cell": [3, 1], "utility": 0.5, "reachable": True}],
                        observation_update={"coverage_rate": 1.0, "coverage_rate_delta": 1.0},
                    )
                ),
                encoding="utf-8",
            )
            manifest_path.write_text(
                json.dumps(
                    {
                        "scenarios": [str(scenario_path)],
                        "planner": {"backend": "straight_line"},
                        "reward_ablations": [
                            {"name": "default", "reward": {}},
                            {
                                "name": "high_path_penalty",
                                "reward": {"path_cost_weight": 1.0, "path_cost_normalizer": 1.0},
                            },
                        ],
                        "outputs": {
                            "rollouts": str(Path(tmpdir) / "rollouts.jsonl"),
                            "evaluation": str(Path(tmpdir) / "evaluation.json"),
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
        default_reward = summary["reward_ablations"]["default"]["rollout_metrics"]["total_reward"]
        high_penalty_reward = summary["reward_ablations"]["high_path_penalty"]["rollout_metrics"]["total_reward"]
        self.assertLess(high_penalty_reward, default_reward)

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not available")
    def test_experiment_runner_train_block_writes_checkpoint_and_loss_log(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            scenario_path = Path(tmpdir) / "scenario-a.json"
            second_scenario_path = Path(tmpdir) / "scenario-b.json"
            manifest_path = Path(tmpdir) / "experiment.json"
            checkpoint_path = Path(tmpdir) / "policy.pt"
            loss_log_path = Path(tmpdir) / "losses.jsonl"
            scenario_payload = minimal_contract(
                goals=[
                    {"cell": [1, 1], "utility": 0.5, "reachable": True},
                    {"cell": [2, 1], "utility": 10.0, "reachable": False},
                ],
                observation_update={"coverage_rate": 0.1, "coverage_rate_delta": 0.1},
            )
            scenario_path.write_text(json.dumps(scenario_payload), encoding="utf-8")
            second_scenario_path.write_text(json.dumps(scenario_payload), encoding="utf-8")
            manifest_path.write_text(
                json.dumps(
                    {
                        "scenarios": [str(scenario_path), str(second_scenario_path)],
                        "max_candidates": 2,
                        "planner": {"backend": "contract_cost"},
                        "outputs": {
                            "rollouts": str(Path(tmpdir) / "rollouts.jsonl"),
                            "evaluation": str(Path(tmpdir) / "evaluation.json"),
                        },
                        "train": {
                            "checkpoint": str(checkpoint_path),
                            "loss_log": str(loss_log_path),
                            "seed": 19,
                            "hidden_size": 16,
                            "epochs": 1,
                            "validation_fraction": 0.5,
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
            checkpoint_exists = checkpoint_path.exists()
            loss_lines = loss_log_path.read_text(encoding="utf-8").splitlines()
            from model_explorer.policy.training import load_policy_checkpoint

            scorer = load_policy_checkpoint(checkpoint_path)
            decision = select_goal(load_contract_from_dict(scenario_payload), policy=scorer)

        self.assertTrue(checkpoint_exists)
        self.assertEqual(summary["training"]["sample_count"], 1)
        self.assertEqual(summary["training"]["train_episode_count"], 1)
        self.assertEqual(summary["training"]["validation_episode_count"], 1)
        self.assertEqual(len(loss_lines), 1)
        self.assertIsNotNone(decision.selected_goal)
        self.assertNotEqual(decision.selected_goal.cell, (2, 1))


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


def _route_fixture(scenario_index, *, action_index):
    base = {
        "schema_version": "path-planner-route/v1",
        "geometric_path": {
            "cells": [[0, 0], [action_index + 1, scenario_index + 1]],
            "world": [[0.0, 0.0], [float(action_index + 1), float(scenario_index + 1)]],
        },
        "diagnostics": {
            "path_length_m": float(action_index + scenario_index + 1),
            "search_mode": "platform_aware_astar",
        },
    }
    if scenario_index == 0 and action_index == 0:
        return {
            **base,
            "reachable": False,
            "path_cost": None,
            "failure_reason": "goal_blocked",
        }
    route = {
        **base,
        "reachable": True,
        "path_cost": float(8 - action_index - scenario_index),
        "failure_reason": None,
        "postprocess": {"fallback_status": "ok", "tracking_safety_report": {"violation_count": 0}},
    }
    route.update(_convex_region_route_fields("fallback_box", fallback_used=True))
    control_point_route = scenario_index == 0 and action_index == 1
    route.update(
        _gcs_trajectory_route_fields(
            success=not (scenario_index == 2 and action_index == 0),
            collision_count=1 if scenario_index == 2 and action_index == 0 else 0,
            backend=(
                "pydrake_control_point_direction_cone_program"
                if control_point_route
                else "pydrake_gcs"
            ),
            reason=(
                "control_point_direction_cone_solution_found"
                if control_point_route
                else None
            ),
            cost_summary=_control_point_cost_summary() if control_point_route else None,
        )
    )
    if scenario_index == 2 and action_index == 0:
        route.update(
            _gcs_motion_feasibility_route_fields(
                evaluated=False,
                feasibility_status="diagnostic_only",
                fallback_reason="gcs_trajectory_failed",
            )
        )
    elif scenario_index == 1 and action_index == 0:
        route.update(
            _gcs_motion_feasibility_route_fields(
                evaluated=True,
                feasibility_status="infeasible",
                fallback_reason="motion_constraint_violation",
                curvature_violation_count=1,
                heading_violation_count=1,
                violation_indices=[1],
            )
        )
    else:
        route.update(
            _gcs_motion_feasibility_route_fields(
                evaluated=True,
                feasibility_status="feasible",
                fallback_reason=None,
            )
        )
    if scenario_index == 2 and action_index == 0:
        route.update(
            _gcs_curvature_constrained_candidate_route_fields(
                available=False,
                selected=False,
                repair_success=False,
                repair_strategy="not_attempted",
                status_before="diagnostic_only",
                status_after="diagnostic_only",
                fallback_reason="gcs_trajectory_failed",
            )
        )
    elif scenario_index == 1 and action_index == 0:
        route.update(
            _gcs_curvature_constrained_candidate_route_fields(
                available=True,
                selected=True,
                repair_success=True,
                repair_strategy="moving_average_smoothing",
                status_before="infeasible",
                status_after="feasible",
                fallback_reason=None,
                curvature_violation_count_before=1,
                heading_violation_count_before=1,
                violation_indices_before=[1],
            )
        )
    else:
        route.update(
            _gcs_curvature_constrained_candidate_route_fields(
                available=True,
                selected=True,
                repair_success=False,
                repair_strategy="none_required",
                status_before="feasible",
                status_after="feasible",
                fallback_reason=None,
            )
        )
    if scenario_index == 2 and action_index == 0:
        route.update(
            _gcs_candidate_route_fields(
                available=False,
                selected=False,
                selection_reason=None,
                fallback_reason="sampled_trajectory_collision",
                collision_count=1,
                cost_delta=None,
                overlap_ratio=None,
            )
        )
    elif scenario_index == 1 and action_index == 0:
        route.update(
            _gcs_candidate_route_fields(
                available=True,
                selected=False,
                selection_reason=None,
                fallback_reason="cost_dominated",
                collision_count=0,
                cost_delta=2.0,
                overlap_ratio=0.25,
            )
        )
    elif scenario_index == 2 and action_index == 1:
        route.update(
            _gcs_candidate_route_fields(
                available=True,
                selected=False,
                selection_reason=None,
                fallback_reason="path_duplicate_with_baseline",
                collision_count=0,
                cost_delta=0.0,
                overlap_ratio=1.0,
            )
        )
    else:
        route.update(
            _gcs_candidate_route_fields(
                available=True,
                selected=True,
                selection_reason="gcs_candidate_quality_improved",
                fallback_reason=None,
                collision_count=0,
                cost_delta=-1.0,
                overlap_ratio=0.25,
                cost_summary=_control_point_candidate_cost_summary() if control_point_route else None,
            )
        )
    if scenario_index == 2 and action_index == 0:
        route["planning_backend_report"] = {
            "requested_backend": "region_graph_guided",
            "selected_backend": "astar",
            "status": "fallback",
            "fallback_reason": "target_component_disconnected",
            "sampled_region_path_report": {
                "schema_version": "sampled_region_path_report/v1",
                "status": "fallback",
                "fallback_reason": "target_component_disconnected",
                "region_sequence": [0, 1],
                "sample_count": 2,
                "start_goal_anchoring": {
                    "start_region_id": 0,
                    "goal_region_id": 1,
                    "start_region_candidates": [0],
                    "goal_region_candidates": [1],
                    "region_sequence_found": True,
                    "start_classification": "covered",
                    "goal_classification": "covered",
                    "start_anchor_region_added": False,
                    "goal_anchor_region_added": False,
                    "start_anchor_region_connected": False,
                    "goal_anchor_region_connected": False,
                    "start_anchor_failure_reason": None,
                    "goal_anchor_failure_reason": None,
                    "reachable_component_report": {
                        "schema_version": "reachable_component_report/v1",
                        "status": "disconnected",
                        "reason": "target_component_disconnected",
                        "passable_source": "inflated_passable_mask",
                        "component_count": 2,
                        "start_component_id": 0,
                        "raw_goal_component_id": 1,
                        "adjusted_goal_component_id": 1,
                        "start_goal_same_component": False,
                        "adjusted_goal_start_component": False,
                    },
                    "anchor_connectivity_closure": {
                        "schema_version": "anchor-connectivity-closure/v1",
                        "attempt_count": 1,
                        "connected_count": 0,
                        "status_counts": {"unavailable": 1},
                        "reason_counts": {"safe_bridge_path_unavailable": 1},
                        "connection_kind_counts": {},
                    },
                },
                "sample_attempt_count": 3,
                "sample_attempts": [
                    {"kind": "region_sample", "region_id": 0, "cell": [0, 0], "strategy": "preferred"},
                    {"kind": "region_sample", "region_id": 1, "cell": [1, 1], "strategy": "preferred"},
                    {
                        "kind": "edge_transition",
                        "from_region_id": 0,
                        "to_region_id": 1,
                        "status": "unavailable",
                    },
                    {
                        "kind": "connector_attempt",
                        "strategy": "cost_aware_constrained_astar",
                        "status": "unavailable",
                    },
                ],
                "candidate_rankings": [
                    {
                        "rank": None,
                        "strategy": "reachable_component_filter",
                        "status": "rejected",
                        "fallback_reason": "target_component_disconnected",
                        "sample_count": 0,
                    }
                ],
                "safety_checks": {"collision_free": False},
                "candidate_comparison": {
                    "candidate_cost_delta": None,
                    "baseline_path_overlap_ratio": None,
                    "path_duplicate_with_baseline": None,
                    "benefit_surface_present": False,
                    "complexity_reason": "candidate_missing_metrics",
                },
                "terminal_adjustment_report": {
                    "schema_version": "terminal_adjustment_report/v1",
                    "status": "not_required",
                    "reason_code": "terminal_adjustment_not_required",
                    "target_adjusted": False,
                    "candidate_count": 0,
                    "reachable_candidate_count": 0,
                    "reachable_component_replacement_selected": False,
                    "reachable_component_report": {
                        "schema_version": "reachable_component_report/v1",
                        "status": "disconnected",
                        "reason": "target_component_disconnected",
                        "passable_source": "inflated_passable_mask",
                        "component_count": 2,
                        "start_component_id": 0,
                        "raw_goal_component_id": 1,
                        "adjusted_goal_component_id": 1,
                        "start_goal_same_component": False,
                        "adjusted_goal_start_component": False,
                    },
                },
                "execution_tie_break": {
                    "status": "fallback",
                    "reason": "execution_tie_break_no_alternative",
                },
            },
        }
        route["region_graph_report"] = {
            "status": "ok",
            "region_source": "grid_box",
            "fallback_used": True,
            "quality_metrics": {
                "requested_region_source": "iris",
                "graph_source": "grid_box",
                "fallback_ratio": 1.0,
                "connected_component_count": 2,
                "start_goal_connected": False,
                "fallback_reason": "iris_region_graph_fallback: start_goal_not_connected",
            },
        }
    if scenario_index == 1 and action_index == 1:
        route["planning_backend_report"] = {
            "requested_backend": "region_graph_guided",
            "selected_backend": "sampled_region_path",
            "status": "selected",
            "fallback_reason": None,
            "sampled_region_path_report": {
                "schema_version": "sampled_region_path_report/v1",
                "status": "selected",
                "fallback_reason": None,
                "region_sequence": [0, 1],
                "sample_count": 2,
                "start_goal_anchoring": {
                    "start_region_id": 0,
                    "goal_region_id": 1,
                    "start_region_candidates": [0],
                    "goal_region_candidates": [1],
                    "region_sequence_found": True,
                    "start_classification": "covered",
                    "goal_classification": "goal_outside_region_coverage",
                    "start_anchor_region_added": False,
                    "goal_anchor_region_added": True,
                    "start_anchor_region_connected": False,
                    "goal_anchor_region_connected": True,
                    "start_anchor_failure_reason": None,
                    "goal_anchor_failure_reason": None,
                    "reachable_component_report": {
                        "schema_version": "reachable_component_report/v1",
                        "status": "adjusted_connected",
                        "reason": "reachable_component_replacement_selected",
                        "passable_source": "inflated_passable_mask",
                        "component_count": 2,
                        "start_component_id": 0,
                        "raw_goal_component_id": None,
                        "adjusted_goal_component_id": 0,
                        "start_goal_same_component": None,
                        "adjusted_goal_start_component": True,
                    },
                    "anchor_connectivity_closure": {
                        "schema_version": "anchor-connectivity-closure/v1",
                        "attempt_count": 2,
                        "connected_count": 1,
                        "status_counts": {"connected": 1, "unavailable": 1},
                        "reason_counts": {
                            "safe_bridge_found": 1,
                            "safe_bridge_path_unavailable": 1,
                        },
                        "connection_kind_counts": {"anchor_region_safe_bridge": 1},
                    },
                },
                "sample_attempt_count": 7,
                "sample_attempts": [
                    {"kind": "region_sample", "region_id": 0, "cell": [0, 0], "strategy": "preferred"},
                    {"kind": "region_sample", "region_id": 0, "cell": [0, 1], "strategy": "low_cost"},
                    {"kind": "region_sample", "region_id": 1, "cell": [1, 1], "strategy": "preferred"},
                    {"kind": "edge_transition", "from_region_id": 0, "to_region_id": 1, "status": "available"},
                    {
                        "kind": "connector_attempt",
                        "strategy": "bridge_aware_constrained_astar",
                        "status": "unavailable",
                        "fallback_reason": "bridge_aware_connector_path_unavailable",
                        "bridge_aware": True,
                        "bridge_connection_count": 1,
                        "bridge_cell_count": 3,
                        "bridge_mask_added_cell_count": 1,
                        "bridge_cells": [[1, 1], [2, 1], [2, 2]],
                    },
                    {
                        "kind": "connector_attempt",
                        "strategy": "bridge_corridor_constrained_astar",
                        "status": "available",
                        "bridge_corridor_expanded": True,
                        "bridge_corridor_radius_cells": 1,
                        "bridge_corridor_added_cell_count": 4,
                        "bridge_corridor_start_goal_connected": True,
                        "bridge_corridor_failure_reason": None,
                    },
                    {
                        "kind": "connector_attempt",
                        "strategy": "cost_aware_constrained_astar",
                        "status": "available",
                    },
                ],
                "candidate_rankings": [
                    {
                        "strategy": "bridge_aware_constrained_astar",
                        "status": "rejected",
                        "fallback_reason": "bridge_aware_connector_path_unavailable",
                        "sample_count": 0,
                        "bridge_aware": True,
                        "bridge_connection_count": 1,
                        "bridge_cell_count": 3,
                        "bridge_mask_added_cell_count": 1,
                        "bridge_cells": [[1, 1], [2, 1], [2, 2]],
                    },
                    {
                        "rank": 1,
                        "strategy": "bridge_corridor_constrained_astar",
                        "status": "selected",
                        "fallback_reason": None,
                        "selection_reason": "execution_tie_break_improved",
                        "execution_tie_break_reason": "execution_tie_break_improved",
                        "sample_count": 2,
                        "candidate_cost_delta": -0.5,
                        "bridge_aware": True,
                        "bridge_connection_count": 1,
                        "bridge_cell_count": 3,
                        "bridge_mask_added_cell_count": 1,
                        "bridge_cells": [[1, 1], [2, 1], [2, 2]],
                        "bridge_corridor_expanded": True,
                        "bridge_corridor_radius_cells": 1,
                        "bridge_corridor_added_cell_count": 4,
                        "bridge_corridor_start_goal_connected": True,
                        "bridge_corridor_failure_reason": None,
                    }
                ],
                "safety_checks": {"collision_free": True},
                "candidate_comparison": {
                    "candidate_cost_delta": -0.5,
                    "baseline_path_overlap_ratio": 0.5,
                    "path_duplicate_with_baseline": False,
                    "benefit_surface_present": True,
                    "complexity_reason": "sampled_candidate_has_quality_gain",
                },
                "terminal_adjustment_report": {
                    "schema_version": "terminal_adjustment_report/v1",
                    "status": "selected",
                    "reason_code": "reachable_terminal_selected_by_component_projection",
                    "target_adjusted": True,
                    "original_goal_cell": [3, 2],
                    "adjusted_goal_cell": [2, 2],
                    "candidate_count": 3,
                    "reachable_candidate_count": 1,
                    "rescue_candidate_count": 2,
                    "reachable_terminal_rescue_used": True,
                    "proxy_goal_anchor_selected": False,
                    "reachable_component_replacement_selected": True,
                    "reachable_component_report": {
                        "schema_version": "reachable_component_report/v1",
                        "status": "adjusted_connected",
                        "reason": "reachable_component_replacement_selected",
                        "passable_source": "inflated_passable_mask",
                        "component_count": 2,
                        "start_component_id": 0,
                        "raw_goal_component_id": None,
                        "adjusted_goal_component_id": 0,
                        "start_goal_same_component": None,
                        "adjusted_goal_start_component": True,
                    },
                },
                "execution_tie_break": {
                    "status": "selected",
                    "reason": "execution_tie_break_improved",
                },
            },
        }
        route["iris_region_report"] = {
            "backend": "workspace_iris",
            "status": "ok",
            "region_count": 2,
            "fallback_used": False,
            "failure_status": "none",
            "failure_reason": None,
        }
        route.update(_convex_region_route_fields("workspace_iris", fallback_used=False))
        route.update(_gcs_trajectory_route_fields(success=True, collision_count=0))
        route.update(
            _gcs_motion_feasibility_route_fields(
                evaluated=True,
                feasibility_status="feasible",
                fallback_reason=None,
            )
        )
        route.update(
            _gcs_candidate_route_fields(
                available=True,
                selected=True,
                selection_reason="gcs_candidate_quality_improved",
                fallback_reason=None,
                collision_count=0,
                cost_delta=-1.0,
                overlap_ratio=0.25,
            )
        )
        route["region_graph_report"] = {
            "status": "ok",
            "region_source": "iris",
            "fallback_used": False,
            "quality_metrics": {
                "requested_region_source": "iris",
                "graph_source": "iris",
                "fallback_ratio": 0.0,
                "connected_component_count": 1,
                "start_goal_connected": True,
                "fallback_reason": None,
            },
        }
    return route


def _convex_region_route_fields(backend, *, fallback_used):
    source = "iris" if backend == "workspace_iris" else "fallback_box"
    sequence = [
        {
            "id": 0,
            "backend": backend,
            "source": source,
            "seed_cell": [0, 0],
            "seed_world": [0.5, 0.5],
            "bounds": {"min": [0, 0], "max": [1, 1]},
            "world_bounds": {"min": [0.0, 0.0], "max": [2.0, 2.0]},
            "hpolyhedron": {
                "A": [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]],
                "b": [2.0, -0.0, 2.0, -0.0],
            },
            "covered_path_indices": [0],
            "validation_status": "valid",
            "fallback_reason": "fallback_box_not_drake_iris" if fallback_used else None,
        },
        {
            "id": 1,
            "backend": backend,
            "source": source,
            "seed_cell": [1, 1],
            "seed_world": [1.5, 1.5],
            "bounds": {"min": [1, 1], "max": [2, 2]},
            "world_bounds": {"min": [1.0, 1.0], "max": [3.0, 3.0]},
            "hpolyhedron": {
                "A": [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]],
                "b": [3.0, -1.0, 3.0, -1.0],
            },
            "covered_path_indices": [1],
            "validation_status": "valid",
            "fallback_reason": "fallback_box_not_drake_iris" if fallback_used else None,
        },
    ]
    return {
        "convex_region_sequence_schema_version": "convex_region_sequence_report/v1",
        "convex_region_sequence": sequence,
        "convex_region_count": len(sequence),
        "convex_region_backend": backend,
        "convex_region_fallback_used": fallback_used,
        "convex_region_coverage_status": "covered",
        "convex_region_start_contained": True,
        "convex_region_goal_contained": True,
        "convex_region_adjacent_overlap_count": 1,
        "convex_region_portal_count": 0,
        "convex_region_blocked_cell_violation_count": 0,
        "convex_region_pydrake_available": True,
        "gcs_ready": True,
        "gcs_ready_reason": "convex_region_sequence_ready",
    }


def _gcs_trajectory_route_fields(
    *,
    success,
    collision_count,
    backend="pydrake_gcs",
    reason=None,
    cost_summary=None,
):
    reason = reason or ("gcs_trajectory_solution_found" if success else "sampled_trajectory_collision")
    fields = {
        "gcs_trajectory_report_schema_version": "gcs_trajectory_report/v1",
        "gcs_trajectory_attempted": True,
        "gcs_trajectory_success": success,
        "gcs_trajectory_backend": backend,
        "gcs_trajectory_result_status": (
            "SolutionResult.kSolutionFound" if success else "sampled_trajectory_collision"
        ),
        "gcs_trajectory_reason": reason,
        "gcs_trajectory_sample_count": 5,
        "gcs_trajectory_collision_count": collision_count,
        "gcs_trajectory_path_length": 2.0,
        "gcs_trajectory_region_count": 2,
        "gcs_trajectory_sampled_points": [[0.5, 0.5], [1.5, 1.5]],
        "gcs_trajectory_constraint_summary": {
            "schema_version": "gcs_direction_cone_constraint/v1",
            "constraint_model": "direction_cone",
            "attempted": True,
            "evaluated": True,
            "backend_enforced": True,
            "violation_count": 0,
            "eta": 1.0,
            "rho_min": 0.025,
            "max_allowed_direction_error_deg": 45.0,
            "constraint_tightness_min": 1.0,
            "risk_flags": [],
            "rho_source_counts": {"seed_distance_portal_support_min": 1},
            "objective_term_weights": {
                "segment_length_quadratic": 1.0,
                "low_cost_anchor_quadratic": 0.1,
                "control_point_terrain_anchor_quadratic": 0.05,
                "control_point_second_difference_quadratic": 0.2,
            },
        },
    }
    if cost_summary is not None:
        fields["gcs_trajectory_cost_summary"] = cost_summary
    return fields


def _gcs_candidate_route_fields(
    *,
    available,
    selected,
    selection_reason,
    fallback_reason,
    collision_count,
    cost_delta,
    overlap_ratio,
    cost_summary=None,
):
    fields = {
        "gcs_candidate_report_schema_version": "gcs_geometric_candidate_report/v1",
        "gcs_candidate_attempted": True,
        "gcs_candidate_available": available,
        "gcs_candidate_selected": selected,
        "gcs_candidate_selection_reason": selection_reason,
        "gcs_candidate_fallback_reason": fallback_reason,
        "gcs_candidate_path_length": 2.0 if available else 0.0,
        "gcs_candidate_path_cost": None if cost_delta is None else 5.0 + cost_delta,
        "gcs_candidate_collision_count": collision_count,
        "gcs_candidate_high_cost_exposure": 0.0,
        "gcs_candidate_baseline_overlap_ratio": overlap_ratio,
        "gcs_candidate_cost_delta_vs_baseline": cost_delta,
        "gcs_candidate_cost_delta_vs_postprocess": cost_delta,
    }
    if cost_summary is not None:
        fields["gcs_candidate_cost_summary"] = cost_summary
    return fields


def _control_point_cost_summary():
    return {
        "high_cost_exposure": 0.0,
        "terrain_path_cost": 6.0,
        "sampled_terrain_cost": 6.0,
        "terrain_objective_source": "region_inverse_cost_weighted_passable_cell_centroid",
        "terrain_objective_weight": 0.05,
        "terrain_objective_boundary": "proxy_not_continuous_field_integral",
        "control_point_terrain_cost": 6.0,
    }


def _control_point_candidate_cost_summary():
    summary = dict(_control_point_cost_summary())
    summary.update(
        {
            "baseline_high_cost_exposure": 1.0,
            "postprocess_high_cost_exposure": 1.0,
            "high_cost_exposure_delta_vs_baseline": -1.0,
            "high_cost_exposure_delta_vs_postprocess": -1.0,
        }
    )
    return summary


def _gcs_motion_feasibility_route_fields(
    *,
    evaluated,
    feasibility_status,
    fallback_reason,
    curvature_violation_count=0,
    heading_violation_count=0,
    violation_indices=None,
):
    return {
        "gcs_motion_feasibility_report_schema_version": "gcs_motion_feasibility_report/v1",
        "gcs_motion_feasibility_evaluated": evaluated,
        "gcs_motion_feasibility_trajectory_source": "gcs_trajectory_sampled_points",
        "gcs_motion_feasibility_motion_model": "curvature_bounded",
        "gcs_motion_feasibility_feasibility_status": feasibility_status,
        "gcs_motion_feasibility_fallback_reason": fallback_reason,
        "gcs_motion_feasibility_min_turning_radius_m": 0.5,
        "gcs_motion_feasibility_max_heading_change_deg": 120.0,
        "gcs_motion_feasibility_curvature_violation_count": curvature_violation_count,
        "gcs_motion_feasibility_heading_violation_count": heading_violation_count,
        "gcs_motion_feasibility_violation_indices": list(violation_indices or []),
        "gcs_motion_feasibility_sample_count": 5 if evaluated else 0,
        "gcs_motion_feasibility_path_length": 2.0 if evaluated else 0.0,
        "gcs_motion_feasibility_constraint_summary": {
            "motion_model": "curvature_bounded",
            "max_curvature": 2.0,
            "min_turning_radius_m": 0.5,
            "max_heading_change_deg": 120.0,
            "max_observed_curvature": 1.0 if curvature_violation_count else 0.0,
            "max_observed_heading_change_deg": 90.0 if heading_violation_count else 0.0,
        },
    }


def _gcs_curvature_constrained_candidate_route_fields(
    *,
    available,
    selected,
    repair_success,
    repair_strategy,
    status_before,
    status_after,
    fallback_reason,
    curvature_violation_count_before=0,
    curvature_violation_count_after=0,
    heading_violation_count_before=0,
    heading_violation_count_after=0,
    violation_indices_before=None,
    violation_indices_after=None,
    collision_count=0,
    region_containment_violation_count=0,
):
    return {
        "gcs_curvature_constrained_report_schema_version": (
            "gcs_curvature_constrained_candidate_report/v1"
        ),
        "gcs_curvature_constrained_attempted": True,
        "gcs_curvature_constrained_available": available,
        "gcs_curvature_constrained_selected": selected,
        "gcs_curvature_constrained_repair_success": repair_success,
        "gcs_curvature_constrained_source": "gcs_trajectory_sampled_points",
        "gcs_curvature_constrained_repair_strategy": repair_strategy,
        "gcs_curvature_constrained_status_before": status_before,
        "gcs_curvature_constrained_status_after": status_after,
        "gcs_curvature_constrained_fallback_reason": fallback_reason,
        "gcs_curvature_constrained_curvature_violation_count_before": curvature_violation_count_before,
        "gcs_curvature_constrained_curvature_violation_count_after": curvature_violation_count_after,
        "gcs_curvature_constrained_heading_violation_count_before": heading_violation_count_before,
        "gcs_curvature_constrained_heading_violation_count_after": heading_violation_count_after,
        "gcs_curvature_constrained_violation_indices_before": list(violation_indices_before or []),
        "gcs_curvature_constrained_violation_indices_after": list(violation_indices_after or []),
        "gcs_curvature_constrained_region_containment_violation_count": (
            region_containment_violation_count
        ),
        "gcs_curvature_constrained_collision_count": collision_count,
        "gcs_curvature_constrained_path_length": 2.0 if available else 0.0,
        "gcs_curvature_constrained_path_cost": 5.0 if available else None,
        "gcs_curvature_constrained_cost_delta_vs_baseline": -1.0 if selected else None,
        "gcs_curvature_constrained_constraint_summary": {
            "max_curvature": 2.0,
            "min_turning_radius_m": 0.5,
            "max_heading_change_deg": 120.0,
            "repair_passes": 4 if repair_strategy == "moving_average_smoothing" else 0,
        },
    }


if __name__ == "__main__":
    unittest.main()
