import json
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

from model_explorer.core.interfaces import ContractValidationError
from model_explorer.decision.selector import select_goal
from model_explorer.io.scenario import load_scenario
from model_explorer.orchestration.loop import run_exploration_loop


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
