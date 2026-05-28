import json
import io
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def minimal_contract(*, goals=None, observation_update=None):
    return {
        "schema_version": "model-explorer-contract/v1",
        "grid": {
            "width": 6,
            "height": 5,
            "resolution": 0.5,
            "frame_id": "moon_local",
            "origin": [0.0, 0.0],
            "layers": ["confidence", "cost"],
        },
        "constraints": {
            "violation_count": 0,
            "passable_ratio": 0.9,
            "reason_counts": {"obstacle": 0},
        },
        "top_goals": goals
        if goals is not None
        else [
            {
                "cell": [2, 1],
                "utility": 0.5,
                "reachable": True,
                "expected_coverage_rate_delta": 0.2,
                "risk": 0.1,
                "path_cost": 1.5,
            }
        ],
        "top_sequences": [{"cells": [[2, 1]], "utility": 0.5, "coverage_area": 1.0}],
        "observation_update": observation_update
        if observation_update is not None
        else {"coverage_rate": 0.2, "coverage_rate_delta": 0.2},
    }


def run_module(args, *, check=True):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC)
    return subprocess.run(
        [sys.executable, "-m", "model_explorer", *args],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=check,
    )


class ExperimentCliEntrypointTests(unittest.TestCase):
    def test_experiment_validate_dry_run_and_run_emit_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            scenario_path = root / "scenario.json"
            manifest_path = root / "experiment.json"
            output_root = root / "out"
            checkpoint_path = root / "policy.pt"
            scenario_path.write_text(json.dumps(minimal_contract()), encoding="utf-8")
            manifest_path.write_text(
                json.dumps(
                    {
                        "name": "cli-smoke",
                        "run_id": "run-001",
                        "scenarios": [str(scenario_path)],
                        "outputs": {"root": str(output_root)},
                        "train": {
                            "checkpoint": str(checkpoint_path),
                            "epochs": 1,
                            "hidden_size": 16,
                        },
                    }
                ),
                encoding="utf-8",
            )

            validate = run_module(["experiment", "validate", str(manifest_path)])
            dry_run = run_module(["experiment", "dry-run", str(manifest_path)])

            self.assertFalse((output_root / "cli-smoke" / "run-001" / "rollouts.jsonl").exists())
            self.assertFalse(checkpoint_path.exists())

            manifest_path.write_text(
                json.dumps(
                    {
                        "name": "cli-smoke",
                        "run_id": "run-001",
                        "scenarios": [str(scenario_path)],
                        "outputs": {"root": str(output_root)},
                    }
                ),
                encoding="utf-8",
            )
            run = run_module(["experiment", "run", str(manifest_path)])
            run_summary = json.loads(run.stdout)

        validate_summary = json.loads(validate.stdout)
        dry_run_summary = json.loads(dry_run.stdout)
        self.assertEqual(validate_summary["status"], "valid")
        self.assertEqual(validate_summary["scenario_count"], 1)
        self.assertEqual(dry_run_summary["status"], "dry_run")
        self.assertIn("rollouts.jsonl", "\n".join(dry_run_summary["would_write"]))
        self.assertEqual(run_summary["scenario_count"], 1)
        self.assertGreater(run_summary["transition_count"], 0)

    def test_experiment_validate_errors_are_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "bad.json"
            manifest_path.write_text(json.dumps({"outputs": {"root": "out"}}), encoding="utf-8")

            completed = run_module(["experiment", "validate", str(manifest_path)], check=False)

        self.assertNotEqual(completed.returncode, 0)
        error = json.loads(completed.stderr)
        self.assertEqual(error["status"], "error")
        self.assertIn("scenarios", error["message"])


class SyntheticBenchmarkGeneratorTests(unittest.TestCase):
    def test_generator_is_reproducible_and_output_runs(self):
        from model_explorer.policy.benchmark import BENCHMARK_GROUPS, generate_synthetic_benchmark_suite
        from model_explorer.policy.experiment import run_experiment_manifest

        with tempfile.TemporaryDirectory() as first_tmp, tempfile.TemporaryDirectory() as second_tmp:
            first = generate_synthetic_benchmark_suite(
                first_tmp,
                seed=123,
                scenario_count=1,
                difficulty="medium",
            )
            second = generate_synthetic_benchmark_suite(
                second_tmp,
                seed=123,
                scenario_count=1,
                difficulty="medium",
            )

            self.assertEqual(_json_snapshot(first_tmp), _json_snapshot(second_tmp))
            summary = run_experiment_manifest(first["manifest"])

        self.assertEqual(tuple(first["groups"]), BENCHMARK_GROUPS)
        self.assertEqual(summary["scenario_count"], len(BENCHMARK_GROUPS))
        self.assertGreater(summary["transition_count"], 0)

    def test_generator_returns_stable_golden_summary_for_fixed_seed(self):
        from model_explorer.policy.benchmark import generate_synthetic_benchmark_suite

        with tempfile.TemporaryDirectory() as tmpdir:
            generated = generate_synthetic_benchmark_suite(
                tmpdir,
                seed=123,
                scenario_count=1,
                difficulty="medium",
            )

        self.assertEqual(
            generated["summary"],
            {
                "schema_version": "model-explorer-synthetic-benchmark-summary/v1",
                "seed": 123,
                "difficulty": "medium",
                "group_count": 6,
                "scenario_count": 6,
                "max_candidates": 5,
                "groups": {
                    "coverage_dominant": {
                        "scenario_count": 1,
                        "candidate_count": 5,
                        "reachable_candidate_count": 3,
                        "missing_experimental_fields": False,
                    },
                    "risk_dominant": {
                        "scenario_count": 1,
                        "candidate_count": 5,
                        "reachable_candidate_count": 3,
                        "missing_experimental_fields": False,
                    },
                    "path_cost_dominant": {
                        "scenario_count": 1,
                        "candidate_count": 5,
                        "reachable_candidate_count": 3,
                        "missing_experimental_fields": False,
                    },
                    "sparse_candidates": {
                        "scenario_count": 1,
                        "candidate_count": 2,
                        "reachable_candidate_count": 2,
                        "missing_experimental_fields": False,
                    },
                    "high_unreachable_rate": {
                        "scenario_count": 1,
                        "candidate_count": 5,
                        "reachable_candidate_count": 0,
                        "missing_experimental_fields": False,
                    },
                    "missing_experimental_fields": {
                        "scenario_count": 1,
                        "candidate_count": 5,
                        "reachable_candidate_count": 4,
                        "missing_experimental_fields": True,
                    },
                },
            },
        )

    def test_benchmark_generate_cli_writes_manifest_and_scenarios(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            completed = run_module(
                [
                    "benchmark",
                    "generate",
                    tmpdir,
                    "--seed",
                    "7",
                    "--scenario-count",
                    "1",
                    "--group",
                    "coverage_dominant",
                    "--difficulty",
                    "easy",
                ]
            )
            summary = json.loads(completed.stdout)
            manifest_path = Path(summary["manifest"])
            manifest_exists = manifest_path.exists()
            dry_run = run_module(["experiment", "dry-run", str(manifest_path)])

        self.assertEqual(summary["status"], "generated")
        self.assertEqual(summary["groups"], ["coverage_dominant"])
        self.assertTrue(manifest_exists)
        self.assertEqual(summary["summary"]["group_count"], 1)
        self.assertEqual(json.loads(dry_run.stdout)["status"], "dry_run")

    def test_single_empty_mask_group_generation_still_runs_experiment(self):
        from model_explorer.policy.benchmark import generate_synthetic_benchmark_suite
        from model_explorer.policy.experiment import run_experiment_manifest

        with tempfile.TemporaryDirectory() as tmpdir:
            generated = generate_synthetic_benchmark_suite(
                tmpdir,
                seed=9,
                scenario_count=1,
                groups=["high_unreachable_rate"],
            )
            summary = run_experiment_manifest(generated["manifest"])

        self.assertEqual(summary["scenario_count"], 1)
        self.assertEqual(summary["dataset_summary"]["empty_action_mask_count"], 1)
        self.assertNotIn("validation_gates", summary["dataset_summary"])

    def test_mixed_empty_mask_group_generation_still_runs_experiment(self):
        from model_explorer.policy.benchmark import generate_synthetic_benchmark_suite
        from model_explorer.policy.experiment import run_experiment_manifest

        with tempfile.TemporaryDirectory() as tmpdir:
            generated = generate_synthetic_benchmark_suite(
                tmpdir,
                seed=10,
                scenario_count=1,
                groups=["high_unreachable_rate", "sparse_candidates"],
            )
            summary = run_experiment_manifest(generated["manifest"])

        self.assertEqual(summary["scenario_count"], 2)
        self.assertEqual(summary["dataset_summary"]["empty_action_mask_count"], 1)
        self.assertEqual(summary["dataset_summary"]["validation_gates"]["status"], "passed")


class BenchmarkReportEnhancementTests(unittest.TestCase):
    def test_benchmark_report_adds_daily_readiness_sections(self):
        from model_explorer.policy.benchmark import generate_synthetic_benchmark_suite
        from model_explorer.policy.experiment import run_experiment_manifest

        with tempfile.TemporaryDirectory() as tmpdir:
            generated = generate_synthetic_benchmark_suite(tmpdir, seed=5, scenario_count=1)
            summary = run_experiment_manifest(generated["manifest"])
            report_path = Path(summary["report_output"])
            report = report_path.read_text(encoding="utf-8")

        self.assertIn("policy_ranking", summary)
        self.assertIn("baseline_deltas", summary)
        self.assertIn("per_group_winners", summary)
        self.assertIn("failure_scenarios", summary)
        self.assertIn("gate_summary", summary)
        self.assertEqual(summary["gate_summary"]["status"], "passed")
        self.assertEqual(set(summary["per_group_winners"]), set(generated["groups"]))
        self.assertEqual(summary["policy_ranking"][0]["rank"], 1)
        self.assertIn(summary["policy_ranking"][0]["policy"], {"utility", "coverage_heuristic"})
        self.assertEqual(summary["baseline_deltas"], {})
        for section in (
            "## Policy Ranking",
            "## Torch Policy Deltas",
            "## Per-Group Winners",
            "## Failure Scenarios",
            "## Gate Summary",
        ):
            self.assertIn(section, report)
        self.assertTrue(summary["failure_scenarios"])


class ContractRegressionFixtureTests(unittest.TestCase):
    def test_contract_intake_readme_documents_no_real_samples_and_json_only_policy(self):
        readme = (ROOT / "tests" / "fixtures" / "contracts" / "README.md").read_text(encoding="utf-8")

        self.assertIn("No real samples", readme)
        self.assertIn("plain JSON", readme)
        self.assertIn("redact", readme)
        self.assertIn("do not import", readme)

    def test_contract_regression_loader_handles_empty_directory(self):
        from model_explorer.io.contracts import load_contract_regression_scenarios

        with tempfile.TemporaryDirectory() as tmpdir:
            scenarios = load_contract_regression_scenarios(tmpdir)

        self.assertEqual(scenarios, ())

    def test_contract_regression_fixtures_load_and_missing_fields_fallback_is_finite(self):
        from model_explorer.decision.selector import select_goal
        from model_explorer.io.contracts import load_contract_regression_scenarios
        from model_explorer.policy.evaluation import evaluate_policy_baseline_scenarios

        scenarios = load_contract_regression_scenarios(ROOT / "tests" / "fixtures" / "contracts")
        missing_field_scenarios = [
            scenario
            for scenario in scenarios
            if any(not goal.experimental for snapshot in scenario.snapshots for goal in snapshot.top_goals)
        ]

        self.assertGreaterEqual(len(scenarios), 2)
        self.assertTrue(missing_field_scenarios)
        decision = select_goal(missing_field_scenarios[0].snapshots[0])
        report = evaluate_policy_baseline_scenarios(missing_field_scenarios)

        self.assertIsNotNone(decision.selected_goal)
        self.assertGreaterEqual(report["utility"]["final_coverage_rate"], 0.0)
        self.assertGreaterEqual(report["coverage_heuristic"]["total_path_cost"], 0.0)

    def test_curated_synthetic_contract_is_not_labeled_as_real_data(self):
        from model_explorer.io.contracts import contract_regression_paths, load_contract_regression_scenarios

        contract_dir = ROOT / "tests" / "fixtures" / "contracts"
        paths = contract_regression_paths(contract_dir)
        curated_paths = [path for path in paths if "curated-synthetic" in path.name]
        scenarios = load_contract_regression_scenarios(contract_dir)

        self.assertTrue(curated_paths)
        self.assertTrue(all("real" not in path.name for path in curated_paths))
        self.assertGreaterEqual(len(scenarios), 3)


class VerifyEntrypointTests(unittest.TestCase):
    def test_verify_dry_run_reports_one_key_verification_steps(self):
        completed = run_module(["verify", "--dry-run"])
        summary = json.loads(completed.stdout)

        self.assertEqual(summary["status"], "dry_run")
        self.assertEqual(
            [step["name"] for step in summary["steps"]],
            ["unittest", "benchmark_smoke", "git_diff_check"],
        )

    def test_verify_supports_json_output_and_skip_benchmark_smoke(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "verify-summary.json"
            completed = run_module(
                [
                    "verify",
                    "--dry-run",
                    "--skip-benchmark-smoke",
                    "--json-output",
                    str(output_path),
                ]
            )
            stdout_summary = json.loads(completed.stdout)
            file_summary = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(stdout_summary, file_summary)
        self.assertEqual(
            [step["name"] for step in stdout_summary["steps"]],
            ["unittest", "git_diff_check"],
        )

    def test_verify_cli_returns_nonzero_when_verification_fails(self):
        from model_explorer.cli import main

        with patch(
            "model_explorer.cli.run_verification",
            return_value={"status": "failed", "steps": [{"name": "unittest", "returncode": 1}]},
        ):
            with patch("sys.stdout", new=io.StringIO()):
                exit_code = main(["verify", "--skip-benchmark-smoke"])

        self.assertEqual(exit_code, 1)


class BenchmarkDocumentationTests(unittest.TestCase):
    def test_benchmark_and_manifest_docs_define_current_synthetic_scope(self):
        benchmark_doc = (ROOT / "docs" / "benchmark-readiness.md").read_text(encoding="utf-8")
        manifest_doc = (ROOT / "docs" / "experiment-manifest.md").read_text(encoding="utf-8")

        for text in (
            "synthetic smoke / regression suite",
            "not a real-world generalization benchmark",
            "coverage_dominant",
            "missing_experimental_fields",
        ):
            self.assertIn(text, benchmark_doc)
        for text in (
            "model-explorer-experiment/v1",
            "outputs.root",
            "splits.benchmark",
            "Policy Ranking",
            "Gate Summary",
        ):
            self.assertIn(text, manifest_doc)


def _json_snapshot(root: str) -> dict[str, str]:
    base = Path(root)
    return {
        str(path.relative_to(base)): path.read_text(encoding="utf-8")
        for path in sorted(base.rglob("*.json"))
    }


if __name__ == "__main__":
    unittest.main()
