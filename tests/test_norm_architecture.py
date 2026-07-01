from __future__ import annotations

import ast
import importlib
import json
from pathlib import Path
import tomllib

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = REPO_ROOT / "model-explorer"
SRC_ROOT = MODEL_ROOT / "src" / "model_explorer"


def _contract_payload(**overrides):
    payload = {
        "schema_version": "model-explorer-contract/v1",
        "grid": {
            "width": 4,
            "height": 3,
            "resolution": 1.0,
            "frame_id": "map",
            "origin": [0.0, 0.0],
            "layers": ["confidence", "risk"],
        },
        "constraints": {
            "violation_count": 0,
            "passable_ratio": 1.0,
            "reason_counts": {},
        },
        "top_goals": [
            {
                "cell": [1, 1],
                "utility": 0.7,
                "reachable": True,
                "expected_coverage_rate_delta": 0.1,
            }
        ],
        "top_sequences": [
            {
                "cells": [[1, 1]],
                "utility": 0.7,
                "coverage_area": 1.0,
            }
        ],
        "observation_update": {"coverage_rate": 0.2},
        "stable_fields": ["schema_version", "grid", "constraints", "top_goals"],
        "experimental_fields": ["expected_coverage_rate_delta"],
    }
    for key, value in overrides.items():
        payload[key] = value
    return payload


def _write_contract(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def test_contract_field_constants_are_the_policy_feature_source() -> None:
    from model_explorer.contracts.fields import (
        CANDIDATE_FEATURE_NAMES,
        CANDIDATE_COST_FIELDS,
        CANDIDATE_BENEFIT_FIELDS,
        EXPERIMENTAL_CANDIDATE_FIELDS,
    )
    from model_explorer.policy.features import (
        CANDIDATE_FEATURE_NAMES as POLICY_CANDIDATE_FEATURE_NAMES,
        EXPERIMENTAL_CANDIDATE_FIELDS as POLICY_EXPERIMENTAL_FIELDS,
    )

    assert CANDIDATE_FEATURE_NAMES == POLICY_CANDIDATE_FEATURE_NAMES
    assert EXPERIMENTAL_CANDIDATE_FIELDS == POLICY_EXPERIMENTAL_FIELDS
    assert CANDIDATE_BENEFIT_FIELDS == (
        "expected_coverage_rate_delta",
        "expected_new_coverage_area",
        "information_gain",
        "confidence_gain",
        "value",
    )
    assert CANDIDATE_COST_FIELDS == ("risk", "path_cost", "energy_cost")


def test_scenario_rejects_string_boolean_reachable(tmp_path: Path) -> None:
    from model_explorer.core.interfaces import ContractValidationError
    from model_explorer.io.scenario import load_scenario

    payload = _contract_payload()
    payload["top_goals"][0]["reachable"] = "false"

    with pytest.raises(ContractValidationError, match="reachable.*bool"):
        load_scenario(_write_contract(tmp_path, payload))


def test_scenario_rejects_string_numeric_grid_width(tmp_path: Path) -> None:
    from model_explorer.core.interfaces import ContractValidationError
    from model_explorer.io.scenario import load_scenario

    payload = _contract_payload()
    payload["grid"]["width"] = "4"

    with pytest.raises(ContractValidationError, match="grid.width.*int"):
        load_scenario(_write_contract(tmp_path, payload))


def test_scenario_rejects_string_stable_fields(tmp_path: Path) -> None:
    from model_explorer.core.interfaces import ContractValidationError
    from model_explorer.io.scenario import load_scenario

    payload = _contract_payload(stable_fields="schema_version")

    with pytest.raises(ContractValidationError, match="stable_fields.*list"):
        load_scenario(_write_contract(tmp_path, payload))


def test_data_package_has_no_static_policy_imports() -> None:
    violations: list[str] = []
    for path in sorted((SRC_ROOT / "data").glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if not isinstance(node, ast.ImportFrom):
                continue
            module = node.module or ""
            if module == "policy" or module.startswith("policy."):
                violations.append(f"{path.relative_to(MODEL_ROOT)}:{node.lineno}")
            if node.level >= 2 and (module == "policy" or module.startswith("policy.")):
                violations.append(f"{path.relative_to(MODEL_ROOT)}:{node.lineno}")

    assert violations == []


def test_compatibility_facades_stay_small() -> None:
    facade_paths = [
        SRC_ROOT / "policy" / "planning.py",
        SRC_ROOT / "policy" / "path_feedback.py",
        SRC_ROOT / "policy" / "experiment.py",
        SRC_ROOT / "data" / "evaluation_matrix.py",
    ]

    oversized = {
        path.relative_to(MODEL_ROOT).as_posix(): len(path.read_text(encoding="utf-8").splitlines())
        for path in facade_paths
        if len(path.read_text(encoding="utf-8").splitlines()) > 250
    }

    assert oversized == {}


def test_legacy_public_import_paths_remain_available() -> None:
    from model_explorer.data.evaluation_matrix import run_quasi_real_evaluation_manifest
    from model_explorer.policy.experiment import run_experiment_manifest
    from model_explorer.policy.path_feedback import run_path_feedback_manifest
    from model_explorer.policy.planning import PathPlanRequest, evaluate_candidate_paths

    experiment_facade = importlib.import_module("model_explorer.policy.experiment")
    path_feedback_facade = importlib.import_module("model_explorer.policy.path_feedback")

    assert callable(run_quasi_real_evaluation_manifest)
    assert callable(run_experiment_manifest)
    assert callable(getattr(experiment_facade, "_run_training"))
    assert callable(run_path_feedback_manifest)
    assert callable(getattr(path_feedback_facade, "_selected_after_feedback"))
    assert PathPlanRequest.__name__ == "PathPlanRequest"
    assert callable(evaluate_candidate_paths)


def test_pyproject_extras_declare_training_raster_and_all() -> None:
    pyproject = tomllib.loads((MODEL_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    extras = pyproject["project"]["optional-dependencies"]

    assert extras["training"] == ["torch>=2.0"]
    assert extras["raster"] == ["Pillow>=10"]
    assert set(extras["all"]) == {"torch>=2.0", "Pillow>=10"}


def test_scripts_are_thin_cli_wrappers() -> None:
    violations: list[str] = []
    allowed_modules = {"model_explorer.cli"}
    for path in sorted((MODEL_ROOT / "scripts").glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import) and any(alias.name == "argparse" for alias in node.names):
                violations.append(f"{path.name}: imports argparse")
            if isinstance(node, ast.ImportFrom) and node.module:
                if node.module.startswith("model_explorer.") and node.module not in allowed_modules:
                    violations.append(f"{path.name}: imports {node.module}")

    assert violations == []


def test_verification_architecture_static_check_passes() -> None:
    from model_explorer.verification import _run_architecture_static_check

    result = _run_architecture_static_check(MODEL_ROOT)

    assert result["returncode"] == 0
    assert result["violations"] == []


def test_impl_modules_are_only_compatibility_shims() -> None:
    impl_paths = [
        SRC_ROOT / "policy" / "planning_impl.py",
        SRC_ROOT / "policy" / "path_feedback_impl.py",
        SRC_ROOT / "experiments" / "experiment_impl.py",
        SRC_ROOT / "experiments" / "quasi_real_matrix" / "evaluation_matrix_impl.py",
    ]

    oversized = {
        path.relative_to(MODEL_ROOT).as_posix(): len(path.read_text(encoding="utf-8").splitlines())
        for path in impl_paths
        if len(path.read_text(encoding="utf-8").splitlines()) > 250
    }

    assert oversized == {}


def test_legacy_facade_exports_do_not_leak_temporary_names() -> None:
    from model_explorer.policy import path_feedback, planning
    from model_explorer.policy import path_feedback_impl, planning_impl

    forbidden = {
        "Any",
        "Counter",
        "Path",
        "Protocol",
        "Sequence",
        "annotations",
        "ceil",
        "dataclass",
        "deque",
        "field",
        "heappop",
        "heappush",
        "hypot",
        "json",
        "load_scenario",
        "os",
        "subprocess",
        "sys",
        "tempfile",
    }

    for module in (planning, planning_impl, path_feedback, path_feedback_impl):
        assert forbidden.isdisjoint(set(module.__all__))

    assert "PathPlanRequest" in planning.__all__
    assert "evaluate_candidate_paths" in planning.__all__
    assert "run_path_feedback_manifest" in path_feedback.__all__
    assert "_selected_after_feedback" in path_feedback.__all__


def test_runner_modules_are_only_orchestration_layers() -> None:
    runner_limits = {
        SRC_ROOT / "policy" / "path_feedback_runner.py": 800,
        SRC_ROOT / "experiments" / "runner.py": 800,
        SRC_ROOT / "experiments" / "quasi_real_matrix" / "runner.py": 700,
    }

    oversized = {
        path.relative_to(MODEL_ROOT).as_posix(): len(path.read_text(encoding="utf-8").splitlines())
        for path, limit in runner_limits.items()
        if len(path.read_text(encoding="utf-8").splitlines()) > limit
    }

    assert oversized == {}


def test_split_modules_do_not_import_their_runner() -> None:
    runner_splits = {
        "model_explorer.policy.path_feedback_runner": [
            SRC_ROOT / "policy" / "feedback_selection.py",
            SRC_ROOT / "policy" / "path_feedback_artifacts.py",
            SRC_ROOT / "policy" / "path_feedback_diagnostics.py",
            SRC_ROOT / "policy" / "path_feedback_manifest.py",
            SRC_ROOT / "policy" / "path_feedback_reports.py",
            SRC_ROOT / "policy" / "path_feedback_summary.py",
        ],
        "model_explorer.experiments.runner": [
            SRC_ROOT / "experiments" / "environment.py",
            SRC_ROOT / "experiments" / "evaluation.py",
            SRC_ROOT / "experiments" / "manifest.py",
            SRC_ROOT / "experiments" / "reports.py",
            SRC_ROOT / "experiments" / "selection.py",
            SRC_ROOT / "experiments" / "training_matrix.py",
        ],
        "model_explorer.experiments.quasi_real_matrix.runner": [
            SRC_ROOT / "experiments" / "quasi_real_matrix" / "manifest.py",
            SRC_ROOT / "experiments" / "quasi_real_matrix" / "metrics.py",
            SRC_ROOT / "experiments" / "quasi_real_matrix" / "reports.py",
            SRC_ROOT / "experiments" / "quasi_real_matrix" / "scenario_generation.py",
            SRC_ROOT / "experiments" / "quasi_real_matrix" / "selection.py",
        ],
    }
    violations: list[str] = []
    for runner_module, paths in runner_splits.items():
        for path in paths:
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name == runner_module:
                            violations.append(f"{path.relative_to(MODEL_ROOT)}:{node.lineno}")
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    if module == runner_module or module.endswith(runner_module):
                        violations.append(f"{path.relative_to(MODEL_ROOT)}:{node.lineno}")
                    if node.level and module == "runner":
                        violations.append(f"{path.relative_to(MODEL_ROOT)}:{node.lineno}")

    assert violations == []


def test_verification_catches_nested_legacy_and_private_runner_imports(tmp_path) -> None:
    from model_explorer.verification import _run_architecture_static_check

    required_files = [
        "src/model_explorer/policy/planning.py",
        "src/model_explorer/policy/path_feedback.py",
        "src/model_explorer/policy/experiment.py",
        "src/model_explorer/data/evaluation_matrix.py",
        "src/model_explorer/policy/planning_impl.py",
        "src/model_explorer/policy/path_feedback_impl.py",
        "src/model_explorer/experiments/experiment_impl.py",
        "src/model_explorer/experiments/quasi_real_matrix/evaluation_matrix_impl.py",
        "src/model_explorer/policy/path_feedback_runner.py",
        "src/model_explorer/experiments/runner.py",
        "src/model_explorer/experiments/quasi_real_matrix/runner.py",
        "src/model_explorer/policy/path_feedback_artifacts.py",
        "src/model_explorer/policy/path_feedback_diagnostics.py",
        "src/model_explorer/policy/path_feedback_manifest.py",
        "src/model_explorer/policy/path_feedback_reports.py",
        "src/model_explorer/policy/path_feedback_summary.py",
        "src/model_explorer/policy/feedback_selection.py",
        "src/model_explorer/experiments/environment.py",
        "src/model_explorer/experiments/evaluation.py",
        "src/model_explorer/experiments/manifest.py",
        "src/model_explorer/experiments/reports.py",
        "src/model_explorer/experiments/selection.py",
        "src/model_explorer/experiments/training_matrix.py",
        "src/model_explorer/experiments/quasi_real_matrix/manifest.py",
        "src/model_explorer/experiments/quasi_real_matrix/metrics.py",
        "src/model_explorer/experiments/quasi_real_matrix/reports.py",
        "src/model_explorer/experiments/quasi_real_matrix/scenario_generation.py",
        "src/model_explorer/experiments/quasi_real_matrix/selection.py",
    ]
    for relative in required_files:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# compatibility test fixture\n", encoding="utf-8")

    offender = tmp_path / "src/model_explorer/offender.py"
    offender.write_text(
        "\n".join(
            [
                "def nested_legacy_import():",
                "    from model_explorer.policy.path_feedback import run_path_feedback_manifest",
                "    return run_path_feedback_manifest",
                "",
                "def private_runner_import():",
                "    from model_explorer.policy.path_feedback_runner import _run_feedback_scenario",
                "    return _run_feedback_scenario",
                "",
            ]
        ),
        encoding="utf-8",
    )

    result = _run_architecture_static_check(tmp_path)
    rule_targets = {(violation["rule"], violation.get("target")) for violation in result["violations"]}

    assert (
        "no_business_code_legacy_import",
        "model_explorer.policy.path_feedback",
    ) in rule_targets
    assert (
        "no_business_code_private_runner_import",
        "model_explorer.policy.path_feedback_runner",
    ) in rule_targets


def test_decision_package_has_no_static_policy_imports() -> None:
    violations: list[str] = []
    for path in sorted((SRC_ROOT / "decision").glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module.startswith("model_explorer.policy") or (
                    node.level >= 2 and (module == "policy" or module.startswith("policy."))
                ):
                    violations.append(f"{path.relative_to(MODEL_ROOT)}:{node.lineno}")
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("model_explorer.policy"):
                        violations.append(f"{path.relative_to(MODEL_ROOT)}:{node.lineno}")

    assert violations == []


def test_tests_do_not_import_private_legacy_symbols() -> None:
    target_modules = {
        "model_explorer.policy.planning",
        "model_explorer.policy.path_feedback",
        "model_explorer.policy.experiment",
        "model_explorer.data.evaluation_matrix",
        "model_explorer.policy.path_feedback_runner",
        "model_explorer.experiments.runner",
        "model_explorer.experiments.quasi_real_matrix.runner",
    }
    violations: list[str] = []
    for path in sorted((MODEL_ROOT / "tests").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or node.module not in target_modules:
                continue
            for alias in node.names:
                if alias.name.startswith("_"):
                    violations.append(
                        f"{path.relative_to(MODEL_ROOT).as_posix()}:{node.lineno}: "
                        f"from {node.module} import {alias.name}"
                    )

    assert violations == []
