from __future__ import annotations

import ast
from collections import Counter
import importlib
import json
from pathlib import Path
import tomllib

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = REPO_ROOT / "model-explorer"
SRC_ROOT = MODEL_ROOT / "src" / "model_explorer"

ARCHITECTURE_FIXTURE_FILES = (
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
    "src/model_explorer/policy/planning_anchor.py",
    "src/model_explorer/policy/planning_diagnostics.py",
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
    "src/model_explorer/policy/path_feedback_diagnostic_aggregate.py",
    "src/model_explorer/policy/path_feedback_diagnostic_interpretation.py",
    "src/model_explorer/policy/path_feedback_backend_diagnostics.py",
    "src/model_explorer/policy/path_feedback_candidate_audits.py",
    "src/model_explorer/policy/feedback_selection_types.py",
    "src/model_explorer/policy/feedback_selection_scoring.py",
    "src/model_explorer/policy/feedback_selection_channel.py",
    "src/model_explorer/policy/feedback_selection_trainability.py",
    "src/model_explorer/policy/feedback_selection_anchor.py",
    "src/model_explorer/policy/feedback_selection_sources.py",
    "src/model_explorer/policy/planning_anchor_evaluation.py",
    "src/model_explorer/policy/planning_anchor_projection.py",
    "src/model_explorer/policy/planning_anchor_grid.py",
    "src/model_explorer/policy/planning_backend_summaries.py",
    "src/model_explorer/policy/planning_platform_feasibility.py",
    "src/model_explorer/policy/planning_diagnostic_interpretation.py",
    "src/model_explorer/experiments/quasi_real_matrix/quality_gates.py",
    "src/model_explorer/experiments/quasi_real_matrix/decision_diagnostics.py",
    "src/model_explorer/experiments/quasi_real_matrix/architecture_selection.py",
    "src/model_explorer/experiments/quasi_real_matrix/stability.py",
    "tests/test_model_explorer.py",
    "tests/test_quasi_real_data_pipeline.py",
)

EXPLICIT_EXPORT_MODULES = (
    "model_explorer.experiments.environment",
    "model_explorer.experiments.evaluation",
    "model_explorer.experiments.manifest",
    "model_explorer.experiments.reports",
    "model_explorer.experiments.selection",
    "model_explorer.experiments.training_matrix",
    "model_explorer.policy.path_feedback_runner",
    "model_explorer.policy.planning_adapters",
    "model_explorer.policy.planning_routes",
    "model_explorer.policy.planning_types",
    "model_explorer.policy.planning_utils",
)

FORBIDDEN_EXPLICIT_EXPORT_NAMES = {
    "Any",
    "Path",
    "Protocol",
    "Sequence",
    "annotations",
    "dataclass",
    "field",
    "heappop",
    "heappush",
    "hypot",
    "importlib_util",
    "isfinite",
    "json",
    "os",
    "sqrt",
    "subprocess",
    "sys",
    "tempfile",
}

EXPECTED_FUNCTION_LINE_LIMITS = {
    ("src/model_explorer/experiments/quasi_real_matrix/reports.py", "_markdown_report"): 180,
    ("src/model_explorer/experiments/reports.py", "_markdown_report"): 180,
    ("src/model_explorer/policy/path_feedback_summary.py", "compact_path_feedback_summary"): 180,
    ("src/model_explorer/experiments/training_matrix.py", "_run_training"): 180,
    ("src/model_explorer/verification.py", "_run_architecture_static_check"): 180,
    ("src/model_explorer/policy/path_feedback_backend_diagnostics.py", "_sampled_region_path_diagnostics"): 180,
    ("src/model_explorer/policy/path_feedback_reports.py", "render_path_feedback_markdown"): 180,
    ("src/model_explorer/policy/collector.py", "collect_dynamic_rollout_episode"): 180,
    ("src/model_explorer/policy/evaluation.py", "_evaluate_strategy"): 180,
    (
        "src/model_explorer/experiments/quasi_real_matrix/architecture_selection.py",
        "_architecture_selection_summary",
    ): 180,
}
EXPECTED_CURRENT_FUNCTION_LINE_VIOLATIONS = {
    target: limit
    for target, limit in EXPECTED_FUNCTION_LINE_LIMITS.items()
    if target
    not in {
        ("src/model_explorer/experiments/quasi_real_matrix/reports.py", "_markdown_report"),
        ("src/model_explorer/experiments/reports.py", "_markdown_report"),
        ("src/model_explorer/policy/path_feedback_summary.py", "compact_path_feedback_summary"),
        ("src/model_explorer/verification.py", "_run_architecture_static_check"),
        ("src/model_explorer/policy/path_feedback_reports.py", "render_path_feedback_markdown"),
    }
}


def _write_architecture_fixture(
    tmp_path: Path,
    *,
    omit: set[str] | None = None,
    overrides: dict[str, str] | None = None,
) -> Path:
    omitted = omit or set()
    content_by_path = overrides or {}
    for relative in ARCHITECTURE_FIXTURE_FILES:
        if relative in omitted:
            continue
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content_by_path.get(relative, "# architecture fixture\n"), encoding="utf-8")
    return tmp_path


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


def test_verification_architecture_static_check_passes_target_governance_state() -> None:
    from model_explorer.verification import _run_architecture_static_check

    result = _run_architecture_static_check(MODEL_ROOT)
    rule_counts = Counter(violation["rule"] for violation in result["violations"])
    target_facade_violations = {
        violation["path"]
        for violation in result["violations"]
        if violation["rule"] == "target_facade_line_limit"
    }
    giant_test_violations = {
        violation["path"]
        for violation in result["violations"]
        if violation["rule"] == "giant_test_line_limit"
    }
    dynamic_globals_all_violations = {
        violation["path"]
        for violation in result["violations"]
        if violation["rule"] == "no_dynamic_globals_all"
    }
    function_limit_violations = {
        (violation["path"], violation["function"], violation["limit"])
        for violation in result["violations"]
        if violation["rule"] == "function_line_limit"
    }

    assert result["returncode"] == 1
    assert rule_counts == Counter({"function_line_limit": len(EXPECTED_CURRENT_FUNCTION_LINE_VIOLATIONS)})
    assert dynamic_globals_all_violations == set()
    assert function_limit_violations == {
        (path, function_name, limit)
        for (path, function_name), limit in EXPECTED_CURRENT_FUNCTION_LINE_VIOLATIONS.items()
    }
    assert (
        "src/model_explorer/verification.py::_run_architecture_static_check"
        in result["function_line_limits"]
    )
    assert target_facade_violations == set()
    assert giant_test_violations == set()


def test_verification_catches_oversized_target_facade(tmp_path: Path) -> None:
    from model_explorer.verification import _run_architecture_static_check

    target = "src/model_explorer/policy/feedback_selection.py"
    oversized_module = "\n".join(f"line_{index} = None" for index in range(251)) + "\n"
    _write_architecture_fixture(tmp_path, overrides={target: oversized_module})

    result = _run_architecture_static_check(tmp_path)

    assert {
        (violation["rule"], violation["path"])
        for violation in result["violations"]
    } >= {("target_facade_line_limit", target)}


def test_verification_catches_missing_planned_split_module(tmp_path: Path) -> None:
    from model_explorer.verification import _run_architecture_static_check

    missing = "src/model_explorer/policy/path_feedback_diagnostic_aggregate.py"
    _write_architecture_fixture(tmp_path, omit={missing})

    result = _run_architecture_static_check(tmp_path)

    assert {
        (violation["rule"], violation["path"])
        for violation in result["violations"]
    } >= {("target_split_module_exists", missing)}


def test_verification_catches_split_module_importing_facade_runner_or_impl(tmp_path: Path) -> None:
    from model_explorer.verification import _run_architecture_static_check

    split_module = "src/model_explorer/policy/path_feedback_diagnostic_aggregate.py"
    _write_architecture_fixture(
        tmp_path,
        overrides={
            split_module: "\n".join(
                [
                    "from model_explorer.policy.path_feedback_diagnostics import channel_aware_astar_diagnostics",
                    "from model_explorer.policy import path_feedback_runner",
                    "from model_explorer.policy.path_feedback_impl import run_path_feedback_manifest",
                    "",
                ]
            ),
        },
    )

    result = _run_architecture_static_check(tmp_path)
    forbidden_targets = {
        violation["target"]
        for violation in result["violations"]
        if violation["rule"] == "no_target_split_module_forbidden_import"
    }

    assert forbidden_targets >= {
        "model_explorer.policy.path_feedback_diagnostics",
        "model_explorer.policy.path_feedback_runner",
        "model_explorer.policy.path_feedback_impl",
    }


def test_verification_catches_production_private_import_from_target_facade(tmp_path: Path) -> None:
    from model_explorer.verification import _run_architecture_static_check

    _write_architecture_fixture(tmp_path)
    offender = tmp_path / "src/model_explorer/policy/offender.py"
    offender.write_text(
        "from model_explorer.policy.feedback_selection import _score_evaluations\n",
        encoding="utf-8",
    )

    result = _run_architecture_static_check(tmp_path)

    assert {
        (violation["rule"], violation.get("target"))
        for violation in result["violations"]
    } >= {("no_target_facade_private_production_import", "model_explorer.policy.feedback_selection")}


def test_verification_catches_test_private_import_from_target_facade(tmp_path: Path) -> None:
    from model_explorer.verification import _run_architecture_static_check

    _write_architecture_fixture(tmp_path)
    offender = tmp_path / "tests/test_private_target_import.py"
    offender.write_text(
        "from model_explorer.policy.feedback_selection import _score_evaluations\n",
        encoding="utf-8",
    )

    result = _run_architecture_static_check(tmp_path)

    assert {
        (violation["rule"], violation.get("target"))
        for violation in result["violations"]
    } >= {("no_target_facade_private_test_import", "model_explorer.policy.feedback_selection")}


def test_verification_catches_dynamic_globals_all_in_any_production_module(tmp_path: Path) -> None:
    from model_explorer.verification import _run_architecture_static_check

    _write_architecture_fixture(tmp_path)
    module_target = "src/model_explorer/policy/planning_utils.py"
    module_path = tmp_path / module_target
    module_path.parent.mkdir(parents=True, exist_ok=True)
    module_path.write_text(
        "__all__ = [name for name in globals() if not name.startswith('__')]\n",
        encoding="utf-8",
    )
    control_flow_target = "src/model_explorer/policy/conditional_dynamic_all.py"
    control_flow_path = tmp_path / control_flow_target
    control_flow_path.write_text(
        "\n".join(
            [
                "if True:",
                "    __all__ = [name for name in globals() if not name.startswith('__')]",
                "",
            ]
        ),
        encoding="utf-8",
    )
    local_target = "src/model_explorer/policy/local_dynamic_all.py"
    local_path = tmp_path / local_target
    local_path.write_text(
        "\n".join(
            [
                "def export_names():",
                "    __all__ = [name for name in globals() if not name.startswith('__')]",
                "    return __all__",
                "",
            ]
        ),
        encoding="utf-8",
    )

    result = _run_architecture_static_check(tmp_path)
    dynamic_globals_all_paths = {
        violation["path"]
        for violation in result["violations"]
        if violation["rule"] == "no_dynamic_globals_all"
    }

    assert dynamic_globals_all_paths == {module_target, control_flow_target}


def test_verification_catches_function_line_budget_violation(tmp_path: Path) -> None:
    from model_explorer.verification import _run_architecture_static_check

    _write_architecture_fixture(tmp_path)
    target = "src/model_explorer/policy/collector.py"
    path = tmp_path / target
    path.parent.mkdir(parents=True, exist_ok=True)
    oversized_function = "\n".join(
        ["def collect_dynamic_rollout_episode():", *(f"    line_{index} = None" for index in range(180))]
    )
    path.write_text(f"{oversized_function}\n", encoding="utf-8")

    result = _run_architecture_static_check(tmp_path)

    assert {
        (violation["rule"], violation["path"], violation["function"], violation["limit"])
        for violation in result["violations"]
    } >= {("function_line_limit", target, "collect_dynamic_rollout_episode", 180)}


def test_verification_catches_missing_function_line_budget_target(tmp_path: Path) -> None:
    from model_explorer.verification import _run_architecture_static_check

    _write_architecture_fixture(tmp_path)
    target = "src/model_explorer/policy/collector.py"
    path = tmp_path / target
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("def other_function():\n    return None\n", encoding="utf-8")

    result = _run_architecture_static_check(tmp_path)

    assert {
        (violation["rule"], violation["path"], violation["function"], violation["limit"])
        for violation in result["violations"]
    } >= {("function_line_target_missing", target, "collect_dynamic_rollout_episode", 180)}


def test_verification_counts_unique_scanned_files(tmp_path: Path) -> None:
    from model_explorer.verification import _run_architecture_static_check

    _write_architecture_fixture(tmp_path)

    result = _run_architecture_static_check(tmp_path)
    expected_scanned_files = len({path.resolve() for path in tmp_path.rglob("*.py")})

    assert result["scanned_files"] == expected_scanned_files


def test_verification_catches_giant_test_budget_violation(tmp_path: Path) -> None:
    from model_explorer.verification import _run_architecture_static_check

    target = "tests/test_model_explorer.py"
    oversized_test = "\n".join("# test budget fixture" for _ in range(5601)) + "\n"
    _write_architecture_fixture(tmp_path, overrides={target: oversized_test})

    result = _run_architecture_static_check(tmp_path)

    assert {
        (violation["rule"], violation["path"])
        for violation in result["violations"]
    } >= {("giant_test_line_limit", target)}


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


def test_target_split_modules_have_explicit_exports_without_temporary_names() -> None:
    for module_name in EXPLICIT_EXPORT_MODULES:
        module = importlib.import_module(module_name)
        assert isinstance(module.__all__, tuple)
        assert FORBIDDEN_EXPLICIT_EXPORT_NAMES.isdisjoint(set(module.__all__))


def test_legacy_facade_imports_expose_key_symbols() -> None:
    from model_explorer.experiments.environment import environment_metadata, git_metadata
    from model_explorer.experiments.evaluation import aggregate_rollout_metrics, collect_episodes
    from model_explorer.experiments.manifest import ExperimentManifest, load_experiment_manifest
    from model_explorer.experiments.reports import render_experiment_markdown
    from model_explorer.experiments.selection import select_best_training_run
    from model_explorer.experiments.training_matrix import run_training
    from model_explorer.policy.path_feedback import run_path_feedback_manifest
    from model_explorer.policy.planning import PathPlanRequest, evaluate_candidate_paths

    path_feedback_facade = importlib.import_module("model_explorer.policy.path_feedback")

    assert callable(environment_metadata)
    assert callable(git_metadata)
    assert callable(aggregate_rollout_metrics)
    assert callable(collect_episodes)
    assert ExperimentManifest.__name__ == "ExperimentManifest"
    assert callable(load_experiment_manifest)
    assert callable(render_experiment_markdown)
    assert callable(select_best_training_run)
    assert callable(run_training)
    assert callable(getattr(path_feedback_facade, "_selected_after_feedback"))
    assert callable(run_path_feedback_manifest)
    assert PathPlanRequest.__name__ == "PathPlanRequest"
    assert callable(evaluate_candidate_paths)


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
            SRC_ROOT / "experiments" / "quasi_real_matrix" / "quality_gates.py",
            SRC_ROOT / "experiments" / "quasi_real_matrix" / "decision_diagnostics.py",
            SRC_ROOT / "experiments" / "quasi_real_matrix" / "architecture_selection.py",
            SRC_ROOT / "experiments" / "quasi_real_matrix" / "stability.py",
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

    _write_architecture_fixture(tmp_path)

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
