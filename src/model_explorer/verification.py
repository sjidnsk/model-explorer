from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


FORBIDDEN_IMPORT_PATTERNS = ("a_gcs_ws", "dev-platform-constraints")
_FORBIDDEN_IMPORT_ALIASES = {
    "a_gcs_ws": ("a_gcs_ws",),
    "dev-platform-constraints": ("dev-platform-constraints", "dev_platform_constraints"),
}
_IMPORT_LINE_RE = re.compile(r"^\s*(?:from|import)\s+")
_FORBIDDEN_IMPORT_SCAN_DIRS = ("src", "tests", "scripts")
_FACADE_LINE_LIMIT = 250
_FACADE_FILES = (
    "src/model_explorer/policy/planning.py",
    "src/model_explorer/policy/path_feedback.py",
    "src/model_explorer/policy/experiment.py",
    "src/model_explorer/data/evaluation_matrix.py",
)
_IMPL_FILES = (
    "src/model_explorer/policy/planning_impl.py",
    "src/model_explorer/policy/path_feedback_impl.py",
    "src/model_explorer/experiments/experiment_impl.py",
    "src/model_explorer/experiments/quasi_real_matrix/evaluation_matrix_impl.py",
)
_RUNNER_LINE_LIMITS = {
    "src/model_explorer/policy/path_feedback_runner.py": 800,
    "src/model_explorer/experiments/runner.py": 800,
    "src/model_explorer/experiments/quasi_real_matrix/runner.py": 700,
}
_RUNNER_SPLIT_MODULES = {
    "model_explorer.policy.path_feedback_runner": (
        "src/model_explorer/policy/path_feedback_artifacts.py",
        "src/model_explorer/policy/path_feedback_diagnostics.py",
        "src/model_explorer/policy/path_feedback_manifest.py",
        "src/model_explorer/policy/path_feedback_reports.py",
        "src/model_explorer/policy/path_feedback_summary.py",
        "src/model_explorer/policy/feedback_selection.py",
    ),
    "model_explorer.experiments.runner": (
        "src/model_explorer/experiments/environment.py",
        "src/model_explorer/experiments/evaluation.py",
        "src/model_explorer/experiments/manifest.py",
        "src/model_explorer/experiments/reports.py",
        "src/model_explorer/experiments/selection.py",
        "src/model_explorer/experiments/training_matrix.py",
    ),
    "model_explorer.experiments.quasi_real_matrix.runner": (
        "src/model_explorer/experiments/quasi_real_matrix/manifest.py",
        "src/model_explorer/experiments/quasi_real_matrix/metrics.py",
        "src/model_explorer/experiments/quasi_real_matrix/reports.py",
        "src/model_explorer/experiments/quasi_real_matrix/scenario_generation.py",
        "src/model_explorer/experiments/quasi_real_matrix/selection.py",
    ),
}
_RUNNER_IMPORT_TARGET_MODULES = set(_RUNNER_SPLIT_MODULES)
_PRIVATE_IMPORT_TARGET_MODULES = {
    "model_explorer.policy.planning",
    "model_explorer.policy.path_feedback",
    "model_explorer.policy.experiment",
    "model_explorer.data.evaluation_matrix",
    "model_explorer.policy.path_feedback_runner",
    "model_explorer.experiments.runner",
    "model_explorer.experiments.quasi_real_matrix.runner",
}
_LEGACY_IMPORT_TARGET_MODULES = {
    "model_explorer.policy.planning",
    "model_explorer.policy.path_feedback",
    "model_explorer.policy.experiment",
    "model_explorer.data.evaluation_matrix",
    "model_explorer.policy.planning_impl",
    "model_explorer.policy.path_feedback_impl",
    "model_explorer.experiments.experiment_impl",
    "model_explorer.experiments.quasi_real_matrix.evaluation_matrix_impl",
}
_LEGACY_IMPORT_ALLOWLIST = set(_FACADE_FILES)

_TARGET_FACADE_LINE_LIMIT = 250
_TARGET_FACADE_FILES = (
    "src/model_explorer/policy/path_feedback_diagnostics.py",
    "src/model_explorer/policy/feedback_selection.py",
    "src/model_explorer/policy/planning_anchor.py",
    "src/model_explorer/policy/planning_diagnostics.py",
    "src/model_explorer/experiments/quasi_real_matrix/selection.py",
)
_TARGET_SPLIT_MODULE_LINE_LIMIT = 800
_TARGET_SPLIT_MODULE_GROUPS = {
    "model_explorer.policy.path_feedback_diagnostics": {
        "runner": "model_explorer.policy.path_feedback_runner",
        "impl": "model_explorer.policy.path_feedback_impl",
        "paths": (
            "src/model_explorer/policy/path_feedback_diagnostic_aggregate.py",
            "src/model_explorer/policy/path_feedback_diagnostic_interpretation.py",
            "src/model_explorer/policy/path_feedback_backend_diagnostics.py",
            "src/model_explorer/policy/path_feedback_candidate_audits.py",
        ),
    },
    "model_explorer.policy.feedback_selection": {
        "runner": "model_explorer.policy.path_feedback_runner",
        "impl": "model_explorer.policy.path_feedback_impl",
        "paths": (
            "src/model_explorer/policy/feedback_selection_types.py",
            "src/model_explorer/policy/feedback_selection_scoring.py",
            "src/model_explorer/policy/feedback_selection_channel.py",
            "src/model_explorer/policy/feedback_selection_trainability.py",
            "src/model_explorer/policy/feedback_selection_anchor.py",
            "src/model_explorer/policy/feedback_selection_sources.py",
        ),
    },
    "model_explorer.policy.planning_anchor": {
        "runner": "model_explorer.policy.path_feedback_runner",
        "impl": "model_explorer.policy.planning_impl",
        "paths": (
            "src/model_explorer/policy/planning_anchor_evaluation.py",
            "src/model_explorer/policy/planning_anchor_projection.py",
            "src/model_explorer/policy/planning_anchor_grid.py",
        ),
    },
    "model_explorer.policy.planning_diagnostics": {
        "runner": "model_explorer.policy.path_feedback_runner",
        "impl": "model_explorer.policy.planning_impl",
        "paths": (
            "src/model_explorer/policy/planning_backend_summaries.py",
            "src/model_explorer/policy/planning_platform_feasibility.py",
            "src/model_explorer/policy/planning_diagnostic_interpretation.py",
        ),
    },
    "model_explorer.experiments.quasi_real_matrix.selection": {
        "runner": "model_explorer.experiments.quasi_real_matrix.runner",
        "impl": "model_explorer.experiments.quasi_real_matrix.evaluation_matrix_impl",
        "paths": (
            "src/model_explorer/experiments/quasi_real_matrix/quality_gates.py",
            "src/model_explorer/experiments/quasi_real_matrix/decision_diagnostics.py",
            "src/model_explorer/experiments/quasi_real_matrix/architecture_selection.py",
            "src/model_explorer/experiments/quasi_real_matrix/stability.py",
        ),
    },
}
_TARGET_SPLIT_MODULES = tuple(
    path
    for group in _TARGET_SPLIT_MODULE_GROUPS.values()
    for path in group["paths"]
)
_TARGET_FACADE_MODULES = frozenset(_TARGET_SPLIT_MODULE_GROUPS)
_TARGET_GOVERNED_PRODUCTION_FILES = (*_TARGET_FACADE_FILES, *_TARGET_SPLIT_MODULES)
_TARGET_SPLIT_MODULE_IMPORT_TARGETS = {
    path: tuple(sorted({facade_module, group["runner"], group["impl"]}))
    for facade_module, group in _TARGET_SPLIT_MODULE_GROUPS.items()
    for path in group["paths"]
}
_GIANT_TEST_LINE_LIMITS = {
    "tests/test_model_explorer.py": 5600,
    "tests/test_quasi_real_data_pipeline.py": 1050,
}
_FUNCTION_LINE_LIMITS = {
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


def run_verification(
    root: str | Path,
    *,
    dry_run: bool = False,
    skip_benchmark_smoke: bool = False,
) -> dict[str, Any]:
    project_root = Path(root)
    steps = _planned_steps(project_root, skip_benchmark_smoke=skip_benchmark_smoke)
    if dry_run:
        return {"status": "dry_run", "steps": steps}

    results: list[dict[str, Any]] = []
    for step in steps:
        if step["name"] == "benchmark_smoke":
            results.append(_run_benchmark_smoke(project_root))
        elif step["name"] == "forbidden_import_check":
            results.append(_run_forbidden_import_check(project_root))
        elif step["name"] == "architecture_static_check":
            results.append(_run_architecture_static_check(project_root))
        else:
            results.append(_run_command_step(step, cwd=project_root))
    status = "passed" if all(result["returncode"] == 0 for result in results) else "failed"
    return {"status": status, "steps": results}


def _planned_steps(project_root: Path, *, skip_benchmark_smoke: bool) -> list[dict[str, Any]]:
    steps = [
        {
            "name": "unittest",
            "kind": "subprocess",
            "command": [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
        },
        {
            "name": "benchmark_smoke",
            "kind": "python",
            "manifest": str(project_root / "tests" / "fixtures" / "synthetic_experiment" / "synthetic-benchmark-experiment.json"),
        },
        {
            "name": "forbidden_import_check",
            "kind": "python_scan",
            "paths": [str(project_root / name) for name in _FORBIDDEN_IMPORT_SCAN_DIRS],
            "forbidden_patterns": list(FORBIDDEN_IMPORT_PATTERNS),
        },
        {
            "name": "architecture_static_check",
            "kind": "python_scan",
            "facade_line_limit": _FACADE_LINE_LIMIT,
            "facades": [str(project_root / name) for name in _FACADE_FILES],
            "impls": [str(project_root / name) for name in _IMPL_FILES],
            "runner_line_limits": {
                str(project_root / name): limit for name, limit in _RUNNER_LINE_LIMITS.items()
            },
            "target_facade_line_limit": _TARGET_FACADE_LINE_LIMIT,
            "target_facades": [str(project_root / name) for name in _TARGET_FACADE_FILES],
            "target_split_module_line_limit": _TARGET_SPLIT_MODULE_LINE_LIMIT,
            "target_split_modules": [str(project_root / name) for name in _TARGET_SPLIT_MODULES],
            "giant_test_line_limits": {
                str(project_root / name): limit for name, limit in _GIANT_TEST_LINE_LIMITS.items()
            },
            "no_dynamic_globals_all_root": str(project_root / "src" / "model_explorer"),
            "function_line_limits": {
                f"{project_root / relative_path}::{function_name}": limit
                for (relative_path, function_name), limit in _FUNCTION_LINE_LIMITS.items()
            },
        },
        {"name": "git_diff_check", "kind": "subprocess", "command": ["git", "diff", "--check"]},
    ]
    if skip_benchmark_smoke:
        return [step for step in steps if step["name"] != "benchmark_smoke"]
    return steps


def _run_command_step(step: dict[str, Any], *, cwd: Path) -> dict[str, Any]:
    completed = subprocess.run(
        step["command"],
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return {
        "name": step["name"],
        "kind": step.get("kind", "subprocess"),
        "command": list(step["command"]),
        "returncode": completed.returncode,
        "stdout_tail": _tail(completed.stdout),
        "stderr_tail": _tail(completed.stderr),
    }


def _run_benchmark_smoke(project_root: Path) -> dict[str, Any]:
    from .experiments.runner import run_experiment_manifest

    source_manifest = project_root / "tests" / "fixtures" / "synthetic_experiment" / "synthetic-benchmark-experiment.json"
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "benchmark-smoke.json"
            payload = json.loads(source_manifest.read_text(encoding="utf-8"))
            fixture_root = source_manifest.parent
            payload["splits"]["benchmark"] = {
                group_name: [str((fixture_root / path).resolve()) for path in scenario_paths]
                for group_name, scenario_paths in payload["splits"]["benchmark"].items()
            }
            payload["outputs"] = {
                "rollouts": str(Path(tmpdir) / "benchmark-rollouts.jsonl"),
                "evaluation": str(Path(tmpdir) / "benchmark-evaluation.json"),
                "dataset_summary": str(Path(tmpdir) / "benchmark-dataset-summary.json"),
                "report": str(Path(tmpdir) / "benchmark-report.md"),
            }
            manifest_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            summary = run_experiment_manifest(manifest_path)
        return {
            "name": "benchmark_smoke",
            "kind": "python",
            "returncode": 0,
            "scenario_count": summary.get("scenario_count", 0),
            "transition_count": summary.get("transition_count", 0),
        }
    except Exception as exc:
        return {
            "name": "benchmark_smoke",
            "kind": "python",
            "returncode": 1,
            "error_type": type(exc).__name__,
            "message": str(exc),
        }


def _run_forbidden_import_check(project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root)
    violations: list[dict[str, Any]] = []
    scanned_files = 0
    for directory_name in _FORBIDDEN_IMPORT_SCAN_DIRS:
        directory = root / directory_name
        if not directory.exists():
            continue
        for path in sorted(directory.rglob("*.py")):
            scanned_files += 1
            for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
                if not _IMPORT_LINE_RE.match(line):
                    continue
                for pattern, aliases in _FORBIDDEN_IMPORT_ALIASES.items():
                    if any(alias in line for alias in aliases):
                        violations.append(
                            {
                                "path": str(path.relative_to(root)),
                                "line": line_number,
                                "pattern": pattern,
                                "text": line.strip(),
                            }
                        )

    return {
        "name": "forbidden_import_check",
        "kind": "python_scan",
        "returncode": 1 if violations else 0,
        "scanned_files": scanned_files,
        "forbidden_patterns": list(FORBIDDEN_IMPORT_PATTERNS),
        "violations": violations,
    }


def _run_architecture_static_check(project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root)
    violations: list[dict[str, Any]] = []
    scanned_files: set[Path] = set()

    data_root = root / "src" / "model_explorer" / "data"
    if data_root.exists():
        for path in sorted(data_root.rglob("*.py")):
            scanned_files.add(path.resolve())
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in tree.body:
                if _is_policy_import(node):
                    violations.append(
                        {
                            "rule": "no_data_to_policy_static_import",
                            "path": str(path.relative_to(root)),
                            "line": node.lineno,
                            "text": _import_text(node),
                        }
                    )

    decision_root = root / "src" / "model_explorer" / "decision"
    if decision_root.exists():
        for path in sorted(decision_root.rglob("*.py")):
            scanned_files.add(path.resolve())
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in tree.body:
                if _is_policy_import(node):
                    violations.append(
                        {
                            "rule": "no_decision_to_policy_static_import",
                            "path": str(path.relative_to(root)),
                            "line": node.lineno,
                            "text": _import_text(node),
                        }
                    )

    source_root = root / "src" / "model_explorer"
    if source_root.exists():
        for path in sorted(source_root.rglob("*.py")):
            relative_path = path.relative_to(root).as_posix()
            scanned_files.add(path.resolve())
            module_name = _module_name_for_source_path(root, path)
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in _module_scope_statements(tree):
                if _is_dynamic_globals_all_assignment(node):
                    violations.append(
                        {
                            "rule": "no_dynamic_globals_all",
                            "path": relative_path,
                            "line": node.lineno,
                            "text": _assignment_text(node),
                        }
                    )
            if relative_path in _LEGACY_IMPORT_ALLOWLIST:
                continue
            for node in ast.walk(tree):
                for target in _legacy_import_targets(
                    node,
                    current_module=module_name,
                    current_is_package=path.name == "__init__.py",
                ):
                    violations.append(
                        {
                            "rule": "no_business_code_legacy_import",
                            "path": relative_path,
                            "line": node.lineno,
                            "text": _import_text(node),
                            "target": target,
                        }
                    )
                for target, imported_name in _private_import_targets(
                    node,
                    current_module=module_name,
                    current_is_package=path.name == "__init__.py",
                    target_modules=_TARGET_FACADE_MODULES,
                ):
                    violations.append(
                        {
                            "rule": "no_target_facade_private_production_import",
                            "path": relative_path,
                            "line": node.lineno,
                            "text": f"from {target} import {imported_name}",
                            "target": target,
                        }
                    )
                if not isinstance(node, ast.ImportFrom):
                    continue
                imported_module = _resolve_import_from_module(
                    module_name,
                    node,
                    current_is_package=path.name == "__init__.py",
                )
                if imported_module not in _RUNNER_IMPORT_TARGET_MODULES:
                    continue
                for alias in node.names:
                    if not alias.name.startswith("_"):
                        continue
                    violations.append(
                        {
                            "rule": "no_business_code_private_runner_import",
                            "path": relative_path,
                            "line": node.lineno,
                            "text": f"from {imported_module} import {alias.name}",
                            "target": imported_module,
                        }
                    )

    for relative_path in (*_FACADE_FILES, *_IMPL_FILES):
        path = root / relative_path
        if not path.exists():
            violations.append(
                {
                    "rule": "compatibility_module_exists",
                    "path": relative_path,
                    "line": None,
                    "text": "missing compatibility module",
                }
            )
            continue
        line_count = len(path.read_text(encoding="utf-8").splitlines())
        if line_count > _FACADE_LINE_LIMIT:
            violations.append(
                {
                    "rule": "compatibility_module_line_limit",
                    "path": relative_path,
                    "line": None,
                    "text": f"{line_count} lines > {_FACADE_LINE_LIMIT}",
                }
            )

    for relative_path, limit in _RUNNER_LINE_LIMITS.items():
        path = root / relative_path
        if not path.exists():
            violations.append(
                {
                    "rule": "runner_module_exists",
                    "path": relative_path,
                    "line": None,
                    "text": "missing runner module",
                }
            )
            continue
        line_count = len(path.read_text(encoding="utf-8").splitlines())
        if line_count > limit:
            violations.append(
                {
                    "rule": "runner_module_line_limit",
                    "path": relative_path,
                    "line": None,
                    "text": f"{line_count} lines > {limit}",
                }
            )

    for runner_module, split_paths in _RUNNER_SPLIT_MODULES.items():
        for relative_path in split_paths:
            path = root / relative_path
            if not path.exists():
                violations.append(
                    {
                        "rule": "split_module_exists",
                        "path": relative_path,
                        "line": None,
                        "text": "missing split module",
                    }
                )
                continue
            scanned_files.add(path.resolve())
            module_name = _module_name_for_source_path(root, path)
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                for target in _static_import_targets(
                    node,
                    current_module=module_name,
                    current_is_package=path.name == "__init__.py",
                    target_modules={runner_module},
                ):
                    violations.append(
                        {
                            "rule": "no_split_module_runner_import",
                            "path": relative_path,
                            "line": node.lineno,
                            "text": _import_text(node),
                            "target": target,
                        }
                    )

    for relative_path in _TARGET_FACADE_FILES:
        path = root / relative_path
        if not path.exists():
            violations.append(
                {
                    "rule": "target_facade_exists",
                    "path": relative_path,
                    "line": None,
                    "text": "missing target facade",
                }
            )
            continue
        line_count = len(path.read_text(encoding="utf-8").splitlines())
        if line_count > _TARGET_FACADE_LINE_LIMIT:
            violations.append(
                {
                    "rule": "target_facade_line_limit",
                    "path": relative_path,
                    "line": None,
                    "text": f"{line_count} lines > {_TARGET_FACADE_LINE_LIMIT}",
                }
            )

    for relative_path in _TARGET_SPLIT_MODULES:
        path = root / relative_path
        if not path.exists():
            violations.append(
                {
                    "rule": "target_split_module_exists",
                    "path": relative_path,
                    "line": None,
                    "text": "missing target split module",
                }
            )
            continue
        line_count = len(path.read_text(encoding="utf-8").splitlines())
        if line_count > _TARGET_SPLIT_MODULE_LINE_LIMIT:
            violations.append(
                {
                    "rule": "target_split_module_line_limit",
                    "path": relative_path,
                    "line": None,
                    "text": f"{line_count} lines > {_TARGET_SPLIT_MODULE_LINE_LIMIT}",
                }
            )
        scanned_files.add(path.resolve())
        module_name = _module_name_for_source_path(root, path)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        forbidden_targets = set(_TARGET_SPLIT_MODULE_IMPORT_TARGETS[relative_path])
        for node in ast.walk(tree):
            for target in _static_import_targets(
                node,
                current_module=module_name,
                current_is_package=path.name == "__init__.py",
                target_modules=forbidden_targets,
            ):
                violations.append(
                    {
                        "rule": "no_target_split_module_forbidden_import",
                        "path": relative_path,
                        "line": node.lineno,
                        "text": _import_text(node),
                        "target": target,
                    }
                )

    for relative_path, limit in _GIANT_TEST_LINE_LIMITS.items():
        path = root / relative_path
        if not path.exists():
            continue
        line_count = len(path.read_text(encoding="utf-8").splitlines())
        if line_count > limit:
            violations.append(
                {
                    "rule": "giant_test_line_limit",
                    "path": relative_path,
                    "line": None,
                    "text": f"{line_count} lines > {limit}",
                }
            )

    for (relative_path, function_name), limit in _FUNCTION_LINE_LIMITS.items():
        path = root / relative_path
        if not path.exists():
            violations.append(
                {
                    "rule": "function_line_target_missing",
                    "path": relative_path,
                    "line": None,
                    "function": function_name,
                    "limit": limit,
                    "text": f"missing function target {function_name}",
                }
            )
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        function_defs = _named_function_defs(tree, function_name)
        if not function_defs:
            violations.append(
                {
                    "rule": "function_line_target_missing",
                    "path": relative_path,
                    "line": None,
                    "function": function_name,
                    "limit": limit,
                    "text": f"missing function target {function_name}",
                }
            )
            continue
        for node in function_defs:
            if node.end_lineno is None:
                continue
            line_count = node.end_lineno - node.lineno + 1
            if line_count <= limit:
                continue
            violations.append(
                {
                    "rule": "function_line_limit",
                    "path": relative_path,
                    "line": node.lineno,
                    "function": function_name,
                    "limit": limit,
                    "line_count": line_count,
                    "text": f"{function_name} has {line_count} lines > {limit}",
                }
            )

    tests_root = root / "tests"
    if tests_root.exists():
        for path in sorted(tests_root.rglob("*.py")):
            scanned_files.add(path.resolve())
            relative_path = path.relative_to(root).as_posix()
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module in _TARGET_FACADE_MODULES:
                    for alias in node.names:
                        if not alias.name.startswith("_"):
                            continue
                        violations.append(
                            {
                                "rule": "no_target_facade_private_test_import",
                                "path": relative_path,
                                "line": node.lineno,
                                "text": f"from {node.module} import {alias.name}",
                                "target": node.module,
                            }
                        )
                if not isinstance(node, ast.ImportFrom) or node.module not in _PRIVATE_IMPORT_TARGET_MODULES:
                    continue
                for alias in node.names:
                    if not alias.name.startswith("_"):
                        continue
                    violations.append(
                        {
                            "rule": "no_private_legacy_test_imports",
                            "path": relative_path,
                            "line": node.lineno,
                            "text": f"from {node.module} import {alias.name}",
                        }
                    )

    return {
        "name": "architecture_static_check",
        "kind": "python_scan",
        "returncode": 1 if violations else 0,
        "scanned_files": len(scanned_files),
        "facade_line_limit": _FACADE_LINE_LIMIT,
        "impl_line_limit": _FACADE_LINE_LIMIT,
        "runner_line_limits": dict(_RUNNER_LINE_LIMITS),
        "target_facade_line_limit": _TARGET_FACADE_LINE_LIMIT,
        "target_facades": list(_TARGET_FACADE_FILES),
        "target_split_module_line_limit": _TARGET_SPLIT_MODULE_LINE_LIMIT,
        "target_split_modules": list(_TARGET_SPLIT_MODULES),
        "giant_test_line_limits": dict(_GIANT_TEST_LINE_LIMITS),
        "no_dynamic_globals_all_root": "src/model_explorer",
        "function_line_limits": {
            f"{relative_path}::{function_name}": limit
            for (relative_path, function_name), limit in _FUNCTION_LINE_LIMITS.items()
        },
        "violations": violations,
    }


def _private_import_targets(
    node: ast.stmt,
    *,
    current_module: str,
    current_is_package: bool,
    target_modules: set[str] | frozenset[str],
) -> list[tuple[str, str]]:
    if not isinstance(node, ast.ImportFrom):
        return []

    module = _resolve_import_from_module(
        current_module,
        node,
        current_is_package=current_is_package,
    )
    if module not in target_modules:
        return []
    return [(module, alias.name) for alias in node.names if alias.name.startswith("_")]


def _is_dynamic_globals_all_assignment(node: ast.AST) -> bool:
    if not isinstance(node, (ast.Assign, ast.AnnAssign)):
        return False
    if not _assignment_targets_name(node, "__all__"):
        return False
    value = node.value
    if value is None:
        return False
    return any(
        isinstance(child, ast.Call)
        and isinstance(child.func, ast.Name)
        and child.func.id == "globals"
        for child in ast.walk(value)
    )


def _module_scope_statements(tree: ast.Module) -> list[ast.stmt]:
    statements: list[ast.stmt] = []

    def visit(node: ast.AST) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return
        if isinstance(node, ast.stmt):
            statements.append(node)
        for child in ast.iter_child_nodes(node):
            visit(child)

    for node in tree.body:
        visit(node)
    return statements


def _assignment_targets_name(node: ast.Assign | ast.AnnAssign, name: str) -> bool:
    if isinstance(node, ast.Assign):
        return any(isinstance(target, ast.Name) and target.id == name for target in node.targets)
    return isinstance(node.target, ast.Name) and node.target.id == name


def _named_function_defs(tree: ast.AST, function_name: str) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name
    ]


def _assignment_text(node: ast.AST) -> str:
    if isinstance(node, (ast.Assign, ast.AnnAssign)) and _assignment_targets_name(node, "__all__"):
        return "__all__ = <dynamic globals export>"
    return type(node).__name__


def _is_policy_import(node: ast.stmt) -> bool:
    if isinstance(node, ast.ImportFrom):
        module = node.module or ""
        return module.startswith("model_explorer.policy") or (node.level > 0 and module.startswith("policy"))
    if isinstance(node, ast.Import):
        return any(alias.name.startswith("model_explorer.policy") for alias in node.names)
    return False


def _module_name_for_source_path(root: Path, path: Path) -> str:
    source_root = root / "src"
    relative = path.relative_to(source_root).with_suffix("")
    parts = list(relative.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _legacy_import_targets(
    node: ast.stmt,
    *,
    current_module: str,
    current_is_package: bool,
) -> list[str]:
    return _static_import_targets(
        node,
        current_module=current_module,
        current_is_package=current_is_package,
        target_modules=_LEGACY_IMPORT_TARGET_MODULES,
    )


def _static_import_targets(
    node: ast.stmt,
    *,
    current_module: str,
    current_is_package: bool,
    target_modules: set[str],
) -> list[str]:
    if isinstance(node, ast.Import):
        return [
            alias.name
            for alias in node.names
            if alias.name in target_modules
        ]
    if not isinstance(node, ast.ImportFrom):
        return []

    module = _resolve_import_from_module(
        current_module,
        node,
        current_is_package=current_is_package,
    )
    targets: list[str] = []
    if module in target_modules:
        targets.append(module)
    for alias in node.names:
        candidate = f"{module}.{alias.name}" if module else alias.name
        if candidate in target_modules:
            targets.append(candidate)
    return targets


def _resolve_import_from_module(
    current_module: str,
    node: ast.ImportFrom,
    *,
    current_is_package: bool,
) -> str:
    if node.level == 0:
        return node.module or ""

    parts = current_module.split(".")
    package_parts = parts if current_is_package else parts[:-1]
    keep = max(0, len(package_parts) - node.level + 1)
    resolved_parts = package_parts[:keep]
    if node.module:
        resolved_parts.extend(node.module.split("."))
    return ".".join(resolved_parts)


def _import_text(node: ast.stmt) -> str:
    if isinstance(node, ast.ImportFrom):
        names = ", ".join(alias.name for alias in node.names)
        dots = "." * node.level
        return f"from {dots}{node.module or ''} import {names}"
    if isinstance(node, ast.Import):
        return "import " + ", ".join(alias.name for alias in node.names)
    return type(node).__name__


def _tail(text: str, *, max_lines: int = 20) -> str:
    lines = text.splitlines()
    return "\n".join(lines[-max_lines:])
