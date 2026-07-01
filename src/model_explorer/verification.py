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
_PRIVATE_IMPORT_TARGET_MODULES = {
    "model_explorer.policy.planning",
    "model_explorer.policy.path_feedback",
    "model_explorer.policy.experiment",
    "model_explorer.data.evaluation_matrix",
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
    scanned_files = 0

    data_root = root / "src" / "model_explorer" / "data"
    if data_root.exists():
        for path in sorted(data_root.rglob("*.py")):
            scanned_files += 1
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
            scanned_files += 1
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
            if relative_path in _LEGACY_IMPORT_ALLOWLIST:
                continue
            scanned_files += 1
            module_name = _module_name_for_source_path(root, path)
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in tree.body:
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

    tests_root = root / "tests"
    if tests_root.exists():
        for path in sorted(tests_root.rglob("*.py")):
            scanned_files += 1
            relative_path = path.relative_to(root).as_posix()
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
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
        "scanned_files": scanned_files,
        "facade_line_limit": _FACADE_LINE_LIMIT,
        "impl_line_limit": _FACADE_LINE_LIMIT,
        "violations": violations,
    }


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
    if isinstance(node, ast.Import):
        return [
            alias.name
            for alias in node.names
            if alias.name in _LEGACY_IMPORT_TARGET_MODULES
        ]
    if not isinstance(node, ast.ImportFrom):
        return []

    module = _resolve_import_from_module(
        current_module,
        node,
        current_is_package=current_is_package,
    )
    targets: list[str] = []
    if module in _LEGACY_IMPORT_TARGET_MODULES:
        targets.append(module)
    for alias in node.names:
        candidate = f"{module}.{alias.name}" if module else alias.name
        if candidate in _LEGACY_IMPORT_TARGET_MODULES:
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
