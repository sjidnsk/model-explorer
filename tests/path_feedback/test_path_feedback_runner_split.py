from __future__ import annotations

import ast
import importlib
import os
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[3]
MODEL_ROOT = REPO_ROOT / "model-explorer"
POLICY_ROOT = MODEL_ROOT / "src" / "model_explorer" / "policy"


def _imports_runner(path: Path) -> list[int]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    lines: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        module = node.module or ""
        if module.endswith("path_feedback_runner"):
            lines.append(node.lineno)
    return lines


def test_path_feedback_runner_is_orchestration_sized() -> None:
    runner = POLICY_ROOT / "path_feedback_runner.py"

    assert len(runner.read_text(encoding="utf-8").splitlines()) <= 800


def test_path_feedback_split_modules_do_not_import_runner() -> None:
    split_modules = [
        POLICY_ROOT / "path_feedback_manifest.py",
        POLICY_ROOT / "path_feedback_summary.py",
        POLICY_ROOT / "path_feedback_reports.py",
        POLICY_ROOT / "path_feedback_diagnostics.py",
        POLICY_ROOT / "path_feedback_artifacts.py",
        POLICY_ROOT / "feedback_selection.py",
    ]

    violations = {
        path.name: _imports_runner(path)
        for path in split_modules
        if _imports_runner(path)
    }

    assert violations == {}


def test_legacy_path_feedback_facade_keeps_old_imports() -> None:
    from model_explorer.policy.path_feedback_impl import PathFeedbackManifest
    from model_explorer.policy.path_feedback import run_path_feedback_manifest

    facade = importlib.import_module("model_explorer.policy.path_feedback")
    impl = importlib.import_module("model_explorer.policy.path_feedback_impl")

    assert callable(run_path_feedback_manifest)
    assert callable(getattr(facade, "_selected_after_feedback"))
    assert callable(getattr(impl, "run_path_feedback_manifest"))
    assert callable(getattr(impl, "_selected_after_feedback"))
    assert PathFeedbackManifest.__name__ == "PathFeedbackManifest"


def test_path_feedback_manifest_import_is_schema_load_light() -> None:
    env = dict(os.environ)
    src_path = str(MODEL_ROOT / "src")
    env["PYTHONPATH"] = src_path if not env.get("PYTHONPATH") else src_path + os.pathsep + env["PYTHONPATH"]
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "import model_explorer.policy.path_feedback_manifest; "
                "print('model_explorer.policy.planning_routes' in sys.modules)"
            ),
        ],
        cwd=MODEL_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False"


def test_path_feedback_impl_all_does_not_leak_temporary_names() -> None:
    impl = importlib.import_module("model_explorer.policy.path_feedback_impl")

    leaked_names = {"Any", "Path", "json", "_MODULES", "_module", "_module_name", "_name"}

    assert leaked_names.isdisjoint(set(impl.__all__))
    assert "run_path_feedback_manifest" in impl.__all__
    assert "_selected_after_feedback" in impl.__all__
