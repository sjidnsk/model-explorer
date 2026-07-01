from __future__ import annotations

import ast
import importlib
from pathlib import Path


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

    assert callable(run_path_feedback_manifest)
    assert callable(getattr(facade, "_selected_after_feedback"))
    assert PathFeedbackManifest.__name__ == "PathFeedbackManifest"
