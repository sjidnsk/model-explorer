from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "model_explorer"
QUASI_REAL = SRC / "experiments" / "quasi_real_matrix"


def test_quasi_real_runner_is_small_orchestration_layer() -> None:
    runner_path = QUASI_REAL / "runner.py"

    assert len(runner_path.read_text(encoding="utf-8").splitlines()) <= 700


def test_quasi_real_split_modules_do_not_import_runner() -> None:
    violations: list[str] = []
    split_modules = {
        "manifest.py",
        "metrics.py",
        "reports.py",
        "scenario_generation.py",
        "selection.py",
        "quality_gates.py",
        "decision_diagnostics.py",
        "architecture_selection.py",
        "stability.py",
    }
    for path in sorted(QUASI_REAL.glob("*.py")):
        if path.name not in split_modules:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module == "runner" and node.level >= 1:
                    violations.append(f"{path.name}:{node.lineno}")
                if module.endswith("quasi_real_matrix.runner"):
                    violations.append(f"{path.name}:{node.lineno}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.endswith("quasi_real_matrix.runner"):
                        violations.append(f"{path.name}:{node.lineno}")

    assert violations == []


def test_quasi_real_compatibility_exports_are_explicit_and_clean() -> None:
    from model_explorer.data import evaluation_matrix
    from model_explorer.experiments.quasi_real_matrix import evaluation_matrix_impl
    from model_explorer.experiments.quasi_real_matrix import runner

    forbidden = {"Any", "Path", "json", "dataclass", "_runner", "_impl", "_name"}
    for module in (runner, evaluation_matrix_impl, evaluation_matrix):
        exported = set(module.__all__)
        assert not (exported & forbidden)

    assert "run_quasi_real_evaluation_manifest" in evaluation_matrix.__all__
    assert evaluation_matrix.run_quasi_real_evaluation_manifest is runner.run_quasi_real_evaluation_manifest
    assert evaluation_matrix.load_quasi_real_evaluation_manifest is evaluation_matrix_impl.load_quasi_real_evaluation_manifest
    assert evaluation_matrix_impl._selection_decision is not None
    assert evaluation_matrix._selection_decision is evaluation_matrix_impl._selection_decision


def test_quasi_real_metric_helpers_live_in_metrics_module() -> None:
    from model_explorer.data import evaluation_matrix
    from model_explorer.experiments.quasi_real_matrix import evaluation_matrix_impl
    from model_explorer.experiments.quasi_real_matrix import metrics
    from model_explorer.experiments.quasi_real_matrix import selection

    public_helpers = {
        "architecture_nested_metric_summary": "_architecture_nested_metric_summary",
        "per_group_action_outcomes": "_per_group_action_outcomes",
        "iter_policy_nested_sections": "_iter_policy_nested_sections",
        "run_selection_metric": "_run_selection_metric",
        "evaluation_metric": "_evaluation_metric",
    }
    for public_name, private_name in public_helpers.items():
        assert public_name in metrics.__all__
        assert private_name in metrics.__all__
        assert getattr(metrics, public_name) is getattr(metrics, private_name)
        assert getattr(selection, private_name) is getattr(metrics, private_name)
        assert getattr(evaluation_matrix_impl, private_name) is getattr(metrics, private_name)
        assert getattr(evaluation_matrix, private_name) is getattr(metrics, private_name)

    selection_path = QUASI_REAL / "selection.py"
    tree = ast.parse(selection_path.read_text(encoding="utf-8"))
    defined_in_selection = {
        node.name for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    assert not (set(public_helpers.values()) & defined_in_selection)
