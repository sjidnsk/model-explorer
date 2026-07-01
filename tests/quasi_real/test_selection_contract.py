from __future__ import annotations

import ast
import importlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
QUASI_REAL = ROOT / "src" / "model_explorer" / "experiments" / "quasi_real_matrix"

MOVED_SYMBOLS = {
    "model_explorer.experiments.quasi_real_matrix.quality_gates": (
        "_selection_quality_gates",
        "_append_selection_min_violation",
        "_mask_stress_coverage",
    ),
    "model_explorer.experiments.quasi_real_matrix.decision_diagnostics": (
        "_decision_diagnostics_summary",
        "_architecture_agreement_matrix",
        "_baseline_agreement_summary",
        "_per_group_disagreement_summary",
        "_all_architectures_identical",
        "_apply_decision_signal_guards",
        "_sample_discriminativeness_summary",
    ),
    "model_explorer.experiments.quasi_real_matrix.architecture_selection": (
        "_architecture_selection_summary",
        "_selection_composite_score",
        "_selection_decision",
        "_per_group_architecture_winners",
        "_held_out_test_audit",
    ),
    "model_explorer.experiments.quasi_real_matrix.stability": (
        "_stability_summary",
        "_manifest_architectures",
        "_manifest_seeds",
        "_architecture_run_count",
    ),
}

PUBLIC_ALIASES = {
    "_selection_quality_gates": "selection_quality_gates",
    "_append_selection_min_violation": "append_selection_min_violation",
    "_mask_stress_coverage": "mask_stress_coverage",
    "_decision_diagnostics_summary": "decision_diagnostics_summary",
    "_architecture_agreement_matrix": "architecture_agreement_matrix",
    "_baseline_agreement_summary": "baseline_agreement_summary",
    "_per_group_disagreement_summary": "per_group_disagreement_summary",
    "_all_architectures_identical": "all_architectures_identical",
    "_apply_decision_signal_guards": "apply_decision_signal_guards",
    "_sample_discriminativeness_summary": "sample_discriminativeness_summary",
    "_architecture_selection_summary": "architecture_selection_summary",
    "_selection_composite_score": "selection_composite_score",
    "_selection_decision": "selection_decision",
    "_per_group_architecture_winners": "per_group_architecture_winners",
    "_held_out_test_audit": "held_out_test_audit",
    "_stability_summary": "stability_summary",
    "_manifest_architectures": "manifest_architectures",
    "_manifest_seeds": "manifest_seeds",
    "_architecture_run_count": "architecture_run_count",
}


def test_quasi_real_selection_split_modules_import_and_facade_exports_are_explicit() -> None:
    selection = importlib.import_module("model_explorer.experiments.quasi_real_matrix.selection")

    assert len((QUASI_REAL / "selection.py").read_text(encoding="utf-8").splitlines()) <= 250

    for module_name, private_names in MOVED_SYMBOLS.items():
        module = importlib.import_module(module_name)
        for private_name in private_names:
            public_name = PUBLIC_ALIASES[private_name]
            assert private_name in selection.__all__
            assert public_name in selection.__all__
            assert getattr(selection, private_name) is getattr(module, private_name)
            assert getattr(selection, public_name) is getattr(module, private_name)

    leaked_names = {"Any", "isfinite", "annotations"}
    assert leaked_names.isdisjoint(set(selection.__all__))
    assert leaked_names.isdisjoint(vars(selection))


def test_quasi_real_private_compat_exports_point_at_new_modules() -> None:
    evaluation_matrix_impl = importlib.import_module(
        "model_explorer.experiments.quasi_real_matrix.evaluation_matrix_impl"
    )
    evaluation_matrix = importlib.import_module("model_explorer.data.evaluation_matrix")

    for module_name, private_names in MOVED_SYMBOLS.items():
        module = importlib.import_module(module_name)
        for private_name in private_names:
            assert getattr(evaluation_matrix_impl, private_name) is getattr(module, private_name)
            assert getattr(evaluation_matrix, private_name) is getattr(module, private_name)


def test_quasi_real_selection_facade_no_longer_defines_moved_private_helpers() -> None:
    tree = ast.parse((QUASI_REAL / "selection.py").read_text(encoding="utf-8"))
    defined = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    moved_private_names = {
        private_name
        for private_names in MOVED_SYMBOLS.values()
        for private_name in private_names
    }

    assert defined.isdisjoint(moved_private_names)
