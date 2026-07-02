from __future__ import annotations

import ast
import importlib
import unittest
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


def test_architecture_selection_summary_stays_within_split_line_budget() -> None:
    tree = ast.parse((QUASI_REAL / "architecture_selection.py").read_text(encoding="utf-8"))
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}

    summary = functions["_architecture_selection_summary"]

    assert summary.end_lineno is not None
    assert summary.end_lineno - summary.lineno + 1 <= 180


class QuasiRealSelectionBehaviorTests(unittest.TestCase):
    def test_selection_decision_is_inconclusive_when_margin_is_within_seed_variance(self):
        from model_explorer.experiments.quasi_real_matrix.architecture_selection import selection_decision

        decision = selection_decision(
            {
                "mlp_v1": {"count": 3, "mean": 0.50, "std": 0.10, "min": 0.40, "max": 0.60},
                "mlp_missing_v1": {"count": 3, "mean": 0.54, "std": 0.08, "min": 0.46, "max": 0.62},
                "candidate_attention_v1": {"count": 3, "mean": 0.49, "std": 0.07, "min": 0.42, "max": 0.56},
            },
            metric="torch_policy.final_coverage_rate",
            mode="max",
            uncertainty_multiplier=1.0,
        )

        self.assertEqual(decision["status"], "inconclusive")
        self.assertIsNone(decision["recommended_architecture"])
        self.assertEqual(decision["decision"], "inconclusive")
        self.assertIn("within seed variance", decision["reason"])

    def test_sample_discriminativeness_warns_when_candidate_spread_is_low(self):
        from model_explorer.experiments.quasi_real_matrix.decision_diagnostics import sample_discriminativeness_summary

        runs = [
            {
                "architecture": "mlp_v1",
                "seed": 1,
                "validation_evaluation": {
                    "per_scenario": [
                        {
                            "path": "/tmp/scenarios/validation/group-a/shared.json",
                            "group": "group-a",
                            "metrics": {
                                "torch_policy": {
                                    "sample_discriminativeness": {
                                        "candidate_coverage_spread": 0.0,
                                        "risk_spread": 0.0,
                                        "path_cost_spread": 0.0,
                                        "value_spread": 0.0,
                                        "oracle_vs_heuristic_action_disagreement_rate": 0.0,
                                    }
                                }
                            },
                        }
                    ]
                },
            }
        ]

        summary = sample_discriminativeness_summary(runs)

        self.assertIn("low_candidate_coverage_spread", summary["warnings"])
        self.assertIn("low_risk_spread", summary["warnings"])
        self.assertEqual(summary["status"], "warning")

    def test_decision_diagnostics_warn_when_policies_match_heuristic_and_actions_are_identical(self):
        from model_explorer.experiments.quasi_real_matrix.decision_diagnostics import decision_diagnostics_summary

        runs = [
            {
                "architecture": architecture,
                "seed": 7,
                "validation_evaluation": {
                    "per_scenario": [
                        {
                            "path": "/tmp/scenarios/validation/group-a/shared.json",
                            "group": "group-a",
                            "metrics": {
                                "torch_policy": {
                                    "action_diagnostics": [
                                        {
                                            "step_index": 0,
                                            "selected_cell": [2, 2],
                                            "selected_index": 1,
                                            "selected_action_mask_valid": True,
                                            "max_masked_action_probability": 0.0,
                                            "agrees_with_utility": False,
                                            "agrees_with_coverage_heuristic": True,
                                        }
                                    ]
                                }
                            },
                        }
                    ]
                },
            }
            for architecture in ("mlp_v1", "mlp_missing_v1", "candidate_attention_v1")
        ]

        diagnostics = decision_diagnostics_summary(
            runs,
            architectures=["mlp_v1", "mlp_missing_v1", "candidate_attention_v1"],
        )

        self.assertTrue(diagnostics["all_architectures_identical"])
        self.assertIn("all_architectures_identical", diagnostics["warnings"])
        self.assertIn("all_trained_policies_match_coverage_heuristic", diagnostics["warnings"])
        self.assertEqual(
            diagnostics["architecture_baseline_agreement"]["mlp_v1"]["coverage_heuristic_agreement_rate"],
            1.0,
        )
