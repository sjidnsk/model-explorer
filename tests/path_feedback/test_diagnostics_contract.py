from __future__ import annotations

import importlib


DIAGNOSTIC_MODULES = {
    "aggregate": "model_explorer.policy.path_feedback_diagnostic_aggregate",
    "interpretation": "model_explorer.policy.path_feedback_diagnostic_interpretation",
    "backend": "model_explorer.policy.path_feedback_backend_diagnostics",
    "candidate_audits": "model_explorer.policy.path_feedback_candidate_audits",
}

CORE_LEGACY_SYMBOLS = (
    "_diagnostic_aggregate",
    "_diagnostic_interpretation_summary",
    "_iris_diagnostics",
    "_gcs_candidate_audit",
    "diagnostic_aggregate",
    "diagnostic_interpretation_summary",
    "iris_diagnostics",
    "gcs_candidate_audit",
)


def test_path_feedback_diagnostic_split_modules_import() -> None:
    for module_name in DIAGNOSTIC_MODULES.values():
        module = importlib.import_module(module_name)
        assert isinstance(module.__all__, tuple)


def test_path_feedback_diagnostics_facade_keeps_core_legacy_symbols() -> None:
    facade = importlib.import_module("model_explorer.policy.path_feedback_diagnostics")

    for symbol in CORE_LEGACY_SYMBOLS:
        assert hasattr(facade, symbol)
        assert symbol in facade.__all__


def test_path_feedback_impl_legacy_diagnostic_symbols_point_to_split_modules() -> None:
    aggregate = importlib.import_module(DIAGNOSTIC_MODULES["aggregate"])
    interpretation = importlib.import_module(DIAGNOSTIC_MODULES["interpretation"])
    backend = importlib.import_module(DIAGNOSTIC_MODULES["backend"])
    candidate_audits = importlib.import_module(DIAGNOSTIC_MODULES["candidate_audits"])
    impl = importlib.import_module("model_explorer.policy.path_feedback_impl")

    assert impl._diagnostic_aggregate is aggregate._diagnostic_aggregate
    assert impl._diagnostic_interpretation_summary is interpretation._diagnostic_interpretation_summary
    assert impl._iris_diagnostics is backend._iris_diagnostics
    assert impl._gcs_candidate_audit is candidate_audits._gcs_candidate_audit


def test_path_feedback_diagnostics_all_does_not_leak_temporary_names() -> None:
    facade = importlib.import_module("model_explorer.policy.path_feedback_diagnostics")
    leaked_names = {"Any", "Counter", "defaultdict", "annotations"}

    assert leaked_names.isdisjoint(set(facade.__all__))
