"""Architecture selection helpers for quasi-real evaluation matrices."""

from .evaluation_matrix_impl import (
    _architecture_selection_summary as architecture_selection_summary,
    _decision_diagnostics_summary as decision_diagnostics_summary,
    _sample_discriminativeness_summary as sample_discriminativeness_summary,
    _selection_decision as selection_decision,
)

__all__ = [
    "architecture_selection_summary",
    "decision_diagnostics_summary",
    "sample_discriminativeness_summary",
    "selection_decision",
]
