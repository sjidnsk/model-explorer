"""Report helpers for quasi-real evaluation matrices."""

from .evaluation_matrix_impl import (
    _inspection_summary as inspection_summary,
    _markdown_report as render_quasi_real_matrix_markdown,
    _run_summary as run_summary,
)

__all__ = [
    "inspection_summary",
    "render_quasi_real_matrix_markdown",
    "run_summary",
]
