"""Experiment report compatibility hooks."""

from .experiment_impl import _markdown_report as render_experiment_markdown

__all__ = ["render_experiment_markdown"]
