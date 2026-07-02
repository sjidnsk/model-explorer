from __future__ import annotations

from typing import Any

from .path_feedback_reports import _legacy_render_path_feedback_markdown


def render_markdown_sections(summary: dict[str, Any]) -> list[str]:
    return _legacy_render_path_feedback_markdown(summary).split("\n")


__all__ = ("render_markdown_sections",)
