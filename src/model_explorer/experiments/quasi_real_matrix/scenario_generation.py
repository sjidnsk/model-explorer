"""Scenario generation orchestration for quasi-real evaluation matrices."""

from .runner import (
    _experiment_manifest_payload as experiment_manifest_payload,
    _roi_specs as roi_specs,
)

__all__ = [
    "experiment_manifest_payload",
    "roi_specs",
]
