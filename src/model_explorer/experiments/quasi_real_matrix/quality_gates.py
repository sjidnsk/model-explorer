from __future__ import annotations

from typing import Any

from .metrics import _int_value
from .stability import _architecture_run_count


def _selection_quality_gates(
    config: dict[str, Any],
    *,
    architectures: list[str],
    seeds: list[int],
    runs: list[dict[str, Any]],
    metric_stats: dict[str, dict[str, Any]],
    loss_stats: dict[str, dict[str, Any]],
    exception_counts: dict[str, int],
    dataset_summary: Any,
) -> dict[str, Any]:
    violations: list[dict[str, Any]] = []
    dataset_summary = dataset_summary if isinstance(dataset_summary, dict) else {}
    _append_selection_min_violation(violations, "min_seed_count", len(seeds), config.get("min_seed_count"))
    _append_selection_min_violation(
        violations,
        "min_architecture_count",
        len(architectures),
        config.get("min_architecture_count"),
    )
    _append_selection_min_violation(
        violations,
        "min_roi_group_count",
        _int_value(dataset_summary.get("roi_count")),
        config.get("min_roi_group_count"),
    )
    _append_selection_min_violation(
        violations,
        "min_unreachable_candidate_count",
        _int_value(dataset_summary.get("unreachable_candidate_count")),
        config.get("min_unreachable_candidate_count"),
    )
    _append_selection_min_violation(
        violations,
        "min_mask_stress_sample_count",
        _int_value(dataset_summary.get("mask_stress_sample_count")),
        config.get("min_mask_stress_sample_count"),
    )
    expected_runs_per_architecture = len(seeds)
    for architecture in architectures:
        run_count = _architecture_run_count(runs, architecture)
        if run_count < expected_runs_per_architecture:
            violations.append(
                {
                    "gate": "architecture_run_count",
                    "architecture": architecture,
                    "expected": expected_runs_per_architecture,
                    "actual": run_count,
                    "message": (
                        f"architecture_run_count expected >= {expected_runs_per_architecture}, "
                        f"actual {run_count} for {architecture}"
                    ),
                }
            )
        if exception_counts.get(architecture, 0) > 0:
            violations.append(
                {
                    "gate": "exception_count",
                    "architecture": architecture,
                    "expected": 0,
                    "actual": exception_counts.get(architecture, 0),
                    "message": f"exception_count expected 0 for {architecture}",
                }
            )
        if loss_stats.get(architecture, {}).get("count", 0) < run_count:
            violations.append(
                {
                    "gate": "finite_loss",
                    "architecture": architecture,
                    "expected": run_count,
                    "actual": loss_stats.get(architecture, {}).get("count", 0),
                    "message": f"finite_loss expected {run_count} finite losses for {architecture}",
                }
            )
        if metric_stats.get(architecture, {}).get("count", 0) < run_count:
            violations.append(
                {
                    "gate": "finite_selection_metric",
                    "architecture": architecture,
                    "expected": run_count,
                    "actual": metric_stats.get(architecture, {}).get("count", 0),
                    "message": f"finite_selection_metric expected {run_count} finite metrics for {architecture}",
                }
            )
    return {
        "status": "failed" if violations else "passed",
        "configured": {key: value for key, value in config.items() if key.startswith("min_") and value is not None},
        "violations": violations,
    }


def _append_selection_min_violation(
    violations: list[dict[str, Any]],
    gate: str,
    actual: int | float,
    expected: int | float | None,
) -> None:
    if expected is None:
        return
    if actual < expected:
        violations.append(
            {
                "gate": gate,
                "expected": expected,
                "actual": actual,
                "message": f"{gate} expected >= {expected}, actual {actual}",
            }
        )


def _mask_stress_coverage(dataset_summary: Any) -> dict[str, Any]:
    dataset_summary = dataset_summary if isinstance(dataset_summary, dict) else {}
    return {
        "unreachable_candidate_count": _int_value(dataset_summary.get("unreachable_candidate_count")),
        "padding_candidate_count": _int_value(dataset_summary.get("padding_candidate_count")),
        "missing_experimental_feature_candidate_count": _int_value(
            dataset_summary.get("missing_experimental_feature_candidate_count")
        ),
        "mask_stress_sample_count": _int_value(dataset_summary.get("mask_stress_sample_count")),
        "mask_stress_augmented": bool(dataset_summary.get("mask_stress_augmented", False)),
    }


selection_quality_gates = _selection_quality_gates
append_selection_min_violation = _append_selection_min_violation
mask_stress_coverage = _mask_stress_coverage

__all__ = [
    "selection_quality_gates",
    "append_selection_min_violation",
    "mask_stress_coverage",
    "_selection_quality_gates",
    "_append_selection_min_violation",
    "_mask_stress_coverage",
]
