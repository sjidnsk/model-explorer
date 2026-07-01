from __future__ import annotations

from math import isfinite
from pathlib import Path
from typing import Any

from .manifest import VALID_SPLITS


def _metric_value(metrics: dict[str, Any], metric: str) -> float:
    metric = metric.removeprefix("torch_policy.")
    current: Any = metrics
    for part in metric.split("."):
        if not isinstance(current, dict):
            current = None
            break
        current = current.get(part)
    value = current
    if value is None and metric == "final_coverage_rate":
        value = metrics.get("average_final_coverage_rate")
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    return numeric if isfinite(numeric) else 0.0


def _cell_tuple(value: Any) -> tuple[int, int] | None:
    if not isinstance(value, list) or len(value) != 2:
        return None
    try:
        return (int(value[0]), int(value[1]))
    except (TypeError, ValueError):
        return None


def _group_name_from_path(path: str) -> str:
    parts = Path(path).parts
    for split in VALID_SPLITS:
        if split not in parts:
            continue
        index = parts.index(split)
        if index + 1 < len(parts):
            return str(parts[index + 1])
    return "unknown"


def _training_runs(experiment: Any) -> list[dict[str, Any]]:
    if not isinstance(experiment, dict):
        return []
    training = experiment.get("training", {})
    runs = training.get("runs", []) if isinstance(training, dict) else []
    return [run for run in runs if isinstance(run, dict)] if isinstance(runs, list) else []


def _append_metric(target: dict[str, list[float]], metric: str, value: Any) -> None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return
    if isfinite(numeric):
        target.setdefault(metric, []).append(numeric)


def _numeric_summary(values: tuple[float, ...]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    average = sum(values) / len(values)
    variance = sum((value - average) ** 2 for value in values) / len(values)
    return {
        "count": len(values),
        "mean": average,
        "std": variance ** 0.5,
        "min": min(values),
        "max": max(values),
    }


def _int_value(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0



metric_value = _metric_value
cell_tuple = _cell_tuple
group_name_from_path = _group_name_from_path
training_runs = _training_runs
append_metric = _append_metric
numeric_summary = _numeric_summary
int_value = _int_value

_PUBLIC_EXPORTS = [
    "metric_value",
    "cell_tuple",
    "group_name_from_path",
    "training_runs",
    "append_metric",
    "numeric_summary",
    "int_value",
]
_PRIVATE_COMPAT_EXPORTS = [
    "_metric_value",
    "_cell_tuple",
    "_group_name_from_path",
    "_training_runs",
    "_append_metric",
    "_numeric_summary",
    "_int_value",
]
__all__ = [*_PUBLIC_EXPORTS, *_PRIVATE_COMPAT_EXPORTS]
