from __future__ import annotations

from collections.abc import Callable, Iterator
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


def _evaluation_policy_metrics(evaluation: Any, policy: str) -> dict[str, Any]:
    if not isinstance(evaluation, dict):
        return {}
    if "aggregate" in evaluation and isinstance(evaluation["aggregate"], dict):
        evaluation = evaluation["aggregate"]
    metrics = evaluation.get(policy) if isinstance(evaluation, dict) else None
    return metrics if isinstance(metrics, dict) else {}


def _architecture_nested_metric_summary(
    runs: list[dict[str, Any]],
    *,
    architectures: list[str],
    section: str,
) -> dict[str, dict[str, dict[str, float | int]]]:
    values: dict[str, dict[str, list[float]]] = {architecture: {} for architecture in architectures}
    for run in runs:
        if not isinstance(run, dict):
            continue
        architecture = str(run.get("architecture", "unknown"))
        architecture_values = values.setdefault(architecture, {})
        for _, nested in _iter_policy_nested_sections(run, section, evaluation_keys=("validation_evaluation",)):
            for metric, value in nested.items():
                _append_metric(architecture_values, str(metric), value)
    return {
        architecture: {
            metric: _numeric_summary(tuple(metric_values))
            for metric, metric_values in metrics.items()
        }
        for architecture, metrics in values.items()
    }


def _per_group_action_outcomes(
    runs: list[dict[str, Any]],
    *,
    architectures: list[str],
    uncertainty_multiplier: float,
    decision_fn: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if decision_fn is None:
        from .selection import _selection_decision as decision_fn

    group_values: dict[str, dict[str, dict[str, list[float]]]] = {}
    for run in runs:
        if not isinstance(run, dict):
            continue
        architecture = str(run.get("architecture", "unknown"))
        for context, regret_metrics in _iter_policy_nested_sections(
            run,
            "oracle_regret",
            evaluation_keys=("validation_evaluation",),
        ):
            group_name = str(context.get("group", "unknown"))
            architecture_values = group_values.setdefault(group_name, {}).setdefault(architecture, {})
            for metric, value in regret_metrics.items():
                _append_metric(architecture_values, str(metric), value)
        for context, action_metrics in _iter_policy_nested_sections(
            run,
            "action_sensitive_metrics",
            evaluation_keys=("validation_evaluation",),
        ):
            group_name = str(context.get("group", "unknown"))
            architecture_values = group_values.setdefault(group_name, {}).setdefault(architecture, {})
            for metric, value in action_metrics.items():
                _append_metric(architecture_values, str(metric), value)

    summary: dict[str, Any] = {}
    for group_name, architecture_values in group_values.items():
        architecture_stats = {
            architecture: {
                metric: _numeric_summary(tuple(metric_values))
                for metric, metric_values in metrics.items()
            }
            for architecture, metrics in architecture_values.items()
        }
        coverage_regret_stats = {
            architecture: metrics.get("coverage_regret", _numeric_summary(()))
            for architecture, metrics in architecture_stats.items()
        }
        decision = decision_fn(
            coverage_regret_stats,
            metric="coverage_regret",
            mode="min",
            uncertainty_multiplier=uncertainty_multiplier,
        )
        summary[group_name] = {
            **decision,
            "reason": (
                f"lower coverage_regret is better; {decision.get('reason', '')}"
                if decision.get("reason")
                else "lower coverage_regret is better"
            ),
            "architectures": {
                architecture: architecture_stats.get(architecture, {})
                for architecture in architectures
            },
        }
    return summary


def _iter_policy_nested_sections(
    run: dict[str, Any],
    section: str,
    *,
    evaluation_keys: tuple[str, ...],
) -> Iterator[tuple[dict[str, str], dict[str, Any]]]:
    for evaluation_key in evaluation_keys:
        evaluation = run.get(evaluation_key, {})
        if not isinstance(evaluation, dict):
            continue
        yielded_per_scenario = False
        per_scenario = evaluation.get("per_scenario", [])
        if isinstance(per_scenario, list):
            for scenario in per_scenario:
                if not isinstance(scenario, dict):
                    continue
                metrics = scenario.get("metrics", {})
                torch_policy = metrics.get("torch_policy", {}) if isinstance(metrics, dict) else {}
                nested = torch_policy.get(section) if isinstance(torch_policy, dict) else None
                if not isinstance(nested, dict):
                    continue
                path = str(scenario.get("path", ""))
                yielded_per_scenario = True
                yield {
                    "evaluation": evaluation_key,
                    "path": path,
                    "group": str(scenario.get("group") or _group_name_from_path(path)),
                }, nested
        if yielded_per_scenario:
            continue
        torch_policy = _evaluation_policy_metrics(evaluation, "torch_policy")
        nested = torch_policy.get(section) if isinstance(torch_policy, dict) else None
        if isinstance(nested, dict):
            yield {"evaluation": evaluation_key, "path": "", "group": "aggregate"}, nested


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


def _run_selection_metric(run: dict[str, Any], metric: str) -> Any:
    if metric in run:
        return run.get(metric)
    evaluation = run.get("validation_evaluation", {})
    return _evaluation_metric(evaluation, metric)


def _evaluation_metric(evaluation: Any, metric: str) -> Any:
    if not isinstance(evaluation, dict):
        return None
    if "aggregate" in evaluation and isinstance(evaluation["aggregate"], dict):
        evaluation = evaluation["aggregate"]
    parts = metric.split(".")
    if len(parts) == 1:
        return evaluation.get(parts[0]) if isinstance(evaluation, dict) else None
    current: Any = evaluation
    for part in parts:
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


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
evaluation_policy_metrics = _evaluation_policy_metrics
architecture_nested_metric_summary = _architecture_nested_metric_summary
per_group_action_outcomes = _per_group_action_outcomes
iter_policy_nested_sections = _iter_policy_nested_sections
cell_tuple = _cell_tuple
group_name_from_path = _group_name_from_path
training_runs = _training_runs
run_selection_metric = _run_selection_metric
evaluation_metric = _evaluation_metric
append_metric = _append_metric
numeric_summary = _numeric_summary
int_value = _int_value

_PUBLIC_EXPORTS = [
    "metric_value",
    "evaluation_policy_metrics",
    "architecture_nested_metric_summary",
    "per_group_action_outcomes",
    "iter_policy_nested_sections",
    "cell_tuple",
    "group_name_from_path",
    "training_runs",
    "run_selection_metric",
    "evaluation_metric",
    "append_metric",
    "numeric_summary",
    "int_value",
]
_PRIVATE_COMPAT_EXPORTS = [
    "_metric_value",
    "_evaluation_policy_metrics",
    "_architecture_nested_metric_summary",
    "_per_group_action_outcomes",
    "_iter_policy_nested_sections",
    "_cell_tuple",
    "_group_name_from_path",
    "_training_runs",
    "_run_selection_metric",
    "_evaluation_metric",
    "_append_metric",
    "_numeric_summary",
    "_int_value",
]
__all__ = [*_PUBLIC_EXPORTS, *_PRIVATE_COMPAT_EXPORTS]
