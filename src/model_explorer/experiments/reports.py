from __future__ import annotations

from typing import Any

from .evaluation import _comparison_from_evaluation
from .report_sections import (
    _loss_summary,
    render_architecture_deltas_section,
    render_baseline_comparison_section,
    render_baselines_section,
    render_benchmark_groups_section,
    render_dataset_summary_section,
    render_failure_scenarios_section,
    render_gate_summary_section,
    render_header_section,
    render_per_group_winners_section,
    render_policy_ranking_section,
    render_rollout_metrics_section,
    render_torch_policy_deltas_section,
    render_training_section,
)
from .selection import _baseline_deltas, _metric_value, _numeric_stats


def _daily_report_summary(summary: dict[str, Any], evaluation: dict[str, Any]) -> dict[str, Any]:
    comparison = _comparison_from_evaluation(evaluation)
    baseline_deltas = _baseline_deltas(comparison)
    return {
        "policy_ranking": _policy_ranking(comparison),
        "baseline_deltas": baseline_deltas,
        "architecture_deltas": _architecture_deltas(summary, baseline_deltas),
        "per_group_winners": _per_group_winners(evaluation),
        "failure_scenarios": _failure_scenarios(evaluation),
        "gate_summary": _gate_summary(summary.get("dataset_summary")),
    }


def _architecture_deltas(summary: dict[str, Any], baseline_deltas: dict[str, Any]) -> dict[str, Any]:
    training = summary.get("training")
    if not isinstance(training, dict):
        return {}
    runs = training.get("runs")
    if isinstance(runs, list):
        per_architecture: dict[str, Any] = {}
        for run in runs:
            if not isinstance(run, dict):
                continue
            architecture = run.get("architecture")
            deltas = run.get("baseline_deltas")
            if not architecture or not isinstance(deltas, dict) or not deltas:
                continue
            per_architecture.setdefault(str(architecture), deltas)
        if per_architecture:
            return per_architecture
    architecture = training.get("architecture")
    torch_deltas = baseline_deltas.get("torch_policy") if isinstance(baseline_deltas, dict) else None
    if not architecture or not isinstance(torch_deltas, dict):
        return {}
    return {str(architecture): torch_deltas}


def _policy_ranking(evaluation_comparison: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for policy_name, metrics in evaluation_comparison.items():
        if not isinstance(metrics, dict):
            continue
        rows.append(
            {
                "policy": str(policy_name),
                "final_coverage_rate": _metric_value(metrics, "final_coverage_rate"),
                "cumulative_coverage_rate_delta": _metric_value(metrics, "cumulative_coverage_rate_delta"),
                "total_path_cost": _metric_value(metrics, "total_path_cost"),
                "average_risk": _metric_value(metrics, "average_risk"),
                "failure_count": int(_metric_value(metrics, "failure_count")),
                "value_coverage": _metric_value(metrics, "value_coverage"),
            }
        )
    rows.sort(
        key=lambda item: (
            -float(item["final_coverage_rate"]),
            int(item["failure_count"]),
            float(item["total_path_cost"]),
            str(item["policy"]),
        )
    )
    for index, item in enumerate(rows, start=1):
        item["rank"] = index
    return rows


def _ordered_policy_names(evaluation_comparison: dict[str, Any]) -> tuple[str, ...]:
    preferred = ("utility", "coverage_heuristic", "feedback_aware", "torch_policy")
    names = [name for name in preferred if name in evaluation_comparison]
    names.extend(
        sorted(
            str(name)
            for name, metrics in evaluation_comparison.items()
            if name not in set(preferred) and isinstance(metrics, dict)
        )
    )
    return tuple(names)


def _per_group_winners(evaluation: dict[str, Any]) -> dict[str, Any]:
    groups = evaluation.get("groups") if isinstance(evaluation, dict) else None
    if not isinstance(groups, dict):
        return {}
    winners: dict[str, Any] = {}
    for group_name, group_evaluation in groups.items():
        if not isinstance(group_evaluation, dict):
            continue
        ranking = _policy_ranking(_comparison_from_evaluation(group_evaluation))
        winners[str(group_name)] = None if not ranking else ranking[0]
    return winners


def _failure_scenarios(evaluation: dict[str, Any]) -> list[dict[str, Any]]:
    per_scenario = evaluation.get("per_scenario") if isinstance(evaluation, dict) else None
    if not isinstance(per_scenario, list):
        return []
    failures: list[dict[str, Any]] = []
    for item in per_scenario:
        if not isinstance(item, dict):
            continue
        metrics = item.get("metrics")
        if not isinstance(metrics, dict):
            continue
        failed_policies: list[str] = []
        for policy_name, policy_metrics in metrics.items():
            if not isinstance(policy_metrics, dict):
                continue
            selected_cells = policy_metrics.get("selected_cells", [])
            has_no_selection = isinstance(selected_cells, list) and any(cell is None for cell in selected_cells)
            if _metric_value(policy_metrics, "failure_count") > 0 or has_no_selection:
                failed_policies.append(str(policy_name))
        if failed_policies:
            failures.append({"path": str(item.get("path", "")), "policies": failed_policies})
    return failures


def _gate_summary(dataset_summary: Any) -> dict[str, Any]:
    if not isinstance(dataset_summary, dict):
        return {"status": "not_configured", "warnings": [], "errors": [], "violation_count": 0}
    validation_gates = dataset_summary.get("validation_gates")
    violations = []
    status = "not_configured"
    if isinstance(validation_gates, dict):
        status = str(validation_gates.get("status", "unknown"))
        raw_violations = validation_gates.get("violations", [])
        violations = raw_violations if isinstance(raw_violations, list) else []
    return {
        "status": status,
        "warnings": list(dataset_summary.get("warnings", [])),
        "errors": list(dataset_summary.get("errors", [])),
        "violation_count": len(violations),
        "violations": violations,
    }


def _markdown_report(summary: dict[str, Any], evaluation: dict[str, Any]) -> str:
    metrics = summary["rollout_metrics"]
    evaluation_comparison = _comparison_from_evaluation(evaluation)
    lines = render_header_section(summary)
    lines.extend(render_rollout_metrics_section(metrics))
    lines.extend(render_dataset_summary_section(summary.get("dataset_summary")))
    lines.extend(render_benchmark_groups_section(summary))
    lines.extend(render_policy_ranking_section(summary))
    lines.extend(render_torch_policy_deltas_section(summary))
    lines.extend(render_architecture_deltas_section(summary))
    lines.extend(render_per_group_winners_section(summary))
    lines.extend(render_failure_scenarios_section(summary))
    lines.extend(render_gate_summary_section(summary))
    lines.extend(render_training_section(summary))
    lines.extend(render_baseline_comparison_section(summary))
    lines.extend(render_baselines_section(evaluation_comparison, _ordered_policy_names(evaluation_comparison)))
    return "\n".join(lines)


# Public aliases
render_experiment_markdown = _markdown_report
daily_report_summary = _daily_report_summary
policy_ranking = _policy_ranking
architecture_deltas = _architecture_deltas
loss_summary = _loss_summary

__all__ = (
    "_comparison_from_evaluation",
    "_baseline_deltas",
    "_metric_value",
    "_numeric_stats",
    "_daily_report_summary",
    "_architecture_deltas",
    "_policy_ranking",
    "_ordered_policy_names",
    "_per_group_winners",
    "_failure_scenarios",
    "_gate_summary",
    "_markdown_report",
    "_loss_summary",
    "render_experiment_markdown",
    "daily_report_summary",
    "policy_ranking",
    "architecture_deltas",
    "loss_summary",
)
