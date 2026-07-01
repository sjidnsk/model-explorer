from __future__ import annotations

import json
from typing import Any

from .evaluation import _comparison_from_evaluation
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
    lines = [
        "# Model Explorer Experiment Report",
        "",
        f"- schema_version: {summary['schema_version']}",
        f"- planner: {summary['planner']}",
        f"- scenario_count: {summary['scenario_count']}",
        f"- transition_count: {summary['transition_count']}",
        "- benchmark_scope: synthetic smoke / regression suite; not a real-world generalization benchmark",
        "",
        "## Rollout Metrics",
        "",
        "| metric | value |",
        "|---|---:|",
    ]
    for key in (
        "final_coverage_rate",
        "cumulative_coverage_rate_delta",
        "total_path_cost",
        "average_risk",
        "failure_count",
        "replan_count",
        "value_coverage",
        "total_reward",
    ):
        lines.append(f"| {key} | {metrics[key]} |")

    dataset_summary = summary.get("dataset_summary")
    if isinstance(dataset_summary, dict):
        lines.extend(["", "## Dataset Summary", "", "| metric | value |", "|---|---:|"])
        for key in (
            "data_class",
            "dataset_id",
            "region",
            "generator_version",
            "roi_count",
            "episode_count",
            "transition_count",
            "trainable_transition_count",
            "no_op_transition_count",
            "failure_transition_count",
            "unreachable_candidate_count",
            "padding_candidate_count",
            "missing_experimental_feature_candidate_count",
            "mask_stress_sample_count",
            "mask_stress_sample_rate",
            "mask_stress_augmented",
            "empty_action_mask_count",
            "invalid_action_mask_count",
            "failure_count",
            "replan_count",
            "coverage_delta_total",
            "total_path_cost",
            "average_risk",
        ):
            lines.append(f"| {key} | {dataset_summary.get(key, 0)} |")
        reward_summary = dataset_summary.get("reward", {})
        if isinstance(reward_summary, dict):
            for key in ("min", "max", "mean"):
                lines.append(f"| reward_{key} | {reward_summary.get(key, 0.0)} |")
        validation_gates = dataset_summary.get("validation_gates")
        if isinstance(validation_gates, dict):
            lines.extend(["", "## Dataset Validation Gates", "", f"- status: {validation_gates.get('status', 'unknown')}"])
            configured = validation_gates.get("configured", {})
            if isinstance(configured, dict):
                lines.extend(["", "| gate | value |", "|---|---:|"])
                for key, value in configured.items():
                    lines.append(f"| {key} | {value} |")
            violations = validation_gates.get("violations", [])
            if isinstance(violations, list) and violations:
                lines.extend(["", "| violation | message |", "|---|---|"])
                for violation in violations:
                    if isinstance(violation, dict):
                        lines.append(f"| {violation.get('gate', '')} | {violation.get('message', '')} |")

    split_summaries = summary.get("split_summaries", {})
    benchmark_summary = split_summaries.get("benchmark") if isinstance(split_summaries, dict) else None
    if isinstance(benchmark_summary, dict):
        lines.extend(
            [
                "",
                "## Benchmark Groups",
                "",
                "| group | scenarios | episodes | transitions |",
                "|---|---:|---:|---:|",
            ]
        )
        groups = benchmark_summary.get("groups", {})
        if isinstance(groups, dict):
            for group_name, group_summary in groups.items():
                if not isinstance(group_summary, dict):
                    continue
                lines.append(
                    "| "
                    + " | ".join(
                        (
                            str(group_name),
                            str(group_summary.get("scenario_count", 0)),
                            str(group_summary.get("episode_count", 0)),
                            str(group_summary.get("transition_count", 0)),
                        )
                    )
                    + " |"
                )

    policy_ranking = summary.get("policy_ranking", [])
    lines.extend(
        [
            "",
            "## Policy Ranking",
            "",
            "| rank | policy | final_coverage_rate | failures | total_path_cost | value_coverage |",
            "|---:|---|---:|---:|---:|---:|",
        ]
    )
    if isinstance(policy_ranking, list) and policy_ranking:
        for row in policy_ranking:
            if not isinstance(row, dict):
                continue
            lines.append(
                "| "
                + " | ".join(
                    (
                        str(row.get("rank", "")),
                        str(row.get("policy", "")),
                        str(row.get("final_coverage_rate", 0.0)),
                        str(row.get("failure_count", 0)),
                        str(row.get("total_path_cost", 0.0)),
                        str(row.get("value_coverage", 0.0)),
                    )
                )
                + " |"
            )

    lines.extend(
        [
            "",
            "## Torch Policy Deltas",
            "",
            "| baseline | metric | delta |",
            "|---|---|---:|",
        ]
    )
    torch_deltas_for_section = summary.get("baseline_deltas", {})
    torch_deltas_for_section = (
        torch_deltas_for_section.get("torch_policy")
        if isinstance(torch_deltas_for_section, dict)
        else None
    )
    if isinstance(torch_deltas_for_section, dict) and torch_deltas_for_section:
        for baseline_name, metrics in torch_deltas_for_section.items():
            if not isinstance(metrics, dict):
                continue
            for metric, delta in metrics.items():
                lines.append(f"| {baseline_name} | {metric} | {delta} |")
    else:
        lines.append("| none | torch_policy unavailable | 0.0 |")

    architecture_deltas = summary.get("architecture_deltas", {})
    lines.extend(
        [
            "",
            "## Architecture Deltas",
            "",
            "| architecture | baseline | metric | delta |",
            "|---|---|---|---:|",
        ]
    )
    if isinstance(architecture_deltas, dict) and architecture_deltas:
        for architecture, baseline_map in architecture_deltas.items():
            if not isinstance(baseline_map, dict):
                continue
            for baseline_name, metrics in baseline_map.items():
                if not isinstance(metrics, dict):
                    continue
                for metric, delta in metrics.items():
                    lines.append(f"| {architecture} | {baseline_name} | {metric} | {delta} |")
    else:
        lines.append("| none | none | none | 0.0 |")

    per_group_winners = summary.get("per_group_winners", {})
    lines.extend(
        [
            "",
            "## Per-Group Winners",
            "",
            "| group | winner | final_coverage_rate | failures |",
            "|---|---|---:|---:|",
        ]
    )
    if isinstance(per_group_winners, dict) and per_group_winners:
        for group_name, winner in per_group_winners.items():
            if isinstance(winner, dict):
                lines.append(
                    "| "
                    + " | ".join(
                        (
                            str(group_name),
                            str(winner.get("policy", "")),
                            str(winner.get("final_coverage_rate", 0.0)),
                            str(winner.get("failure_count", 0)),
                        )
                    )
                    + " |"
                )
            else:
                lines.append(f"| {group_name} | none | 0.0 | 0 |")

    failure_scenarios = summary.get("failure_scenarios", [])
    lines.extend(["", "## Failure Scenarios", "", "| scenario | policies |", "|---|---|"])
    if isinstance(failure_scenarios, list) and failure_scenarios:
        for item in failure_scenarios:
            if not isinstance(item, dict):
                continue
            policies = item.get("policies", [])
            policy_text = ", ".join(str(policy) for policy in policies) if isinstance(policies, list) else ""
            lines.append(f"| {item.get('path', '')} | {policy_text} |")
    else:
        lines.append("| none | none |")

    gate_summary = summary.get("gate_summary", {})
    lines.extend(["", "## Gate Summary", ""])
    if isinstance(gate_summary, dict):
        lines.append(f"- status: {gate_summary.get('status', 'unknown')}")
        lines.append(f"- warning_count: {len(gate_summary.get('warnings', []))}")
        lines.append(f"- violation_count: {gate_summary.get('violation_count', 0)}")

    if "training" in summary:
        training = summary["training"]
        lines.extend(["", "## Training", "", "| field | value |", "|---|---:|"])
        for key in (
            "checkpoint",
            "architecture",
            "best_checkpoint_path",
            "last_checkpoint_path",
            "best_seed",
            "seed",
            "run_count",
            "epochs",
            "sample_count",
            "train_episode_count",
            "validation_episode_count",
            "loss",
            "policy_loss",
            "value_loss",
            "entropy",
        ):
            if key in training:
                lines.append(f"| {key} | {training[key]} |")

        source_comparison = training.get("source_comparison")
        if isinstance(source_comparison, dict) and source_comparison:
            lines.extend(
                [
                    "",
                    "## Training Source Comparison",
                    "",
                    "| source | primary_selection_strategy | feedback_aware_action_agreement_rate | feedback_aware_top2_action_agreement_rate |",
                    "|---|---|---:|---:|",
                ]
            )
            for source_name, source_summary in source_comparison.items():
                if not isinstance(source_summary, dict):
                    continue
                training_source = source_summary.get("training_source", {})
                training_source = training_source if isinstance(training_source, dict) else {}
                teacher_agreement = source_summary.get("teacher_agreement", {})
                teacher_agreement = teacher_agreement if isinstance(teacher_agreement, dict) else {}
                lines.append(
                    "| "
                    + " | ".join(
                        (
                            str(source_name),
                            str(training_source.get("primary_selection_strategy", "")),
                            str(teacher_agreement.get("feedback_aware_action_agreement_rate", 0.0)),
                            str(teacher_agreement.get("feedback_aware_top2_action_agreement_rate", 0.0)),
                        )
                    )
                    + " |"
                )

        distillation_matrix = training.get("distillation_matrix")
        if isinstance(distillation_matrix, list) and distillation_matrix:
            lines.extend(
                [
                    "",
                    "## Distillation Matrix",
                    "",
                    "| source | teacher_imitation_weight | teacher_quality_gate_status | feedback_aware_action_agreement_rate | feedback_aware_delta_final_coverage_rate |",
                    "|---|---:|---|---:|---:|",
                ]
            )
            for record in distillation_matrix:
                if not isinstance(record, dict):
                    continue
                gates = record.get("teacher_quality_gates", {})
                gates = gates if isinstance(gates, dict) else {}
                teacher_agreement = record.get("teacher_agreement", {})
                teacher_agreement = teacher_agreement if isinstance(teacher_agreement, dict) else {}
                baseline_deltas = record.get("baseline_deltas", {})
                baseline_deltas = baseline_deltas if isinstance(baseline_deltas, dict) else {}
                feedback_delta = baseline_deltas.get("feedback_aware", {})
                feedback_delta = feedback_delta if isinstance(feedback_delta, dict) else {}
                lines.append(
                    "| "
                    + " | ".join(
                        (
                            str(record.get("source_selection_strategy", "")),
                            str(record.get("teacher_imitation_weight", 0.0)),
                            str(gates.get("status", "unknown")),
                            str(teacher_agreement.get("feedback_aware_action_agreement_rate", 0.0)),
                            str(feedback_delta.get("final_coverage_rate", 0.0)),
                        )
                    )
                    + " |"
                )

        architecture_config = training.get("architecture_config")
        architecture_diagnostics = training.get("architecture_diagnostics")
        if isinstance(architecture_config, dict) or isinstance(architecture_diagnostics, dict):
            lines.extend(["", "## Architecture Diagnostics", "", "| field | value |", "|---|---|"])
            if isinstance(architecture_config, dict):
                for key in sorted(architecture_config):
                    lines.append(f"| {key} | {architecture_config[key]} |")
            if isinstance(architecture_diagnostics, dict):
                for key in (
                    "architecture",
                    "observation_schema_version",
                    "candidate_feature_dim",
                    "global_feature_dim",
                    "missing_indicator_dim",
                    "mask_valid_action_count",
                ):
                    if key not in architecture_diagnostics:
                        continue
                    value = architecture_diagnostics[key]
                    if isinstance(value, dict):
                        value = json.dumps(value, ensure_ascii=False, sort_keys=True)
                    lines.append(f"| {key} | {value} |")
        lines.extend(["", "### dataset_summary", "", "| metric | value |", "|---|---:|"])
        training_dataset = training.get("dataset_summary", {})
        if isinstance(training_dataset, dict):
            for key in (
                "data_class",
                "dataset_id",
                "region",
                "generator_version",
                "roi_count",
                "episode_count",
                "transition_count",
                "trainable_transition_count",
                "failure_transition_count",
                "coverage_delta_total",
                "total_path_cost",
                "average_risk",
            ):
                lines.append(f"| {key} | {training_dataset.get(key, 0)} |")

        best_selection = training.get("best_selection", {})
        if isinstance(best_selection, dict):
            lines.extend(["", "## Best Checkpoint", "", "| field | value |", "|---|---|"])
            for key in ("best_seed", "best_checkpoint_path", "last_checkpoint_path"):
                if key in training:
                    lines.append(f"| {key} | {training[key]} |")
            for key in ("policy", "metric", "mode", "value", "reason"):
                if key in best_selection:
                    lines.append(f"| {key} | {best_selection[key]} |")

        multi_seed_summary = training.get("multi_seed_summary", {})
        if isinstance(multi_seed_summary, dict) and multi_seed_summary:
            lines.extend(
                [
                    "",
                    "## Multi-Seed Summary",
                    "",
                    "| policy | metric | mean | std | min | max |",
                    "|---|---|---:|---:|---:|---:|",
                ]
            )
            for policy, metrics in multi_seed_summary.items():
                if not isinstance(metrics, dict):
                    continue
                for metric, stats in metrics.items():
                    if not isinstance(stats, dict):
                        continue
                    lines.append(
                        "| "
                        + " | ".join(
                            (
                                str(policy),
                                str(metric),
                                str(stats.get("mean", 0.0)),
                                str(stats.get("std", 0.0)),
                                str(stats.get("min", 0.0)),
                                str(stats.get("max", 0.0)),
                            )
                        )
                        + " |"
                    )

        runs = training.get("runs", [])
        if isinstance(runs, list) and runs:
            lines.extend(
                [
                    "",
                    "## Per-Seed Metrics",
                    "",
                    "| architecture | seed | checkpoint | final_coverage_rate | total_path_cost | loss | policy_loss | value_loss | entropy |",
                    "|---|---:|---|---:|---:|---:|---:|---:|---:|",
                ]
            )
            for run in runs:
                if not isinstance(run, dict):
                    continue
                torch_metrics = {}
                validation_evaluation = run.get("validation_evaluation", {})
                if isinstance(validation_evaluation, dict):
                    torch_metrics = validation_evaluation.get("torch_policy", {}) or {}
                lines.append(
                    "| "
                    + " | ".join(
                        (
                            str(run.get("architecture", "")),
                            str(run.get("seed", "")),
                            str(run.get("checkpoint", "")),
                            str(torch_metrics.get("final_coverage_rate", 0.0)),
                            str(torch_metrics.get("total_path_cost", 0.0)),
                            str(run.get("loss", 0.0)),
                            str(run.get("policy_loss", 0.0)),
                            str(run.get("value_loss", 0.0)),
                            str(run.get("entropy", 0.0)),
                        )
                    )
                    + " |"
                )

            loss_summary = _loss_summary(runs)
            lines.extend(
                [
                    "",
                    "## Loss Summary",
                    "",
                    "| metric | mean | std | min | max |",
                    "|---|---:|---:|---:|---:|",
                ]
            )
            for metric, stats in loss_summary.items():
                lines.append(
                    "| "
                    + " | ".join(
                        (
                            metric,
                            str(stats.get("mean", 0.0)),
                            str(stats.get("std", 0.0)),
                            str(stats.get("min", 0.0)),
                            str(stats.get("max", 0.0)),
                        )
                    )
                    + " |"
                )

        lines.extend(["", "## Training Quality", "", "| seed | warnings |", "|---:|---|"])
        for run in runs if isinstance(runs, list) else []:
            if not isinstance(run, dict):
                continue
            warnings = run.get("warnings", [])
            warning_text = ", ".join(str(item) for item in warnings) if isinstance(warnings, list) else ""
            lines.append(f"| {run.get('seed', '')} | {warning_text} |")

    baseline_deltas = summary.get("baseline_deltas", {})
    torch_deltas = baseline_deltas.get("torch_policy") if isinstance(baseline_deltas, dict) else None
    if isinstance(torch_deltas, dict) and torch_deltas:
        lines.extend(
            [
                "",
                "## Baseline Comparison",
                "",
                "| baseline | metric | delta |",
                "|---|---|---:|",
            ]
        )
        for baseline_name, metrics in torch_deltas.items():
            if not isinstance(metrics, dict):
                continue
            for metric, delta in metrics.items():
                lines.append(f"| {baseline_name} | {metric} | {delta} |")

    lines.extend(["", "## Baselines", "", "| policy | final_coverage_rate | cumulative_coverage_rate_delta | total_path_cost | average_risk | failure_count | replan_count | value_coverage |", "|---|---:|---:|---:|---:|---:|---:|---:|"])
    for policy_name in _ordered_policy_names(evaluation_comparison):
        if policy_name not in evaluation_comparison:
            continue
        policy_metrics = evaluation_comparison[policy_name]
        lines.append(
            "| "
            + " | ".join(
                (
                    policy_name,
                    str(policy_metrics.get("final_coverage_rate", policy_metrics.get("average_final_coverage_rate", 0.0))),
                    str(policy_metrics.get("cumulative_coverage_rate_delta", 0.0)),
                    str(policy_metrics.get("total_path_cost", 0.0)),
                    str(policy_metrics.get("average_risk", 0.0)),
                    str(policy_metrics.get("failure_count", 0)),
                    str(policy_metrics.get("replan_count", 0)),
                    str(policy_metrics.get("value_coverage", 0.0)),
                )
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def _loss_summary(runs: list[Any]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for metric in ("loss", "policy_loss", "value_loss", "entropy"):
        values: list[float] = []
        for run in runs:
            if not isinstance(run, dict):
                continue
            try:
                values.append(float(run[metric]))
            except (KeyError, TypeError, ValueError):
                continue
        summary[metric] = _numeric_stats(tuple(values))
    return summary


# Public aliases
render_experiment_markdown = _markdown_report
daily_report_summary = _daily_report_summary
policy_ranking = _policy_ranking
architecture_deltas = _architecture_deltas
loss_summary = _loss_summary

__all__ = [name for name in globals() if not name.startswith("__")]
