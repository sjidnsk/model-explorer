from __future__ import annotations

import json
from typing import Any

from .selection import _numeric_stats


def render_header_section(summary: dict[str, Any]) -> list[str]:
    return [
        "# Model Explorer Experiment Report",
        "",
        f"- schema_version: {summary['schema_version']}",
        f"- planner: {summary['planner']}",
        f"- scenario_count: {summary['scenario_count']}",
        f"- transition_count: {summary['transition_count']}",
        "- benchmark_scope: synthetic smoke / regression suite; not a real-world generalization benchmark",
    ]


def render_rollout_metrics_section(metrics: dict[str, Any]) -> list[str]:
    lines = ["", "## Rollout Metrics", "", "| metric | value |", "|---|---:|"]
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
    return lines


def render_dataset_summary_section(dataset_summary: Any) -> list[str]:
    if not isinstance(dataset_summary, dict):
        return []
    lines = ["", "## Dataset Summary", "", "| metric | value |", "|---|---:|"]
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
    return lines


def render_benchmark_groups_section(summary: dict[str, Any]) -> list[str]:
    split_summaries = summary.get("split_summaries", {})
    benchmark_summary = split_summaries.get("benchmark") if isinstance(split_summaries, dict) else None
    if not isinstance(benchmark_summary, dict):
        return []
    lines = ["", "## Benchmark Groups", "", "| group | scenarios | episodes | transitions |", "|---|---:|---:|---:|"]
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
    return lines


def render_policy_ranking_section(summary: dict[str, Any]) -> list[str]:
    lines = [
        "",
        "## Policy Ranking",
        "",
        "| rank | policy | final_coverage_rate | failures | total_path_cost | value_coverage |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    policy_ranking = summary.get("policy_ranking", [])
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
    return lines


def render_torch_policy_deltas_section(summary: dict[str, Any]) -> list[str]:
    lines = ["", "## Torch Policy Deltas", "", "| baseline | metric | delta |", "|---|---|---:|"]
    torch_deltas = summary.get("baseline_deltas", {})
    torch_deltas = torch_deltas.get("torch_policy") if isinstance(torch_deltas, dict) else None
    if isinstance(torch_deltas, dict) and torch_deltas:
        for baseline_name, metrics in torch_deltas.items():
            if not isinstance(metrics, dict):
                continue
            for metric, delta in metrics.items():
                lines.append(f"| {baseline_name} | {metric} | {delta} |")
    else:
        lines.append("| none | torch_policy unavailable | 0.0 |")
    return lines


def render_architecture_deltas_section(summary: dict[str, Any]) -> list[str]:
    lines = ["", "## Architecture Deltas", "", "| architecture | baseline | metric | delta |", "|---|---|---|---:|"]
    architecture_deltas = summary.get("architecture_deltas", {})
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
    return lines


def render_per_group_winners_section(summary: dict[str, Any]) -> list[str]:
    lines = ["", "## Per-Group Winners", "", "| group | winner | final_coverage_rate | failures |", "|---|---|---:|---:|"]
    per_group_winners = summary.get("per_group_winners", {})
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
    return lines


def render_failure_scenarios_section(summary: dict[str, Any]) -> list[str]:
    lines = ["", "## Failure Scenarios", "", "| scenario | policies |", "|---|---|"]
    failure_scenarios = summary.get("failure_scenarios", [])
    if isinstance(failure_scenarios, list) and failure_scenarios:
        for item in failure_scenarios:
            if not isinstance(item, dict):
                continue
            policies = item.get("policies", [])
            policy_text = ", ".join(str(policy) for policy in policies) if isinstance(policies, list) else ""
            lines.append(f"| {item.get('path', '')} | {policy_text} |")
    else:
        lines.append("| none | none |")
    return lines


def render_gate_summary_section(summary: dict[str, Any]) -> list[str]:
    lines = ["", "## Gate Summary", ""]
    gate_summary = summary.get("gate_summary", {})
    if isinstance(gate_summary, dict):
        lines.append(f"- status: {gate_summary.get('status', 'unknown')}")
        lines.append(f"- warning_count: {len(gate_summary.get('warnings', []))}")
        lines.append(f"- violation_count: {gate_summary.get('violation_count', 0)}")
    return lines


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


def render_training_section(summary: dict[str, Any]) -> list[str]:
    if "training" not in summary:
        return []
    training = summary["training"]
    lines = ["", "## Training", "", "| field | value |", "|---|---:|"]
    for key in (
        "checkpoint", "architecture", "best_checkpoint_path", "last_checkpoint_path", "best_seed", "seed",
        "run_count", "epochs", "sample_count", "train_episode_count", "validation_episode_count", "loss",
        "policy_loss", "value_loss", "entropy",
    ):
        if key in training:
            lines.append(f"| {key} | {training[key]} |")
    lines.extend(_render_training_source_comparison(training))
    lines.extend(_render_distillation_matrix(training))
    lines.extend(_render_architecture_diagnostics(training))
    lines.extend(_render_training_dataset_summary(training))
    lines.extend(_render_best_checkpoint(training))
    lines.extend(_render_multi_seed_summary(training))
    lines.extend(_render_per_seed_metrics(training))
    lines.extend(_render_training_quality(training))
    return lines


def _render_training_source_comparison(training: dict[str, Any]) -> list[str]:
    source_comparison = training.get("source_comparison")
    if not isinstance(source_comparison, dict) or not source_comparison:
        return []
    lines = ["", "## Training Source Comparison", "", "| source | primary_selection_strategy | feedback_aware_action_agreement_rate | feedback_aware_top2_action_agreement_rate |", "|---|---|---:|---:|"]
    for source_name, source_summary in source_comparison.items():
        if not isinstance(source_summary, dict):
            continue
        training_source = source_summary.get("training_source", {})
        training_source = training_source if isinstance(training_source, dict) else {}
        teacher_agreement = source_summary.get("teacher_agreement", {})
        teacher_agreement = teacher_agreement if isinstance(teacher_agreement, dict) else {}
        lines.append(f"| {source_name} | {training_source.get('primary_selection_strategy', '')} | {teacher_agreement.get('feedback_aware_action_agreement_rate', 0.0)} | {teacher_agreement.get('feedback_aware_top2_action_agreement_rate', 0.0)} |")
    return lines


def _render_distillation_matrix(training: dict[str, Any]) -> list[str]:
    distillation_matrix = training.get("distillation_matrix")
    if not isinstance(distillation_matrix, list) or not distillation_matrix:
        return []
    lines = ["", "## Distillation Matrix", "", "| source | teacher_imitation_weight | teacher_quality_gate_status | feedback_aware_action_agreement_rate | feedback_aware_delta_final_coverage_rate |", "|---|---:|---|---:|---:|"]
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
        lines.append(f"| {record.get('source_selection_strategy', '')} | {record.get('teacher_imitation_weight', 0.0)} | {gates.get('status', 'unknown')} | {teacher_agreement.get('feedback_aware_action_agreement_rate', 0.0)} | {feedback_delta.get('final_coverage_rate', 0.0)} |")
    return lines


def _render_architecture_diagnostics(training: dict[str, Any]) -> list[str]:
    architecture_config = training.get("architecture_config")
    architecture_diagnostics = training.get("architecture_diagnostics")
    if not isinstance(architecture_config, dict) and not isinstance(architecture_diagnostics, dict):
        return []
    lines = ["", "## Architecture Diagnostics", "", "| field | value |", "|---|---|"]
    if isinstance(architecture_config, dict):
        for key in sorted(architecture_config):
            lines.append(f"| {key} | {architecture_config[key]} |")
    if isinstance(architecture_diagnostics, dict):
        for key in ("architecture", "observation_schema_version", "candidate_feature_dim", "global_feature_dim", "missing_indicator_dim", "mask_valid_action_count"):
            if key not in architecture_diagnostics:
                continue
            value = architecture_diagnostics[key]
            if isinstance(value, dict):
                value = json.dumps(value, ensure_ascii=False, sort_keys=True)
            lines.append(f"| {key} | {value} |")
    return lines


def _render_training_dataset_summary(training: dict[str, Any]) -> list[str]:
    lines = ["", "### dataset_summary", "", "| metric | value |", "|---|---:|"]
    training_dataset = training.get("dataset_summary", {})
    if isinstance(training_dataset, dict):
        for key in ("data_class", "dataset_id", "region", "generator_version", "roi_count", "episode_count", "transition_count", "trainable_transition_count", "failure_transition_count", "coverage_delta_total", "total_path_cost", "average_risk"):
            lines.append(f"| {key} | {training_dataset.get(key, 0)} |")
    return lines


def _render_best_checkpoint(training: dict[str, Any]) -> list[str]:
    best_selection = training.get("best_selection", {})
    if not isinstance(best_selection, dict):
        return []
    lines = ["", "## Best Checkpoint", "", "| field | value |", "|---|---|"]
    for key in ("best_seed", "best_checkpoint_path", "last_checkpoint_path"):
        if key in training:
            lines.append(f"| {key} | {training[key]} |")
    for key in ("policy", "metric", "mode", "value", "reason"):
        if key in best_selection:
            lines.append(f"| {key} | {best_selection[key]} |")
    return lines


def _render_multi_seed_summary(training: dict[str, Any]) -> list[str]:
    multi_seed_summary = training.get("multi_seed_summary", {})
    if not isinstance(multi_seed_summary, dict) or not multi_seed_summary:
        return []
    lines = ["", "## Multi-Seed Summary", "", "| policy | metric | mean | std | min | max |", "|---|---|---:|---:|---:|---:|"]
    for policy, metrics in multi_seed_summary.items():
        if not isinstance(metrics, dict):
            continue
        for metric, stats in metrics.items():
            if isinstance(stats, dict):
                lines.append(f"| {policy} | {metric} | {stats.get('mean', 0.0)} | {stats.get('std', 0.0)} | {stats.get('min', 0.0)} | {stats.get('max', 0.0)} |")
    return lines


def _render_per_seed_metrics(training: dict[str, Any]) -> list[str]:
    runs = training.get("runs", [])
    if not isinstance(runs, list) or not runs:
        return []
    lines = ["", "## Per-Seed Metrics", "", "| architecture | seed | checkpoint | final_coverage_rate | total_path_cost | loss | policy_loss | value_loss | entropy |", "|---|---:|---|---:|---:|---:|---:|---:|---:|"]
    for run in runs:
        if not isinstance(run, dict):
            continue
        torch_metrics = {}
        validation_evaluation = run.get("validation_evaluation", {})
        if isinstance(validation_evaluation, dict):
            torch_metrics = validation_evaluation.get("torch_policy", {}) or {}
        lines.append(f"| {run.get('architecture', '')} | {run.get('seed', '')} | {run.get('checkpoint', '')} | {torch_metrics.get('final_coverage_rate', 0.0)} | {torch_metrics.get('total_path_cost', 0.0)} | {run.get('loss', 0.0)} | {run.get('policy_loss', 0.0)} | {run.get('value_loss', 0.0)} | {run.get('entropy', 0.0)} |")
    lines.extend(["", "## Loss Summary", "", "| metric | mean | std | min | max |", "|---|---:|---:|---:|---:|"])
    for metric, stats in _loss_summary(runs).items():
        lines.append(f"| {metric} | {stats.get('mean', 0.0)} | {stats.get('std', 0.0)} | {stats.get('min', 0.0)} | {stats.get('max', 0.0)} |")
    return lines


def _render_training_quality(training: dict[str, Any]) -> list[str]:
    lines = ["", "## Training Quality", "", "| seed | warnings |", "|---:|---|"]
    runs = training.get("runs", [])
    for run in runs if isinstance(runs, list) else []:
        if not isinstance(run, dict):
            continue
        warnings = run.get("warnings", [])
        warning_text = ", ".join(str(item) for item in warnings) if isinstance(warnings, list) else ""
        lines.append(f"| {run.get('seed', '')} | {warning_text} |")
    return lines


def render_baseline_comparison_section(summary: dict[str, Any]) -> list[str]:
    baseline_deltas = summary.get("baseline_deltas", {})
    torch_deltas = baseline_deltas.get("torch_policy") if isinstance(baseline_deltas, dict) else None
    if not isinstance(torch_deltas, dict) or not torch_deltas:
        return []
    lines = ["", "## Baseline Comparison", "", "| baseline | metric | delta |", "|---|---|---:|"]
    for baseline_name, metrics in torch_deltas.items():
        if not isinstance(metrics, dict):
            continue
        for metric, delta in metrics.items():
            lines.append(f"| {baseline_name} | {metric} | {delta} |")
    return lines


def render_baselines_section(evaluation_comparison: dict[str, Any], ordered_policy_names: tuple[str, ...]) -> list[str]:
    lines = ["", "## Baselines", "", "| policy | final_coverage_rate | cumulative_coverage_rate_delta | total_path_cost | average_risk | failure_count | replan_count | value_coverage |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for policy_name in ordered_policy_names:
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
    return lines


__all__ = (
    "_loss_summary",
    "render_architecture_deltas_section",
    "render_baseline_comparison_section",
    "render_baselines_section",
    "render_benchmark_groups_section",
    "render_dataset_summary_section",
    "render_failure_scenarios_section",
    "render_gate_summary_section",
    "render_header_section",
    "render_per_group_winners_section",
    "render_policy_ranking_section",
    "render_rollout_metrics_section",
    "render_torch_policy_deltas_section",
    "render_training_section",
)
