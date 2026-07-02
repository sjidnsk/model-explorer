from __future__ import annotations

import json
from typing import Any


def render_header_section(summary: dict[str, Any]) -> list[str]:
    return [
        "# Quasi-real South Pole Evaluation Matrix",
        "",
        f"- evaluation_scope: {summary['evaluation_scope']}",
        f"- data_class: {summary['data_class']}",
        f"- dataset_id: {summary['dataset_id']}",
        f"- region: {summary['region']}",
        f"- roi_count: {summary['roi_count']}",
        f"- mask_stress_augmented: {summary.get('mask_stress_augmented', False)}",
    ]


def render_roi_splits_section(summary: dict[str, Any]) -> list[str]:
    lines = ["", "## ROI Splits", "", "| split | roi | x | y | width | height | scenarios |", "|---|---|---:|---:|---:|---:|---:|"]
    for roi in summary.get("rois", []):
        bounds = roi.get("bounds", {}) if isinstance(roi, dict) else {}
        lines.append(
            "| "
            + " | ".join(
                (
                    str(roi.get("split", "")),
                    str(roi.get("name", "")),
                    str(bounds.get("x", 0)),
                    str(bounds.get("y", 0)),
                    str(bounds.get("width", 0)),
                    str(bounds.get("height", 0)),
                    str(roi.get("scenario_count", 0)),
                )
            )
            + " |"
        )
    return lines


def render_split_counts_section(summary: dict[str, Any]) -> list[str]:
    split_counts = summary.get("splits", {})
    if not isinstance(split_counts, dict) or not split_counts:
        return []
    lines = ["", "## Split Counts", "", "| split | scenarios |", "|---|---:|"]
    for split, count in split_counts.items():
        lines.append(f"| {split} | {count} |")
    return lines


def render_dataset_quality_section(dataset_summary: Any) -> list[str]:
    lines = ["", "## Dataset Quality", "", "| metric | value |", "|---|---:|"]
    if isinstance(dataset_summary, dict):
        for key in (
            "episode_count",
            "transition_count",
            "trainable_transition_count",
            "roi_count",
            "unreachable_candidate_count",
            "action_mask_valid_mean",
            "unreachable_candidate_rate",
            "padding_candidate_count",
            "padding_candidate_rate",
            "missing_experimental_feature_candidate_count",
            "mask_stress_sample_count",
            "mask_stress_sample_rate",
            "mask_stress_augmented",
            "non_finite_reward_count",
        ):
            lines.append(f"| {key} | {dataset_summary.get(key, 0)} |")
        reward = dataset_summary.get("reward", {})
        if isinstance(reward, dict):
            lines.append(f"| reward_mean | {reward.get('mean', 0.0)} |")
            lines.append(f"| reward_std | {reward.get('std', 0.0)} |")
    return lines


def render_mask_stress_section(summary: dict[str, Any], dataset_summary: Any) -> list[str]:
    mask_stress = summary.get("mask_stress", {})
    lines = ["", "## Mask-Stress Coverage", "", "| metric | value |", "|---|---:|"]
    if isinstance(mask_stress, dict):
        for key in ("enabled", "label", "profile", "unreachable_candidate_count", "missing_experimental_fields"):
            lines.append(f"| {key} | {mask_stress.get(key, '')} |")
    if isinstance(dataset_summary, dict):
        for key in (
            "mask_stress_augmented",
            "mask_stress_sample_count",
            "unreachable_candidate_count",
            "padding_candidate_count",
            "missing_experimental_feature_candidate_count",
        ):
            lines.append(f"| {key} | {dataset_summary.get(key, 0)} |")
    lines.append(f"| evaluation_scope | {summary['evaluation_scope']} |")
    return lines


def render_coverage_warnings_section(summary: dict[str, Any]) -> list[str]:
    lines = ["", "## Sample Coverage Warnings", ""]
    coverage_warnings = summary.get("coverage_warnings", [])
    if isinstance(coverage_warnings, list) and coverage_warnings:
        for warning in coverage_warnings:
            lines.append(f"- {warning}")
    else:
        lines.append("- none")
    return lines


def render_architecture_selection_section(selection: Any) -> list[str]:
    if not isinstance(selection, dict):
        return []
    lines = ["", "## Architecture Selection Gate", "", "| field | value |", "|---|---|"]
    for key in (
        "enabled",
        "status",
        "decision",
        "recommended_architecture",
        "metric",
        "mode",
        "reason",
        "decision_boundary",
        "baseline_delta_distribution",
    ):
        value = selection.get(key)
        if isinstance(value, (dict, list)):
            value = json.dumps(value, ensure_ascii=False, sort_keys=True)
        lines.append(f"| {key} | {value} |")
    lines.extend(_render_selection_quality_rows(selection))
    lines.extend(_render_architecture_selection_metrics(selection))
    lines.extend(_render_architecture_per_group_winners(selection))
    return lines


def _render_selection_quality_rows(selection: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    quality = selection.get("quality_gates", {})
    if isinstance(quality, dict):
        lines.append(f"| quality_gate_status | {quality.get('status', 'unknown')} |")
        violations = quality.get("violations", [])
        if isinstance(violations, list):
            lines.append(f"| quality_gate_violation_count | {len(violations)} |")
    mask_coverage = selection.get("mask_stress_coverage", {})
    if isinstance(mask_coverage, dict):
        for key in (
            "unreachable_candidate_count",
            "padding_candidate_count",
            "missing_experimental_feature_candidate_count",
            "mask_stress_sample_count",
        ):
            lines.append(f"| {key} | {mask_coverage.get(key, 0)} |")
    return lines


def _render_architecture_selection_metrics(selection: dict[str, Any]) -> list[str]:
    architectures = selection.get("architectures", {})
    if not isinstance(architectures, dict) or not architectures:
        return []
    lines = ["", "### Architecture Selection Metrics", "", "| architecture | run_count | exception_count | failure_count_mean | selection_metric_mean | selection_metric_std | loss_mean | loss_std |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for architecture, details in architectures.items():
        if not isinstance(details, dict):
            continue
        metric_stats = details.get("selection_metric", {})
        loss_stats = details.get("loss", {})
        failure_stats = details.get("failure_count", {})
        lines.append(
            "| "
            + " | ".join(
                (
                    str(architecture),
                    str(details.get("run_count", 0)),
                    str(details.get("exception_count", 0)),
                    str(failure_stats.get("mean", 0.0) if isinstance(failure_stats, dict) else 0.0),
                    str(metric_stats.get("mean", 0.0) if isinstance(metric_stats, dict) else 0.0),
                    str(metric_stats.get("std", 0.0) if isinstance(metric_stats, dict) else 0.0),
                    str(loss_stats.get("mean", 0.0) if isinstance(loss_stats, dict) else 0.0),
                    str(loss_stats.get("std", 0.0) if isinstance(loss_stats, dict) else 0.0),
                )
            )
            + " |"
        )
    return lines


def _render_architecture_per_group_winners(selection: dict[str, Any]) -> list[str]:
    per_group = selection.get("per_group_winners", {})
    if not isinstance(per_group, dict) or not per_group:
        return []
    lines = ["", "### Architecture Per-Group Winners", "", "| group | winner | reason |", "|---|---|---|"]
    for group_name, winner in per_group.items():
        if isinstance(winner, dict):
            lines.append(f"| {group_name} | {winner.get('decision', winner.get('recommended_architecture'))} | {winner.get('reason', '')} |")
    return lines


def render_policy_decision_diagnostics_section(selection: Any) -> list[str]:
    if not isinstance(selection, dict):
        return []
    decision_diagnostics = selection.get("decision_diagnostics", {})
    if not isinstance(decision_diagnostics, dict):
        return []
    lines = ["", "## Policy Decision Diagnostics", "", "- architecture_agreement_matrix: present"]
    warnings = decision_diagnostics.get("warnings", [])
    if isinstance(warnings, list) and warnings:
        for warning in warnings:
            lines.append(f"- warning: {warning}")
    else:
        lines.append("- warning: none")
    lines.extend(_render_agreement_matrix(decision_diagnostics))
    lines.extend(_render_baseline_agreement(decision_diagnostics))
    lines.extend(_render_per_group_disagreement(decision_diagnostics))
    return lines


def _render_agreement_matrix(decision_diagnostics: dict[str, Any]) -> list[str]:
    matrix = decision_diagnostics.get("architecture_agreement_matrix", {})
    if not isinstance(matrix, dict) or not matrix:
        return []
    lines = ["", "| left_architecture | right_architecture | compared | agreement_rate |", "|---|---|---:|---:|"]
    for left, row in matrix.items():
        if not isinstance(row, dict):
            continue
        for right, cell in row.items():
            if isinstance(cell, dict):
                lines.append(f"| {left} | {right} | {cell.get('compared', 0)} | {cell.get('agreement_rate', 0.0)} |")
    return lines


def _render_baseline_agreement(decision_diagnostics: dict[str, Any]) -> list[str]:
    baseline_agreement = decision_diagnostics.get("architecture_baseline_agreement", {})
    if not isinstance(baseline_agreement, dict) or not baseline_agreement:
        return []
    lines = ["", "| architecture | samples | utility_agreement_rate | coverage_heuristic_agreement_rate |", "|---|---:|---:|---:|"]
    for architecture, agreement in baseline_agreement.items():
        if isinstance(agreement, dict):
            lines.append(f"| {architecture} | {agreement.get('sample_count', 0)} | {agreement.get('utility_agreement_rate', 0.0)} | {agreement.get('coverage_heuristic_agreement_rate', 0.0)} |")
    return lines


def _render_per_group_disagreement(decision_diagnostics: dict[str, Any]) -> list[str]:
    per_group_disagreement = decision_diagnostics.get("per_group_disagreement", {})
    if not isinstance(per_group_disagreement, dict) or not per_group_disagreement:
        return []
    lines = ["", "| group | compared | disagreement_rate |", "|---|---:|---:|"]
    for group, disagreement in per_group_disagreement.items():
        if isinstance(disagreement, dict):
            lines.append(f"| {group} | {disagreement.get('compared', 0)} | {disagreement.get('disagreement_rate', 0.0)} |")
    return lines


def render_selection_composite_section(selection: Any) -> list[str]:
    if not isinstance(selection, dict):
        return []
    composite_selection = selection.get("composite_selection", {})
    lines = ["", "## Selection Composite Metrics", "", "| field | value |", "|---|---|"]
    lines.append("| selection_composite_weights | " + json.dumps(selection.get("selection_composite_weights", {}), ensure_ascii=False, sort_keys=True) + " |")
    if isinstance(composite_selection, dict):
        for key in ("status", "decision", "recommended_architecture", "reason"):
            lines.append(f"| composite_{key} | {composite_selection.get(key)} |")
    architectures = selection.get("architectures", {})
    if isinstance(architectures, dict) and architectures:
        lines.extend(["", "| architecture | selection_composite_score_mean | selection_composite_score_std |", "|---|---:|---:|"])
        for architecture, details in architectures.items():
            if not isinstance(details, dict):
                continue
            composite_stats = details.get("selection_composite_score", {})
            if not isinstance(composite_stats, dict):
                composite_stats = {}
            lines.append(f"| {architecture} | {composite_stats.get('mean', 0.0)} | {composite_stats.get('std', 0.0)} |")
    return lines


def render_action_sensitive_section(selection: Any) -> list[str]:
    return _render_stats_by_architecture_section(selection, "action_sensitive_summary", "## Action-Sensitive Metrics")


def render_oracle_regret_section(selection: Any) -> list[str]:
    return _render_stats_by_architecture_section(selection, "oracle_regret_summary", "## Oracle Regret Summary")


def _render_stats_by_architecture_section(selection: Any, source_key: str, heading: str) -> list[str]:
    if not isinstance(selection, dict):
        return []
    source = selection.get(source_key, {})
    if not isinstance(source, dict):
        return []
    lines = ["", heading, "", "| architecture | metric | mean | std | min | max | samples |", "|---|---|---:|---:|---:|---:|---:|"]
    for architecture, metrics in source.items():
        if not isinstance(metrics, dict):
            continue
        for metric, stats in metrics.items():
            if isinstance(stats, dict):
                lines.append(f"| {architecture} | {metric} | {stats.get('mean', 0.0)} | {stats.get('std', 0.0)} | {stats.get('min', 0.0)} | {stats.get('max', 0.0)} | {stats.get('count', 0)} |")
    return lines


def render_sample_discriminativeness_section(selection: Any) -> list[str]:
    if not isinstance(selection, dict):
        return []
    sample_discriminativeness = selection.get("sample_discriminativeness", {})
    if not isinstance(sample_discriminativeness, dict):
        return []
    lines = ["", "## Sample Discriminativeness", ""]
    warnings = sample_discriminativeness.get("warnings", [])
    lines.append(f"- status: {sample_discriminativeness.get('status', 'unknown')}")
    if isinstance(warnings, list) and warnings:
        for warning in warnings:
            lines.append(f"- warning: {warning}")
    else:
        lines.append("- warning: none")
    sample_metrics = sample_discriminativeness.get("metrics", {})
    if isinstance(sample_metrics, dict):
        lines.extend(["", "| metric | mean | std | min | max | samples |", "|---|---:|---:|---:|---:|---:|"])
        for metric, stats in sample_metrics.items():
            if isinstance(stats, dict):
                lines.append(f"| {metric} | {stats.get('mean', 0.0)} | {stats.get('std', 0.0)} | {stats.get('min', 0.0)} | {stats.get('max', 0.0)} | {stats.get('count', 0)} |")
    return lines


def render_per_roi_action_outcomes_section(selection: Any) -> list[str]:
    if not isinstance(selection, dict):
        return []
    per_group_action = selection.get("per_group_action_outcomes", {})
    if not isinstance(per_group_action, dict):
        return []
    lines = ["", "## Per-ROI Action Outcomes", "", "| group | decision | architecture | coverage_regret_mean | composite_regret_mean | selected_expected_coverage_delta_mean | reason |", "|---|---|---|---:|---:|---:|---|"]
    for group_name, group_summary in per_group_action.items():
        if not isinstance(group_summary, dict):
            continue
        group_architectures = group_summary.get("architectures", {})
        if not isinstance(group_architectures, dict):
            group_architectures = {}
        for architecture, metrics in group_architectures.items():
            if not isinstance(metrics, dict):
                continue
            coverage_regret = metrics.get("coverage_regret", {})
            composite_regret = metrics.get("composite_regret", {})
            selected_coverage = metrics.get("selected_expected_coverage_delta", {})
            lines.append(f"| {group_name} | {group_summary.get('decision', 'inconclusive')} | {architecture} | {coverage_regret.get('mean', 0.0) if isinstance(coverage_regret, dict) else 0.0} | {composite_regret.get('mean', 0.0) if isinstance(composite_regret, dict) else 0.0} | {selected_coverage.get('mean', 0.0) if isinstance(selected_coverage, dict) else 0.0} | {group_summary.get('reason', '')} |")
    return lines


def render_held_out_test_audit_section(selection: Any) -> list[str]:
    if not isinstance(selection, dict):
        return []
    held_out = selection.get("held_out_test_audit", {})
    if not isinstance(held_out, dict):
        return []
    lines = ["", "## Held-out Test Audit", "", "| field | value |", "|---|---|"]
    for key in ("status", "used_for_selection", "split", "validation_decision", "validation_recommended_architecture", "stable_with_validation", "reason", "evaluation_scope"):
        if key in held_out:
            lines.append(f"| {key} | {held_out.get(key)} |")
    return lines


def render_quality_gates_section(summary: dict[str, Any]) -> list[str]:
    lines = ["", "## Quality Gates", "", "| gate | value |", "|---|---:|"]
    quality_gates = summary.get("quality_gates", {})
    if isinstance(quality_gates, dict) and quality_gates:
        for key, value in quality_gates.items():
            lines.append(f"| {key} | {value} |")
    else:
        lines.append("| none | not configured |")
    return lines


def render_architectures_section(training: Any) -> list[str]:
    lines = ["", "## Architectures", "", "| architecture |", "|---|"]
    if isinstance(training, dict):
        for architecture in training.get("architectures", []):
            lines.append(f"| {architecture} |")
    return lines


def render_architecture_stability_section(stability_summary: Any) -> list[str]:
    architecture_stability = stability_summary.get("architectures", {}) if isinstance(stability_summary, dict) else {}
    if not isinstance(architecture_stability, dict) or not architecture_stability:
        return []
    lines = ["", "## Architecture Stability", "", "| architecture | metric | mean | std | min | max | samples |", "|---|---|---:|---:|---:|---:|---:|"]
    for architecture, metrics in architecture_stability.items():
        if not isinstance(metrics, dict):
            continue
        for metric, stats in metrics.items():
            if metric != "run_count" and isinstance(stats, dict):
                lines.append(f"| {architecture} | {metric} | {stats.get('mean', 0.0)} | {stats.get('std', 0.0)} | {stats.get('min', 0.0)} | {stats.get('max', 0.0)} | {stats.get('count', 0)} |")
    return lines


def render_loss_distribution_section(stability_summary: Any) -> list[str]:
    loss_distribution = stability_summary.get("loss_distribution", {}) if isinstance(stability_summary, dict) else {}
    if not isinstance(loss_distribution, dict) or not loss_distribution:
        return []
    lines = ["", "## Loss Distribution", "", "| metric | mean | std | min | max | samples |", "|---|---:|---:|---:|---:|---:|"]
    for metric, stats in loss_distribution.items():
        if isinstance(stats, dict):
            lines.append(f"| {metric} | {stats.get('mean', 0.0)} | {stats.get('std', 0.0)} | {stats.get('min', 0.0)} | {stats.get('max', 0.0)} | {stats.get('count', 0)} |")
    return lines


def render_baseline_comparison_section() -> list[str]:
    return [
        "",
        "## Baseline Comparison",
        "",
        "| section | status |",
        "|---|---|",
        "| utility | present |",
        "| coverage_heuristic | present |",
        "| torch_policy | present |",
    ]


def render_policy_ranking_section(experiment: Any) -> list[str]:
    policy_ranking = experiment.get("policy_ranking", {}) if isinstance(experiment, dict) else {}
    if not isinstance(policy_ranking, list) or not policy_ranking:
        return []
    lines = ["", "## Policy Ranking", "", "| rank | policy | final_coverage_rate | total_path_cost | failures |", "|---:|---|---:|---:|---:|"]
    for row in policy_ranking:
        if isinstance(row, dict):
            lines.append(f"| {row.get('rank', '')} | {row.get('policy', '')} | {row.get('final_coverage_rate', 0.0)} | {row.get('total_path_cost', 0.0)} | {row.get('failure_count', 0)} |")
    return lines


def render_per_group_winners_section(experiment: Any) -> list[str]:
    per_group_winners = experiment.get("per_group_winners", {}) if isinstance(experiment, dict) else {}
    if not isinstance(per_group_winners, dict) or not per_group_winners:
        return []
    lines = ["", "## Per-Group Winners", "", "| group | winner | final_coverage_rate | total_path_cost | failures |", "|---|---|---:|---:|---:|"]
    for group_name, winner in per_group_winners.items():
        if isinstance(winner, dict):
            lines.append(f"| {group_name} | {winner.get('policy', '')} | {winner.get('final_coverage_rate', 0.0)} | {winner.get('total_path_cost', 0.0)} | {winner.get('failure_count', 0)} |")
    return lines


def render_failure_scenarios_section(experiment: Any) -> list[str]:
    lines = ["", "## Failure Scenarios", "", "| scenario | policies |", "|---|---|"]
    failure_scenarios = experiment.get("failure_scenarios", []) if isinstance(experiment, dict) else []
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


def render_baseline_delta_summary_section(stability_summary: Any) -> list[str]:
    baseline_delta_summary = stability_summary.get("baseline_deltas", {}) if isinstance(stability_summary, dict) else {}
    if not isinstance(baseline_delta_summary, dict) or not baseline_delta_summary:
        return []
    lines = ["", "## Baseline Delta Summary", "", "| architecture | metric | mean | std | min | max | samples |", "|---|---|---:|---:|---:|---:|---:|"]
    for architecture, metrics in baseline_delta_summary.items():
        if not isinstance(metrics, dict):
            continue
        for metric, stats in metrics.items():
            if isinstance(stats, dict):
                lines.append(f"| {architecture} | {metric} | {stats.get('mean', 0.0)} | {stats.get('std', 0.0)} | {stats.get('min', 0.0)} | {stats.get('max', 0.0)} | {stats.get('count', 0)} |")
    return lines


def render_architecture_delta_details_section(experiment: Any) -> list[str]:
    architecture_deltas = experiment.get("architecture_deltas", {}) if isinstance(experiment, dict) else {}
    if not architecture_deltas:
        return []
    lines = ["", "## Architecture Delta Details", "", "| architecture | baseline | metric | delta |", "|---|---|---|---:|"]
    for architecture, baselines in architecture_deltas.items():
        if not isinstance(baselines, dict):
            continue
        for baseline, metrics in baselines.items():
            if not isinstance(metrics, dict):
                continue
            for metric, delta in metrics.items():
                lines.append(f"| {architecture} | {baseline} | {metric} | {delta} |")
    return lines


__all__ = (
    "render_action_sensitive_section",
    "render_architecture_delta_details_section",
    "render_architecture_selection_section",
    "render_architecture_stability_section",
    "render_architectures_section",
    "render_baseline_comparison_section",
    "render_baseline_delta_summary_section",
    "render_coverage_warnings_section",
    "render_dataset_quality_section",
    "render_failure_scenarios_section",
    "render_header_section",
    "render_held_out_test_audit_section",
    "render_loss_distribution_section",
    "render_mask_stress_section",
    "render_oracle_regret_section",
    "render_per_group_winners_section",
    "render_per_roi_action_outcomes_section",
    "render_policy_decision_diagnostics_section",
    "render_policy_ranking_section",
    "render_quality_gates_section",
    "render_roi_splits_section",
    "render_sample_discriminativeness_section",
    "render_selection_composite_section",
    "render_split_counts_section",
)
