from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .manifest import (
    EVALUATION_SCOPE,
    QUASI_REAL_EVALUATION_SCHEMA_VERSION,
    QuasiRealEvaluationManifest,
    _mask_stress_enabled,
    _mask_stress_summary,
    _roi_summary,
    _selection_config_summary,
    _split_counts,
)
from .metrics import _int_value
from .selection import _architecture_selection_summary, _stability_summary


def _inspection_summary(
    manifest: QuasiRealEvaluationManifest,
    *,
    status: str,
    data_validation: dict[str, Any],
) -> dict[str, Any]:
    split_counts = _split_counts(manifest.rois)
    return {
        "status": status,
        "schema_version": QUASI_REAL_EVALUATION_SCHEMA_VERSION,
        "evaluation_scope": EVALUATION_SCOPE,
        "name": manifest.name,
        "run_id": manifest.run_id,
        "dataset_manifest": str(manifest.dataset_manifest),
        "dataset_id": data_validation.get("dataset_id"),
        "data_class": data_validation.get("data_class"),
        "output_root": str(manifest.output_root),
        "roi_count": len({roi.name for roi in manifest.rois}),
        "rois": [_roi_summary(roi) for roi in manifest.rois],
        "splits": split_counts,
        "architectures": list(manifest.train_config.get("architectures", [manifest.train_config.get("architecture", "mlp_v1")])),
        "mask_stress": _mask_stress_summary(manifest.mask_stress_config),
        "selection": _selection_config_summary(manifest.selection_config),
        "dataset_validation": data_validation,
        "quality_gates": dict(manifest.dataset_validation),
    }


def _would_write(manifest: QuasiRealEvaluationManifest) -> list[str]:
    paths = [
        manifest.output_root / "experiment.json",
        manifest.output_root / "summary.json",
        manifest.output_root / "matrix-report.md",
        manifest.output_root / "out",
    ]
    for roi in manifest.rois:
        for index in range(roi.episode_count):
            paths.append(manifest.output_root / "scenarios" / roi.split / roi.name / f"lola-south-pole-{index:03d}.json")
    return [str(path) for path in paths]


def _run_summary(
    manifest: QuasiRealEvaluationManifest,
    *,
    data_manifest: dict[str, Any],
    data_validation: dict[str, Any],
    generated_scenarios: list[dict[str, Any]],
    experiment_manifest_path: Path,
    experiment_summary: dict[str, Any],
) -> dict[str, Any]:
    stability_summary = _stability_summary(experiment_summary)
    return {
        "status": "completed",
        "schema_version": QUASI_REAL_EVALUATION_SCHEMA_VERSION,
        "evaluation_scope": EVALUATION_SCOPE,
        "name": manifest.name,
        "run_id": manifest.run_id,
        "dataset_manifest": str(manifest.dataset_manifest),
        "dataset_id": str(data_manifest.get("dataset_id", "unknown")),
        "data_class": str(data_manifest.get("data_class", "unknown")),
        "region": str(data_manifest.get("region", "unknown")),
        "output_root": str(manifest.output_root),
        "roi_count": len({roi.name for roi in manifest.rois}),
        "rois": generated_scenarios,
        "splits": _split_counts(manifest.rois),
        "quality_gates": dict(manifest.dataset_validation),
        "mask_stress": _mask_stress_summary(manifest.mask_stress_config),
        "mask_stress_augmented": _mask_stress_enabled(manifest.mask_stress_config),
        "data_validation": data_validation,
        "experiment_manifest": str(experiment_manifest_path),
        "experiment": experiment_summary,
        "coverage_warnings": _coverage_warnings(experiment_summary.get("dataset_summary", {})),
        "stability_summary": stability_summary,
        "architecture_selection": _architecture_selection_summary(
            manifest,
            experiment_summary,
            stability_summary=stability_summary,
        ),
    }


def _markdown_report(summary: dict[str, Any]) -> str:
    experiment = summary.get("experiment", {})
    dataset_summary = experiment.get("dataset_summary", {}) if isinstance(experiment, dict) else {}
    training = experiment.get("training", {}) if isinstance(experiment, dict) else {}
    lines = [
        "# Quasi-real South Pole Evaluation Matrix",
        "",
        f"- evaluation_scope: {summary['evaluation_scope']}",
        f"- data_class: {summary['data_class']}",
        f"- dataset_id: {summary['dataset_id']}",
        f"- region: {summary['region']}",
        f"- roi_count: {summary['roi_count']}",
        f"- mask_stress_augmented: {summary.get('mask_stress_augmented', False)}",
        "",
        "## ROI Splits",
        "",
        "| split | roi | x | y | width | height | scenarios |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
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
    split_counts = summary.get("splits", {})
    if isinstance(split_counts, dict) and split_counts:
        lines.extend(["", "## Split Counts", "", "| split | scenarios |", "|---|---:|"])
        for split, count in split_counts.items():
            lines.append(f"| {split} | {count} |")
    lines.extend(["", "## Dataset Quality", "", "| metric | value |", "|---|---:|"])
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
    mask_stress = summary.get("mask_stress", {})
    lines.extend(["", "## Mask-Stress Coverage", "", "| metric | value |", "|---|---:|"])
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
    coverage_warnings = summary.get("coverage_warnings", [])
    lines.extend(["", "## Sample Coverage Warnings", ""])
    if isinstance(coverage_warnings, list) and coverage_warnings:
        for warning in coverage_warnings:
            lines.append(f"- {warning}")
    else:
        lines.append("- none")
    selection = summary.get("architecture_selection", {})
    if isinstance(selection, dict):
        lines.extend(["", "## Architecture Selection Gate", "", "| field | value |", "|---|---|"])
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
        architectures = selection.get("architectures", {})
        if isinstance(architectures, dict) and architectures:
            lines.extend(
                [
                    "",
                    "### Architecture Selection Metrics",
                    "",
                    "| architecture | run_count | exception_count | failure_count_mean | selection_metric_mean | selection_metric_std | loss_mean | loss_std |",
                    "|---|---:|---:|---:|---:|---:|---:|---:|",
                ]
            )
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
        per_group = selection.get("per_group_winners", {})
        if isinstance(per_group, dict) and per_group:
            lines.extend(
                [
                    "",
                    "### Architecture Per-Group Winners",
                    "",
                    "| group | winner | reason |",
                    "|---|---|---|",
                ]
            )
            for group_name, winner in per_group.items():
                if not isinstance(winner, dict):
                    continue
                lines.append(
                    f"| {group_name} | {winner.get('decision', winner.get('recommended_architecture'))} | {winner.get('reason', '')} |"
                )
        decision_diagnostics = selection.get("decision_diagnostics", {})
        if isinstance(decision_diagnostics, dict):
            lines.extend(["", "## Policy Decision Diagnostics", "", "- architecture_agreement_matrix: present"])
            warnings = decision_diagnostics.get("warnings", [])
            if isinstance(warnings, list) and warnings:
                for warning in warnings:
                    lines.append(f"- warning: {warning}")
            else:
                lines.append("- warning: none")
            matrix = decision_diagnostics.get("architecture_agreement_matrix", {})
            if isinstance(matrix, dict) and matrix:
                lines.extend(
                    [
                        "",
                        "| left_architecture | right_architecture | compared | agreement_rate |",
                        "|---|---|---:|---:|",
                    ]
                )
                for left, row in matrix.items():
                    if not isinstance(row, dict):
                        continue
                    for right, cell in row.items():
                        if not isinstance(cell, dict):
                            continue
                        lines.append(
                            "| "
                            + " | ".join(
                                (
                                    str(left),
                                    str(right),
                                    str(cell.get("compared", 0)),
                                    str(cell.get("agreement_rate", 0.0)),
                                )
                            )
                            + " |"
                        )
            baseline_agreement = decision_diagnostics.get("architecture_baseline_agreement", {})
            if isinstance(baseline_agreement, dict) and baseline_agreement:
                lines.extend(
                    [
                        "",
                        "| architecture | samples | utility_agreement_rate | coverage_heuristic_agreement_rate |",
                        "|---|---:|---:|---:|",
                    ]
                )
                for architecture, agreement in baseline_agreement.items():
                    if not isinstance(agreement, dict):
                        continue
                    lines.append(
                        "| "
                        + " | ".join(
                            (
                                str(architecture),
                                str(agreement.get("sample_count", 0)),
                                str(agreement.get("utility_agreement_rate", 0.0)),
                                str(agreement.get("coverage_heuristic_agreement_rate", 0.0)),
                            )
                        )
                        + " |"
                    )
            per_group_disagreement = decision_diagnostics.get("per_group_disagreement", {})
            if isinstance(per_group_disagreement, dict) and per_group_disagreement:
                lines.extend(
                    [
                        "",
                        "| group | compared | disagreement_rate |",
                        "|---|---:|---:|",
                    ]
                )
                for group, disagreement in per_group_disagreement.items():
                    if not isinstance(disagreement, dict):
                        continue
                    lines.append(
                        f"| {group} | {disagreement.get('compared', 0)} | {disagreement.get('disagreement_rate', 0.0)} |"
                    )
        composite_selection = selection.get("composite_selection", {})
        lines.extend(["", "## Selection Composite Metrics", "", "| field | value |", "|---|---|"])
        lines.append(
            "| selection_composite_weights | "
            + json.dumps(selection.get("selection_composite_weights", {}), ensure_ascii=False, sort_keys=True)
            + " |"
        )
        if isinstance(composite_selection, dict):
            for key in ("status", "decision", "recommended_architecture", "reason"):
                lines.append(f"| composite_{key} | {composite_selection.get(key)} |")
        architectures = selection.get("architectures", {})
        if isinstance(architectures, dict) and architectures:
            lines.extend(
                [
                    "",
                    "| architecture | selection_composite_score_mean | selection_composite_score_std |",
                    "|---|---:|---:|",
                ]
            )
            for architecture, details in architectures.items():
                if not isinstance(details, dict):
                    continue
                composite_stats = details.get("selection_composite_score", {})
                if not isinstance(composite_stats, dict):
                    composite_stats = {}
                lines.append(
                    f"| {architecture} | {composite_stats.get('mean', 0.0)} | {composite_stats.get('std', 0.0)} |"
                )
        action_sensitive = selection.get("action_sensitive_summary", {})
        if isinstance(action_sensitive, dict):
            lines.extend(
                [
                    "",
                    "## Action-Sensitive Metrics",
                    "",
                    "| architecture | metric | mean | std | min | max | samples |",
                    "|---|---|---:|---:|---:|---:|---:|",
                ]
            )
            for architecture, metrics in action_sensitive.items():
                if not isinstance(metrics, dict):
                    continue
                for metric, stats in metrics.items():
                    if not isinstance(stats, dict):
                        continue
                    lines.append(
                        "| "
                        + " | ".join(
                            (
                                str(architecture),
                                str(metric),
                                str(stats.get("mean", 0.0)),
                                str(stats.get("std", 0.0)),
                                str(stats.get("min", 0.0)),
                                str(stats.get("max", 0.0)),
                                str(stats.get("count", 0)),
                            )
                        )
                        + " |"
                    )
        oracle_regret = selection.get("oracle_regret_summary", {})
        if isinstance(oracle_regret, dict):
            lines.extend(
                [
                    "",
                    "## Oracle Regret Summary",
                    "",
                    "| architecture | metric | mean | std | min | max | samples |",
                    "|---|---|---:|---:|---:|---:|---:|",
                ]
            )
            for architecture, metrics in oracle_regret.items():
                if not isinstance(metrics, dict):
                    continue
                for metric, stats in metrics.items():
                    if not isinstance(stats, dict):
                        continue
                    lines.append(
                        "| "
                        + " | ".join(
                            (
                                str(architecture),
                                str(metric),
                                str(stats.get("mean", 0.0)),
                                str(stats.get("std", 0.0)),
                                str(stats.get("min", 0.0)),
                                str(stats.get("max", 0.0)),
                                str(stats.get("count", 0)),
                            )
                        )
                        + " |"
                    )
        sample_discriminativeness = selection.get("sample_discriminativeness", {})
        if isinstance(sample_discriminativeness, dict):
            lines.extend(["", "## Sample Discriminativeness", ""])
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
                    if not isinstance(stats, dict):
                        continue
                    lines.append(
                        "| "
                        + " | ".join(
                            (
                                str(metric),
                                str(stats.get("mean", 0.0)),
                                str(stats.get("std", 0.0)),
                                str(stats.get("min", 0.0)),
                                str(stats.get("max", 0.0)),
                                str(stats.get("count", 0)),
                            )
                        )
                        + " |"
                    )
        per_group_action = selection.get("per_group_action_outcomes", {})
        if isinstance(per_group_action, dict):
            lines.extend(
                [
                    "",
                    "## Per-ROI Action Outcomes",
                    "",
                    "| group | decision | architecture | coverage_regret_mean | composite_regret_mean | selected_expected_coverage_delta_mean | reason |",
                    "|---|---|---|---:|---:|---:|---|",
                ]
            )
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
                    lines.append(
                        "| "
                        + " | ".join(
                            (
                                str(group_name),
                                str(group_summary.get("decision", "inconclusive")),
                                str(architecture),
                                str(coverage_regret.get("mean", 0.0) if isinstance(coverage_regret, dict) else 0.0),
                                str(composite_regret.get("mean", 0.0) if isinstance(composite_regret, dict) else 0.0),
                                str(selected_coverage.get("mean", 0.0) if isinstance(selected_coverage, dict) else 0.0),
                                str(group_summary.get("reason", "")),
                            )
                        )
                        + " |"
                    )
        held_out = selection.get("held_out_test_audit", {})
        if isinstance(held_out, dict):
            lines.extend(["", "## Held-out Test Audit", "", "| field | value |", "|---|---|"])
            for key in (
                "status",
                "used_for_selection",
                "split",
                "validation_decision",
                "validation_recommended_architecture",
                "stable_with_validation",
                "reason",
                "evaluation_scope",
            ):
                if key in held_out:
                    lines.append(f"| {key} | {held_out.get(key)} |")
    quality_gates = summary.get("quality_gates", {})
    lines.extend(["", "## Quality Gates", "", "| gate | value |", "|---|---:|"])
    if isinstance(quality_gates, dict) and quality_gates:
        for key, value in quality_gates.items():
            lines.append(f"| {key} | {value} |")
    else:
        lines.append("| none | not configured |")
    lines.extend(["", "## Architectures", "", "| architecture |", "|---|"])
    for architecture in training.get("architectures", []):
        lines.append(f"| {architecture} |")
    stability_summary = summary.get("stability_summary", {})
    architecture_stability = (
        stability_summary.get("architectures", {}) if isinstance(stability_summary, dict) else {}
    )
    if isinstance(architecture_stability, dict) and architecture_stability:
        lines.extend(
            [
                "",
                "## Architecture Stability",
                "",
                "| architecture | metric | mean | std | min | max | samples |",
                "|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        for architecture, metrics in architecture_stability.items():
            if not isinstance(metrics, dict):
                continue
            for metric, stats in metrics.items():
                if metric == "run_count" or not isinstance(stats, dict):
                    continue
                lines.append(
                    "| "
                    + " | ".join(
                        (
                            str(architecture),
                            str(metric),
                            str(stats.get("mean", 0.0)),
                            str(stats.get("std", 0.0)),
                            str(stats.get("min", 0.0)),
                            str(stats.get("max", 0.0)),
                            str(stats.get("count", 0)),
                        )
                    )
                    + " |"
                )
    loss_distribution = (
        stability_summary.get("loss_distribution", {}) if isinstance(stability_summary, dict) else {}
    )
    if isinstance(loss_distribution, dict) and loss_distribution:
        lines.extend(["", "## Loss Distribution", "", "| metric | mean | std | min | max | samples |", "|---|---:|---:|---:|---:|---:|"])
        for metric, stats in loss_distribution.items():
            if not isinstance(stats, dict):
                continue
            lines.append(
                "| "
                + " | ".join(
                    (
                        str(metric),
                        str(stats.get("mean", 0.0)),
                        str(stats.get("std", 0.0)),
                        str(stats.get("min", 0.0)),
                        str(stats.get("max", 0.0)),
                        str(stats.get("count", 0)),
                    )
                )
                + " |"
            )
    lines.extend(["", "## Baseline Comparison", "", "| section | status |", "|---|---|"])
    lines.append("| utility | present |")
    lines.append("| coverage_heuristic | present |")
    lines.append("| torch_policy | present |")
    policy_ranking = experiment.get("policy_ranking", {}) if isinstance(experiment, dict) else {}
    if isinstance(policy_ranking, list) and policy_ranking:
        lines.extend(
            [
                "",
                "## Policy Ranking",
                "",
                "| rank | policy | final_coverage_rate | total_path_cost | failures |",
                "|---:|---|---:|---:|---:|",
            ]
        )
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
                        str(row.get("total_path_cost", 0.0)),
                        str(row.get("failure_count", 0)),
                    )
                )
                + " |"
            )
    per_group_winners = experiment.get("per_group_winners", {}) if isinstance(experiment, dict) else {}
    if isinstance(per_group_winners, dict) and per_group_winners:
        lines.extend(
            [
                "",
                "## Per-Group Winners",
                "",
                "| group | winner | final_coverage_rate | total_path_cost | failures |",
                "|---|---|---:|---:|---:|",
            ]
        )
        for group_name, winner in per_group_winners.items():
            if not isinstance(winner, dict):
                continue
            lines.append(
                "| "
                + " | ".join(
                    (
                        str(group_name),
                        str(winner.get("policy", "")),
                        str(winner.get("final_coverage_rate", 0.0)),
                        str(winner.get("total_path_cost", 0.0)),
                        str(winner.get("failure_count", 0)),
                    )
                )
                + " |"
            )
    failure_scenarios = experiment.get("failure_scenarios", []) if isinstance(experiment, dict) else []
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
    baseline_delta_summary = (
        stability_summary.get("baseline_deltas", {}) if isinstance(stability_summary, dict) else {}
    )
    if isinstance(baseline_delta_summary, dict) and baseline_delta_summary:
        lines.extend(
            [
                "",
                "## Baseline Delta Summary",
                "",
                "| architecture | metric | mean | std | min | max | samples |",
                "|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        for architecture, metrics in baseline_delta_summary.items():
            if not isinstance(metrics, dict):
                continue
            for metric, stats in metrics.items():
                if not isinstance(stats, dict):
                    continue
                lines.append(
                    "| "
                    + " | ".join(
                        (
                            str(architecture),
                            str(metric),
                            str(stats.get("mean", 0.0)),
                            str(stats.get("std", 0.0)),
                            str(stats.get("min", 0.0)),
                            str(stats.get("max", 0.0)),
                            str(stats.get("count", 0)),
                        )
                    )
                    + " |"
                )
    architecture_deltas = experiment.get("architecture_deltas", {}) if isinstance(experiment, dict) else {}
    if architecture_deltas:
        lines.extend(["", "## Architecture Delta Details", "", "| architecture | baseline | metric | delta |", "|---|---|---|---:|"])
        for architecture, baselines in architecture_deltas.items():
            if not isinstance(baselines, dict):
                continue
            for baseline, metrics in baselines.items():
                if not isinstance(metrics, dict):
                    continue
                for metric, delta in metrics.items():
                    lines.append(f"| {architecture} | {baseline} | {metric} | {delta} |")
    lines.append("")
    return "\n".join(lines)


def _coverage_warnings(dataset_summary: Any) -> list[str]:
    if not isinstance(dataset_summary, dict):
        return ["dataset_summary_missing"]
    warnings: list[str] = []
    unreachable_count = _int_value(dataset_summary.get("unreachable_candidate_count"))
    empty_mask_count = _int_value(dataset_summary.get("empty_action_mask_count"))
    invalid_mask_count = _int_value(dataset_summary.get("invalid_action_mask_count"))
    mask_stress_sample_count = _int_value(dataset_summary.get("mask_stress_sample_count"))
    if unreachable_count == 0:
        warnings.append("no_unreachable_candidates")
    if mask_stress_sample_count == 0 and unreachable_count == 0 and empty_mask_count == 0 and invalid_mask_count == 0:
        warnings.append("no_mask_stress_samples")
    return warnings



inspection_summary = _inspection_summary
would_write = _would_write
run_summary = _run_summary
render_quasi_real_matrix_markdown = _markdown_report
coverage_warnings = _coverage_warnings

_PUBLIC_EXPORTS = [
    "inspection_summary",
    "would_write",
    "run_summary",
    "render_quasi_real_matrix_markdown",
    "coverage_warnings",
]
_PRIVATE_COMPAT_EXPORTS = [
    "_inspection_summary",
    "_would_write",
    "_run_summary",
    "_markdown_report",
    "_coverage_warnings",
]
__all__ = [*_PUBLIC_EXPORTS, *_PRIVATE_COMPAT_EXPORTS]
