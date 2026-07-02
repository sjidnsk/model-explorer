from __future__ import annotations

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
from .architecture_selection import _architecture_selection_summary
from .report_sections import (
    render_action_sensitive_section,
    render_architecture_delta_details_section,
    render_architecture_selection_section,
    render_architecture_stability_section,
    render_architectures_section,
    render_baseline_comparison_section,
    render_baseline_delta_summary_section,
    render_coverage_warnings_section,
    render_dataset_quality_section,
    render_failure_scenarios_section,
    render_header_section,
    render_held_out_test_audit_section,
    render_loss_distribution_section,
    render_mask_stress_section,
    render_oracle_regret_section,
    render_per_group_winners_section,
    render_per_roi_action_outcomes_section,
    render_policy_decision_diagnostics_section,
    render_policy_ranking_section,
    render_quality_gates_section,
    render_roi_splits_section,
    render_sample_discriminativeness_section,
    render_selection_composite_section,
    render_split_counts_section,
)
from .stability import _stability_summary


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
    selection = summary.get("architecture_selection", {})
    stability_summary = summary.get("stability_summary", {})
    lines = render_header_section(summary)
    lines.extend(render_roi_splits_section(summary))
    lines.extend(render_split_counts_section(summary))
    lines.extend(render_dataset_quality_section(dataset_summary))
    lines.extend(render_mask_stress_section(summary, dataset_summary))
    lines.extend(render_coverage_warnings_section(summary))
    lines.extend(render_architecture_selection_section(selection))
    lines.extend(render_policy_decision_diagnostics_section(selection))
    lines.extend(render_selection_composite_section(selection))
    lines.extend(render_action_sensitive_section(selection))
    lines.extend(render_oracle_regret_section(selection))
    lines.extend(render_sample_discriminativeness_section(selection))
    lines.extend(render_per_roi_action_outcomes_section(selection))
    lines.extend(render_held_out_test_audit_section(selection))
    lines.extend(render_quality_gates_section(summary))
    lines.extend(render_architectures_section(training))
    lines.extend(render_architecture_stability_section(stability_summary))
    lines.extend(render_loss_distribution_section(stability_summary))
    lines.extend(render_baseline_comparison_section())
    lines.extend(render_policy_ranking_section(experiment))
    lines.extend(render_per_group_winners_section(experiment))
    lines.extend(render_failure_scenarios_section(experiment))
    lines.extend(render_baseline_delta_summary_section(stability_summary))
    lines.extend(render_architecture_delta_details_section(experiment))
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
