from __future__ import annotations

import json
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any

from .lola_south_pole import (
    LolaSouthPoleRoiConfig,
    write_lola_south_pole_scenarios_json,
)
from .manifest import load_data_manifest, validate_data_manifest
from .raster import read_raster_window


QUASI_REAL_EVALUATION_SCHEMA_VERSION = "model-explorer-quasi-real-evaluation/v1"
EVALUATION_SCOPE = "quasi-real evaluation; not real-world generalization benchmark"
REQUIRED_ROI_NAMES = (
    "smooth_high_confidence",
    "rim_or_steep_slope",
    "low_observation_count",
    "mixed_risk",
)
VALID_SPLITS = ("train", "validation", "test", "benchmark")


@dataclass(frozen=True)
class RoiSpec:
    name: str
    split: str
    roi_x: int
    roi_y: int
    roi_width: int
    roi_height: int
    seed: int
    candidate_count: int
    episode_count: int

    @property
    def bounds(self) -> dict[str, int]:
        return {
            "x": self.roi_x,
            "y": self.roi_y,
            "width": self.roi_width,
            "height": self.roi_height,
        }


@dataclass(frozen=True)
class QuasiRealEvaluationManifest:
    path: Path
    name: str
    run_id: str
    dataset_manifest: Path
    output_root: Path
    rois: tuple[RoiSpec, ...]
    mask_stress_config: dict[str, Any]
    selection_config: dict[str, Any]
    dataset_validation: dict[str, Any]
    train_config: dict[str, Any]
    planner_config: dict[str, Any]
    reward_config: dict[str, Any]


def validate_quasi_real_evaluation_manifest(path: str | Path) -> dict[str, Any]:
    manifest = load_quasi_real_evaluation_manifest(path)
    data_validation = validate_data_manifest(manifest.dataset_manifest)
    return _inspection_summary(manifest, status="valid", data_validation=data_validation.to_dict())


def dry_run_quasi_real_evaluation_manifest(path: str | Path) -> dict[str, Any]:
    manifest = load_quasi_real_evaluation_manifest(path)
    data_validation = validate_data_manifest(manifest.dataset_manifest)
    summary = _inspection_summary(manifest, status="dry_run", data_validation=data_validation.to_dict())
    summary["would_write"] = _would_write(manifest)
    return summary


def run_quasi_real_evaluation_manifest(path: str | Path) -> dict[str, Any]:
    from ..policy.experiment import run_experiment_manifest

    manifest = load_quasi_real_evaluation_manifest(path)
    data_validation = validate_data_manifest(manifest.dataset_manifest)
    data_validation.require_valid()
    data_manifest = load_data_manifest(manifest.dataset_manifest)
    raw_dir = Path(data_validation.raw_dir)
    dem_path = raw_dir / _product_file_name(data_manifest, role="shape_map_radius")
    count_path = raw_dir / _product_file_name(data_manifest, role="observation_count")
    resolution = _manifest_resolution(data_manifest)

    manifest.output_root.mkdir(parents=True, exist_ok=True)
    split_groups: dict[str, list[dict[str, Any]]] = {split: [] for split in VALID_SPLITS}
    generated_scenarios: list[dict[str, Any]] = []
    for roi in manifest.rois:
        dem_window = read_raster_window(
            dem_path,
            x=roi.roi_x,
            y=roi.roi_y,
            width=roi.roi_width,
            height=roi.roi_height,
        )
        count_window = read_raster_window(
            count_path,
            x=roi.roi_x,
            y=roi.roi_y,
            width=roi.roi_width,
            height=roi.roi_height,
        )
        scenario_dir = manifest.output_root / "scenarios" / roi.split / roi.name
        source_config = LolaSouthPoleRoiConfig(
            roi_x=roi.roi_x,
            roi_y=roi.roi_y,
            roi_width=roi.roi_width,
            roi_height=roi.roi_height,
            candidate_count=roi.candidate_count,
            episode_count=roi.episode_count,
            seed=roi.seed,
        )
        metadata_extra = {
            "roi_name": roi.name,
            "split": roi.split,
            "roi": roi.bounds,
        }
        if _mask_stress_enabled(manifest.mask_stress_config):
            metadata_extra.update(_mask_stress_metadata(manifest.mask_stress_config))
        scenario_paths = write_lola_south_pole_scenarios_json(
            scenario_dir,
            dem_window.values,
            count_window.values,
            dataset_id=str(data_manifest.get("dataset_id", "unknown")),
            data_class=str(data_manifest.get("data_class", "unknown")),
            region=str(data_manifest.get("region", "unknown")),
            resolution=resolution,
            config=LolaSouthPoleRoiConfig(
                roi_x=0,
                roi_y=0,
                roi_width=roi.roi_width,
                roi_height=roi.roi_height,
                candidate_count=roi.candidate_count,
                episode_count=roi.episode_count,
                seed=roi.seed,
            ),
            source_config=source_config,
            metadata_extra=metadata_extra,
        )
        if _mask_stress_enabled(manifest.mask_stress_config):
            for scenario_path in scenario_paths:
                _apply_mask_stress_to_scenario_json(scenario_path, manifest.mask_stress_config)
        relative_paths = [str(scenario_path.relative_to(manifest.output_root)) for scenario_path in scenario_paths]
        split_groups[roi.split].append({"name": roi.name, "scenarios": relative_paths})
        generated_scenarios.append(
            {
                "name": roi.name,
                "split": roi.split,
                "bounds": roi.bounds,
                "scenario_count": len(scenario_paths),
                "scenarios": relative_paths,
            }
        )

    experiment_manifest_path = manifest.output_root / "experiment.json"
    experiment_payload = _experiment_manifest_payload(manifest, split_groups)
    experiment_manifest_path.write_text(
        json.dumps(experiment_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    experiment_summary = run_experiment_manifest(experiment_manifest_path)

    summary = _run_summary(
        manifest,
        data_manifest=data_manifest,
        data_validation=data_validation.to_dict(),
        generated_scenarios=generated_scenarios,
        experiment_manifest_path=experiment_manifest_path,
        experiment_summary=experiment_summary,
    )
    summary_path = manifest.output_root / "summary.json"
    report_path = manifest.output_root / "matrix-report.md"
    summary["summary_output"] = str(summary_path)
    summary["report_output"] = str(report_path)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    report_path.write_text(_markdown_report(summary), encoding="utf-8")
    return summary


def load_quasi_real_evaluation_manifest(path: str | Path) -> QuasiRealEvaluationManifest:
    manifest_path = Path(path).resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("quasi-real evaluation manifest must be a JSON object")
    schema_version = str(payload.get("schema_version", QUASI_REAL_EVALUATION_SCHEMA_VERSION))
    if schema_version != QUASI_REAL_EVALUATION_SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {QUASI_REAL_EVALUATION_SCHEMA_VERSION!r}")
    base_dir = manifest_path.parent
    seed = int(payload.get("seed", 0))
    candidate_count = int(payload.get("candidate_count", 8))
    episode_count = int(payload.get("episode_count", 1))
    rois = _roi_specs(
        payload.get("rois"),
        seed=seed,
        candidate_count=candidate_count,
        episode_count=episode_count,
    )
    _validate_roi_coverage(rois)
    return QuasiRealEvaluationManifest(
        path=manifest_path,
        name=str(payload.get("name", manifest_path.stem)),
        run_id=str(payload.get("run_id", "default")),
        dataset_manifest=_resolve_path(base_dir, payload.get("dataset_manifest")),
        output_root=_resolve_path(base_dir, payload.get("output_root", "data/processed/quasi_real/evaluation-matrix")),
        rois=rois,
        mask_stress_config=_normalize_mask_stress_config(payload.get("mask_stress", {})),
        selection_config=_normalize_selection_config(payload.get("selection", {})),
        dataset_validation=dict(payload.get("dataset_validation", {})),
        train_config=dict(payload.get("train", _default_train_config(seed))),
        planner_config=dict(payload.get("planner", {"backend": "contract_cost"})),
        reward_config=dict(payload.get("reward", _default_reward_config())),
    )


def _roi_specs(
    value: Any,
    *,
    seed: int,
    candidate_count: int,
    episode_count: int,
) -> tuple[RoiSpec, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("rois must be a non-empty list")
    specs: list[RoiSpec] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(f"rois[{index}] must be an object")
        split = str(item.get("split", "train"))
        if split not in VALID_SPLITS:
            raise ValueError(f"rois[{index}].split must be one of {', '.join(VALID_SPLITS)}")
        width = int(item.get("roi_width", 0))
        height = int(item.get("roi_height", 0))
        if width <= 0 or height <= 0:
            raise ValueError(f"rois[{index}] ROI width and height must be positive")
        specs.append(
            RoiSpec(
                name=str(item.get("name", f"roi-{index}")),
                split=split,
                roi_x=int(item.get("roi_x", 0)),
                roi_y=int(item.get("roi_y", 0)),
                roi_width=width,
                roi_height=height,
                seed=int(item.get("seed", seed + index)),
                candidate_count=int(item.get("candidate_count", candidate_count)),
                episode_count=int(item.get("episode_count", episode_count)),
            )
        )
    return tuple(specs)


def _validate_roi_coverage(rois: tuple[RoiSpec, ...]) -> None:
    names = {roi.name for roi in rois}
    missing = [name for name in REQUIRED_ROI_NAMES if name not in names]
    if missing:
        raise ValueError(f"quasi-real evaluation manifest is missing required ROI: {missing[0]}")


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


def _experiment_manifest_payload(
    manifest: QuasiRealEvaluationManifest,
    split_groups: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    splits = {split: groups for split, groups in split_groups.items() if groups}
    return {
        "schema_version": "model-explorer-experiment/v1",
        "name": manifest.name,
        "run_id": manifest.run_id,
        "splits": splits,
        "max_candidates": max(roi.candidate_count for roi in manifest.rois),
        "planner": dict(manifest.planner_config),
        "reward": dict(manifest.reward_config),
        "dataset_validation": dict(manifest.dataset_validation),
        "train": dict(manifest.train_config),
        "outputs": {"root": "out"},
    }


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


def _normalize_mask_stress_config(value: Any) -> dict[str, Any]:
    if value is None:
        value = {}
    if not isinstance(value, dict):
        raise ValueError("mask_stress must be an object")
    enabled = bool(value.get("enabled", False))
    fields = value.get("missing_experimental_fields", ())
    if fields is None:
        fields = ()
    if not isinstance(fields, (list, tuple)):
        raise ValueError("mask_stress.missing_experimental_fields must be a list")
    return {
        "enabled": enabled,
        "label": str(value.get("label", "mask_stress_augmented")),
        "profile": str(value.get("profile", "deterministic-v1")),
        "unreachable_candidate_count": max(0, int(value.get("unreachable_candidate_count", 1))),
        "missing_experimental_fields": tuple(str(field) for field in fields),
    }


def _mask_stress_enabled(config: dict[str, Any]) -> bool:
    return bool(config.get("enabled", False))


def _mask_stress_summary(config: dict[str, Any]) -> dict[str, Any]:
    normalized = _normalize_mask_stress_config(config)
    return {
        "enabled": bool(normalized["enabled"]),
        "label": str(normalized["label"]),
        "profile": str(normalized["profile"]),
        "unreachable_candidate_count": int(normalized["unreachable_candidate_count"]),
        "missing_experimental_fields": list(normalized["missing_experimental_fields"]),
    }


def _mask_stress_metadata(config: dict[str, Any]) -> dict[str, Any]:
    normalized = _normalize_mask_stress_config(config)
    return {
        "mask_stress_augmented": True,
        "mask_stress_label": str(normalized["label"]),
        "mask_stress_profile": str(normalized["profile"]),
        "evaluation_scope": EVALUATION_SCOPE,
    }


def _normalize_selection_config(value: Any) -> dict[str, Any]:
    if value is None:
        value = {}
    if not isinstance(value, dict):
        raise ValueError("selection must be an object")
    mode = str(value.get("mode", "max"))
    if mode not in {"max", "min"}:
        raise ValueError("selection.mode must be 'max' or 'min'")
    return {
        "enabled": bool(value.get("enabled", False)),
        "metric": str(value.get("metric", "torch_policy.final_coverage_rate")),
        "mode": mode,
        "min_seed_count": _optional_selection_int(value, "min_seed_count"),
        "min_architecture_count": _optional_selection_int(value, "min_architecture_count"),
        "min_roi_group_count": _optional_selection_int(value, "min_roi_group_count"),
        "min_unreachable_candidate_count": _optional_selection_int(value, "min_unreachable_candidate_count"),
        "min_mask_stress_sample_count": _optional_selection_int(value, "min_mask_stress_sample_count"),
        "uncertainty_multiplier": float(value.get("uncertainty_multiplier", 1.0)),
    }


def _optional_selection_int(value: dict[str, Any], key: str) -> int | None:
    return None if value.get(key) is None else int(value[key])


def _selection_config_summary(config: dict[str, Any]) -> dict[str, Any]:
    normalized = _normalize_selection_config(config)
    return {key: value for key, value in normalized.items() if value is not None}


def _apply_mask_stress_to_scenario_json(path: Path, config: dict[str, Any]) -> None:
    normalized = _normalize_mask_stress_config(config)
    if not bool(normalized["enabled"]):
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("scenario JSON root must be an object")
    metadata = payload.setdefault("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("scenario metadata must be an object")
    metadata.update(_mask_stress_metadata(normalized))
    snapshots = payload.get("snapshots", [])
    if not isinstance(snapshots, list):
        raise ValueError("scenario snapshots must be a list")
    for snapshot in snapshots:
        if not isinstance(snapshot, dict):
            continue
        _apply_mask_stress_to_contract_payload(snapshot, normalized)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _apply_mask_stress_to_contract_payload(contract: dict[str, Any], config: dict[str, Any]) -> None:
    goals = contract.get("top_goals", [])
    if not isinstance(goals, list) or not goals:
        return
    missing_fields = tuple(config.get("missing_experimental_fields", ()))
    for index, goal in enumerate(goals):
        if not isinstance(goal, dict):
            continue
        removed_fields = []
        for field in missing_fields:
            if field in goal:
                goal.pop(field)
                removed_fields.append(field)
        if removed_fields:
            goal["mask_stress_missing_fields"] = removed_fields
        goal["mask_stress_augmented"] = True

    reachable_indices = [
        index
        for index, goal in enumerate(goals)
        if isinstance(goal, dict) and bool(goal.get("reachable", False))
    ]
    target_unreachable_count = int(config.get("unreachable_candidate_count", 0))
    changed = 0
    for index in reversed(reachable_indices):
        if changed >= target_unreachable_count:
            break
        if len(reachable_indices) - changed <= 1:
            break
        goal = goals[index]
        if not isinstance(goal, dict):
            continue
        goal["reachable"] = False
        goal["mask_stress_reason"] = "deterministic_unreachable"
        changed += 1

    observation_update = contract.setdefault("observation_update", {})
    if isinstance(observation_update, dict):
        observation_update.update(_mask_stress_metadata(config))
    experimental_fields = contract.get("experimental_fields", [])
    if isinstance(experimental_fields, list):
        for field in ("mask_stress_augmented", "mask_stress_reason", "mask_stress_missing_fields"):
            if field not in experimental_fields:
                experimental_fields.append(field)


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


def _architecture_selection_summary(
    manifest: QuasiRealEvaluationManifest,
    experiment: Any,
    *,
    stability_summary: dict[str, Any],
) -> dict[str, Any]:
    config = _normalize_selection_config(manifest.selection_config)
    metric = str(config["metric"])
    mode = str(config["mode"])
    runs = _training_runs(experiment)
    architectures = _manifest_architectures(manifest)
    seeds = _manifest_seeds(manifest)
    dataset_summary = experiment.get("dataset_summary", {}) if isinstance(experiment, dict) else {}
    per_architecture_values: dict[str, list[float]] = {architecture: [] for architecture in architectures}
    loss_values: dict[str, list[float]] = {architecture: [] for architecture in architectures}
    exception_counts: dict[str, int] = {architecture: 0 for architecture in architectures}

    for run in runs:
        architecture = str(run.get("architecture", "unknown")) if isinstance(run, dict) else "unknown"
        if architecture not in per_architecture_values:
            per_architecture_values[architecture] = []
            loss_values[architecture] = []
            exception_counts[architecture] = 0
        if not isinstance(run, dict):
            exception_counts[architecture] += 1
            continue
        if "error" in run or "exception" in run:
            exception_counts[architecture] += 1
        _append_metric({architecture: per_architecture_values[architecture]}, architecture, _run_selection_metric(run, metric))
        _append_metric({architecture: loss_values[architecture]}, architecture, run.get("loss"))

    metric_stats = {
        architecture: _numeric_summary(tuple(values))
        for architecture, values in per_architecture_values.items()
    }
    loss_stats = {
        architecture: _numeric_summary(tuple(values))
        for architecture, values in loss_values.items()
    }
    quality_gates = _selection_quality_gates(
        config,
        architectures=architectures,
        seeds=seeds,
        runs=runs,
        metric_stats=metric_stats,
        loss_stats=loss_stats,
        exception_counts=exception_counts,
        dataset_summary=dataset_summary,
    )
    architecture_stability = (
        stability_summary.get("architectures", {}) if isinstance(stability_summary, dict) else {}
    )
    architecture_details = {}
    for architecture in architectures:
        stability_metrics = architecture_stability.get(architecture, {}) if isinstance(architecture_stability, dict) else {}
        architecture_details[architecture] = {
            "run_count": _architecture_run_count(runs, architecture),
            "exception_count": exception_counts.get(architecture, 0),
            "failure_count": (
                stability_metrics.get("torch_policy.failure_count", _numeric_summary(()))
                if isinstance(stability_metrics, dict)
                else _numeric_summary(())
            ),
            "selection_metric": metric_stats.get(architecture, _numeric_summary(())),
            "loss": loss_stats.get(architecture, _numeric_summary(())),
            "baseline_deltas": (
                stability_summary.get("baseline_deltas", {}).get(architecture, {})
                if isinstance(stability_summary.get("baseline_deltas", {}), dict)
                else {}
            ),
            "dataset": {
                key: stability_metrics.get(f"dataset.{key}", {})
                for key in (
                    "unreachable_candidate_count",
                    "padding_candidate_count",
                    "missing_experimental_feature_candidate_count",
                    "mask_stress_sample_count",
                )
            }
            if isinstance(stability_metrics, dict)
            else {},
        }

    base_summary: dict[str, Any] = {
        "enabled": bool(config["enabled"]),
        "metric": metric,
        "mode": mode,
        "decision_boundary": "recommended_architecture or inconclusive based on seed variance",
        "quality_gates": quality_gates,
        "architectures": architecture_details,
        "loss_distribution": stability_summary.get("loss_distribution", {}) if isinstance(stability_summary, dict) else {},
        "baseline_delta_distribution": (
            stability_summary.get("baseline_deltas", {}) if isinstance(stability_summary, dict) else {}
        ),
        "per_group_winners": _per_group_architecture_winners(
            runs,
            metric=metric,
            mode=mode,
            uncertainty_multiplier=float(config["uncertainty_multiplier"]),
        ),
        "mask_stress_coverage": _mask_stress_coverage(dataset_summary),
        "evaluation_scope": EVALUATION_SCOPE,
    }
    if not bool(config["enabled"]):
        base_summary.update(
            {
                "status": "not_configured",
                "decision": "inconclusive",
                "recommended_architecture": None,
                "reason": "selection.enabled is false",
            }
        )
        return base_summary
    if quality_gates["status"] != "passed":
        base_summary.update(
            {
                "status": "failed",
                "decision": "inconclusive",
                "recommended_architecture": None,
                "reason": "selection quality gates failed",
            }
        )
        return base_summary
    decision = _selection_decision(
        metric_stats,
        metric=metric,
        mode=mode,
        uncertainty_multiplier=float(config["uncertainty_multiplier"]),
    )
    base_summary.update(decision)
    return base_summary


def _training_runs(experiment: Any) -> list[dict[str, Any]]:
    if not isinstance(experiment, dict):
        return []
    training = experiment.get("training", {})
    runs = training.get("runs", []) if isinstance(training, dict) else []
    return [run for run in runs if isinstance(run, dict)] if isinstance(runs, list) else []


def _manifest_architectures(manifest: QuasiRealEvaluationManifest) -> list[str]:
    raw = manifest.train_config.get("architectures")
    if isinstance(raw, list) and raw:
        return [str(architecture) for architecture in raw]
    architecture = manifest.train_config.get("architecture", "mlp_v1")
    return ["mlp_v1" if architecture is None or str(architecture).strip() == "" else str(architecture)]


def _manifest_seeds(manifest: QuasiRealEvaluationManifest) -> list[int]:
    raw = manifest.train_config.get("seeds")
    if isinstance(raw, list) and raw:
        return [int(seed) for seed in raw]
    return [int(manifest.train_config.get("seed", 0))]


def _architecture_run_count(runs: list[dict[str, Any]], architecture: str) -> int:
    return sum(1 for run in runs if str(run.get("architecture", "unknown")) == architecture)


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


def _selection_decision(
    architecture_stats: dict[str, dict[str, Any]],
    *,
    metric: str,
    mode: str,
    uncertainty_multiplier: float,
) -> dict[str, Any]:
    candidates = [
        (architecture, stats)
        for architecture, stats in architecture_stats.items()
        if isinstance(stats, dict) and int(stats.get("count", 0)) > 0 and isfinite(float(stats.get("mean", 0.0)))
    ]
    if not candidates:
        return {
            "status": "inconclusive",
            "decision": "inconclusive",
            "recommended_architecture": None,
            "reason": f"no finite values for {metric}",
        }
    reverse = mode == "max"
    candidates.sort(key=lambda item: float(item[1].get("mean", 0.0)), reverse=reverse)
    best_architecture, best_stats = candidates[0]
    if len(candidates) == 1:
        return {
            "status": "selected",
            "decision": str(best_architecture),
            "recommended_architecture": str(best_architecture),
            "reason": f"only architecture with finite {metric}",
        }
    second_architecture, second_stats = candidates[1]
    best_mean = float(best_stats.get("mean", 0.0))
    second_mean = float(second_stats.get("mean", 0.0))
    margin = best_mean - second_mean if mode == "max" else second_mean - best_mean
    uncertainty = max(float(best_stats.get("std", 0.0)), float(second_stats.get("std", 0.0))) * uncertainty_multiplier
    if margin <= uncertainty:
        return {
            "status": "inconclusive",
            "decision": "inconclusive",
            "recommended_architecture": None,
            "reason": (
                f"best {metric} margin {margin} between {best_architecture} and "
                f"{second_architecture} is within seed variance {uncertainty}"
            ),
            "best_candidate": str(best_architecture),
            "runner_up": str(second_architecture),
            "margin": margin,
            "uncertainty": uncertainty,
        }
    return {
        "status": "selected",
        "decision": str(best_architecture),
        "recommended_architecture": str(best_architecture),
        "reason": (
            f"best {metric} margin {margin} over {second_architecture} exceeds "
            f"seed variance threshold {uncertainty}"
        ),
        "runner_up": str(second_architecture),
        "margin": margin,
        "uncertainty": uncertainty,
    }


def _per_group_architecture_winners(
    runs: list[dict[str, Any]],
    *,
    metric: str,
    mode: str,
    uncertainty_multiplier: float,
) -> dict[str, Any]:
    group_values: dict[str, dict[str, list[float]]] = {}
    for run in runs:
        architecture = str(run.get("architecture", "unknown"))
        evaluation = run.get("validation_evaluation", {})
        groups = evaluation.get("groups", {}) if isinstance(evaluation, dict) else {}
        if not isinstance(groups, dict):
            continue
        for group_name, group_evaluation in groups.items():
            value = _evaluation_metric(group_evaluation, metric)
            group_architectures = group_values.setdefault(str(group_name), {})
            _append_metric(group_architectures, architecture, value)
    winners: dict[str, Any] = {}
    for group_name, architecture_values in group_values.items():
        stats = {
            architecture: _numeric_summary(tuple(values))
            for architecture, values in architecture_values.items()
        }
        decision = _selection_decision(
            stats,
            metric=metric,
            mode=mode,
            uncertainty_multiplier=uncertainty_multiplier,
        )
        decision["architectures"] = stats
        winners[group_name] = decision
    return winners


def _stability_summary(experiment: Any) -> dict[str, Any]:
    if not isinstance(experiment, dict):
        return {"architectures": {}, "loss_distribution": {}, "baseline_deltas": {}}
    training = experiment.get("training", {})
    runs = training.get("runs", []) if isinstance(training, dict) else []
    if not isinstance(runs, list):
        runs = []

    architecture_values: dict[str, dict[str, list[float]]] = {}
    architecture_run_counts: dict[str, int] = {}
    baseline_values: dict[str, dict[str, list[float]]] = {}
    loss_values: dict[str, list[float]] = {}
    for run in runs:
        if not isinstance(run, dict):
            continue
        architecture = str(run.get("architecture", "unknown"))
        architecture_run_counts[architecture] = architecture_run_counts.get(architecture, 0) + 1
        arch_metrics = architecture_values.setdefault(architecture, {})
        for metric in ("loss", "policy_loss", "value_loss", "entropy"):
            _append_metric(arch_metrics, metric, run.get(metric))
            _append_metric(loss_values, metric, run.get(metric))
        run_dataset = run.get("dataset_summary", {})
        if isinstance(run_dataset, dict):
            for metric in (
                "unreachable_candidate_count",
                "padding_candidate_count",
                "missing_experimental_feature_candidate_count",
                "mask_stress_sample_count",
            ):
                _append_metric(arch_metrics, f"dataset.{metric}", run_dataset.get(metric))
        validation = run.get("validation_evaluation", {})
        if isinstance(validation, dict) and "aggregate" in validation and isinstance(validation["aggregate"], dict):
            validation = validation["aggregate"]
        torch_policy = validation.get("torch_policy", {}) if isinstance(validation, dict) else {}
        if isinstance(torch_policy, dict):
            for metric in ("final_coverage_rate", "total_path_cost", "average_risk", "failure_count"):
                _append_metric(arch_metrics, f"torch_policy.{metric}", torch_policy.get(metric))
        deltas = run.get("baseline_deltas", {})
        arch_deltas = baseline_values.setdefault(architecture, {})
        if isinstance(deltas, dict):
            for baseline_name, metrics in deltas.items():
                if not isinstance(metrics, dict):
                    continue
                for metric, value in metrics.items():
                    _append_metric(arch_deltas, f"{baseline_name}.{metric}", value)

    return {
        "architectures": {
            architecture: {
                "run_count": architecture_run_counts.get(architecture, 0),
                **{metric: _numeric_summary(tuple(values)) for metric, values in metrics.items()},
            }
            for architecture, metrics in architecture_values.items()
        },
        "loss_distribution": {
            metric: _numeric_summary(tuple(values))
            for metric, values in loss_values.items()
        },
        "baseline_deltas": {
            architecture: {
                metric: _numeric_summary(tuple(values))
                for metric, values in metrics.items()
            }
            for architecture, metrics in baseline_values.items()
        },
    }


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


def _roi_summary(roi: RoiSpec) -> dict[str, Any]:
    return {
        "name": roi.name,
        "split": roi.split,
        "bounds": roi.bounds,
        "seed": roi.seed,
        "candidate_count": roi.candidate_count,
        "episode_count": roi.episode_count,
    }


def _split_counts(rois: tuple[RoiSpec, ...]) -> dict[str, int]:
    counts = {split: 0 for split in VALID_SPLITS}
    for roi in rois:
        counts[roi.split] += roi.episode_count
    return {split: count for split, count in counts.items() if count}


def _product_file_name(manifest: dict[str, Any], *, role: str) -> str:
    products = manifest.get("products", [])
    if not isinstance(products, list):
        raise ValueError("manifest.products must be a list")
    for product in products:
        if not isinstance(product, dict) or product.get("role") != role:
            continue
        for file_info in product.get("files", []):
            if isinstance(file_info, dict) and str(file_info.get("name", "")).lower().endswith(".jp2"):
                return str(file_info["name"])
    raise ValueError(f"manifest is missing product file for role {role!r}")


def _manifest_resolution(manifest: dict[str, Any]) -> float:
    projection = manifest.get("projection", {})
    if isinstance(projection, dict) and projection.get("map_scale_meters_per_pixel") is not None:
        return float(projection["map_scale_meters_per_pixel"])
    return 1.0


def _resolve_path(base_dir: Path, value: Any) -> Path:
    if value is None:
        raise ValueError("path value is required")
    path = Path(str(value))
    return path.resolve() if path.is_absolute() else (base_dir / path).resolve()


def _default_train_config(seed: int) -> dict[str, Any]:
    return {
        "seed": int(seed),
        "architectures": ["mlp_v1", "mlp_missing_v1", "candidate_attention_v1"],
        "hidden_size": 16,
        "learning_rate": 1.0e-3,
        "epochs": 1,
        "evaluate_trained_policy": True,
    }


def _default_reward_config() -> dict[str, float]:
    return {
        "path_cost_weight": 0.1,
        "path_cost_normalizer": 1000.0,
        "risk_weight": 0.2,
        "failure_penalty": 1.0,
    }
