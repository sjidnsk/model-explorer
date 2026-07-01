from __future__ import annotations

from pathlib import Path
from typing import Any

from ..policy.dataset import summarize_rollout_dataset, summarize_teacher_quality_gates, validate_rollout_dataset
from ..policy.evaluation import evaluate_policy_baseline_scenarios
from ..policy.planning_adapters import planner_from_config
from ..policy.rollout_io import write_rollout_episodes_jsonl
from .environment import _environment_metadata
from .evaluation import (
    _aggregate_rollout_metrics,
    _all_split_episodes,
    _all_split_scenarios,
    _collect_episodes,
    _comparison_from_evaluation,
    _evaluation_split_name,
    _grouped_evaluation,
    _load_split_scenarios,
    _run_reward_ablations,
    _should_evaluate_trained_policy,
    _split_summaries,
)
from .manifest import (
    EXPERIMENT_SCHEMA_VERSION,
    ExperimentManifest,
    ExperimentScenarioGroup,
    ExperimentSplit,
    _ensure_parent_dir,
    _manifest_inspection_summary,
    _resolved_manifest_payload,
    _would_write_paths,
    _write_json,
    load_experiment_manifest,
)
from .reports import _daily_report_summary, _markdown_report
from .selection import _baseline_deltas
from .training_matrix import _run_training


def run_experiment_manifest(path: str | Path) -> dict[str, Any]:
    manifest = load_experiment_manifest(path)
    planner = planner_from_config(manifest.planner_config)
    split_scenarios = _load_split_scenarios(manifest)
    scenarios = _all_split_scenarios(split_scenarios)
    split_episodes = {
        split_name: _collect_episodes(
            split_items,
            planner=planner,
            max_candidates=manifest.max_candidates,
            selection_strategy=manifest.selection_strategy,
            reward_config=manifest.reward_config,
        )
        for split_name, split_items in split_scenarios.items()
    }
    episodes = _all_split_episodes(split_episodes)
    _ensure_parent_dir(manifest.rollout_output)
    write_rollout_episodes_jsonl(manifest.rollout_output, episodes)
    dataset_summary = (
        validate_rollout_dataset(episodes, gates=manifest.dataset_validation)
        if manifest.dataset_validation is not None
        else summarize_rollout_dataset(episodes)
    )
    if manifest.dataset_summary_output is not None:
        _write_json(manifest.dataset_summary_output, dataset_summary)

    evaluation_split_name = _evaluation_split_name(manifest)
    evaluation_scenarios = split_scenarios[evaluation_split_name]
    evaluation_groups = manifest.splits[evaluation_split_name].scenario_groups
    evaluation_paths = manifest.splits[evaluation_split_name].scenarios
    base_evaluation = evaluate_policy_baseline_scenarios(evaluation_scenarios, planning_adapter=planner)
    evaluation = (
        _grouped_evaluation(
            evaluation_groups,
            evaluation_scenarios,
            evaluation_paths,
            planner=planner,
            aggregate=base_evaluation,
        )
        if manifest.explicit_splits or len(evaluation_groups) > 1
        else base_evaluation
    )

    summary = {
        "schema_version": manifest.schema_version,
        "experiment_name": manifest.experiment_name,
        "run_id": manifest.run_id,
        "scenario_count": len(scenarios),
        "group_count": len(manifest.scenario_groups),
        "groups": [
            {"name": group.name, "scenario_count": len(group.scenarios)}
            for group in manifest.scenario_groups
        ],
        "planner": str(manifest.planner_config.get("backend", "contract_cost")),
        "rollout_output": str(manifest.rollout_output),
        "evaluation_output": str(manifest.evaluation_output),
        "dataset_summary_output": None
        if manifest.dataset_summary_output is None
        else str(manifest.dataset_summary_output),
        "selection_strategy": manifest.selection_strategy,
        "output_layout": {
            "root": None if manifest.output_root is None else str(manifest.output_root),
            "run_dir": None if manifest.run_output_dir is None else str(manifest.run_output_dir),
        },
        "transition_count": sum(len(episode.transitions) for episode in episodes),
        "rollout_metrics": _aggregate_rollout_metrics(episodes),
        "dataset_summary": dataset_summary,
        "split_summaries": _split_summaries(manifest, split_episodes),
        "resolved_manifest_output": str(manifest.resolved_manifest_output),
        "environment": _environment_metadata(base_dir=Path(path).parent),
        "reward": dict(manifest.reward_config or {}),
    }
    _write_json(manifest.resolved_manifest_output, _resolved_manifest_payload(manifest))
    if manifest.reward_ablations:
        summary["reward_ablations"] = _run_reward_ablations(manifest, scenarios, planner=planner)
    if manifest.train_config is not None:
        summary["training"] = _run_training(
            episodes,
            manifest.train_config,
            base_dir=Path(path).parent,
            run_output_dir=manifest.run_output_dir,
            scenarios=scenarios,
            planner=planner,
            max_candidates=manifest.max_candidates,
            reward_config=manifest.reward_config,
            train_episodes=split_episodes.get("train") if manifest.explicit_splits else None,
            train_scenarios=split_scenarios.get("train") if manifest.explicit_splits else None,
            validation_episodes=split_episodes.get("validation") if manifest.explicit_splits else None,
            validation_scenarios=split_scenarios.get("validation") if manifest.explicit_splits else None,
            validation_groups=(
                manifest.splits["validation"].scenario_groups
                if manifest.explicit_splits and "validation" in manifest.splits
                else None
            ),
            validation_paths=(
                manifest.splits["validation"].scenarios
                if manifest.explicit_splits and "validation" in manifest.splits
                else None
            ),
            test_scenarios=split_scenarios.get("test") if manifest.explicit_splits else None,
            test_groups=(
                manifest.splits["test"].scenario_groups
                if manifest.explicit_splits and "test" in manifest.splits
                else None
            ),
            test_paths=(
                manifest.splits["test"].scenarios
                if manifest.explicit_splits and "test" in manifest.splits
                else None
            ),
        )
        if _should_evaluate_trained_policy(manifest.train_config):
            from ..policy.training import load_policy_checkpoint

            trained_policy = load_policy_checkpoint(summary["training"]["checkpoint"])
            trained_evaluation = evaluate_policy_baseline_scenarios(
                evaluation_scenarios,
                torch_policy=trained_policy,
                planning_adapter=planner,
            )
            evaluation = (
                _grouped_evaluation(
                    evaluation_groups,
                    evaluation_scenarios,
                    evaluation_paths,
                    planner=planner,
                    aggregate=trained_evaluation,
                    torch_policy=trained_policy,
                )
                if manifest.explicit_splits or len(evaluation_groups) > 1
                else trained_evaluation
            )
            summary["training"]["baseline_evaluation"] = _comparison_from_evaluation(evaluation)
            summary["baseline_deltas"] = _baseline_deltas(_comparison_from_evaluation(evaluation))
    summary.update(_daily_report_summary(summary, evaluation))
    _write_json(manifest.evaluation_output, evaluation)
    if manifest.report_output is not None:
        _ensure_parent_dir(manifest.report_output)
        manifest.report_output.write_text(_markdown_report(summary, evaluation), encoding="utf-8")
        summary["report_output"] = str(manifest.report_output)
    return summary


def validate_experiment_manifest(path: str | Path) -> dict[str, Any]:
    manifest = load_experiment_manifest(path)
    split_scenarios = _load_split_scenarios(manifest)
    return _manifest_inspection_summary(manifest, split_scenarios=split_scenarios, status="valid")


def dry_run_experiment_manifest(path: str | Path) -> dict[str, Any]:
    manifest = load_experiment_manifest(path)
    split_scenarios = _load_split_scenarios(manifest)
    summary = _manifest_inspection_summary(manifest, split_scenarios=split_scenarios, status="dry_run")
    summary["would_write"] = _would_write_paths(manifest, base_dir=Path(path).parent)
    summary["training_enabled"] = manifest.train_config is not None
    return summary


# Explicit compatibility aliases for symbols that lived in this module before
# the runner split. Keep this list narrow; temporary imports stay out of
# runner.__all__.
from .environment import _git_metadata, _git_output
from .evaluation import (
    _all_split_episodes,
    _reward_ablations,
)
from .manifest import (
    _evaluation_groups_from_splits,
    _manifest_splits,
    _optional_mapping,
    _planner_config_with_sidecar,
    _resolve_output_path,
    _resolve_path,
    _scenario_groups,
    _scenario_paths,
    _split_scenario_groups,
    _system_calibration_config,
    _system_path_feedback_gate_config,
    _system_path_feedback_gate_enabled,
    _system_sample_quality_config,
    _training_would_write_paths,
    _unique_scenario_paths,
)
from .reports import (
    _architecture_deltas,
    _failure_scenarios,
    _gate_summary,
    _loss_summary,
    _ordered_policy_names,
    _per_group_winners,
    _policy_ranking,
)
from .selection import (
    _BASELINE_DELTA_METRICS,
    _baseline_names,
    _best_selection_record,
    _calibration_profile_key,
    _calibration_profile_records,
    _calibration_recommendation,
    _collect_confidence_values,
    _distillation_evaluation_scope,
    _distillation_profile_dimension_enabled,
    _distillation_run_selection_decision,
    _distillation_stability_group_summary,
    _distillation_stability_summary,
    _excluded_run_record,
    _metric_value,
    _multi_seed_delta_summary,
    _multi_seed_evaluation_summary,
    _nested_metric_mean,
    _normalize_curriculum_profile_name,
    _normalize_teacher_weight_name,
    _normalize_training_source_name,
    _numeric_stats,
    _numeric_stats_by_metric,
    _optional_numeric,
    _recommended_run,
    _run_curriculum_profile_name,
    _select_best_training_run,
    _teacher_agreement_summary,
    _teacher_quality_gate_failed,
    _teacher_quality_gate_status,
    _training_distillation_matrix,
    _training_run_metric,
    _training_source_comparison,
)
from .training_matrix import (
    _builtin_teacher_margin_curriculum_profile,
    _coerce_teacher_margin_curriculum_profile,
    _normalize_training_architecture_name,
    _path_parent_contains_any_placeholder,
    _path_parent_contains_placeholder,
    _split_training_episodes,
    _split_training_scenarios,
    _training_architecture_config,
    _training_architectures,
    _training_episodes_for_source,
    _training_output_path,
    _training_seeds,
    _training_source_selection_strategies,
    _training_teacher_imitation_weights,
    _training_teacher_margin_curriculum_profiles,
)

__all__ = (
    "EXPERIMENT_SCHEMA_VERSION",
    "ExperimentScenarioGroup",
    "ExperimentSplit",
    "ExperimentManifest",
    "load_experiment_manifest",
    "run_experiment_manifest",
    "validate_experiment_manifest",
    "dry_run_experiment_manifest",
    "_planner_config_with_sidecar",
    "_scenario_groups",
    "_manifest_splits",
    "_split_scenario_groups",
    "_evaluation_groups_from_splits",
    "_unique_scenario_paths",
    "_scenario_paths",
    "_resolve_path",
    "_resolve_output_path",
    "_ensure_parent_dir",
    "_write_json",
    "_optional_mapping",
    "_system_calibration_config",
    "_system_path_feedback_gate_enabled",
    "_system_path_feedback_gate_config",
    "_system_sample_quality_config",
    "_reward_ablations",
    "_collect_episodes",
    "_load_split_scenarios",
    "_all_split_scenarios",
    "_all_split_episodes",
    "_manifest_inspection_summary",
    "_would_write_paths",
    "_training_would_write_paths",
    "_evaluation_split_name",
    "_split_summaries",
    "_grouped_evaluation",
    "_run_reward_ablations",
    "_run_training",
    "_should_evaluate_trained_policy",
    "_comparison_from_evaluation",
    "_training_episodes_for_source",
    "_training_source_comparison",
    "_training_distillation_matrix",
    "_distillation_evaluation_scope",
    "_distillation_run_selection_decision",
    "_teacher_agreement_summary",
    "_distillation_stability_summary",
    "_distillation_stability_group_summary",
    "_collect_confidence_values",
    "_nested_metric_mean",
    "_distillation_profile_dimension_enabled",
    "_numeric_stats_by_metric",
    "_optional_numeric",
    "_split_training_episodes",
    "_split_training_scenarios",
    "_training_seeds",
    "_training_source_selection_strategies",
    "_training_teacher_imitation_weights",
    "_training_teacher_margin_curriculum_profiles",
    "_coerce_teacher_margin_curriculum_profile",
    "_builtin_teacher_margin_curriculum_profile",
    "_training_architectures",
    "_training_architecture_config",
    "_normalize_training_architecture_name",
    "_training_output_path",
    "_normalize_training_source_name",
    "_normalize_teacher_weight_name",
    "_normalize_curriculum_profile_name",
    "_path_parent_contains_placeholder",
    "_path_parent_contains_any_placeholder",
    "_calibration_recommendation",
    "_calibration_profile_records",
    "_recommended_run",
    "_calibration_profile_key",
    "_run_curriculum_profile_name",
    "_select_best_training_run",
    "_best_selection_record",
    "_excluded_run_record",
    "_teacher_quality_gate_status",
    "_teacher_quality_gate_failed",
    "_training_run_metric",
    "_multi_seed_evaluation_summary",
    "_BASELINE_DELTA_METRICS",
    "_baseline_deltas",
    "_baseline_names",
    "_multi_seed_delta_summary",
    "_metric_value",
    "_numeric_stats",
    "_environment_metadata",
    "_git_metadata",
    "_git_output",
    "_resolved_manifest_payload",
    "_aggregate_rollout_metrics",
    "_daily_report_summary",
    "_architecture_deltas",
    "_policy_ranking",
    "_ordered_policy_names",
    "_per_group_winners",
    "_failure_scenarios",
    "_gate_summary",
    "_markdown_report",
    "_loss_summary",
)
