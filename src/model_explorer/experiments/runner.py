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


# Compatibility re-exports for legacy private imports.
from .environment import _git_metadata, _git_output, environment_metadata, git_metadata
from .evaluation import *
from .manifest import *
from .reports import *
from .selection import *
from .training_matrix import *

__all__ = [name for name in globals() if not name.startswith("__")]
