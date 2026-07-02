from __future__ import annotations

from pathlib import Path
from typing import Any

from ..io.scenario import Scenario
from ..policy.dataset import summarize_teacher_quality_gates
from ..policy.evaluation import evaluate_policy_baseline_scenarios
from ..policy.rollout import RolloutEpisode
from ..policy.system_calibration import filter_episodes_by_sample_quality
from .evaluation import _collect_episodes, _comparison_from_evaluation, _grouped_evaluation
from .manifest import ExperimentScenarioGroup, _ensure_parent_dir
from .selection import _baseline_deltas
from .training_dimensions import _normalize_training_architecture_name, _training_architecture_config
from .training_outputs import (
    _training_output_path,
    _write_training_evaluation_output,
    _write_training_loss_log,
    _write_training_summary,
)


def _execute_training_runs(
    episodes: tuple[RolloutEpisode, ...],
    config: dict[str, Any],
    *,
    base_dir: Path,
    run_output_dir: Path | None,
    train_scenarios: tuple[Scenario, ...],
    fallback_scenarios: tuple[Scenario, ...],
    planner,
    max_candidates: int | None,
    reward_config: dict[str, Any] | None,
    validation_episodes: tuple[RolloutEpisode, ...],
    evaluation_scenarios: tuple[Scenario, ...],
    validation_groups: tuple[ExperimentScenarioGroup, ...] | None,
    validation_paths: tuple[Path, ...] | None,
    test_scenarios: tuple[Scenario, ...] | None,
    test_groups: tuple[ExperimentScenarioGroup, ...] | None,
    test_paths: tuple[Path, ...] | None,
    seeds: tuple[int, ...],
    architectures: tuple[str | None, ...],
    source_strategies: tuple[str | None, ...],
    teacher_weights: tuple[float, ...],
    curriculum_profiles: tuple[dict[str, Any], ...],
    sample_quality_summary: dict[str, Any] | None,
    sample_quality_config: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    matrix_flags = {
        "multi_seed": len(seeds) > 1,
        "multi_architecture": len(architectures) > 1,
        "multi_source": len(source_strategies) > 1,
        "multi_teacher_weight": len(teacher_weights) > 1,
        "multi_curriculum_profile": len(curriculum_profiles) > 1,
    }
    for source_strategy in source_strategies:
        source_episodes = _training_episodes_for_source(
            source_strategy,
            train_episodes=episodes,
            train_scenarios=train_scenarios or fallback_scenarios,
            planner=planner,
            max_candidates=max_candidates,
            reward_config=reward_config,
        )
        source_sample_quality_summary = None
        if sample_quality_summary is not None:
            source_episodes, source_sample_quality_summary = filter_episodes_by_sample_quality(
                source_episodes,
                sample_quality_summary,
                sample_quality_config,
            )
        runs.extend(
            _execute_source_training_runs(
                source_episodes,
                config,
                base_dir=base_dir,
                run_output_dir=run_output_dir,
                source_strategy=source_strategy,
                source_sample_quality_summary=source_sample_quality_summary,
                validation_episodes=validation_episodes,
                evaluation_scenarios=evaluation_scenarios,
                validation_groups=validation_groups,
                validation_paths=validation_paths,
                test_scenarios=test_scenarios,
                test_groups=test_groups,
                test_paths=test_paths,
                planner=planner,
                seeds=seeds,
                architectures=architectures,
                teacher_weights=teacher_weights,
                curriculum_profiles=curriculum_profiles,
                matrix_flags=matrix_flags,
            )
        )
    return runs


def _execute_source_training_runs(
    source_episodes: tuple[RolloutEpisode, ...],
    config: dict[str, Any],
    *,
    base_dir: Path,
    run_output_dir: Path | None,
    source_strategy: str | None,
    source_sample_quality_summary: dict[str, Any] | None,
    validation_episodes: tuple[RolloutEpisode, ...],
    evaluation_scenarios: tuple[Scenario, ...],
    validation_groups: tuple[ExperimentScenarioGroup, ...] | None,
    validation_paths: tuple[Path, ...] | None,
    test_scenarios: tuple[Scenario, ...] | None,
    test_groups: tuple[ExperimentScenarioGroup, ...] | None,
    test_paths: tuple[Path, ...] | None,
    planner,
    seeds: tuple[int, ...],
    architectures: tuple[str | None, ...],
    teacher_weights: tuple[float, ...],
    curriculum_profiles: tuple[dict[str, Any], ...],
    matrix_flags: dict[str, bool],
) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for teacher_weight in teacher_weights:
        for curriculum_profile in curriculum_profiles:
            for architecture in architectures:
                for seed in seeds:
                    runs.append(
                        _execute_single_training_run(
                            source_episodes,
                            config,
                            base_dir=base_dir,
                            run_output_dir=run_output_dir,
                            source_strategy=source_strategy,
                            source_sample_quality_summary=source_sample_quality_summary,
                            validation_episodes=validation_episodes,
                            evaluation_scenarios=evaluation_scenarios,
                            validation_groups=validation_groups,
                            validation_paths=validation_paths,
                            test_scenarios=test_scenarios,
                            test_groups=test_groups,
                            test_paths=test_paths,
                            planner=planner,
                            seed=seed,
                            architecture=architecture,
                            teacher_weight=teacher_weight,
                            curriculum_profile=curriculum_profile,
                            matrix_flags=matrix_flags,
                        )
                    )
    return runs


def _execute_single_training_run(
    source_episodes: tuple[RolloutEpisode, ...],
    config: dict[str, Any],
    *,
    base_dir: Path,
    run_output_dir: Path | None,
    source_strategy: str | None,
    source_sample_quality_summary: dict[str, Any] | None,
    validation_episodes: tuple[RolloutEpisode, ...],
    evaluation_scenarios: tuple[Scenario, ...],
    validation_groups: tuple[ExperimentScenarioGroup, ...] | None,
    validation_paths: tuple[Path, ...] | None,
    test_scenarios: tuple[Scenario, ...] | None,
    test_groups: tuple[ExperimentScenarioGroup, ...] | None,
    test_paths: tuple[Path, ...] | None,
    planner,
    seed: int,
    architecture: str | None,
    teacher_weight: float,
    curriculum_profile: dict[str, Any],
    matrix_flags: dict[str, bool],
) -> dict[str, Any]:
    from ..policy.training import load_policy_checkpoint, train_policy_on_episodes

    profile_name = str(curriculum_profile["name"])
    architecture_name = _normalize_training_architecture_name(architecture)
    checkpoint = _run_output_path(
        config,
        "checkpoint",
        seed=seed,
        architecture=architecture_name,
        source_strategy=source_strategy,
        teacher_weight=teacher_weight,
        profile_name=profile_name,
        base_dir=base_dir,
        run_output_dir=run_output_dir,
        default_name="checkpoint.pt",
        matrix_flags=matrix_flags,
        required=True,
    )
    loss_log = _run_output_path(
        config,
        "loss_log",
        seed=seed,
        architecture=architecture_name,
        source_strategy=source_strategy,
        teacher_weight=teacher_weight,
        profile_name=profile_name,
        base_dir=base_dir,
        run_output_dir=run_output_dir,
        default_name="losses.jsonl",
        matrix_flags=matrix_flags,
        required=False,
    )
    if checkpoint is None:
        raise ValueError("train.checkpoint is required when outputs.root is not configured")
    _ensure_parent_dir(checkpoint)
    result = train_policy_on_episodes(
        source_episodes,
        checkpoint_path=checkpoint,
        seed=seed,
        hidden_size=int(config.get("hidden_size", 64)),
        learning_rate=float(config.get("learning_rate", 1.0e-3)),
        epochs=int(config.get("epochs", 1)),
        return_mode=str(config.get("return_mode", "reward_as_return")),
        discount_factor=float(config.get("discount_factor", 0.99)),
        architecture=architecture,
        architecture_config=_training_architecture_config(config, architecture_name),
        teacher_imitation_weight=teacher_weight,
        teacher_margin_weighting=curriculum_profile.get("teacher_margin_weighting"),
    )
    _write_training_loss_log(loss_log, result)
    _annotate_training_result(
        result,
        checkpoint=checkpoint,
        loss_log=loss_log,
        source_strategy=source_strategy,
        teacher_weight=teacher_weight,
        curriculum_profile=profile_name,
        teacher_margin_weighting=curriculum_profile.get("teacher_margin_weighting"),
        profile_matrix_configured="teacher_margin_curriculum_profiles" in config,
        source_sample_quality_summary=source_sample_quality_summary,
        validation_episode_count=len(validation_episodes),
        train_episode_count=len(source_episodes),
        teacher_quality_gates=config.get("teacher_quality_gates"),
    )
    if evaluation_scenarios:
        _attach_validation_evaluation(
            result,
            checkpoint=checkpoint,
            scenarios=evaluation_scenarios,
            groups=validation_groups,
            paths=validation_paths,
            planner=planner,
            trained_policy=load_policy_checkpoint(checkpoint),
        )
    if test_scenarios:
        _attach_test_evaluation(
            result,
            checkpoint=checkpoint,
            scenarios=test_scenarios,
            groups=test_groups,
            paths=test_paths,
            planner=planner,
            trained_policy=load_policy_checkpoint(checkpoint),
        )
    summary_output = _write_training_summary(checkpoint, result)
    result["training_summary_output"] = str(summary_output)
    return result


def _run_output_path(
    config: dict[str, Any],
    key: str,
    *,
    seed: int,
    architecture: str,
    source_strategy: str | None,
    teacher_weight: float,
    profile_name: str,
    base_dir: Path,
    run_output_dir: Path | None,
    default_name: str,
    matrix_flags: dict[str, bool],
    required: bool,
) -> Path | None:
    return _training_output_path(
        config,
        key,
        seed=seed,
        architecture=architecture,
        selection_strategy=source_strategy,
        teacher_imitation_weight=teacher_weight,
        curriculum_profile=profile_name,
        base_dir=base_dir,
        run_output_dir=run_output_dir,
        default_name=default_name,
        required=required,
        **matrix_flags,
    )


def _annotate_training_result(
    result: dict[str, Any],
    *,
    checkpoint: Path,
    loss_log: Path | None,
    source_strategy: str | None,
    teacher_weight: float,
    curriculum_profile: str,
    teacher_margin_weighting: Any,
    profile_matrix_configured: bool,
    source_sample_quality_summary: dict[str, Any] | None,
    validation_episode_count: int,
    train_episode_count: int,
    teacher_quality_gates: Any,
) -> None:
    result["checkpoint"] = str(checkpoint)
    if loss_log is not None:
        result["loss_log"] = str(loss_log)
    result["training_data_selection_strategy"] = None if source_strategy is None else str(source_strategy)
    result["teacher_imitation_weight"] = float(teacher_weight)
    if profile_matrix_configured:
        result["teacher_margin_curriculum_profile"] = curriculum_profile
    result["teacher_margin_weighting"] = dict(teacher_margin_weighting or {})
    if source_sample_quality_summary is not None:
        result["sample_quality_summary"] = dict(source_sample_quality_summary)
        audit_summary = source_sample_quality_summary.get("sample_quality_audit_summary")
        if isinstance(audit_summary, dict):
            result["sample_quality_audit_summary"] = dict(audit_summary)
    result["teacher_quality_gates"] = summarize_teacher_quality_gates(
        result.get("dataset_summary", {}),
        teacher_quality_gates,
    )
    result["train_episode_count"] = train_episode_count
    result["validation_episode_count"] = validation_episode_count


def _attach_validation_evaluation(
    result: dict[str, Any],
    *,
    checkpoint: Path,
    scenarios: tuple[Scenario, ...],
    groups: tuple[ExperimentScenarioGroup, ...] | None,
    paths: tuple[Path, ...] | None,
    planner,
    trained_policy,
) -> None:
    evaluation = _training_run_evaluation(
        scenarios,
        groups=groups,
        paths=paths,
        planner=planner,
        trained_policy=trained_policy,
    )
    result["validation_evaluation"] = evaluation
    result["baseline_deltas"] = _baseline_deltas(_comparison_from_evaluation(evaluation)).get("torch_policy", {})
    output = _write_training_evaluation_output(checkpoint, "validation-evaluation.json", evaluation)
    result["validation_evaluation_output"] = str(output)


def _attach_test_evaluation(
    result: dict[str, Any],
    *,
    checkpoint: Path,
    scenarios: tuple[Scenario, ...],
    groups: tuple[ExperimentScenarioGroup, ...] | None,
    paths: tuple[Path, ...] | None,
    planner,
    trained_policy,
) -> None:
    evaluation = _training_run_evaluation(
        scenarios,
        groups=groups,
        paths=paths,
        planner=planner,
        trained_policy=trained_policy,
    )
    result["test_evaluation"] = evaluation
    output = _write_training_evaluation_output(checkpoint, "test-evaluation.json", evaluation)
    result["test_evaluation_output"] = str(output)


def _training_run_evaluation(
    scenarios: tuple[Scenario, ...],
    *,
    groups: tuple[ExperimentScenarioGroup, ...] | None,
    paths: tuple[Path, ...] | None,
    planner,
    trained_policy,
) -> dict[str, Any]:
    aggregate = evaluate_policy_baseline_scenarios(
        scenarios,
        torch_policy=trained_policy,
        planning_adapter=planner,
    )
    if groups and paths:
        return _grouped_evaluation(
            groups,
            scenarios,
            paths,
            planner=planner,
            aggregate=aggregate,
            torch_policy=trained_policy,
        )
    return aggregate


def _training_episodes_for_source(
    selection_strategy: str | None,
    *,
    train_episodes: tuple[RolloutEpisode, ...],
    train_scenarios: tuple[Scenario, ...],
    planner,
    max_candidates: int | None,
    reward_config: dict[str, Any] | None,
) -> tuple[RolloutEpisode, ...]:
    if selection_strategy is None:
        return train_episodes
    return _collect_episodes(
        train_scenarios,
        planner=planner,
        max_candidates=max_candidates,
        selection_strategy=selection_strategy,
        reward_config=reward_config,
    )


__all__ = (
    "_execute_training_runs",
    "_execute_source_training_runs",
    "_execute_single_training_run",
    "_run_output_path",
    "_annotate_training_result",
    "_attach_validation_evaluation",
    "_attach_test_evaluation",
    "_training_run_evaluation",
    "_training_episodes_for_source",
)
