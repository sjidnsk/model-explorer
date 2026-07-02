from __future__ import annotations

from pathlib import Path
from typing import Any

from ..io.scenario import Scenario
from ..policy.dataset import summarize_teacher_quality_gates
from ..policy.evaluation import evaluate_policy_baseline_scenarios
from ..policy.rollout import RolloutEpisode
from ..policy.system_calibration import (
    annotate_runs_with_path_feedback_gates,
    build_sample_quality_summary,
    build_system_calibration_summary,
    filter_episodes_by_sample_quality,
    load_path_feedback_summary_entries,
    select_system_best_run,
)
from .evaluation import (
    _collect_episodes,
    _comparison_from_evaluation,
    _grouped_evaluation,
)
from .manifest import (
    ExperimentScenarioGroup,
    _ensure_parent_dir,
    _resolve_path,
    _system_calibration_config,
    _system_path_feedback_gate_config,
    _system_path_feedback_gate_enabled,
    _system_sample_quality_config,
    _write_json,
)
from .reports import _loss_summary
from .selection import (
    _baseline_deltas,
    _best_selection_record,
    _calibration_recommendation,
    _distillation_stability_summary,
    _multi_seed_delta_summary,
    _multi_seed_evaluation_summary,
    _normalize_curriculum_profile_name,
    _normalize_teacher_weight_name,
    _normalize_training_source_name,
    _recommended_run,
    _select_best_training_run,
    _training_distillation_matrix,
    _training_source_comparison,
)
from .training_dimensions import (
    _builtin_teacher_margin_curriculum_profile,
    _coerce_teacher_margin_curriculum_profile,
    _normalize_training_architecture_name,
    _training_architecture_config,
    _training_architectures,
    _training_seeds,
    _training_source_selection_strategies,
    _training_teacher_imitation_weights,
    _training_teacher_margin_curriculum_profiles,
)
from .training_execution import _execute_training_runs, _training_episodes_for_source
from .training_outputs import (
    _path_parent_contains_any_placeholder,
    _path_parent_contains_placeholder,
    _write_system_calibration_summary_output,
    _training_output_path,
)


def _run_training(
    episodes: tuple[RolloutEpisode, ...],
    config: dict[str, Any],
    *,
    base_dir: Path,
    run_output_dir: Path | None = None,
    scenarios: tuple[Scenario, ...] = (),
    planner=None,
    max_candidates: int | None = None,
    reward_config: dict[str, Any] | None = None,
    train_episodes: tuple[RolloutEpisode, ...] | None = None,
    train_scenarios: tuple[Scenario, ...] | None = None,
    validation_episodes: tuple[RolloutEpisode, ...] | None = None,
    validation_scenarios: tuple[Scenario, ...] | None = None,
    validation_groups: tuple[ExperimentScenarioGroup, ...] | None = None,
    validation_paths: tuple[Path, ...] | None = None,
    test_scenarios: tuple[Scenario, ...] | None = None,
    test_groups: tuple[ExperimentScenarioGroup, ...] | None = None,
    test_paths: tuple[Path, ...] | None = None,
) -> dict[str, Any]:
    validation_fraction = float(config.get("validation_fraction", 0.0))
    if train_episodes is None:
        train_episodes, inferred_validation_episodes = _split_training_episodes(
            episodes,
            validation_fraction=validation_fraction,
        )
        if validation_episodes is None:
            validation_episodes = inferred_validation_episodes
    if validation_scenarios is None:
        _, inferred_validation_scenarios = _split_training_scenarios(
            scenarios,
            validation_fraction=validation_fraction,
        )
        validation_scenarios = inferred_validation_scenarios
    validation_episodes = validation_episodes or ()
    evaluation_scenarios = validation_scenarios or scenarios
    seeds = _training_seeds(config)
    architectures = _training_architectures(config)
    source_strategies = _training_source_selection_strategies(config)
    teacher_weights = _training_teacher_imitation_weights(config)
    curriculum_profiles = _training_teacher_margin_curriculum_profiles(config)
    calibration_inputs = _training_calibration_inputs(config, base_dir=base_dir)
    runs = _execute_training_runs(
        train_episodes,
        config,
        base_dir=base_dir,
        run_output_dir=run_output_dir,
        train_scenarios=train_scenarios or scenarios,
        fallback_scenarios=scenarios,
        planner=planner,
        max_candidates=max_candidates,
        reward_config=reward_config,
        validation_episodes=validation_episodes,
        evaluation_scenarios=evaluation_scenarios,
        validation_groups=validation_groups,
        validation_paths=validation_paths,
        test_scenarios=test_scenarios,
        test_groups=test_groups,
        test_paths=test_paths,
        seeds=seeds,
        architectures=architectures,
        source_strategies=source_strategies,
        teacher_weights=teacher_weights,
        curriculum_profiles=curriculum_profiles,
        sample_quality_summary=calibration_inputs["sample_quality_summary"],
        sample_quality_config=calibration_inputs["sample_quality_config"],
    )
    best_policy = str(config.get("best_policy", "torch_policy"))
    best_metric = str(config.get("best_metric", "final_coverage_rate"))
    best_run, calibration_recommendation = _select_training_result(
        runs,
        config,
        calibration_inputs=calibration_inputs,
        best_policy=best_policy,
        best_metric=best_metric,
    )
    selected = _training_final_summary(
        runs,
        best_run=best_run,
        calibration_recommendation=calibration_recommendation,
        seeds=seeds,
        architectures=architectures,
        source_strategies=source_strategies,
        teacher_weights=teacher_weights,
        curriculum_profiles=curriculum_profiles,
        best_policy=best_policy,
        best_metric=best_metric,
    )
    _attach_system_calibration_summary(
        selected,
        runs,
        calibration_recommendation=calibration_recommendation,
        calibration_inputs=calibration_inputs,
        base_dir=base_dir,
        best_policy=best_policy,
        best_metric=best_metric,
    )
    return selected


def _training_calibration_inputs(config: dict[str, Any], *, base_dir: Path) -> dict[str, Any]:
    system_calibration_config = _system_calibration_config(config)
    path_feedback_summary_entries: list[dict[str, Any]] = []
    sample_quality_config = None
    sample_quality_summary = None
    system_path_gate_enabled = (
        False
        if system_calibration_config is None
        else _system_path_feedback_gate_enabled(system_calibration_config)
    )
    if system_calibration_config is not None:
        path_feedback_summary_entries = load_path_feedback_summary_entries(
            system_calibration_config,
            base_dir=base_dir,
        )
        sample_quality_config = _system_sample_quality_config(system_calibration_config)
        if sample_quality_config is not None:
            if not path_feedback_summary_entries:
                raise ValueError("system_calibration.sample_quality requires path feedback summaries")
            sample_quality_summary = build_sample_quality_summary(
                path_feedback_summary_entries,
                sample_quality_config,
            )
    return {
        "system_calibration_config": system_calibration_config,
        "system_path_gate_enabled": system_path_gate_enabled,
        "path_feedback_summary_entries": path_feedback_summary_entries,
        "sample_quality_config": sample_quality_config,
        "sample_quality_summary": sample_quality_summary,
    }


def _select_training_result(
    runs: list[dict[str, Any]],
    config: dict[str, Any],
    *,
    calibration_inputs: dict[str, Any],
    best_policy: str,
    best_metric: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    system_calibration_config = calibration_inputs["system_calibration_config"]
    if system_calibration_config is not None and calibration_inputs["system_path_gate_enabled"]:
        path_feedback_summary_entries = calibration_inputs["path_feedback_summary_entries"]
        if not path_feedback_summary_entries:
            raise ValueError("system_calibration.path_feedback_gate requires path feedback summaries")
        annotate_runs_with_path_feedback_gates(
            runs,
            path_feedback_summaries=path_feedback_summary_entries,
            gate_config=_system_path_feedback_gate_config(system_calibration_config),
        )
        system_selection = select_system_best_run(
            runs,
            policy=best_policy,
            metric=best_metric,
            system_gate_configured=True,
        )
        if system_selection.get("run") is None:
            raise ValueError("system calibration found no run passing teacher and path-feedback gates")
        best_run = system_selection["run"]
        calibration_recommendation = _calibration_recommendation(
            runs,
            metric=best_metric,
            policy=best_policy,
            selected_run=best_run,
        )
        calibration_recommendation["mode"] = "system_gate_best_run"
        calibration_recommendation["profile_selection_reason_codes"] = ["system_gate_best_run_selection"]
        return best_run, calibration_recommendation
    if "teacher_margin_curriculum_profiles" in config:
        calibration_recommendation = _calibration_recommendation(
            runs,
            metric=best_metric,
            policy=best_policy,
        )
        return _recommended_run(runs, calibration_recommendation), calibration_recommendation
    best_run = _select_best_training_run(runs, metric=best_metric, policy=best_policy)
    calibration_recommendation = _calibration_recommendation(
        runs,
        metric=best_metric,
        policy=best_policy,
        selected_run=best_run,
    )
    return best_run, calibration_recommendation


def _training_final_summary(
    runs: list[dict[str, Any]],
    *,
    best_run: dict[str, Any],
    calibration_recommendation: dict[str, Any],
    seeds: tuple[int, ...],
    architectures: tuple[str | None, ...],
    source_strategies: tuple[str | None, ...],
    teacher_weights: tuple[float, ...],
    curriculum_profiles: tuple[dict[str, Any], ...],
    best_policy: str,
    best_metric: str,
) -> dict[str, Any]:
    selected = dict(best_run)
    selected["checkpoint"] = best_run["checkpoint"]
    selected["best_seed"] = int(best_run["seed"])
    selected["best_checkpoint"] = best_run["checkpoint"]
    selected["best_checkpoint_path"] = best_run["checkpoint"]
    selected["last_checkpoint"] = runs[-1]["checkpoint"]
    selected["last_checkpoint_path"] = runs[-1]["checkpoint"]
    selected["seeds"] = list(seeds)
    selected["architectures"] = [_normalize_training_architecture_name(architecture) for architecture in architectures]
    selected["architecture_count"] = len(architectures)
    selected["run_count"] = len(runs)
    selected["runs"] = runs
    selected["source_selection_strategies"] = [
        _normalize_training_source_name(strategy) for strategy in source_strategies if strategy is not None
    ]
    selected["teacher_imitation_weights"] = [float(weight) for weight in teacher_weights]
    selected["teacher_margin_curriculum_profiles"] = [str(profile["name"]) for profile in curriculum_profiles]
    selected["source_comparison"] = _training_source_comparison(runs)
    selected["distillation_matrix"] = _training_distillation_matrix(
        runs,
        selected_run=best_run,
        policy=best_policy,
        metric=best_metric,
    )
    selected["source_weight_comparison"] = list(selected["distillation_matrix"])
    selected["multi_seed_summary"] = _multi_seed_evaluation_summary(runs)
    selected["multi_seed_loss_summary"] = _loss_summary(runs)
    selected["multi_seed_delta_summary"] = _multi_seed_delta_summary(runs)
    selected["distillation_stability_summary"] = _distillation_stability_summary(runs)
    selected["calibration_recommendation"] = calibration_recommendation
    selected["best_selection"] = _best_selection_record(
        runs,
        best_run,
        policy=best_policy,
        metric=best_metric,
    )
    return selected


def _attach_system_calibration_summary(
    selected: dict[str, Any],
    runs: list[dict[str, Any]],
    *,
    calibration_recommendation: dict[str, Any],
    calibration_inputs: dict[str, Any],
    base_dir: Path,
    best_policy: str,
    best_metric: str,
) -> None:
    system_calibration_config = calibration_inputs["system_calibration_config"]
    if system_calibration_config is None:
        return
    system_summary = build_system_calibration_summary(
        {
            "runs": runs,
            "calibration_recommendation": calibration_recommendation,
        },
        path_feedback_summaries=calibration_inputs["path_feedback_summary_entries"],
        config=system_calibration_config,
        policy=best_policy,
        metric=best_metric,
    )
    selected["system_calibration_summary"] = system_summary
    system_selection_record = system_summary["selection"]
    selected["best_selection"]["mode"] = system_selection_record["mode"]
    selected["best_selection"]["reason_codes"] = list(system_selection_record["reason_codes"])
    selected["best_selection"]["excluded_run_count"] = system_selection_record["excluded_run_count"]
    selected["best_selection"]["excluded_runs"] = list(system_selection_record["excluded_runs"])
    selected["best_selection"]["system_gate_selection"] = dict(system_selection_record)
    summary_output_path = _write_system_calibration_summary_output(
        base_dir,
        system_calibration_config,
        system_summary,
    )
    if summary_output_path is not None:
        selected["system_calibration_summary_output"] = str(summary_output_path)


def _split_training_episodes(
    episodes: tuple[RolloutEpisode, ...],
    *,
    validation_fraction: float,
) -> tuple[tuple[RolloutEpisode, ...], tuple[RolloutEpisode, ...]]:
    if validation_fraction <= 0.0 or len(episodes) <= 1:
        return episodes, ()
    validation_count = max(1, int(round(len(episodes) * min(validation_fraction, 0.9))))
    train_count = max(1, len(episodes) - validation_count)
    return episodes[:train_count], episodes[train_count:]


def _split_training_scenarios(
    scenarios: tuple[Scenario, ...],
    *,
    validation_fraction: float,
) -> tuple[tuple[Scenario, ...], tuple[Scenario, ...]]:
    if validation_fraction <= 0.0 or len(scenarios) <= 1:
        return scenarios, ()
    validation_count = max(1, int(round(len(scenarios) * min(validation_fraction, 0.9))))
    train_count = max(1, len(scenarios) - validation_count)
    return scenarios[:train_count], scenarios[train_count:]


# Public aliases
run_training = _run_training
training_architectures = _training_architectures
training_distillation_matrix = _training_distillation_matrix
training_output_path = _training_output_path
training_seeds = _training_seeds
training_source_selection_strategies = _training_source_selection_strategies
training_teacher_imitation_weights = _training_teacher_imitation_weights
training_teacher_margin_curriculum_profiles = _training_teacher_margin_curriculum_profiles

__all__ = (
    "Scenario",
    "summarize_teacher_quality_gates",
    "evaluate_policy_baseline_scenarios",
    "RolloutEpisode",
    "annotate_runs_with_path_feedback_gates",
    "build_sample_quality_summary",
    "build_system_calibration_summary",
    "filter_episodes_by_sample_quality",
    "load_path_feedback_summary_entries",
    "select_system_best_run",
    "_collect_episodes",
    "_comparison_from_evaluation",
    "_grouped_evaluation",
    "ExperimentScenarioGroup",
    "_ensure_parent_dir",
    "_resolve_path",
    "_system_calibration_config",
    "_system_path_feedback_gate_config",
    "_system_path_feedback_gate_enabled",
    "_system_sample_quality_config",
    "_write_json",
    "_loss_summary",
    "_baseline_deltas",
    "_best_selection_record",
    "_calibration_recommendation",
    "_distillation_stability_summary",
    "_multi_seed_delta_summary",
    "_multi_seed_evaluation_summary",
    "_normalize_curriculum_profile_name",
    "_normalize_teacher_weight_name",
    "_normalize_training_source_name",
    "_recommended_run",
    "_select_best_training_run",
    "_training_distillation_matrix",
    "_training_source_comparison",
    "_run_training",
    "_training_episodes_for_source",
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
    "_path_parent_contains_placeholder",
    "_path_parent_contains_any_placeholder",
    "run_training",
    "training_architectures",
    "training_distillation_matrix",
    "training_output_path",
    "training_seeds",
    "training_source_selection_strategies",
    "training_teacher_imitation_weights",
    "training_teacher_margin_curriculum_profiles",
)
