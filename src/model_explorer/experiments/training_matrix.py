from __future__ import annotations

import json
from importlib import util as importlib_util
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
    from ..policy.training import train_policy_on_episodes

    from ..policy.training import load_policy_checkpoint

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
    multi_seed = len(seeds) > 1
    multi_architecture = len(architectures) > 1
    multi_source = len(source_strategies) > 1
    multi_teacher_weight = len(teacher_weights) > 1
    multi_curriculum_profile = len(curriculum_profiles) > 1
    profile_matrix_configured = "teacher_margin_curriculum_profiles" in config
    system_calibration_config = _system_calibration_config(config)
    system_path_gate_enabled = (
        False
        if system_calibration_config is None
        else _system_path_feedback_gate_enabled(system_calibration_config)
    )
    path_feedback_summary_entries: list[dict[str, Any]] = []
    sample_quality_config = None
    sample_quality_summary = None
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
    runs: list[dict[str, Any]] = []

    for source_strategy in source_strategies:
        source_episodes = _training_episodes_for_source(
            source_strategy,
            train_episodes=train_episodes,
            train_scenarios=train_scenarios or scenarios,
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
        for teacher_weight in teacher_weights:
            for curriculum_profile in curriculum_profiles:
                profile_name = str(curriculum_profile["name"])
                teacher_margin_weighting = curriculum_profile.get("teacher_margin_weighting")
                for architecture in architectures:
                    architecture_name = _normalize_training_architecture_name(architecture)
                    for seed in seeds:
                        checkpoint = _training_output_path(
                            config,
                            "checkpoint",
                            seed=seed,
                            architecture=architecture_name,
                            selection_strategy=source_strategy,
                            teacher_imitation_weight=teacher_weight,
                            curriculum_profile=profile_name,
                            base_dir=base_dir,
                            run_output_dir=run_output_dir,
                            default_name="checkpoint.pt",
                            multi_seed=multi_seed,
                            multi_architecture=multi_architecture,
                            multi_source=multi_source,
                            multi_teacher_weight=multi_teacher_weight,
                            multi_curriculum_profile=multi_curriculum_profile,
                            required=True,
                        )
                        loss_log = _training_output_path(
                            config,
                            "loss_log",
                            seed=seed,
                            architecture=architecture_name,
                            selection_strategy=source_strategy,
                            teacher_imitation_weight=teacher_weight,
                            curriculum_profile=profile_name,
                            base_dir=base_dir,
                            run_output_dir=run_output_dir,
                            default_name="losses.jsonl",
                            multi_seed=multi_seed,
                            multi_architecture=multi_architecture,
                            multi_source=multi_source,
                            multi_teacher_weight=multi_teacher_weight,
                            multi_curriculum_profile=multi_curriculum_profile,
                            required=False,
                        )
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
                            teacher_margin_weighting=teacher_margin_weighting,
                        )
                        if loss_log is not None:
                            _ensure_parent_dir(loss_log)
                            loss_records = result.get("epoch_losses", [])
                            if not isinstance(loss_records, list) or not loss_records:
                                loss_records = [result]
                            loss_log.write_text(
                                "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in loss_records),
                                encoding="utf-8",
                            )
                        result["checkpoint"] = str(checkpoint)
                        if loss_log is not None:
                            result["loss_log"] = str(loss_log)
                        result["training_data_selection_strategy"] = (
                            None if source_strategy is None else str(source_strategy)
                        )
                        result["teacher_imitation_weight"] = float(teacher_weight)
                        if profile_matrix_configured:
                            result["teacher_margin_curriculum_profile"] = profile_name
                        result["teacher_margin_weighting"] = dict(teacher_margin_weighting or {})
                        if source_sample_quality_summary is not None:
                            result["sample_quality_summary"] = dict(source_sample_quality_summary)
                            audit_summary = source_sample_quality_summary.get("sample_quality_audit_summary")
                            if isinstance(audit_summary, dict):
                                result["sample_quality_audit_summary"] = dict(audit_summary)
                        result["teacher_quality_gates"] = summarize_teacher_quality_gates(
                            result.get("dataset_summary", {}),
                            config.get("teacher_quality_gates"),
                        )
                        result["train_episode_count"] = len(source_episodes)
                        result["validation_episode_count"] = len(validation_episodes)
                        if evaluation_scenarios:
                            trained_policy = load_policy_checkpoint(checkpoint)
                            aggregate_validation_evaluation = evaluate_policy_baseline_scenarios(
                                evaluation_scenarios,
                                torch_policy=trained_policy,
                                planning_adapter=planner,
                            )
                            validation_evaluation = (
                                _grouped_evaluation(
                                    validation_groups,
                                    evaluation_scenarios,
                                    validation_paths or (),
                                    planner=planner,
                                    aggregate=aggregate_validation_evaluation,
                                    torch_policy=trained_policy,
                                )
                                if validation_groups and validation_paths
                                else aggregate_validation_evaluation
                            )
                            result["validation_evaluation"] = validation_evaluation
                            result["baseline_deltas"] = _baseline_deltas(
                                _comparison_from_evaluation(validation_evaluation)
                            ).get("torch_policy", {})
                            validation_output = checkpoint.parent / "validation-evaluation.json"
                            _write_json(validation_output, validation_evaluation)
                            result["validation_evaluation_output"] = str(validation_output)
                        if test_scenarios:
                            trained_policy = load_policy_checkpoint(checkpoint)
                            aggregate_test_evaluation = evaluate_policy_baseline_scenarios(
                                test_scenarios,
                                torch_policy=trained_policy,
                                planning_adapter=planner,
                            )
                            test_evaluation = (
                                _grouped_evaluation(
                                    test_groups,
                                    test_scenarios,
                                    test_paths or (),
                                    planner=planner,
                                    aggregate=aggregate_test_evaluation,
                                    torch_policy=trained_policy,
                                )
                                if test_groups and test_paths
                                else aggregate_test_evaluation
                            )
                            result["test_evaluation"] = test_evaluation
                            test_output = checkpoint.parent / "test-evaluation.json"
                            _write_json(test_output, test_evaluation)
                            result["test_evaluation_output"] = str(test_output)
                        training_summary_output = checkpoint.parent / "training-summary.json"
                        _write_json(training_summary_output, result)
                        result["training_summary_output"] = str(training_summary_output)
                        runs.append(result)

    best_policy = str(config.get("best_policy", "torch_policy"))
    best_metric = str(config.get("best_metric", "final_coverage_rate"))
    system_selection: dict[str, Any] | None = None
    if system_calibration_config is not None and system_path_gate_enabled:
        if path_feedback_summary_entries:
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
            calibration_recommendation["profile_selection_reason_codes"] = [
                "system_gate_best_run_selection"
            ]
        else:
            raise ValueError("system_calibration.path_feedback_gate requires path feedback summaries")
    if system_selection is None:
        if "teacher_margin_curriculum_profiles" in config:
            calibration_recommendation = _calibration_recommendation(
                runs,
                metric=best_metric,
                policy=best_policy,
            )
            best_run = _recommended_run(runs, calibration_recommendation)
        else:
            best_run = _select_best_training_run(runs, metric=best_metric, policy=best_policy)
            calibration_recommendation = _calibration_recommendation(
                runs,
                metric=best_metric,
                policy=best_policy,
                selected_run=best_run,
            )
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
    if system_calibration_config is not None:
        system_summary = build_system_calibration_summary(
            {
                "runs": runs,
                "calibration_recommendation": calibration_recommendation,
            },
            path_feedback_summaries=path_feedback_summary_entries,
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
        summary_output = system_calibration_config.get("summary_output")
        if summary_output is not None:
            summary_output_path = _resolve_path(base_dir, summary_output)
            _write_json(summary_output_path, system_summary)
            selected["system_calibration_summary_output"] = str(summary_output_path)
    return selected


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


def _training_seeds(config: dict[str, Any]) -> tuple[int, ...]:
    if "seeds" not in config:
        return (int(config.get("seed", 0)),)
    raw_seeds = config["seeds"]
    if not isinstance(raw_seeds, list) or not raw_seeds:
        raise ValueError("train.seeds must be a non-empty list")
    return tuple(int(seed) for seed in raw_seeds)


def _training_source_selection_strategies(config: dict[str, Any]) -> tuple[str | None, ...]:
    raw_value = config.get("source_selection_strategies")
    if raw_value is None:
        raw_value = config.get("selection_strategies")
    if raw_value is None:
        return (None,)
    if not isinstance(raw_value, list) or not raw_value:
        raise ValueError("train.source_selection_strategies must be a non-empty list")
    strategies = tuple(str(value).strip() for value in raw_value)
    if any(not value for value in strategies):
        raise ValueError("train.source_selection_strategies entries must be non-empty")
    return strategies


def _training_teacher_imitation_weights(config: dict[str, Any]) -> tuple[float, ...]:
    raw_value = config.get("teacher_imitation_weights")
    if raw_value is None:
        return (max(0.0, float(config.get("teacher_imitation_weight", 0.0))),)
    if not isinstance(raw_value, list) or not raw_value:
        raise ValueError("train.teacher_imitation_weights must be a non-empty list")
    return tuple(max(0.0, float(value)) for value in raw_value)


def _training_teacher_margin_curriculum_profiles(config: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    raw_value = config.get("teacher_margin_curriculum_profiles")
    if raw_value is None:
        weighting = config.get("teacher_margin_weighting")
        profile_name = "default"
        if isinstance(weighting, dict):
            profile_name = str(weighting.get("profile_name", "custom")).strip() or "custom"
        return ({"name": profile_name, "teacher_margin_weighting": weighting},)
    if isinstance(raw_value, dict):
        raw_entries = [
            {"name": name, **(value if isinstance(value, dict) else {"bucket_weights": value})}
            for name, value in raw_value.items()
        ]
    else:
        if not isinstance(raw_value, list) or not raw_value:
            raise ValueError("train.teacher_margin_curriculum_profiles must be a non-empty list or mapping")
        raw_entries = list(raw_value)
    profiles: list[dict[str, Any]] = []
    for entry in raw_entries:
        profiles.append(_coerce_teacher_margin_curriculum_profile(entry))
    names = [profile["name"] for profile in profiles]
    if len(set(names)) != len(names):
        raise ValueError("train.teacher_margin_curriculum_profiles names must be unique")
    return tuple(profiles)


def _coerce_teacher_margin_curriculum_profile(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        return _builtin_teacher_margin_curriculum_profile(value)
    if not isinstance(value, dict):
        raise ValueError("teacher margin curriculum profile must be a name or object")
    name = str(value.get("name", "")).strip()
    if not name:
        raise ValueError("teacher margin curriculum profile requires a non-empty name")
    if "teacher_margin_weighting" in value:
        weighting = value["teacher_margin_weighting"]
        if weighting is not None and not isinstance(weighting, dict):
            raise ValueError("teacher_margin_curriculum_profile.teacher_margin_weighting must be an object")
        weighting_config = dict(weighting or {})
    else:
        weighting_config = {
            key: value[key]
            for key in ("bucket_weights", "teacher_low_margin_threshold", "teacher_high_margin_threshold")
            if key in value
        }
    weighting_config["profile_name"] = name
    return {"name": name, "teacher_margin_weighting": weighting_config}


def _builtin_teacher_margin_curriculum_profile(name: str) -> dict[str, Any]:
    profile_name = str(name).strip()
    profiles = {
        "high_only": {"high": 1.0, "medium": 0.0, "low": 0.0, "missing": 0.0},
        "high_medium": {"high": 1.0, "medium": 0.5, "low": 0.0, "missing": 0.0},
        "soft_all_valid": {"high": 1.0, "medium": 0.5, "low": 0.1, "missing": 0.0},
    }
    if profile_name not in profiles:
        raise ValueError(f"unknown teacher margin curriculum profile: {profile_name}")
    return {
        "name": profile_name,
        "teacher_margin_weighting": {
            "profile_name": profile_name,
            "bucket_weights": dict(profiles[profile_name]),
        },
    }


def _training_architectures(config: dict[str, Any]) -> tuple[str | None, ...]:
    if "architectures" not in config:
        return (config.get("architecture"),)
    raw_architectures = config["architectures"]
    if not isinstance(raw_architectures, list) or not raw_architectures:
        raise ValueError("train.architectures must be a non-empty list")
    return tuple(str(architecture) for architecture in raw_architectures)


def _training_architecture_config(config: dict[str, Any], architecture: str) -> dict[str, Any] | None:
    base_config = config.get("architecture_config")
    architecture_configs = config.get("architecture_configs")
    selected_config = base_config
    if architecture_configs is not None:
        if not isinstance(architecture_configs, dict):
            raise ValueError("train.architecture_configs must be a mapping of architecture name to config")
        selected_config = architecture_configs.get(architecture, base_config)
    if selected_config is None:
        return None
    if not isinstance(selected_config, dict):
        raise ValueError("train.architecture_config must be a mapping")
    return dict(selected_config)


def _normalize_training_architecture_name(value: str | None) -> str:
    return "mlp_v1" if value is None or str(value).strip() == "" else str(value)


def _training_output_path(
    config: dict[str, Any],
    key: str,
    *,
    seed: int,
    architecture: str,
    selection_strategy: str | None,
    teacher_imitation_weight: float,
    curriculum_profile: str,
    base_dir: Path,
    run_output_dir: Path | None,
    default_name: str,
    multi_seed: bool,
    multi_architecture: bool,
    multi_source: bool,
    multi_teacher_weight: bool,
    multi_curriculum_profile: bool,
    required: bool,
) -> Path | None:
    source_name = _normalize_training_source_name(selection_strategy)
    teacher_weight_name = _normalize_teacher_weight_name(teacher_imitation_weight)
    curriculum_profile_name = _normalize_curriculum_profile_name(curriculum_profile)
    value = config.get(key)
    if value is not None:
        text = str(value)
        formatted = text.format(
            seed=seed,
            architecture=architecture,
            selection_strategy=source_name,
            teacher_imitation_weight=teacher_weight_name,
            teacher_weight=teacher_weight_name,
            teacher_margin_curriculum_profile=curriculum_profile_name,
            curriculum_profile=curriculum_profile_name,
        )
        path = _resolve_path(base_dir, formatted)
        parent = path.parent
        if multi_source and not _path_parent_contains_placeholder(text, "{selection_strategy}"):
            parent = parent / f"source-{source_name}"
        if multi_teacher_weight and not _path_parent_contains_any_placeholder(
            text,
            ("{teacher_imitation_weight}", "{teacher_weight}"),
        ):
            parent = parent / f"teacher-weight-{teacher_weight_name}"
        if multi_curriculum_profile and not _path_parent_contains_any_placeholder(
            text,
            ("{teacher_margin_curriculum_profile}", "{curriculum_profile}"),
        ):
            parent = parent / f"curriculum-profile-{curriculum_profile_name}"
        if multi_architecture and not _path_parent_contains_placeholder(text, "{architecture}"):
            parent = parent / architecture
        if (multi_seed or multi_architecture) and not _path_parent_contains_placeholder(text, "{seed}"):
            parent = parent / f"seed-{seed}"
        path = parent / path.name
        return path
    if run_output_dir is not None:
        parent = run_output_dir
        if multi_source:
            parent = parent / f"source-{source_name}"
        if multi_teacher_weight:
            parent = parent / f"teacher-weight-{teacher_weight_name}"
        if multi_curriculum_profile:
            parent = parent / f"curriculum-profile-{curriculum_profile_name}"
        if multi_architecture:
            parent = parent / architecture
        return parent / f"seed-{seed}" / default_name
    if required:
        raise ValueError(f"train.{key} is required when outputs.root is not configured")
    return None


def _path_parent_contains_placeholder(path_text: str, placeholder: str) -> bool:
    return any(placeholder in part for part in Path(path_text).parent.parts)


def _path_parent_contains_any_placeholder(path_text: str, placeholders: tuple[str, ...]) -> bool:
    return any(_path_parent_contains_placeholder(path_text, placeholder) for placeholder in placeholders)


# Public aliases
run_training = _run_training
training_architectures = _training_architectures
training_distillation_matrix = _training_distillation_matrix
training_output_path = _training_output_path
training_seeds = _training_seeds
training_source_selection_strategies = _training_source_selection_strategies
training_teacher_imitation_weights = _training_teacher_imitation_weights
training_teacher_margin_curriculum_profiles = _training_teacher_margin_curriculum_profiles

__all__ = [name for name in globals() if not name.startswith("__")]
