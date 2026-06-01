from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from importlib import util as importlib_util
from math import sqrt
from pathlib import Path
from typing import Any

from ..io.scenario import Scenario, load_scenario
from .collector import collect_rollout_episode
from .dataset import summarize_rollout_dataset, summarize_teacher_quality_gates, validate_rollout_dataset
from .evaluation import evaluate_policy_baseline_scenarios, evaluate_policy_baselines
from .planning import planner_from_config
from .rollout import RolloutEpisode
from .rollout_io import write_rollout_episodes_jsonl


EXPERIMENT_SCHEMA_VERSION = "model-explorer-experiment/v1"


@dataclass(frozen=True)
class ExperimentScenarioGroup:
    name: str
    scenarios: tuple[Path, ...]


@dataclass(frozen=True)
class ExperimentSplit:
    name: str
    scenario_groups: tuple[ExperimentScenarioGroup, ...]

    @property
    def scenarios(self) -> tuple[Path, ...]:
        return tuple(path for group in self.scenario_groups for path in group.scenarios)


@dataclass(frozen=True)
class ExperimentManifest:
    schema_version: str
    experiment_name: str
    run_id: str
    scenarios: tuple[Path, ...]
    scenario_groups: tuple[ExperimentScenarioGroup, ...]
    splits: dict[str, ExperimentSplit]
    explicit_splits: bool
    planner_config: dict[str, Any]
    rollout_output: Path
    evaluation_output: Path
    resolved_manifest_output: Path
    report_output: Path | None = None
    dataset_summary_output: Path | None = None
    output_root: Path | None = None
    run_output_dir: Path | None = None
    max_candidates: int | None = None
    selection_strategy: str = "auto"
    reward_config: dict[str, Any] | None = None
    reward_ablations: tuple[dict[str, Any], ...] = ()
    train_config: dict[str, Any] | None = None
    dataset_validation: dict[str, Any] | None = None


def load_experiment_manifest(path: str | Path) -> ExperimentManifest:
    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("experiment manifest must be a JSON object")

    schema_version = str(payload.get("schema_version", EXPERIMENT_SCHEMA_VERSION))
    if schema_version != EXPERIMENT_SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {EXPERIMENT_SCHEMA_VERSION!r}")

    base_dir = manifest_path.parent
    experiment_name = str(payload.get("name", payload.get("experiment_name", manifest_path.stem)))
    run_id = str(payload.get("run_id", "default"))
    explicit_splits = payload.get("splits") is not None
    splits = (
        _manifest_splits(payload["splits"], base_dir=base_dir)
        if explicit_splits
        else {"all": ExperimentSplit(name="all", scenario_groups=_scenario_groups(payload, base_dir=base_dir))}
    )
    scenario_groups = _evaluation_groups_from_splits(splits, explicit_splits=explicit_splits)
    scenario_paths = _unique_scenario_paths(splits)

    outputs = payload.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError("experiment manifest requires outputs")
    output_root = None if outputs.get("root") is None else _resolve_path(base_dir, outputs["root"])
    run_output_dir = None if output_root is None else output_root / experiment_name / run_id
    if "rollouts" not in outputs and run_output_dir is None:
        raise ValueError("experiment outputs must include outputs.rollouts")
    if "evaluation" not in outputs and run_output_dir is None:
        raise ValueError("experiment outputs must include outputs.evaluation")

    planner_config = _planner_config_with_sidecar(payload.get("planner", {}), base_dir=base_dir)
    max_candidates = payload.get("max_candidates")
    selection_strategy = str(payload.get("selection_strategy", "auto"))
    report_output_value = (
        outputs["report"]
        if "report" in outputs
        else (None if run_output_dir is None else run_output_dir / "report.md")
    )
    dataset_summary_output_value = (
        outputs["dataset_summary"]
        if "dataset_summary" in outputs
        else (None if run_output_dir is None else run_output_dir / "dataset-summary.json")
    )
    rollout_output = _resolve_output_path(base_dir, outputs.get("rollouts"), run_output_dir, "rollouts.jsonl")
    evaluation_output = _resolve_output_path(base_dir, outputs.get("evaluation"), run_output_dir, "evaluation.json")
    resolved_manifest_output = (
        _resolve_path(base_dir, outputs["resolved_manifest"])
        if outputs.get("resolved_manifest") is not None
        else (run_output_dir / "manifest.resolved.json" if run_output_dir is not None else evaluation_output.parent / "manifest.resolved.json")
    )
    return ExperimentManifest(
        schema_version=schema_version,
        experiment_name=experiment_name,
        run_id=run_id,
        scenarios=scenario_paths,
        scenario_groups=scenario_groups,
        splits=splits,
        explicit_splits=explicit_splits,
        planner_config=planner_config,
        rollout_output=rollout_output,
        evaluation_output=evaluation_output,
        resolved_manifest_output=resolved_manifest_output,
        report_output=(None if report_output_value is None else _resolve_path(base_dir, report_output_value)),
        dataset_summary_output=(
            None if dataset_summary_output_value is None else _resolve_path(base_dir, dataset_summary_output_value)
        ),
        output_root=output_root,
        run_output_dir=run_output_dir,
        max_candidates=None if max_candidates is None else int(max_candidates),
        selection_strategy=selection_strategy,
        reward_config=_optional_mapping(payload.get("reward")),
        reward_ablations=_reward_ablations(payload.get("reward_ablations")),
        train_config=_optional_mapping(payload.get("train")),
        dataset_validation=_optional_mapping(payload.get("dataset_validation")),
    )


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
            from .training import load_policy_checkpoint

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


def _planner_config_with_sidecar(config: Any, *, base_dir: Path) -> dict[str, Any]:
    if config is None:
        return {}
    if not isinstance(config, dict):
        raise ValueError("planner config must be an object")

    planner_config = dict(config)
    sidecar_path = planner_config.pop("sidecar_grid", None)
    if sidecar_path is not None:
        sidecar_payload = json.loads(_resolve_path(base_dir, sidecar_path).read_text(encoding="utf-8"))
        planner_config["passable_grid"] = (
            sidecar_payload["passable_grid"] if isinstance(sidecar_payload, dict) else sidecar_payload
        )
    path_planner_sidecar = planner_config.get("path_planner_sidecar")
    if path_planner_sidecar is not None:
        planner_config["path_planner_sidecar"] = str(_resolve_path(base_dir, path_planner_sidecar))
    return planner_config


def _scenario_groups(payload: dict[str, Any], *, base_dir: Path) -> tuple[ExperimentScenarioGroup, ...]:
    raw_groups = payload.get("scenario_groups")
    if raw_groups is not None:
        if not isinstance(raw_groups, list) or not raw_groups:
            raise ValueError("scenario_groups must be a non-empty list")
        groups: list[ExperimentScenarioGroup] = []
        for index, item in enumerate(raw_groups):
            if not isinstance(item, dict):
                raise ValueError(f"scenario_groups[{index}] must be an object")
            name = str(item.get("name", f"group-{index}"))
            groups.append(ExperimentScenarioGroup(name=name, scenarios=_scenario_paths(item.get("scenarios"), base_dir=base_dir)))
        return tuple(groups)

    return (ExperimentScenarioGroup(name="default", scenarios=_scenario_paths(payload.get("scenarios"), base_dir=base_dir)),)


def _manifest_splits(value: Any, *, base_dir: Path) -> dict[str, ExperimentSplit]:
    if not isinstance(value, dict):
        raise ValueError("splits must be an object")
    splits: dict[str, ExperimentSplit] = {}
    for split_name in ("train", "validation", "test", "benchmark"):
        raw_split = value.get(split_name)
        groups = _split_scenario_groups(raw_split, base_dir=base_dir, default_name=split_name)
        if groups:
            splits[split_name] = ExperimentSplit(name=split_name, scenario_groups=groups)
    if not splits:
        raise ValueError("splits must define at least one scenario")
    return splits


def _split_scenario_groups(
    value: Any,
    *,
    base_dir: Path,
    default_name: str,
) -> tuple[ExperimentScenarioGroup, ...]:
    if value is None:
        return ()
    if isinstance(value, list):
        if not value:
            return ()
        if all(isinstance(item, dict) for item in value):
            groups = []
            for index, item in enumerate(value):
                name = str(item.get("name", f"{default_name}-{index}"))
                groups.append(
                    ExperimentScenarioGroup(
                        name=name,
                        scenarios=_scenario_paths(item.get("scenarios"), base_dir=base_dir),
                    )
                )
            return tuple(groups)
        return (ExperimentScenarioGroup(name=default_name, scenarios=_scenario_paths(value, base_dir=base_dir)),)
    if isinstance(value, dict):
        if "scenarios" in value:
            return (
                ExperimentScenarioGroup(
                    name=str(value.get("name", default_name)),
                    scenarios=_scenario_paths(value.get("scenarios"), base_dir=base_dir),
                ),
            )
        groups = []
        for group_name, paths in value.items():
            groups.append(
                ExperimentScenarioGroup(
                    name=str(group_name),
                    scenarios=_scenario_paths(paths, base_dir=base_dir),
                )
            )
        return tuple(groups)
    raise ValueError(f"splits.{default_name} must be a list or object")


def _evaluation_groups_from_splits(
    splits: dict[str, ExperimentSplit],
    *,
    explicit_splits: bool,
) -> tuple[ExperimentScenarioGroup, ...]:
    if not explicit_splits:
        return splits["all"].scenario_groups
    for split_name in ("benchmark", "test", "validation", "train"):
        split = splits.get(split_name)
        if split is not None and split.scenario_groups:
            return split.scenario_groups
    return ()


def _unique_scenario_paths(splits: dict[str, ExperimentSplit]) -> tuple[Path, ...]:
    seen: set[Path] = set()
    paths: list[Path] = []
    for split in splits.values():
        for path in split.scenarios:
            if path in seen:
                continue
            seen.add(path)
            paths.append(path)
    return tuple(paths)


def _scenario_paths(value: Any, *, base_dir: Path) -> tuple[Path, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("experiment manifest requires a non-empty scenarios list")
    return tuple(_resolve_path(base_dir, path) for path in value)


def _resolve_path(base_dir: Path, value: Any) -> Path:
    path = Path(str(value))
    return (path if path.is_absolute() else base_dir / path).resolve()


def _resolve_output_path(base_dir: Path, value: Any, run_output_dir: Path | None, default_name: str) -> Path:
    if value is not None:
        return _resolve_path(base_dir, value)
    if run_output_dir is None:
        raise ValueError(f"experiment outputs must include outputs.{default_name}")
    return run_output_dir / default_name


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, payload: Any) -> None:
    _ensure_parent_dir(path)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _optional_mapping(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("reward config must be an object")
    return dict(value)


def _reward_ablations(value: Any) -> tuple[dict[str, Any], ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError("reward_ablations must be a list")
    ablations = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(f"reward_ablations[{index}] must be an object")
        name = str(item.get("name", f"ablation-{index}"))
        reward = item.get("reward", {})
        if not isinstance(reward, dict):
            raise ValueError(f"reward_ablations[{index}].reward must be an object")
        ablations.append({"name": name, "reward": dict(reward)})
    return tuple(ablations)


def _collect_episodes(
    scenarios: tuple[Scenario, ...],
    *,
    planner,
    max_candidates: int | None,
    selection_strategy: str,
    reward_config: dict[str, Any] | None,
) -> tuple[RolloutEpisode, ...]:
    return tuple(
        collect_rollout_episode(
            scenario,
            max_candidates=max_candidates,
            planning_adapter=planner,
            reward_config=reward_config,
            selection_strategy=selection_strategy,
        )
        for scenario in scenarios
    )


def _load_split_scenarios(manifest: ExperimentManifest) -> dict[str, tuple[Scenario, ...]]:
    return {
        split_name: tuple(load_scenario(path) for path in split.scenarios)
        for split_name, split in manifest.splits.items()
    }


def _all_split_scenarios(split_scenarios: dict[str, tuple[Scenario, ...]]) -> tuple[Scenario, ...]:
    return tuple(scenario for scenarios in split_scenarios.values() for scenario in scenarios)


def _all_split_episodes(split_episodes: dict[str, tuple[RolloutEpisode, ...]]) -> tuple[RolloutEpisode, ...]:
    return tuple(episode for episodes in split_episodes.values() for episode in episodes)


def _manifest_inspection_summary(
    manifest: ExperimentManifest,
    *,
    split_scenarios: dict[str, tuple[Scenario, ...]],
    status: str,
) -> dict[str, Any]:
    scenarios = _all_split_scenarios(split_scenarios)
    return {
        "status": status,
        "schema_version": manifest.schema_version,
        "experiment_name": manifest.experiment_name,
        "run_id": manifest.run_id,
        "scenario_count": len(scenarios),
        "group_count": len(manifest.scenario_groups),
        "groups": [
            {"name": group.name, "scenario_count": len(group.scenarios)}
            for group in manifest.scenario_groups
        ],
        "splits": {
            split_name: {
                "scenario_count": len(split.scenarios),
                "groups": {
                    group.name: len(group.scenarios)
                    for group in split.scenario_groups
                },
            }
            for split_name, split in manifest.splits.items()
        },
        "planner": str(manifest.planner_config.get("backend", "contract_cost")),
        "selection_strategy": manifest.selection_strategy,
        "outputs": {
            "rollouts": str(manifest.rollout_output),
            "evaluation": str(manifest.evaluation_output),
            "report": None if manifest.report_output is None else str(manifest.report_output),
            "dataset_summary": None
            if manifest.dataset_summary_output is None
            else str(manifest.dataset_summary_output),
            "resolved_manifest": str(manifest.resolved_manifest_output),
        },
    }


def _would_write_paths(manifest: ExperimentManifest, *, base_dir: Path) -> list[str]:
    paths = [
        str(manifest.rollout_output),
        str(manifest.evaluation_output),
        str(manifest.resolved_manifest_output),
    ]
    if manifest.report_output is not None:
        paths.append(str(manifest.report_output))
    if manifest.dataset_summary_output is not None:
        paths.append(str(manifest.dataset_summary_output))
    if manifest.train_config is not None:
        paths.extend(str(path) for path in _training_would_write_paths(manifest, base_dir=base_dir))
    return paths


def _training_would_write_paths(manifest: ExperimentManifest, *, base_dir: Path) -> tuple[Path, ...]:
    if manifest.train_config is None:
        return ()
    config = manifest.train_config
    seeds = _training_seeds(config)
    architectures = _training_architectures(config)
    source_strategies = _training_source_selection_strategies(config)
    teacher_weights = _training_teacher_imitation_weights(config)
    multi_seed = len(seeds) > 1
    multi_architecture = len(architectures) > 1
    multi_source = len(source_strategies) > 1
    multi_teacher_weight = len(teacher_weights) > 1
    paths: list[Path] = []
    for source_strategy in source_strategies:
        for teacher_weight in teacher_weights:
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
                        base_dir=base_dir,
                        run_output_dir=manifest.run_output_dir,
                        default_name="checkpoint.pt",
                        multi_seed=multi_seed,
                        multi_architecture=multi_architecture,
                        multi_source=multi_source,
                        multi_teacher_weight=multi_teacher_weight,
                        required=True,
                    )
                    loss_log = _training_output_path(
                        config,
                        "loss_log",
                        seed=seed,
                        architecture=architecture_name,
                        selection_strategy=source_strategy,
                        teacher_imitation_weight=teacher_weight,
                        base_dir=base_dir,
                        run_output_dir=manifest.run_output_dir,
                        default_name="losses.jsonl",
                        multi_seed=multi_seed,
                        multi_architecture=multi_architecture,
                        multi_source=multi_source,
                        multi_teacher_weight=multi_teacher_weight,
                        required=False,
                    )
                    paths.append(checkpoint)
                    if loss_log is not None:
                        paths.append(loss_log)
                    paths.append(checkpoint.parent / "training-summary.json")
                    paths.append(checkpoint.parent / "validation-evaluation.json")
    return tuple(paths)


def _evaluation_split_name(manifest: ExperimentManifest) -> str:
    if not manifest.explicit_splits:
        return "all"
    for split_name in ("benchmark", "test", "validation", "train"):
        if split_name in manifest.splits and manifest.splits[split_name].scenarios:
            return split_name
    raise ValueError("experiment manifest has no scenarios to evaluate")


def _split_summaries(
    manifest: ExperimentManifest,
    split_episodes: dict[str, tuple[RolloutEpisode, ...]],
) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    for split_name, split in manifest.splits.items():
        episodes = split_episodes.get(split_name, ())
        split_summary = {
            "scenario_count": len(split.scenarios),
            "episode_count": len(episodes),
            "transition_count": sum(len(episode.transitions) for episode in episodes),
            "groups": {},
        }
        cursor = 0
        groups: dict[str, Any] = {}
        for group in split.scenario_groups:
            group_episodes = episodes[cursor : cursor + len(group.scenarios)]
            cursor += len(group.scenarios)
            groups[group.name] = {
                "scenario_count": len(group.scenarios),
                "episode_count": len(group_episodes),
                "transition_count": sum(len(episode.transitions) for episode in group_episodes),
            }
        split_summary["groups"] = groups
        summaries[split_name] = split_summary
    return summaries


def _grouped_evaluation(
    groups: tuple[ExperimentScenarioGroup, ...],
    scenarios: tuple[Scenario, ...],
    scenario_paths: tuple[Path, ...],
    *,
    planner,
    aggregate: dict[str, Any],
    torch_policy=None,
) -> dict[str, Any]:
    group_results: dict[str, Any] = {}
    per_scenario: list[dict[str, Any]] = []
    cursor = 0
    for group in groups:
        group_scenarios = scenarios[cursor : cursor + len(group.scenarios)]
        group_paths = scenario_paths[cursor : cursor + len(group.scenarios)]
        cursor += len(group.scenarios)
        group_results[group.name] = evaluate_policy_baseline_scenarios(
            group_scenarios,
            torch_policy=torch_policy,
            planning_adapter=planner,
        )
        per_scenario.extend(
            {
                "path": str(path),
                "group": group.name,
                "metrics": evaluate_policy_baselines(
                    scenario,
                    torch_policy=torch_policy,
                    planning_adapter=planner,
                ),
            }
            for path, scenario in zip(group_paths, group_scenarios)
        )

    return {
        "aggregate": aggregate,
        "groups": group_results,
        "per_scenario": per_scenario,
    }


def _run_reward_ablations(manifest: ExperimentManifest, scenarios: tuple[Scenario, ...], *, planner) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for ablation in manifest.reward_ablations:
        episodes = _collect_episodes(
            scenarios,
            planner=planner,
            max_candidates=manifest.max_candidates,
            selection_strategy=manifest.selection_strategy,
            reward_config=ablation["reward"],
        )
        results[str(ablation["name"])] = {
            "reward": dict(ablation["reward"]),
            "rollout_metrics": _aggregate_rollout_metrics(episodes),
        }
    return results


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
    from .training import train_policy_on_episodes

    from .training import load_policy_checkpoint

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
    multi_seed = len(seeds) > 1
    multi_architecture = len(architectures) > 1
    multi_source = len(source_strategies) > 1
    multi_teacher_weight = len(teacher_weights) > 1
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
        for teacher_weight in teacher_weights:
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
                        base_dir=base_dir,
                        run_output_dir=run_output_dir,
                        default_name="checkpoint.pt",
                        multi_seed=multi_seed,
                        multi_architecture=multi_architecture,
                        multi_source=multi_source,
                        multi_teacher_weight=multi_teacher_weight,
                        required=True,
                    )
                    loss_log = _training_output_path(
                        config,
                        "loss_log",
                        seed=seed,
                        architecture=architecture_name,
                        selection_strategy=source_strategy,
                        teacher_imitation_weight=teacher_weight,
                        base_dir=base_dir,
                        run_output_dir=run_output_dir,
                        default_name="losses.jsonl",
                        multi_seed=multi_seed,
                        multi_architecture=multi_architecture,
                        multi_source=multi_source,
                        multi_teacher_weight=multi_teacher_weight,
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

    best_run = _select_best_training_run(
        runs,
        metric=str(config.get("best_metric", "final_coverage_rate")),
        policy=str(config.get("best_policy", "torch_policy")),
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
    selected["source_comparison"] = _training_source_comparison(runs)
    selected["distillation_matrix"] = _training_distillation_matrix(runs)
    selected["source_weight_comparison"] = list(selected["distillation_matrix"])
    selected["multi_seed_summary"] = _multi_seed_evaluation_summary(runs)
    selected["multi_seed_loss_summary"] = _loss_summary(runs)
    selected["multi_seed_delta_summary"] = _multi_seed_delta_summary(runs)
    selected["best_selection"] = {
        "policy": str(config.get("best_policy", "torch_policy")),
        "metric": str(config.get("best_metric", "final_coverage_rate")),
        "mode": "max",
        "value": _training_run_metric(
            best_run,
            policy=str(config.get("best_policy", "torch_policy")),
            metric=str(config.get("best_metric", "final_coverage_rate")),
        ),
        "reason": (
            f"max {config.get('best_policy', 'torch_policy')}."
            f"{config.get('best_metric', 'final_coverage_rate')} on validation evaluation; "
            f"source={_normalize_training_source_name(best_run.get('training_data_selection_strategy'))}; "
            f"teacher_imitation_weight={float(best_run.get('teacher_imitation_weight', 0.0))}"
        ),
    }
    return selected


def _should_evaluate_trained_policy(config: dict[str, Any]) -> bool:
    return bool(config.get("evaluate_trained_policy", True))


def _comparison_from_evaluation(evaluation: dict[str, Any]) -> dict[str, Any]:
    if "aggregate" in evaluation and isinstance(evaluation["aggregate"], dict):
        return evaluation["aggregate"]
    return evaluation


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


def _training_source_comparison(runs: list[dict[str, Any]]) -> dict[str, Any]:
    comparison: dict[str, Any] = {}
    for run in runs:
        source = run.get("training_data_selection_strategy")
        if source is None:
            source = run.get("training_source", {}).get("primary_selection_strategy")
        source_name = _normalize_training_source_name(None if source is None else str(source))
        existing = comparison.get(source_name)
        if existing is not None:
            continue
        torch_metrics = _comparison_from_evaluation(run.get("validation_evaluation", {})).get("torch_policy", {})
        torch_metrics = torch_metrics if isinstance(torch_metrics, dict) else {}
        comparison[source_name] = {
            "checkpoint": run.get("checkpoint"),
            "training_source": dict(run.get("training_source", {})),
            "teacher_imitation": dict(run.get("teacher_imitation", {})),
            "teacher_margin_summary": dict(run.get("teacher_margin_summary", {})),
            "teacher_agreement": _teacher_agreement_summary(torch_metrics),
            "baseline_deltas": dict(run.get("baseline_deltas", {})),
        }
    return comparison


def _training_distillation_matrix(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    matrix: list[dict[str, Any]] = []
    for run in runs:
        torch_metrics = _comparison_from_evaluation(run.get("validation_evaluation", {})).get("torch_policy", {})
        torch_metrics = torch_metrics if isinstance(torch_metrics, dict) else {}
        dataset_summary = run.get("dataset_summary", {})
        dataset_summary = dataset_summary if isinstance(dataset_summary, dict) else {}
        source = run.get("training_data_selection_strategy")
        matrix.append(
            {
                "source_selection_strategy": _normalize_training_source_name(None if source is None else str(source)),
                "teacher_imitation_weight": float(run.get("teacher_imitation_weight", 0.0)),
                "architecture": run.get("architecture"),
                "seed": run.get("seed"),
                "checkpoint": run.get("checkpoint"),
                "training_source": dict(run.get("training_source", {})),
                "teacher_imitation": dict(run.get("teacher_imitation", {})),
                "teacher_margin_summary": dict(run.get("teacher_margin_summary", {})),
                "teacher_quality_gates": dict(run.get("teacher_quality_gates", {})),
                "teacher_agreement": _teacher_agreement_summary(torch_metrics),
                "baseline_deltas": dict(run.get("baseline_deltas", {})),
                "data_class": dataset_summary.get("data_class"),
                "dataset_id": dataset_summary.get("dataset_id"),
                "mask_stress_augmented": bool(dataset_summary.get("mask_stress_augmented", False)),
            }
        )
    return matrix


def _teacher_agreement_summary(torch_metrics: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "feedback_aware_action_agreement_rate",
        "feedback_aware_selected_cell_agreement_rate",
        "feedback_aware_top2_action_agreement_rate",
        "feedback_aware_topk_action_agreement_rate",
        "feedback_aware_teacher_rank_mean",
        "feedback_aware_margin_bucket_agreement",
    )
    return {field: torch_metrics[field] for field in fields if field in torch_metrics}


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
    base_dir: Path,
    run_output_dir: Path | None,
    default_name: str,
    multi_seed: bool,
    multi_architecture: bool,
    multi_source: bool,
    multi_teacher_weight: bool,
    required: bool,
) -> Path | None:
    source_name = _normalize_training_source_name(selection_strategy)
    teacher_weight_name = _normalize_teacher_weight_name(teacher_imitation_weight)
    value = config.get(key)
    if value is not None:
        text = str(value)
        formatted = text.format(
            seed=seed,
            architecture=architecture,
            selection_strategy=source_name,
            teacher_imitation_weight=teacher_weight_name,
            teacher_weight=teacher_weight_name,
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
        if multi_architecture:
            parent = parent / architecture
        return parent / f"seed-{seed}" / default_name
    if required:
        raise ValueError(f"train.{key} is required when outputs.root is not configured")
    return None


def _normalize_training_source_name(selection_strategy: str | None) -> str:
    return "manifest" if selection_strategy is None else str(selection_strategy).strip().replace("/", "_")


def _normalize_teacher_weight_name(value: float) -> str:
    text = f"{float(value):g}"
    return text.replace("-", "neg-").replace(".", "p")


def _path_parent_contains_placeholder(path_text: str, placeholder: str) -> bool:
    return any(placeholder in part for part in Path(path_text).parent.parts)


def _path_parent_contains_any_placeholder(path_text: str, placeholders: tuple[str, ...]) -> bool:
    return any(_path_parent_contains_placeholder(path_text, placeholder) for placeholder in placeholders)


def _select_best_training_run(runs: list[dict[str, Any]], *, policy: str, metric: str) -> dict[str, Any]:
    if not runs:
        raise ValueError("training must produce at least one run")
    return max(runs, key=lambda run: _training_run_metric(run, policy=policy, metric=metric))


def _training_run_metric(run: dict[str, Any], *, policy: str, metric: str) -> float:
    evaluation = run.get("validation_evaluation", {})
    if not isinstance(evaluation, dict):
        return float("-inf")
    evaluation = _comparison_from_evaluation(evaluation)
    policy_metrics = evaluation.get(policy, {})
    if not isinstance(policy_metrics, dict):
        return float("-inf")
    value = policy_metrics.get(metric)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("-inf")


def _multi_seed_evaluation_summary(runs: list[dict[str, Any]]) -> dict[str, Any]:
    values: dict[str, dict[str, list[float]]] = {}
    for run in runs:
        evaluation = run.get("validation_evaluation", {})
        if not isinstance(evaluation, dict):
            continue
        evaluation = _comparison_from_evaluation(evaluation)
        for policy, metrics in evaluation.items():
            if not isinstance(metrics, dict):
                continue
            policy_values = values.setdefault(str(policy), {})
            for metric, value in metrics.items():
                if isinstance(value, bool):
                    continue
                try:
                    numeric = float(value)
                except (TypeError, ValueError):
                    continue
                metric_values = policy_values.setdefault(str(metric), [])
                metric_values.append(numeric)
    return {
        policy: {metric: _numeric_stats(tuple(metric_values)) for metric, metric_values in metrics.items()}
        for policy, metrics in values.items()
    }


_BASELINE_DELTA_METRICS = (
    "final_coverage_rate",
    "cumulative_coverage_rate_delta",
    "total_path_cost",
    "average_risk",
    "failure_count",
    "value_coverage",
)


def _baseline_deltas(evaluation: dict[str, Any], *, policy: str = "torch_policy") -> dict[str, Any]:
    policy_metrics = evaluation.get(policy)
    if not isinstance(policy_metrics, dict):
        return {}
    deltas: dict[str, Any] = {policy: {}}
    for baseline_name in _baseline_names(evaluation, policy=policy):
        baseline_metrics = evaluation.get(baseline_name)
        if not isinstance(baseline_metrics, dict):
            continue
        deltas[policy][baseline_name] = {
            metric: _metric_value(policy_metrics, metric) - _metric_value(baseline_metrics, metric)
            for metric in _BASELINE_DELTA_METRICS
        }
    return deltas


def _baseline_names(evaluation: dict[str, Any], *, policy: str) -> tuple[str, ...]:
    preferred = ("utility", "coverage_heuristic", "feedback_aware")
    names = [name for name in preferred if name in evaluation and name != policy]
    names.extend(
        sorted(
            str(name)
            for name, metrics in evaluation.items()
            if name not in set(preferred)
            and name != policy
            and isinstance(metrics, dict)
        )
    )
    return tuple(names)


def _multi_seed_delta_summary(runs: list[dict[str, Any]]) -> dict[str, Any]:
    values: dict[str, dict[str, list[float]]] = {}
    for run in runs:
        evaluation = run.get("validation_evaluation", {})
        if not isinstance(evaluation, dict):
            continue
        evaluation = _comparison_from_evaluation(evaluation)
        deltas = _baseline_deltas(evaluation).get("torch_policy", {})
        if not isinstance(deltas, dict):
            continue
        for baseline_name, metrics in deltas.items():
            if not isinstance(metrics, dict):
                continue
            baseline_values = values.setdefault(str(baseline_name), {})
            for metric, value in metrics.items():
                baseline_values.setdefault(str(metric), []).append(float(value))
    return {
        baseline: {metric: _numeric_stats(tuple(metric_values)) for metric, metric_values in metrics.items()}
        for baseline, metrics in values.items()
    }


def _metric_value(metrics: dict[str, Any], metric: str) -> float:
    value = metrics.get(metric)
    if value is None and metric == "final_coverage_rate":
        value = metrics.get("average_final_coverage_rate")
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _numeric_stats(values: tuple[float, ...]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    average = sum(values) / len(values)
    variance = sum((value - average) ** 2 for value in values) / len(values)
    return {
        "mean": average,
        "std": sqrt(variance),
        "min": min(values),
        "max": max(values),
    }


def _environment_metadata(*, base_dir: Path) -> dict[str, Any]:
    torch_available = importlib_util.find_spec("torch") is not None
    torch_version = None
    if torch_available:
        try:
            import torch

            torch_version = str(torch.__version__)
        except Exception:
            torch_available = False
            torch_version = None
    return {
        "python_version": sys.version,
        "torch": {"available": torch_available, "version": torch_version},
        "git": _git_metadata(base_dir=base_dir),
    }


def _git_metadata(*, base_dir: Path) -> dict[str, Any]:
    commit = _git_output(base_dir, "rev-parse", "HEAD")
    status = _git_output(base_dir, "status", "--porcelain")
    return {
        "commit": commit,
        "dirty": bool(status),
    }


def _git_output(base_dir: Path, *args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=base_dir,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
    except Exception:
        return None
    return completed.stdout.strip()


def _resolved_manifest_payload(manifest: ExperimentManifest) -> dict[str, Any]:
    return {
        "schema_version": manifest.schema_version,
        "experiment_name": manifest.experiment_name,
        "run_id": manifest.run_id,
        "scenario_paths": [str(path) for path in manifest.scenarios],
        "splits": {
            name: [str(path) for path in split.scenarios]
            for name, split in manifest.splits.items()
        },
        "split_groups": {
            name: {
                group.name: [str(path) for path in group.scenarios]
                for group in split.scenario_groups
            }
            for name, split in manifest.splits.items()
        },
        "outputs": {
            "rollouts": str(manifest.rollout_output),
            "evaluation": str(manifest.evaluation_output),
            "report": None if manifest.report_output is None else str(manifest.report_output),
            "dataset_summary": None
            if manifest.dataset_summary_output is None
            else str(manifest.dataset_summary_output),
            "resolved_manifest": str(manifest.resolved_manifest_output),
            "root": None if manifest.output_root is None else str(manifest.output_root),
            "run_dir": None if manifest.run_output_dir is None else str(manifest.run_output_dir),
        },
        "planner": dict(manifest.planner_config),
        "selection_strategy": manifest.selection_strategy,
        "reward": dict(manifest.reward_config or {}),
        "dataset_gates": dict(manifest.dataset_validation or {}),
        "train_config": dict(manifest.train_config or {}),
        "split_config": {
            name: {
                "scenario_count": len(split.scenarios),
                "groups": {
                    group.name: len(group.scenarios) for group in split.scenario_groups
                },
            }
            for name, split in manifest.splits.items()
        },
    }


def _aggregate_rollout_metrics(episodes: tuple[RolloutEpisode, ...]) -> dict[str, Any]:
    return {
        "episode_count": len(episodes),
        "final_coverage_rate": (
            sum(episode.metrics.final_coverage_rate for episode in episodes) / len(episodes)
            if episodes
            else 0.0
        ),
        "cumulative_coverage_rate_delta": sum(
            episode.metrics.cumulative_coverage_rate_delta for episode in episodes
        ),
        "total_path_cost": sum(episode.metrics.total_path_cost for episode in episodes),
        "average_risk": (
            sum(episode.metrics.average_risk for episode in episodes) / len(episodes)
            if episodes
            else 0.0
        ),
        "failure_count": sum(episode.metrics.failure_count for episode in episodes),
        "replan_count": sum(episode.metrics.replan_count for episode in episodes),
        "value_coverage": sum(episode.metrics.value_coverage for episode in episodes),
        "total_reward": sum(
            transition.reward
            for episode in episodes
            for transition in episode.transitions
        ),
    }


def _daily_report_summary(summary: dict[str, Any], evaluation: dict[str, Any]) -> dict[str, Any]:
    comparison = _comparison_from_evaluation(evaluation)
    baseline_deltas = _baseline_deltas(comparison)
    return {
        "policy_ranking": _policy_ranking(comparison),
        "baseline_deltas": baseline_deltas,
        "architecture_deltas": _architecture_deltas(summary, baseline_deltas),
        "per_group_winners": _per_group_winners(evaluation),
        "failure_scenarios": _failure_scenarios(evaluation),
        "gate_summary": _gate_summary(summary.get("dataset_summary")),
    }


def _architecture_deltas(summary: dict[str, Any], baseline_deltas: dict[str, Any]) -> dict[str, Any]:
    training = summary.get("training")
    if not isinstance(training, dict):
        return {}
    runs = training.get("runs")
    if isinstance(runs, list):
        per_architecture: dict[str, Any] = {}
        for run in runs:
            if not isinstance(run, dict):
                continue
            architecture = run.get("architecture")
            deltas = run.get("baseline_deltas")
            if not architecture or not isinstance(deltas, dict) or not deltas:
                continue
            per_architecture.setdefault(str(architecture), deltas)
        if per_architecture:
            return per_architecture
    architecture = training.get("architecture")
    torch_deltas = baseline_deltas.get("torch_policy") if isinstance(baseline_deltas, dict) else None
    if not architecture or not isinstance(torch_deltas, dict):
        return {}
    return {str(architecture): torch_deltas}


def _policy_ranking(evaluation_comparison: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for policy_name, metrics in evaluation_comparison.items():
        if not isinstance(metrics, dict):
            continue
        rows.append(
            {
                "policy": str(policy_name),
                "final_coverage_rate": _metric_value(metrics, "final_coverage_rate"),
                "cumulative_coverage_rate_delta": _metric_value(metrics, "cumulative_coverage_rate_delta"),
                "total_path_cost": _metric_value(metrics, "total_path_cost"),
                "average_risk": _metric_value(metrics, "average_risk"),
                "failure_count": int(_metric_value(metrics, "failure_count")),
                "value_coverage": _metric_value(metrics, "value_coverage"),
            }
        )
    rows.sort(
        key=lambda item: (
            -float(item["final_coverage_rate"]),
            int(item["failure_count"]),
            float(item["total_path_cost"]),
            str(item["policy"]),
        )
    )
    for index, item in enumerate(rows, start=1):
        item["rank"] = index
    return rows


def _ordered_policy_names(evaluation_comparison: dict[str, Any]) -> tuple[str, ...]:
    preferred = ("utility", "coverage_heuristic", "feedback_aware", "torch_policy")
    names = [name for name in preferred if name in evaluation_comparison]
    names.extend(
        sorted(
            str(name)
            for name, metrics in evaluation_comparison.items()
            if name not in set(preferred) and isinstance(metrics, dict)
        )
    )
    return tuple(names)


def _per_group_winners(evaluation: dict[str, Any]) -> dict[str, Any]:
    groups = evaluation.get("groups") if isinstance(evaluation, dict) else None
    if not isinstance(groups, dict):
        return {}
    winners: dict[str, Any] = {}
    for group_name, group_evaluation in groups.items():
        if not isinstance(group_evaluation, dict):
            continue
        ranking = _policy_ranking(_comparison_from_evaluation(group_evaluation))
        winners[str(group_name)] = None if not ranking else ranking[0]
    return winners


def _failure_scenarios(evaluation: dict[str, Any]) -> list[dict[str, Any]]:
    per_scenario = evaluation.get("per_scenario") if isinstance(evaluation, dict) else None
    if not isinstance(per_scenario, list):
        return []
    failures: list[dict[str, Any]] = []
    for item in per_scenario:
        if not isinstance(item, dict):
            continue
        metrics = item.get("metrics")
        if not isinstance(metrics, dict):
            continue
        failed_policies: list[str] = []
        for policy_name, policy_metrics in metrics.items():
            if not isinstance(policy_metrics, dict):
                continue
            selected_cells = policy_metrics.get("selected_cells", [])
            has_no_selection = isinstance(selected_cells, list) and any(cell is None for cell in selected_cells)
            if _metric_value(policy_metrics, "failure_count") > 0 or has_no_selection:
                failed_policies.append(str(policy_name))
        if failed_policies:
            failures.append({"path": str(item.get("path", "")), "policies": failed_policies})
    return failures


def _gate_summary(dataset_summary: Any) -> dict[str, Any]:
    if not isinstance(dataset_summary, dict):
        return {"status": "not_configured", "warnings": [], "errors": [], "violation_count": 0}
    validation_gates = dataset_summary.get("validation_gates")
    violations = []
    status = "not_configured"
    if isinstance(validation_gates, dict):
        status = str(validation_gates.get("status", "unknown"))
        raw_violations = validation_gates.get("violations", [])
        violations = raw_violations if isinstance(raw_violations, list) else []
    return {
        "status": status,
        "warnings": list(dataset_summary.get("warnings", [])),
        "errors": list(dataset_summary.get("errors", [])),
        "violation_count": len(violations),
        "violations": violations,
    }


def _markdown_report(summary: dict[str, Any], evaluation: dict[str, Any]) -> str:
    metrics = summary["rollout_metrics"]
    evaluation_comparison = _comparison_from_evaluation(evaluation)
    lines = [
        "# Model Explorer Experiment Report",
        "",
        f"- schema_version: {summary['schema_version']}",
        f"- planner: {summary['planner']}",
        f"- scenario_count: {summary['scenario_count']}",
        f"- transition_count: {summary['transition_count']}",
        "- benchmark_scope: synthetic smoke / regression suite; not a real-world generalization benchmark",
        "",
        "## Rollout Metrics",
        "",
        "| metric | value |",
        "|---|---:|",
    ]
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

    dataset_summary = summary.get("dataset_summary")
    if isinstance(dataset_summary, dict):
        lines.extend(["", "## Dataset Summary", "", "| metric | value |", "|---|---:|"])
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

    split_summaries = summary.get("split_summaries", {})
    benchmark_summary = split_summaries.get("benchmark") if isinstance(split_summaries, dict) else None
    if isinstance(benchmark_summary, dict):
        lines.extend(
            [
                "",
                "## Benchmark Groups",
                "",
                "| group | scenarios | episodes | transitions |",
                "|---|---:|---:|---:|",
            ]
        )
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

    policy_ranking = summary.get("policy_ranking", [])
    lines.extend(
        [
            "",
            "## Policy Ranking",
            "",
            "| rank | policy | final_coverage_rate | failures | total_path_cost | value_coverage |",
            "|---:|---|---:|---:|---:|---:|",
        ]
    )
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

    lines.extend(
        [
            "",
            "## Torch Policy Deltas",
            "",
            "| baseline | metric | delta |",
            "|---|---|---:|",
        ]
    )
    torch_deltas_for_section = summary.get("baseline_deltas", {})
    torch_deltas_for_section = (
        torch_deltas_for_section.get("torch_policy")
        if isinstance(torch_deltas_for_section, dict)
        else None
    )
    if isinstance(torch_deltas_for_section, dict) and torch_deltas_for_section:
        for baseline_name, metrics in torch_deltas_for_section.items():
            if not isinstance(metrics, dict):
                continue
            for metric, delta in metrics.items():
                lines.append(f"| {baseline_name} | {metric} | {delta} |")
    else:
        lines.append("| none | torch_policy unavailable | 0.0 |")

    architecture_deltas = summary.get("architecture_deltas", {})
    lines.extend(
        [
            "",
            "## Architecture Deltas",
            "",
            "| architecture | baseline | metric | delta |",
            "|---|---|---|---:|",
        ]
    )
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

    per_group_winners = summary.get("per_group_winners", {})
    lines.extend(
        [
            "",
            "## Per-Group Winners",
            "",
            "| group | winner | final_coverage_rate | failures |",
            "|---|---|---:|---:|",
        ]
    )
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

    failure_scenarios = summary.get("failure_scenarios", [])
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

    gate_summary = summary.get("gate_summary", {})
    lines.extend(["", "## Gate Summary", ""])
    if isinstance(gate_summary, dict):
        lines.append(f"- status: {gate_summary.get('status', 'unknown')}")
        lines.append(f"- warning_count: {len(gate_summary.get('warnings', []))}")
        lines.append(f"- violation_count: {gate_summary.get('violation_count', 0)}")

    if "training" in summary:
        training = summary["training"]
        lines.extend(["", "## Training", "", "| field | value |", "|---|---:|"])
        for key in (
            "checkpoint",
            "architecture",
            "best_checkpoint_path",
            "last_checkpoint_path",
            "best_seed",
            "seed",
            "run_count",
            "epochs",
            "sample_count",
            "train_episode_count",
            "validation_episode_count",
            "loss",
            "policy_loss",
            "value_loss",
            "entropy",
        ):
            if key in training:
                lines.append(f"| {key} | {training[key]} |")

        source_comparison = training.get("source_comparison")
        if isinstance(source_comparison, dict) and source_comparison:
            lines.extend(
                [
                    "",
                    "## Training Source Comparison",
                    "",
                    "| source | primary_selection_strategy | feedback_aware_action_agreement_rate | feedback_aware_top2_action_agreement_rate |",
                    "|---|---|---:|---:|",
                ]
            )
            for source_name, source_summary in source_comparison.items():
                if not isinstance(source_summary, dict):
                    continue
                training_source = source_summary.get("training_source", {})
                training_source = training_source if isinstance(training_source, dict) else {}
                teacher_agreement = source_summary.get("teacher_agreement", {})
                teacher_agreement = teacher_agreement if isinstance(teacher_agreement, dict) else {}
                lines.append(
                    "| "
                    + " | ".join(
                        (
                            str(source_name),
                            str(training_source.get("primary_selection_strategy", "")),
                            str(teacher_agreement.get("feedback_aware_action_agreement_rate", 0.0)),
                            str(teacher_agreement.get("feedback_aware_top2_action_agreement_rate", 0.0)),
                        )
                    )
                    + " |"
                )

        distillation_matrix = training.get("distillation_matrix")
        if isinstance(distillation_matrix, list) and distillation_matrix:
            lines.extend(
                [
                    "",
                    "## Distillation Matrix",
                    "",
                    "| source | teacher_imitation_weight | teacher_quality_gate_status | feedback_aware_action_agreement_rate | feedback_aware_delta_final_coverage_rate |",
                    "|---|---:|---|---:|---:|",
                ]
            )
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
                lines.append(
                    "| "
                    + " | ".join(
                        (
                            str(record.get("source_selection_strategy", "")),
                            str(record.get("teacher_imitation_weight", 0.0)),
                            str(gates.get("status", "unknown")),
                            str(teacher_agreement.get("feedback_aware_action_agreement_rate", 0.0)),
                            str(feedback_delta.get("final_coverage_rate", 0.0)),
                        )
                    )
                    + " |"
                )

        architecture_config = training.get("architecture_config")
        architecture_diagnostics = training.get("architecture_diagnostics")
        if isinstance(architecture_config, dict) or isinstance(architecture_diagnostics, dict):
            lines.extend(["", "## Architecture Diagnostics", "", "| field | value |", "|---|---|"])
            if isinstance(architecture_config, dict):
                for key in sorted(architecture_config):
                    lines.append(f"| {key} | {architecture_config[key]} |")
            if isinstance(architecture_diagnostics, dict):
                for key in (
                    "architecture",
                    "observation_schema_version",
                    "candidate_feature_dim",
                    "global_feature_dim",
                    "missing_indicator_dim",
                    "mask_valid_action_count",
                ):
                    if key not in architecture_diagnostics:
                        continue
                    value = architecture_diagnostics[key]
                    if isinstance(value, dict):
                        value = json.dumps(value, ensure_ascii=False, sort_keys=True)
                    lines.append(f"| {key} | {value} |")
        lines.extend(["", "### dataset_summary", "", "| metric | value |", "|---|---:|"])
        training_dataset = training.get("dataset_summary", {})
        if isinstance(training_dataset, dict):
            for key in (
                "data_class",
                "dataset_id",
                "region",
                "generator_version",
                "roi_count",
                "episode_count",
                "transition_count",
                "trainable_transition_count",
                "failure_transition_count",
                "coverage_delta_total",
                "total_path_cost",
                "average_risk",
            ):
                lines.append(f"| {key} | {training_dataset.get(key, 0)} |")

        best_selection = training.get("best_selection", {})
        if isinstance(best_selection, dict):
            lines.extend(["", "## Best Checkpoint", "", "| field | value |", "|---|---|"])
            for key in ("best_seed", "best_checkpoint_path", "last_checkpoint_path"):
                if key in training:
                    lines.append(f"| {key} | {training[key]} |")
            for key in ("policy", "metric", "mode", "value", "reason"):
                if key in best_selection:
                    lines.append(f"| {key} | {best_selection[key]} |")

        multi_seed_summary = training.get("multi_seed_summary", {})
        if isinstance(multi_seed_summary, dict) and multi_seed_summary:
            lines.extend(
                [
                    "",
                    "## Multi-Seed Summary",
                    "",
                    "| policy | metric | mean | std | min | max |",
                    "|---|---|---:|---:|---:|---:|",
                ]
            )
            for policy, metrics in multi_seed_summary.items():
                if not isinstance(metrics, dict):
                    continue
                for metric, stats in metrics.items():
                    if not isinstance(stats, dict):
                        continue
                    lines.append(
                        "| "
                        + " | ".join(
                            (
                                str(policy),
                                str(metric),
                                str(stats.get("mean", 0.0)),
                                str(stats.get("std", 0.0)),
                                str(stats.get("min", 0.0)),
                                str(stats.get("max", 0.0)),
                            )
                        )
                        + " |"
                    )

        runs = training.get("runs", [])
        if isinstance(runs, list) and runs:
            lines.extend(
                [
                    "",
                    "## Per-Seed Metrics",
                    "",
                    "| architecture | seed | checkpoint | final_coverage_rate | total_path_cost | loss | policy_loss | value_loss | entropy |",
                    "|---|---:|---|---:|---:|---:|---:|---:|---:|",
                ]
            )
            for run in runs:
                if not isinstance(run, dict):
                    continue
                torch_metrics = {}
                validation_evaluation = run.get("validation_evaluation", {})
                if isinstance(validation_evaluation, dict):
                    torch_metrics = validation_evaluation.get("torch_policy", {}) or {}
                lines.append(
                    "| "
                    + " | ".join(
                        (
                            str(run.get("architecture", "")),
                            str(run.get("seed", "")),
                            str(run.get("checkpoint", "")),
                            str(torch_metrics.get("final_coverage_rate", 0.0)),
                            str(torch_metrics.get("total_path_cost", 0.0)),
                            str(run.get("loss", 0.0)),
                            str(run.get("policy_loss", 0.0)),
                            str(run.get("value_loss", 0.0)),
                            str(run.get("entropy", 0.0)),
                        )
                    )
                    + " |"
                )

            loss_summary = _loss_summary(runs)
            lines.extend(
                [
                    "",
                    "## Loss Summary",
                    "",
                    "| metric | mean | std | min | max |",
                    "|---|---:|---:|---:|---:|",
                ]
            )
            for metric, stats in loss_summary.items():
                lines.append(
                    "| "
                    + " | ".join(
                        (
                            metric,
                            str(stats.get("mean", 0.0)),
                            str(stats.get("std", 0.0)),
                            str(stats.get("min", 0.0)),
                            str(stats.get("max", 0.0)),
                        )
                    )
                    + " |"
                )

        lines.extend(["", "## Training Quality", "", "| seed | warnings |", "|---:|---|"])
        for run in runs if isinstance(runs, list) else []:
            if not isinstance(run, dict):
                continue
            warnings = run.get("warnings", [])
            warning_text = ", ".join(str(item) for item in warnings) if isinstance(warnings, list) else ""
            lines.append(f"| {run.get('seed', '')} | {warning_text} |")

    baseline_deltas = summary.get("baseline_deltas", {})
    torch_deltas = baseline_deltas.get("torch_policy") if isinstance(baseline_deltas, dict) else None
    if isinstance(torch_deltas, dict) and torch_deltas:
        lines.extend(
            [
                "",
                "## Baseline Comparison",
                "",
                "| baseline | metric | delta |",
                "|---|---|---:|",
            ]
        )
        for baseline_name, metrics in torch_deltas.items():
            if not isinstance(metrics, dict):
                continue
            for metric, delta in metrics.items():
                lines.append(f"| {baseline_name} | {metric} | {delta} |")

    lines.extend(["", "## Baselines", "", "| policy | final_coverage_rate | cumulative_coverage_rate_delta | total_path_cost | average_risk | failure_count | replan_count | value_coverage |", "|---|---:|---:|---:|---:|---:|---:|---:|"])
    for policy_name in _ordered_policy_names(evaluation_comparison):
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
    return "\n".join(lines)


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
