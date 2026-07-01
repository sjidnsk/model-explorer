from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
    from .evaluation import _reward_ablations

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


def _system_calibration_config(config: dict[str, Any]) -> dict[str, Any] | None:
    value = config.get("system_calibration")
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("train.system_calibration must be an object")
    return dict(value)


def _system_path_feedback_gate_enabled(config: dict[str, Any]) -> bool:
    return "path_feedback_gate" in config or "acceptance_gate" in config


def _system_path_feedback_gate_config(config: dict[str, Any]) -> dict[str, Any]:
    gate = config.get("path_feedback_gate", {})
    if gate is None:
        gate = {}
    if not isinstance(gate, dict):
        raise ValueError("system_calibration.path_feedback_gate must be an object")
    result = dict(gate)
    if "acceptance_gate" in config:
        result["acceptance_gate"] = config["acceptance_gate"]
    return result


def _system_sample_quality_config(config: dict[str, Any]) -> dict[str, Any] | None:
    value = config.get("sample_quality")
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("system_calibration.sample_quality must be an object")
    if not bool(value.get("enabled", False)):
        return None
    return dict(value)


def _manifest_inspection_summary(
    manifest: ExperimentManifest,
    *,
    split_scenarios: dict[str, tuple[Scenario, ...]],
    status: str,
) -> dict[str, Any]:
    from .evaluation import _all_split_scenarios

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


def _training_would_write_paths(manifest: ExperimentManifest, *, base_dir: Path) -> tuple[Path, ...]:
    from .training_matrix import (
        _normalize_training_architecture_name,
        _training_architectures,
        _training_output_path,
        _training_seeds,
        _training_source_selection_strategies,
        _training_teacher_imitation_weights,
        _training_teacher_margin_curriculum_profiles,
    )

    if manifest.train_config is None:
        return ()
    config = manifest.train_config
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
    paths: list[Path] = []
    for source_strategy in source_strategies:
        for teacher_weight in teacher_weights:
            for curriculum_profile in curriculum_profiles:
                profile_name = str(curriculum_profile["name"])
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
                            run_output_dir=manifest.run_output_dir,
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
                            run_output_dir=manifest.run_output_dir,
                            default_name="losses.jsonl",
                            multi_seed=multi_seed,
                            multi_architecture=multi_architecture,
                            multi_source=multi_source,
                            multi_teacher_weight=multi_teacher_weight,
                            multi_curriculum_profile=multi_curriculum_profile,
                            required=False,
                        )
                        paths.append(checkpoint)
                        if loss_log is not None:
                            paths.append(loss_log)
                        paths.append(checkpoint.parent / "training-summary.json")
                        paths.append(checkpoint.parent / "validation-evaluation.json")
    return tuple(paths)


# Public aliases
resolve_path = _resolve_path
resolve_output_path = _resolve_output_path
manifest_inspection_summary = _manifest_inspection_summary
would_write_paths = _would_write_paths
resolved_manifest_payload = _resolved_manifest_payload

__all__ = [name for name in globals() if not name.startswith("__")]
