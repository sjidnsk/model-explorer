from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..io.scenario import Scenario, load_scenario
from .collector import collect_rollout_episode
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
class ExperimentManifest:
    schema_version: str
    scenarios: tuple[Path, ...]
    scenario_groups: tuple[ExperimentScenarioGroup, ...]
    planner_config: dict[str, Any]
    rollout_output: Path
    evaluation_output: Path
    report_output: Path | None = None
    max_candidates: int | None = None
    reward_config: dict[str, Any] | None = None
    reward_ablations: tuple[dict[str, Any], ...] = ()
    train_config: dict[str, Any] | None = None


def load_experiment_manifest(path: str | Path) -> ExperimentManifest:
    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("experiment manifest must be a JSON object")

    schema_version = str(payload.get("schema_version", EXPERIMENT_SCHEMA_VERSION))
    if schema_version != EXPERIMENT_SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {EXPERIMENT_SCHEMA_VERSION!r}")

    base_dir = manifest_path.parent
    scenario_groups = _scenario_groups(payload, base_dir=base_dir)
    scenario_paths = tuple(path for group in scenario_groups for path in group.scenarios)

    outputs = payload.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError("experiment manifest requires outputs")
    if "rollouts" not in outputs:
        raise ValueError("experiment outputs must include outputs.rollouts")
    if "evaluation" not in outputs:
        raise ValueError("experiment outputs must include outputs.evaluation")

    planner_config = _planner_config_with_sidecar(payload.get("planner", {}), base_dir=base_dir)
    max_candidates = payload.get("max_candidates")
    return ExperimentManifest(
        schema_version=schema_version,
        scenarios=scenario_paths,
        scenario_groups=scenario_groups,
        planner_config=planner_config,
        rollout_output=_resolve_path(base_dir, outputs["rollouts"]),
        evaluation_output=_resolve_path(base_dir, outputs["evaluation"]),
        report_output=(None if outputs.get("report") is None else _resolve_path(base_dir, outputs["report"])),
        max_candidates=None if max_candidates is None else int(max_candidates),
        reward_config=_optional_mapping(payload.get("reward")),
        reward_ablations=_reward_ablations(payload.get("reward_ablations")),
        train_config=_optional_mapping(payload.get("train")),
    )


def run_experiment_manifest(path: str | Path) -> dict[str, Any]:
    manifest = load_experiment_manifest(path)
    scenarios = tuple(load_scenario(path) for path in manifest.scenarios)
    planner = planner_from_config(manifest.planner_config)
    episodes = _collect_episodes(
        scenarios,
        planner=planner,
        max_candidates=manifest.max_candidates,
        reward_config=manifest.reward_config,
    )
    write_rollout_episodes_jsonl(manifest.rollout_output, episodes)

    base_evaluation = evaluate_policy_baseline_scenarios(scenarios, planning_adapter=planner)
    evaluation = (
        _grouped_evaluation(manifest, scenarios, planner=planner, aggregate=base_evaluation)
        if len(manifest.scenario_groups) > 1
        else base_evaluation
    )
    manifest.evaluation_output.write_text(json.dumps(evaluation, indent=2, ensure_ascii=False), encoding="utf-8")

    summary = {
        "schema_version": manifest.schema_version,
        "scenario_count": len(scenarios),
        "group_count": len(manifest.scenario_groups),
        "groups": [
            {"name": group.name, "scenario_count": len(group.scenarios)}
            for group in manifest.scenario_groups
        ],
        "planner": str(manifest.planner_config.get("backend", "contract_cost")),
        "rollout_output": str(manifest.rollout_output),
        "evaluation_output": str(manifest.evaluation_output),
        "transition_count": sum(len(episode.transitions) for episode in episodes),
        "rollout_metrics": _aggregate_rollout_metrics(episodes),
        "reward": dict(manifest.reward_config or {}),
    }
    if manifest.reward_ablations:
        summary["reward_ablations"] = _run_reward_ablations(manifest, scenarios, planner=planner)
    if manifest.train_config is not None:
        summary["training"] = _run_training(episodes, manifest.train_config, base_dir=Path(path).parent)
    if manifest.report_output is not None:
        manifest.report_output.write_text(_markdown_report(summary, base_evaluation), encoding="utf-8")
        summary["report_output"] = str(manifest.report_output)
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


def _scenario_paths(value: Any, *, base_dir: Path) -> tuple[Path, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("experiment manifest requires a non-empty scenarios list")
    return tuple(_resolve_path(base_dir, path) for path in value)


def _resolve_path(base_dir: Path, value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else base_dir / path


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
    reward_config: dict[str, Any] | None,
) -> tuple[RolloutEpisode, ...]:
    return tuple(
        collect_rollout_episode(
            scenario,
            max_candidates=max_candidates,
            planning_adapter=planner,
            reward_config=reward_config,
        )
        for scenario in scenarios
    )


def _grouped_evaluation(
    manifest: ExperimentManifest,
    scenarios: tuple[Scenario, ...],
    *,
    planner,
    aggregate: dict[str, Any],
) -> dict[str, Any]:
    groups: dict[str, Any] = {}
    cursor = 0
    for group in manifest.scenario_groups:
        group_scenarios = scenarios[cursor : cursor + len(group.scenarios)]
        cursor += len(group.scenarios)
        groups[group.name] = evaluate_policy_baseline_scenarios(group_scenarios, planning_adapter=planner)

    return {
        "aggregate": aggregate,
        "groups": groups,
        "per_scenario": [
            {
                "path": str(path),
                "metrics": evaluate_policy_baselines(scenario, planning_adapter=planner),
            }
            for path, scenario in zip(manifest.scenarios, scenarios)
        ],
    }


def _run_reward_ablations(manifest: ExperimentManifest, scenarios: tuple[Scenario, ...], *, planner) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for ablation in manifest.reward_ablations:
        episodes = _collect_episodes(
            scenarios,
            planner=planner,
            max_candidates=manifest.max_candidates,
            reward_config=ablation["reward"],
        )
        results[str(ablation["name"])] = {
            "reward": dict(ablation["reward"]),
            "rollout_metrics": _aggregate_rollout_metrics(episodes),
        }
    return results


def _run_training(episodes: tuple[RolloutEpisode, ...], config: dict[str, Any], *, base_dir: Path) -> dict[str, Any]:
    from .training import train_policy_on_episodes

    train_episodes, validation_episodes = _split_training_episodes(
        episodes,
        validation_fraction=float(config.get("validation_fraction", 0.0)),
    )
    checkpoint = _resolve_path(base_dir, config["checkpoint"])
    result = train_policy_on_episodes(
        train_episodes,
        checkpoint_path=checkpoint,
        seed=int(config.get("seed", 0)),
        hidden_size=int(config.get("hidden_size", 64)),
        learning_rate=float(config.get("learning_rate", 1.0e-3)),
        epochs=int(config.get("epochs", 1)),
    )
    loss_log = config.get("loss_log")
    if loss_log is not None:
        _resolve_path(base_dir, loss_log).write_text(json.dumps(result, ensure_ascii=False) + "\n", encoding="utf-8")
    result["checkpoint"] = str(checkpoint)
    if loss_log is not None:
        result["loss_log"] = str(_resolve_path(base_dir, loss_log))
    result["train_episode_count"] = len(train_episodes)
    result["validation_episode_count"] = len(validation_episodes)
    return result


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


def _markdown_report(summary: dict[str, Any], evaluation: dict[str, Any]) -> str:
    metrics = summary["rollout_metrics"]
    lines = [
        "# Model Explorer Experiment Report",
        "",
        f"- schema_version: {summary['schema_version']}",
        f"- planner: {summary['planner']}",
        f"- scenario_count: {summary['scenario_count']}",
        f"- transition_count: {summary['transition_count']}",
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

    lines.extend(["", "## Baselines", "", "| policy | final_coverage_rate | total_path_cost | failure_count | replan_count | value_coverage |", "|---|---:|---:|---:|---:|---:|"])
    for policy_name in ("utility", "coverage_heuristic", "torch_policy"):
        if policy_name not in evaluation:
            continue
        policy_metrics = evaluation[policy_name]
        lines.append(
            "| "
            + " | ".join(
                (
                    policy_name,
                    str(policy_metrics.get("final_coverage_rate", policy_metrics.get("average_final_coverage_rate", 0.0))),
                    str(policy_metrics.get("total_path_cost", 0.0)),
                    str(policy_metrics.get("failure_count", 0)),
                    str(policy_metrics.get("replan_count", 0)),
                    str(policy_metrics.get("value_coverage", 0.0)),
                )
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)
