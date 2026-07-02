from __future__ import annotations

from pathlib import Path
from typing import Any

from ..io.scenario import Scenario, load_scenario
from ..policy.collector import collect_rollout_episode
from ..policy.evaluation import evaluate_policy_baseline_scenarios, evaluate_policy_baselines
from ..policy.rollout import RolloutEpisode
from .manifest import ExperimentManifest, ExperimentScenarioGroup


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


def _should_evaluate_trained_policy(config: dict[str, Any]) -> bool:
    return bool(config.get("evaluate_trained_policy", True))


def _comparison_from_evaluation(evaluation: dict[str, Any]) -> dict[str, Any]:
    if "aggregate" in evaluation and isinstance(evaluation["aggregate"], dict):
        return evaluation["aggregate"]
    return evaluation


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


# Public aliases
collect_episodes = _collect_episodes
load_split_scenarios = _load_split_scenarios
grouped_evaluation = _grouped_evaluation
run_reward_ablations = _run_reward_ablations
comparison_from_evaluation = _comparison_from_evaluation
aggregate_rollout_metrics = _aggregate_rollout_metrics

__all__ = (
    "Scenario",
    "collect_rollout_episode",
    "evaluate_policy_baseline_scenarios",
    "evaluate_policy_baselines",
    "RolloutEpisode",
    "ExperimentManifest",
    "ExperimentScenarioGroup",
    "_reward_ablations",
    "_collect_episodes",
    "_load_split_scenarios",
    "_all_split_scenarios",
    "_all_split_episodes",
    "_evaluation_split_name",
    "_split_summaries",
    "_grouped_evaluation",
    "_run_reward_ablations",
    "_should_evaluate_trained_policy",
    "_comparison_from_evaluation",
    "_aggregate_rollout_metrics",
    "collect_episodes",
    "load_split_scenarios",
    "grouped_evaluation",
    "run_reward_ablations",
    "comparison_from_evaluation",
    "aggregate_rollout_metrics",
)
