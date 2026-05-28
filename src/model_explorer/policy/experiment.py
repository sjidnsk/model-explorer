from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..io.scenario import Scenario, load_scenario
from .collector import collect_rollout_episode
from .evaluation import evaluate_policy_baseline_scenarios
from .planning import planner_from_config
from .rollout import RolloutEpisode
from .rollout_io import write_rollout_episodes_jsonl


@dataclass(frozen=True)
class ExperimentManifest:
    scenarios: tuple[Path, ...]
    planner_config: dict[str, Any]
    rollout_output: Path
    evaluation_output: Path
    max_candidates: int | None = None
    reward_config: dict[str, Any] | None = None


def load_experiment_manifest(path: str | Path) -> ExperimentManifest:
    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("experiment manifest must be a JSON object")

    base_dir = manifest_path.parent
    scenario_paths = payload.get("scenarios")
    if not isinstance(scenario_paths, list) or not scenario_paths:
        raise ValueError("experiment manifest requires a non-empty scenarios list")

    outputs = payload.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError("experiment manifest requires outputs")
    if "rollouts" not in outputs or "evaluation" not in outputs:
        raise ValueError("experiment outputs must include rollouts and evaluation")

    planner_config = _planner_config_with_sidecar(payload.get("planner", {}), base_dir=base_dir)
    max_candidates = payload.get("max_candidates")
    return ExperimentManifest(
        scenarios=tuple(_resolve_path(base_dir, path) for path in scenario_paths),
        planner_config=planner_config,
        rollout_output=_resolve_path(base_dir, outputs["rollouts"]),
        evaluation_output=_resolve_path(base_dir, outputs["evaluation"]),
        max_candidates=None if max_candidates is None else int(max_candidates),
        reward_config=_optional_mapping(payload.get("reward")),
    )


def run_experiment_manifest(path: str | Path) -> dict[str, Any]:
    manifest = load_experiment_manifest(path)
    scenarios = tuple(load_scenario(path) for path in manifest.scenarios)
    planner = planner_from_config(manifest.planner_config)
    episodes = tuple(
        collect_rollout_episode(
            scenario,
            max_candidates=manifest.max_candidates,
            planning_adapter=planner,
            reward_config=manifest.reward_config,
        )
        for scenario in scenarios
    )
    write_rollout_episodes_jsonl(manifest.rollout_output, episodes)

    evaluation = evaluate_policy_baseline_scenarios(scenarios, planning_adapter=planner)
    manifest.evaluation_output.write_text(json.dumps(evaluation, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "scenario_count": len(scenarios),
        "planner": str(manifest.planner_config.get("backend", "contract_cost")),
        "rollout_output": str(manifest.rollout_output),
        "evaluation_output": str(manifest.evaluation_output),
        "transition_count": sum(len(episode.transitions) for episode in episodes),
        "rollout_metrics": _aggregate_rollout_metrics(episodes),
        "reward": dict(manifest.reward_config or {}),
    }


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


def _resolve_path(base_dir: Path, value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else base_dir / path


def _optional_mapping(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("reward config must be an object")
    return dict(value)


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
        "failure_count": sum(episode.metrics.failure_count for episode in episodes),
        "replan_count": sum(episode.metrics.replan_count for episode in episodes),
        "value_coverage": sum(episode.metrics.value_coverage for episode in episodes),
    }
