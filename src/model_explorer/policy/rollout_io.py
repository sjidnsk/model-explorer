from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .features import PolicyObservation
from .rollout import EpisodeMetrics, RolloutEpisode, RolloutInfo, RolloutTransition


def write_rollout_episode(path: str | Path, episode: RolloutEpisode) -> None:
    Path(path).write_text(json.dumps(episode.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")


def read_rollout_episode(path: str | Path) -> RolloutEpisode:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return _episode_from_dict(payload)


def write_rollout_episodes_jsonl(path: str | Path, episodes: list[RolloutEpisode] | tuple[RolloutEpisode, ...]) -> None:
    lines = [json.dumps(episode.to_dict(), ensure_ascii=False) for episode in episodes]
    Path(path).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def read_rollout_episodes(path: str | Path) -> tuple[RolloutEpisode, ...]:
    source = Path(path)
    if source.is_dir():
        episodes: list[RolloutEpisode] = []
        for child in sorted(source.iterdir()):
            if child.suffix.lower() not in {".json", ".jsonl"}:
                continue
            episodes.extend(read_rollout_episodes(child))
        return tuple(episodes)

    if source.suffix.lower() == ".jsonl":
        episodes = []
        for line in source.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            episodes.append(_episode_from_dict(json.loads(line)))
        return tuple(episodes)

    return (read_rollout_episode(source),)


def _episode_from_dict(payload: dict[str, Any]) -> RolloutEpisode:
    transitions = tuple(_transition_from_dict(item) for item in payload["transitions"])
    return RolloutEpisode(
        transitions=transitions,
        metrics=_metrics_from_dict(payload.get("metrics", {})),
    )


def _transition_from_dict(payload: dict[str, Any]) -> RolloutTransition:
    return RolloutTransition(
        observation=_observation_from_dict(payload["observation"]),
        action_index=int(payload["action_index"]),
        log_prob=payload["log_prob"],
        value=payload["value"],
        reward=float(payload["reward"]),
        next_observation=(
            None
            if payload["next_observation"] is None
            else _observation_from_dict(payload["next_observation"])
        ),
        done=bool(payload["done"]),
        info=_rollout_info_from_dict(payload["info"]),
    )


def _observation_from_dict(payload: dict[str, Any]) -> PolicyObservation:
    return PolicyObservation(
        candidate_feature_names=tuple(payload["candidate_feature_names"]),
        candidate_features=tuple(tuple(float(value) for value in row) for row in payload["candidate_features"]),
        global_feature_names=tuple(payload["global_feature_names"]),
        global_features=tuple(float(value) for value in payload["global_features"]),
        action_mask=tuple(bool(value) for value in payload["action_mask"]),
        candidate_cells=tuple(
            None if cell is None else (int(cell[0]), int(cell[1])) for cell in payload["candidate_cells"]
        ),
    )


def _metrics_from_dict(payload: dict[str, Any]) -> EpisodeMetrics:
    return EpisodeMetrics(
        final_coverage_rate=float(payload.get("final_coverage_rate", 0.0)),
        cumulative_coverage_rate_delta=float(payload.get("cumulative_coverage_rate_delta", 0.0)),
        total_path_cost=float(payload.get("total_path_cost", 0.0)),
        average_risk=float(payload.get("average_risk", 0.0)),
        failure_count=int(payload.get("failure_count", 0)),
        replan_count=int(payload.get("replan_count", 0)),
        value_coverage=float(payload.get("value_coverage", 0.0)),
    )


def _rollout_info_from_dict(payload: dict[str, Any]) -> RolloutInfo:
    known_fields = {
        "selected_cell",
        "coverage_rate_delta",
        "path_cost",
        "risk",
        "failure_reason",
        "final_coverage_rate",
        "total_cost",
        "failure_count",
        "replan_count",
    }
    cell = payload.get("selected_cell")
    selected_cell = None if cell is None else (int(cell[0]), int(cell[1]))
    return RolloutInfo(
        selected_cell=selected_cell,
        coverage_rate_delta=float(payload.get("coverage_rate_delta", 0.0)),
        path_cost=float(payload.get("path_cost", 0.0)),
        risk=float(payload.get("risk", 0.0)),
        failure_reason=payload.get("failure_reason"),
        final_coverage_rate=(
            None
            if payload.get("final_coverage_rate") is None
            else float(payload.get("final_coverage_rate"))
        ),
        total_cost=float(payload.get("total_cost", 0.0)),
        failure_count=int(payload.get("failure_count", 0)),
        replan_count=int(payload.get("replan_count", 0)),
        extra={key: value for key, value in payload.items() if key not in known_fields},
    )
