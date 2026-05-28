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
    transitions = tuple(_transition_from_dict(item) for item in payload["transitions"])
    return RolloutEpisode(
        transitions=transitions,
        metrics=EpisodeMetrics(**payload["metrics"]),
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
        info=RolloutInfo(**payload["info"]),
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
