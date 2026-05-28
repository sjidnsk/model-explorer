from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .features import PolicyObservation


@dataclass(frozen=True)
class RolloutInfo:
    selected_cell: tuple[int, int] | None = None
    coverage_rate_delta: float = 0.0
    path_cost: float = 0.0
    risk: float = 0.0
    failure_reason: str | None = None
    final_coverage_rate: float | None = None
    total_cost: float = 0.0
    failure_count: int = 0
    replan_count: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "selected_cell": _cell_to_list(self.selected_cell),
            "coverage_rate_delta": float(self.coverage_rate_delta),
            "path_cost": float(self.path_cost),
            "risk": float(self.risk),
            "failure_reason": self.failure_reason,
            "final_coverage_rate": None
            if self.final_coverage_rate is None
            else float(self.final_coverage_rate),
            "total_cost": float(self.total_cost),
            "failure_count": int(self.failure_count),
            "replan_count": int(self.replan_count),
        }
        payload.update(self.extra)
        return payload


@dataclass(frozen=True)
class RolloutTransition:
    observation: PolicyObservation
    action_index: int
    log_prob: float | None
    value: float | None
    reward: float
    next_observation: PolicyObservation | None
    done: bool
    info: RolloutInfo

    def __post_init__(self) -> None:
        if self.action_index == -1:
            if self.info.failure_reason is None:
                raise ValueError("no-op action requires a failure reason")
            return
        if self.action_index < 0 or self.action_index >= len(self.observation.action_mask):
            raise ValueError("action_index is outside the observation action mask")
        if not self.observation.action_mask[self.action_index]:
            raise ValueError("masked action index cannot be recorded as selected")

    def to_dict(self) -> dict[str, Any]:
        return {
            "observation": _observation_to_dict(self.observation),
            "action_index": int(self.action_index),
            "action_mask": [bool(value) for value in self.observation.action_mask],
            "log_prob": None if self.log_prob is None else float(self.log_prob),
            "value": None if self.value is None else float(self.value),
            "reward": float(self.reward),
            "next_observation": None
            if self.next_observation is None
            else _observation_to_dict(self.next_observation),
            "done": bool(self.done),
            "info": self.info.to_dict(),
        }


@dataclass(frozen=True)
class EpisodeMetrics:
    final_coverage_rate: float = 0.0
    cumulative_coverage_rate_delta: float = 0.0
    total_path_cost: float = 0.0
    average_risk: float = 0.0
    failure_count: int = 0
    replan_count: int = 0
    value_coverage: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "final_coverage_rate": float(self.final_coverage_rate),
            "cumulative_coverage_rate_delta": float(self.cumulative_coverage_rate_delta),
            "total_path_cost": float(self.total_path_cost),
            "average_risk": float(self.average_risk),
            "failure_count": int(self.failure_count),
            "replan_count": int(self.replan_count),
            "value_coverage": float(self.value_coverage),
        }


@dataclass(frozen=True)
class RolloutEpisode:
    transitions: tuple[RolloutTransition, ...]
    metrics: EpisodeMetrics

    def to_dict(self) -> dict[str, Any]:
        return {
            "transitions": [transition.to_dict() for transition in self.transitions],
            "metrics": self.metrics.to_dict(),
        }


def _observation_to_dict(observation: PolicyObservation) -> dict[str, Any]:
    return {
        "candidate_feature_names": list(observation.candidate_feature_names),
        "candidate_features": [list(features) for features in observation.candidate_features],
        "global_feature_names": list(observation.global_feature_names),
        "global_features": list(observation.global_features),
        "action_mask": [bool(value) for value in observation.action_mask],
        "candidate_cells": [_cell_to_list(cell) for cell in observation.candidate_cells],
        "candidate_missing_feature_names": [
            list(names) for names in observation.candidate_missing_feature_names
        ],
    }


def _cell_to_list(cell: tuple[int, int] | None) -> list[int] | None:
    if cell is None:
        return None
    return [int(cell[0]), int(cell[1])]
