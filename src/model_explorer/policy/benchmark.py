from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any


BENCHMARK_GROUPS = (
    "coverage_dominant",
    "risk_dominant",
    "path_cost_dominant",
    "sparse_candidates",
    "high_unreachable_rate",
    "missing_experimental_fields",
)

_DIFFICULTY_CANDIDATES = {
    "easy": 4,
    "medium": 5,
    "hard": 6,
}


def generate_synthetic_benchmark_suite(
    output_dir: str | Path,
    *,
    seed: int = 0,
    scenario_count: int = 1,
    groups: tuple[str, ...] | list[str] | None = None,
    difficulty: str = "medium",
    experiment_name: str = "synthetic-benchmark",
    run_id: str | None = None,
) -> dict[str, Any]:
    selected_groups = _selected_groups(groups)
    count = int(scenario_count)
    if count <= 0:
        raise ValueError("scenario_count must be positive")
    if difficulty not in _DIFFICULTY_CANDIDATES:
        raise ValueError(f"difficulty must be one of {', '.join(sorted(_DIFFICULTY_CANDIDATES))}")

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    benchmark_split: dict[str, list[str]] = {}
    group_summaries: dict[str, dict[str, Any]] = {}
    for group in selected_groups:
        paths: list[str] = []
        candidate_count = 0
        reachable_candidate_count = 0
        for index in range(count):
            scenario_name = f"{group}-{index:03d}.json"
            payload = _scenario_payload(
                group,
                index=index,
                seed=seed,
                difficulty=difficulty,
            )
            _write_json(root / scenario_name, payload)
            paths.append(scenario_name)
            goals = payload["top_goals"]
            candidate_count += len(goals)
            reachable_candidate_count += sum(1 for goal in goals if bool(goal["reachable"]))
        benchmark_split[group] = paths
        group_summaries[group] = {
            "scenario_count": count,
            "candidate_count": candidate_count,
            "reachable_candidate_count": reachable_candidate_count,
            "missing_experimental_fields": group == "missing_experimental_fields",
        }

    all_empty_action_masks = tuple(selected_groups) == ("high_unreachable_rate",)
    manifest = {
        "schema_version": "model-explorer-experiment/v1",
        "name": experiment_name,
        "run_id": run_id or f"seed-{int(seed)}-{difficulty}",
        "splits": {"benchmark": benchmark_split},
        "max_candidates": _DIFFICULTY_CANDIDATES[difficulty],
        "planner": {"backend": "contract_cost"},
        "reward": {
            "path_cost_weight": 0.1,
            "path_cost_normalizer": 100.0,
            "risk_weight": 0.2,
            "failure_penalty": 1.0,
        },
        "outputs": {"root": "out"},
    }
    if not all_empty_action_masks:
        manifest["dataset_validation"] = _dataset_gates(
            group_count=len(selected_groups),
            scenario_count=count,
            has_high_unreachable_rate="high_unreachable_rate" in selected_groups,
        )
    summary = {
        "schema_version": "model-explorer-synthetic-benchmark-summary/v1",
        "seed": int(seed),
        "difficulty": difficulty,
        "group_count": len(selected_groups),
        "scenario_count": len(selected_groups) * count,
        "max_candidates": _DIFFICULTY_CANDIDATES[difficulty],
        "groups": group_summaries,
    }
    manifest_path = root / "synthetic-benchmark-experiment.json"
    _write_json(manifest_path, manifest)
    _write_json(root / "synthetic-benchmark-summary.json", summary)
    return {
        "status": "generated",
        "manifest": str(manifest_path),
        "groups": list(selected_groups),
        "scenario_count": len(selected_groups) * count,
        "seed": int(seed),
        "difficulty": difficulty,
        "summary": summary,
    }


def _selected_groups(groups: tuple[str, ...] | list[str] | None) -> tuple[str, ...]:
    if groups is None or not groups:
        return BENCHMARK_GROUPS
    unknown = sorted(set(groups) - set(BENCHMARK_GROUPS))
    if unknown:
        raise ValueError(f"unknown benchmark group: {unknown[0]}")
    return tuple(str(group) for group in groups)


def _scenario_payload(group: str, *, index: int, seed: int, difficulty: str) -> dict[str, Any]:
    rng = random.Random(f"{seed}:{difficulty}:{group}:{index}")
    candidate_count = _candidate_count(group, difficulty)
    width = 8 + (index % 3)
    height = 6 + (index % 2)
    goals = [
        _goal_payload(group, goal_index, rng=rng, width=width, height=height, difficulty=difficulty)
        for goal_index in range(candidate_count)
    ]
    coverage_delta = _coverage_delta_for_group(group, index=index, difficulty=difficulty)
    return {
        "schema_version": "model-explorer-contract/v1",
        "grid": {
            "width": width,
            "height": height,
            "resolution": 0.5,
            "frame_id": "moon_local",
            "origin": [0.0, 0.0],
            "layers": ["confidence", "cost", "risk"],
        },
        "constraints": {
            "violation_count": 0 if group != "high_unreachable_rate" else max(candidate_count - 1, 1),
            "passable_ratio": 0.85 if group != "high_unreachable_rate" else 0.35,
            "reason_counts": {"obstacle": 0 if group != "high_unreachable_rate" else candidate_count},
        },
        "top_goals": goals,
        "top_sequences": [
            {
                "cells": [goals[0]["cell"]] if goals else [[0, 0]],
                "utility": goals[0]["utility"] if goals else 0.0,
                "coverage_area": 1.0 + index,
            }
        ],
        "observation_update": {
            "coverage_rate": min(1.0, coverage_delta + 0.05 * index),
            "coverage_rate_delta": coverage_delta,
            "visible_cell_count": int(10 + 2 * index),
            "updated_cell_count": int(8 + index),
            "value_coverage": 0.1 + coverage_delta,
        },
        "experimental_fields": [] if group == "missing_experimental_fields" else [
            "expected_coverage_rate_delta",
            "expected_new_coverage_area",
            "information_gain",
            "confidence_gain",
            "value",
            "risk",
            "path_cost",
            "energy_cost",
        ],
    }


def _candidate_count(group: str, difficulty: str) -> int:
    if group == "sparse_candidates":
        return 1 if difficulty == "easy" else 2
    return _DIFFICULTY_CANDIDATES[difficulty]


def _goal_payload(
    group: str,
    index: int,
    *,
    rng: random.Random,
    width: int,
    height: int,
    difficulty: str,
) -> dict[str, Any]:
    reachable = _reachable(group, index=index)
    cell = [int((index * 2 + rng.randrange(width)) % width), int((index + rng.randrange(height)) % height)]
    utility = round(0.9 - 0.08 * index + rng.random() * 0.04, 4)
    if group == "coverage_dominant" and index == 1:
        utility = 0.25
    if group == "missing_experimental_fields":
        return {"cell": cell, "utility": utility, "reachable": reachable}

    difficulty_scale = {"easy": 0.8, "medium": 1.0, "hard": 1.25}[difficulty]
    coverage = 0.08 + 0.03 * index
    risk = 0.1 + 0.08 * index
    path_cost = 1.0 + 1.5 * index
    if group == "coverage_dominant":
        coverage = 0.55 if index == 1 else 0.12 + 0.02 * index
    elif group == "risk_dominant":
        risk = 0.75 if index == 0 else 0.08 + 0.04 * index
    elif group == "path_cost_dominant":
        path_cost = 18.0 if index == 0 else 1.0 + 0.7 * index
    elif group == "high_unreachable_rate":
        coverage = 0.4 + 0.05 * index
        risk = 0.35 + 0.05 * index

    return {
        "cell": cell,
        "utility": utility,
        "reachable": reachable,
        "expected_coverage_rate_delta": round(coverage, 4),
        "expected_new_coverage_area": round(coverage * 100.0, 4),
        "information_gain": round(0.2 + 0.05 * index, 4),
        "confidence_gain": round(0.15 + 0.03 * index, 4),
        "value": round(0.3 + 0.04 * index, 4),
        "risk": round(risk * difficulty_scale, 4),
        "path_cost": round(path_cost * difficulty_scale, 4),
        "energy_cost": round((0.5 + 0.5 * index) * difficulty_scale, 4),
    }


def _reachable(group: str, *, index: int) -> bool:
    if group == "high_unreachable_rate":
        return False
    if group in {"coverage_dominant", "risk_dominant", "path_cost_dominant"}:
        return index < 3
    return index == 0 or index % 3 != 0


def _coverage_delta_for_group(group: str, *, index: int, difficulty: str) -> float:
    base = {
        "coverage_dominant": 0.35,
        "risk_dominant": 0.18,
        "path_cost_dominant": 0.16,
        "sparse_candidates": 0.1,
        "high_unreachable_rate": 0.0,
        "missing_experimental_fields": 0.08,
    }[group]
    scale = {"easy": 0.8, "medium": 1.0, "hard": 1.15}[difficulty]
    return round(base * scale + 0.01 * index, 4)


def _dataset_gates(*, group_count: int, scenario_count: int, has_high_unreachable_rate: bool) -> dict[str, Any]:
    total = group_count * scenario_count
    expected_empty = scenario_count if has_high_unreachable_rate else 0
    expected_failure_rate = expected_empty / total if total else 0.0
    return {
        "min_episode_count": total,
        "min_transition_count": total,
        "min_trainable_transition_count": max(total - expected_empty, 1),
        "max_empty_action_mask_count": expected_empty,
        "max_invalid_action_mask_count": 0,
        "max_failure_rate": max(0.25, expected_failure_rate),
        "require_finite_reward": True,
        "max_missing_experimental_feature_rate": 0.8,
        "min_action_mask_valid_mean": 0.2,
        "max_unreachable_candidate_rate": 0.75,
        "min_reward_std": 0.01,
    }


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
