from __future__ import annotations

from collections.abc import Iterable

from ..core.interfaces import ModelExplorerContract
from ..decision.selector import select_goal
from ..io.scenario import Scenario
from .features import extract_policy_observation
from .reward import compute_step_reward
from .rollout import EpisodeMetrics, RolloutEpisode, RolloutInfo, RolloutTransition


def collect_rollout_episode(
    scenario_or_snapshots: Scenario | Iterable[ModelExplorerContract],
    *,
    policy=None,
    max_candidates: int | None = None,
) -> RolloutEpisode:
    snapshots = (
        scenario_or_snapshots.snapshots
        if isinstance(scenario_or_snapshots, Scenario)
        else tuple(scenario_or_snapshots)
    )

    transitions: list[RolloutTransition] = []
    total_path_cost = 0.0
    total_risk = 0.0
    cumulative_coverage_rate_delta = 0.0
    failure_count = 0
    final_coverage_rate = 0.0

    for index, contract in enumerate(snapshots):
        observation = extract_policy_observation(
            contract,
            step_index=index,
            remaining_steps=max(len(snapshots) - index - 1, 0),
            max_candidates=max_candidates,
        )
        decision = select_goal(contract, policy=policy)
        selected_goal = decision.selected_goal
        failure_reason = None if selected_goal is not None else "no_reachable_goal"
        if failure_reason is not None:
            failure_count += 1
            continue

        action_index = _selected_action_index(contract, selected_goal.cell)
        reward_info = compute_step_reward(selected_goal, contract.observation_update, failure_reason=failure_reason)
        next_observation = (
            extract_policy_observation(
                snapshots[index + 1],
                step_index=index + 1,
                remaining_steps=max(len(snapshots) - index - 2, 0),
                max_candidates=max_candidates,
            )
            if index + 1 < len(snapshots)
            else None
        )
        final_coverage_rate = _coverage_rate(contract.observation_update, fallback=final_coverage_rate)
        cumulative_coverage_rate_delta += reward_info.coverage_rate_delta
        total_path_cost += reward_info.path_cost
        total_risk += reward_info.risk

        transitions.append(
            RolloutTransition(
                observation=observation,
                action_index=action_index,
                log_prob=None,
                value=None,
                reward=reward_info.reward,
                next_observation=next_observation,
                done=index + 1 >= len(snapshots),
                info=RolloutInfo(
                    selected_cell=selected_goal.cell,
                    coverage_rate_delta=reward_info.coverage_rate_delta,
                    path_cost=reward_info.path_cost,
                    risk=reward_info.risk,
                    failure_reason=reward_info.failure_reason,
                    final_coverage_rate=final_coverage_rate if index + 1 >= len(snapshots) else None,
                    total_cost=total_path_cost,
                    failure_count=failure_count,
                    replan_count=0,
                ),
            )
        )

    average_risk = total_risk / len(transitions) if transitions else 0.0
    return RolloutEpisode(
        transitions=tuple(transitions),
        metrics=EpisodeMetrics(
            final_coverage_rate=final_coverage_rate,
            cumulative_coverage_rate_delta=cumulative_coverage_rate_delta,
            total_path_cost=total_path_cost,
            average_risk=average_risk,
            failure_count=failure_count,
            replan_count=0,
            value_coverage=0.0,
        ),
    )


def _selected_action_index(contract: ModelExplorerContract, selected_cell: tuple[int, int]) -> int:
    for index, goal in enumerate(contract.top_goals):
        if goal.cell == selected_cell:
            return index
    raise ValueError(f"selected cell {selected_cell!r} is not present in top_goals")


def _coverage_rate(observation_update: dict, *, fallback: float) -> float:
    value = observation_update.get("coverage_rate")
    if isinstance(value, bool) or value is None:
        return fallback
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback
