from __future__ import annotations

from ..core.interfaces import ExplorerDecision, ModelExplorerContract


def select_goal(contract: ModelExplorerContract) -> ExplorerDecision:
    reachable_goals = tuple(goal for goal in contract.top_goals if goal.reachable)
    ranked_goals = tuple(sorted(reachable_goals, key=lambda goal: (-goal.utility, goal.cell[0], goal.cell[1])))

    if not ranked_goals:
        return ExplorerDecision(status="no_reachable_goal", selected_goal=None, ranked_goals=ranked_goals)

    return ExplorerDecision(status="selected", selected_goal=ranked_goals[0], ranked_goals=ranked_goals)
