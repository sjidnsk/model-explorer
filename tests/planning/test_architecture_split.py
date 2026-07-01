from __future__ import annotations


def test_planning_facade_matches_split_modules() -> None:
    from model_explorer.policy import planning
    from model_explorer.policy.planning_adapters import ContractCostPlanner
    from model_explorer.policy.planning_anchor import evaluate_candidate_paths
    from model_explorer.policy.planning_types import PathPlanRequest

    assert planning.PathPlanRequest is PathPlanRequest
    assert planning.ContractCostPlanner is ContractCostPlanner
    assert planning.evaluate_candidate_paths is evaluate_candidate_paths
