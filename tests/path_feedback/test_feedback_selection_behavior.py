import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from model_explorer.io.scenario import load_scenario


def minimal_contract(
    *,
    goals=None,
    sequences=None,
    observation_update=None,
    schema_version="model-explorer-contract/v1",
):
    return {
        "schema_version": schema_version,
        "grid": {
            "width": 4,
            "height": 3,
            "resolution": 0.5,
            "frame_id": "moon_local",
            "origin": [1.0, 2.0],
            "layers": ["confidence", "cost"],
        },
        "constraints": {
            "violation_count": 0,
            "passable_ratio": 1.0,
            "reason_counts": {"obstacle": 0},
        },
        "top_goals": goals
        if goals is not None
        else [
            {
                "cell": [2, 1],
                "utility": 0.42,
                "reachable": True,
            }
        ],
        "top_sequences": sequences
        if sequences is not None
        else [
            {
                "cells": [[2, 1]],
                "utility": 0.42,
                "coverage_area": 1.0,
            }
        ],
        "observation_update": observation_update
        if observation_update is not None
        else {"delta_c": 0.25, "visible_cell_count": 3, "updated_cell_count": 3},
    }


def load_contract_from_dict(payload):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "contract.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return load_scenario(path).snapshots[0]


class FeedbackSelectionBehaviorTests(unittest.TestCase):
    def test_feedback_aware_selection_prefers_feasible_alternative_over_blocked_high_coverage_goal(self):
        from model_explorer.policy.feedback_selection_scoring import select_goal_with_path_feedback
        from model_explorer.policy.planning import PathPlanResult

        class FixedPlanner:
            def __init__(self):
                self.action_indices = []

            def plan(self, request):
                self.action_indices.append(request.action_index)
                if request.action_index == 0:
                    return PathPlanResult(
                        feasible=False,
                        path_cost=0.0,
                        path_length=0.0,
                        risk=0.0,
                        failure_reason="goal_blocked",
                        replan_required=True,
                    )
                return PathPlanResult(feasible=True, path_cost=5.0, path_length=5.0, risk=0.2)

        contract = load_contract_from_dict(
            minimal_contract(
                goals=[
                    {
                        "cell": [1, 1],
                        "utility": 0.9,
                        "reachable": True,
                        "expected_coverage_rate_delta": 0.9,
                        "value": 0.8,
                    },
                    {
                        "cell": [2, 1],
                        "utility": 0.2,
                        "reachable": True,
                        "expected_coverage_rate_delta": 0.2,
                        "value": 0.1,
                    },
                ]
            )
        )
        planner = FixedPlanner()

        selection = select_goal_with_path_feedback(
            contract,
            planner=planner,
            current_cell=(0, 0),
            top_k=2,
        )

        self.assertEqual(selection.decision.selected_goal.cell, (2, 1))
        self.assertEqual(selection.decision.status, "selected")
        self.assertEqual(planner.action_indices, [0, 1])
        self.assertLess(selection.scores_by_action_index[0], selection.scores_by_action_index[1])

    def test_feedback_aware_selection_records_top_two_teacher_scores_and_margin(self):
        from model_explorer.policy.feedback_selection_scoring import select_goal_with_path_feedback
        from model_explorer.policy.planning import PathPlanResult

        class FixedPlanner:
            def __init__(self):
                self.action_indices = []

            def plan(self, request):
                self.action_indices.append(request.action_index)
                return PathPlanResult(feasible=True, path_cost=1.0, path_length=1.0, risk=0.1)

        contract = load_contract_from_dict(
            minimal_contract(
                goals=[
                    {
                        "cell": [1, 1],
                        "utility": 0.8,
                        "reachable": True,
                        "expected_coverage_rate_delta": 0.9,
                    },
                    {
                        "cell": [2, 1],
                        "utility": 0.7,
                        "reachable": True,
                        "expected_coverage_rate_delta": 0.4,
                    },
                    {
                        "cell": [3, 1],
                        "utility": 99.0,
                        "reachable": False,
                        "expected_coverage_rate_delta": 1.0,
                    },
                ]
            )
        )

        selection = select_goal_with_path_feedback(
            contract,
            planner=FixedPlanner(),
            current_cell=(0, 0),
            top_k=3,
        )

        self.assertEqual(selection.selected_action_index, 0)
        self.assertEqual(selection.runner_up_action_index, 1)
        self.assertEqual(selection.ranked_action_indices, (0, 1))
        self.assertNotIn(2, selection.ranked_action_indices)
        self.assertAlmostEqual(selection.selected_score, selection.scores_by_action_index[0])
        self.assertAlmostEqual(selection.runner_up_score, selection.scores_by_action_index[1])
        self.assertAlmostEqual(
            selection.score_margin,
            selection.scores_by_action_index[0] - selection.scores_by_action_index[1],
        )
        self.assertGreater(selection.score_margin, 0.0)

    def test_feedback_aware_selection_penalizes_path_cost_risk_and_replan(self):
        from model_explorer.policy.feedback_selection_scoring import select_goal_with_path_feedback
        from model_explorer.policy.planning import PathPlanResult

        class FixedPlanner:
            def plan(self, request):
                results = {
                    0: PathPlanResult(
                        feasible=True,
                        path_cost=1.0,
                        path_length=1.0,
                        risk=0.05,
                        replan_required=True,
                    ),
                    1: PathPlanResult(feasible=True, path_cost=4.0, path_length=4.0, risk=0.2),
                    2: PathPlanResult(feasible=True, path_cost=80.0, path_length=80.0, risk=0.9),
                }
                return results[request.action_index]

        contract = load_contract_from_dict(
            minimal_contract(
                goals=[
                    {
                        "cell": [1, 1],
                        "utility": 0.9,
                        "reachable": True,
                        "expected_coverage_rate_delta": 0.6,
                    },
                    {
                        "cell": [2, 1],
                        "utility": 0.4,
                        "reachable": True,
                        "expected_coverage_rate_delta": 0.55,
                    },
                    {
                        "cell": [3, 1],
                        "utility": 0.8,
                        "reachable": True,
                        "expected_coverage_rate_delta": 0.55,
                    },
                ]
            )
        )

        selection = select_goal_with_path_feedback(
            contract,
            planner=FixedPlanner(),
            current_cell=(0, 0),
            top_k=3,
        )

        self.assertEqual(selection.decision.selected_goal.cell, (2, 1))
        self.assertLess(selection.scores_by_action_index[0], selection.scores_by_action_index[1])
        self.assertLess(selection.scores_by_action_index[2], selection.scores_by_action_index[1])

    def test_feedback_aware_selection_never_evaluates_or_selects_contract_unreachable_candidates(self):
        from model_explorer.policy.feedback_selection_scoring import select_goal_with_path_feedback
        from model_explorer.policy.planning import PathPlanResult

        class RecordingPlanner:
            def __init__(self):
                self.action_indices = []

            def plan(self, request):
                self.action_indices.append(request.action_index)
                return PathPlanResult(feasible=True, path_cost=1.0, path_length=1.0, risk=0.0)

        contract = load_contract_from_dict(
            minimal_contract(
                goals=[
                    {
                        "cell": [1, 1],
                        "utility": 99.0,
                        "reachable": False,
                        "expected_coverage_rate_delta": 1.0,
                    },
                    {
                        "cell": [2, 1],
                        "utility": 0.1,
                        "reachable": True,
                        "expected_coverage_rate_delta": 0.1,
                    },
                ]
            )
        )
        planner = RecordingPlanner()

        selection = select_goal_with_path_feedback(
            contract,
            planner=planner,
            current_cell=(0, 0),
            top_k=2,
        )

        self.assertEqual(selection.decision.selected_goal.cell, (2, 1))
        self.assertEqual(planner.action_indices, [1])
        self.assertNotIn(0, selection.scores_by_action_index)

    def test_feedback_aware_selection_audits_channel_aware_quality_signal_and_blockers(self):
        from model_explorer.policy.feedback_selection_scoring import select_goal_with_path_feedback
        from model_explorer.policy.planning import PathPlanResult

        class ChannelAwarePlanner:
            def plan(self, request):
                reports = {
                    0: {
                        "requested_backend": "channel_aware_astar",
                        "selected_backend": "channel_aware_astar",
                        "status": "selected",
                        "fallback_reason": None,
                        "comparison": {
                            "path_changed": True,
                            "path_cost_delta": 2.0,
                            "channel_cost_delta": -4.0,
                            "high_cost_exposure_delta": -3.0,
                        },
                    },
                    1: {
                        "requested_backend": "channel_aware_astar",
                        "selected_backend": "astar",
                        "status": "fallback",
                        "fallback_reason": "channel_search_failed:goal_blocked",
                        "comparison": {},
                    },
                    2: {
                        "requested_backend": "channel_aware_astar",
                        "selected_backend": "astar",
                        "status": "fallback",
                        "fallback_reason": "channel_candidate_same_as_baseline",
                        "comparison": {},
                    },
                    3: {
                        "requested_backend": "channel_aware_astar",
                        "selected_backend": "astar",
                        "status": "fallback",
                        "fallback_reason": "channel_candidate_not_lower_risk",
                        "comparison": {},
                    },
                }
                return PathPlanResult(
                    feasible=True,
                    path_cost=5.0 + request.action_index,
                    path_length=5.0 + request.action_index,
                    risk=0.1,
                    metadata={"planning_backend_report": reports[request.action_index]},
                )

        contract = load_contract_from_dict(
            minimal_contract(
                goals=[
                    {"cell": [1, 1], "utility": 0.9, "reachable": True, "expected_coverage_rate_delta": 0.9},
                    {"cell": [2, 1], "utility": 0.8, "reachable": True, "expected_coverage_rate_delta": 0.8},
                    {"cell": [3, 1], "utility": 0.7, "reachable": True, "expected_coverage_rate_delta": 0.7},
                    {"cell": [4, 1], "utility": 0.6, "reachable": True, "expected_coverage_rate_delta": 0.6},
                ]
            )
        )

        selection = select_goal_with_path_feedback(
            contract,
            planner=ChannelAwarePlanner(),
            current_cell=(0, 0),
            top_k=4,
        )

        improved = selection.channel_aware_evidence_by_action_index[0]
        self.assertTrue(improved["quality_improvement"])
        self.assertTrue(improved["path_cost_tradeoff"])
        self.assertEqual(improved["recommendation"], "keep")
        self.assertIn("channel_aware_quality_improved", improved["reason_codes"])
        self.assertIn("path_cost_tradeoff", improved["reason_codes"])
        self.assertNotIn("path_cost_regression_failure", improved["reason_codes"])
        self.assertGreater(selection.channel_aware_score_adjustments_by_action_index[0], 0.0)

        self.assertIn("goal_blocked", selection.channel_aware_evidence_by_action_index[1]["reason_codes"])
        self.assertIn("same_as_baseline", selection.channel_aware_evidence_by_action_index[2]["reason_codes"])
        self.assertIn("not_lower_risk", selection.channel_aware_evidence_by_action_index[3]["reason_codes"])

    def test_contract_aware_selection_rejects_quality_regression_before_preference(self):
        from model_explorer.policy.feedback_selection_sources import selected_after_feedback
        from model_explorer.policy.planning import (
            AnchorProjectionCandidateConfig,
            PathCandidateEvaluation,
            PathPlanResult,
        )

        generation = {
            "schema_version": "anchor-projection-candidate/v1",
            "candidate_role": "projected_execution_target",
            "target_binding_mode": "same_action_execution_substitute",
            "source_action_index": 0,
            "policy_target_cell": [2, 1],
            "execution_goal_cell": [1, 1],
            "projected_anchor_cell": [1, 1],
            "projection_distance_cells": 1,
            "projection_distance_m": 1.0,
            "anchor_reachable": True,
            "comparison_scope": "projected_target_anchor_contrast",
            "scope": "projected_target_anchor_contrast",
            "training_use": "not_positive_evidence",
            "sample_weight": 0.0,
            "reject_reason": "pending_source_selection",
            "source_selection_status": "pending_source_selection",
            "evidence_boundary": "source_candidate_pending_selection_not_audit_proxy",
            "audit_proxy_positive_evidence": False,
            "ppo_consumable_action": True,
            "contract_safe": True,
            "trainability_gate": {"status": "eligible_if_source_selected", "reason_codes": []},
        }
        same_action = PathCandidateEvaluation(
            action_index=0,
            cell=(2, 1),
            utility=0.9,
            result=PathPlanResult(feasible=True, path_cost=10.0, path_length=10.0, risk=0.6),
            source_action_index=0,
            candidate_generation=dict(generation),
        )
        alternative = PathCandidateEvaluation(
            action_index=1,
            cell=(0, 2),
            utility=0.5,
            result=PathPlanResult(feasible=True, path_cost=3.0, path_length=3.0, risk=0.2),
        )

        selected = selected_after_feedback(
            (same_action, alternative),
            anchor_projection_candidate_config=AnchorProjectionCandidateConfig(
                enabled=True,
                contract_aware_trainable_target_generation=True,
                prefer_contract_safe_trainable_targets=True,
                max_source_selection_path_cost_regression=2.0,
                max_source_selection_risk_regression=0.2,
            ),
        )

        self.assertIs(selected, alternative)

    def test_planner_validated_selection_can_prefer_distance_exception_without_quality_regression(self):
        from model_explorer.policy.feedback_selection_sources import selected_after_feedback
        from model_explorer.policy.planning import (
            AnchorProjectionCandidateConfig,
            PathCandidateEvaluation,
            PathPlanResult,
        )

        generation = {
            "schema_version": "anchor-projection-candidate/v1",
            "candidate_role": "projected_execution_target",
            "target_binding_mode": "same_action_execution_substitute",
            "source_action_index": 0,
            "policy_target_cell": [3, 1],
            "execution_goal_cell": [0, 1],
            "projected_anchor_cell": [0, 1],
            "projection_distance_cells": 3,
            "projection_distance_m": 1.5,
            "anchor_reachable": True,
            "comparison_scope": "projected_target_anchor_contrast",
            "scope": "projected_target_anchor_contrast",
            "training_use": "not_positive_evidence",
            "sample_weight": 0.0,
            "reject_reason": "pending_source_selection",
            "source_selection_status": "pending_source_selection",
            "evidence_boundary": "source_candidate_pending_selection_not_audit_proxy",
            "audit_proxy_positive_evidence": False,
            "ppo_consumable_action": True,
            "contract_safe": False,
            "default_distance_contract_safe": False,
            "planner_validated_distance_exception": True,
            "planner_validated_exception_safe": True,
            "trainability_gate": {"status": "eligible_if_source_selected", "reason_codes": []},
        }
        exception_candidate = PathCandidateEvaluation(
            action_index=0,
            cell=(3, 1),
            utility=0.9,
            result=PathPlanResult(feasible=True, path_cost=5.0, path_length=5.0, risk=0.1),
            source_action_index=0,
            candidate_generation=dict(generation),
        )
        alternative = PathCandidateEvaluation(
            action_index=1,
            cell=(0, 2),
            utility=0.5,
            result=PathPlanResult(feasible=True, path_cost=4.0, path_length=4.0, risk=0.1),
        )

        selected = selected_after_feedback(
            (exception_candidate, alternative),
            anchor_projection_candidate_config=AnchorProjectionCandidateConfig(
                enabled=True,
                contract_aware_trainable_target_generation=True,
                prefer_contract_safe_trainable_targets=True,
                planner_validated_trainable_target_mining=True,
                allow_planner_validated_distance_exception=True,
                max_source_selection_path_cost_regression=2.0,
                max_source_selection_risk_regression=0.0,
            ),
        )

        self.assertIs(selected, exception_candidate)

    def test_anchor_projection_selection_bonus_is_opt_in_and_bounded_to_projected_candidates(self):
        from model_explorer.policy.feedback_selection_sources import selected_after_feedback
        from model_explorer.policy.planning import (
            AnchorProjectionCandidateConfig,
            PathPlanResult,
            evaluate_candidate_paths,
        )

        request_payload = {
            "schema_version": "path-planner-request/v1",
            "grid": {
                "width": 4,
                "height": 3,
                "resolution": 1.0,
                "origin": [0.0, 0.0],
                "frame_id": "moon_local",
            },
            "cost": [[1.0, 1.0, 1.0, 1.0] for _ in range(3)],
            "passable_mask": [
                [True, True, False, True],
                [True, True, True, True],
                [True, True, True, True],
            ],
            "start": [0, 0],
            "goal": [2, 1],
            "metadata": {"passable_mask_source": "configured"},
        }

        class SelectionBonusPlanner:
            def plan(self, request):
                payload = json.loads(json.dumps(request_payload))
                payload["goal"] = [request.selected_goal.cell[0], request.selected_goal.cell[1]]
                metadata = {
                    "request_payload": payload,
                    "diagnostics": {
                        "search_mode": "platform_aware_astar",
                        "passable_source": "inflated_passable_mask",
                        "footprint_radius_m": 1.0,
                    },
                }
                if request.selected_goal.cell == (2, 1):
                    return PathPlanResult(
                        feasible=False,
                        failure_reason="goal_blocked",
                        replan_required=True,
                        metadata=metadata,
                    )
                if request.selected_goal.cell == (1, 1):
                    return PathPlanResult(
                        feasible=True,
                        path_cost=5.0,
                        path_length=5.0,
                        risk=0.1,
                        metadata=metadata,
                    )
                return PathPlanResult(
                    feasible=True,
                    path_cost=3.0,
                    path_length=3.0,
                    risk=0.1,
                    metadata=metadata,
                )

        contract = load_contract_from_dict(
            minimal_contract(
                goals=[
                    {"cell": [2, 1], "utility": 0.9, "reachable": True},
                    {"cell": [0, 2], "utility": 0.8, "reachable": True},
                ]
            )
        )
        evaluations = evaluate_candidate_paths(
            contract,
            current_cell=(0, 0),
            top_k=2,
            planner=SelectionBonusPlanner(),
            anchor_projection_candidate_config=AnchorProjectionCandidateConfig(enabled=True),
        )

        without_bonus = selected_after_feedback(
            evaluations,
            anchor_projection_candidate_config=AnchorProjectionCandidateConfig(enabled=True),
        )
        with_bonus = selected_after_feedback(
            evaluations,
            anchor_projection_candidate_config=AnchorProjectionCandidateConfig(
                enabled=True,
                source_selection_path_cost_bonus=3.0,
            ),
        )

        self.assertEqual(without_bonus.cell, (0, 2))
        self.assertEqual(with_bonus.cell, (1, 1))
        self.assertEqual(with_bonus.candidate_generation["candidate_role"], "projected_execution_target")



if __name__ == "__main__":
    unittest.main()
