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


class AnchorProjectionSourceSelectionTests(unittest.TestCase):
    def test_source_selected_anchor_projection_is_marked_trainable_only_after_selection(self):
        from model_explorer.policy.feedback_selection_anchor import annotate_source_selected_anchor_projection
        from model_explorer.policy.planning import (
            AnchorProjectionCandidateConfig,
            PathPlanResult,
            evaluate_candidate_paths,
            path_feedback_summary,
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

        class SelectionPlanner:
            def plan(self, request):
                payload = json.loads(json.dumps(request_payload))
                payload["goal"] = [request.selected_goal.cell[0], request.selected_goal.cell[1]]
                common_metadata = {
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
                        metadata=common_metadata,
                    )
                return PathPlanResult(
                    feasible=True,
                    path_cost=1.0 if request.selected_goal.cell == (1, 1) else 10.0,
                    path_length=1.0 if request.selected_goal.cell == (1, 1) else 10.0,
                    metadata=common_metadata,
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
            planner=SelectionPlanner(),
            anchor_projection_candidate_config=AnchorProjectionCandidateConfig(enabled=True),
        )
        selected = min(
            [item for item in evaluations if item.result.feasible],
            key=lambda item: item.result.path_cost,
        )
        feedback = annotate_source_selected_anchor_projection(
            path_feedback_summary(evaluations),
            selected_evaluation=selected,
        )
        projected = next(
            item for item in feedback["candidates"] if item["candidate_role"] == "projected_execution_target"
        )
        projection = projected["platform_goal_feasibility"]["anchor_projection"]

        self.assertEqual(selected.cell, (1, 1))
        self.assertEqual(projected["candidate_generation"]["source_selection_status"], "source_selected")
        self.assertEqual(projected["candidate_generation"]["training_use"], "trainable_anchor_projection_contrast")
        self.assertEqual(projected["candidate_generation"]["sample_weight"], 1.0)
        self.assertIsNone(projected["candidate_generation"]["reject_reason"])
        self.assertEqual(projection["training_use"], "trainable_anchor_projection_contrast")
        self.assertEqual(projection["comparison_scope"], "projected_target_anchor_contrast")
        self.assertEqual(projection["sample_weight"], 1.0)
        self.assertIsNone(projection["reject_reason"])

    def test_source_selected_anchor_projection_quality_regression_is_not_trainable(self):
        from model_explorer.policy.feedback_selection_anchor import annotate_source_selected_anchor_projection
        from model_explorer.policy.planning import (
            AnchorProjectionCandidateConfig,
            PathCandidateEvaluation,
            PathPlanResult,
        )

        selected_generation = {
            "schema_version": "anchor-projection-candidate/v1",
            "candidate_role": "projected_execution_target",
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
        }
        feedback = {
            "candidates": [
                {
                    "action_index": 2,
                    "source_action_index": 0,
                    "cell": [1, 1],
                    "candidate_role": "projected_execution_target",
                    "reachable": True,
                    "replan_required": False,
                    "path_cost": 10.0,
                    "risk": 0.6,
                    "utility": 0.8,
                    "candidate_generation": dict(selected_generation),
                    "platform_goal_feasibility": {
                        "classification": "platform_inflated_goal_blocked",
                        "anchor_projection": dict(selected_generation),
                    },
                },
                {
                    "action_index": 1,
                    "cell": [0, 2],
                    "candidate_role": "policy_target",
                    "reachable": True,
                    "replan_required": False,
                    "path_cost": 3.0,
                    "risk": 0.2,
                    "utility": 0.7,
                },
            ]
        }
        selected = PathCandidateEvaluation(
            action_index=2,
            cell=(1, 1),
            utility=0.8,
            result=PathPlanResult(feasible=True, path_cost=10.0, risk=0.6),
            source_action_index=0,
            candidate_generation=dict(selected_generation),
        )

        annotated = annotate_source_selected_anchor_projection(
            feedback,
            selected_evaluation=selected,
            anchor_projection_candidate_config=AnchorProjectionCandidateConfig(
                enabled=True,
                max_source_selection_path_cost_regression=2.0,
                max_source_selection_risk_regression=0.2,
            ),
        )

        projected = annotated["candidates"][0]["candidate_generation"]
        projection = annotated["candidates"][0]["platform_goal_feasibility"]["anchor_projection"]
        self.assertEqual(projected["source_selection_status"], "source_selected_quality_regression")
        self.assertEqual(projected["training_use"], "not_positive_evidence")
        self.assertEqual(projected["sample_weight"], 0.0)
        self.assertEqual(projected["reject_reason"], "source_selection_quality_regression")
        self.assertEqual(projected["source_selection_path_cost_margin_vs_best_alternative"], 7.0)
        self.assertAlmostEqual(projected["source_selection_risk_margin_vs_best_alternative"], 0.4)
        self.assertEqual(
            projected["source_selection_best_alternative_scope"],
            "reachable_non_replan_candidates_including_policy_and_projected_targets",
        )
        self.assertEqual(projected["source_selection_best_alternative_candidate_role"], "policy_target")
        self.assertEqual(projection["reject_reason"], "source_selection_quality_regression")



if __name__ == "__main__":
    unittest.main()
