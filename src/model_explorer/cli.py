from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .core.interfaces import ModelExplorerContract
from .io.scenario import load_scenario
from .orchestration.loop import run_exploration_loop


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the model-explorer orchestration loop.")
    parser.add_argument("scenario", type=Path, help="Path to a model-explorer-contract/v1 JSON scenario.")
    args = parser.parse_args(argv)

    scenario = load_scenario(args.scenario)
    results = run_exploration_loop(scenario)
    summaries = [
        _result_to_summary(result, contract)
        for result, contract in zip(results, scenario.snapshots, strict=True)
    ]
    print(json.dumps(summaries, indent=2, ensure_ascii=False))
    return 0


def _result_to_summary(result, contract: ModelExplorerContract) -> dict[str, Any]:
    selected_goal = result.decision.selected_goal
    selected_cell = list(selected_goal.cell) if selected_goal is not None else None
    selected_world = list(contract.cell_to_world(selected_goal.cell)) if selected_goal is not None else None
    return {
        "step_index": result.step_index,
        "status": result.decision.status,
        "selected_cell": selected_cell,
        "selected_world": selected_world,
        "selected_utility": selected_goal.utility if selected_goal is not None else None,
        "observation_update": result.observation_update,
        "replan_reasons": list(result.replan_reasons),
    }
