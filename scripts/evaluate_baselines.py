from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from model_explorer.io.scenario import load_scenario
from model_explorer.policy.evaluation import evaluate_policy_baseline_scenarios, evaluate_policy_baselines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate model-explorer policy baselines.")
    parser.add_argument("scenario", type=Path, nargs="+", help="Path to one or more contract/scenario JSON files.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Optional torch policy checkpoint.")
    args = parser.parse_args(argv)

    scenarios = [load_scenario(path) for path in args.scenario]
    policy = None
    if args.checkpoint is not None:
        from model_explorer.policy.training import load_policy_checkpoint

        policy = load_policy_checkpoint(args.checkpoint)
    report = (
        evaluate_policy_baselines(scenarios[0], torch_policy=policy)
        if len(scenarios) == 1
        else evaluate_policy_baseline_scenarios(scenarios, torch_policy=policy)
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
