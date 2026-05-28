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
from model_explorer.policy.collector import collect_rollout_episode
from model_explorer.policy.rollout_io import write_rollout_episode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Collect a rollout episode from model-explorer contracts.")
    parser.add_argument("scenario", type=Path, help="Path to a contract or scenario JSON file.")
    parser.add_argument("output", type=Path, help="Path to write rollout episode JSON.")
    parser.add_argument("--max-candidates", type=int, default=None)
    args = parser.parse_args(argv)

    scenario = load_scenario(args.scenario)
    episode = collect_rollout_episode(scenario, max_candidates=args.max_candidates)
    write_rollout_episode(args.output, episode)
    print(
        json.dumps(
            {
                "transition_count": len(episode.transitions),
                "output": str(args.output),
                "metrics": episode.metrics.to_dict(),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
