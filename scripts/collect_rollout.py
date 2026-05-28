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
from model_explorer.policy.rollout_io import write_rollout_episode, write_rollout_episodes_jsonl


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Collect a rollout episode from model-explorer contracts.")
    parser.add_argument(
        "paths",
        type=Path,
        nargs="+",
        help="One or more contract/scenario JSON files followed by the output path.",
    )
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--jsonl", action="store_true", help="Write JSONL even when there is one input scenario.")
    args = parser.parse_args(argv)
    if len(args.paths) < 2:
        parser.error("provide at least one scenario path and one output path")

    scenario_paths = args.paths[:-1]
    output_path = args.paths[-1]
    episodes = [
        collect_rollout_episode(load_scenario(path), max_candidates=args.max_candidates)
        for path in scenario_paths
    ]
    if len(episodes) == 1 and not args.jsonl and output_path.suffix.lower() != ".jsonl":
        write_rollout_episode(output_path, episodes[0])
    else:
        write_rollout_episodes_jsonl(output_path, episodes)
    print(
        json.dumps(
            {
                "episode_count": len(episodes),
                "transition_count": sum(len(episode.transitions) for episode in episodes),
                "output": str(output_path),
                "metrics": episodes[0].metrics.to_dict() if len(episodes) == 1 else None,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
