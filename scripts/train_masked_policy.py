from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from model_explorer.policy.rollout_io import read_rollout_episode
from model_explorer.policy.training import train_policy_on_episode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a minimal masked PPO training step.")
    parser.add_argument("rollout", type=Path, help="Path to rollout episode JSON.")
    parser.add_argument("checkpoint", type=Path, help="Path to write torch policy checkpoint.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args(argv)

    episode = read_rollout_episode(args.rollout)
    result = train_policy_on_episode(
        episode,
        checkpoint_path=args.checkpoint,
        seed=args.seed,
        hidden_size=args.hidden_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
