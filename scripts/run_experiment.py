from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from model_explorer.policy.experiment import run_experiment_manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a model-explorer experiment manifest.")
    parser.add_argument("manifest", type=Path, help="Path to experiment manifest JSON.")
    args = parser.parse_args(argv)

    summary = run_experiment_manifest(args.manifest)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
