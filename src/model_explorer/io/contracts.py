from __future__ import annotations

from pathlib import Path

from .scenario import Scenario, load_scenario


def contract_regression_paths(directory: str | Path) -> tuple[Path, ...]:
    root = Path(directory)
    if not root.exists():
        return ()
    if not root.is_dir():
        raise ValueError(f"contract regression path is not a directory: {root}")
    return tuple(sorted(path for path in root.glob("*.json") if path.is_file()))


def load_contract_regression_scenarios(directory: str | Path) -> tuple[Scenario, ...]:
    return tuple(load_scenario(path) for path in contract_regression_paths(directory))
