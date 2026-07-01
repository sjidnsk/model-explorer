from __future__ import annotations

import subprocess
import sys
from importlib import util as importlib_util
from pathlib import Path
from typing import Any


def _environment_metadata(*, base_dir: Path) -> dict[str, Any]:
    torch_available = importlib_util.find_spec("torch") is not None
    torch_version = None
    if torch_available:
        try:
            import torch

            torch_version = str(torch.__version__)
        except Exception:
            torch_available = False
            torch_version = None
    return {
        "python_version": sys.version,
        "torch": {"available": torch_available, "version": torch_version},
        "git": _git_metadata(base_dir=base_dir),
    }


def _git_metadata(*, base_dir: Path) -> dict[str, Any]:
    commit = _git_output(base_dir, "rev-parse", "HEAD")
    status = _git_output(base_dir, "status", "--porcelain")
    return {
        "commit": commit,
        "dirty": bool(status),
    }


def _git_output(base_dir: Path, *args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=base_dir,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
    except Exception:
        return None
    return completed.stdout.strip()


# Public aliases
environment_metadata = _environment_metadata
git_metadata = _git_metadata

__all__ = [name for name in globals() if not name.startswith("__")]
