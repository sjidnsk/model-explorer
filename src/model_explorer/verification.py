from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from .policy.experiment import run_experiment_manifest


def run_verification(
    root: str | Path,
    *,
    dry_run: bool = False,
    skip_benchmark_smoke: bool = False,
) -> dict[str, Any]:
    project_root = Path(root)
    steps = _planned_steps(project_root, skip_benchmark_smoke=skip_benchmark_smoke)
    if dry_run:
        return {"status": "dry_run", "steps": steps}

    results: list[dict[str, Any]] = []
    for step in steps:
        if step["name"] == "benchmark_smoke":
            results.append(_run_benchmark_smoke(project_root))
        else:
            results.append(_run_command_step(step, cwd=project_root))
    status = "passed" if all(result["returncode"] == 0 for result in results) else "failed"
    return {"status": status, "steps": results}


def _planned_steps(project_root: Path, *, skip_benchmark_smoke: bool) -> list[dict[str, Any]]:
    steps = [
        {
            "name": "unittest",
            "command": [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
        },
        {
            "name": "benchmark_smoke",
            "manifest": str(project_root / "tests" / "fixtures" / "synthetic_experiment" / "synthetic-benchmark-experiment.json"),
        },
        {"name": "git_diff_check", "command": ["git", "diff", "--check"]},
    ]
    if skip_benchmark_smoke:
        return [step for step in steps if step["name"] != "benchmark_smoke"]
    return steps


def _run_command_step(step: dict[str, Any], *, cwd: Path) -> dict[str, Any]:
    completed = subprocess.run(
        step["command"],
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return {
        "name": step["name"],
        "command": list(step["command"]),
        "returncode": completed.returncode,
        "stdout_tail": _tail(completed.stdout),
        "stderr_tail": _tail(completed.stderr),
    }


def _run_benchmark_smoke(project_root: Path) -> dict[str, Any]:
    source_manifest = project_root / "tests" / "fixtures" / "synthetic_experiment" / "synthetic-benchmark-experiment.json"
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "benchmark-smoke.json"
            payload = json.loads(source_manifest.read_text(encoding="utf-8"))
            fixture_root = source_manifest.parent
            payload["splits"]["benchmark"] = {
                group_name: [str((fixture_root / path).resolve()) for path in scenario_paths]
                for group_name, scenario_paths in payload["splits"]["benchmark"].items()
            }
            payload["outputs"] = {
                "rollouts": str(Path(tmpdir) / "benchmark-rollouts.jsonl"),
                "evaluation": str(Path(tmpdir) / "benchmark-evaluation.json"),
                "dataset_summary": str(Path(tmpdir) / "benchmark-dataset-summary.json"),
                "report": str(Path(tmpdir) / "benchmark-report.md"),
            }
            manifest_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            summary = run_experiment_manifest(manifest_path)
        return {
            "name": "benchmark_smoke",
            "returncode": 0,
            "scenario_count": summary.get("scenario_count", 0),
            "transition_count": summary.get("transition_count", 0),
        }
    except Exception as exc:
        return {
            "name": "benchmark_smoke",
            "returncode": 1,
            "error_type": type(exc).__name__,
            "message": str(exc),
        }


def _tail(text: str, *, max_lines: int = 20) -> str:
    lines = text.splitlines()
    return "\n".join(lines[-max_lines:])
