from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from .policy.experiment import run_experiment_manifest


FORBIDDEN_IMPORT_PATTERNS = ("a_gcs_ws", "dev-platform-constraints")
_FORBIDDEN_IMPORT_ALIASES = {
    "a_gcs_ws": ("a_gcs_ws",),
    "dev-platform-constraints": ("dev-platform-constraints", "dev_platform_constraints"),
}
_IMPORT_LINE_RE = re.compile(r"^\s*(?:from|import)\s+")
_FORBIDDEN_IMPORT_SCAN_DIRS = ("src", "tests", "scripts")


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
        elif step["name"] == "forbidden_import_check":
            results.append(_run_forbidden_import_check(project_root))
        else:
            results.append(_run_command_step(step, cwd=project_root))
    status = "passed" if all(result["returncode"] == 0 for result in results) else "failed"
    return {"status": status, "steps": results}


def _planned_steps(project_root: Path, *, skip_benchmark_smoke: bool) -> list[dict[str, Any]]:
    steps = [
        {
            "name": "unittest",
            "kind": "subprocess",
            "command": [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
        },
        {
            "name": "benchmark_smoke",
            "kind": "python",
            "manifest": str(project_root / "tests" / "fixtures" / "synthetic_experiment" / "synthetic-benchmark-experiment.json"),
        },
        {
            "name": "forbidden_import_check",
            "kind": "python_scan",
            "paths": [str(project_root / name) for name in _FORBIDDEN_IMPORT_SCAN_DIRS],
            "forbidden_patterns": list(FORBIDDEN_IMPORT_PATTERNS),
        },
        {"name": "git_diff_check", "kind": "subprocess", "command": ["git", "diff", "--check"]},
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
        "kind": step.get("kind", "subprocess"),
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
            "kind": "python",
            "returncode": 0,
            "scenario_count": summary.get("scenario_count", 0),
            "transition_count": summary.get("transition_count", 0),
        }
    except Exception as exc:
        return {
            "name": "benchmark_smoke",
            "kind": "python",
            "returncode": 1,
            "error_type": type(exc).__name__,
            "message": str(exc),
        }


def _run_forbidden_import_check(project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root)
    violations: list[dict[str, Any]] = []
    scanned_files = 0
    for directory_name in _FORBIDDEN_IMPORT_SCAN_DIRS:
        directory = root / directory_name
        if not directory.exists():
            continue
        for path in sorted(directory.rglob("*.py")):
            scanned_files += 1
            for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
                if not _IMPORT_LINE_RE.match(line):
                    continue
                for pattern, aliases in _FORBIDDEN_IMPORT_ALIASES.items():
                    if any(alias in line for alias in aliases):
                        violations.append(
                            {
                                "path": str(path.relative_to(root)),
                                "line": line_number,
                                "pattern": pattern,
                                "text": line.strip(),
                            }
                        )

    return {
        "name": "forbidden_import_check",
        "kind": "python_scan",
        "returncode": 1 if violations else 0,
        "scanned_files": scanned_files,
        "forbidden_patterns": list(FORBIDDEN_IMPORT_PATTERNS),
        "violations": violations,
    }


def _tail(text: str, *, max_lines: int = 20) -> str:
    lines = text.splitlines()
    return "\n".join(lines[-max_lines:])
