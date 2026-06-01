from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .core.interfaces import ModelExplorerContract
from .io.scenario import load_scenario
from .orchestration.loop import run_exploration_loop
from .policy.benchmark import BENCHMARK_GROUPS, generate_synthetic_benchmark_suite
from .policy.experiment import (
    dry_run_experiment_manifest,
    run_experiment_manifest,
    validate_experiment_manifest,
)
from .policy.path_feedback import (
    dry_run_path_feedback_manifest,
    run_path_feedback_manifest,
    validate_path_feedback_manifest,
)
from .data.evaluation_matrix import (
    dry_run_quasi_real_evaluation_manifest,
    run_quasi_real_evaluation_manifest,
    validate_quasi_real_evaluation_manifest,
)
from .verification import run_verification


def main(argv: list[str] | None = None) -> int:
    args_list = list(sys.argv[1:] if argv is None else argv)
    try:
        return _main(args_list)
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                },
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 1


def _main(args_list: list[str]) -> int:
    if args_list and args_list[0] not in {"experiment", "benchmark", "path-feedback", "quasi-real", "verify", "-h", "--help"}:
        return _run_legacy_scenario(args_list)

    parser = argparse.ArgumentParser(description="model-explorer benchmark and orchestration entrypoint.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    experiment_parser = subparsers.add_parser("experiment", help="Validate, dry-run, or run experiment manifests.")
    experiment_subparsers = experiment_parser.add_subparsers(dest="experiment_command", required=True)
    for command in ("run", "validate", "dry-run"):
        command_parser = experiment_subparsers.add_parser(command)
        command_parser.add_argument("manifest", type=Path, help="Path to experiment manifest JSON.")

    benchmark_parser = subparsers.add_parser("benchmark", help="Synthetic benchmark utilities.")
    benchmark_subparsers = benchmark_parser.add_subparsers(dest="benchmark_command", required=True)
    generate_parser = benchmark_subparsers.add_parser("generate")
    generate_parser.add_argument("output_dir", type=Path, help="Directory to receive generated manifest and scenarios.")
    generate_parser.add_argument("--seed", type=int, default=0)
    generate_parser.add_argument("--scenario-count", type=int, default=1)
    generate_parser.add_argument("--group", choices=BENCHMARK_GROUPS, action="append", default=None)
    generate_parser.add_argument("--difficulty", choices=("easy", "medium", "hard"), default="medium")

    path_feedback_parser = subparsers.add_parser(
        "path-feedback",
        help="Run semi-real path feedback summaries from contract + sidecar manifests.",
    )
    path_feedback_subparsers = path_feedback_parser.add_subparsers(dest="path_feedback_command", required=True)
    for command in ("run", "validate", "dry-run"):
        command_parser = path_feedback_subparsers.add_parser(command)
        command_parser.add_argument("manifest", type=Path, help="Path to path-feedback manifest JSON.")

    quasi_real_parser = subparsers.add_parser("quasi-real", help="Quasi-real LOLA evaluation matrix utilities.")
    quasi_real_subparsers = quasi_real_parser.add_subparsers(dest="quasi_real_command", required=True)
    for command in ("run", "validate", "dry-run"):
        command_parser = quasi_real_subparsers.add_parser(command)
        command_parser.add_argument("manifest", type=Path, help="Path to quasi-real evaluation matrix JSON.")

    verify_parser = subparsers.add_parser("verify", help="Run the daily verification chain.")
    verify_parser.add_argument("--dry-run", action="store_true", help="Report verification steps without running them.")
    verify_parser.add_argument(
        "--skip-benchmark-smoke",
        action="store_true",
        help="Skip the fixture benchmark smoke run.",
    )
    verify_parser.add_argument("--json-output", type=Path, help="Write the verification JSON summary to this path.")

    args = parser.parse_args(args_list)
    if args.command == "experiment":
        if args.experiment_command == "run":
            _print_json(run_experiment_manifest(args.manifest))
        elif args.experiment_command == "validate":
            _print_json(validate_experiment_manifest(args.manifest))
        elif args.experiment_command == "dry-run":
            _print_json(dry_run_experiment_manifest(args.manifest))
        return 0
    if args.command == "benchmark" and args.benchmark_command == "generate":
        _print_json(
            generate_synthetic_benchmark_suite(
                args.output_dir,
                seed=args.seed,
                scenario_count=args.scenario_count,
                groups=args.group,
                difficulty=args.difficulty,
            )
        )
        return 0
    if args.command == "path-feedback":
        if args.path_feedback_command == "run":
            _print_json(run_path_feedback_manifest(args.manifest))
        elif args.path_feedback_command == "validate":
            _print_json(validate_path_feedback_manifest(args.manifest))
        elif args.path_feedback_command == "dry-run":
            _print_json(dry_run_path_feedback_manifest(args.manifest))
        return 0
    if args.command == "quasi-real":
        if args.quasi_real_command == "run":
            _print_json(run_quasi_real_evaluation_manifest(args.manifest))
        elif args.quasi_real_command == "validate":
            _print_json(validate_quasi_real_evaluation_manifest(args.manifest))
        elif args.quasi_real_command == "dry-run":
            _print_json(dry_run_quasi_real_evaluation_manifest(args.manifest))
        return 0
    if args.command == "verify":
        summary = run_verification(
            Path.cwd(),
            dry_run=args.dry_run,
            skip_benchmark_smoke=args.skip_benchmark_smoke,
        )
        if args.json_output is not None:
            args.json_output.parent.mkdir(parents=True, exist_ok=True)
            args.json_output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        _print_json(summary)
        return 0 if summary.get("status") in {"passed", "dry_run"} else 1
    return 1


def _run_legacy_scenario(args_list: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Run the model-explorer orchestration loop.")
    parser.add_argument("scenario", type=Path, help="Path to a model-explorer-contract/v1 JSON scenario.")
    args = parser.parse_args(args_list)

    scenario = load_scenario(args.scenario)
    results = run_exploration_loop(scenario)
    summaries = [
        _result_to_summary(result, contract)
        for result, contract in zip(results, scenario.snapshots, strict=True)
    ]
    _print_json(summaries)
    return 0


def _print_json(payload: Any) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False))


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
