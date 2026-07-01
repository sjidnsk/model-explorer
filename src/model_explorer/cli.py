from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_COMMANDS = {
    "benchmark",
    "collect-rollout",
    "evaluate-baselines",
    "experiment",
    "path-feedback",
    "quasi-real",
    "train",
    "verify",
}
_BENCHMARK_GROUPS = (
    "coverage_dominant",
    "risk_dominant",
    "path_cost_dominant",
    "sparse_candidates",
    "high_unreachable_rate",
    "missing_experimental_fields",
)


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
    if args_list and args_list[0] not in _COMMANDS | {"-h", "--help"}:
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
    generate_parser.add_argument("--group", choices=_BENCHMARK_GROUPS, action="append", default=None)
    generate_parser.add_argument("--difficulty", choices=("easy", "medium", "hard"), default="medium")

    collect_parser = subparsers.add_parser("collect-rollout", help="Collect rollout episodes from scenarios.")
    collect_parser.add_argument(
        "paths",
        type=Path,
        nargs="+",
        help="One or more contract/scenario JSON files followed by the output path.",
    )
    collect_parser.add_argument("--max-candidates", type=int, default=None)
    collect_parser.add_argument("--jsonl", action="store_true", help="Write JSONL even when there is one input scenario.")

    baselines_parser = subparsers.add_parser("evaluate-baselines", help="Evaluate policy baselines.")
    baselines_parser.add_argument("scenario", type=Path, nargs="+", help="Path to one or more contract/scenario JSON files.")
    baselines_parser.add_argument("--checkpoint", type=Path, default=None, help="Optional torch policy checkpoint.")

    train_parser = subparsers.add_parser("train", help="Run a minimal masked PPO training step.")
    train_parser.add_argument("rollout", type=Path, help="Path to rollout JSON, JSONL, or directory.")
    train_parser.add_argument("checkpoint", type=Path, help="Path to write torch policy checkpoint.")
    train_parser.add_argument("--seed", type=int, default=0)
    train_parser.add_argument("--hidden-size", type=int, default=64)
    train_parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    train_parser.add_argument("--epochs", type=int, default=1)

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
        return _run_experiment_command(args)
    if args.command == "benchmark" and args.benchmark_command == "generate":
        return _run_benchmark_generate(args)
    if args.command == "collect-rollout":
        return _run_collect_rollout(args)
    if args.command == "evaluate-baselines":
        return _run_evaluate_baselines(args)
    if args.command == "train":
        return _run_train(args)
    if args.command == "path-feedback":
        return _run_path_feedback_command(args)
    if args.command == "quasi-real":
        return _run_quasi_real_command(args)
    if args.command == "verify":
        return _run_verify_command(args)
    return 1


def _run_experiment_command(args: argparse.Namespace) -> int:
    from .experiments.runner import (
        dry_run_experiment_manifest,
        run_experiment_manifest,
        validate_experiment_manifest,
    )

    if args.experiment_command == "run":
        _print_json(run_experiment_manifest(args.manifest))
    elif args.experiment_command == "validate":
        _print_json(validate_experiment_manifest(args.manifest))
    elif args.experiment_command == "dry-run":
        _print_json(dry_run_experiment_manifest(args.manifest))
    return 0


def _run_benchmark_generate(args: argparse.Namespace) -> int:
    from .policy.benchmark import generate_synthetic_benchmark_suite

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


def _run_collect_rollout(args: argparse.Namespace) -> int:
    from .io.scenario import load_scenario
    from .policy.collector import collect_rollout_episode
    from .policy.rollout_io import write_rollout_episode, write_rollout_episodes_jsonl

    if len(args.paths) < 2:
        raise ValueError("provide at least one scenario path and one output path")

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
    _print_json(
        {
            "episode_count": len(episodes),
            "transition_count": sum(len(episode.transitions) for episode in episodes),
            "output": str(output_path),
            "metrics": episodes[0].metrics.to_dict() if len(episodes) == 1 else None,
        }
    )
    return 0


def _run_evaluate_baselines(args: argparse.Namespace) -> int:
    from .io.scenario import load_scenario
    from .policy.evaluation import evaluate_policy_baseline_scenarios, evaluate_policy_baselines

    scenarios = [load_scenario(path) for path in args.scenario]
    policy = None
    if args.checkpoint is not None:
        from .policy.training import load_policy_checkpoint

        policy = load_policy_checkpoint(args.checkpoint)
    report = (
        evaluate_policy_baselines(scenarios[0], torch_policy=policy)
        if len(scenarios) == 1
        else evaluate_policy_baseline_scenarios(scenarios, torch_policy=policy)
    )
    _print_json(report)
    return 0


def _run_train(args: argparse.Namespace) -> int:
    from .policy.rollout_io import read_rollout_episodes
    from .policy.training import train_policy_on_episodes

    episodes = read_rollout_episodes(args.rollout)
    result = train_policy_on_episodes(
        episodes,
        checkpoint_path=args.checkpoint,
        seed=args.seed,
        hidden_size=args.hidden_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
    )
    _print_json(result)
    return 0


def _run_path_feedback_command(args: argparse.Namespace) -> int:
    if args.path_feedback_command == "run":
        from .policy.path_feedback_manifest import load_path_feedback_manifest
        from .policy.path_feedback_reports import render_path_feedback_markdown
        from .policy.path_feedback_runner import run_path_feedback
        from .policy.path_feedback_summary import compact_path_feedback_summary

        manifest = load_path_feedback_manifest(args.manifest)
        summary = run_path_feedback(manifest)
        if manifest.summary_output is not None:
            manifest.summary_output.parent.mkdir(parents=True, exist_ok=True)
            manifest.summary_output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        if manifest.report_output is not None:
            manifest.report_output.parent.mkdir(parents=True, exist_ok=True)
            manifest.report_output.write_text(render_path_feedback_markdown(summary), encoding="utf-8")
        _print_json(
            compact_path_feedback_summary(
                summary,
                summary_output=manifest.summary_output,
                report_output=manifest.report_output,
            )
        )
    elif args.path_feedback_command == "validate":
        from .policy.path_feedback_manifest import validate_path_feedback_manifest

        _print_json(validate_path_feedback_manifest(args.manifest))
    elif args.path_feedback_command == "dry-run":
        from .policy.path_feedback_runner import dry_run_path_feedback_manifest

        _print_json(dry_run_path_feedback_manifest(args.manifest))
    return 0


def _run_quasi_real_command(args: argparse.Namespace) -> int:
    from .experiments.quasi_real_matrix.runner import (
        dry_run_quasi_real_evaluation_manifest,
        run_quasi_real_evaluation_manifest,
        validate_quasi_real_evaluation_manifest,
    )

    if args.quasi_real_command == "run":
        _print_json(run_quasi_real_evaluation_manifest(args.manifest))
    elif args.quasi_real_command == "validate":
        _print_json(validate_quasi_real_evaluation_manifest(args.manifest))
    elif args.quasi_real_command == "dry-run":
        _print_json(dry_run_quasi_real_evaluation_manifest(args.manifest))
    return 0


def _run_verify_command(args: argparse.Namespace) -> int:
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


def run_verification(*args: Any, **kwargs: Any) -> dict[str, Any]:
    from .verification import run_verification as _run_verification

    return _run_verification(*args, **kwargs)


def _run_legacy_scenario(args_list: list[str]) -> int:
    from .io.scenario import load_scenario
    from .orchestration.loop import run_exploration_loop

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
