import ast
import subprocess
import sys
import textwrap
import typing
from pathlib import Path


PROJECT_ROOT = Path(__file__).parents[2]
SRC_ROOT = PROJECT_ROOT / "src"

TEMPORARY_EXPORT_NAMES = {
    "Any",
    "Path",
    "json",
    "dataclass",
    "Scenario",
    "RolloutEpisode",
    "evaluate_policy_baselines",
    "collect_rollout_episode",
    "_runner",
    "_impl",
    "_name",
}

LEGACY_PRIVATE_HELPERS = {
    "_run_training",
    "_select_best_training_run",
    "_baseline_deltas",
    "_calibration_recommendation",
    "_training_distillation_matrix",
    "_distillation_stability_summary",
}


def test_runner_keeps_only_orchestration_surface() -> None:
    runner_path = SRC_ROOT / "model_explorer" / "experiments" / "runner.py"

    assert len(runner_path.read_text(encoding="utf-8").splitlines()) <= 800
    assert "import *" not in runner_path.read_text(encoding="utf-8")


def test_legacy_experiment_import_paths_remain_available() -> None:
    from model_explorer.experiments.experiment_impl import (
        ExperimentManifest,
        _markdown_report,
        _run_training,
        run_experiment_manifest,
    )
    from model_explorer.policy.experiment import (
        dry_run_experiment_manifest,
        load_experiment_manifest,
        validate_experiment_manifest,
    )

    assert ExperimentManifest.__name__ == "ExperimentManifest"
    assert callable(_markdown_report)
    assert callable(_run_training)
    assert callable(run_experiment_manifest)
    assert callable(dry_run_experiment_manifest)
    assert callable(load_experiment_manifest)
    assert callable(validate_experiment_manifest)


def test_experiment_export_lists_are_explicit_and_do_not_leak_temporary_names() -> None:
    from model_explorer.experiments import experiment_impl, runner
    from model_explorer.policy import experiment

    for module in (runner, experiment_impl, experiment):
        assert isinstance(module.__all__, tuple)
        assert not (set(module.__all__) & TEMPORARY_EXPORT_NAMES)
        for name in LEGACY_PRIVATE_HELPERS:
            assert name in module.__all__
            assert callable(getattr(module, name))
        assert "run_experiment_manifest" in module.__all__
        assert callable(module.run_experiment_manifest)


def test_split_experiment_modules_do_not_import_runner() -> None:
    modules = (
        SRC_ROOT / "model_explorer" / "experiments" / "environment.py",
        SRC_ROOT / "model_explorer" / "experiments" / "evaluation.py",
        SRC_ROOT / "model_explorer" / "experiments" / "manifest.py",
        SRC_ROOT / "model_explorer" / "experiments" / "reports.py",
        SRC_ROOT / "model_explorer" / "experiments" / "selection.py",
        SRC_ROOT / "model_explorer" / "experiments" / "training_matrix.py",
    )

    for path in modules:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert node.module not in {"runner", "model_explorer.experiments.runner"}
            if isinstance(node, ast.Import):
                assert all(alias.name != "model_explorer.experiments.runner" for alias in node.names)


def test_importing_experiment_runner_does_not_load_torch() -> None:
    code = textwrap.dedent(
        f"""
        import sys
        sys.path.insert(0, {str(SRC_ROOT)!r})
        import model_explorer.experiments.runner
        raise SystemExit(1 if "torch" in sys.modules else 0)
        """
    )

    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_moved_manifest_type_hints_resolve() -> None:
    from model_explorer.experiments import evaluation, training_matrix

    for function in (
        evaluation._load_split_scenarios,
        evaluation._grouped_evaluation,
        training_matrix._run_training,
    ):
        typing.get_type_hints(function)
