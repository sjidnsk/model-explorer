from pathlib import Path


def test_runner_keeps_only_orchestration_surface() -> None:
    runner_path = (
        Path(__file__).parents[2]
        / "src"
        / "model_explorer"
        / "experiments"
        / "runner.py"
    )

    assert len(runner_path.read_text(encoding="utf-8").splitlines()) <= 800


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
