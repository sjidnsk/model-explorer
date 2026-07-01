from __future__ import annotations

from typing import Any

from .manifest import QuasiRealEvaluationManifest
from .metrics import _append_metric, _numeric_summary


def _stability_summary(experiment: Any) -> dict[str, Any]:
    if not isinstance(experiment, dict):
        return {"architectures": {}, "loss_distribution": {}, "baseline_deltas": {}}
    training = experiment.get("training", {})
    runs = training.get("runs", []) if isinstance(training, dict) else []
    if not isinstance(runs, list):
        runs = []

    architecture_values: dict[str, dict[str, list[float]]] = {}
    architecture_run_counts: dict[str, int] = {}
    baseline_values: dict[str, dict[str, list[float]]] = {}
    loss_values: dict[str, list[float]] = {}
    for run in runs:
        if not isinstance(run, dict):
            continue
        architecture = str(run.get("architecture", "unknown"))
        architecture_run_counts[architecture] = architecture_run_counts.get(architecture, 0) + 1
        arch_metrics = architecture_values.setdefault(architecture, {})
        for metric in ("loss", "policy_loss", "value_loss", "entropy"):
            _append_metric(arch_metrics, metric, run.get(metric))
            _append_metric(loss_values, metric, run.get(metric))
        run_dataset = run.get("dataset_summary", {})
        if isinstance(run_dataset, dict):
            for metric in (
                "unreachable_candidate_count",
                "padding_candidate_count",
                "missing_experimental_feature_candidate_count",
                "mask_stress_sample_count",
            ):
                _append_metric(arch_metrics, f"dataset.{metric}", run_dataset.get(metric))
        validation = run.get("validation_evaluation", {})
        if isinstance(validation, dict) and "aggregate" in validation and isinstance(validation["aggregate"], dict):
            validation = validation["aggregate"]
        torch_policy = validation.get("torch_policy", {}) if isinstance(validation, dict) else {}
        if isinstance(torch_policy, dict):
            for metric in ("final_coverage_rate", "total_path_cost", "average_risk", "failure_count"):
                _append_metric(arch_metrics, f"torch_policy.{metric}", torch_policy.get(metric))
        deltas = run.get("baseline_deltas", {})
        arch_deltas = baseline_values.setdefault(architecture, {})
        if isinstance(deltas, dict):
            for baseline_name, metrics in deltas.items():
                if not isinstance(metrics, dict):
                    continue
                for metric, value in metrics.items():
                    _append_metric(arch_deltas, f"{baseline_name}.{metric}", value)

    return {
        "architectures": {
            architecture: {
                "run_count": architecture_run_counts.get(architecture, 0),
                **{metric: _numeric_summary(tuple(values)) for metric, values in metrics.items()},
            }
            for architecture, metrics in architecture_values.items()
        },
        "loss_distribution": {
            metric: _numeric_summary(tuple(values))
            for metric, values in loss_values.items()
        },
        "baseline_deltas": {
            architecture: {
                metric: _numeric_summary(tuple(values))
                for metric, values in metrics.items()
            }
            for architecture, metrics in baseline_values.items()
        },
    }


def _manifest_architectures(manifest: QuasiRealEvaluationManifest) -> list[str]:
    raw = manifest.train_config.get("architectures")
    if isinstance(raw, list) and raw:
        return [str(architecture) for architecture in raw]
    architecture = manifest.train_config.get("architecture", "mlp_v1")
    return ["mlp_v1" if architecture is None or str(architecture).strip() == "" else str(architecture)]


def _manifest_seeds(manifest: QuasiRealEvaluationManifest) -> list[int]:
    raw = manifest.train_config.get("seeds")
    if isinstance(raw, list) and raw:
        return [int(seed) for seed in raw]
    return [int(manifest.train_config.get("seed", 0))]


def _architecture_run_count(runs: list[dict[str, Any]], architecture: str) -> int:
    return sum(1 for run in runs if str(run.get("architecture", "unknown")) == architecture)


stability_summary = _stability_summary
manifest_architectures = _manifest_architectures
manifest_seeds = _manifest_seeds
architecture_run_count = _architecture_run_count

__all__ = [
    "stability_summary",
    "manifest_architectures",
    "manifest_seeds",
    "architecture_run_count",
    "_stability_summary",
    "_manifest_architectures",
    "_manifest_seeds",
    "_architecture_run_count",
]
