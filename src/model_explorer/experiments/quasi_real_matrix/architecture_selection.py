from __future__ import annotations

from math import isfinite
from typing import Any

from .decision_diagnostics import (
    _apply_decision_signal_guards,
    _decision_diagnostics_summary,
    _sample_discriminativeness_summary,
)
from .manifest import EVALUATION_SCOPE, QuasiRealEvaluationManifest, _normalize_selection_config
from .metrics import (
    _append_metric,
    _architecture_nested_metric_summary,
    _evaluation_metric,
    _evaluation_policy_metrics,
    _metric_value,
    _numeric_summary,
    _per_group_action_outcomes,
    _run_selection_metric,
    _training_runs,
)
from .quality_gates import _mask_stress_coverage, _selection_quality_gates
from .stability import _architecture_run_count, _manifest_architectures, _manifest_seeds


def _architecture_selection_summary(
    manifest: QuasiRealEvaluationManifest,
    experiment: Any,
    *,
    stability_summary: dict[str, Any],
) -> dict[str, Any]:
    config = _normalize_selection_config(manifest.selection_config)
    metric = str(config["metric"])
    mode = str(config["mode"])
    composite_weights = dict(config["composite_weights"])
    runs = _training_runs(experiment)
    architectures = _manifest_architectures(manifest)
    seeds = _manifest_seeds(manifest)
    dataset_summary = experiment.get("dataset_summary", {}) if isinstance(experiment, dict) else {}
    per_architecture_values: dict[str, list[float]] = {architecture: [] for architecture in architectures}
    loss_values: dict[str, list[float]] = {architecture: [] for architecture in architectures}
    composite_values: dict[str, list[float]] = {architecture: [] for architecture in architectures}
    composite_component_values: dict[str, dict[str, list[float]]] = {
        architecture: {metric_name: [] for metric_name in composite_weights}
        for architecture in architectures
    }
    exception_counts: dict[str, int] = {architecture: 0 for architecture in architectures}

    for run in runs:
        architecture = str(run.get("architecture", "unknown")) if isinstance(run, dict) else "unknown"
        if architecture not in per_architecture_values:
            per_architecture_values[architecture] = []
            loss_values[architecture] = []
            composite_values[architecture] = []
            composite_component_values[architecture] = {
                metric_name: [] for metric_name in composite_weights
            }
            exception_counts[architecture] = 0
        if not isinstance(run, dict):
            exception_counts[architecture] += 1
            continue
        if "error" in run or "exception" in run:
            exception_counts[architecture] += 1
        _append_metric({architecture: per_architecture_values[architecture]}, architecture, _run_selection_metric(run, metric))
        _append_metric({architecture: loss_values[architecture]}, architecture, run.get("loss"))
        validation_metrics = _evaluation_policy_metrics(run.get("validation_evaluation", {}), "torch_policy")
        _append_metric(
            {architecture: composite_values[architecture]},
            architecture,
            _selection_composite_score(validation_metrics, composite_weights),
        )
        for metric_name in composite_weights:
            _append_metric(
                composite_component_values[architecture],
                metric_name,
                _metric_value(validation_metrics, metric_name),
            )

    metric_stats = {
        architecture: _numeric_summary(tuple(values))
        for architecture, values in per_architecture_values.items()
    }
    loss_stats = {
        architecture: _numeric_summary(tuple(values))
        for architecture, values in loss_values.items()
    }
    composite_stats = {
        architecture: _numeric_summary(tuple(values))
        for architecture, values in composite_values.items()
    }
    composite_component_stats = {
        architecture: {
            metric_name: _numeric_summary(tuple(values))
            for metric_name, values in metrics.items()
        }
        for architecture, metrics in composite_component_values.items()
    }
    quality_gates = _selection_quality_gates(
        config,
        architectures=architectures,
        seeds=seeds,
        runs=runs,
        metric_stats=metric_stats,
        loss_stats=loss_stats,
        exception_counts=exception_counts,
        dataset_summary=dataset_summary,
    )
    architecture_stability = (
        stability_summary.get("architectures", {}) if isinstance(stability_summary, dict) else {}
    )
    architecture_details = {}
    for architecture in architectures:
        stability_metrics = architecture_stability.get(architecture, {}) if isinstance(architecture_stability, dict) else {}
        architecture_details[architecture] = {
            "run_count": _architecture_run_count(runs, architecture),
            "exception_count": exception_counts.get(architecture, 0),
            "failure_count": (
                stability_metrics.get("torch_policy.failure_count", _numeric_summary(()))
                if isinstance(stability_metrics, dict)
                else _numeric_summary(())
            ),
            "selection_metric": metric_stats.get(architecture, _numeric_summary(())),
            "selection_composite_score": composite_stats.get(architecture, _numeric_summary(())),
            "selection_composite_components": composite_component_stats.get(architecture, {}),
            "loss": loss_stats.get(architecture, _numeric_summary(())),
            "baseline_deltas": (
                stability_summary.get("baseline_deltas", {}).get(architecture, {})
                if isinstance(stability_summary.get("baseline_deltas", {}), dict)
                else {}
            ),
            "dataset": {
                key: stability_metrics.get(f"dataset.{key}", {})
                for key in (
                    "unreachable_candidate_count",
                    "padding_candidate_count",
                    "missing_experimental_feature_candidate_count",
                    "mask_stress_sample_count",
                )
            }
            if isinstance(stability_metrics, dict)
            else {},
        }

    decision_diagnostics = _decision_diagnostics_summary(runs, architectures=architectures)
    action_sensitive_summary = _architecture_nested_metric_summary(
        runs,
        architectures=architectures,
        section="action_sensitive_metrics",
    )
    oracle_regret_summary = _architecture_nested_metric_summary(
        runs,
        architectures=architectures,
        section="oracle_regret",
    )
    sample_discriminativeness = _sample_discriminativeness_summary(runs)
    per_group_action_outcomes = _per_group_action_outcomes(
        runs,
        architectures=architectures,
        uncertainty_multiplier=float(config["uncertainty_multiplier"]),
        decision_fn=_selection_decision,
    )
    composite_decision = _selection_decision(
        composite_stats,
        metric="selection_composite_score",
        mode="max",
        uncertainty_multiplier=float(config["uncertainty_multiplier"]),
    )
    base_summary: dict[str, Any] = {
        "enabled": bool(config["enabled"]),
        "metric": metric,
        "mode": mode,
        "decision_boundary": "recommended_architecture or inconclusive based on seed variance",
        "selection_composite_weights": composite_weights,
        "composite_selection": composite_decision,
        "decision_diagnostics": decision_diagnostics,
        "action_sensitive_summary": action_sensitive_summary,
        "oracle_regret_summary": oracle_regret_summary,
        "sample_discriminativeness": sample_discriminativeness,
        "per_group_action_outcomes": per_group_action_outcomes,
        "quality_gates": quality_gates,
        "architectures": architecture_details,
        "loss_distribution": stability_summary.get("loss_distribution", {}) if isinstance(stability_summary, dict) else {},
        "baseline_delta_distribution": (
            stability_summary.get("baseline_deltas", {}) if isinstance(stability_summary, dict) else {}
        ),
        "per_group_winners": _per_group_architecture_winners(
            runs,
            metric=metric,
            mode=mode,
            uncertainty_multiplier=float(config["uncertainty_multiplier"]),
        ),
        "mask_stress_coverage": _mask_stress_coverage(dataset_summary),
        "evaluation_scope": EVALUATION_SCOPE,
    }
    if not bool(config["enabled"]):
        base_summary.update(
            {
                "status": "not_configured",
                "decision": "inconclusive",
                "recommended_architecture": None,
                "reason": "selection.enabled is false",
            }
        )
        return base_summary
    if quality_gates["status"] != "passed":
        base_summary.update(
            {
                "status": "failed",
                "decision": "inconclusive",
                "recommended_architecture": None,
                "reason": "selection quality gates failed",
            }
        )
        return base_summary
    decision = _selection_decision(
        metric_stats,
        metric=metric,
        mode=mode,
        uncertainty_multiplier=float(config["uncertainty_multiplier"]),
    )
    decision = _apply_decision_signal_guards(
        decision,
        composite_decision=composite_decision,
        decision_diagnostics=decision_diagnostics,
    )
    base_summary.update(decision)
    base_summary["held_out_test_audit"] = _held_out_test_audit(
        runs,
        architectures=architectures,
        metric=metric,
        mode=mode,
        uncertainty_multiplier=float(config["uncertainty_multiplier"]),
        validation_decision=base_summary,
        composite_weights=composite_weights,
    )
    return base_summary


def _selection_composite_score(metrics: dict[str, Any], weights: dict[str, float]) -> float:
    score = 0.0
    for metric, weight in weights.items():
        score += float(weight) * _metric_value(metrics, metric)
    return score if isfinite(score) else 0.0


def _selection_decision(
    architecture_stats: dict[str, dict[str, Any]],
    *,
    metric: str,
    mode: str,
    uncertainty_multiplier: float,
) -> dict[str, Any]:
    candidates = [
        (architecture, stats)
        for architecture, stats in architecture_stats.items()
        if isinstance(stats, dict) and int(stats.get("count", 0)) > 0 and isfinite(float(stats.get("mean", 0.0)))
    ]
    if not candidates:
        return {
            "status": "inconclusive",
            "decision": "inconclusive",
            "recommended_architecture": None,
            "reason": f"no finite values for {metric}",
        }
    reverse = mode == "max"
    candidates.sort(key=lambda item: float(item[1].get("mean", 0.0)), reverse=reverse)
    best_architecture, best_stats = candidates[0]
    if len(candidates) == 1:
        return {
            "status": "selected",
            "decision": str(best_architecture),
            "recommended_architecture": str(best_architecture),
            "reason": f"only architecture with finite {metric}",
        }
    second_architecture, second_stats = candidates[1]
    best_mean = float(best_stats.get("mean", 0.0))
    second_mean = float(second_stats.get("mean", 0.0))
    margin = best_mean - second_mean if mode == "max" else second_mean - best_mean
    uncertainty = max(float(best_stats.get("std", 0.0)), float(second_stats.get("std", 0.0))) * uncertainty_multiplier
    if margin <= uncertainty:
        return {
            "status": "inconclusive",
            "decision": "inconclusive",
            "recommended_architecture": None,
            "reason": (
                f"best {metric} margin {margin} between {best_architecture} and "
                f"{second_architecture} is within seed variance {uncertainty}"
            ),
            "best_candidate": str(best_architecture),
            "runner_up": str(second_architecture),
            "margin": margin,
            "uncertainty": uncertainty,
        }
    return {
        "status": "selected",
        "decision": str(best_architecture),
        "recommended_architecture": str(best_architecture),
        "reason": (
            f"best {metric} margin {margin} over {second_architecture} exceeds "
            f"seed variance threshold {uncertainty}"
        ),
        "runner_up": str(second_architecture),
        "margin": margin,
        "uncertainty": uncertainty,
    }


def _per_group_architecture_winners(
    runs: list[dict[str, Any]],
    *,
    metric: str,
    mode: str,
    uncertainty_multiplier: float,
) -> dict[str, Any]:
    group_values: dict[str, dict[str, list[float]]] = {}
    for run in runs:
        architecture = str(run.get("architecture", "unknown"))
        evaluation = run.get("validation_evaluation", {})
        groups = evaluation.get("groups", {}) if isinstance(evaluation, dict) else {}
        if not isinstance(groups, dict):
            continue
        for group_name, group_evaluation in groups.items():
            value = _evaluation_metric(group_evaluation, metric)
            group_architectures = group_values.setdefault(str(group_name), {})
            _append_metric(group_architectures, architecture, value)
    winners: dict[str, Any] = {}
    for group_name, architecture_values in group_values.items():
        stats = {
            architecture: _numeric_summary(tuple(values))
            for architecture, values in architecture_values.items()
        }
        decision = _selection_decision(
            stats,
            metric=metric,
            mode=mode,
            uncertainty_multiplier=uncertainty_multiplier,
        )
        decision["architectures"] = stats
        winners[group_name] = decision
    return winners


def _held_out_test_audit(
    runs: list[dict[str, Any]],
    *,
    architectures: list[str],
    metric: str,
    mode: str,
    uncertainty_multiplier: float,
    validation_decision: dict[str, Any],
    composite_weights: dict[str, float],
) -> dict[str, Any]:
    metric_values: dict[str, list[float]] = {architecture: [] for architecture in architectures}
    composite_values: dict[str, list[float]] = {architecture: [] for architecture in architectures}
    for run in runs:
        if not isinstance(run, dict) or "test_evaluation" not in run:
            continue
        architecture = str(run.get("architecture", "unknown"))
        metric_values.setdefault(architecture, [])
        composite_values.setdefault(architecture, [])
        test_evaluation = run.get("test_evaluation", {})
        _append_metric(
            {architecture: metric_values[architecture]},
            architecture,
            _evaluation_metric(test_evaluation, metric),
        )
        test_policy_metrics = _evaluation_policy_metrics(test_evaluation, "torch_policy")
        _append_metric(
            {architecture: composite_values[architecture]},
            architecture,
            _selection_composite_score(test_policy_metrics, composite_weights),
        )
    metric_stats = {
        architecture: _numeric_summary(tuple(values))
        for architecture, values in metric_values.items()
    }
    composite_stats = {
        architecture: _numeric_summary(tuple(values))
        for architecture, values in composite_values.items()
    }
    if not any(stats.get("count", 0) for stats in metric_stats.values()):
        return {
            "status": "not_available",
            "used_for_selection": False,
            "split": "test",
            "reason": "no held-out test evaluation was recorded",
            "evaluation_scope": EVALUATION_SCOPE,
        }
    test_decision = _selection_decision(
        metric_stats,
        metric=metric,
        mode=mode,
        uncertainty_multiplier=uncertainty_multiplier,
    )
    test_composite_decision = _selection_decision(
        composite_stats,
        metric="selection_composite_score",
        mode="max",
        uncertainty_multiplier=uncertainty_multiplier,
    )
    validation_architecture = validation_decision.get("recommended_architecture")
    test_architecture = test_decision.get("recommended_architecture")
    stable = bool(validation_architecture and validation_architecture == test_architecture)
    return {
        "status": "available",
        "used_for_selection": False,
        "split": "test",
        "validation_decision": validation_decision.get("decision"),
        "validation_recommended_architecture": validation_architecture,
        "test_decision": test_decision,
        "test_composite_selection": test_composite_decision,
        "stable_with_validation": stable,
        "architectures": {
            architecture: {
                "test_metric": metric_stats.get(architecture, _numeric_summary(())),
                "test_composite_score": composite_stats.get(architecture, _numeric_summary(())),
            }
            for architecture in architectures
        },
        "reason": (
            "validation decision matches held-out test decision"
            if stable
            else "validation decision is inconclusive or differs from held-out test audit"
        ),
        "evaluation_scope": EVALUATION_SCOPE,
    }


architecture_selection_summary = _architecture_selection_summary
selection_composite_score = _selection_composite_score
selection_decision = _selection_decision
per_group_architecture_winners = _per_group_architecture_winners
held_out_test_audit = _held_out_test_audit

__all__ = [
    "architecture_selection_summary",
    "selection_composite_score",
    "selection_decision",
    "per_group_architecture_winners",
    "held_out_test_audit",
    "_architecture_selection_summary",
    "_selection_composite_score",
    "_selection_decision",
    "_per_group_architecture_winners",
    "_held_out_test_audit",
]
