from __future__ import annotations

from math import isfinite
from typing import Any

from .manifest import EVALUATION_SCOPE, QuasiRealEvaluationManifest, _normalize_selection_config
from .metrics import (
    _append_metric,
    _architecture_nested_metric_summary,
    _cell_tuple,
    _evaluation_metric,
    _evaluation_policy_metrics,
    _group_name_from_path,
    _int_value,
    _iter_policy_nested_sections,
    _metric_value,
    _numeric_summary,
    _per_group_action_outcomes,
    _run_selection_metric,
    _training_runs,
)


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


def _sample_discriminativeness_summary(runs: list[dict[str, Any]]) -> dict[str, Any]:
    values: dict[str, list[float]] = {}
    for run in runs:
        if not isinstance(run, dict):
            continue
        for _, nested in _iter_policy_nested_sections(
            run,
            "sample_discriminativeness",
            evaluation_keys=("validation_evaluation", "test_evaluation"),
        ):
            for metric, value in nested.items():
                _append_metric(values, str(metric), value)

    metrics = {
        metric: _numeric_summary(tuple(metric_values))
        for metric, metric_values in values.items()
    }
    warnings: list[str] = []
    low_spread_metrics = {
        "candidate_coverage_spread": "low_candidate_coverage_spread",
        "risk_spread": "low_risk_spread",
        "path_cost_spread": "low_path_cost_spread",
        "value_spread": "low_value_spread",
    }
    for metric, warning in low_spread_metrics.items():
        stats = metrics.get(metric)
        if not isinstance(stats, dict) or int(stats.get("count", 0)) == 0:
            warnings.append(f"missing_{metric}")
        elif float(stats.get("mean", 0.0)) <= 1.0e-12:
            warnings.append(warning)
    disagreement = metrics.get("oracle_vs_heuristic_action_disagreement_rate")
    if isinstance(disagreement, dict) and int(disagreement.get("count", 0)) > 0:
        if float(disagreement.get("mean", 0.0)) <= 1.0e-12:
            warnings.append("low_oracle_vs_heuristic_action_disagreement_rate")
    elif not metrics:
        warnings.append("missing_sample_discriminativeness_metrics")

    return {
        "status": "warning" if warnings else "ok",
        "metrics": metrics,
        "warnings": warnings,
    }


def _decision_diagnostics_summary(
    runs: list[dict[str, Any]],
    *,
    architectures: list[str],
) -> dict[str, Any]:
    action_records: dict[str, dict[tuple[str, str, int], dict[str, Any]]] = {
        architecture: {} for architecture in architectures
    }
    baseline_counts: dict[str, dict[str, int]] = {
        architecture: {
            "sample_count": 0,
            "utility_agreement_count": 0,
            "coverage_heuristic_agreement_count": 0,
        }
        for architecture in architectures
    }
    mask_violation_count = 0

    for run in runs:
        if not isinstance(run, dict):
            continue
        architecture = str(run.get("architecture", "unknown"))
        if architecture not in action_records:
            action_records[architecture] = {}
            baseline_counts[architecture] = {
                "sample_count": 0,
                "utility_agreement_count": 0,
                "coverage_heuristic_agreement_count": 0,
            }
        seed = str(run.get("seed", ""))
        validation = run.get("validation_evaluation", {})
        per_scenario = validation.get("per_scenario", []) if isinstance(validation, dict) else []
        if not isinstance(per_scenario, list):
            continue
        for scenario in per_scenario:
            if not isinstance(scenario, dict):
                continue
            path = str(scenario.get("path", ""))
            group = str(scenario.get("group") or _group_name_from_path(path))
            metrics = scenario.get("metrics", {})
            torch_metrics = metrics.get("torch_policy", {}) if isinstance(metrics, dict) else {}
            diagnostics = torch_metrics.get("action_diagnostics", []) if isinstance(torch_metrics, dict) else []
            if not isinstance(diagnostics, list):
                continue
            for diagnostic in diagnostics:
                if not isinstance(diagnostic, dict):
                    continue
                step_index = _int_value(diagnostic.get("step_index"))
                key = (seed, path, step_index)
                selected_cell = _cell_tuple(diagnostic.get("selected_cell"))
                action_records[architecture][key] = {
                    "selected_cell": selected_cell,
                    "group": group,
                }
                counts = baseline_counts[architecture]
                counts["sample_count"] += 1
                if bool(diagnostic.get("agrees_with_utility", False)):
                    counts["utility_agreement_count"] += 1
                if bool(diagnostic.get("agrees_with_coverage_heuristic", False)):
                    counts["coverage_heuristic_agreement_count"] += 1
                try:
                    max_masked_probability = float(diagnostic.get("max_masked_action_probability", 0.0))
                except (TypeError, ValueError):
                    max_masked_probability = 0.0
                if max_masked_probability != 0.0 or not bool(diagnostic.get("selected_action_mask_valid", True)):
                    mask_violation_count += 1

    agreement_matrix = _architecture_agreement_matrix(action_records, architectures)
    baseline_agreement = _baseline_agreement_summary(baseline_counts, architectures)
    per_group_disagreement = _per_group_disagreement_summary(action_records, architectures)
    all_identical = _all_architectures_identical(agreement_matrix, architectures)
    warnings: list[str] = []
    if all_identical:
        warnings.append("all_architectures_identical")
    matching_coverage_architectures = [
        architecture
        for architecture, summary in baseline_agreement.items()
        if summary.get("sample_count", 0) > 0
        and float(summary.get("coverage_heuristic_agreement_rate", 0.0)) == 1.0
    ]
    for architecture in matching_coverage_architectures:
        warnings.append(f"trained_policy_matches_coverage_heuristic:{architecture}")
    if len(matching_coverage_architectures) == len(architectures) and architectures:
        warnings.append("all_trained_policies_match_coverage_heuristic")
    if mask_violation_count:
        warnings.append("masked_action_diagnostic_violation")

    return {
        "architecture_agreement_matrix": agreement_matrix,
        "architecture_baseline_agreement": baseline_agreement,
        "per_group_disagreement": per_group_disagreement,
        "all_architectures_identical": all_identical,
        "mask_violation_count": mask_violation_count,
        "sample_count": sum(int(summary.get("sample_count", 0)) for summary in baseline_agreement.values()),
        "warnings": warnings,
    }


def _architecture_agreement_matrix(
    action_records: dict[str, dict[tuple[str, str, int], dict[str, Any]]],
    architectures: list[str],
) -> dict[str, dict[str, dict[str, float | int]]]:
    matrix: dict[str, dict[str, dict[str, float | int]]] = {}
    for left in architectures:
        matrix[left] = {}
        left_records = action_records.get(left, {})
        for right in architectures:
            right_records = action_records.get(right, {})
            shared_keys = sorted(set(left_records).intersection(right_records))
            compared = len(shared_keys)
            agreement_count = (
                compared
                if left == right
                else sum(
                    1
                    for key in shared_keys
                    if left_records[key].get("selected_cell") == right_records[key].get("selected_cell")
                )
            )
            matrix[left][right] = {
                "compared": compared,
                "agreement_count": agreement_count,
                "agreement_rate": agreement_count / compared if compared else 0.0,
            }
    return matrix


def _baseline_agreement_summary(
    baseline_counts: dict[str, dict[str, int]],
    architectures: list[str],
) -> dict[str, dict[str, float | int | bool]]:
    summary: dict[str, dict[str, float | int | bool]] = {}
    for architecture in architectures:
        counts = baseline_counts.get(architecture, {})
        sample_count = int(counts.get("sample_count", 0))
        utility_count = int(counts.get("utility_agreement_count", 0))
        coverage_count = int(counts.get("coverage_heuristic_agreement_count", 0))
        summary[architecture] = {
            "sample_count": sample_count,
            "utility_agreement_count": utility_count,
            "utility_agreement_rate": utility_count / sample_count if sample_count else 0.0,
            "coverage_heuristic_agreement_count": coverage_count,
            "coverage_heuristic_agreement_rate": coverage_count / sample_count if sample_count else 0.0,
            "matches_coverage_heuristic": bool(sample_count and coverage_count == sample_count),
        }
    return summary


def _per_group_disagreement_summary(
    action_records: dict[str, dict[tuple[str, str, int], dict[str, Any]]],
    architectures: list[str],
) -> dict[str, dict[str, float | int]]:
    grouped_keys: dict[str, set[tuple[str, str, int]]] = {}
    for records in action_records.values():
        for key, record in records.items():
            group = str(record.get("group", "unknown"))
            grouped_keys.setdefault(group, set()).add(key)

    summary: dict[str, dict[str, float | int]] = {}
    for group, keys in grouped_keys.items():
        compared = 0
        disagreement_count = 0
        for key in keys:
            cells = [
                action_records.get(architecture, {}).get(key, {}).get("selected_cell")
                for architecture in architectures
                if key in action_records.get(architecture, {})
            ]
            if len(cells) < 2:
                continue
            compared += 1
            if len(set(cells)) > 1:
                disagreement_count += 1
        summary[group] = {
            "compared": compared,
            "disagreement_count": disagreement_count,
            "disagreement_rate": disagreement_count / compared if compared else 0.0,
        }
    return summary


def _all_architectures_identical(
    agreement_matrix: dict[str, dict[str, dict[str, float | int]]],
    architectures: list[str],
) -> bool:
    if len(architectures) < 2:
        return False
    for left in architectures:
        for right in architectures:
            if left == right:
                continue
            cell = agreement_matrix.get(left, {}).get(right, {})
            if int(cell.get("compared", 0)) <= 0:
                return False
            if float(cell.get("agreement_rate", 0.0)) != 1.0:
                return False
    return True


def _apply_decision_signal_guards(
    decision: dict[str, Any],
    *,
    composite_decision: dict[str, Any],
    decision_diagnostics: dict[str, Any],
) -> dict[str, Any]:
    guarded = dict(decision)
    if decision_diagnostics.get("all_architectures_identical"):
        guarded.update(
            {
                "status": "inconclusive",
                "decision": "inconclusive",
                "recommended_architecture": None,
                "reason": "all architectures selected identical actions; selection signal is insufficient",
            }
        )
    elif guarded.get("status") == "selected":
        composite_status = composite_decision.get("status")
        composite_architecture = composite_decision.get("recommended_architecture")
        recommended = guarded.get("recommended_architecture")
        if composite_status != "selected":
            guarded.update(
                {
                    "status": "inconclusive",
                    "decision": "inconclusive",
                    "recommended_architecture": None,
                    "reason": (
                        f"primary metric selected {recommended}, but selection_composite_score is "
                        f"{composite_status}: {composite_decision.get('reason', '')}"
                    ),
                }
            )
        elif composite_architecture != recommended:
            guarded.update(
                {
                    "status": "inconclusive",
                    "decision": "inconclusive",
                    "recommended_architecture": None,
                    "reason": (
                        f"primary metric selected {recommended}, but selection_composite_score selected "
                        f"{composite_architecture}"
                    ),
                }
            )
    warnings = decision_diagnostics.get("warnings", [])
    if isinstance(warnings, list) and warnings:
        guarded["decision_signal_warnings"] = list(warnings)
    return guarded


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


def _selection_quality_gates(
    config: dict[str, Any],
    *,
    architectures: list[str],
    seeds: list[int],
    runs: list[dict[str, Any]],
    metric_stats: dict[str, dict[str, Any]],
    loss_stats: dict[str, dict[str, Any]],
    exception_counts: dict[str, int],
    dataset_summary: Any,
) -> dict[str, Any]:
    violations: list[dict[str, Any]] = []
    dataset_summary = dataset_summary if isinstance(dataset_summary, dict) else {}
    _append_selection_min_violation(violations, "min_seed_count", len(seeds), config.get("min_seed_count"))
    _append_selection_min_violation(
        violations,
        "min_architecture_count",
        len(architectures),
        config.get("min_architecture_count"),
    )
    _append_selection_min_violation(
        violations,
        "min_roi_group_count",
        _int_value(dataset_summary.get("roi_count")),
        config.get("min_roi_group_count"),
    )
    _append_selection_min_violation(
        violations,
        "min_unreachable_candidate_count",
        _int_value(dataset_summary.get("unreachable_candidate_count")),
        config.get("min_unreachable_candidate_count"),
    )
    _append_selection_min_violation(
        violations,
        "min_mask_stress_sample_count",
        _int_value(dataset_summary.get("mask_stress_sample_count")),
        config.get("min_mask_stress_sample_count"),
    )
    expected_runs_per_architecture = len(seeds)
    for architecture in architectures:
        run_count = _architecture_run_count(runs, architecture)
        if run_count < expected_runs_per_architecture:
            violations.append(
                {
                    "gate": "architecture_run_count",
                    "architecture": architecture,
                    "expected": expected_runs_per_architecture,
                    "actual": run_count,
                    "message": (
                        f"architecture_run_count expected >= {expected_runs_per_architecture}, "
                        f"actual {run_count} for {architecture}"
                    ),
                }
            )
        if exception_counts.get(architecture, 0) > 0:
            violations.append(
                {
                    "gate": "exception_count",
                    "architecture": architecture,
                    "expected": 0,
                    "actual": exception_counts.get(architecture, 0),
                    "message": f"exception_count expected 0 for {architecture}",
                }
            )
        if loss_stats.get(architecture, {}).get("count", 0) < run_count:
            violations.append(
                {
                    "gate": "finite_loss",
                    "architecture": architecture,
                    "expected": run_count,
                    "actual": loss_stats.get(architecture, {}).get("count", 0),
                    "message": f"finite_loss expected {run_count} finite losses for {architecture}",
                }
            )
        if metric_stats.get(architecture, {}).get("count", 0) < run_count:
            violations.append(
                {
                    "gate": "finite_selection_metric",
                    "architecture": architecture,
                    "expected": run_count,
                    "actual": metric_stats.get(architecture, {}).get("count", 0),
                    "message": f"finite_selection_metric expected {run_count} finite metrics for {architecture}",
                }
            )
    return {
        "status": "failed" if violations else "passed",
        "configured": {key: value for key, value in config.items() if key.startswith("min_") and value is not None},
        "violations": violations,
    }


def _append_selection_min_violation(
    violations: list[dict[str, Any]],
    gate: str,
    actual: int | float,
    expected: int | float | None,
) -> None:
    if expected is None:
        return
    if actual < expected:
        violations.append(
            {
                "gate": gate,
                "expected": expected,
                "actual": actual,
                "message": f"{gate} expected >= {expected}, actual {actual}",
            }
        )


def _mask_stress_coverage(dataset_summary: Any) -> dict[str, Any]:
    dataset_summary = dataset_summary if isinstance(dataset_summary, dict) else {}
    return {
        "unreachable_candidate_count": _int_value(dataset_summary.get("unreachable_candidate_count")),
        "padding_candidate_count": _int_value(dataset_summary.get("padding_candidate_count")),
        "missing_experimental_feature_candidate_count": _int_value(
            dataset_summary.get("missing_experimental_feature_candidate_count")
        ),
        "mask_stress_sample_count": _int_value(dataset_summary.get("mask_stress_sample_count")),
        "mask_stress_augmented": bool(dataset_summary.get("mask_stress_augmented", False)),
    }


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



architecture_selection_summary = _architecture_selection_summary
selection_composite_score = _selection_composite_score
sample_discriminativeness_summary = _sample_discriminativeness_summary
decision_diagnostics_summary = _decision_diagnostics_summary
architecture_agreement_matrix = _architecture_agreement_matrix
baseline_agreement_summary = _baseline_agreement_summary
per_group_disagreement_summary = _per_group_disagreement_summary
all_architectures_identical = _all_architectures_identical
apply_decision_signal_guards = _apply_decision_signal_guards
held_out_test_audit = _held_out_test_audit
manifest_architectures = _manifest_architectures
manifest_seeds = _manifest_seeds
architecture_run_count = _architecture_run_count
selection_quality_gates = _selection_quality_gates
append_selection_min_violation = _append_selection_min_violation
mask_stress_coverage = _mask_stress_coverage
selection_decision = _selection_decision
per_group_architecture_winners = _per_group_architecture_winners
stability_summary = _stability_summary

_PUBLIC_EXPORTS = [
    "architecture_selection_summary",
    "selection_composite_score",
    "sample_discriminativeness_summary",
    "decision_diagnostics_summary",
    "architecture_agreement_matrix",
    "baseline_agreement_summary",
    "per_group_disagreement_summary",
    "all_architectures_identical",
    "apply_decision_signal_guards",
    "held_out_test_audit",
    "manifest_architectures",
    "manifest_seeds",
    "architecture_run_count",
    "selection_quality_gates",
    "append_selection_min_violation",
    "mask_stress_coverage",
    "selection_decision",
    "per_group_architecture_winners",
    "stability_summary",
]
_PRIVATE_COMPAT_EXPORTS = [
    "_architecture_selection_summary",
    "_selection_composite_score",
    "_sample_discriminativeness_summary",
    "_decision_diagnostics_summary",
    "_architecture_agreement_matrix",
    "_baseline_agreement_summary",
    "_per_group_disagreement_summary",
    "_all_architectures_identical",
    "_apply_decision_signal_guards",
    "_held_out_test_audit",
    "_manifest_architectures",
    "_manifest_seeds",
    "_architecture_run_count",
    "_selection_quality_gates",
    "_append_selection_min_violation",
    "_mask_stress_coverage",
    "_selection_decision",
    "_per_group_architecture_winners",
    "_stability_summary",
]
__all__ = [*_PUBLIC_EXPORTS, *_PRIVATE_COMPAT_EXPORTS]
