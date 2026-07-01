from __future__ import annotations

from typing import Any

from .metrics import (
    _append_metric,
    _cell_tuple,
    _group_name_from_path,
    _int_value,
    _iter_policy_nested_sections,
    _numeric_summary,
)


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


decision_diagnostics_summary = _decision_diagnostics_summary
architecture_agreement_matrix = _architecture_agreement_matrix
baseline_agreement_summary = _baseline_agreement_summary
per_group_disagreement_summary = _per_group_disagreement_summary
all_architectures_identical = _all_architectures_identical
apply_decision_signal_guards = _apply_decision_signal_guards
sample_discriminativeness_summary = _sample_discriminativeness_summary

__all__ = [
    "decision_diagnostics_summary",
    "architecture_agreement_matrix",
    "baseline_agreement_summary",
    "per_group_disagreement_summary",
    "all_architectures_identical",
    "apply_decision_signal_guards",
    "sample_discriminativeness_summary",
    "_decision_diagnostics_summary",
    "_architecture_agreement_matrix",
    "_baseline_agreement_summary",
    "_per_group_disagreement_summary",
    "_all_architectures_identical",
    "_apply_decision_signal_guards",
    "_sample_discriminativeness_summary",
]
