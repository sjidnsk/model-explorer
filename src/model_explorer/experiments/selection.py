from __future__ import annotations

from math import isfinite, sqrt
from typing import Any

from .evaluation import _comparison_from_evaluation


def _training_source_comparison(runs: list[dict[str, Any]]) -> dict[str, Any]:
    comparison: dict[str, Any] = {}
    for run in runs:
        source = run.get("training_data_selection_strategy")
        if source is None:
            source = run.get("training_source", {}).get("primary_selection_strategy")
        source_name = _normalize_training_source_name(None if source is None else str(source))
        existing = comparison.get(source_name)
        if existing is not None:
            continue
        torch_metrics = _comparison_from_evaluation(run.get("validation_evaluation", {})).get("torch_policy", {})
        torch_metrics = torch_metrics if isinstance(torch_metrics, dict) else {}
        comparison[source_name] = {
            "checkpoint": run.get("checkpoint"),
            "training_source": dict(run.get("training_source", {})),
            "teacher_imitation": dict(run.get("teacher_imitation", {})),
            "teacher_margin_summary": dict(run.get("teacher_margin_summary", {})),
            "teacher_agreement": _teacher_agreement_summary(torch_metrics),
            "baseline_deltas": dict(run.get("baseline_deltas", {})),
        }
    return comparison


def _training_distillation_matrix(
    runs: list[dict[str, Any]],
    *,
    selected_run: dict[str, Any] | None = None,
    policy: str = "torch_policy",
    metric: str = "final_coverage_rate",
) -> list[dict[str, Any]]:
    matrix: list[dict[str, Any]] = []
    for run in runs:
        torch_metrics = _comparison_from_evaluation(run.get("validation_evaluation", {})).get("torch_policy", {})
        torch_metrics = torch_metrics if isinstance(torch_metrics, dict) else {}
        dataset_summary = run.get("dataset_summary", {})
        dataset_summary = dataset_summary if isinstance(dataset_summary, dict) else {}
        source = run.get("training_data_selection_strategy")
        matrix.append(
            {
                "source_selection_strategy": _normalize_training_source_name(None if source is None else str(source)),
                "teacher_imitation_weight": float(run.get("teacher_imitation_weight", 0.0)),
                "architecture": run.get("architecture"),
                "seed": run.get("seed"),
                "checkpoint": run.get("checkpoint"),
                "training_source": dict(run.get("training_source", {})),
                "teacher_imitation": dict(run.get("teacher_imitation", {})),
                "teacher_margin_summary": dict(run.get("teacher_margin_summary", {})),
                "teacher_margin_curriculum_profile": _run_curriculum_profile_name(run),
                "teacher_curriculum": dict(run.get("teacher_curriculum", {})),
                "teacher_quality_gates": dict(run.get("teacher_quality_gates", {})),
                "teacher_agreement": _teacher_agreement_summary(torch_metrics),
                "confidence_calibration": dict(torch_metrics.get("feedback_aware_confidence_calibration", {})),
                "baseline_deltas": dict(run.get("baseline_deltas", {})),
                "data_class": dataset_summary.get("data_class"),
                "dataset_id": dataset_summary.get("dataset_id"),
                "mask_stress_augmented": bool(dataset_summary.get("mask_stress_augmented", False)),
                "evaluation_scope": _distillation_evaluation_scope(dataset_summary),
                "selection_decision": _distillation_run_selection_decision(
                    run,
                    selected_run=selected_run,
                    policy=policy,
                    metric=metric,
                    torch_metrics=torch_metrics,
                ),
            }
        )
    return matrix


def _distillation_evaluation_scope(dataset_summary: dict[str, Any]) -> str:
    if dataset_summary.get("data_class") == "quasi_real" or bool(dataset_summary.get("mask_stress_augmented", False)):
        return "calibration evidence; not real-world generalization benchmark"
    return "synthetic calibration evidence; not real-world generalization benchmark"


def _distillation_run_selection_decision(
    run: dict[str, Any],
    *,
    selected_run: dict[str, Any] | None,
    policy: str,
    metric: str,
    torch_metrics: dict[str, Any],
) -> dict[str, Any]:
    gate_status = _teacher_quality_gate_status(run)
    metric_value = _training_run_metric(run, policy=policy, metric=metric)
    source = _normalize_training_source_name(run.get("training_data_selection_strategy"))
    teacher_weight = float(run.get("teacher_imitation_weight", 0.0))
    baseline_deltas = run.get("baseline_deltas", {})
    feedback_delta = baseline_deltas.get("feedback_aware", {}) if isinstance(baseline_deltas, dict) else {}
    margin_bucket = torch_metrics.get("feedback_aware_margin_bucket_agreement", {})
    is_selected = selected_run is not None and run is selected_run
    reason_codes: list[str] = []
    status = "candidate"
    if _teacher_quality_gate_failed(run) and not is_selected:
        status = "excluded"
        reason_codes.append("teacher_quality_gate_failed")
    elif is_selected:
        status = "selected"
        reason_codes.append("selected_best_run")
        if _teacher_quality_gate_failed(run):
            reason_codes.append("gate_failed_selection")
    else:
        reason_codes.append("eligible_not_selected")
    return {
        "status": status,
        "reason_codes": reason_codes,
        "policy": str(policy),
        "metric": str(metric),
        "metric_value": metric_value,
        "source_selection_strategy": source,
        "teacher_imitation_weight": teacher_weight,
        "teacher_margin_curriculum_profile": _run_curriculum_profile_name(run),
        "gate_status": gate_status,
        "baseline_delta": dict(feedback_delta) if isinstance(feedback_delta, dict) else {},
        "margin_bucket_performance": dict(margin_bucket) if isinstance(margin_bucket, dict) else {},
        "reason": (
            f"{status}: source={source}; teacher_imitation_weight={teacher_weight}; "
            f"gate_status={gate_status}; {policy}.{metric}={metric_value}"
        ),
    }


def _teacher_agreement_summary(torch_metrics: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "feedback_aware_action_agreement_rate",
        "feedback_aware_selected_cell_agreement_rate",
        "feedback_aware_top2_action_agreement_rate",
        "feedback_aware_topk_action_agreement_rate",
        "feedback_aware_teacher_rank_mean",
        "feedback_aware_margin_bucket_agreement",
    )
    return {field: torch_metrics[field] for field in fields if field in torch_metrics}


def _distillation_stability_summary(runs: list[dict[str, Any]]) -> dict[str, Any]:
    profile_dimension = _distillation_profile_dimension_enabled(runs)
    grouped: dict[str, dict[str, dict[str, list[dict[str, Any]]] | list[dict[str, Any]]]] = {}
    for run in runs:
        source = _normalize_training_source_name(run.get("training_data_selection_strategy"))
        teacher_weight = f"{float(run.get('teacher_imitation_weight', 0.0)):g}"
        if profile_dimension:
            profile = _run_curriculum_profile_name(run)
            source_group = grouped.setdefault(source, {})
            weight_group = source_group.setdefault(teacher_weight, {})
            assert isinstance(weight_group, dict)
            weight_group.setdefault(profile, []).append(run)
        else:
            source_group = grouped.setdefault(source, {})
            weight_runs = source_group.setdefault(teacher_weight, [])
            assert isinstance(weight_runs, list)
            weight_runs.append(run)
    if profile_dimension:
        return {
            source: {
                weight: {
                    profile: _distillation_stability_group_summary(profile_runs)
                    for profile, profile_runs in sorted(profile_groups.items())
                }
                for weight, profile_groups in sorted(weight_groups.items())
                if isinstance(profile_groups, dict)
            }
            for source, weight_groups in sorted(grouped.items())
        }
    return {
        source: {
            weight: _distillation_stability_group_summary(group_runs)
            for weight, group_runs in sorted(weight_groups.items())
            if isinstance(group_runs, list)
        }
        for source, weight_groups in sorted(grouped.items())
    }


def _distillation_stability_group_summary(runs: list[dict[str, Any]]) -> dict[str, Any]:
    teacher_agreement_values: dict[str, list[float]] = {}
    margin_bucket_values: dict[str, dict[str, list[float]]] = {}
    confidence_values: dict[str, list[float]] = {}
    confidence_bucket_values: dict[str, dict[str, list[float]]] = {}
    feedback_delta_values: dict[str, list[float]] = {}
    gate_pass_count = 0
    for run in runs:
        if _teacher_quality_gate_status(run) == "passed":
            gate_pass_count += 1
        evaluation = run.get("validation_evaluation", {})
        evaluation = _comparison_from_evaluation(evaluation) if isinstance(evaluation, dict) else {}
        torch_metrics = evaluation.get("torch_policy", {})
        torch_metrics = torch_metrics if isinstance(torch_metrics, dict) else {}
        teacher_agreement = _teacher_agreement_summary(torch_metrics)
        for metric, value in teacher_agreement.items():
            if metric == "feedback_aware_margin_bucket_agreement":
                continue
            numeric = _optional_numeric(value)
            if numeric is not None:
                teacher_agreement_values.setdefault(metric, []).append(numeric)
        margin_report = teacher_agreement.get("feedback_aware_margin_bucket_agreement", {})
        if isinstance(margin_report, dict):
            for bucket_name, bucket_metrics in margin_report.items():
                if not isinstance(bucket_metrics, dict):
                    continue
                bucket_values = margin_bucket_values.setdefault(str(bucket_name), {})
                for metric, value in bucket_metrics.items():
                        numeric = _optional_numeric(value)
                        if numeric is not None:
                            bucket_values.setdefault(str(metric), []).append(numeric)
        confidence = torch_metrics.get("feedback_aware_confidence_calibration", {})
        if isinstance(confidence, dict):
            _collect_confidence_values(
                confidence,
                confidence_values=confidence_values,
                confidence_bucket_values=confidence_bucket_values,
            )
        baseline_deltas = run.get("baseline_deltas", {})
        feedback_delta = baseline_deltas.get("feedback_aware", {}) if isinstance(baseline_deltas, dict) else {}
        if isinstance(feedback_delta, dict):
            for metric, value in feedback_delta.items():
                numeric = _optional_numeric(value)
                if numeric is not None:
                    feedback_delta_values.setdefault(str(metric), []).append(numeric)
    run_count = len(runs)
    return {
        "run_count": run_count,
        "seed_count": len({int(run["seed"]) for run in runs if run.get("seed") is not None}),
        "teacher_quality_gate_pass_rate": gate_pass_count / run_count if run_count else 0.0,
        "teacher_quality_gate_pass_count": gate_pass_count,
        "teacher_agreement": _numeric_stats_by_metric(teacher_agreement_values),
        "margin_bucket_agreement": {
            bucket: _numeric_stats_by_metric(metrics)
            for bucket, metrics in sorted(margin_bucket_values.items())
        },
        "confidence_calibration": {
            **_numeric_stats_by_metric(confidence_values),
            "margin_buckets": {
                bucket: _numeric_stats_by_metric(metrics)
                for bucket, metrics in sorted(confidence_bucket_values.items())
            },
        },
        "feedback_aware_baseline_delta": _numeric_stats_by_metric(feedback_delta_values),
    }


def _collect_confidence_values(
    confidence: dict[str, Any],
    *,
    confidence_values: dict[str, list[float]],
    confidence_bucket_values: dict[str, dict[str, list[float]]],
) -> None:
    for metric, value in confidence.items():
        if metric in {"margin_buckets", "warnings"}:
            continue
        numeric = _nested_metric_mean(value)
        if numeric is not None:
            confidence_values.setdefault(str(metric), []).append(numeric)
    margin_buckets = confidence.get("margin_buckets", {})
    if isinstance(margin_buckets, dict):
        for bucket_name, bucket_metrics in margin_buckets.items():
            if not isinstance(bucket_metrics, dict):
                continue
            bucket_values = confidence_bucket_values.setdefault(str(bucket_name), {})
            for metric, value in bucket_metrics.items():
                numeric = _nested_metric_mean(value)
                if numeric is not None:
                    bucket_values.setdefault(str(metric), []).append(numeric)


def _nested_metric_mean(value: Any) -> float | None:
    if isinstance(value, dict) and "mean" in value:
        return _optional_numeric(value.get("mean"))
    return _optional_numeric(value)


def _distillation_profile_dimension_enabled(runs: list[dict[str, Any]]) -> bool:
    return any("teacher_margin_curriculum_profile" in run for run in runs)


def _numeric_stats_by_metric(values: dict[str, list[float]]) -> dict[str, dict[str, float]]:
    return {
        metric: _numeric_stats(tuple(metric_values))
        for metric, metric_values in sorted(values.items())
    }


def _optional_numeric(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if isfinite(numeric) else None


def _normalize_training_source_name(selection_strategy: str | None) -> str:
    return "manifest" if selection_strategy is None else str(selection_strategy).strip().replace("/", "_")


def _normalize_teacher_weight_name(value: float) -> str:
    text = f"{float(value):g}"
    return text.replace("-", "neg-").replace(".", "p")


def _normalize_curriculum_profile_name(value: str | None) -> str:
    text = "default" if value is None else str(value).strip()
    return (text or "default").replace("/", "_").replace(" ", "_")


def _calibration_recommendation(
    runs: list[dict[str, Any]],
    *,
    policy: str,
    metric: str,
    selected_run: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not runs:
        raise ValueError("training must produce at least one run")
    profile_records = _calibration_profile_records(runs, policy=policy, metric=metric)
    eligible_profiles = [record for record in profile_records if not record["gate_failed_profile"]]
    if selected_run is None:
        candidate_profiles = eligible_profiles if eligible_profiles else profile_records
        best_profile = max(
            candidate_profiles,
            key=lambda record: (
                record["metric"]["mean"],
                record["teacher_quality_gate_pass_rate"],
                record["source_selection_strategy"],
                str(record["teacher_imitation_weight"]),
                record["teacher_margin_curriculum_profile"],
            ),
        )
    else:
        selected_profile_key = _calibration_profile_key(selected_run)
        best_profile = next(record for record in profile_records if record["profile_key"] == selected_profile_key)
    profile_runs = [
        run
        for run in runs
        if _calibration_profile_key(run) == best_profile["profile_key"]
    ]
    best_run = selected_run or _select_best_training_run(profile_runs, policy=policy, metric=metric)
    gate_failed_selection = best_profile["gate_failed_profile"] or _teacher_quality_gate_failed(best_run)
    reason_codes = ["selected_best_profile"] if selected_run is None else ["legacy_best_run_selection"]
    if gate_failed_selection:
        reason_codes.append("gate_failed_selection")
    excluded_profiles = [
        {
            key: record[key]
            for key in (
                "profile_key",
                "source_selection_strategy",
                "teacher_imitation_weight",
                "teacher_margin_curriculum_profile",
                "gate_status",
                "teacher_quality_gate_pass_rate",
                "metric",
                "reason_codes",
            )
        }
        for record in profile_records
        if record["profile_key"] != best_profile["profile_key"] and record["gate_failed_profile"]
    ]
    return {
        "status": "selected",
        "policy": str(policy),
        "metric": str(metric),
        "mode": "max_mean" if selected_run is None else "legacy_best_run",
        "profile_key": best_profile["profile_key"],
        "source_selection_strategy": best_profile["source_selection_strategy"],
        "teacher_imitation_weight": best_profile["teacher_imitation_weight"],
        "teacher_margin_curriculum_profile": best_profile["teacher_margin_curriculum_profile"],
        "recommended_checkpoint": best_run.get("checkpoint"),
        "recommended_seed": best_run.get("seed"),
        "gate_status": best_profile["gate_status"],
        "gate_failed_selection": gate_failed_selection,
        "inconclusive": False,
        "profile_selection_reason_codes": reason_codes,
        "eligible_profile_count": len(eligible_profiles),
        "excluded_profile_count": len(excluded_profiles),
        "excluded_profiles": excluded_profiles,
        "profiles": [
            {key: value for key, value in record.items() if key != "_runs"}
            for record in profile_records
        ],
        "selected_profile": {key: value for key, value in best_profile.items() if key != "_runs"},
        "reason": (
            f"selected profile {best_profile['profile_key']} by mean {policy}.{metric}; "
            f"gate_status={best_profile['gate_status']}"
        ),
    }


def _calibration_profile_records(
    runs: list[dict[str, Any]],
    *,
    policy: str,
    metric: str,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        grouped.setdefault(_calibration_profile_key(run), []).append(run)
    records: list[dict[str, Any]] = []
    for profile_key, profile_runs in sorted(grouped.items()):
        metric_values = tuple(_training_run_metric(run, policy=policy, metric=metric) for run in profile_runs)
        failed_count = sum(1 for run in profile_runs if _teacher_quality_gate_failed(run))
        pass_count = sum(1 for run in profile_runs if _teacher_quality_gate_status(run) == "passed")
        gate_failed_profile = failed_count == len(profile_runs)
        gate_status = (
            "failed"
            if gate_failed_profile
            else "passed"
            if pass_count == len(profile_runs)
            else "mixed"
            if failed_count or pass_count
            else "not_configured"
        )
        first = profile_runs[0]
        reason_codes = ["teacher_quality_gate_failed"] if gate_failed_profile else ["eligible_profile"]
        records.append(
            {
                "profile_key": profile_key,
                "source_selection_strategy": _normalize_training_source_name(
                    first.get("training_data_selection_strategy")
                ),
                "teacher_imitation_weight": float(first.get("teacher_imitation_weight", 0.0)),
                "teacher_margin_curriculum_profile": _run_curriculum_profile_name(first),
                "run_count": len(profile_runs),
                "seed_count": len({int(run["seed"]) for run in profile_runs if run.get("seed") is not None}),
                "gate_status": gate_status,
                "gate_failed_profile": gate_failed_profile,
                "teacher_quality_gate_pass_rate": pass_count / len(profile_runs) if profile_runs else 0.0,
                "teacher_quality_gate_pass_count": pass_count,
                "metric": _numeric_stats(metric_values),
                "stability": _distillation_stability_group_summary(profile_runs),
                "reason_codes": reason_codes,
                "_runs": profile_runs,
            }
        )
    return records


def _recommended_run(runs: list[dict[str, Any]], recommendation: dict[str, Any]) -> dict[str, Any]:
    checkpoint = recommendation.get("recommended_checkpoint")
    for run in runs:
        if run.get("checkpoint") == checkpoint:
            return run
    return _select_best_training_run(runs, policy=str(recommendation["policy"]), metric=str(recommendation["metric"]))


def _calibration_profile_key(run: dict[str, Any]) -> str:
    return "|".join(
        (
            _normalize_training_source_name(run.get("training_data_selection_strategy")),
            f"{float(run.get('teacher_imitation_weight', 0.0)):g}",
            _run_curriculum_profile_name(run),
        )
    )


def _run_curriculum_profile_name(run: dict[str, Any]) -> str:
    value = run.get("teacher_margin_curriculum_profile")
    if value is None:
        curriculum = run.get("teacher_curriculum", {})
        if isinstance(curriculum, dict):
            value = curriculum.get("profile_name")
    return _normalize_curriculum_profile_name(None if value is None else str(value))


def _select_best_training_run(runs: list[dict[str, Any]], *, policy: str, metric: str) -> dict[str, Any]:
    if not runs:
        raise ValueError("training must produce at least one run")
    eligible_runs = [run for run in runs if not _teacher_quality_gate_failed(run)]
    candidates = eligible_runs if eligible_runs else runs
    return max(candidates, key=lambda run: _training_run_metric(run, policy=policy, metric=metric))


def _best_selection_record(
    runs: list[dict[str, Any]],
    best_run: dict[str, Any],
    *,
    policy: str,
    metric: str,
) -> dict[str, Any]:
    excluded_runs = [
        _excluded_run_record(run, policy=policy, metric=metric)
        for run in runs
        if run is not best_run and _teacher_quality_gate_failed(run)
    ]
    gate_status = _teacher_quality_gate_status(best_run)
    gate_failed_selection = _teacher_quality_gate_failed(best_run)
    source = _normalize_training_source_name(best_run.get("training_data_selection_strategy"))
    teacher_weight = float(best_run.get("teacher_imitation_weight", 0.0))
    curriculum_profile = _run_curriculum_profile_name(best_run)
    metric_value = _training_run_metric(best_run, policy=policy, metric=metric)
    reason_codes = ["selected_best_run"]
    if gate_failed_selection:
        reason_codes.append("gate_failed_selection")
    return {
        "policy": str(policy),
        "metric": str(metric),
        "mode": "max",
        "value": metric_value,
        "source_selection_strategy": source,
        "teacher_imitation_weight": teacher_weight,
        "teacher_margin_curriculum_profile": curriculum_profile,
        "gate_status": gate_status,
        "gate_failed_selection": gate_failed_selection,
        "reason_codes": reason_codes,
        "eligible_run_count": sum(1 for run in runs if not _teacher_quality_gate_failed(run)),
        "excluded_run_count": len(excluded_runs),
        "excluded_runs": excluded_runs,
        "reason": (
            f"max {policy}.{metric} on validation evaluation; "
            f"source={source}; teacher_imitation_weight={teacher_weight}; gate_status={gate_status}"
        ),
    }


def _excluded_run_record(run: dict[str, Any], *, policy: str, metric: str) -> dict[str, Any]:
    return {
        "checkpoint": run.get("checkpoint"),
        "seed": run.get("seed"),
        "source_selection_strategy": _normalize_training_source_name(run.get("training_data_selection_strategy")),
        "teacher_imitation_weight": float(run.get("teacher_imitation_weight", 0.0)),
        "teacher_margin_curriculum_profile": _run_curriculum_profile_name(run),
        "gate_status": _teacher_quality_gate_status(run),
        "metric_value": _training_run_metric(run, policy=policy, metric=metric),
        "reason_codes": ["teacher_quality_gate_failed"],
    }


def _teacher_quality_gate_status(run: dict[str, Any]) -> str:
    gates = run.get("teacher_quality_gates")
    if not isinstance(gates, dict):
        return "not_configured"
    return str(gates.get("status", "not_configured"))


def _teacher_quality_gate_failed(run: dict[str, Any]) -> bool:
    return _teacher_quality_gate_status(run) == "failed"


def _training_run_metric(run: dict[str, Any], *, policy: str, metric: str) -> float:
    evaluation = run.get("validation_evaluation", {})
    if not isinstance(evaluation, dict):
        return float("-inf")
    evaluation = _comparison_from_evaluation(evaluation)
    policy_metrics = evaluation.get(policy, {})
    if not isinstance(policy_metrics, dict):
        return float("-inf")
    value = policy_metrics.get(metric)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("-inf")


def _multi_seed_evaluation_summary(runs: list[dict[str, Any]]) -> dict[str, Any]:
    values: dict[str, dict[str, list[float]]] = {}
    for run in runs:
        evaluation = run.get("validation_evaluation", {})
        if not isinstance(evaluation, dict):
            continue
        evaluation = _comparison_from_evaluation(evaluation)
        for policy, metrics in evaluation.items():
            if not isinstance(metrics, dict):
                continue
            policy_values = values.setdefault(str(policy), {})
            for metric, value in metrics.items():
                if isinstance(value, bool):
                    continue
                try:
                    numeric = float(value)
                except (TypeError, ValueError):
                    continue
                metric_values = policy_values.setdefault(str(metric), [])
                metric_values.append(numeric)
    return {
        policy: {metric: _numeric_stats(tuple(metric_values)) for metric, metric_values in metrics.items()}
        for policy, metrics in values.items()
    }


_BASELINE_DELTA_METRICS = (
    "final_coverage_rate",
    "cumulative_coverage_rate_delta",
    "total_path_cost",
    "average_risk",
    "failure_count",
    "value_coverage",
)


def _baseline_deltas(evaluation: dict[str, Any], *, policy: str = "torch_policy") -> dict[str, Any]:
    policy_metrics = evaluation.get(policy)
    if not isinstance(policy_metrics, dict):
        return {}
    deltas: dict[str, Any] = {policy: {}}
    for baseline_name in _baseline_names(evaluation, policy=policy):
        baseline_metrics = evaluation.get(baseline_name)
        if not isinstance(baseline_metrics, dict):
            continue
        deltas[policy][baseline_name] = {
            metric: _metric_value(policy_metrics, metric) - _metric_value(baseline_metrics, metric)
            for metric in _BASELINE_DELTA_METRICS
        }
    return deltas


def _baseline_names(evaluation: dict[str, Any], *, policy: str) -> tuple[str, ...]:
    preferred = ("utility", "coverage_heuristic", "feedback_aware")
    names = [name for name in preferred if name in evaluation and name != policy]
    names.extend(
        sorted(
            str(name)
            for name, metrics in evaluation.items()
            if name not in set(preferred)
            and name != policy
            and isinstance(metrics, dict)
        )
    )
    return tuple(names)


def _multi_seed_delta_summary(runs: list[dict[str, Any]]) -> dict[str, Any]:
    values: dict[str, dict[str, list[float]]] = {}
    for run in runs:
        evaluation = run.get("validation_evaluation", {})
        if not isinstance(evaluation, dict):
            continue
        evaluation = _comparison_from_evaluation(evaluation)
        deltas = _baseline_deltas(evaluation).get("torch_policy", {})
        if not isinstance(deltas, dict):
            continue
        for baseline_name, metrics in deltas.items():
            if not isinstance(metrics, dict):
                continue
            baseline_values = values.setdefault(str(baseline_name), {})
            for metric, value in metrics.items():
                baseline_values.setdefault(str(metric), []).append(float(value))
    return {
        baseline: {metric: _numeric_stats(tuple(metric_values)) for metric, metric_values in metrics.items()}
        for baseline, metrics in values.items()
    }


def _metric_value(metrics: dict[str, Any], metric: str) -> float:
    value = metrics.get(metric)
    if value is None and metric == "final_coverage_rate":
        value = metrics.get("average_final_coverage_rate")
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _numeric_stats(values: tuple[float, ...]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    average = sum(values) / len(values)
    variance = sum((value - average) ** 2 for value in values) / len(values)
    return {
        "mean": average,
        "std": sqrt(variance),
        "min": min(values),
        "max": max(values),
    }


# Public aliases
baseline_deltas = _baseline_deltas
best_selection_record = _best_selection_record
calibration_recommendation = _calibration_recommendation
distillation_stability_summary = _distillation_stability_summary
select_best_training_run = _select_best_training_run
training_distillation_matrix = _training_distillation_matrix
training_source_comparison = _training_source_comparison
multi_seed_delta_summary = _multi_seed_delta_summary

__all__ = [name for name in globals() if not name.startswith("__")]
