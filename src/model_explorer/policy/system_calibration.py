from __future__ import annotations

import json
from math import isfinite
from pathlib import Path
from typing import Any


SYSTEM_CALIBRATION_SCHEMA_VERSION = "system-calibration-summary/v1"
CALIBRATION_EVALUATION_SCOPE = "calibration evidence; not real-world generalization benchmark"


def load_path_feedback_summary_entries(config: dict[str, Any], *, base_dir: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for item in _raw_path_feedback_summary_entries(config):
        if not isinstance(item, dict):
            raise ValueError("system_calibration.path_feedback_summaries entries must be objects")
        entry = {key: value for key, value in item.items() if key not in {"path", "summary"}}
        if isinstance(item.get("summary"), dict):
            entry["summary"] = dict(item["summary"])
        else:
            path_value = item.get("path", item.get("summary"))
            if path_value is None:
                raise ValueError("path feedback summary entries require path or summary")
            summary_path = _resolve_path(base_dir, path_value)
            entry["path"] = str(summary_path)
            entry["summary"] = json.loads(summary_path.read_text(encoding="utf-8"))
        entries.append(entry)
    return entries


def annotate_runs_with_path_feedback_gates(
    runs: list[dict[str, Any]],
    *,
    path_feedback_summaries: list[dict[str, Any]],
    gate_config: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    for run in runs:
        entry = _matching_path_feedback_entry(run, path_feedback_summaries)
        if entry is None:
            run["path_feedback_gate"] = {
                "status": "not_configured",
                "reason_codes": ["path_feedback_summary_not_configured"],
            }
            continue
        run["path_feedback_gate"] = evaluate_path_feedback_gate(
            entry["summary"],
            gate_config or {},
        )
    return runs


def evaluate_path_feedback_gate(summary: dict[str, Any], gate_config: dict[str, Any] | None = None) -> dict[str, Any]:
    config = dict(gate_config or {})
    metrics = _path_feedback_metrics(summary)
    reason_codes: list[str] = []
    warning_reason_codes: list[str] = []
    if bool(config.get("require_open_grid_fallback_used_false", True)) and metrics["open_grid_fallback_used"]:
        reason_codes.append("open_grid_fallback_used")
    acceptance_metadata = _evaluate_acceptance_metadata(summary, config.get("acceptance_gate"))
    warning_reason_codes.extend(acceptance_metadata["warning_reason_codes"])
    reason_codes.extend(acceptance_metadata["exclusion_reason_codes"])
    _append_rate_violation(
        reason_codes,
        config,
        metrics,
        config_key="max_path_planning_failure_rate",
        metric_key="path_planning_failure_rate",
        reason_code="path_planning_failure_rate_exceeded",
    )
    _append_rate_violation(
        reason_codes,
        config,
        metrics,
        config_key="max_replan_rate",
        metric_key="replan_rate",
        reason_code="replan_rate_exceeded",
    )
    _append_rate_violation(
        reason_codes,
        config,
        metrics,
        config_key="max_region_graph_disconnected_rate",
        metric_key="region_graph_disconnected_rate",
        reason_code="region_graph_disconnected_rate_exceeded",
    )
    _append_rate_violation(
        reason_codes,
        config,
        metrics,
        config_key="max_iris_fallback_rate",
        metric_key="iris_fallback_rate",
        reason_code="iris_fallback_rate_exceeded",
    )
    _append_rate_violation(
        reason_codes,
        config,
        metrics,
        config_key="max_region_graph_fallback_rate",
        metric_key="region_graph_fallback_rate",
        reason_code="region_graph_fallback_rate_exceeded",
    )
    _append_count_violation(
        reason_codes,
        config,
        metrics,
        config_key="max_path_planning_failure_count",
        metric_key="path_planning_failure_count",
        reason_code="path_planning_failure_count_exceeded",
    )
    _append_count_violation(
        reason_codes,
        config,
        metrics,
        config_key="max_replan_count",
        metric_key="replan_count",
        reason_code="replan_count_exceeded",
    )

    return {
        "status": "failed" if reason_codes else "passed",
        "reason_codes": reason_codes or ["path_feedback_gate_passed"],
        "warning_reason_codes": warning_reason_codes,
        "metrics": metrics,
        "acceptance_metadata": acceptance_metadata,
        "stress_diagnostics": _path_feedback_group_diagnostics(summary, "stress"),
        "mixed_stress_diagnostics": _path_feedback_group_diagnostics(summary, "mixed_stress"),
        "quality_signal_use": "calibration_only",
        "not_real_world_performance_claim": True,
    }


def build_system_calibration_summary(
    training_summary: dict[str, Any],
    *,
    path_feedback_summaries: list[dict[str, Any]] | None = None,
    config: dict[str, Any] | None = None,
    policy: str = "torch_policy",
    metric: str = "final_coverage_rate",
) -> dict[str, Any]:
    system_config = dict(config or {})
    run_copies = [dict(run) for run in training_summary.get("runs", [])]
    if not run_copies:
        raise ValueError("system calibration requires training runs")
    entries = list(path_feedback_summaries or [])
    system_gate_configured = _path_feedback_gate_enabled(system_config)
    if system_gate_configured:
        annotate_runs_with_path_feedback_gates(
            run_copies,
            path_feedback_summaries=entries,
            gate_config=_path_feedback_gate_config(system_config),
        )
    else:
        for run in run_copies:
            run["path_feedback_gate"] = {
                "status": "not_configured",
                "reason_codes": ["system_path_feedback_gate_not_configured"],
            }

    selection = select_system_best_run(
        run_copies,
        policy=policy,
        metric=metric,
        system_gate_configured=system_gate_configured,
    )
    selected_run = selection.get("run")
    summary_runs = [
        _system_run_record(
            run,
            selected_run=selected_run,
            policy=policy,
            metric=metric,
            system_gate_configured=system_gate_configured,
        )
        for run in run_copies
    ]
    data_class = _first_non_empty(
        system_config.get("data_class"),
        *(
            run.get("dataset_summary", {}).get("data_class")
            for run in run_copies
            if isinstance(run.get("dataset_summary"), dict)
        ),
        default="unknown",
    )
    mask_stress_augmented = bool(
        system_config.get(
            "mask_stress_augmented",
            any(
                bool(run.get("dataset_summary", {}).get("mask_stress_augmented"))
                for run in run_copies
                if isinstance(run.get("dataset_summary"), dict)
            ),
        )
    )
    sample_quality_config = _sample_quality_config(system_config)
    sample_quality_summary = (
        build_sample_quality_summary(entries, sample_quality_config)
        if sample_quality_config is not None
        else None
    )
    summary = {
        "schema_version": SYSTEM_CALIBRATION_SCHEMA_VERSION,
        "status": "selected" if selected_run is not None else "no_eligible_run",
        "policy": str(policy),
        "metric": str(metric),
        "data_class": data_class,
        "benchmark_scope": str(system_config.get("benchmark_scope", "not real-world generalization benchmark")),
        "mask_stress_augmented": mask_stress_augmented,
        "evaluation_scope": CALIBRATION_EVALUATION_SCOPE,
        "calibration_recommendation": dict(training_summary.get("calibration_recommendation", {})),
        "path_feedback_gate_configured": system_gate_configured,
        "gate_summary": _joint_gate_summary(run_copies, system_gate_configured=system_gate_configured),
        "path_feedback_diagnostics": _combined_path_feedback_diagnostics(entries),
        "selection": _selection_summary(selection),
        "runs": summary_runs,
    }
    if sample_quality_summary is not None:
        summary["sample_quality_summary"] = sample_quality_summary
    return summary


def select_system_best_run(
    runs: list[dict[str, Any]],
    *,
    policy: str,
    metric: str,
    system_gate_configured: bool,
) -> dict[str, Any]:
    if not runs:
        raise ValueError("best-run selection requires at least one run")
    excluded_runs = [
        _excluded_run_record(run, policy=policy, metric=metric, system_gate_configured=system_gate_configured)
        for run in runs
        if _run_exclusion_reason_codes(run, system_gate_configured=system_gate_configured)
    ]
    if system_gate_configured:
        candidates = [
            run
            for run in runs
            if not _run_exclusion_reason_codes(run, system_gate_configured=True)
        ]
        mode = "system_gate_best_metric"
    else:
        candidates = [run for run in runs if not _teacher_quality_gate_failed(run)]
        if not candidates:
            candidates = list(runs)
        mode = "legacy_best_metric"
        excluded_runs = [
            _excluded_run_record(run, policy=policy, metric=metric, system_gate_configured=False)
            for run in runs
            if _teacher_quality_gate_failed(run)
        ]
    if not candidates:
        return {
            "status": "no_eligible_run",
            "mode": mode,
            "run": None,
            "reason_codes": ["no_run_passed_teacher_and_path_feedback_gates"],
            "excluded_runs": excluded_runs,
        }
    best_run = max(candidates, key=lambda run: _training_run_metric(run, policy=policy, metric=metric))
    return {
        "status": "selected",
        "mode": mode,
        "run": best_run,
        "reason_codes": ["selected_best_run"],
        "excluded_runs": excluded_runs,
    }


def _raw_path_feedback_summary_entries(config: dict[str, Any]) -> list[dict[str, Any]]:
    if "path_feedback_summaries" in config:
        value = config["path_feedback_summaries"]
        if not isinstance(value, list):
            raise ValueError("system_calibration.path_feedback_summaries must be a list")
        return list(value)
    if "path_feedback_summary" in config:
        return [{"path": config["path_feedback_summary"]}]
    return []


def _path_feedback_gate_config(config: dict[str, Any]) -> dict[str, Any]:
    value = config.get("path_feedback_gate", {})
    if value is None:
        value = {}
    if not isinstance(value, dict):
        raise ValueError("system_calibration.path_feedback_gate must be an object")
    gate_config = dict(value)
    if "acceptance_gate" in config:
        gate_config["acceptance_gate"] = config["acceptance_gate"]
    return gate_config


def _path_feedback_gate_enabled(config: dict[str, Any]) -> bool:
    return "path_feedback_gate" in config or "acceptance_gate" in config


def _sample_quality_config(config: dict[str, Any]) -> dict[str, Any] | None:
    value = config.get("sample_quality")
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("system_calibration.sample_quality must be an object")
    if not bool(value.get("enabled", True)):
        return None
    return dict(value)


def _evaluate_acceptance_metadata(summary: dict[str, Any], expected_gate: Any) -> dict[str, Any]:
    if expected_gate is None:
        return {
            "status": "not_configured",
            "reason_codes": ["acceptance_metadata_check_not_configured"],
            "warning_reason_codes": [],
            "exclusion_reason_codes": [],
            "mismatches": [],
        }
    if not isinstance(expected_gate, dict):
        raise ValueError("system_calibration.acceptance_gate must be an object")
    metadata = summary.get("acceptance_metadata")
    if not isinstance(metadata, dict):
        return {
            "status": "missing",
            "reason_codes": ["acceptance_metadata_missing"],
            "warning_reason_codes": ["acceptance_metadata_missing"],
            "exclusion_reason_codes": ["acceptance_metadata_missing"],
            "mismatches": [],
            "expected": dict(expected_gate),
        }
    checks = (
        ("scenario_set", expected_gate.get("scenario_set")),
        ("diagnostic_profile", expected_gate.get("diagnostic_profile")),
        ("top_k", expected_gate.get("top_k")),
    )
    mismatches = []
    for field, expected in checks:
        if expected is None:
            continue
        actual = metadata.get(field, summary.get(field))
        if str(actual) != str(expected):
            mismatches.append({"field": field, "expected": expected, "actual": actual})
    if "planner_extra_args" in expected_gate:
        expected_args = [str(value) for value in expected_gate.get("planner_extra_args", [])]
        actual_args = [str(value) for value in metadata.get("planner_extra_args", summary.get("planner_extra_args", []))]
        if actual_args != expected_args:
            mismatches.append({"field": "planner_extra_args", "expected": expected_args, "actual": actual_args})
    required_open_grid = expected_gate.get("require_open_grid_fallback_used")
    if required_open_grid is not None:
        actual_open_grid = bool(metadata.get("open_grid_fallback_used", summary.get("open_grid_fallback_used")))
        if actual_open_grid is not bool(required_open_grid):
            mismatches.append(
                {
                    "field": "open_grid_fallback_used",
                    "expected": bool(required_open_grid),
                    "actual": actual_open_grid,
                }
            )
    if mismatches:
        return {
            "status": "mismatched",
            "reason_codes": ["acceptance_metadata_mismatch"],
            "warning_reason_codes": ["acceptance_metadata_mismatch"],
            "exclusion_reason_codes": ["acceptance_metadata_mismatch"],
            "mismatches": mismatches,
            "expected": dict(expected_gate),
            "actual": dict(metadata),
        }
    return {
        "status": "passed",
        "reason_codes": ["acceptance_metadata_matched"],
        "warning_reason_codes": [],
        "exclusion_reason_codes": [],
        "mismatches": [],
        "expected": dict(expected_gate),
        "actual": dict(metadata),
    }


def build_sample_quality_summary(
    path_feedback_summaries: list[dict[str, Any]],
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    quality_config = dict(config or {})
    exclude_reason_codes = _string_set(quality_config.get("exclude_reason_codes", ()))
    downweight_reason_codes = _string_set(quality_config.get("downweight_reason_codes", ()))
    downweight_factor = _optional_float(quality_config.get("downweight_factor", 0.5))
    if downweight_factor is None:
        downweight_factor = 0.5
    records: list[dict[str, Any]] = []
    for entry in path_feedback_summaries:
        summary = entry.get("summary") if isinstance(entry, dict) and "summary" in entry else entry
        if not isinstance(summary, dict):
            continue
        for scenario in summary.get("scenarios", []):
            if not isinstance(scenario, dict):
                continue
            reason_codes = _sample_quality_reason_codes(scenario)
            decision = "keep"
            sample_weight = 1.0
            if exclude_reason_codes.intersection(reason_codes):
                decision = "exclude"
                sample_weight = 0.0
            elif downweight_reason_codes.intersection(reason_codes):
                decision = "downweight"
                sample_weight = max(0.0, float(downweight_factor))
            records.append(
                {
                    "scenario_id": str(scenario.get("scenario_id", "")),
                    "scenario_group": str(scenario.get("scenario_group", "unknown")),
                    "decision": decision,
                    "sample_weight": sample_weight,
                    "reason_codes": reason_codes,
                    "quality_signal_use": "calibration_only",
                    "not_real_world_performance_claim": True,
                }
            )
    decision_counts = _counts(record["decision"] for record in records)
    return {
        "schema_version": "sample-quality-summary/v1",
        "enabled": True,
        "quality_signal_scope": str(quality_config.get("quality_signal_scope", "calibration_only")),
        "benchmark_scope": str(
            quality_config.get("benchmark_scope", "not real-world generalization benchmark")
        ),
        "quality_signal_use": "calibration_only",
        "not_real_world_performance_claim": True,
        "record_count": len(records),
        "excluded_sample_count": int(decision_counts.get("exclude", 0)),
        "downweighted_sample_count": int(decision_counts.get("downweight", 0)),
        "kept_sample_count": int(decision_counts.get("keep", 0)),
        "decision_counts": decision_counts,
        "reason_code_counts": _counts(
            reason
            for record in records
            for reason in record["reason_codes"]
        ),
        "records": records,
    }


def filter_episodes_by_sample_quality(
    episodes,
    sample_quality_summary: dict[str, Any],
    config: dict[str, Any] | None = None,
):
    quality_config = dict(config or {})
    match_key = str(quality_config.get("match_key", "scenario_id"))
    episode_tuple = tuple(episodes)
    records_by_id = {
        str(record.get(match_key, record.get("scenario_id"))): record
        for record in sample_quality_summary.get("records", [])
        if isinstance(record, dict) and record.get(match_key, record.get("scenario_id")) is not None
    }
    kept = []
    applied_records = []
    for episode in episode_tuple:
        sample_id = _episode_quality_match_value(episode, match_key)
        record = records_by_id.get(sample_id)
        if record is not None:
            applied_records.append(dict(record))
        if record is not None and record.get("decision") == "exclude":
            continue
        kept.append(episode)
    decision_counts = _counts(record.get("decision", "keep") for record in applied_records)
    applied_summary = dict(sample_quality_summary)
    applied_summary.update(
        {
            "match_key": match_key,
            "input_episode_count": len(episode_tuple),
            "matched_sample_count": len(applied_records),
            "excluded_sample_count": int(decision_counts.get("exclude", 0)),
            "downweighted_sample_count": int(decision_counts.get("downweight", 0)),
            "kept_episode_count": len(kept),
            "applied_records": applied_records,
        }
    )
    return tuple(kept), applied_summary


def _sample_quality_reason_codes(scenario: dict[str, Any]) -> list[str]:
    codes: list[str] = []
    feedback = scenario.get("path_feedback", {})
    if isinstance(feedback, dict):
        if _int_value(feedback.get("failure_count")) > 0:
            codes.append("path_planning_failure")
        if _int_value(feedback.get("replan_count")) > 0:
            codes.append("replan_required")
        for candidate in feedback.get("candidates", []):
            if not isinstance(candidate, dict):
                continue
            interpretation = candidate.get("diagnostic_interpretation", {})
            if isinstance(interpretation, dict):
                for flag in interpretation.get("diagnostic_flags", []):
                    codes.append(_normalize_sample_quality_reason(flag))
                if bool(interpretation.get("open_grid_fallback_used")):
                    codes.append("open_grid_fallback")
            if candidate.get("failure_reason"):
                codes.append("path_planning_failure")
            if bool(candidate.get("replan_required")):
                codes.append("replan_required")
    interpretation = scenario.get("diagnostic_interpretation", {})
    if isinstance(interpretation, dict):
        for source in interpretation.get("failure_sources", []):
            codes.append(_normalize_sample_quality_reason(source))
        if bool(interpretation.get("open_grid_fallback_used")):
            codes.append("open_grid_fallback")
    if bool(scenario.get("open_grid_fallback_used")):
        codes.append("open_grid_fallback")
    return _dedupe_strings(code for code in codes if code and code != "none") or ["sample_quality_passed"]


def _normalize_sample_quality_reason(value: Any) -> str:
    text = str(value).strip()
    aliases = {
        "replan": "replan_required",
        "path_failure": "path_planning_failure",
        "failure": "path_planning_failure",
        "iris_region_fallback": "iris_fallback",
        "region_graph_start_goal_disconnected": "region_graph_disconnected",
    }
    return aliases.get(text, text)


def _episode_quality_match_value(episode, match_key: str) -> str:
    for transition in getattr(episode, "transitions", ()):
        extra = getattr(getattr(transition, "info", None), "extra", {})
        if not isinstance(extra, dict):
            continue
        if match_key in extra:
            return str(extra[match_key])
        provenance = extra.get("provenance")
        if isinstance(provenance, dict) and match_key in provenance:
            return str(provenance[match_key])
    return ""


def _string_set(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        return {value}
    if not isinstance(value, list | tuple | set):
        raise ValueError("sample quality reason code lists must be arrays")
    return {str(item) for item in value}


def _counts(values) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _dedupe_strings(values) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value)
        if text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _resolve_path(base_dir: Path, value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else base_dir / path


def _matching_path_feedback_entry(run: dict[str, Any], entries: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not entries:
        return None
    matches = [entry for entry in entries if _entry_matches_run(entry, run)]
    if matches:
        return matches[0]
    unconstrained = [entry for entry in entries if not _entry_match_keys(entry)]
    return unconstrained[0] if len(unconstrained) == 1 else None


def _entry_matches_run(entry: dict[str, Any], run: dict[str, Any]) -> bool:
    keys = _entry_match_keys(entry)
    if not keys:
        return True
    for key in keys:
        if key == "source_selection_strategy":
            if _normalize_source(entry[key]) != _normalize_source(run.get("training_data_selection_strategy")):
                return False
        elif key == "teacher_imitation_weight":
            if float(entry[key]) != float(run.get("teacher_imitation_weight", 0.0)):
                return False
        elif key == "teacher_margin_curriculum_profile":
            if _normalize_profile(entry[key]) != _run_curriculum_profile_name(run):
                return False
        elif key == "seed":
            if int(entry[key]) != int(run.get("seed", -1)):
                return False
        elif key == "architecture":
            if str(entry[key]) != str(run.get("architecture", "mlp_v1")):
                return False
    return True


def _entry_match_keys(entry: dict[str, Any]) -> tuple[str, ...]:
    return tuple(
        key
        for key in (
            "source_selection_strategy",
            "teacher_imitation_weight",
            "teacher_margin_curriculum_profile",
            "seed",
            "architecture",
        )
        if key in entry
    )


def _path_feedback_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    candidate_count = max(0, _int_value(summary.get("candidate_count")))
    denominator = candidate_count if candidate_count else 1
    failure_count = _int_value(summary.get("path_planning_failure_count"))
    replan_count = _int_value(summary.get("replan_count"))
    region_disconnected_count = _int_value(
        summary.get("region_graph_disconnected_count", summary.get("region_graph_start_goal_disconnected_count"))
    )
    iris_fallback_count = _int_value(summary.get("iris_fallback_count"))
    region_graph_fallback_count = _int_value(summary.get("region_graph_fallback_count"))
    return {
        "scenario_count": _int_value(summary.get("scenario_count")),
        "top_k": _int_value(summary.get("top_k")),
        "candidate_count": candidate_count,
        "reachable_count": _int_value(summary.get("reachable_count")),
        "path_planning_failure_count": failure_count,
        "path_planning_failure_rate": failure_count / denominator,
        "replan_count": replan_count,
        "replan_rate": replan_count / denominator,
        "region_graph_disconnected_count": region_disconnected_count,
        "region_graph_disconnected_rate": region_disconnected_count / denominator,
        "iris_fallback_count": iris_fallback_count,
        "iris_fallback_rate": iris_fallback_count / denominator,
        "region_graph_fallback_count": region_graph_fallback_count,
        "region_graph_fallback_rate": region_graph_fallback_count / denominator,
        "open_grid_fallback_used": bool(summary.get("open_grid_fallback_used")),
    }


def _append_rate_violation(
    reason_codes: list[str],
    config: dict[str, Any],
    metrics: dict[str, Any],
    *,
    config_key: str,
    metric_key: str,
    reason_code: str,
) -> None:
    if config_key not in config:
        return
    threshold = _optional_float(config.get(config_key))
    actual = _optional_float(metrics.get(metric_key))
    if threshold is not None and actual is not None and actual > threshold:
        reason_codes.append(reason_code)


def _append_count_violation(
    reason_codes: list[str],
    config: dict[str, Any],
    metrics: dict[str, Any],
    *,
    config_key: str,
    metric_key: str,
    reason_code: str,
) -> None:
    if config_key not in config:
        return
    threshold = _optional_float(config.get(config_key))
    actual = _optional_float(metrics.get(metric_key))
    if threshold is not None and actual is not None and actual > threshold:
        reason_codes.append(reason_code)


def _path_feedback_group_diagnostics(summary: dict[str, Any], group_name: str) -> dict[str, Any]:
    groups = summary.get("scenario_group_summary", {})
    group = groups.get(group_name, {}) if isinstance(groups, dict) else {}
    if not isinstance(group, dict):
        group = {}
    return {
        "scenario_count": _int_value(group.get("scenario_count")),
        "candidate_count": _int_value(group.get("candidate_count")),
        "reachable_count": _int_value(group.get("reachable_count")),
        "failure_count": _int_value(group.get("failure_count")),
        "replan_count": _int_value(group.get("replan_count")),
        "iris_fallback_count": _int_value(group.get("iris_fallback_count")),
        "region_graph_fallback_count": _int_value(group.get("region_graph_fallback_count")),
        "region_graph_start_goal_disconnected_count": _int_value(
            group.get("region_graph_start_goal_disconnected_count")
        ),
    }


def _combined_path_feedback_diagnostics(entries: list[dict[str, Any]]) -> dict[str, Any]:
    result = {
        "stress": _empty_path_feedback_group(),
        "mixed_stress": _empty_path_feedback_group(),
    }
    for entry in entries:
        summary = entry.get("summary")
        if not isinstance(summary, dict):
            continue
        for group_name in ("stress", "mixed_stress"):
            group = _path_feedback_group_diagnostics(summary, group_name)
            for key, value in group.items():
                result[group_name][key] += int(value)
    return result


def _empty_path_feedback_group() -> dict[str, int]:
    return {
        "scenario_count": 0,
        "candidate_count": 0,
        "reachable_count": 0,
        "failure_count": 0,
        "replan_count": 0,
        "iris_fallback_count": 0,
        "region_graph_fallback_count": 0,
        "region_graph_start_goal_disconnected_count": 0,
    }


def _joint_gate_summary(runs: list[dict[str, Any]], *, system_gate_configured: bool) -> dict[str, Any]:
    run_count = len(runs)
    teacher_pass_count = sum(1 for run in runs if _teacher_quality_gate_status(run) == "passed")
    teacher_fail_count = sum(1 for run in runs if _teacher_quality_gate_failed(run))
    path_pass_count = sum(1 for run in runs if _path_feedback_gate_status(run) == "passed")
    path_fail_count = sum(1 for run in runs if _path_feedback_gate_failed(run))
    joint_pass_count = sum(
        1
        for run in runs
        if not _teacher_quality_gate_failed(run)
        and (not system_gate_configured or not _path_feedback_gate_failed(run))
        and (not system_gate_configured or _path_feedback_gate_status(run) == "passed")
    )
    return {
        "run_count": run_count,
        "teacher_quality_gate_pass_count": teacher_pass_count,
        "teacher_quality_gate_fail_count": teacher_fail_count,
        "teacher_quality_gate_pass_rate": teacher_pass_count / run_count if run_count else 0.0,
        "path_feedback_gate_configured": system_gate_configured,
        "path_feedback_gate_pass_count": path_pass_count,
        "path_feedback_gate_fail_count": path_fail_count,
        "path_feedback_gate_pass_rate": path_pass_count / run_count if run_count else 0.0,
        "joint_gate_pass_count": joint_pass_count,
        "joint_gate_pass_rate": joint_pass_count / run_count if run_count else 0.0,
    }


def _selection_summary(selection: dict[str, Any]) -> dict[str, Any]:
    run = selection.get("run")
    return {
        "status": selection["status"],
        "mode": selection["mode"],
        "recommended_checkpoint": None if run is None else run.get("checkpoint"),
        "recommended_seed": None if run is None else run.get("seed"),
        "reason_codes": list(selection.get("reason_codes", [])),
        "excluded_run_count": len(selection.get("excluded_runs", [])),
        "excluded_runs": list(selection.get("excluded_runs", [])),
    }


def _system_run_record(
    run: dict[str, Any],
    *,
    selected_run: dict[str, Any] | None,
    policy: str,
    metric: str,
    system_gate_configured: bool,
) -> dict[str, Any]:
    reason_codes = _run_exclusion_reason_codes(run, system_gate_configured=system_gate_configured)
    selected = selected_run is run
    status = "selected" if selected else "excluded" if reason_codes else "eligible_not_selected"
    if selected:
        reason_codes = ["selected_best_run"]
    metric_value = _training_run_metric(run, policy=policy, metric=metric)
    source = _normalize_source(run.get("training_data_selection_strategy"))
    teacher_weight = float(run.get("teacher_imitation_weight", 0.0))
    curriculum_profile = _run_curriculum_profile_name(run)
    return {
        "checkpoint": run.get("checkpoint"),
        "seed": run.get("seed"),
        "source_selection_strategy": source,
        "teacher_imitation_weight": teacher_weight,
        "teacher_margin_curriculum_profile": curriculum_profile,
        "teacher_quality_gates": dict(run.get("teacher_quality_gates", {})),
        "path_feedback_gate": dict(run.get("path_feedback_gate", {})),
        "metric_value": metric_value,
        "selection_decision": {
            "status": status,
            "reason_codes": reason_codes,
            "policy": str(policy),
            "metric": str(metric),
            "metric_value": metric_value,
            "source_selection_strategy": source,
            "teacher_imitation_weight": teacher_weight,
            "teacher_margin_curriculum_profile": curriculum_profile,
            "seed": run.get("seed"),
        },
        "quality_signal_use": "calibration_only",
        "not_real_world_performance_claim": True,
    }


def _excluded_run_record(
    run: dict[str, Any],
    *,
    policy: str,
    metric: str,
    system_gate_configured: bool,
) -> dict[str, Any]:
    return {
        "checkpoint": run.get("checkpoint"),
        "seed": run.get("seed"),
        "source_selection_strategy": _normalize_source(run.get("training_data_selection_strategy")),
        "teacher_imitation_weight": float(run.get("teacher_imitation_weight", 0.0)),
        "teacher_margin_curriculum_profile": _run_curriculum_profile_name(run),
        "teacher_quality_gates": dict(run.get("teacher_quality_gates", {})),
        "path_feedback_gate": dict(run.get("path_feedback_gate", {})),
        "metric_value": _training_run_metric(run, policy=policy, metric=metric),
        "reason_codes": _run_exclusion_reason_codes(run, system_gate_configured=system_gate_configured),
    }


def _run_exclusion_reason_codes(run: dict[str, Any], *, system_gate_configured: bool) -> list[str]:
    reason_codes: list[str] = []
    if _teacher_quality_gate_failed(run):
        reason_codes.append("teacher_quality_gate_failed")
    if system_gate_configured and _path_feedback_gate_failed(run):
        gate = run.get("path_feedback_gate", {})
        gate_reasons = gate.get("reason_codes", []) if isinstance(gate, dict) else []
        reason_codes.extend(str(reason) for reason in gate_reasons)
    return reason_codes


def _teacher_quality_gate_status(run: dict[str, Any]) -> str:
    gates = run.get("teacher_quality_gates")
    if not isinstance(gates, dict):
        return "not_configured"
    return str(gates.get("status", "not_configured"))


def _teacher_quality_gate_failed(run: dict[str, Any]) -> bool:
    return _teacher_quality_gate_status(run) == "failed"


def _path_feedback_gate_status(run: dict[str, Any]) -> str:
    gate = run.get("path_feedback_gate")
    if not isinstance(gate, dict):
        return "not_configured"
    return str(gate.get("status", "not_configured"))


def _path_feedback_gate_failed(run: dict[str, Any]) -> bool:
    return _path_feedback_gate_status(run) == "failed"


def _training_run_metric(run: dict[str, Any], *, policy: str, metric: str) -> float:
    evaluation = run.get("validation_evaluation", {})
    if isinstance(evaluation, dict) and isinstance(evaluation.get("aggregate"), dict):
        evaluation = evaluation["aggregate"]
    if not isinstance(evaluation, dict):
        return float("-inf")
    policy_metrics = evaluation.get(policy, {})
    if not isinstance(policy_metrics, dict):
        return float("-inf")
    value = policy_metrics.get(metric)
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float("-inf")
    return numeric if isfinite(numeric) else float("-inf")


def _run_curriculum_profile_name(run: dict[str, Any]) -> str:
    value = run.get("teacher_margin_curriculum_profile")
    if value is None:
        curriculum = run.get("teacher_curriculum", {})
        if isinstance(curriculum, dict):
            value = curriculum.get("profile_name")
    return _normalize_profile(value)


def _normalize_source(value: Any) -> str:
    return "manifest" if value is None else str(value).strip().replace("/", "_")


def _normalize_profile(value: Any) -> str:
    text = "default" if value is None else str(value).strip()
    return (text or "default").replace("/", "_").replace(" ", "_")


def _int_value(value: Any) -> int:
    if isinstance(value, bool) or value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _optional_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if isfinite(numeric) else None


def _first_non_empty(*values: Any, default: str) -> str:
    for value in values:
        if value is not None and str(value).strip():
            return str(value)
    return default
