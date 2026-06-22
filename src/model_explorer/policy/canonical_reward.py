from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION_V2 = "xunce-canonical-reward-guard-profile/v2"
SCHEMA_VERSION_V3 = "xunce-canonical-reward-guard-profile/v3"
SCHEMA_VERSION = SCHEMA_VERSION_V2

SOFT_REWARD_COMPONENT_KEYS_V2 = (
    "coverage_weight",
    "valuable_coverage_weight",
    "information_weight",
    "path_cost_weight",
    "risk_weight",
    "fallback_penalty",
    "failure_penalty",
)

SOFT_REWARD_COMPONENT_KEYS_V3 = (
    "coverage_weight",
    "roi_coverage_weight",
    "information_weight",
    "path_cost_weight",
    "soft_risk_weight",
    "fallback_penalty",
    "failure_penalty",
)

NORMALIZER_KEYS_V2 = (
    "coverage_gain_rate",
    "valuable_coverage",
    "information_gain",
    "path_cost_m",
    "risk_proxy",
)

NORMALIZER_KEYS_V3 = (
    "coverage_gain_rate",
    "roi_coverage",
    "information_gain",
    "path_cost_m",
    "soft_risk_exposure",
)

GUARD_KEYS_V2 = (
    "min_coverage_delta_cells",
    "max_acceptable_path_cost_delta_m",
    "min_coverage_per_100m_delta",
    "max_acceptable_risk_delta",
    "max_acceptable_risk_cost_weighted_delta",
    "coverage_gain_per_path_cost_delta_mode",
    "aggregation_policy",
)

GUARD_KEYS_V3 = (
    "min_coverage_delta_cells",
    "max_acceptable_path_cost_delta_m",
    "min_coverage_per_100m_delta",
    "max_soft_risk_exposure_delta",
    "max_hard_risk_violation_count",
)

RISK_POLICY_KEYS_V3 = (
    "path_cost_includes_risk_proxy",
    "soft_risk_component_mode",
)

PROFILE_KEYS_V2 = (
    "schema_version",
    "profile_id",
    "profile_version",
    "soft_reward_components",
    "normalizers",
    "guards",
)

PROFILE_KEYS_V3 = (
    "schema_version",
    "profile_id",
    "profile_version",
    "soft_reward_components",
    "normalizers",
    "risk_policy",
    "trajectory_guards",
)

CANONICAL_REWARD_COMPONENTS_V2 = (
    "coverage_component",
    "valuable_coverage_component",
    "information_component",
    "path_cost_component",
    "risk_component",
    "fallback_component",
    "failure_component",
)

CANONICAL_REWARD_COMPONENTS_V3 = (
    "coverage_component",
    "roi_coverage_component",
    "information_component",
    "path_cost_component",
    "soft_risk_component",
    "fallback_component",
    "failure_component",
)
CANONICAL_REWARD_COMPONENTS = CANONICAL_REWARD_COMPONENTS_V2

GUARD_METRIC_FIELDS_V2 = (
    "coverage_cells",
    "path_cost_m",
    "risk_proxy",
    "risk_cost_weighted",
    "coverage_per_100m",
)

GUARD_METRIC_FIELDS_V3 = (
    "coverage_cells",
    "path_cost_m",
    "soft_risk_exposure",
    "hard_risk_violation_count",
    "coverage_per_100m",
)


@dataclass(frozen=True)
class CanonicalRewardGuardProfile:
    schema_version: str
    profile_id: str
    profile_version: str
    soft_reward_components: dict[str, float]
    normalizers: dict[str, float]
    guards: dict[str, Any]
    profile_hash: str
    risk_policy: dict[str, Any] | None = None

    def to_content_dict(self) -> dict[str, Any]:
        if self.schema_version == SCHEMA_VERSION_V3:
            return {
                "schema_version": self.schema_version,
                "profile_id": self.profile_id,
                "profile_version": self.profile_version,
                "soft_reward_components": dict(self.soft_reward_components),
                "normalizers": dict(self.normalizers),
                "risk_policy": dict(self.risk_policy or {}),
                "trajectory_guards": dict(self.guards),
            }
        return {
            "schema_version": self.schema_version,
            "profile_id": self.profile_id,
            "profile_version": self.profile_version,
            "soft_reward_components": dict(self.soft_reward_components),
            "normalizers": dict(self.normalizers),
            "guards": dict(self.guards),
        }


@dataclass(frozen=True)
class RewardComponentResult:
    components: dict[str, float]
    reward: float
    profile_id: str
    profile_version: str
    profile_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "components": dict(self.components),
            "reward": float(self.reward),
            "profile_id": self.profile_id,
            "profile_version": self.profile_version,
            "profile_hash": self.profile_hash,
        }


@dataclass(frozen=True)
class GuardResult:
    passed: bool
    failed_guards: list[str]
    reason_codes: list[str]
    observed: dict[str, float | None]
    thresholds: dict[str, Any]
    profile_id: str
    profile_version: str
    profile_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": bool(self.passed),
            "failed_guards": list(self.failed_guards),
            "reason_codes": list(self.reason_codes),
            "observed": dict(self.observed),
            "thresholds": dict(self.thresholds),
            "profile_id": self.profile_id,
            "profile_version": self.profile_version,
            "profile_hash": self.profile_hash,
        }


def load_canonical_reward_profile(path: str | Path) -> CanonicalRewardGuardProfile:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("canonical reward profile root must be a JSON object")
    return _profile_from_payload(payload)


def canonical_profile_hash(profile: CanonicalRewardGuardProfile | Mapping[str, Any]) -> str:
    payload = _profile_content(profile)
    stable = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(stable.encode("utf-8")).hexdigest()


def compute_canonical_reward_components(
    metrics: Mapping[str, Any],
    profile: CanonicalRewardGuardProfile,
) -> RewardComponentResult:
    if profile.schema_version == SCHEMA_VERSION_V3:
        return _compute_canonical_reward_components_v3(metrics, profile)

    weights = profile.soft_reward_components
    normalizers = profile.normalizers
    components = {
        "coverage_component": _round(
            _nonnegative(metrics.get("coverage_gain_rate"))
            / normalizers["coverage_gain_rate"]
            * weights["coverage_weight"]
        ),
        "valuable_coverage_component": _round(
            _nonnegative(metrics.get("valuable_coverage"))
            / normalizers["valuable_coverage"]
            * weights["valuable_coverage_weight"]
        ),
        "information_component": _round(
            _nonnegative(metrics.get("information_gain"))
            / normalizers["information_gain"]
            * weights["information_weight"]
        ),
        "path_cost_component": _round(
            -_nonnegative(metrics.get("path_cost_m"))
            / normalizers["path_cost_m"]
            * weights["path_cost_weight"]
        ),
        "risk_component": _round(
            -_nonnegative(metrics.get("risk_proxy"))
            / normalizers["risk_proxy"]
            * weights["risk_weight"]
        ),
        "fallback_component": _round(-weights["fallback_penalty"] if _truthy(metrics.get("fallback_used")) else 0.0),
        "failure_component": _round(-weights["failure_penalty"] if _failure(metrics) else 0.0),
    }
    reward = _round(sum(components.values()))
    return RewardComponentResult(
        components=components,
        reward=reward,
        profile_id=profile.profile_id,
        profile_version=profile.profile_version,
        profile_hash=profile.profile_hash,
    )


def evaluate_canonical_guard(
    candidate_metrics: Mapping[str, Any],
    baseline_metrics: Mapping[str, Any],
    profile: CanonicalRewardGuardProfile,
) -> GuardResult:
    if profile.schema_version == SCHEMA_VERSION_V3:
        return _evaluate_canonical_guard_v3(candidate_metrics, baseline_metrics, profile)

    missing: list[str] = []
    candidate_values: dict[str, float] = {}
    baseline_values: dict[str, float] = {}
    for field in GUARD_METRIC_FIELDS_V2:
        candidate_value = _finite_float(candidate_metrics.get(field))
        baseline_value = _finite_float(baseline_metrics.get(field))
        if candidate_value is None:
            missing.append(f"candidate.{field}")
        else:
            candidate_values[field] = candidate_value
        if baseline_value is None:
            missing.append(f"baseline.{field}")
        else:
            baseline_values[field] = baseline_value

    observed: dict[str, float | None] = {
        "coverage_delta_cells": None,
        "path_cost_delta_m": None,
        "risk_delta": None,
        "risk_cost_weighted_delta": None,
        "coverage_per_100m_delta": None,
        "coverage_gain_per_path_cost_delta": _optional_delta(
            candidate_metrics,
            baseline_metrics,
            "coverage_gain_per_path_cost",
        ),
    }
    failed_guards: list[str] = []
    reason_codes: list[str] = []
    if missing:
        failed_guards.append("missing_guard_metric")
        reason_codes.extend(missing)
    else:
        observed["coverage_delta_cells"] = _round(
            candidate_values["coverage_cells"] - baseline_values["coverage_cells"]
        )
        observed["path_cost_delta_m"] = _round(
            candidate_values["path_cost_m"] - baseline_values["path_cost_m"]
        )
        observed["risk_delta"] = _round(
            candidate_values["risk_proxy"] - baseline_values["risk_proxy"]
        )
        observed["risk_cost_weighted_delta"] = _round(
            candidate_values["risk_cost_weighted"] - baseline_values["risk_cost_weighted"]
        )
        observed["coverage_per_100m_delta"] = _round(
            candidate_values["coverage_per_100m"] - baseline_values["coverage_per_100m"]
        )

        guards = profile.guards
        if observed["coverage_delta_cells"] < guards["min_coverage_delta_cells"]:
            failed_guards.append("coverage_gain_insufficient")
        if observed["path_cost_delta_m"] > guards["max_acceptable_path_cost_delta_m"]:
            failed_guards.append("path_cost_budget_exceeded")
        if observed["risk_delta"] > guards["max_acceptable_risk_delta"]:
            failed_guards.append("risk_budget_exceeded")
        if observed["risk_cost_weighted_delta"] > guards["max_acceptable_risk_cost_weighted_delta"]:
            failed_guards.append("risk_cost_weighted_budget_exceeded")
        if observed["coverage_per_100m_delta"] < guards["min_coverage_per_100m_delta"]:
            failed_guards.append("coverage_efficiency_regression")
        reason_codes.extend(failed_guards)

    return GuardResult(
        passed=not failed_guards,
        failed_guards=_unique(failed_guards),
        reason_codes=_unique(reason_codes),
        observed=observed,
        thresholds=dict(profile.guards),
        profile_id=profile.profile_id,
        profile_version=profile.profile_version,
        profile_hash=profile.profile_hash,
    )


def _profile_from_payload(payload: Mapping[str, Any]) -> CanonicalRewardGuardProfile:
    schema_version = payload.get("schema_version")
    if schema_version == SCHEMA_VERSION_V3:
        return _profile_v3_from_payload(payload)
    if schema_version != SCHEMA_VERSION_V2:
        raise ValueError(f"expected schema_version {SCHEMA_VERSION_V2} or {SCHEMA_VERSION_V3}")
    _validate_keys("profile", payload, PROFILE_KEYS_V2)
    profile_id = _required_string(payload, "profile_id")
    profile_version = _required_string(payload, "profile_version")
    soft_reward_components = _numeric_section(
        payload.get("soft_reward_components"),
        SOFT_REWARD_COMPONENT_KEYS_V2,
        section_name="soft_reward_components",
        strictly_positive=False,
    )
    normalizers = _numeric_section(
        payload.get("normalizers"),
        NORMALIZER_KEYS_V2,
        section_name="normalizers",
        strictly_positive=True,
    )
    guards = _guard_section_v2(payload.get("guards"))
    content = {
        "schema_version": SCHEMA_VERSION_V2,
        "profile_id": profile_id,
        "profile_version": profile_version,
        "soft_reward_components": soft_reward_components,
        "normalizers": normalizers,
        "guards": guards,
    }
    profile_hash = canonical_profile_hash(content)
    return CanonicalRewardGuardProfile(
        schema_version=SCHEMA_VERSION_V2,
        profile_id=profile_id,
        profile_version=profile_version,
        soft_reward_components=soft_reward_components,
        normalizers=normalizers,
        guards=guards,
        profile_hash=profile_hash,
    )


def _profile_content(profile: CanonicalRewardGuardProfile | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(profile, CanonicalRewardGuardProfile):
        return profile.to_content_dict()
    payload = dict(profile)
    if "profile_hash" in payload:
        payload.pop("profile_hash")
    if payload.get("schema_version") == SCHEMA_VERSION_V3:
        _validate_keys("profile", payload, PROFILE_KEYS_V3)
        return {
            "schema_version": payload["schema_version"],
            "profile_id": payload["profile_id"],
            "profile_version": payload["profile_version"],
            "soft_reward_components": dict(payload["soft_reward_components"]),
            "normalizers": dict(payload["normalizers"]),
            "risk_policy": dict(payload["risk_policy"]),
            "trajectory_guards": dict(payload["trajectory_guards"]),
        }
    _validate_keys("profile", payload, PROFILE_KEYS_V2)
    return {
        "schema_version": payload["schema_version"],
        "profile_id": payload["profile_id"],
        "profile_version": payload["profile_version"],
        "soft_reward_components": dict(payload["soft_reward_components"]),
        "normalizers": dict(payload["normalizers"]),
        "guards": dict(payload["guards"]),
    }


def _profile_v3_from_payload(payload: Mapping[str, Any]) -> CanonicalRewardGuardProfile:
    _validate_keys("profile", payload, PROFILE_KEYS_V3)
    profile_id = _required_string(payload, "profile_id")
    profile_version = _required_string(payload, "profile_version")
    soft_reward_components = _numeric_section(
        payload.get("soft_reward_components"),
        SOFT_REWARD_COMPONENT_KEYS_V3,
        section_name="soft_reward_components",
        strictly_positive=False,
    )
    normalizers = _numeric_section(
        payload.get("normalizers"),
        NORMALIZER_KEYS_V3,
        section_name="normalizers",
        strictly_positive=True,
    )
    risk_policy = _risk_policy_section_v3(payload.get("risk_policy"))
    guards = _guard_section_v3(payload.get("trajectory_guards"))
    content = {
        "schema_version": SCHEMA_VERSION_V3,
        "profile_id": profile_id,
        "profile_version": profile_version,
        "soft_reward_components": soft_reward_components,
        "normalizers": normalizers,
        "risk_policy": risk_policy,
        "trajectory_guards": guards,
    }
    profile_hash = canonical_profile_hash(content)
    return CanonicalRewardGuardProfile(
        schema_version=SCHEMA_VERSION_V3,
        profile_id=profile_id,
        profile_version=profile_version,
        soft_reward_components=soft_reward_components,
        normalizers=normalizers,
        guards=guards,
        profile_hash=profile_hash,
        risk_policy=risk_policy,
    )


def _compute_canonical_reward_components_v3(
    metrics: Mapping[str, Any],
    profile: CanonicalRewardGuardProfile,
) -> RewardComponentResult:
    weights = profile.soft_reward_components
    normalizers = profile.normalizers
    failure_triggered = _failure(metrics) or _hard_risk_violation(metrics)
    components = {
        "coverage_component": _round(
            _nonnegative(metrics.get("coverage_gain_rate"))
            / normalizers["coverage_gain_rate"]
            * weights["coverage_weight"]
        ),
        "roi_coverage_component": _round(
            _nonnegative(metrics.get("roi_coverage"))
            / normalizers["roi_coverage"]
            * weights["roi_coverage_weight"]
        ),
        "information_component": _round(
            _nonnegative(metrics.get("information_gain"))
            / normalizers["information_gain"]
            * weights["information_weight"]
        ),
        "path_cost_component": _round(
            -_nonnegative(metrics.get("path_cost_m"))
            / normalizers["path_cost_m"]
            * weights["path_cost_weight"]
        ),
        "soft_risk_component": _round(
            -_nonnegative(metrics.get("soft_risk_exposure"))
            / normalizers["soft_risk_exposure"]
            * weights["soft_risk_weight"]
        ),
        "fallback_component": _round(-weights["fallback_penalty"] if _truthy(metrics.get("fallback_used")) else 0.0),
        "failure_component": _round(-weights["failure_penalty"] if failure_triggered else 0.0),
    }
    reward = _round(sum(components.values()))
    if failure_triggered and reward > 0.0:
        components["failure_component"] = _round(components["failure_component"] - reward)
        reward = _round(sum(components.values()))
    return RewardComponentResult(
        components=components,
        reward=reward,
        profile_id=profile.profile_id,
        profile_version=profile.profile_version,
        profile_hash=profile.profile_hash,
    )


def _numeric_section(
    section: Any,
    keys: tuple[str, ...],
    *,
    section_name: str,
    strictly_positive: bool,
) -> dict[str, float]:
    if not isinstance(section, dict):
        raise ValueError(f"{section_name} must be an object")
    _validate_keys(section_name, section, keys)
    result: dict[str, float] = {}
    for key in keys:
        value = _finite_float(section.get(key))
        if value is None:
            raise ValueError(f"{section_name}.{key} must be finite")
        if strictly_positive and value <= 0.0:
            raise ValueError(f"{section_name}.{key} must be > 0")
        if not strictly_positive and value < 0.0:
            raise ValueError(f"{section_name}.{key} must be >= 0")
        result[key] = value
    return result


def _evaluate_canonical_guard_v3(
    candidate_metrics: Mapping[str, Any],
    baseline_metrics: Mapping[str, Any],
    profile: CanonicalRewardGuardProfile,
) -> GuardResult:
    missing: list[str] = []
    candidate_values: dict[str, float] = {}
    baseline_values: dict[str, float] = {}
    for field in GUARD_METRIC_FIELDS_V3:
        candidate_value = _finite_float(candidate_metrics.get(field))
        baseline_value = _finite_float(baseline_metrics.get(field))
        if candidate_value is None:
            missing.append(f"candidate.{field}")
        else:
            candidate_values[field] = candidate_value
        if baseline_value is None:
            missing.append(f"baseline.{field}")
        else:
            baseline_values[field] = baseline_value

    observed: dict[str, float | None] = {
        "coverage_delta_cells": None,
        "path_cost_delta_m": None,
        "soft_risk_exposure_delta": None,
        "hard_risk_violation_count": None,
        "coverage_per_100m_delta": None,
    }
    failed_guards: list[str] = []
    reason_codes: list[str] = []
    if missing:
        failed_guards.append("missing_guard_metric")
        reason_codes.extend(missing)
    else:
        observed["coverage_delta_cells"] = _round(
            candidate_values["coverage_cells"] - baseline_values["coverage_cells"]
        )
        observed["path_cost_delta_m"] = _round(
            candidate_values["path_cost_m"] - baseline_values["path_cost_m"]
        )
        observed["soft_risk_exposure_delta"] = _round(
            candidate_values["soft_risk_exposure"] - baseline_values["soft_risk_exposure"]
        )
        observed["hard_risk_violation_count"] = _round(candidate_values["hard_risk_violation_count"])
        observed["coverage_per_100m_delta"] = _round(
            candidate_values["coverage_per_100m"] - baseline_values["coverage_per_100m"]
        )

        guards = profile.guards
        if observed["coverage_delta_cells"] < guards["min_coverage_delta_cells"]:
            failed_guards.append("coverage_gain_insufficient")
        if observed["path_cost_delta_m"] > guards["max_acceptable_path_cost_delta_m"]:
            failed_guards.append("path_cost_budget_exceeded")
        if observed["soft_risk_exposure_delta"] > guards["max_soft_risk_exposure_delta"]:
            failed_guards.append("soft_risk_exposure_budget_exceeded")
        if observed["hard_risk_violation_count"] > guards["max_hard_risk_violation_count"]:
            failed_guards.append("hard_risk_boundary_violation")
        if observed["coverage_per_100m_delta"] < guards["min_coverage_per_100m_delta"]:
            failed_guards.append("coverage_efficiency_regression")
        reason_codes.extend(failed_guards)

    return GuardResult(
        passed=not failed_guards,
        failed_guards=_unique(failed_guards),
        reason_codes=_unique(reason_codes),
        observed=observed,
        thresholds=dict(profile.guards),
        profile_id=profile.profile_id,
        profile_version=profile.profile_version,
        profile_hash=profile.profile_hash,
    )


def _guard_section_v2(section: Any) -> dict[str, Any]:
    if not isinstance(section, dict):
        raise ValueError("guards must be an object")
    _validate_keys("guards", section, GUARD_KEYS_V2)
    guards: dict[str, Any] = {}
    for key in GUARD_KEYS_V2:
        if key in {"coverage_gain_per_path_cost_delta_mode", "aggregation_policy"}:
            guards[key] = _required_string(section, key)
        else:
            value = _finite_float(section.get(key))
            if value is None:
                raise ValueError(f"guards.{key} must be finite")
            guards[key] = value
    if guards["max_acceptable_path_cost_delta_m"] < 0.0:
        raise ValueError("guards.max_acceptable_path_cost_delta_m must be >= 0")
    if guards["max_acceptable_risk_delta"] < 0.0:
        raise ValueError("guards.max_acceptable_risk_delta must be >= 0")
    if guards["max_acceptable_risk_cost_weighted_delta"] < 0.0:
        raise ValueError("guards.max_acceptable_risk_cost_weighted_delta must be >= 0")
    if guards["coverage_gain_per_path_cost_delta_mode"] != "audit_only":
        raise ValueError("guards.coverage_gain_per_path_cost_delta_mode must be audit_only")
    if guards["aggregation_policy"] != "scenario_worst_case":
        raise ValueError("guards.aggregation_policy must be scenario_worst_case")
    return guards


def _risk_policy_section_v3(section: Any) -> dict[str, Any]:
    if not isinstance(section, dict):
        raise ValueError("risk_policy must be an object")
    _validate_keys("risk_policy", section, RISK_POLICY_KEYS_V3)
    path_cost_includes_risk_proxy = section.get("path_cost_includes_risk_proxy")
    if not isinstance(path_cost_includes_risk_proxy, bool):
        raise ValueError("risk_policy.path_cost_includes_risk_proxy must be boolean")
    soft_mode = _required_string(section, "soft_risk_component_mode")
    if soft_mode not in {"audit_weighted_tiny", "weighted"}:
        raise ValueError("risk_policy.soft_risk_component_mode must be audit_weighted_tiny or weighted")
    return {
        "path_cost_includes_risk_proxy": path_cost_includes_risk_proxy,
        "soft_risk_component_mode": soft_mode,
    }


def _guard_section_v3(section: Any) -> dict[str, Any]:
    if not isinstance(section, dict):
        raise ValueError("trajectory_guards must be an object")
    _validate_keys("trajectory_guards", section, GUARD_KEYS_V3)
    guards: dict[str, float] = {}
    for key in GUARD_KEYS_V3:
        value = _finite_float(section.get(key))
        if value is None:
            raise ValueError(f"trajectory_guards.{key} must be finite")
        guards[key] = value
    if guards["max_acceptable_path_cost_delta_m"] < 0.0:
        raise ValueError("trajectory_guards.max_acceptable_path_cost_delta_m must be >= 0")
    if guards["max_soft_risk_exposure_delta"] < 0.0:
        raise ValueError("trajectory_guards.max_soft_risk_exposure_delta must be >= 0")
    if guards["max_hard_risk_violation_count"] < 0.0:
        raise ValueError("trajectory_guards.max_hard_risk_violation_count must be >= 0")
    return guards


def _validate_keys(section_name: str, payload: Mapping[str, Any], allowed: tuple[str, ...]) -> None:
    allowed_set = set(allowed)
    actual = set(payload)
    missing = sorted(allowed_set - actual)
    unknown = sorted(actual - allowed_set)
    if missing:
        raise ValueError(f"{section_name} missing required keys: {', '.join(missing)}")
    if unknown:
        raise ValueError(f"{section_name} contains unknown keys: {', '.join(unknown)}")


def _required_string(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value.strip()


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _nonnegative(value: Any) -> float:
    number = _finite_float(value)
    if number is None:
        return 0.0
    return max(number, 0.0)


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _failure(metrics: Mapping[str, Any]) -> bool:
    if _truthy(metrics.get("failure")):
        return True
    reason = metrics.get("failure_reason")
    return isinstance(reason, str) and bool(reason.strip())


def _hard_risk_violation(metrics: Mapping[str, Any]) -> bool:
    if metrics.get("path_allowed_by_risk") is False:
        return True
    count = _finite_float(metrics.get("hard_risk_violation_count"))
    return count is not None and count > 0.0


def _optional_delta(
    candidate_metrics: Mapping[str, Any],
    baseline_metrics: Mapping[str, Any],
    field: str,
) -> float | None:
    candidate_value = _finite_float(candidate_metrics.get(field))
    baseline_value = _finite_float(baseline_metrics.get(field))
    if candidate_value is None or baseline_value is None:
        return None
    return _round(candidate_value - baseline_value)


def _round(value: float) -> float:
    return round(float(value), 12)


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result
