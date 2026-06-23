from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = "xunce-stage21-coverage-first-ppo-reward-profile/v1"
SCHEMA_VERSION_V2 = "xunce-stage21-coverage-constrained-ppo-reward-profile/v2"

PROFILE_KEYS = (
    "schema_version",
    "profile_id",
    "profile_version",
    "target_final_coverage_rate",
    "coverage_cap_rate",
    "horizon_steps",
    "mission_budget_route_when_below_target",
    "weights",
    "normalizers",
    "risk_policy",
    "hard_risk_policy",
    "component_source_map",
)

WEIGHT_KEYS = (
    "step_coverage_gain",
    "coverage_progress",
    "final_coverage",
    "success_99pct",
    "path_cost",
    "soft_risk",
    "failure",
    "hard_risk_failure",
)

WEIGHT_KEYS_V2 = (
    "coverage_gain",
    "coverage_progress",
    "coverage_per_cost",
    "final_coverage",
    "success_99pct",
    "path_cost",
    "soft_risk",
    "failure",
    "hard_risk_failure",
)

NORMALIZER_KEYS = (
    "coverage_rate_delta",
    "remaining_coverage_gap",
    "final_coverage_rate",
    "path_cost_m",
    "soft_risk_exposure",
)

NORMALIZER_KEYS_V2 = (
    "coverage_rate_delta",
    "remaining_coverage_gap",
    "coverage_per_cost",
    "final_coverage_rate",
    "path_cost_m",
    "soft_risk_exposure",
)

RISK_POLICY_KEYS = (
    "path_cost_includes_risk_proxy",
    "soft_risk_component_mode",
    "max_soft_risk_to_path_cost_penalty_ratio",
)

REWARD_POLICY_KEYS_V2 = (
    "coverage_per_cost_path_cost_floor",
    "coverage_per_cost_component_cap_ratio",
    "path_cost_penalty_cap_ratio_when_below_target",
)

HARD_RISK_POLICY_KEYS = (
    "reject_before_reward",
    "clamp_positive_reward_to_non_positive",
    "failure_floor",
)

COMPONENT_KEYS = (
    "step_coverage_gain_component",
    "coverage_progress_component",
    "final_coverage_bonus_component",
    "success_99pct_bonus_component",
    "path_cost_component",
    "soft_risk_component",
    "failure_component",
    "hard_risk_component",
)

COMPONENT_KEYS_V2 = (
    "coverage_gain_component",
    "coverage_progress_component",
    "coverage_per_cost_component",
    "path_cost_component",
    "soft_risk_component",
    "final_coverage_bonus_component",
    "success_99pct_bonus_component",
    "failure_component",
    "hard_risk_component",
)


@dataclass(frozen=True)
class CoverageFirstRewardProfile:
    schema_version: str
    profile_id: str
    profile_version: str
    target_final_coverage_rate: float
    coverage_cap_rate: float
    horizon_steps: int
    mission_budget_route_when_below_target: str
    weights: dict[str, float]
    normalizers: dict[str, float]
    risk_policy: dict[str, Any]
    reward_policy: dict[str, Any]
    hard_risk_policy: dict[str, Any]
    component_source_map: dict[str, str]
    component_keys: tuple[str, ...]
    profile_hash: str

    def to_content_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "profile_id": self.profile_id,
            "profile_version": self.profile_version,
            "target_final_coverage_rate": self.target_final_coverage_rate,
            "coverage_cap_rate": self.coverage_cap_rate,
            "horizon_steps": self.horizon_steps,
            "mission_budget_route_when_below_target": self.mission_budget_route_when_below_target,
            "weights": dict(self.weights),
            "normalizers": dict(self.normalizers),
            "risk_policy": dict(self.risk_policy),
            "hard_risk_policy": dict(self.hard_risk_policy),
            "component_source_map": dict(self.component_source_map),
        }
        if self.schema_version == SCHEMA_VERSION_V2:
            payload["reward_policy"] = dict(self.reward_policy)
        return payload


@dataclass(frozen=True)
class CoverageFirstRewardResult:
    components: dict[str, float]
    reward: float
    trainable: bool
    reason_codes: list[str]
    profile_id: str
    profile_version: str
    profile_hash: str
    risk_deduplication_applied: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "components": dict(self.components),
            "reward": float(self.reward),
            "trainable": bool(self.trainable),
            "reason_codes": list(self.reason_codes),
            "profile_id": self.profile_id,
            "profile_version": self.profile_version,
            "profile_hash": self.profile_hash,
            "risk_deduplication_applied": bool(self.risk_deduplication_applied),
        }


def load_coverage_first_reward_profile(path: str | Path) -> CoverageFirstRewardProfile:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("coverage-first reward profile root must be a JSON object")
    return _profile_from_payload(payload)


def coverage_first_profile_hash(profile: CoverageFirstRewardProfile | Mapping[str, Any]) -> str:
    payload = profile.to_content_dict() if isinstance(profile, CoverageFirstRewardProfile) else dict(profile)
    payload.pop("profile_hash", None)
    stable = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(stable.encode("utf-8")).hexdigest()


def compute_coverage_first_reward_components(
    metrics: Mapping[str, Any],
    profile: CoverageFirstRewardProfile,
) -> CoverageFirstRewardResult:
    if profile.schema_version == SCHEMA_VERSION_V2:
        return _compute_coverage_constrained_v2_reward_components(metrics, profile)
    return _compute_coverage_first_v1_reward_components(metrics, profile)


def _compute_coverage_first_v1_reward_components(
    metrics: Mapping[str, Any],
    profile: CoverageFirstRewardProfile,
) -> CoverageFirstRewardResult:
    hard_risk = _hard_risk(metrics)
    failure = hard_risk or _truthy(metrics.get("failure")) or bool(str(metrics.get("failure_reason") or "").strip())
    final_coverage = min(profile.coverage_cap_rate, _nonnegative(metrics.get("final_coverage_rate")))
    coverage_delta = _nonnegative(metrics.get("coverage_rate_delta"))
    progress = min(profile.coverage_cap_rate, _nonnegative(metrics.get("coverage_progress_rate", final_coverage)))
    remaining_gap = max(0.0, profile.target_final_coverage_rate - progress)
    done = _truthy(metrics.get("done"))
    success_99 = done and final_coverage >= profile.target_final_coverage_rate

    weights = profile.weights
    normalizers = profile.normalizers
    positive_allowed = not hard_risk
    step_component = _round(
        coverage_delta / normalizers["coverage_rate_delta"] * weights["step_coverage_gain"]
    ) if positive_allowed else 0.0
    progress_component = _round(
        (profile.target_final_coverage_rate - remaining_gap)
        / normalizers["remaining_coverage_gap"]
        * weights["coverage_progress"]
    ) if positive_allowed else 0.0
    final_component = _round(
        final_coverage / normalizers["final_coverage_rate"] * weights["final_coverage"]
    ) if done and positive_allowed else 0.0
    success_component = _round(weights["success_99pct"]) if success_99 and positive_allowed else 0.0
    path_cost_component = _round(
        -_nonnegative(metrics.get("path_cost_m")) / normalizers["path_cost_m"] * weights["path_cost"]
    )
    raw_soft_risk_component = _round(
        -_nonnegative(metrics.get("soft_risk_exposure"))
        / normalizers["soft_risk_exposure"]
        * weights["soft_risk"]
    )
    soft_risk_component, dedup_applied = _dedup_soft_risk(raw_soft_risk_component, path_cost_component, profile)
    failure_component = _round(-weights["failure"]) if failure and not hard_risk else 0.0
    hard_risk_component = _round(-weights["hard_risk_failure"]) if hard_risk else 0.0
    components = {
        "step_coverage_gain_component": step_component,
        "coverage_progress_component": progress_component,
        "final_coverage_bonus_component": final_component,
        "success_99pct_bonus_component": success_component,
        "path_cost_component": path_cost_component,
        "soft_risk_component": soft_risk_component,
        "failure_component": failure_component,
        "hard_risk_component": hard_risk_component,
    }
    reward = _round(sum(components.values()))
    reason_codes: list[str] = []
    if hard_risk:
        reason_codes.append("hard_risk_rejected")
    if failure and not hard_risk:
        reason_codes.append("failure_penalty_applied")
    if done and final_coverage < profile.target_final_coverage_rate:
        reason_codes.append("final_coverage_below_99pct_target")
    if hard_risk and profile.hard_risk_policy["clamp_positive_reward_to_non_positive"] and reward > 0.0:
        components["hard_risk_component"] = _round(components["hard_risk_component"] - reward)
        reward = _round(sum(components.values()))
    floor = float(profile.hard_risk_policy["failure_floor"])
    if hard_risk and reward > floor:
        components["hard_risk_component"] = _round(components["hard_risk_component"] + (floor - reward))
        reward = _round(sum(components.values()))
    return CoverageFirstRewardResult(
        components=components,
        reward=reward,
        trainable=not hard_risk,
        reason_codes=_unique(reason_codes),
        profile_id=profile.profile_id,
        profile_version=profile.profile_version,
        profile_hash=profile.profile_hash,
        risk_deduplication_applied=dedup_applied,
    )


def _compute_coverage_constrained_v2_reward_components(
    metrics: Mapping[str, Any],
    profile: CoverageFirstRewardProfile,
) -> CoverageFirstRewardResult:
    hard_risk = _hard_risk(metrics)
    failure = hard_risk or _truthy(metrics.get("failure")) or bool(str(metrics.get("failure_reason") or "").strip())
    final_coverage = min(profile.coverage_cap_rate, _nonnegative(metrics.get("final_coverage_rate")))
    coverage_delta = _nonnegative(metrics.get("coverage_rate_delta"))
    progress = min(profile.coverage_cap_rate, _nonnegative(metrics.get("coverage_progress_rate", final_coverage)))
    remaining_gap = max(0.0, profile.target_final_coverage_rate - progress)
    done = _truthy(metrics.get("done"))
    success_99 = done and final_coverage >= profile.target_final_coverage_rate

    weights = profile.weights
    normalizers = profile.normalizers
    reward_policy = profile.reward_policy
    positive_allowed = not hard_risk
    coverage_gain_component = _round(
        coverage_delta / normalizers["coverage_rate_delta"] * weights["coverage_gain"]
    ) if positive_allowed else 0.0
    coverage_progress_component = _round(
        (profile.target_final_coverage_rate - remaining_gap)
        / normalizers["remaining_coverage_gap"]
        * weights["coverage_progress"]
    ) if positive_allowed else 0.0

    path_cost = _nonnegative(metrics.get("path_cost_m"))
    path_cost_floor = float(reward_policy["coverage_per_cost_path_cost_floor"])
    coverage_per_cost_value = _coverage_per_cost_value(metrics, coverage_delta, path_cost, path_cost_floor)
    if positive_allowed and coverage_delta > 0.0:
        normalized_cpc = coverage_per_cost_value / normalizers["coverage_per_cost"]
        coverage_gain_gate = min(1.0, coverage_delta / normalizers["coverage_rate_delta"])
        raw_cpc_component = _round(math.log1p(normalized_cpc) * weights["coverage_per_cost"])
        cpc_cap = abs(weights["coverage_per_cost"]) * float(reward_policy["coverage_per_cost_component_cap_ratio"])
        coverage_per_cost_component = _round(min(raw_cpc_component, cpc_cap) * coverage_gain_gate) if cpc_cap > 0.0 else 0.0
    else:
        coverage_per_cost_component = 0.0

    raw_path_cost_component = _round(
        -path_cost / normalizers["path_cost_m"] * weights["path_cost"]
    )
    path_cost_component = _cap_path_cost_component(raw_path_cost_component, coverage_gain_component, progress, profile)
    raw_soft_risk_component = _round(
        -_nonnegative(metrics.get("soft_risk_exposure"))
        / normalizers["soft_risk_exposure"]
        * weights["soft_risk"]
    )
    soft_risk_component, dedup_applied = _dedup_soft_risk(raw_soft_risk_component, path_cost_component, profile)
    final_component = _round(
        final_coverage / normalizers["final_coverage_rate"] * weights["final_coverage"]
    ) if done and positive_allowed else 0.0
    success_component = _round(weights["success_99pct"]) if success_99 and positive_allowed else 0.0
    failure_component = _round(-weights["failure"]) if failure and not hard_risk else 0.0
    hard_risk_component = _round(-weights["hard_risk_failure"]) if hard_risk else 0.0

    components = {
        "coverage_gain_component": coverage_gain_component,
        "coverage_progress_component": coverage_progress_component,
        "coverage_per_cost_component": coverage_per_cost_component,
        "path_cost_component": path_cost_component,
        "soft_risk_component": soft_risk_component,
        "final_coverage_bonus_component": final_component,
        "success_99pct_bonus_component": success_component,
        "failure_component": failure_component,
        "hard_risk_component": hard_risk_component,
    }
    reward = _round(sum(components.values()))
    reason_codes: list[str] = []
    if hard_risk:
        reason_codes.append("hard_risk_rejected")
    if failure and not hard_risk:
        reason_codes.append("failure_penalty_applied")
    if done and final_coverage < profile.target_final_coverage_rate:
        reason_codes.append("final_coverage_below_99pct_target")
    if coverage_delta <= 0.0:
        reason_codes.append("coverage_per_cost_component_inactive_without_coverage_gain")
    if hard_risk and profile.hard_risk_policy["clamp_positive_reward_to_non_positive"] and reward > 0.0:
        components["hard_risk_component"] = _round(components["hard_risk_component"] - reward)
        reward = _round(sum(components.values()))
    floor = float(profile.hard_risk_policy["failure_floor"])
    if hard_risk and reward > floor:
        components["hard_risk_component"] = _round(components["hard_risk_component"] + (floor - reward))
        reward = _round(sum(components.values()))
    return CoverageFirstRewardResult(
        components=components,
        reward=reward,
        trainable=not hard_risk,
        reason_codes=_unique(reason_codes),
        profile_id=profile.profile_id,
        profile_version=profile.profile_version,
        profile_hash=profile.profile_hash,
        risk_deduplication_applied=dedup_applied,
    )


def _profile_from_payload(payload: Mapping[str, Any]) -> CoverageFirstRewardProfile:
    schema_version = payload.get("schema_version")
    profile_keys = PROFILE_KEYS + (("reward_policy",) if schema_version == SCHEMA_VERSION_V2 else ())
    _validate_keys("profile", payload, profile_keys)
    if schema_version not in {SCHEMA_VERSION, SCHEMA_VERSION_V2}:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION} or {SCHEMA_VERSION_V2}")
    weight_keys = WEIGHT_KEYS_V2 if schema_version == SCHEMA_VERSION_V2 else WEIGHT_KEYS
    normalizer_keys = NORMALIZER_KEYS_V2 if schema_version == SCHEMA_VERSION_V2 else NORMALIZER_KEYS
    component_keys = COMPONENT_KEYS_V2 if schema_version == SCHEMA_VERSION_V2 else COMPONENT_KEYS
    weights = _numeric_section(payload.get("weights"), weight_keys, "weights", strictly_positive=False)
    normalizers = _numeric_section(payload.get("normalizers"), normalizer_keys, "normalizers", strictly_positive=True)
    risk_policy = _risk_policy(payload.get("risk_policy"))
    reward_policy = _reward_policy_v2(payload.get("reward_policy")) if schema_version == SCHEMA_VERSION_V2 else {}
    hard_risk_policy = _hard_risk_policy(payload.get("hard_risk_policy"))
    component_source_map = payload.get("component_source_map")
    if not isinstance(component_source_map, dict):
        raise ValueError("component_source_map must be an object")
    _validate_keys("component_source_map", component_source_map, component_keys)
    profile = CoverageFirstRewardProfile(
        schema_version=str(schema_version),
        profile_id=_required_string(payload, "profile_id"),
        profile_version=_required_string(payload, "profile_version"),
        target_final_coverage_rate=_range_float(payload.get("target_final_coverage_rate"), "target_final_coverage_rate"),
        coverage_cap_rate=_range_float(payload.get("coverage_cap_rate"), "coverage_cap_rate"),
        horizon_steps=_positive_int(payload.get("horizon_steps"), "horizon_steps"),
        mission_budget_route_when_below_target=_required_string(payload, "mission_budget_route_when_below_target"),
        weights=weights,
        normalizers=normalizers,
        risk_policy=risk_policy,
        reward_policy=reward_policy,
        hard_risk_policy=hard_risk_policy,
        component_source_map={str(key): str(value) for key, value in component_source_map.items()},
        component_keys=component_keys,
        profile_hash="",
    )
    return CoverageFirstRewardProfile(**{**profile.__dict__, "profile_hash": coverage_first_profile_hash(profile)})


def _coverage_per_cost_value(metrics: Mapping[str, Any], coverage_delta: float, path_cost: float, path_cost_floor: float) -> float:
    supplied = _finite_float(metrics.get("coverage_per_cost"))
    if supplied is not None:
        return max(0.0, supplied)
    return coverage_delta / max(path_cost, path_cost_floor)


def _cap_path_cost_component(
    path_cost_component: float,
    coverage_gain_component: float,
    progress: float,
    profile: CoverageFirstRewardProfile,
) -> float:
    if progress >= profile.target_final_coverage_rate:
        return path_cost_component
    cap_ratio = float(profile.reward_policy["path_cost_penalty_cap_ratio_when_below_target"])
    cap = abs(coverage_gain_component) * cap_ratio
    if cap <= 0.0:
        return path_cost_component
    return _round(max(path_cost_component, -cap))


def _dedup_soft_risk(
    soft_risk_component: float,
    path_cost_component: float,
    profile: CoverageFirstRewardProfile,
) -> tuple[float, bool]:
    if profile.risk_policy["path_cost_includes_risk_proxy"] is not True:
        return soft_risk_component, False
    ratio = float(profile.risk_policy["max_soft_risk_to_path_cost_penalty_ratio"])
    cap = abs(path_cost_component) * ratio
    if cap <= 0.0:
        cap = ratio
    if abs(soft_risk_component) <= cap:
        return soft_risk_component, True
    return _round(-cap), True


def _hard_risk(metrics: Mapping[str, Any]) -> bool:
    if metrics.get("path_allowed_by_risk") is False:
        return True
    if _truthy(metrics.get("open_grid_fallback_used")):
        return True
    flags = metrics.get("hard_risk_flags")
    if isinstance(flags, (list, tuple)) and len(flags) > 0:
        return True
    count = _finite_float(metrics.get("hard_risk_violation_count"))
    return count is not None and count > 0.0


def _risk_policy(section: Any) -> dict[str, Any]:
    if not isinstance(section, dict):
        raise ValueError("risk_policy must be an object")
    _validate_keys("risk_policy", section, RISK_POLICY_KEYS)
    if not isinstance(section["path_cost_includes_risk_proxy"], bool):
        raise ValueError("risk_policy.path_cost_includes_risk_proxy must be boolean")
    mode = _required_string(section, "soft_risk_component_mode")
    if mode not in {"audit_weighted_tiny", "weighted"}:
        raise ValueError("risk_policy.soft_risk_component_mode must be audit_weighted_tiny or weighted")
    ratio = _finite_float(section.get("max_soft_risk_to_path_cost_penalty_ratio"))
    if ratio is None or ratio < 0.0:
        raise ValueError("risk_policy.max_soft_risk_to_path_cost_penalty_ratio must be finite and >= 0")
    return {
        "path_cost_includes_risk_proxy": section["path_cost_includes_risk_proxy"],
        "soft_risk_component_mode": mode,
        "max_soft_risk_to_path_cost_penalty_ratio": ratio,
    }


def _reward_policy_v2(section: Any) -> dict[str, Any]:
    if not isinstance(section, dict):
        raise ValueError("reward_policy must be an object")
    _validate_keys("reward_policy", section, REWARD_POLICY_KEYS_V2)
    floor = _finite_float(section.get("coverage_per_cost_path_cost_floor"))
    cpc_cap = _finite_float(section.get("coverage_per_cost_component_cap_ratio"))
    path_cap = _finite_float(section.get("path_cost_penalty_cap_ratio_when_below_target"))
    if floor is None or floor <= 0.0:
        raise ValueError("reward_policy.coverage_per_cost_path_cost_floor must be finite and > 0")
    if cpc_cap is None or cpc_cap < 0.0:
        raise ValueError("reward_policy.coverage_per_cost_component_cap_ratio must be finite and >= 0")
    if path_cap is None or path_cap < 0.0:
        raise ValueError("reward_policy.path_cost_penalty_cap_ratio_when_below_target must be finite and >= 0")
    return {
        "coverage_per_cost_path_cost_floor": floor,
        "coverage_per_cost_component_cap_ratio": cpc_cap,
        "path_cost_penalty_cap_ratio_when_below_target": path_cap,
    }


def _hard_risk_policy(section: Any) -> dict[str, Any]:
    if not isinstance(section, dict):
        raise ValueError("hard_risk_policy must be an object")
    _validate_keys("hard_risk_policy", section, HARD_RISK_POLICY_KEYS)
    for key in ("reject_before_reward", "clamp_positive_reward_to_non_positive"):
        if not isinstance(section[key], bool):
            raise ValueError(f"hard_risk_policy.{key} must be boolean")
    floor = _finite_float(section.get("failure_floor"))
    if floor is None or floor > 0.0:
        raise ValueError("hard_risk_policy.failure_floor must be finite and <= 0")
    return {
        "reject_before_reward": section["reject_before_reward"],
        "clamp_positive_reward_to_non_positive": section["clamp_positive_reward_to_non_positive"],
        "failure_floor": floor,
    }


def _numeric_section(section: Any, keys: tuple[str, ...], name: str, *, strictly_positive: bool) -> dict[str, float]:
    if not isinstance(section, dict):
        raise ValueError(f"{name} must be an object")
    _validate_keys(name, section, keys)
    result: dict[str, float] = {}
    for key in keys:
        value = _finite_float(section.get(key))
        if value is None:
            raise ValueError(f"{name}.{key} must be finite")
        if strictly_positive and value <= 0.0:
            raise ValueError(f"{name}.{key} must be > 0")
        if not strictly_positive and value < 0.0:
            raise ValueError(f"{name}.{key} must be >= 0")
        result[key] = value
    return result


def _validate_keys(section_name: str, payload: Mapping[str, Any], allowed: tuple[str, ...]) -> None:
    actual = set(payload)
    allowed_set = set(allowed)
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


def _positive_int(value: Any, key: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{key} must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be an integer") from exc
    if parsed <= 0:
        raise ValueError(f"{key} must be > 0")
    return parsed


def _range_float(value: Any, key: str) -> float:
    parsed = _finite_float(value)
    if parsed is None or parsed <= 0.0 or parsed > 1.0:
        raise ValueError(f"{key} must be in (0, 1]")
    return parsed


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _nonnegative(value: Any) -> float:
    parsed = _finite_float(value)
    return 0.0 if parsed is None else max(0.0, parsed)


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _round(value: float) -> float:
    return round(float(value), 12)


def _unique(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result
