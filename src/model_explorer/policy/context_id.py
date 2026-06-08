from __future__ import annotations

import hashlib
import json
from typing import Any


POLICY_CONTEXT_ID_SCHEMA_VERSION = "policy-context-id/v1"
POLICY_CONTEXT_ID_SOURCE = "stable_semantic_fields"

POLICY_CONTEXT_ID_FIELDS = (
    "scenario_id",
    "scenario_group",
    "scenario_seed",
    "scenario_variant_id",
    "diagnostic_profile",
    "planning_backend",
    "top_k",
    "sample_type",
    "candidate_role",
    "source_action_index",
    "policy_target_cell",
    "execution_goal_cell",
    "target_binding_mode",
)


def build_policy_context_id(payload: dict[str, Any]) -> str | None:
    fields, missing = policy_context_id_fields(payload)
    if missing:
        return None
    encoded = json.dumps(fields, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def policy_context_id_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    fields, missing = policy_context_id_fields(payload)
    context_id = None if missing else build_policy_context_id(fields)
    return {
        "context_id": context_id,
        "context_id_schema_version": POLICY_CONTEXT_ID_SCHEMA_VERSION,
        "context_id_source": POLICY_CONTEXT_ID_SOURCE,
        "legacy_identity_fallback_used": False,
        "context_id_missing": bool(missing),
        "missing_context_id_fields": missing,
    }


def policy_context_id_fields(payload: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    fields = {
        "scenario_id": _string_or_none(payload.get("scenario_id")),
        "scenario_group": _string_or_none(payload.get("scenario_group")),
        "scenario_seed": _scalar(payload.get("scenario_seed")),
        "scenario_variant_id": _string_or_none(payload.get("scenario_variant_id")),
        "diagnostic_profile": _string_or_none(payload.get("diagnostic_profile")),
        "planning_backend": _planning_backend(payload.get("planning_backend")),
        "top_k": _int_or_none(payload.get("top_k")),
        "sample_type": _string_or_none(payload.get("sample_type") or payload.get("candidate_role")),
        "candidate_role": _string_or_none(payload.get("candidate_role") or payload.get("sample_type")),
        "source_action_index": _int_or_none(payload.get("source_action_index")),
        "policy_target_cell": _cell(payload.get("policy_target_cell")),
        "execution_goal_cell": _cell(payload.get("execution_goal_cell")),
        "target_binding_mode": _string_or_none(payload.get("target_binding_mode")),
    }
    missing = [
        field
        for field in POLICY_CONTEXT_ID_FIELDS
        if fields.get(field) in (None, "", [])
    ]
    return fields, missing


def _planning_backend(value: Any) -> str | None:
    if isinstance(value, dict):
        for key in ("backend", "name", "source"):
            parsed = _string_or_none(value.get(key))
            if parsed:
                return parsed
        return None
    return _string_or_none(value)


def _scalar(value: Any) -> int | float | str | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else value
    parsed = _string_or_none(value)
    if parsed is None:
        return None
    try:
        return int(parsed)
    except ValueError:
        return parsed


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    parsed = str(value)
    return parsed if parsed else None


def _cell(value: Any) -> list[int] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        return [int(value[0]), int(value[1])]
    except (TypeError, ValueError):
        return None
