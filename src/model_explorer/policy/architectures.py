from __future__ import annotations

from typing import Any

from .features import PolicyObservation
from .torch_policy import (
    CandidateAttentionPolicyNetwork,
    MaskedCandidatePolicyNetwork,
    MissingIndicatorCandidatePolicyNetwork,
)


MLP_V1 = "mlp_v1"
MLP_MISSING_V1 = "mlp_missing_v1"
CANDIDATE_ATTENTION_V1 = "candidate_attention_v1"

SUPPORTED_ARCHITECTURES = (MLP_V1, MLP_MISSING_V1, CANDIDATE_ATTENTION_V1)
_COMMON_CONFIG_FIELDS = frozenset(("hidden_dim", "dropout"))
_CONFIG_FIELDS_BY_ARCHITECTURE = {
    MLP_V1: _COMMON_CONFIG_FIELDS,
    MLP_MISSING_V1: _COMMON_CONFIG_FIELDS,
    CANDIDATE_ATTENTION_V1: _COMMON_CONFIG_FIELDS | frozenset(("attention_heads",)),
}


def normalize_architecture_name(value: str | None) -> str:
    return MLP_V1 if value is None or str(value).strip() == "" else str(value)


def build_policy_network(
    architecture: str | None,
    *,
    observation: PolicyObservation,
    hidden_size: int = 64,
    architecture_config: dict[str, Any] | None = None,
):
    return build_policy_network_from_metadata(
        architecture,
        candidate_feature_count=len(observation.candidate_feature_names),
        global_feature_count=len(observation.global_feature_names),
        missing_indicator_count=len(observation.candidate_missing_indicator_names),
        hidden_size=hidden_size,
        architecture_config=architecture_config,
    )


def build_policy_network_from_metadata(
    architecture: str | None,
    *,
    candidate_feature_count: int,
    global_feature_count: int,
    missing_indicator_count: int = 0,
    hidden_size: int = 64,
    architecture_config: dict[str, Any] | None = None,
):
    name = normalize_architecture_name(architecture)
    config = normalize_architecture_config(
        name,
        hidden_size=hidden_size,
        architecture_config=architecture_config,
    )
    if name == MLP_V1:
        return MaskedCandidatePolicyNetwork(
            candidate_feature_count=candidate_feature_count,
            global_feature_count=global_feature_count,
            hidden_size=config["hidden_dim"],
            missing_indicator_count=missing_indicator_count,
            dropout=config["dropout"],
            architecture_config=config,
        )
    if name == MLP_MISSING_V1:
        return MissingIndicatorCandidatePolicyNetwork(
            candidate_feature_count=candidate_feature_count,
            global_feature_count=global_feature_count,
            hidden_size=config["hidden_dim"],
            missing_indicator_count=missing_indicator_count,
            dropout=config["dropout"],
            architecture_config=config,
        )
    if name == CANDIDATE_ATTENTION_V1:
        return CandidateAttentionPolicyNetwork(
            candidate_feature_count=candidate_feature_count,
            global_feature_count=global_feature_count,
            hidden_size=config["hidden_dim"],
            missing_indicator_count=missing_indicator_count,
            dropout=config["dropout"],
            attention_heads=config["attention_heads"],
            architecture_config=config,
        )
    raise ValueError(
        f"unknown architecture {name!r}; supported architectures: {', '.join(SUPPORTED_ARCHITECTURES)}"
    )


def normalize_architecture_config(
    architecture: str | None,
    *,
    hidden_size: int = 64,
    architecture_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    name = normalize_architecture_name(architecture)
    if name not in SUPPORTED_ARCHITECTURES:
        raise ValueError(
            f"unknown architecture {name!r}; supported architectures: {', '.join(SUPPORTED_ARCHITECTURES)}"
        )
    if architecture_config is None:
        raw_config: dict[str, Any] = {}
    elif isinstance(architecture_config, dict):
        raw_config = dict(architecture_config)
    else:
        raise ValueError("architecture_config must be a mapping")

    allowed_fields = _CONFIG_FIELDS_BY_ARCHITECTURE[name]
    unknown_fields = sorted(set(raw_config) - allowed_fields)
    if unknown_fields:
        fields = ", ".join(unknown_fields)
        allowed = ", ".join(sorted(allowed_fields))
        raise ValueError(f"unknown architecture config field(s) for {name}: {fields}; allowed fields: {allowed}")

    hidden_dim = _positive_int(raw_config.get("hidden_dim", hidden_size), field_name="hidden_dim")
    dropout = _unit_interval_float(raw_config.get("dropout", 0.0), field_name="dropout")
    config: dict[str, Any] = {
        "hidden_dim": hidden_dim,
        "dropout": dropout,
    }
    if name == CANDIDATE_ATTENTION_V1:
        attention_heads = _positive_int(raw_config.get("attention_heads", 1), field_name="attention_heads")
        if hidden_dim % attention_heads != 0:
            raise ValueError("attention_heads must divide hidden_dim for candidate_attention_v1")
        config["attention_heads"] = attention_heads
    return config


def _positive_int(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a positive integer")
    try:
        numeric = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a positive integer") from exc
    if numeric <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return numeric


def _unit_interval_float(value: Any, *, field_name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be in [0.0, 1.0)")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be in [0.0, 1.0)") from exc
    if numeric < 0.0 or numeric >= 1.0:
        raise ValueError(f"{field_name} must be in [0.0, 1.0)")
    return numeric
