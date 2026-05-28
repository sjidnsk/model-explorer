from __future__ import annotations

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


def normalize_architecture_name(value: str | None) -> str:
    return MLP_V1 if value is None or str(value).strip() == "" else str(value)


def build_policy_network(
    architecture: str | None,
    *,
    observation: PolicyObservation,
    hidden_size: int = 64,
):
    return build_policy_network_from_metadata(
        architecture,
        candidate_feature_count=len(observation.candidate_feature_names),
        global_feature_count=len(observation.global_feature_names),
        missing_indicator_count=len(observation.candidate_missing_indicator_names),
        hidden_size=hidden_size,
    )


def build_policy_network_from_metadata(
    architecture: str | None,
    *,
    candidate_feature_count: int,
    global_feature_count: int,
    missing_indicator_count: int = 0,
    hidden_size: int = 64,
):
    name = normalize_architecture_name(architecture)
    if name == MLP_V1:
        return MaskedCandidatePolicyNetwork(
            candidate_feature_count=candidate_feature_count,
            global_feature_count=global_feature_count,
            hidden_size=hidden_size,
            missing_indicator_count=missing_indicator_count,
        )
    if name == MLP_MISSING_V1:
        return MissingIndicatorCandidatePolicyNetwork(
            candidate_feature_count=candidate_feature_count,
            global_feature_count=global_feature_count,
            hidden_size=hidden_size,
            missing_indicator_count=missing_indicator_count,
        )
    if name == CANDIDATE_ATTENTION_V1:
        return CandidateAttentionPolicyNetwork(
            candidate_feature_count=candidate_feature_count,
            global_feature_count=global_feature_count,
            hidden_size=hidden_size,
            missing_indicator_count=missing_indicator_count,
        )
    raise ValueError(
        f"unknown architecture {name!r}; supported architectures: {', '.join(SUPPORTED_ARCHITECTURES)}"
    )
