from __future__ import annotations

from typing import Any


def _training_seeds(config: dict[str, Any]) -> tuple[int, ...]:
    if "seeds" not in config:
        return (int(config.get("seed", 0)),)
    raw_seeds = config["seeds"]
    if not isinstance(raw_seeds, list) or not raw_seeds:
        raise ValueError("train.seeds must be a non-empty list")
    return tuple(int(seed) for seed in raw_seeds)


def _training_source_selection_strategies(config: dict[str, Any]) -> tuple[str | None, ...]:
    raw_value = config.get("source_selection_strategies")
    if raw_value is None:
        raw_value = config.get("selection_strategies")
    if raw_value is None:
        return (None,)
    if not isinstance(raw_value, list) or not raw_value:
        raise ValueError("train.source_selection_strategies must be a non-empty list")
    strategies = tuple(str(value).strip() for value in raw_value)
    if any(not value for value in strategies):
        raise ValueError("train.source_selection_strategies entries must be non-empty")
    return strategies


def _training_teacher_imitation_weights(config: dict[str, Any]) -> tuple[float, ...]:
    raw_value = config.get("teacher_imitation_weights")
    if raw_value is None:
        return (max(0.0, float(config.get("teacher_imitation_weight", 0.0))),)
    if not isinstance(raw_value, list) or not raw_value:
        raise ValueError("train.teacher_imitation_weights must be a non-empty list")
    return tuple(max(0.0, float(value)) for value in raw_value)


def _training_teacher_margin_curriculum_profiles(config: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    raw_value = config.get("teacher_margin_curriculum_profiles")
    if raw_value is None:
        weighting = config.get("teacher_margin_weighting")
        profile_name = "default"
        if isinstance(weighting, dict):
            profile_name = str(weighting.get("profile_name", "custom")).strip() or "custom"
        return ({"name": profile_name, "teacher_margin_weighting": weighting},)
    if isinstance(raw_value, dict):
        raw_entries = [
            {"name": name, **(value if isinstance(value, dict) else {"bucket_weights": value})}
            for name, value in raw_value.items()
        ]
    else:
        if not isinstance(raw_value, list) or not raw_value:
            raise ValueError("train.teacher_margin_curriculum_profiles must be a non-empty list or mapping")
        raw_entries = list(raw_value)
    profiles: list[dict[str, Any]] = []
    for entry in raw_entries:
        profiles.append(_coerce_teacher_margin_curriculum_profile(entry))
    names = [profile["name"] for profile in profiles]
    if len(set(names)) != len(names):
        raise ValueError("train.teacher_margin_curriculum_profiles names must be unique")
    return tuple(profiles)


def _coerce_teacher_margin_curriculum_profile(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        return _builtin_teacher_margin_curriculum_profile(value)
    if not isinstance(value, dict):
        raise ValueError("teacher margin curriculum profile must be a name or object")
    name = str(value.get("name", "")).strip()
    if not name:
        raise ValueError("teacher margin curriculum profile requires a non-empty name")
    if "teacher_margin_weighting" in value:
        weighting = value["teacher_margin_weighting"]
        if weighting is not None and not isinstance(weighting, dict):
            raise ValueError("teacher_margin_curriculum_profile.teacher_margin_weighting must be an object")
        weighting_config = dict(weighting or {})
    else:
        weighting_config = {
            key: value[key]
            for key in ("bucket_weights", "teacher_low_margin_threshold", "teacher_high_margin_threshold")
            if key in value
        }
    weighting_config["profile_name"] = name
    return {"name": name, "teacher_margin_weighting": weighting_config}


def _builtin_teacher_margin_curriculum_profile(name: str) -> dict[str, Any]:
    profile_name = str(name).strip()
    profiles = {
        "high_only": {"high": 1.0, "medium": 0.0, "low": 0.0, "missing": 0.0},
        "high_medium": {"high": 1.0, "medium": 0.5, "low": 0.0, "missing": 0.0},
        "soft_all_valid": {"high": 1.0, "medium": 0.5, "low": 0.1, "missing": 0.0},
    }
    if profile_name not in profiles:
        raise ValueError(f"unknown teacher margin curriculum profile: {profile_name}")
    return {
        "name": profile_name,
        "teacher_margin_weighting": {
            "profile_name": profile_name,
            "bucket_weights": dict(profiles[profile_name]),
        },
    }


def _training_architectures(config: dict[str, Any]) -> tuple[str | None, ...]:
    if "architectures" not in config:
        return (config.get("architecture"),)
    raw_architectures = config["architectures"]
    if not isinstance(raw_architectures, list) or not raw_architectures:
        raise ValueError("train.architectures must be a non-empty list")
    return tuple(str(architecture) for architecture in raw_architectures)


def _training_architecture_config(config: dict[str, Any], architecture: str) -> dict[str, Any] | None:
    base_config = config.get("architecture_config")
    architecture_configs = config.get("architecture_configs")
    selected_config = base_config
    if architecture_configs is not None:
        if not isinstance(architecture_configs, dict):
            raise ValueError("train.architecture_configs must be a mapping of architecture name to config")
        selected_config = architecture_configs.get(architecture, base_config)
    if selected_config is None:
        return None
    if not isinstance(selected_config, dict):
        raise ValueError("train.architecture_config must be a mapping")
    return dict(selected_config)


def _normalize_training_architecture_name(value: str | None) -> str:
    return "mlp_v1" if value is None or str(value).strip() == "" else str(value)


training_architectures = _training_architectures
training_seeds = _training_seeds
training_source_selection_strategies = _training_source_selection_strategies
training_teacher_imitation_weights = _training_teacher_imitation_weights
training_teacher_margin_curriculum_profiles = _training_teacher_margin_curriculum_profiles

__all__ = (
    "_training_seeds",
    "_training_source_selection_strategies",
    "_training_teacher_imitation_weights",
    "_training_teacher_margin_curriculum_profiles",
    "_coerce_teacher_margin_curriculum_profile",
    "_builtin_teacher_margin_curriculum_profile",
    "_training_architectures",
    "_training_architecture_config",
    "_normalize_training_architecture_name",
    "training_architectures",
    "training_seeds",
    "training_source_selection_strategies",
    "training_teacher_imitation_weights",
    "training_teacher_margin_curriculum_profiles",
)
