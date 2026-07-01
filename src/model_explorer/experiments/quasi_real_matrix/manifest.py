from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


QUASI_REAL_EVALUATION_SCHEMA_VERSION = "model-explorer-quasi-real-evaluation/v1"


EVALUATION_SCOPE = "quasi-real evaluation; not real-world generalization benchmark"


REQUIRED_ROI_NAMES = (
    "smooth_high_confidence",
    "rim_or_steep_slope",
    "low_observation_count",
    "mixed_risk",
)


VALID_SPLITS = ("train", "validation", "test", "benchmark")


_DEFAULT_SELECTION_COMPOSITE_WEIGHTS = {
    "final_coverage_rate": 1.0,
    "cumulative_coverage_rate_delta": 1.0,
    "value_coverage": 0.25,
    "total_path_cost": -0.05,
    "average_risk": -0.25,
    "failure_count": -1.0,
}


@dataclass(frozen=True)
class RoiSpec:
    name: str
    split: str
    roi_x: int
    roi_y: int
    roi_width: int
    roi_height: int
    seed: int
    candidate_count: int
    episode_count: int
    start_cell: tuple[int, int] = (0, 0)

    @property
    def bounds(self) -> dict[str, int]:
        return {
            "x": self.roi_x,
            "y": self.roi_y,
            "width": self.roi_width,
            "height": self.roi_height,
        }


@dataclass(frozen=True)
class QuasiRealEvaluationManifest:
    path: Path
    name: str
    run_id: str
    dataset_manifest: Path
    output_root: Path
    rois: tuple[RoiSpec, ...]
    mask_stress_config: dict[str, Any]
    selection_config: dict[str, Any]
    dataset_validation: dict[str, Any]
    train_config: dict[str, Any]
    planner_config: dict[str, Any]
    reward_config: dict[str, Any]


def load_quasi_real_evaluation_manifest(path: str | Path) -> QuasiRealEvaluationManifest:
    manifest_path = Path(path).resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("quasi-real evaluation manifest must be a JSON object")
    schema_version = str(payload.get("schema_version", QUASI_REAL_EVALUATION_SCHEMA_VERSION))
    if schema_version != QUASI_REAL_EVALUATION_SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {QUASI_REAL_EVALUATION_SCHEMA_VERSION!r}")
    base_dir = manifest_path.parent
    seed = int(payload.get("seed", 0))
    candidate_count = int(payload.get("candidate_count", 8))
    episode_count = int(payload.get("episode_count", 1))
    rois = _roi_specs(
        payload.get("rois"),
        seed=seed,
        candidate_count=candidate_count,
        episode_count=episode_count,
    )
    _validate_roi_coverage(rois)
    return QuasiRealEvaluationManifest(
        path=manifest_path,
        name=str(payload.get("name", manifest_path.stem)),
        run_id=str(payload.get("run_id", "default")),
        dataset_manifest=_resolve_path(base_dir, payload.get("dataset_manifest")),
        output_root=_resolve_path(base_dir, payload.get("output_root", "data/processed/quasi_real/evaluation-matrix")),
        rois=rois,
        mask_stress_config=_normalize_mask_stress_config(payload.get("mask_stress", {})),
        selection_config=_normalize_selection_config(payload.get("selection", {})),
        dataset_validation=dict(payload.get("dataset_validation", {})),
        train_config=dict(payload.get("train", _default_train_config(seed))),
        planner_config=dict(payload.get("planner", {"backend": "contract_cost"})),
        reward_config=dict(payload.get("reward", _default_reward_config())),
    )


def _roi_specs(
    value: Any,
    *,
    seed: int,
    candidate_count: int,
    episode_count: int,
) -> tuple[RoiSpec, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("rois must be a non-empty list")
    specs: list[RoiSpec] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(f"rois[{index}] must be an object")
        split = str(item.get("split", "train"))
        if split not in VALID_SPLITS:
            raise ValueError(f"rois[{index}].split must be one of {', '.join(VALID_SPLITS)}")
        width = int(item.get("roi_width", 0))
        height = int(item.get("roi_height", 0))
        if width <= 0 or height <= 0:
            raise ValueError(f"rois[{index}] ROI width and height must be positive")
        specs.append(
            RoiSpec(
                name=str(item.get("name", f"roi-{index}")),
                split=split,
                roi_x=int(item.get("roi_x", 0)),
                roi_y=int(item.get("roi_y", 0)),
                roi_width=width,
                roi_height=height,
                seed=int(item.get("seed", seed + index)),
                candidate_count=int(item.get("candidate_count", candidate_count)),
                episode_count=int(item.get("episode_count", episode_count)),
                start_cell=_start_cell(item.get("start_cell"), width=width, height=height),
            )
        )
    return tuple(specs)


def _start_cell(value: Any, *, width: int, height: int) -> tuple[int, int]:
    if value is None:
        return (0, 0)
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 2
    ):
        raise ValueError("roi.start_cell must be a two-item [x, y] cell")
    x = int(value[0])
    y = int(value[1])
    if x < 0 or y < 0 or x >= width or y >= height:
        raise ValueError("roi.start_cell must be inside the ROI bounds")
    return (x, y)


def _validate_roi_coverage(rois: tuple[RoiSpec, ...]) -> None:
    names = {roi.name for roi in rois}
    missing = [name for name in REQUIRED_ROI_NAMES if name not in names]
    if missing:
        raise ValueError(f"quasi-real evaluation manifest is missing required ROI: {missing[0]}")


def _normalize_mask_stress_config(value: Any) -> dict[str, Any]:
    if value is None:
        value = {}
    if not isinstance(value, dict):
        raise ValueError("mask_stress must be an object")
    enabled = bool(value.get("enabled", False))
    fields = value.get("missing_experimental_fields", ())
    if fields is None:
        fields = ()
    if not isinstance(fields, (list, tuple)):
        raise ValueError("mask_stress.missing_experimental_fields must be a list")
    return {
        "enabled": enabled,
        "label": str(value.get("label", "mask_stress_augmented")),
        "profile": str(value.get("profile", "deterministic-v1")),
        "unreachable_candidate_count": max(0, int(value.get("unreachable_candidate_count", 1))),
        "missing_experimental_fields": tuple(str(field) for field in fields),
    }


def _mask_stress_enabled(config: dict[str, Any]) -> bool:
    return bool(config.get("enabled", False))


def _mask_stress_summary(config: dict[str, Any]) -> dict[str, Any]:
    normalized = _normalize_mask_stress_config(config)
    return {
        "enabled": bool(normalized["enabled"]),
        "label": str(normalized["label"]),
        "profile": str(normalized["profile"]),
        "unreachable_candidate_count": int(normalized["unreachable_candidate_count"]),
        "missing_experimental_fields": list(normalized["missing_experimental_fields"]),
    }


def _mask_stress_metadata(config: dict[str, Any]) -> dict[str, Any]:
    normalized = _normalize_mask_stress_config(config)
    return {
        "mask_stress_augmented": True,
        "mask_stress_label": str(normalized["label"]),
        "mask_stress_profile": str(normalized["profile"]),
        "evaluation_scope": EVALUATION_SCOPE,
    }


def _normalize_selection_config(value: Any) -> dict[str, Any]:
    if value is None:
        value = {}
    if not isinstance(value, dict):
        raise ValueError("selection must be an object")
    mode = str(value.get("mode", "max"))
    if mode not in {"max", "min"}:
        raise ValueError("selection.mode must be 'max' or 'min'")
    composite_weights = dict(_DEFAULT_SELECTION_COMPOSITE_WEIGHTS)
    raw_composite_weights = value.get("composite_weights", {})
    if raw_composite_weights is not None:
        if not isinstance(raw_composite_weights, dict):
            raise ValueError("selection.composite_weights must be an object")
        for metric, weight in raw_composite_weights.items():
            composite_weights[str(metric)] = float(weight)
    return {
        "enabled": bool(value.get("enabled", False)),
        "metric": str(value.get("metric", "torch_policy.final_coverage_rate")),
        "mode": mode,
        "min_seed_count": _optional_selection_int(value, "min_seed_count"),
        "min_architecture_count": _optional_selection_int(value, "min_architecture_count"),
        "min_roi_group_count": _optional_selection_int(value, "min_roi_group_count"),
        "min_unreachable_candidate_count": _optional_selection_int(value, "min_unreachable_candidate_count"),
        "min_mask_stress_sample_count": _optional_selection_int(value, "min_mask_stress_sample_count"),
        "uncertainty_multiplier": float(value.get("uncertainty_multiplier", 1.0)),
        "composite_weights": composite_weights,
    }


def _optional_selection_int(value: dict[str, Any], key: str) -> int | None:
    return None if value.get(key) is None else int(value[key])


def _selection_config_summary(config: dict[str, Any]) -> dict[str, Any]:
    normalized = _normalize_selection_config(config)
    return {key: value for key, value in normalized.items() if value is not None}


def _roi_summary(roi: RoiSpec) -> dict[str, Any]:
    return {
        "name": roi.name,
        "split": roi.split,
        "bounds": roi.bounds,
        "seed": roi.seed,
        "candidate_count": roi.candidate_count,
        "episode_count": roi.episode_count,
    }


def _split_counts(rois: tuple[RoiSpec, ...]) -> dict[str, int]:
    counts = {split: 0 for split in VALID_SPLITS}
    for roi in rois:
        counts[roi.split] += roi.episode_count
    return {split: count for split, count in counts.items() if count}


def _product_file_name(manifest: dict[str, Any], *, role: str) -> str:
    products = manifest.get("products", [])
    if not isinstance(products, list):
        raise ValueError("manifest.products must be a list")
    for product in products:
        if not isinstance(product, dict) or product.get("role") != role:
            continue
        for file_info in product.get("files", []):
            if isinstance(file_info, dict) and str(file_info.get("name", "")).lower().endswith(".jp2"):
                return str(file_info["name"])
    raise ValueError(f"manifest is missing product file for role {role!r}")


def _manifest_resolution(manifest: dict[str, Any]) -> float:
    projection = manifest.get("projection", {})
    if isinstance(projection, dict) and projection.get("map_scale_meters_per_pixel") is not None:
        return float(projection["map_scale_meters_per_pixel"])
    return 1.0


def _resolve_path(base_dir: Path, value: Any) -> Path:
    if value is None:
        raise ValueError("path value is required")
    path = Path(str(value))
    return path.resolve() if path.is_absolute() else (base_dir / path).resolve()


def _default_train_config(seed: int) -> dict[str, Any]:
    return {
        "seed": int(seed),
        "architectures": ["mlp_v1", "mlp_missing_v1", "candidate_attention_v1"],
        "hidden_size": 16,
        "learning_rate": 1.0e-3,
        "epochs": 1,
        "evaluate_trained_policy": True,
    }


def _default_reward_config() -> dict[str, float]:
    return {
        "path_cost_weight": 0.1,
        "path_cost_normalizer": 1000.0,
        "risk_weight": 0.2,
        "failure_penalty": 1.0,
    }



mask_stress_enabled = _mask_stress_enabled
mask_stress_summary = _mask_stress_summary
mask_stress_metadata = _mask_stress_metadata
normalize_mask_stress_config = _normalize_mask_stress_config
normalize_selection_config = _normalize_selection_config
selection_config_summary = _selection_config_summary
roi_summary = _roi_summary
split_counts = _split_counts
product_file_name = _product_file_name
manifest_resolution = _manifest_resolution
resolve_path = _resolve_path

_PUBLIC_EXPORTS = [
    "QUASI_REAL_EVALUATION_SCHEMA_VERSION",
    "EVALUATION_SCOPE",
    "REQUIRED_ROI_NAMES",
    "VALID_SPLITS",
    "RoiSpec",
    "QuasiRealEvaluationManifest",
    "load_quasi_real_evaluation_manifest",
    "mask_stress_enabled",
    "mask_stress_summary",
    "mask_stress_metadata",
    "normalize_mask_stress_config",
    "normalize_selection_config",
    "selection_config_summary",
    "roi_summary",
    "split_counts",
    "product_file_name",
    "manifest_resolution",
    "resolve_path",
]
_PRIVATE_COMPAT_EXPORTS = [
    "_roi_specs",
    "_start_cell",
    "_validate_roi_coverage",
    "_normalize_mask_stress_config",
    "_mask_stress_enabled",
    "_mask_stress_summary",
    "_mask_stress_metadata",
    "_normalize_selection_config",
    "_optional_selection_int",
    "_selection_config_summary",
    "_roi_summary",
    "_split_counts",
    "_product_file_name",
    "_manifest_resolution",
    "_resolve_path",
    "_default_train_config",
    "_default_reward_config",
]
__all__ = [*_PUBLIC_EXPORTS, *_PRIVATE_COMPAT_EXPORTS]
