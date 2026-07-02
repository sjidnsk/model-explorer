from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .manifest import _ensure_parent_dir, _resolve_path, _write_json
from .selection import (
    _normalize_curriculum_profile_name,
    _normalize_teacher_weight_name,
    _normalize_training_source_name,
)


def _training_output_path(
    config: dict[str, Any],
    key: str,
    *,
    seed: int,
    architecture: str,
    selection_strategy: str | None,
    teacher_imitation_weight: float,
    curriculum_profile: str,
    base_dir: Path,
    run_output_dir: Path | None,
    default_name: str,
    multi_seed: bool,
    multi_architecture: bool,
    multi_source: bool,
    multi_teacher_weight: bool,
    multi_curriculum_profile: bool,
    required: bool,
) -> Path | None:
    source_name = _normalize_training_source_name(selection_strategy)
    teacher_weight_name = _normalize_teacher_weight_name(teacher_imitation_weight)
    curriculum_profile_name = _normalize_curriculum_profile_name(curriculum_profile)
    value = config.get(key)
    if value is not None:
        text = str(value)
        formatted = text.format(
            seed=seed,
            architecture=architecture,
            selection_strategy=source_name,
            teacher_imitation_weight=teacher_weight_name,
            teacher_weight=teacher_weight_name,
            teacher_margin_curriculum_profile=curriculum_profile_name,
            curriculum_profile=curriculum_profile_name,
        )
        path = _resolve_path(base_dir, formatted)
        parent = path.parent
        if multi_source and not _path_parent_contains_placeholder(text, "{selection_strategy}"):
            parent = parent / f"source-{source_name}"
        if multi_teacher_weight and not _path_parent_contains_any_placeholder(
            text,
            ("{teacher_imitation_weight}", "{teacher_weight}"),
        ):
            parent = parent / f"teacher-weight-{teacher_weight_name}"
        if multi_curriculum_profile and not _path_parent_contains_any_placeholder(
            text,
            ("{teacher_margin_curriculum_profile}", "{curriculum_profile}"),
        ):
            parent = parent / f"curriculum-profile-{curriculum_profile_name}"
        if multi_architecture and not _path_parent_contains_placeholder(text, "{architecture}"):
            parent = parent / architecture
        if (multi_seed or multi_architecture) and not _path_parent_contains_placeholder(text, "{seed}"):
            parent = parent / f"seed-{seed}"
        path = parent / path.name
        return path
    if run_output_dir is not None:
        parent = run_output_dir
        if multi_source:
            parent = parent / f"source-{source_name}"
        if multi_teacher_weight:
            parent = parent / f"teacher-weight-{teacher_weight_name}"
        if multi_curriculum_profile:
            parent = parent / f"curriculum-profile-{curriculum_profile_name}"
        if multi_architecture:
            parent = parent / architecture
        return parent / f"seed-{seed}" / default_name
    if required:
        raise ValueError(f"train.{key} is required when outputs.root is not configured")
    return None


def _write_training_loss_log(loss_log: Path | None, result: dict[str, Any]) -> None:
    if loss_log is None:
        return
    _ensure_parent_dir(loss_log)
    loss_records = result.get("epoch_losses", [])
    if not isinstance(loss_records, list) or not loss_records:
        loss_records = [result]
    loss_log.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in loss_records),
        encoding="utf-8",
    )


def _write_training_evaluation_output(checkpoint: Path, name: str, evaluation: dict[str, Any]) -> Path:
    output = checkpoint.parent / name
    _write_json(output, evaluation)
    return output


def _write_training_summary(checkpoint: Path, result: dict[str, Any]) -> Path:
    output = checkpoint.parent / "training-summary.json"
    _write_json(output, result)
    return output


def _write_system_calibration_summary_output(
    base_dir: Path,
    config: dict[str, Any],
    summary: dict[str, Any],
) -> Path | None:
    summary_output = config.get("summary_output")
    if summary_output is None:
        return None
    summary_output_path = _resolve_path(base_dir, summary_output)
    _write_json(summary_output_path, summary)
    return summary_output_path


def _path_parent_contains_placeholder(path_text: str, placeholder: str) -> bool:
    return any(placeholder in part for part in Path(path_text).parent.parts)


def _path_parent_contains_any_placeholder(path_text: str, placeholders: tuple[str, ...]) -> bool:
    return any(_path_parent_contains_placeholder(path_text, placeholder) for placeholder in placeholders)


training_output_path = _training_output_path

__all__ = (
    "_training_output_path",
    "_write_training_loss_log",
    "_write_training_evaluation_output",
    "_write_training_summary",
    "_write_system_calibration_summary_output",
    "_path_parent_contains_placeholder",
    "_path_parent_contains_any_placeholder",
    "training_output_path",
)
