from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ...data.manifest import load_data_manifest, validate_data_manifest
from .manifest import QuasiRealEvaluationManifest, RoiSpec, load_quasi_real_evaluation_manifest
from .reports import inspection_summary, render_quasi_real_matrix_markdown, run_summary, would_write
from .scenario_generation import experiment_manifest_payload, generate_quasi_real_scenarios

__all__ = [
    "RoiSpec",
    "QuasiRealEvaluationManifest",
    "load_quasi_real_evaluation_manifest",
    "validate_quasi_real_evaluation_manifest",
    "dry_run_quasi_real_evaluation_manifest",
    "run_quasi_real_evaluation_manifest",
]


def validate_quasi_real_evaluation_manifest(path: str | Path) -> dict[str, Any]:
    manifest = load_quasi_real_evaluation_manifest(path)
    data_validation = validate_data_manifest(manifest.dataset_manifest)
    return inspection_summary(manifest, status="valid", data_validation=data_validation.to_dict())


def dry_run_quasi_real_evaluation_manifest(path: str | Path) -> dict[str, Any]:
    manifest = load_quasi_real_evaluation_manifest(path)
    data_validation = validate_data_manifest(manifest.dataset_manifest)
    summary = inspection_summary(manifest, status="dry_run", data_validation=data_validation.to_dict())
    summary["would_write"] = would_write(manifest)
    return summary


def run_quasi_real_evaluation_manifest(path: str | Path) -> dict[str, Any]:
    from ..runner import run_experiment_manifest

    manifest = load_quasi_real_evaluation_manifest(path)
    data_validation = validate_data_manifest(manifest.dataset_manifest)
    data_validation.require_valid()
    data_manifest = load_data_manifest(manifest.dataset_manifest)
    raw_dir = Path(data_validation.raw_dir)

    manifest.output_root.mkdir(parents=True, exist_ok=True)
    split_groups, generated_scenarios = generate_quasi_real_scenarios(
        manifest,
        data_manifest=data_manifest,
        raw_dir=raw_dir,
    )

    experiment_manifest_path = manifest.output_root / "experiment.json"
    experiment_payload = experiment_manifest_payload(manifest, split_groups)
    experiment_manifest_path.write_text(
        json.dumps(experiment_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    experiment_summary = run_experiment_manifest(experiment_manifest_path)

    summary = run_summary(
        manifest,
        data_manifest=data_manifest,
        data_validation=data_validation.to_dict(),
        generated_scenarios=generated_scenarios,
        experiment_manifest_path=experiment_manifest_path,
        experiment_summary=experiment_summary,
    )
    summary_path = manifest.output_root / "summary.json"
    report_path = manifest.output_root / "matrix-report.md"
    summary["summary_output"] = str(summary_path)
    summary["report_output"] = str(report_path)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    report_path.write_text(render_quasi_real_matrix_markdown(summary), encoding="utf-8")
    return summary
