from __future__ import annotations


def test_experiment_report_sections_lock_key_headings_and_fields() -> None:
    from model_explorer.experiments.report_sections import (
        render_dataset_summary_section,
        render_header_section,
    )

    summary = {
        "schema_version": "experiment/v1",
        "planner": "astar",
        "scenario_count": 1,
        "transition_count": 2,
        "dataset_summary": {
            "data_class": "synthetic",
            "dataset_id": "dataset-1",
            "mask_stress_sample_count": 3,
        },
    }

    rendered = "\n".join(
        [
            *render_header_section(summary),
            *render_dataset_summary_section(summary["dataset_summary"]),
        ]
    )

    assert "# Model Explorer Experiment Report" in rendered
    assert "## Dataset Summary" in rendered
    assert "| mask_stress_sample_count | 3 |" in rendered


def test_experiment_report_sections_use_explicit_exports() -> None:
    import model_explorer.experiments.report_sections as report_sections

    assert isinstance(report_sections.__all__, tuple)
    assert "render_header_section" in report_sections.__all__
    assert "render_dataset_summary_section" in report_sections.__all__
