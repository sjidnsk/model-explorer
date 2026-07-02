from __future__ import annotations


def test_quasi_real_report_sections_lock_key_headings_and_fields() -> None:
    from model_explorer.experiments.quasi_real_matrix.report_sections import (
        render_architecture_selection_section,
        render_dataset_quality_section,
        render_sample_discriminativeness_section,
        render_selection_composite_section,
    )

    summary = {"evaluation_scope": "quasi-real/v1"}
    dataset_summary = {
        "episode_count": 1,
        "transition_count": 2,
        "mask_stress_sample_count": 4,
    }
    selection = {
        "enabled": True,
        "status": "passed",
        "decision": "select",
        "recommended_architecture": "mlp_v1",
        "selection_composite_weights": {"coverage": 1.0},
        "architectures": {
            "mlp_v1": {"selection_composite_score": {"mean": 0.75, "std": 0.1}},
        },
        "sample_discriminativeness": {
            "status": "passed",
            "metrics": {"coverage_gap": {"mean": 0.1, "std": 0.0, "min": 0.1, "max": 0.1, "count": 2}},
        },
    }

    rendered = "\n".join(
        [
            *render_dataset_quality_section(dataset_summary),
            *render_architecture_selection_section(selection),
            *render_selection_composite_section(selection),
            *render_sample_discriminativeness_section(selection),
        ]
    )

    assert "## Dataset Quality" in rendered
    assert "## Architecture Selection Gate" in rendered
    assert "## Sample Discriminativeness" in rendered
    assert "selection_composite_score" in rendered
    assert "| mask_stress_sample_count | 4 |" in rendered


def test_quasi_real_report_sections_use_explicit_exports() -> None:
    import model_explorer.experiments.quasi_real_matrix.report_sections as report_sections

    assert isinstance(report_sections.__all__, tuple)
    assert "render_architecture_selection_section" in report_sections.__all__
    assert "render_sample_discriminativeness_section" in report_sections.__all__
