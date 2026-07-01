from __future__ import annotations

CANDIDATE_BENEFIT_FIELDS = (
    "expected_coverage_rate_delta",
    "expected_new_coverage_area",
    "information_gain",
    "confidence_gain",
    "value",
)

CANDIDATE_COST_FIELDS = ("risk", "path_cost", "energy_cost")

CANDIDATE_FEATURE_NAMES = (
    "cell_x",
    "cell_y",
    "relative_dx",
    "relative_dy",
    "relative_distance",
    "utility",
    "reachable",
    *CANDIDATE_BENEFIT_FIELDS,
    *CANDIDATE_COST_FIELDS,
)

GLOBAL_FEATURE_NAMES = (
    "grid_width",
    "grid_height",
    "grid_resolution",
    "passable_ratio",
    "violation_count",
    "coverage_rate",
    "step_index",
    "remaining_steps",
)

EXPERIMENTAL_CANDIDATE_FIELDS = CANDIDATE_BENEFIT_FIELDS + CANDIDATE_COST_FIELDS
MISSING_INDICATOR_NAMES = tuple(f"{field}_missing" for field in EXPERIMENTAL_CANDIDATE_FIELDS)
