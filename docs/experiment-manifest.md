# Experiment Manifest

`model-explorer-experiment/v1` describes local benchmark and experiment runs.
It is intentionally file based so the runner does not import external projects.

## Minimum Shape

- `schema_version`: optional; defaults to `model-explorer-experiment/v1`.
- `scenarios`: list of `model-explorer-contract/v1` JSON files for simple runs.
- `splits.benchmark`: grouped scenario lists for benchmark reports.
- `planner`: optional planner configuration; default behavior remains local.
- `outputs.root`: root directory used to derive run artifacts, or explicit
  `outputs.rollouts` and `outputs.evaluation` paths.

## Stable Report Sections

Markdown reports keep existing rollout, dataset, benchmark, training, and
baseline sections. The current benchmark hardening fields add:

- `Policy Ranking`
- `Torch Policy Deltas`
- `Per-Group Winners`
- `Failure Scenarios`
- `Gate Summary`

The JSON summary exposes matching machine-readable keys:
`policy_ranking`, `baseline_deltas`, `per_group_winners`, `failure_scenarios`,
and `gate_summary`.
