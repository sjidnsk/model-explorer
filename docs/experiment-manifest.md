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
- `train.architecture`: optional policy network architecture. Defaults to
  `mlp_v1`; supported values are `mlp_v1`, `mlp_missing_v1`, and
  `candidate_attention_v1`.
- `train.architectures`: optional architecture matrix. When present, the
  runner trains every listed architecture for every configured seed and writes
  separate checkpoints under architecture and seed-specific output directories.
- `train.architecture_config`: optional parsed config for the selected
  architecture. Supported common fields are `hidden_dim` and `dropout`.
- `train.architecture_configs`: optional per-architecture config mapping used
  by architecture matrix runs.
- `train.source_selection_strategies`: optional training-source matrix over
  rollout selection sources such as `coverage_heuristic` and `feedback_aware`.
- `train.teacher_imitation_weight`: optional scalar weight for the auxiliary
  teacher imitation loss. Defaults to `0.0`.
- `train.teacher_imitation_weights`: optional weight matrix. When present, the
  runner trains every source strategy for every listed imitation weight.
- `train.teacher_quality_gates`: optional non-fatal teacher data-quality gates
  recorded per source/weight run. Supported keys include
  `min_feedback_aware_sample_count`, `min_teacher_high_margin_sample_count`,
  `max_missing_teacher_signal_rate`, and
  `max_low_margin_only_dataset_rate`.

`train.architecture` and `train.architectures` are compatible with older
single-architecture manifests. `train.architecture` keeps the previous default
single-model behavior. `train.architectures` takes precedence when present and
trains each listed architecture for every configured seed.

`train.architecture_config` is recorded after parsing in checkpoint metadata,
JSON training summaries, and Markdown reports. `candidate_attention_v1` also
supports `attention_heads`; the value must divide `hidden_dim`. Unknown config
fields fail with a readable error instead of being silently ignored.

With `outputs.root`, matrix checkpoints are derived as:

```text
`outputs.root/<name>/<run_id>/<architecture>/seed-<seed>/checkpoint.pt`
```

`train.checkpoint` and `train.loss_log` may use `{architecture}` and `{seed}`
placeholders. They may also use `{selection_strategy}` and
`{teacher_imitation_weight}` placeholders for feedback-aware distillation
matrix runs. For matrix or multi-seed runs, placeholders in parent directories
are treated as output dimensions. If an explicit path does not place
architecture, seed, source, or teacher weight in parent directories, the runner
appends the missing directories to avoid overwriting checkpoints, loss logs,
and sidecar summaries.

## Stable Report Sections

Markdown reports keep existing rollout, dataset, benchmark, training, and
baseline sections. The current benchmark hardening fields add:

- `Policy Ranking`
- `Torch Policy Deltas`
- `Per-Group Winners`
- `Failure Scenarios`
- `Gate Summary`
- `Training` includes the selected architecture when training is enabled.
- `Architecture Diagnostics` includes parsed `architecture_config`,
  `observation_schema_version`, feature dimensions, missing indicator
  dimension, and valid action-mask count distribution.
- `Architecture Deltas` reports the trained architecture's `torch_policy`
  deltas against `utility` and `coverage_heuristic` when trained-policy
  evaluation is enabled.
- Matrix training keeps per-run `architecture`, `seed`, `checkpoint`, and
  baseline deltas in the JSON summary so reports can compare `mlp_v1`,
  `mlp_missing_v1`, and `candidate_attention_v1`.
- Feedback-aware distillation matrix runs keep per-run source, teacher
  imitation weight, `teacher_quality_gates`, teacher agreement, margin bucket
  agreement, and feedback-aware baseline deltas in `distillation_matrix`.

The JSON summary exposes matching machine-readable keys:
`policy_ranking`, `baseline_deltas`, `per_group_winners`, `failure_scenarios`,
`gate_summary`, and `distillation_matrix`.
