# Network Architecture v1.2 Release Readiness

This checklist defines the release-candidate gates for the Ubuntu-ready
candidate-list architecture baseline. It is a readiness checklist, not a claim
that synthetic benchmark deltas generalize to real lunar terrain.

## Install And Dependency Gates

- [x] default install does not require PyTorch.
- [x] Training installs PyTorch through `model-explorer[training]`.
- [x] Non-training dry-run paths do not import `torch`.
- [x] Ubuntu commands are documented for Python 3.12 and Ubuntu 24.04.

## Verification Gates

- [x] `verify JSON` reports machine-readable steps for `unittest`,
  `benchmark_smoke`, `forbidden_import_check`, and `git_diff_check`.
- [x] `forbidden_import_check` scans `src`, `tests`, and `scripts` without
  relying on PowerShell-only commands.
- [x] Windows local command:
  `$env:PYTHONPATH='src'; python -m model_explorer verify`
- [x] Linux CI command:
  `PYTHONPATH=src python -m model_explorer verify`

## Architecture Matrix Gates

- [x] `train.architecture` keeps single-architecture compatibility.
- [x] `train.architectures` runs `mlp_v1`, `mlp_missing_v1`, and
  `candidate_attention_v1` as a comparable matrix.
- [x] Matrix runs preserve architecture, seed, checkpoint, and baseline deltas
  in JSON summaries and Markdown reports.
- [x] Checkpoints include `observation_schema_version`,
  candidate/global feature names, missing indicator names, architecture, seed,
  and hidden size.

## Safety Gates

- [x] action mask safety remains the action boundary for unreachable candidates
  and padding actions.
- [x] Training and selection remain limited to
  `ModelExplorerContract.top_goals`.
- [x] No full-map action space is introduced.
- [x] No import or call to `a_gcs_ws-2.0.1` or `dev-platform-constraints` is
  introduced.

## Remaining Release Risk

- [ ] Ubuntu 24.04 target environment has not been executed in this Windows
  session.
- [ ] Synthetic benchmark remains smoke/regression evidence only and must not
  be presented as real-world generalization evidence.
