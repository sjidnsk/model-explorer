# Network Architecture v1.1 Progress

## Status Summary

| Field | Value |
|---|---|
| Phase | Network Architecture v1.2 release candidate |
| Current baseline | `mlp_v1` masked candidate policy |
| Scope | Candidate-list policy over `ModelExplorerContract.top_goals` |
| Out of scope | Full-map action space, external project imports, contract v1 breaking changes |
| Last updated | 2026-05-28 |

## Progress Table

| ID | Task | Status | Evidence | Next Action |
|---|---|---|---|---|
| NA-0 | Baseline Snapshot | Completed | `MaskedCandidatePolicyNetwork.architecture_name = mlp_v1`; checkpoint metadata/result include `architecture`; old checkpoint fallback loads as `mlp_v1`; `python -m unittest discover -s tests -v` passed 91 tests | Continue with NA-1 missing indicators |
| NA-1 | Observation Schema v1.1 | Completed | `PolicyObservation` now carries `candidate_missing_indicators`; real `0.0` and fallback `0.0` are distinguishable; old rollout JSON fallback uses zero indicators; `python -m unittest discover -s tests -v` passed 94 tests | Continue with architecture registry |
| NA-2 | Feature Normalization Policy | Completed | Candidate benefit/cost features and global grid/progress fields are normalized with finite-value guards; extreme-scale fixture remains finite; `python -m unittest discover -s tests -v` passed 94 tests | Continue with architecture registry |
| NA-3 | Architecture Config and Registry | Completed | Added `policy/architectures.py`; `train.architecture` selects `mlp_v1` or `mlp_missing_v1`; unknown names raise readable errors; checkpoint loader restores architecture metadata; `python -m unittest discover -s tests -v` passed 98 tests | Continue with candidate attention |
| NA-4 | MLP Missing Indicators v1 | Completed | `mlp_missing_v1` concatenates missing indicators into candidate encoder input; mask safety, finite loss, checkpoint save/load, JSON summary, and Markdown report are covered; `python -m unittest discover -s tests -v` passed 98 tests | Continue with candidate attention |
| NA-5 | Candidate Attention v1 | Completed | `candidate_attention_v1` adds one masked self-attention layer with `action_mask` key padding; padding candidates do not affect valid probabilities; finite training/checkpoint smoke covered; `python -m unittest discover -s tests -v` passed 100 tests | Run final verification |
| NA-6 | Architecture Benchmark Matrix | Completed | Training JSON/Markdown report include architecture; `architecture_deltas` records trained architecture deltas against utility and coverage heuristic; `python -m unittest discover -s tests -v` passed 100 tests | Run final verification |
| NA-7 | Final Verification and Readiness Review | Completed | Final verification passed: unittest 100 tests, `python -m model_explorer verify`, `git diff --check`, and forbidden import `rg` check | Keep v1.1 as candidate-list architecture baseline for future experiments |

## Ubuntu Readiness / Network Architecture v1.2 Progress

| ID | Task | Status | Evidence | Next Action |
|---|---|---|---|---|
| UR-0 | Ubuntu installation and validation documentation | Completed | Added `docs/ubuntu-readiness.md` with Ubuntu 24.04, Python 3.12, Linux shell verify commands, default install, `model-explorer[training]`, and Windows-vs-Ubuntu validation boundary; `python -m unittest discover -s tests -v` passed 110 tests | Validate commands in target Ubuntu environment when available |
| UR-1 | Dependency extras | Completed | `pyproject.toml` defines `project.optional-dependencies.training` with PyTorch while default dependencies do not require torch; docs and tests cover optional training install; dry-run matrix test guards against importing torch; `python -m unittest discover -s tests -v` passed 110 tests | Keep non-training paths importable without PyTorch |
| UR-2 | Verify cross-platform hardening | Completed | `model_explorer verify --dry-run` now reports `unittest`, `benchmark_smoke`, `forbidden_import_check`, and `git_diff_check` with machine-readable `kind`; forbidden import check is a Python scan over `src`, `tests`, and `scripts`; `python -m unittest discover -s tests -v` passed 110 tests | Run final verify and external `rg` gate before completion |
| AR-8 | Multi-architecture benchmark manifest | Completed | `architecture-smoke-experiment.json` uses fixed seed 17 and `train.architectures` for `mlp_v1`, `mlp_missing_v1`, and `candidate_attention_v1`; experiment runner writes per-architecture/per-seed checkpoints and per-run baseline deltas; reports mark the suite as synthetic smoke / regression; `python -m unittest discover -s tests -v` passed 110 tests | Validate matrix manifest in Ubuntu with `model-explorer[training]` installed |
| AR-9 | Reproducibility metadata audit | Completed | Checkpoint metadata records architecture, seed, hidden size, candidate/global/missing feature names, and `observation_schema_version = policy-observation/v1.1`; old checkpoint fallback still loads as `mlp_v1`; `python -m unittest discover -s tests -v` passed 110 tests | Do not require bitwise-identical floating point training results |
| UR-3 | Linux CI/script preparation | Completed | `docs/ubuntu-readiness.md` provides POSIX shell install and validation commands; `model_explorer verify` uses Python-native forbidden import scanning instead of PowerShell-only commands; `python -m unittest discover -s tests -v` passed 110 tests | Re-run final gates before handoff |

## Current Architecture Snapshot

```text
contract JSON
-> extract_policy_observation
-> candidate_features + global_features + action_mask
-> MaskedCandidatePolicyNetwork
-> masked logits + value
-> PPO loss / TorchPolicyScorer
```

Current release-candidate properties:

- Candidate action space is limited to `top_goals`.
- `reachable=false` and padding actions are masked.
- Missing experimental fields use compatibility fallback values plus explicit missing indicators.
- Candidate and global features use bounded, finite normalization rules.
- `train.architecture` supports single-model compatibility; `train.architectures` supports architecture matrix runs.
- `mlp_v1`, `mlp_missing_v1`, and `candidate_attention_v1` are selectable, trainable, checkpointable, and mask-safe.
- PPO training, checkpoint save/load, report integration, and architecture deltas exist.
- Ubuntu Readiness documents default install, `model-explorer[training]`, Linux shell verification, and Windows-vs-Ubuntu validation boundaries.

## Decision Log

| Date | Decision | Reason |
|---|---|---|
| 2026-05-28 | Start with `mlp_v1` hardening instead of full-map network | Existing contract exposes candidate summaries, not full map tensors |
| 2026-05-28 | Prioritize missing indicators before attention | Distinguishing fallback values from real zero values is lower risk and higher leverage |
| 2026-05-28 | Add architecture config before adding multiple models | Experiments need reproducible manifest-level model selection |

## Verification Snapshot

Latest completed stage:

```text
NA-0 Baseline Snapshot
python -m unittest discover -s tests -v
Result: passed 91 tests

NA-1/NA-2 Observation Schema and Feature Normalization
python -m unittest discover -s tests -v
Result: passed 94 tests

NA-3/NA-4 Architecture Registry and MLP Missing Indicators
python -m unittest discover -s tests -v
Result: passed 98 tests

NA-5/NA-6 Candidate Attention and Architecture Report Matrix
python -m unittest discover -s tests -v
Result: passed 100 tests

NA-7 Final Verification
python -m unittest discover -s tests -v
Result: passed 100 tests
$env:PYTHONPATH='src'; python -m model_explorer verify
Result: passed; unittest, benchmark_smoke, and git_diff_check returned 0
git diff --check
Result: returned 0; Windows line-ending warnings only
rg forbidden import check
Result: no forbidden import matches in src, tests, or scripts

UR/AR v1.2 Ubuntu readiness and architecture matrix
python -m unittest discover -s tests -v
Result: passed 110 tests
$env:PYTHONPATH='src'; python -m model_explorer verify
Result: passed; unittest, benchmark_smoke, forbidden_import_check, and git_diff_check returned 0
git diff --check
Result: returned 0; Windows line-ending warnings only
rg forbidden import check
Result: no forbidden import matches in src, tests, or scripts
```

## Risk Register

| Risk | Impact | Mitigation |
|---|---|---|
| Synthetic benchmark overfitting | Network changes may look better without real-world value | Treat synthetic results as smoke/regression only |
| Synthetic benchmark remains smoke/regression evidence only | Architecture deltas can be misread as real-world generalization | Reports and release checklist explicitly avoid real-world generalization claims |
| Ubuntu 24.04 target environment has not been executed in this Windows session | Linux install commands are documented but not proven in a target VM/container here | Run the documented Ubuntu command block in target CI or an Ubuntu 24.04 environment before external release |
| Feature schema churn | Checkpoints and rollout logs may become hard to load | Version observation schema and preserve loader fallback |
| Mask regression | Policy could score unreachable or padding actions | Add tests for unreachable and padding masks for every architecture |
| Config sprawl | Too many knobs make experiments hard to compare | Keep v1.1 architecture options to `mlp_v1`, `mlp_missing_v1`, `candidate_attention_v1` |
| PyTorch optional dependency | Non-training workflows could fail when PyTorch is absent | Keep train-specific tests skippable and non-training paths independent |

## Completion Criteria

- TODO file has every v1.1 task marked complete.
- Progress table records evidence for each completed task.
- `mlp_v1`, `mlp_missing_v1`, and `candidate_attention_v1` can be selected by manifest.
- All architectures preserve action mask safety.
- Reports include architecture identity and baseline deltas.
- Final verification commands pass.
