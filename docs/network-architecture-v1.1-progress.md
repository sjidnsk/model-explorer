# Network Architecture v1.1 Progress

## Status Summary

| Field | Value |
|---|---|
| Phase | Network Architecture v1.1 verified |
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

## Current Architecture Snapshot

```text
contract JSON
-> extract_policy_observation
-> candidate_features + global_features + action_mask
-> MaskedCandidatePolicyNetwork
-> masked logits + value
-> PPO loss / TorchPolicyScorer
```

Current properties:

- Candidate action space is limited to `top_goals`.
- `reachable=false` and padding actions are masked.
- Missing experimental fields use compatibility fallback values.
- PPO training, checkpoint save/load, and report integration exist.

Current gaps:

- Missing experimental fields are not exposed as tensor indicators.
- Global and cost feature scales are not fully normalized.
- Architecture selection is not configurable in manifest.
- Candidate self-attention is not implemented.
- Synthetic benchmark supports regression, not real-world generalization claims.

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
```

## Risk Register

| Risk | Impact | Mitigation |
|---|---|---|
| Synthetic benchmark overfitting | Network changes may look better without real-world value | Treat synthetic results as smoke/regression only |
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
