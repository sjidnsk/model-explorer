# Network Architecture v1.1 Progress

## Status Summary

| Field | Value |
|---|---|
| Phase | Network Architecture v1.3 network architecture experiments |
| Current stage | v1 mask-stress data quality gate |
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

## Network Architecture v1.3 Progress

| ID | Task | Status | Evidence | Next Action |
|---|---|---|---|---|
| NA-8 | Architecture Config Hardening | Completed | `architecture_config` is parsed per architecture; `hidden_dim`, `dropout`, and `candidate_attention_v1.attention_heads` are recorded in checkpoint metadata and training result; old checkpoints without config still load with `mlp_v1` fallback; `python -m unittest discover -s tests -v` passed 112 tests | Run final verify |
| NA-9 | Candidate Attention Mask Invariance | Completed | Regression test mutates unreachable and padding candidate tensors and reorders candidate rows; valid action logits/probabilities remain invariant up to the same permutation; masked candidates stay probability zero; `python -m unittest discover -s tests -v` passed 112 tests | Run final verify |
| NA-10 | Architecture Diagnostics | Completed | Training JSON summary and Markdown report include `architecture_diagnostics` with architecture, parsed config, observation schema, feature dimensions, missing indicator dimension, and valid action mask count distribution; `python -m unittest discover -s tests -v` passed 112 tests | Run final verify |

## Quasi-real South Pole Dataset Pipeline v1 Progress

| ID | Task | Status | Evidence | Next Action |
|---|---|---|---|---|
| QD-0 | LRO LOLA local manifest | Completed | `data/manifests/lunar_south_pole_lro_lola_gdr_875s_20m.json` records `dataset_id`, `data_class=quasi_real`, source URLs, byte sizes, SHA-256 hashes, projection, and intended model-explorer use; `data/raw/` and `data/processed/` are git-ignored | Keep raw and processed products out of commits |
| QD-1 | Manifest validation | Completed | `model_explorer.data.manifest.validate_data_manifest` checks existence, bytes, and SHA-256; missing and mismatched files return readable `DataManifestIssue` entries; tests cover valid, missing, hash mismatch, and readable `require_valid()` failure | Use validation before generating training samples |
| QD-2 | Optional raster adapter | Completed | `model_explorer.data.raster.read_raster_window` reads finite raster windows through optional Pillow/JPEG2000 support and raises `RasterDecodeUnavailable` with a readable diagnostic when decoding is unavailable; tests use a small fixture instead of large JP2 files | Add a dependency note if Pillow/JPEG2000 is absent in another environment |
| QD-3 | LOLA south-pole ROI contract generation | Completed | `model_explorer.data.lola_south_pole` derives relative elevation, slope proxy, roughness proxy, observation-count confidence, risk, path cost, energy cost, and coverage/value signals; generated contracts stay in `model-explorer-contract/v1` and only expose candidate-list `top_goals` | Add more ROIs after first smoke remains stable |
| QD-4 | Quasi-real rollout JSONL | Completed | `write_lola_south_pole_rollouts_from_manifest_jsonl` produced `data/processed/quasi_real/lunar_south_pole/lro_lola_gdr_875s_20m/rollouts_roi_3700_3700_32.jsonl` with 3 episodes, 3 trainable transitions, finite rewards, `dataset_id=lunar_south_pole_lro_lola_gdr_875s_20m`, and `data_class=quasi_real` | Treat this as smoke data, not a performance benchmark |
| QD-5 | Quasi-real architecture smoke | Completed | Real LOLA ROI experiment `quasi-real-lola-south-pole-architecture-smoke/roi-3700-3700-32` trained `mlp_v1`, `mlp_missing_v1`, and `candidate_attention_v1`; JSON summary and Markdown report include `data_class`, `dataset_id`, and architecture metadata; `python -m unittest discover -s tests -v` passed 119 tests | Expand ROI set only after smoke data review |
| QD-6 | Final verification gates | Completed | `$env:PYTHONPATH='src'; python -m model_explorer verify` passed with 120 unittest cases, benchmark smoke, forbidden import scan over 43 files, and `git diff --check`; explicit forbidden import `rg` returned no matches | Keep goal outputs uncommitted until user asks for commit |

## Quasi-real South Pole Evaluation Matrix v1 Progress

| ID | Task | Status | Evidence | Next Action |
|---|---|---|---|---|
| EM-0 | Multi-ROI evaluation manifest | Completed | Added `model-explorer-quasi-real-evaluation/v1` manifest loader/validator/dry-run and `model_explorer quasi-real validate/dry-run/run`; real manifest `data/processed/qreal_eval_roi32_v1/matrix.json` validates `lunar_south_pole_lro_lola_gdr_875s_20m` with 6 checked files / 40,631,113 bytes and reports scope `quasi-real evaluation; not real-world generalization benchmark` | Keep generated manifest under ignored `data/processed/` |
| EM-1 | Multi-ROI sample generation | Completed | Real matrix run generated 10 scenarios from 5 ROI entries / 4 ROI groups: `smooth_high_confidence`, `rim_or_steep_slope`, `low_observation_count`, `mixed_risk`; split counts are train 4, validation 2, test 2, benchmark 2; provenance records `dataset_id`, `data_class=quasi_real`, region, ROI, resolution, seed, and generator version; source ROI offsets are preserved in contract `grid.origin` | Add broader ROI discovery only after v1 matrix review |
| EM-2 | Dataset quality gates | Completed | Real run dataset summary: 10 episodes, 10 transitions, 10 trainable transitions, reward std 0.016618254713438904, action_mask_valid_mean 1.0, unreachable_candidate_rate 0.0, non_finite_reward_count 0; configured gates passed with no violations | Continue treating labels as derived quasi-real proxies |
| EM-3 | Architecture evaluation matrix | Completed | Real run trained `mlp_v1`, `mlp_missing_v1`, and `candidate_attention_v1`; each run wrote checkpoint, `training-summary.json`, `validation-evaluation.json`, and `losses.jsonl`; `matrix-report.md` includes quality gates, policy ranking, per-group winners, and architecture delta details against `utility` and `coverage_heuristic` | Compare stability over more seeds only after this v1 matrix is accepted |
| EM-4 | Final verification gates | Completed | Final verification passed after matrix report and ROI origin enhancements: `python -m unittest discover -s tests -v` passed 123 tests; `$env:PYTHONPATH='src'; python -m model_explorer verify` passed 123 unittest cases plus benchmark smoke, forbidden import scan over 44 files, and `git diff --check`; explicit forbidden import `rg` over `src tests scripts docs data` returned no matches; `git status --short --ignored` shows raw/processed outputs only as ignored data | Keep generated matrix outputs out of commits; do not claim real-world generalization |

## Quasi-real v1 Stability Evaluation Progress

| ID | Task | Status | Evidence | Next Action |
|---|---|---|---|---|
| VS-0 | Stability manifest template | Completed | Added tracked `data/manifests/lunar_south_pole_lro_lola_stability_matrix_v1.json`; it references the existing LOLA GDR 875S 20 m data manifest, writes under ignored `data/processed/qreal_stability_v1`, covers 12 ROI windows across 4 ROI groups, has train/validation/test split counts of 4 each, and configures 3 seeds x 3 v1 architectures | Keep raw and processed products ignored |
| VS-1 | Multi-seed stability summary | Completed | `summary.json` now includes `stability_summary` with per-architecture `run_count`, mean/std/min/max for loss metrics and `torch_policy` metrics, plus global loss distribution and baseline delta distributions; Markdown report adds `Architecture Stability`, `Loss Distribution`, and `Baseline Delta Summary` sections | Treat values as quasi-real stability evidence only |
| VS-2 | ROI/split/group report consistency | Completed | Fixture regression checks JSON summary and Markdown report agree on ROI groups, split counts, and report sections; real run produced 12 scenarios from `smooth_high_confidence`, `rim_or_steep_slope`, `low_observation_count`, and `mixed_risk` | Add broader ROI discovery later, not in v1 stability hardening |
| VS-3 | Sample coverage warnings | Completed | Report emits non-fatal warnings `no_unreachable_candidates` and `no_mask_stress_samples` when the matrix lacks unreachable candidates or mask-stress rows; real stability run returned both warnings while keeping finite metrics and passing quality gates | Add dedicated mask-stress ROI/sample generation in a later data-quality pass |
| VS-4 | Real stability run | Completed | `$env:PYTHONPATH='src'; python -m model_explorer quasi-real run data\manifests\lunar_south_pole_lro_lola_stability_matrix_v1.json` completed 9 architecture/seed runs; loss distribution count 9, mean 0.05742432177066803, std 0.01835056203905026; per-architecture run counts are 3 for `mlp_v1`, `mlp_missing_v1`, and `candidate_attention_v1` | Run final repository verification gates |
| VS-5 | Final verification gates | Completed | `python -m unittest discover -s tests -v` passed 126 tests; `$env:PYTHONPATH='src'; python -m model_explorer verify` passed 126 unittest cases, benchmark smoke, forbidden import scan over 44 files, and `git diff --check`; external `git diff --check` returned 0 with CRLF warnings only; external forbidden import `rg` returned no matches | Keep stability outputs under ignored `data/processed/` and do not claim real-world generalization |

## v1 Mask-Stress Data Quality Gate Progress

| ID | Task | Status | Evidence | Next Action |
|---|---|---|---|---|
| MS-0 | Deterministic mask-stress manifest | Completed | Added tracked `data/manifests/lunar_south_pole_lro_lola_mask_stress_matrix_v1.json`; it references the existing LOLA GDR 875S 20 m data manifest, writes under ignored `data/processed/qreal_mask_stress_v1`, enables `mask_stress_augmented`, and keeps the run labeled `quasi_real` / `not real-world generalization benchmark` | Keep augmentation explicit and default-off for old manifests |
| MS-1 | Mask-stress augmentation | Completed | `model_explorer.data.evaluation_matrix` now accepts optional `mask_stress`; when enabled it marks generated scenario provenance and contract observation metadata with `mask_stress_augmented`, deterministically flips configured candidate-list entries to `reachable=false`, and removes configured experimental fields for missing-field fallback coverage without changing `model-explorer-contract/v1` stable field names | Do not use augmented labels as mission ground truth |
| MS-2 | Dataset summary and gates | Completed | Dataset summary now records `observation_slot_count`, `padding_candidate_count/rate`, `missing_experimental_feature_candidate_count`, `mask_stress_sample_count/rate`, and `mask_stress_augmented`; explicit gates `min_unreachable_candidate_count` and `min_mask_stress_sample_count` fail only when configured | Keep old quasi-real/stability manifests warning-compatible |
| MS-3 | Report coverage | Completed | Matrix Markdown report adds `Mask-Stress Coverage`; architecture stability tables include per-architecture dataset metrics such as `dataset.padding_candidate_count`, `dataset.missing_experimental_feature_candidate_count`, and `dataset.mask_stress_sample_count`; report continues to display `quasi_real` and `not real-world generalization benchmark` | Use report as data-quality evidence, not performance proof |
| MS-4 | Real mask-stress run | Completed | `$env:PYTHONPATH='src'; python -m model_explorer quasi-real run data\manifests\lunar_south_pole_lro_lola_mask_stress_matrix_v1.json` completed 4 scenarios and 3 architecture runs; dataset summary: `unreachable_candidate_count=4`, `padding_candidate_count=2`, `missing_experimental_feature_candidate_count=30`, `mask_stress_sample_count=4`, `non_finite_reward_count=0`, coverage warnings empty, validation gates passed | Run full repository verification gates |
| MS-5 | Final verification gates | Completed | `python -m unittest discover -s tests -v` passed 129 tests; `$env:PYTHONPATH='src'; python -m model_explorer verify` passed 129 unittest cases, benchmark smoke, forbidden import scan over 44 files, and `git diff --check`; external `git diff --check` and forbidden import `rg` are part of final handoff gates | Keep generated outputs under ignored `data/processed/`; do not use mask-stress scores as real-world performance evidence |

## Current Architecture Snapshot

```text
contract JSON
-> extract_policy_observation
-> candidate_features + global_features + action_mask
-> architecture registry + architecture_config
-> MaskedCandidatePolicyNetwork / MissingIndicatorCandidatePolicyNetwork / CandidateAttentionPolicyNetwork
-> masked logits + value
-> PPO loss / TorchPolicyScorer
```

Current network-architecture properties:

- Candidate action space is limited to `top_goals`.
- `reachable=false` and padding actions are masked.
- Missing experimental fields use compatibility fallback values plus explicit missing indicators.
- Candidate and global features use bounded, finite normalization rules.
- `train.architecture` supports single-model compatibility; `train.architectures` supports architecture matrix runs.
- `train.architecture_config` and `train.architecture_configs` support parsed per-architecture knobs without changing the contract.
- `mlp_v1`, `mlp_missing_v1`, and `candidate_attention_v1` are selectable, trainable, checkpointable, and mask-safe.
- `candidate_attention_v1` uses masked self-attention and regression tests guard against unreachable/padding candidate information leakage.
- Training summaries and reports include architecture diagnostics for reproducibility and comparison.
- PPO training, checkpoint save/load, report integration, and architecture deltas exist.
- Ubuntu Readiness documents default install, `model-explorer[training]`, Linux shell verification, and Windows-vs-Ubuntu validation boundaries.
- Quasi-real LRO LOLA south-pole data can now be validated from manifest, decoded through an optional raster adapter, converted into candidate-list contracts, written as rollout JSONL, and used for three-architecture training smoke without expanding the action space beyond `top_goals`.
- The v1 stability matrix can compare `mlp_v1`, `mlp_missing_v1`, and `candidate_attention_v1` over multiple seeds with mean/std summaries while preserving the `quasi_real` and `not real-world generalization benchmark` labels.
- The v1 mask-stress matrix can produce deterministic unreachable candidates, padding coverage, missing experimental field fallback, and per-architecture finite training metrics while keeping augmentation explicitly labeled.

## Decision Log

| Date | Decision | Reason |
|---|---|---|
| 2026-05-28 | Start with `mlp_v1` hardening instead of full-map network | Existing contract exposes candidate summaries, not full map tensors |
| 2026-05-28 | Prioritize missing indicators before attention | Distinguishing fallback values from real zero values is lower risk and higher leverage |
| 2026-05-28 | Add architecture config before adding multiple models | Experiments need reproducible manifest-level model selection |
| 2026-05-28 | Keep v1.3 focused on network architecture, not release readiness | User explicitly deferred real Ubuntu validation and release convergence |

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

Network Architecture v1.3 architecture config and diagnostics
python -m unittest tests.test_model_explorer.TorchPolicyNetworkTests tests.test_training_closure.TrainingClosureTests -v
Result: passed 15 tests
python -m unittest discover -s tests -v
Result: passed 112 tests

Quasi-real South Pole Dataset Pipeline v1
python -m unittest tests.test_quasi_real_data_pipeline -v
Result: passed 8 tests
python -m unittest discover -s tests -v
Result: passed 120 tests
Real LOLA ROI smoke:
manifest validation passed for 6 files / 40,631,113 bytes
generated rollout JSONL has 3 episodes, 3 trainable transitions, and 0 non-finite rewards
real ROI experiment trained `mlp_v1`, `mlp_missing_v1`, and `candidate_attention_v1`
$env:PYTHONPATH='src'; python -m model_explorer verify
Result: passed; unittest, benchmark_smoke, forbidden_import_check, and git_diff_check returned 0
rg forbidden import check
Result: no forbidden import matches in src, tests, scripts, docs, or data

Quasi-real v1 Stability Evaluation
python -m unittest tests.test_quasi_real_data_pipeline.QuasiRealEvaluationMatrixTests -v
Result: passed 6 tests
$env:PYTHONPATH='src'; python -m model_explorer quasi-real validate data\manifests\lunar_south_pole_lro_lola_stability_matrix_v1.json
Result: valid; 12 ROI windows, train/validation/test split counts of 4 each, 3 architectures
$env:PYTHONPATH='src'; python -m model_explorer quasi-real dry-run data\manifests\lunar_south_pole_lro_lola_stability_matrix_v1.json
Result: dry_run; output paths remain under ignored data\processed\qreal_stability_v1
$env:PYTHONPATH='src'; python -m model_explorer quasi-real run data\manifests\lunar_south_pole_lro_lola_stability_matrix_v1.json
Result: completed; 12 scenarios, 9 architecture/seed runs, loss mean 0.05742432177066803, loss std 0.01835056203905026, warnings no_unreachable_candidates/no_mask_stress_samples
python -m unittest discover -s tests -v
Result: passed 126 tests
$env:PYTHONPATH='src'; python -m model_explorer verify
Result: passed; unittest, benchmark_smoke, forbidden_import_check, and git_diff_check returned 0
git diff --check
Result: returned 0; Windows line-ending warnings only
rg forbidden import check
Result: no forbidden import matches in src, tests, scripts, docs, or data

v1 Mask-Stress Data Quality Gate
python -m unittest tests.test_quasi_real_data_pipeline.QuasiRealEvaluationMatrixTests -v
Result: initially failed as expected for missing tracked manifest, unknown `min_mask_stress_sample_count`, and absent mask-stress summary/report fields
python -m unittest tests.test_quasi_real_data_pipeline.QuasiRealEvaluationMatrixTests -v
Result: passed 9 tests after implementation
$env:PYTHONPATH='src'; python -m model_explorer quasi-real validate data\manifests\lunar_south_pole_lro_lola_mask_stress_matrix_v1.json
Result: valid; 4 ROI windows, explicit `mask_stress_augmented`, 3 architectures, configured mask-stress gates
$env:PYTHONPATH='src'; python -m model_explorer quasi-real dry-run data\manifests\lunar_south_pole_lro_lola_mask_stress_matrix_v1.json
Result: dry_run; output paths remain under ignored data\processed\qreal_mask_stress_v1
$env:PYTHONPATH='src'; python -m model_explorer quasi-real run data\manifests\lunar_south_pole_lro_lola_mask_stress_matrix_v1.json
Result: completed; 4 scenarios, 3 architecture runs, unreachable_candidate_count 4, padding_candidate_count 2, missing_experimental_feature_candidate_count 30, mask_stress_sample_count 4, non_finite_reward_count 0, coverage warnings empty
python -m unittest discover -s tests -v
Result: passed 129 tests
$env:PYTHONPATH='src'; python -m model_explorer verify
Result: passed; unittest, benchmark_smoke, forbidden_import_check, and git_diff_check returned 0
```

## Risk Register

| Risk | Impact | Mitigation |
|---|---|---|
| Synthetic benchmark overfitting | Network changes may look better without real-world value | Treat synthetic results as smoke/regression only |
| Synthetic benchmark remains smoke/regression evidence only | Architecture deltas can be misread as real-world generalization | Reports and release checklist explicitly avoid real-world generalization claims |
| Ubuntu 24.04 target environment has not been executed in this Windows session | Linux install commands are documented but not proven in a target VM/container here | Run the documented Ubuntu command block in target CI or an Ubuntu 24.04 environment before external release |
| Feature schema churn | Checkpoints and rollout logs may become hard to load | Version observation schema and preserve loader fallback |
| Mask regression | Policy could score unreachable or padding actions | Add tests for unreachable and padding masks for every architecture |
| Attention leakage | Masked candidates could affect valid candidate logits through attention context | v1.3 regression mutates unreachable/padding tensors and checks valid output invariance |
| Config sprawl | Too many knobs make experiments hard to compare | Keep v1.1 architecture options to `mlp_v1`, `mlp_missing_v1`, `candidate_attention_v1` |
| PyTorch optional dependency | Non-training workflows could fail when PyTorch is absent | Keep train-specific tests skippable and non-training paths independent |
| JP2 optional decoder availability | Another environment may lack Pillow/JPEG2000 support and be unable to decode LOLA JP2 products | Keep raster decoding optional and return `RasterDecodeUnavailable` diagnostics; fixture tests avoid requiring large JP2 files |
| Quasi-real ROI smoke is not mission ground truth | Terrain-derived labels are generated proxies and can bias policy evaluation | Mark data as `quasi_real`, keep provenance in rollout/report, and avoid claiming real-world generalization |
| Windows long output paths | Deep ignored output roots plus long experiment names can exceed legacy Windows path limits during report writes | Use short local output roots such as `data/processed/qreal_eval_roi32_v1` for current v1 matrix; defer path-prefix hardening unless it blocks tracked workflows |
| Stability matrix lacks mask-stress samples | Current real stability run has no unreachable candidates, so mask-stress coverage is only fixture-tested | Emit non-fatal report warnings and keep separate mask-safety unit tests; add a dedicated mask-stress sample source before using stability results for stronger claims |
| Mask-stress augmentation can distort quasi-real costs | Removing experimental fields intentionally exercises fallback paths and can make selected path/risk metrics less representative | Keep `mask_stress_augmented` explicit in provenance/report and use this gate only for mask/data-quality evidence |

## Completion Criteria

- TODO file has every v1.1 task marked complete.
- Progress table records evidence for each completed task.
- `mlp_v1`, `mlp_missing_v1`, and `candidate_attention_v1` can be selected by manifest.
- All architectures preserve action mask safety.
- Reports include architecture identity and baseline deltas.
- Checkpoint metadata, JSON summary, and Markdown report include parsed `architecture_config`.
- `candidate_attention_v1` passes mask invariance tests for unreachable, padding, and reordered candidates.
- Final verification commands pass.
