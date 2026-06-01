# Network Architecture v1.1 Progress

## Status Summary

| Field | Value |
|---|---|
| Phase | Network Architecture v1.3 network architecture experiments |
| Current stage | v1 action-sensitive training and evaluation calibration |
| Current baseline | `mlp_v1` masked candidate policy |
| Scope | Candidate-list policy over `ModelExplorerContract.top_goals` |
| Out of scope | Full-map action space, external project imports, contract v1 breaking changes |
| Last updated | 2026-06-01 |

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

## v1 Architecture Selection Gate Progress

| ID | Task | Status | Evidence | Next Action |
|---|---|---|---|---|
| AS-0 | Selection manifest template | Completed | Added tracked `data/manifests/lunar_south_pole_lro_lola_selection_matrix_v1.json`; it references the LOLA GDR 875S 20 m manifest, writes under ignored `data/processed/qreal_selection_v1`, covers 12 ROI windows across 4 ROI groups, enables explicit `mask_stress_augmented`, and configures 3 seeds x 3 architectures | Keep raw/processed outputs ignored |
| AS-1 | Selection quality gates | Completed | Dataset validation now accepts `min_roi_group_count`; selection config records `min_seed_count`, `min_architecture_count`, `min_roi_group_count`, `min_unreachable_candidate_count`, and `min_mask_stress_sample_count`; gate violations remain readable and old stability/mask-stress manifests stay compatible | Keep selection gates evidence-oriented, not release gates |
| AS-2 | Grouped validation per architecture | Completed | Training run validation now preserves validation split groups when explicit splits exist, so each architecture/seed run can report per-group `torch_policy` metrics while keeping aggregate fallback behavior for old summaries | Use grouped metrics only for v1 architecture comparison |
| AS-3 | Selection report | Completed | Matrix JSON/Markdown now include `architecture_selection` / `Architecture Selection Gate` with `recommended_architecture` or `inconclusive`, architecture mean/std/loss metrics, baseline delta distribution, per-group winners, exception count, and mask-stress coverage | Treat `inconclusive` as a valid outcome when margins are within seed variance |
| AS-4 | Real selection run | Completed | `$env:PYTHONPATH='src'; python -m model_explorer quasi-real run data\manifests\lunar_south_pole_lro_lola_selection_matrix_v1.json` completed 12 scenarios and 9 architecture/seed runs; dataset summary: `unreachable_candidate_count=12`, `padding_candidate_count=8`, `missing_experimental_feature_candidate_count=88`, `mask_stress_sample_count=12`; selection quality gates passed and decision was `inconclusive` because validation coverage margins were within seed variance | Do not promote a default v1 architecture from this evidence alone |
| AS-5 | Test evidence | Completed | New fixture tests cover tracked selection manifest, selection report fields, mask-stress coverage in JSON/Markdown, and the no-forced-winner rule; `python -m unittest discover -s tests -v` passed 132 tests | Run final verify, diff check, and forbidden import scan before completion |
| AS-6 | Final verification gates | Completed | `$env:PYTHONPATH='src'; python -m model_explorer verify` passed 132 unittest cases, benchmark smoke, forbidden import scan over 44 files, and `git diff --check`; external `git diff --check` returned 0 with CRLF warnings only; external forbidden import `rg` returned no matches | Keep selection outputs under ignored `data/processed/`; do not claim a default architecture from an `inconclusive` selection result |

## v1 Policy Decision Signal Enrichment Progress

| ID | Task | Status | Evidence | Next Action |
|---|---|---|---|---|
| DS-0 | Action-level diagnostics | Completed | `evaluate_policy_baselines(..., torch_policy=...)` records per-step selected index/cell, selected probability, rank, entropy, valid action count, baseline agreement flags, selected mask validity, and max masked probability; regression test proves a policy preferring an unreachable action still records `max_masked_action_probability=0.0` and selects a valid candidate | Keep diagnostics read-only and outside training loss |
| DS-1 | Architecture agreement report | Completed | Matrix summary/report now include `decision_diagnostics` with architecture agreement matrix, baseline agreement rates, per-group disagreement, warnings, and mask violation count; real selection run recorded 36 diagnostic samples, 0 mask violations, and non-identical architecture choices (`mlp_v1` vs `mlp_missing_v1` agreement 0.25; `mlp_v1` vs `candidate_attention_v1` agreement 0.8333333333333334) | Use disagreement as interpretability evidence, not as a winner by itself |
| DS-2 | Composite selection metric | Completed | Selection config accepts `composite_weights`; JSON/Markdown report `selection_composite_score` and component stats for final coverage, coverage delta, value coverage, path cost, risk, and failures; real run stayed `inconclusive` because composite margin was within seed variance | Tune weights only with a documented manifest change |
| DS-3 | Held-out test audit | Completed | Training runs now write per-run `test_evaluation` when an explicit test split exists; best checkpoint selection still cites validation evaluation; `held_out_test_audit.used_for_selection=false`; real selection run test audit was available and also inconclusive | Keep test split audit out of checkpoint/architecture selection |
| DS-4 | Test evidence | Completed | New regression tests cover masked action diagnostics, decision diagnostics warning behavior, agreement/composite/report fields, and validation-only checkpoint selection; `python -m unittest discover -s tests -v` passed 134 tests | Keep tests focused on v1 decision signal behavior |
| DS-5 | Final verification gates | Completed | `$env:PYTHONPATH='src'; python -m model_explorer verify` passed 134 unittest cases, benchmark smoke, forbidden import scan over 44 files, and `git diff --check`; external `git diff --check` returned 0 with CRLF warnings only; external forbidden import `rg` returned `no forbidden imports` | Keep generated selection outputs under ignored `data/processed/` |

## v1 Action-Sensitive Training and Evaluation Calibration Progress

| ID | Task | Status | Evidence | Next Action |
|---|---|---|---|---|
| AC-0 | Action-sensitive metrics | Completed | `evaluate_policy_baselines` now reports selected expected coverage delta, selected value coverage, selected risk, selected path cost, selected composite utility, and selected count for every strategy; missing experimental fields fall back to finite values | Keep `final_coverage_rate` as report context, not the only architecture signal |
| AC-1 | Reachable-only oracle and regret | Completed | Per-scenario metrics now include coverage, low-risk, low-cost, and composite oracles plus coverage/risk/path/composite regret; regression test proves unreachable candidates are excluded from oracle actions | Keep oracle constrained to `ModelExplorerContract.top_goals` |
| AC-2 | Sample discriminativeness diagnostics | Completed | Matrix summary adds candidate coverage/risk/path/value spread and oracle-vs-heuristic action disagreement rate; low-spread samples emit warning without failing training/evaluation | Separate mask-stress safety evidence from performance selection evidence |
| AC-3 | Selection/report upgrade | Completed | Selection JSON and Markdown now include `Action-Sensitive Metrics`, `Oracle Regret Summary`, `Sample Discriminativeness`, and `Per-ROI Action Outcomes`; composite weights can read nested metrics such as `action_sensitive_metrics.selected_expected_coverage_delta` | Run full verification gates |
| AC-4 | Final verification gates | Completed | Targeted tests passed, related test classes passed, `python -m unittest discover -s tests -v` passed 136 tests, `$env:PYTHONPATH='src'; python -m model_explorer verify` passed 136 unittest cases plus benchmark smoke / forbidden import scan / git diff check, external `git diff --check` returned 0 with CRLF warnings only, and explicit forbidden import `rg` returned `no forbidden imports` | Keep generated selection outputs under ignored `data/processed/` |

## Feedback-Aware Distillation Evaluation Matrix Progress

| ID | Task | Status | Evidence | Next Action |
|---|---|---|---|---|
| FA-0 | Feedback-aware training source closure | Completed | Rollout, dataset, training, evaluation, and experiment summaries record `feedback_aware` selection source, teacher action/cell, score margin, top-k agreement, margin bucket agreement, and feedback-aware baseline deltas without changing `model-explorer-contract/v1` or the candidate-list action space | Keep `teacher_imitation_weight` defaulted to 0.0 |
| FA-1 | Source x teacher-weight matrix | Completed | Experiment manifests can use `train.source_selection_strategies`, `train.teacher_imitation_weights`, and `train.teacher_quality_gates`; JSON summary records per-run `distillation_matrix` entries with teacher agreement, teacher-quality gates, margin bucket agreement, and baseline deltas | Use this as distillation evidence only |
| FA-2 | Quasi-real/mask-stress labels | Completed | Distillation matrix summaries preserve `quasi_real`, `mask_stress_augmented`, and `not real-world generalization benchmark` scope labels | Do not treat quasi-real or mask-stress results as real-world generalization proof |
| FA-3 | Distillation Calibration v4 | Completed | Training accepts optional `teacher_margin_weighting`, records `teacher_curriculum` in summaries/checkpoints, excludes failed `teacher_quality_gates` from default best-run selection when non-failed runs exist, and adds machine-readable `selection_decision`, `evaluation_scope`, and `distillation_stability_summary` fields; `python -m unittest discover -s tests -v` passed 165 tests | Keep this as calibration/curriculum/model-selection support, not a motion feasibility solver |
| FA-4 | Distillation Calibration v5 | Completed | Added curriculum profile matrix support, profile-aware checkpoint paths, group-level `calibration_recommendation`, per-run `teacher_margin_curriculum_profile`, policy-teacher `feedback_aware_confidence_calibration`, profile-dimensional `distillation_stability_summary`, and tracked manifest `data/manifests/feedback_aware_distillation_calibration_v5.json` | Continue treating quasi-real/mask-stress as calibration evidence only |
| FA-5 | Semi-Real Closed-Loop Calibration v1 | Completed | Added optional `train.system_calibration` path-feedback gate over external `path-feedback-summary/v1` JSON; system summary combines v5 `calibration_recommendation`, teacher gates, path-feedback gates, acceptance metadata, stress/mixed-stress diagnostics, machine-readable exclusion reasons, and explicit `sample_quality_summary` filtering/downweighting records. Full validation passed with `model_explorer verify`, `path-planner` pytest, `dev-platform-constraints` unittest, parent pytest, and `bash scripts/run_path_feedback_validation.sh --scenario-set all --diagnostic-profile all --top-k 3 --output-root outputs/path_feedback_validation_next_stage`; `open_grid_fallback_used=false` | Keep this as calibration/dataset-quality evidence only; no new network, action-space expansion, GCS backend, Ackermann/skid-steer/differential feasibility solver, or real-world generalization claim is introduced |
| FA-6 | Semi-Real Calibration Dataset Application v2 | Completed | Added explicit `system_calibration.sample_quality.enabled=true` application for quasi-real/mask-stress sample filtering/downweighting, hard `open_grid_fallback` sample exclusion, cross-summary `sample_quality_audit_summary` aggregation by scenario, ROI/group, reason code, action, source path, acceptance metadata, scenario set, diagnostic profile, and `top_k`, plus fixture `tests/fixtures/synthetic_experiment/semi-real-calibration-dataset-application-v2.json` | Validate with the repository acceptance command `bash scripts/run_path_feedback_validation.sh --scenario-set all --diagnostic-profile all --top-k 3 --output-root outputs/path_feedback_validation_dataset_v2`; do not treat IRIS/region-graph diagnostics as GCS trajectory, vehicle feasibility, or real-world generalization evidence |

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
- The v1 architecture selection gate combines multi-seed architecture comparison with explicit mask-stress coverage and reports either `recommended_architecture` or `inconclusive` instead of forcing a winner when margins are within seed variance.
- The v1 policy decision signal report explains whether architectures chose the same actions, matched baselines, or disagreed by ROI group, and keeps validation selection separate from held-out test audit.

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

v1 Architecture Selection Gate
python -m unittest tests.test_quasi_real_data_pipeline.QuasiRealEvaluationMatrixTests -v
Result: initially failed as expected for missing tracked selection manifest, unknown `min_roi_group_count`, and absent selection decision/report fields
python -m unittest tests.test_quasi_real_data_pipeline.QuasiRealEvaluationMatrixTests -v
Result: passed 12 tests after implementation
$env:PYTHONPATH='src'; python -m model_explorer quasi-real validate data\manifests\lunar_south_pole_lro_lola_selection_matrix_v1.json
Result: valid; 12 ROI windows, train/validation/test split counts of 4 each, explicit mask-stress coverage, 3 seeds, 3 architectures, selection gates configured
$env:PYTHONPATH='src'; python -m model_explorer quasi-real dry-run data\manifests\lunar_south_pole_lro_lola_selection_matrix_v1.json
Result: dry_run; output paths remain under ignored data\processed\qreal_selection_v1
$env:PYTHONPATH='src'; python -m model_explorer quasi-real run data\manifests\lunar_south_pole_lro_lola_selection_matrix_v1.json
Result: completed; 12 scenarios, 9 architecture/seed runs, unreachable_candidate_count 12, padding_candidate_count 8, missing_experimental_feature_candidate_count 88, mask_stress_sample_count 12, selection gate status passed, architecture selection decision inconclusive
python -m unittest discover -s tests -v
Result: passed 132 tests
$env:PYTHONPATH='src'; python -m model_explorer verify
Result: passed; 132 unittest cases, benchmark_smoke, forbidden_import_check over 44 files, and git_diff_check returned 0
git diff --check
Result: returned 0; Windows line-ending warnings only
rg forbidden import check
Result: no forbidden import matches in src, tests, scripts, docs, or data

v1 Policy Decision Signal Enrichment
python -m unittest tests.test_model_explorer.BaselineEvaluationTests.test_policy_action_diagnostics_keep_masked_actions_at_zero_probability -v
Result: initially failed for missing `action_diagnostics`, then passed after implementation
python -m unittest tests.test_quasi_real_data_pipeline.QuasiRealEvaluationMatrixTests.test_decision_diagnostics_warn_when_policies_match_heuristic_and_actions_are_identical -v
Result: initially failed for missing `_decision_diagnostics_summary`, then passed after implementation
python -m unittest tests.test_quasi_real_data_pipeline.QuasiRealEvaluationMatrixTests.test_selection_report_summarizes_architecture_decision_and_quality_gates -v
Result: initially failed for missing `decision_diagnostics`, then passed after implementation
python -m unittest discover -s tests -v
Result: passed 134 tests
$env:PYTHONPATH='src'; python -m model_explorer quasi-real run data\manifests\lunar_south_pole_lro_lola_selection_matrix_v1.json
Result: completed; 12 scenarios, 9 architecture/seed runs, 36 action diagnostic samples, 0 mask diagnostic violations, held-out test audit available and not used for selection, selection decision remains inconclusive
$env:PYTHONPATH='src'; python -m model_explorer verify
Result: passed; 134 unittest cases, benchmark_smoke, forbidden_import_check over 44 files, and git_diff_check returned 0
git diff --check
Result: returned 0; Windows line-ending warnings only
rg forbidden import check
Result: no forbidden import matches in src, tests, scripts, docs, or data

v1 Action-Sensitive Training and Evaluation Calibration
python -m unittest tests.test_model_explorer.BaselineEvaluationTests.test_action_sensitive_metrics_and_oracle_regret_ignore_unreachable_candidates -v
Result: initially failed for missing `oracle_actions`, then passed after implementation
python -m unittest tests.test_quasi_real_data_pipeline.QuasiRealEvaluationMatrixTests.test_sample_discriminativeness_warns_when_candidate_spread_is_low -v
Result: initially failed for missing `_sample_discriminativeness_summary`, then passed after implementation
python -m unittest tests.test_quasi_real_data_pipeline.QuasiRealEvaluationMatrixTests.test_selection_report_summarizes_architecture_decision_and_quality_gates -v
Result: initially failed for missing action-sensitive / oracle-regret report fields, then passed after implementation
python -m unittest discover -s tests -v
Result: passed 136 tests
$env:PYTHONPATH='src'; python -m model_explorer quasi-real run data\manifests\lunar_south_pole_lro_lola_selection_matrix_v1.json
Result: completed; selection remained inconclusive; action-sensitive and oracle-regret summaries covered `mlp_v1`, `mlp_missing_v1`, and `candidate_attention_v1`; per-group action outcomes covered all 4 ROI groups; sample discriminativeness warned for low risk/path spread and low oracle-vs-heuristic action disagreement
$env:PYTHONPATH='src'; python -m model_explorer verify
Result: passed; 136 unittest cases, benchmark smoke, forbidden import scan over 44 files, and git_diff_check returned 0
git diff --check
Result: returned 0; Windows line-ending warnings only
rg forbidden import check
Result: no forbidden imports
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
| Selection gate can be inconclusive | Current quasi-real selection run ties architectures on validation coverage, so a default model choice would be weaker than the measured variance | Report `inconclusive` and keep `mlp_v1` as compatibility default until broader data or a stronger metric separates architectures |
| Action disagreement may not imply policy value | Current decision diagnostics can show different selected actions while coverage/composite metrics remain tied | Treat agreement/disagreement as explanatory signal and require metric separation before recommending a default architecture |
| Current quasi-real mask-stress selection has low risk/path spread | Action-sensitive regret can still be small or tied when augmented samples remove risk/path fields or candidates are nearly equivalent | Report sample-discriminativeness warnings and avoid treating these rows as performance-selection evidence |
| Path-feedback diagnostics can be overread as performance improvement | Failure, replan, IRIS fallback, region-graph disconnect/fallback, and open-grid fallback are execution-side diagnostics, not proof of real-world performance | Use them only for calibration, exclusion, or downweighting; keep `not real-world generalization benchmark` on quasi-real/mask-stress summaries |

## Completion Criteria

- TODO file has every v1.1 task marked complete.
- Progress table records evidence for each completed task.
- `mlp_v1`, `mlp_missing_v1`, and `candidate_attention_v1` can be selected by manifest.
- All architectures preserve action mask safety.
- Reports include architecture identity and baseline deltas.
- Selection reports include `recommended_architecture` or `inconclusive`.
- Checkpoint metadata, JSON summary, and Markdown report include parsed `architecture_config`.
- `candidate_attention_v1` passes mask invariance tests for unreachable, padding, and reordered candidates.
- Final verification commands pass.
