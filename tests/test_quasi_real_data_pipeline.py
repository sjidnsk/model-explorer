from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class DataManifestValidationTests(unittest.TestCase):
    def test_manifest_validation_checks_bytes_hash_and_missing_files(self):
        from model_explorer.data.manifest import validate_data_manifest

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_dir = root / "raw"
            raw_dir.mkdir()
            dem_bytes = b"dem"
            count_bytes = b"count"
            (raw_dir / "dem.jp2").write_bytes(dem_bytes)
            (raw_dir / "count.jp2").write_bytes(count_bytes)
            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "dataset_id": "fixture_lola",
                        "data_class": "quasi_real",
                        "local_raw_dir": str(raw_dir),
                        "products": [
                            {
                                "product_id": "DEM",
                                "role": "shape_map_radius",
                                "files": [
                                    {
                                        "name": "dem.jp2",
                                        "bytes": len(dem_bytes),
                                        "sha256": hashlib.sha256(dem_bytes).hexdigest().upper(),
                                    }
                                ],
                            },
                            {
                                "product_id": "COUNT",
                                "role": "observation_count",
                                "files": [
                                    {
                                        "name": "count.jp2",
                                        "bytes": len(count_bytes),
                                        "sha256": hashlib.sha256(count_bytes).hexdigest().upper(),
                                    }
                                ],
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            valid = validate_data_manifest(manifest_path)
            self.assertEqual(valid.status, "passed")
            self.assertEqual(valid.dataset_id, "fixture_lola")
            self.assertEqual(valid.data_class, "quasi_real")
            self.assertEqual(valid.checked_file_count, 2)
            self.assertEqual(valid.total_bytes, len(dem_bytes) + len(count_bytes))
            self.assertEqual(valid.issues, ())

            (raw_dir / "count.jp2").write_bytes(b"wrong-data")
            (raw_dir / "dem.jp2").unlink()

            invalid = validate_data_manifest(manifest_path)
            self.assertEqual(invalid.status, "failed")
            self.assertEqual({issue.code for issue in invalid.issues}, {"file_missing", "sha256_mismatch", "bytes_mismatch"})
            self.assertTrue(all(issue.message for issue in invalid.issues))

    def test_manifest_require_valid_raises_readable_error(self):
        from model_explorer.data.manifest import DataManifestError, validate_data_manifest

        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "dataset_id": "fixture_lola",
                        "data_class": "quasi_real",
                        "local_raw_dir": "missing",
                        "products": [
                            {
                                "product_id": "DEM",
                                "role": "shape_map_radius",
                                "files": [{"name": "dem.jp2", "bytes": 3, "sha256": "0"}],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            result = validate_data_manifest(manifest_path)
            with self.assertRaisesRegex(DataManifestError, "fixture_lola.*file_missing.*dem.jp2"):
                result.require_valid()


class RasterAdapterTests(unittest.TestCase):
    def test_raster_window_reads_finite_values_without_importing_policy_network(self):
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("Pillow is not available in this environment")
        from model_explorer.data.raster import read_raster_window

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "fixture.jp2"
            image = Image.new("L", (4, 3))
            image.putdata([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
            image.save(path, format="PNG")

            window = read_raster_window(path, x=1, y=1, width=2, height=2)

        self.assertEqual(window.width, 2)
        self.assertEqual(window.height, 2)
        self.assertEqual(window.values, ((6.0, 7.0), (10.0, 11.0)))
        self.assertTrue(all(math.isfinite(value) for row in window.values for value in row))


class LolaSouthPoleGenerationTests(unittest.TestCase):
    def test_single_contract_top_level_metadata_is_not_treated_as_scenario_provenance(self):
        from model_explorer.io.scenario import load_scenario

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "single-contract.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": "model-explorer-contract/v1",
                        "metadata": "legacy contract-side note",
                        "grid": {
                            "width": 2,
                            "height": 2,
                            "resolution": 1.0,
                            "frame_id": "moon_local",
                            "origin": [0.0, 0.0],
                            "layers": ["confidence"],
                        },
                        "constraints": {
                            "violation_count": 0,
                            "passable_ratio": 1.0,
                            "reason_counts": {},
                        },
                        "top_goals": [{"cell": [1, 1], "utility": 0.5, "reachable": True}],
                        "top_sequences": [{"cells": [[1, 1]], "utility": 0.5, "coverage_area": 1.0}],
                        "observation_update": {"coverage_rate": 0.1, "coverage_rate_delta": 0.1},
                    }
                ),
                encoding="utf-8",
            )

            scenario = load_scenario(path)

        self.assertEqual(len(scenario.snapshots), 1)
        self.assertEqual(scenario.metadata, {})

    def test_array_generator_outputs_mask_safe_contract_and_quasi_real_rollout_jsonl(self):
        from model_explorer.data.lola_south_pole import (
            LolaSouthPoleRoiConfig,
            generate_lola_south_pole_scenarios,
            write_lola_south_pole_rollouts_jsonl,
        )
        from model_explorer.policy.dataset import summarize_rollout_dataset
        from model_explorer.policy.features import extract_policy_observation
        from model_explorer.policy.rollout_io import read_rollout_episodes

        dem = (
            (100.0, 101.0, 103.0, 110.0, 125.0),
            (100.5, 101.5, 104.0, 112.0, 135.0),
            (99.5, 101.0, 103.5, 116.0, 145.0),
            (98.0, 100.0, 102.0, 122.0, 160.0),
            (97.0, 99.0, 101.0, 130.0, 180.0),
        )
        counts = (
            (8.0, 8.0, 7.0, 1.0, 0.0),
            (8.0, 7.0, 6.0, 1.0, 0.0),
            (7.0, 7.0, 5.0, 1.0, 0.0),
            (6.0, 5.0, 4.0, 1.0, 0.0),
            (5.0, 4.0, 3.0, 0.0, 0.0),
        )
        config = LolaSouthPoleRoiConfig(
            roi_x=0,
            roi_y=0,
            roi_width=5,
            roi_height=5,
            candidate_count=4,
            episode_count=2,
            seed=7,
        )

        scenarios = generate_lola_south_pole_scenarios(
            dem,
            counts,
            dataset_id="fixture_lola",
            data_class="quasi_real",
            region="lunar_south_pole",
            resolution=20.0,
            config=config,
        )
        self.assertEqual(len(scenarios), 2)
        contract = scenarios[0].snapshots[0]
        self.assertEqual(contract.schema_version, "model-explorer-contract/v1")
        self.assertLessEqual(len(contract.top_goals), 4)
        observation = extract_policy_observation(contract, max_candidates=6)
        self.assertEqual(observation.action_mask[-2:], (False, False))
        self.assertIn(False, observation.action_mask[: len(contract.top_goals)])
        self.assertTrue(any(observation.action_mask[: len(contract.top_goals)]))
        for row in observation.candidate_features:
            self.assertTrue(all(math.isfinite(value) for value in row))

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "rollouts.jsonl"
            write_lola_south_pole_rollouts_jsonl(
                output,
                dem,
                counts,
                dataset_id="fixture_lola",
                data_class="quasi_real",
                region="lunar_south_pole",
                resolution=20.0,
                config=config,
            )
            episodes = read_rollout_episodes(output)

        summary = summarize_rollout_dataset(episodes)
        self.assertEqual(summary["data_class"], "quasi_real")
        self.assertEqual(summary["dataset_id"], "fixture_lola")
        self.assertEqual(summary["region"], "lunar_south_pole")
        self.assertEqual(summary["episode_count"], 2)
        self.assertGreaterEqual(summary["trainable_transition_count"], 2)
        self.assertEqual(summary["non_finite_reward_count"], 0)
        self.assertEqual(episodes[0].transitions[0].info.extra["provenance"]["dataset_id"], "fixture_lola")

    def test_manifest_generator_reads_raster_window_and_writes_rollout_jsonl(self):
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("Pillow is not available in this environment")
        from model_explorer.data.lola_south_pole import (
            LolaSouthPoleRoiConfig,
            write_lola_south_pole_rollouts_from_manifest_jsonl,
        )
        from model_explorer.policy.dataset import summarize_rollout_dataset
        from model_explorer.policy.rollout_io import read_rollout_episodes

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_dir = root / "raw"
            raw_dir.mkdir()
            dem_path = raw_dir / "dem.jp2"
            count_path = raw_dir / "count.jp2"
            dem_image = Image.new("L", (4, 4))
            dem_image.putdata([0, 1, 2, 8, 0, 1, 3, 10, 1, 2, 4, 12, 1, 3, 6, 20])
            dem_image.save(dem_path, format="PNG")
            count_image = Image.new("L", (4, 4))
            count_image.putdata([9, 8, 3, 0, 9, 8, 3, 0, 8, 7, 2, 0, 7, 6, 1, 0])
            count_image.save(count_path, format="PNG")
            dem_bytes = dem_path.read_bytes()
            count_bytes = count_path.read_bytes()
            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "dataset_id": "fixture_manifest_lola",
                        "data_class": "quasi_real",
                        "region": "lunar_south_pole",
                        "local_raw_dir": str(raw_dir),
                        "projection": {"map_scale_meters_per_pixel": 20},
                        "products": [
                            {
                                "product_id": "DEM",
                                "role": "shape_map_radius",
                                "files": [
                                    {
                                        "name": "dem.jp2",
                                        "bytes": len(dem_bytes),
                                        "sha256": hashlib.sha256(dem_bytes).hexdigest().upper(),
                                    }
                                ],
                            },
                            {
                                "product_id": "COUNT",
                                "role": "observation_count",
                                "files": [
                                    {
                                        "name": "count.jp2",
                                        "bytes": len(count_bytes),
                                        "sha256": hashlib.sha256(count_bytes).hexdigest().upper(),
                                    }
                                ],
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            output = root / "rollouts.jsonl"

            episodes = write_lola_south_pole_rollouts_from_manifest_jsonl(
                manifest_path,
                output,
                config=LolaSouthPoleRoiConfig(
                    roi_x=0,
                    roi_y=0,
                    roi_width=4,
                    roi_height=4,
                    candidate_count=4,
                    episode_count=1,
                    seed=13,
                ),
            )
            loaded = read_rollout_episodes(output)

        summary = summarize_rollout_dataset(loaded)
        self.assertEqual(len(episodes), 1)
        self.assertEqual(summary["dataset_id"], "fixture_manifest_lola")
        self.assertEqual(summary["data_class"], "quasi_real")
        self.assertEqual(summary["region"], "lunar_south_pole")

    def test_training_smoke_runs_all_architectures_with_quasi_real_dataset_summary(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("PyTorch is not available in this environment")
        from model_explorer.data.lola_south_pole import (
            LolaSouthPoleRoiConfig,
            generate_lola_south_pole_rollout_episodes,
        )
        from model_explorer.policy.training import load_policy_checkpoint, train_policy_on_episodes

        dem = (
            (0.0, 1.0, 2.0, 6.0),
            (0.5, 1.5, 2.5, 7.0),
            (1.0, 2.0, 3.0, 9.0),
            (1.5, 2.5, 4.0, 12.0),
        )
        counts = (
            (9.0, 8.0, 3.0, 0.0),
            (9.0, 8.0, 3.0, 0.0),
            (8.0, 7.0, 2.0, 0.0),
            (7.0, 6.0, 1.0, 0.0),
        )
        episodes = generate_lola_south_pole_rollout_episodes(
            dem,
            counts,
            dataset_id="fixture_lola",
            data_class="quasi_real",
            region="lunar_south_pole",
            resolution=20.0,
            config=LolaSouthPoleRoiConfig(
                roi_x=0,
                roi_y=0,
                roi_width=4,
                roi_height=4,
                candidate_count=4,
                episode_count=3,
                seed=11,
            ),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            for architecture in ("mlp_v1", "mlp_missing_v1", "candidate_attention_v1"):
                checkpoint = Path(tmpdir) / f"{architecture}.pt"
                result = train_policy_on_episodes(
                    episodes,
                    checkpoint_path=checkpoint,
                    seed=3,
                    hidden_size=16,
                    learning_rate=0.001,
                    epochs=1,
                    architecture=architecture,
                    architecture_config={"dropout": 0.0},
                )
                scorer = load_policy_checkpoint(checkpoint)

                self.assertEqual(result["architecture"], architecture)
                self.assertEqual(result["dataset_summary"]["data_class"], "quasi_real")
                self.assertEqual(result["dataset_summary"]["dataset_id"], "fixture_lola")
                self.assertTrue(math.isfinite(result["loss"]))
                self.assertEqual(scorer.network.architecture_name, architecture)

    def test_experiment_report_includes_quasi_real_dataset_and_architecture_metadata(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("PyTorch is not available in this environment")
        from model_explorer.data.lola_south_pole import (
            LolaSouthPoleRoiConfig,
            write_lola_south_pole_scenarios_json,
        )
        from model_explorer.policy.experiment import run_experiment_manifest

        dem = (
            (0.0, 1.0, 2.0, 8.0),
            (0.0, 1.0, 3.0, 10.0),
            (1.0, 2.0, 4.0, 12.0),
            (1.0, 3.0, 6.0, 20.0),
        )
        counts = (
            (9.0, 8.0, 3.0, 0.0),
            (9.0, 8.0, 3.0, 0.0),
            (8.0, 7.0, 2.0, 0.0),
            (7.0, 6.0, 1.0, 0.0),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            scenario_paths = write_lola_south_pole_scenarios_json(
                root / "scenarios",
                dem,
                counts,
                dataset_id="fixture_report_lola",
                data_class="quasi_real",
                region="lunar_south_pole",
                resolution=20.0,
                config=LolaSouthPoleRoiConfig(
                    roi_x=0,
                    roi_y=0,
                    roi_width=4,
                    roi_height=4,
                    candidate_count=4,
                    episode_count=2,
                    seed=19,
                ),
            )
            manifest = root / "experiment.json"
            manifest.write_text(
                json.dumps(
                    {
                        "schema_version": "model-explorer-experiment/v1",
                        "name": "quasi-real-report-smoke",
                        "run_id": "fixture",
                        "scenarios": [str(path.relative_to(root)) for path in scenario_paths],
                        "max_candidates": 4,
                        "planner": {"backend": "contract_cost"},
                        "dataset_validation": {
                            "min_episode_count": 2,
                            "min_trainable_transition_count": 2,
                            "max_empty_action_mask_count": 0,
                            "max_invalid_action_mask_count": 0,
                            "require_finite_reward": True,
                        },
                        "train": {
                            "seed": 23,
                            "architecture": "mlp_v1",
                            "hidden_size": 16,
                            "learning_rate": 0.001,
                            "epochs": 1,
                            "evaluate_trained_policy": True,
                        },
                        "outputs": {"root": "out"},
                    }
                ),
                encoding="utf-8",
            )

            summary = run_experiment_manifest(manifest)
            run_dir = root / "out" / "quasi-real-report-smoke" / "fixture"
            report = (run_dir / "report.md").read_text(encoding="utf-8")

        self.assertEqual(summary["dataset_summary"]["data_class"], "quasi_real")
        self.assertEqual(summary["dataset_summary"]["dataset_id"], "fixture_report_lola")
        self.assertEqual(summary["training"]["architecture"], "mlp_v1")
        self.assertEqual(summary["training"]["dataset_summary"]["data_class"], "quasi_real")
        self.assertEqual(summary["training"]["dataset_summary"]["dataset_id"], "fixture_report_lola")
        self.assertIn("| data_class | quasi_real |", report)
        self.assertIn("| dataset_id | fixture_report_lola |", report)
        self.assertIn("| architecture | mlp_v1 |", report)


class QuasiRealEvaluationMatrixTests(unittest.TestCase):
    def test_stability_manifest_template_is_tracked_and_expands_roi_and_seed_coverage(self):
        from model_explorer.data.evaluation_matrix import load_quasi_real_evaluation_manifest

        manifest_path = ROOT / "data" / "manifests" / "lunar_south_pole_lro_lola_stability_matrix_v1.json"

        manifest = load_quasi_real_evaluation_manifest(manifest_path)
        roi_counts: dict[str, int] = {}
        for roi in manifest.rois:
            roi_counts[roi.name] = roi_counts.get(roi.name, 0) + 1

        self.assertTrue(manifest_path.exists())
        self.assertEqual(manifest.dataset_manifest.name, "lunar_south_pole_lro_lola_gdr_875s_20m.json")
        self.assertEqual(manifest.output_root.name, "qreal_stability_v1")
        self.assertEqual(manifest.output_root.parent.name, "processed")
        self.assertEqual(manifest.output_root.parent.parent.name, "data")
        self.assertEqual(set(manifest.train_config["architectures"]), {"mlp_v1", "mlp_missing_v1", "candidate_attention_v1"})
        self.assertGreaterEqual(len(manifest.train_config["seeds"]), 3)
        self.assertGreaterEqual(set(roi.split for roi in manifest.rois), {"train", "validation", "test"})
        for roi_name in ("smooth_high_confidence", "rim_or_steep_slope", "low_observation_count", "mixed_risk"):
            self.assertGreaterEqual(roi_counts.get(roi_name, 0), 2)

    def test_evaluation_manifest_validate_and_dry_run_do_not_write_outputs(self):
        from model_explorer.data.evaluation_matrix import (
            dry_run_quasi_real_evaluation_manifest,
            validate_quasi_real_evaluation_manifest,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            matrix_manifest = _write_fixture_matrix_manifest(Path(tmpdir))

            validation = validate_quasi_real_evaluation_manifest(matrix_manifest)
            dry_run = dry_run_quasi_real_evaluation_manifest(matrix_manifest)

            output_root = Path(tmpdir) / "matrix-out"
            self.assertFalse(output_root.exists())

        self.assertEqual(validation["status"], "valid")
        self.assertEqual(validation["roi_count"], 4)
        self.assertEqual(validation["data_class"], "quasi_real")
        self.assertEqual(set(validation["splits"]), {"train", "validation", "test", "benchmark"})
        self.assertEqual(dry_run["status"], "dry_run")
        self.assertEqual(dry_run["roi_count"], 4)
        self.assertEqual(dry_run["evaluation_scope"], "quasi-real evaluation; not real-world generalization benchmark")
        self.assertTrue(any(path.endswith("experiment.json") for path in dry_run["would_write"]))

    def test_quasi_real_cli_validate_and_dry_run_emit_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            matrix_manifest = _write_fixture_matrix_manifest(root)
            env = os.environ.copy()
            env["PYTHONPATH"] = str(SRC)

            validate = subprocess.run(
                [sys.executable, "-m", "model_explorer", "quasi-real", "validate", str(matrix_manifest)],
                cwd=ROOT,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
            dry_run = subprocess.run(
                [sys.executable, "-m", "model_explorer", "quasi-real", "dry-run", str(matrix_manifest)],
                cwd=ROOT,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )

            self.assertFalse((root / "matrix-out").exists())

        self.assertEqual(json.loads(validate.stdout)["status"], "valid")
        self.assertEqual(json.loads(dry_run.stdout)["status"], "dry_run")

    def test_evaluation_matrix_run_generates_multi_roi_report_and_architecture_matrix(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("PyTorch is not available in this environment")
        from model_explorer.data.evaluation_matrix import run_quasi_real_evaluation_manifest
        from model_explorer.io.scenario import load_scenario

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            matrix_manifest = _write_fixture_matrix_manifest(root)

            summary = run_quasi_real_evaluation_manifest(matrix_manifest)
            report = Path(summary["report_output"]).read_text(encoding="utf-8")
            rim_scenario = load_scenario(
                Path(summary["output_root"]) / summary["rois"][1]["scenarios"][0]
            )

        self.assertEqual(summary["status"], "completed")
        self.assertEqual(summary["roi_count"], 4)
        self.assertEqual(summary["dataset_id"], "fixture_matrix_lola")
        self.assertEqual(summary["data_class"], "quasi_real")
        self.assertEqual(summary["evaluation_scope"], "quasi-real evaluation; not real-world generalization benchmark")
        self.assertEqual(summary["experiment"]["training"]["architectures"], ["mlp_v1", "mlp_missing_v1", "candidate_attention_v1"])
        self.assertEqual(set(summary["experiment"]["architecture_deltas"]), {"mlp_v1", "mlp_missing_v1", "candidate_attention_v1"})
        self.assertEqual(summary["experiment"]["dataset_summary"]["roi_count"], 4)
        self.assertEqual(summary["experiment"]["dataset_summary"]["split_counts"]["train"], 2)
        self.assertEqual(rim_scenario.snapshots[0].grid.origin, (80.0, 80.0))
        for text in (
            "quasi-real evaluation; not real-world generalization benchmark",
            "## Quality Gates",
            "min_trainable_transition_count",
            "| train | 2 |",
            "| validation | 1 |",
            "| test | 1 |",
            "## Policy Ranking",
            "## Per-Group Winners",
            "## Architecture Delta Details",
            "smooth_high_confidence",
            "rim_or_steep_slope",
            "low_observation_count",
            "mixed_risk",
            "mlp_v1",
            "mlp_missing_v1",
            "candidate_attention_v1",
            "coverage_heuristic",
        ):
            self.assertIn(text, report)

    def test_stability_report_summarizes_multi_seed_architecture_metrics(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("PyTorch is not available in this environment")
        from model_explorer.data.evaluation_matrix import run_quasi_real_evaluation_manifest

        with tempfile.TemporaryDirectory() as tmpdir:
            matrix_manifest = _write_fixture_matrix_manifest(Path(tmpdir), seeds=[31, 37])

            summary = run_quasi_real_evaluation_manifest(matrix_manifest)
            report = Path(summary["report_output"]).read_text(encoding="utf-8")

        stability = summary["stability_summary"]
        self.assertEqual(set(stability["architectures"]), {"mlp_v1", "mlp_missing_v1", "candidate_attention_v1"})
        for architecture in ("mlp_v1", "mlp_missing_v1", "candidate_attention_v1"):
            self.assertEqual(stability["architectures"][architecture]["run_count"], 2)
            self.assertIn("loss", stability["architectures"][architecture])
            self.assertIn("torch_policy.total_path_cost", stability["architectures"][architecture])
        for text in (
            "## Architecture Stability",
            "## Loss Distribution",
            "## Baseline Delta Summary",
            "torch_policy.total_path_cost",
            "mlp_missing_v1",
            "candidate_attention_v1",
        ):
            self.assertIn(text, report)

    def test_matrix_report_warns_when_dataset_has_no_mask_stress_samples(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("PyTorch is not available in this environment")
        from model_explorer.data.evaluation_matrix import run_quasi_real_evaluation_manifest

        with tempfile.TemporaryDirectory() as tmpdir:
            matrix_manifest = _write_fixture_matrix_manifest(Path(tmpdir), all_reachable=True)

            summary = run_quasi_real_evaluation_manifest(matrix_manifest)
            report = Path(summary["report_output"]).read_text(encoding="utf-8")

        self.assertEqual(summary["experiment"]["dataset_summary"]["unreachable_candidate_count"], 0)
        self.assertIn("no_unreachable_candidates", summary["coverage_warnings"])
        self.assertIn("no_mask_stress_samples", summary["coverage_warnings"])
        self.assertEqual(summary["experiment"]["dataset_summary"]["non_finite_reward_count"], 0)
        self.assertIn("## Sample Coverage Warnings", report)
        self.assertIn("no_mask_stress_samples", report)


def _write_fixture_matrix_manifest(root: Path, *, seeds: list[int] | None = None, all_reachable: bool = False) -> Path:
    try:
        from PIL import Image
    except ImportError as exc:
        raise unittest.SkipTest("Pillow is not available in this environment") from exc

    raw_dir = root / "raw"
    raw_dir.mkdir()
    dem_path = raw_dir / "dem.jp2"
    count_path = raw_dir / "count.jp2"
    dem_values = []
    count_values = []
    for y in range(8):
        for x in range(8):
            dem_values.append(10 if all_reachable else (x + y) * 2 + (20 if x >= 4 and y >= 4 else 0))
            count_values.append(9 if all_reachable else max(0, 9 - x - (2 if y >= 4 else 0)))
    dem_image = Image.new("L", (8, 8))
    dem_image.putdata(dem_values)
    dem_image.save(dem_path, format="PNG")
    count_image = Image.new("L", (8, 8))
    count_image.putdata(count_values)
    count_image.save(count_path, format="PNG")
    dem_bytes = dem_path.read_bytes()
    count_bytes = count_path.read_bytes()
    data_manifest = root / "lola-manifest.json"
    data_manifest.write_text(
        json.dumps(
            {
                "dataset_id": "fixture_matrix_lola",
                "data_class": "quasi_real",
                "region": "lunar_south_pole",
                "local_raw_dir": str(raw_dir),
                "projection": {"map_scale_meters_per_pixel": 20},
                "products": [
                    {
                        "product_id": "DEM",
                        "role": "shape_map_radius",
                        "files": [
                            {
                                "name": "dem.jp2",
                                "bytes": len(dem_bytes),
                                "sha256": hashlib.sha256(dem_bytes).hexdigest().upper(),
                            }
                        ],
                    },
                    {
                        "product_id": "COUNT",
                        "role": "observation_count",
                        "files": [
                            {
                                "name": "count.jp2",
                                "bytes": len(count_bytes),
                                "sha256": hashlib.sha256(count_bytes).hexdigest().upper(),
                            }
                        ],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    matrix_manifest = root / "matrix.json"
    matrix_manifest.write_text(
        json.dumps(
            {
                "schema_version": "model-explorer-quasi-real-evaluation/v1",
                "name": "fixture-quasi-real-matrix",
                "run_id": "run-001",
                "dataset_manifest": str(data_manifest),
                "output_root": str(root / "matrix-out"),
                "candidate_count": 4,
                "episode_count": 1,
                "seed": 31,
                "rois": [
                    {"name": "smooth_high_confidence", "split": "train", "roi_x": 0, "roi_y": 0, "roi_width": 4, "roi_height": 4},
                    {"name": "rim_or_steep_slope", "split": "train", "roi_x": 4, "roi_y": 4, "roi_width": 4, "roi_height": 4},
                    {"name": "low_observation_count", "split": "validation", "roi_x": 4, "roi_y": 0, "roi_width": 4, "roi_height": 4},
                    {"name": "mixed_risk", "split": "test", "roi_x": 0, "roi_y": 4, "roi_width": 4, "roi_height": 4},
                    {"name": "mixed_risk", "split": "benchmark", "roi_x": 0, "roi_y": 4, "roi_width": 4, "roi_height": 4, "seed": 41},
                ],
                "dataset_validation": {
                    "min_trainable_transition_count": 4,
                    "require_finite_reward": True,
                    "min_reward_std": 0.0,
                    "min_action_mask_valid_mean": 0.1,
                    "max_unreachable_candidate_rate": 0.95,
                },
                "train": {
                    "seed": 31,
                    "architectures": ["mlp_v1", "mlp_missing_v1", "candidate_attention_v1"],
                    "hidden_size": 16,
                    "learning_rate": 0.001,
                    "epochs": 1,
                    "evaluate_trained_policy": True,
                    **({"seeds": seeds} if seeds is not None else {}),
                },
            }
        ),
        encoding="utf-8",
    )
    return matrix_manifest


if __name__ == "__main__":
    unittest.main()
