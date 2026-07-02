import subprocess
import sys
import textwrap
from pathlib import Path


PROJECT_ROOT = Path(__file__).parents[2]
SRC_ROOT = PROJECT_ROOT / "src"


def test_training_dimension_helpers_remain_available_through_legacy_aliases() -> None:
    from model_explorer.experiments import training_dimensions, training_matrix

    config = {
        "seeds": [3, "5"],
        "source_selection_strategies": ["coverage_heuristic", "feedback_aware"],
        "teacher_imitation_weights": [0.0, "0.25"],
        "teacher_margin_curriculum_profiles": ["high_only", {"name": "custom", "bucket_weights": {"high": 1}}],
        "architectures": ["mlp_v1", "attention_v1"],
        "architecture_configs": {"attention_v1": {"heads": 2}},
    }

    assert training_matrix.training_seeds(config) == training_dimensions.training_seeds(config) == (3, 5)
    assert training_matrix.training_source_selection_strategies(
        config
    ) == training_dimensions.training_source_selection_strategies(config)
    assert training_matrix.training_teacher_imitation_weights(
        config
    ) == training_dimensions.training_teacher_imitation_weights(config)
    assert training_matrix.training_teacher_margin_curriculum_profiles(
        config
    ) == training_dimensions.training_teacher_margin_curriculum_profiles(config)
    assert training_matrix.training_architectures(config) == training_dimensions.training_architectures(config)
    assert training_matrix._training_architecture_config(config, "attention_v1") == {"heads": 2}


def test_training_output_path_helper_remains_available_through_legacy_alias() -> None:
    from model_explorer.experiments import training_matrix, training_outputs

    config = {"checkpoint": "runs/{architecture}/checkpoint.pt"}
    kwargs = {
        "seed": 7,
        "architecture": "attention_v1",
        "selection_strategy": "feedback_aware",
        "teacher_imitation_weight": 0.25,
        "curriculum_profile": "soft_all_valid",
        "base_dir": PROJECT_ROOT,
        "run_output_dir": None,
        "default_name": "checkpoint.pt",
        "multi_seed": True,
        "multi_architecture": True,
        "multi_source": True,
        "multi_teacher_weight": True,
        "multi_curriculum_profile": True,
        "required": True,
    }

    assert training_matrix.training_output_path(config, "checkpoint", **kwargs) == training_outputs.training_output_path(
        config,
        "checkpoint",
        **kwargs,
    )


def test_training_matrix_import_keeps_torch_lazy_after_split() -> None:
    code = textwrap.dedent(
        f"""
        import sys
        sys.path.insert(0, {str(SRC_ROOT)!r})
        import model_explorer.experiments.training_matrix
        raise SystemExit(1 if "torch" in sys.modules else 0)
        """
    )

    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
