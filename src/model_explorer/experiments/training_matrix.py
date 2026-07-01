"""Training matrix helpers for experiment manifests."""

from .experiment_impl import (
    _run_training as run_training,
    _training_architectures as training_architectures,
    _training_distillation_matrix as training_distillation_matrix,
    _training_output_path as training_output_path,
    _training_seeds as training_seeds,
    _training_source_selection_strategies as training_source_selection_strategies,
    _training_teacher_imitation_weights as training_teacher_imitation_weights,
    _training_teacher_margin_curriculum_profiles as training_teacher_margin_curriculum_profiles,
)

__all__ = [
    "run_training",
    "training_architectures",
    "training_distillation_matrix",
    "training_output_path",
    "training_seeds",
    "training_source_selection_strategies",
    "training_teacher_imitation_weights",
    "training_teacher_margin_curriculum_profiles",
]
