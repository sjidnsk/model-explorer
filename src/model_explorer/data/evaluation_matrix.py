"""Legacy compatibility facade for quasi-real evaluation matrix APIs."""

from ..experiments.quasi_real_matrix import evaluation_matrix_impl as _impl

for _name in dir(_impl):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_impl, _name)

__all__ = [_name for _name in globals() if not _name.startswith("__")]
