"""Legacy compatibility facade for path-feedback APIs."""

from . import path_feedback_impl as _impl

for _name in dir(_impl):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_impl, _name)

del _name
del _impl

__all__ = [_name for _name in globals() if not _name.startswith("__")]
