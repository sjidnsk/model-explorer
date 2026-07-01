"""Legacy compatibility facade for path-feedback APIs."""

from . import path_feedback_impl as _impl

_EXPORT_NAMES = tuple(getattr(_impl, "__all__", ()))
_exports = {_name: getattr(_impl, _name) for _name in _EXPORT_NAMES}

globals().update(_exports)
__all__ = _EXPORT_NAMES

del _EXPORT_NAMES
del _exports
del _impl
