"""Legacy compatibility facade for experiment orchestration APIs."""

from importlib import import_module as _import_module

_impl = _import_module("model_explorer.experiments.experiment_impl")
_EXPORT_NAMES = tuple(getattr(_impl, "__all__", ()))
_exports = {name: getattr(_impl, name) for name in _EXPORT_NAMES}

globals().update(_exports)
__all__ = _EXPORT_NAMES

del _EXPORT_NAMES
del _exports
del _import_module
del _impl
