"""Compatibility re-export for migrated implementation symbols."""

from importlib import import_module as _import_module

_runner = _import_module(f"{__package__}.runner")
_EXPORT_NAMES = tuple(getattr(_runner, "__all__", ()))
_exports = {name: getattr(_runner, name) for name in _EXPORT_NAMES}

globals().update(_exports)
__all__ = _EXPORT_NAMES

del _EXPORT_NAMES
del _exports
del _import_module
del _runner
