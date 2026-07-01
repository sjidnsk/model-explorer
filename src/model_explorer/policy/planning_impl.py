"""Compatibility re-export for legacy planning implementation imports."""

from __future__ import annotations

from . import planning_adapters as _planning_adapters
from . import planning_anchor as _planning_anchor
from . import planning_diagnostics as _planning_diagnostics
from . import planning_routes as _planning_routes
from . import planning_types as _planning_types
from . import planning_utils as _planning_utils

_MODULES = (
    _planning_types,
    _planning_routes,
    _planning_diagnostics,
    _planning_anchor,
    _planning_adapters,
    _planning_utils,
)

for _module in _MODULES:
    for _name in dir(_module):
        if _name.startswith("__"):
            continue
        globals()[_name] = getattr(_module, _name)

del _module
del _name
del _MODULES
del _planning_adapters
del _planning_anchor
del _planning_diagnostics
del _planning_routes
del _planning_types
del _planning_utils

__all__ = [_name for _name in globals() if not _name.startswith("__")]
