"""Compatibility re-export for legacy planning implementation imports."""

from __future__ import annotations

from importlib import import_module as _import_module

_PUBLIC_MODULES = (
    "planning_types",
    "planning_routes",
    "planning_backend_summaries",
    "planning_platform_feasibility",
    "planning_diagnostic_interpretation",
    "planning_anchor_evaluation",
    "planning_anchor_projection",
    "planning_anchor_grid",
    "planning_adapters",
    "planning_utils",
)

_TEMPORARY_EXPORT_NAMES = {
    "Any",
    "Counter",
    "Path",
    "Protocol",
    "Sequence",
    "annotations",
    "ceil",
    "dataclass",
    "deque",
    "field",
    "heappop",
    "heappush",
    "hypot",
    "json",
    "os",
    "subprocess",
    "sys",
    "tempfile",
}

_exports = {}
for _module_name in _PUBLIC_MODULES:
    _module = _import_module(f"{__package__}.{_module_name}")
    for _name in getattr(_module, "__all__", ()):
        if _name in _TEMPORARY_EXPORT_NAMES:
            continue
        if hasattr(_module, _name):
            _exports.setdefault(_name, getattr(_module, _name))

globals().update(_exports)
__all__ = tuple(sorted(_exports))

del _module
del _module_name
del _name
del _exports
del _import_module
del _PUBLIC_MODULES
del _TEMPORARY_EXPORT_NAMES
