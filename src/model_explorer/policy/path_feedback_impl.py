"""Compatibility re-export for migrated path-feedback symbols."""

_MODULES = (
    "path_feedback_manifest",
    "path_feedback_summary",
    "path_feedback_reports",
    "path_feedback_diagnostics",
    "path_feedback_artifacts",
    "feedback_selection",
    "path_feedback_runner",
)

for _module_name in _MODULES:
    _module = __import__(f"{__package__}.{_module_name}", fromlist=["*"])
    for _name in dir(_module):
        if not _name.startswith("__"):
            globals()[_name] = getattr(_module, _name)

del _module
del _module_name
del _name

__all__ = [_name for _name in globals() if not _name.startswith("__")]
