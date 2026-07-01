from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..core.interfaces import ContractValidationError


def required(payload: dict[str, Any], key: str, prefix: str) -> Any:
    if key not in payload:
        raise ContractValidationError(f"{prefix}.{key} is required")
    return payload[key]


def strict_bool(value: Any, prefix: str) -> bool:
    if not isinstance(value, bool):
        raise ContractValidationError(f"{prefix} must be a bool")
    return value


def strict_int(value: Any, prefix: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractValidationError(f"{prefix} must be an int")
    return value


def strict_float(value: Any, prefix: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractValidationError(f"{prefix} must be a float")
    return float(value)


def strict_string_list(value: Any, prefix: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ContractValidationError(f"{prefix} must be a list of strings")
    return tuple(value)


def cell_like(value: Any, prefix: str, *, item_type: type = int) -> tuple:
    if not isinstance(value, list) or len(value) != 2:
        raise ContractValidationError(f"{prefix} must be a list with 2 values")
    try:
        if item_type is int:
            return tuple(strict_int(item, f"{prefix}[{index}]") for index, item in enumerate(value))
        if item_type is float:
            return tuple(strict_float(item, f"{prefix}[{index}]") for index, item in enumerate(value))
        return tuple(item_type(item) for item in value)
    except ContractValidationError:
        raise
    except (TypeError, ValueError) as exc:
        raise ContractValidationError(f"{prefix} contains invalid values") from exc
