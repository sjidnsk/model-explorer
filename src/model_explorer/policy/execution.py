from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class ExecutionFeasibilityRequest:
    schema_version: str
    step_index: int
    action_index: int
    selected_cell: tuple[int, int]
    selected_world: tuple[float, float]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExecutionFeasibilityResponse:
    feasible: bool
    failure_reason: str | None = None
    replan_required: bool = False
    metrics: dict[str, Any] = field(default_factory=dict)


class ExecutionFeasibilityAdapter(Protocol):
    def check_feasibility(self, request: ExecutionFeasibilityRequest) -> ExecutionFeasibilityResponse:
        ...


class FakeExecutionFeasibilityAdapter:
    def __init__(
        self,
        *,
        failing_cells: Mapping[tuple[int, int], str] | None = None,
        default_failure_reason: str = "execution_infeasible",
    ) -> None:
        self._failing_cells = dict(failing_cells or {})
        self._default_failure_reason = default_failure_reason

    def check_feasibility(self, request: ExecutionFeasibilityRequest) -> ExecutionFeasibilityResponse:
        failure_reason = self._failing_cells.get(request.selected_cell)
        if failure_reason is None:
            return ExecutionFeasibilityResponse(
                feasible=True,
                metrics={"adapter": "fake", "selected_world": list(request.selected_world)},
            )

        return ExecutionFeasibilityResponse(
            feasible=False,
            failure_reason=failure_reason or self._default_failure_reason,
            replan_required=True,
            metrics={"adapter": "fake", "selected_world": list(request.selected_world)},
        )
