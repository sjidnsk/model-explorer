from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from ..core.interfaces import GoalCandidate, ModelExplorerContract


@dataclass(frozen=True)
class ProviderStepRequest:
    step_index: int
    contract: ModelExplorerContract
    action_index: int
    selected_goal: GoalCandidate | None
    failure_reason: str | None = None
    info: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProviderStepResult:
    next_contract: ModelExplorerContract | None = None
    done: bool = False
    failure_reason: str | None = None
    replan_reasons: tuple[str, ...] = ()
    info: dict[str, Any] = field(default_factory=dict)


class ContractProvider(Protocol):
    def initial_contract(self) -> ModelExplorerContract:
        ...

    def advance(self, request: ProviderStepRequest) -> ProviderStepResult:
        ...


class SequenceContractProvider:
    def __init__(
        self,
        contracts: Iterable[ModelExplorerContract],
        *,
        failures_by_step: Mapping[int, str] | None = None,
        replan_reasons_by_step: Mapping[int, Iterable[str]] | None = None,
        info_by_step: Mapping[int, Mapping[str, Any]] | None = None,
    ) -> None:
        self._contracts = tuple(contracts)
        if not self._contracts:
            raise ValueError("contracts must contain at least one contract")
        self._failures_by_step = dict(failures_by_step or {})
        self._replan_reasons_by_step = {
            int(step): tuple(str(reason) for reason in reasons)
            for step, reasons in (replan_reasons_by_step or {}).items()
        }
        self._info_by_step = {
            int(step): dict(info)
            for step, info in (info_by_step or {}).items()
        }
        self.total_steps = len(self._contracts)

    def initial_contract(self) -> ModelExplorerContract:
        return self._contracts[0]

    def advance(self, request: ProviderStepRequest) -> ProviderStepResult:
        next_index = request.step_index + 1
        next_contract = self._contracts[next_index] if next_index < len(self._contracts) else None
        return ProviderStepResult(
            next_contract=next_contract,
            done=next_contract is None,
            failure_reason=self._failures_by_step.get(request.step_index),
            replan_reasons=self._replan_reasons_by_step.get(request.step_index, ()),
            info=self._info_by_step.get(request.step_index, {}),
        )
