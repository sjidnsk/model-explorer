from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


CUDA_REQUESTED_BUT_UNAVAILABLE = "cuda_requested_but_unavailable"
TRAINING_DEVICE_CONTRACT_INVALID = "training_device_contract_invalid"
VALID_TRAINING_DEVICES = ("cpu", "cuda", "auto")


@dataclass(frozen=True)
class ResolvedTrainingDevice:
    requested_device: str
    resolved_device: str
    cuda_available: bool
    cuda_device_name: str | None
    fallback_to_cpu: bool
    reason_codes: tuple[str, ...] = ()

    def to_summary(self) -> dict[str, Any]:
        return asdict(self)


def resolve_training_device(
    requested_device: str | None = None,
    *,
    torch_module=None,
) -> ResolvedTrainingDevice:
    requested = "cpu" if requested_device is None else str(requested_device).strip().lower()
    if not requested:
        requested = "cpu"
    if requested not in VALID_TRAINING_DEVICES:
        return ResolvedTrainingDevice(
            requested_device=requested,
            resolved_device="cpu",
            cuda_available=False,
            cuda_device_name=None,
            fallback_to_cpu=False,
            reason_codes=(TRAINING_DEVICE_CONTRACT_INVALID,),
        )

    torch = torch_module if torch_module is not None else _load_torch()
    cuda_available = bool(torch.cuda.is_available())
    cuda_device_name = _cuda_device_name(torch) if cuda_available else None
    if requested == "cpu":
        return ResolvedTrainingDevice(
            requested_device=requested,
            resolved_device="cpu",
            cuda_available=cuda_available,
            cuda_device_name=cuda_device_name,
            fallback_to_cpu=False,
        )
    if requested == "auto":
        return ResolvedTrainingDevice(
            requested_device=requested,
            resolved_device="cuda" if cuda_available else "cpu",
            cuda_available=cuda_available,
            cuda_device_name=cuda_device_name,
            fallback_to_cpu=not cuda_available,
        )
    if cuda_available:
        return ResolvedTrainingDevice(
            requested_device=requested,
            resolved_device="cuda",
            cuda_available=True,
            cuda_device_name=cuda_device_name,
            fallback_to_cpu=False,
        )
    return ResolvedTrainingDevice(
        requested_device=requested,
        resolved_device="cuda",
        cuda_available=False,
        cuda_device_name=None,
        fallback_to_cpu=False,
        reason_codes=(CUDA_REQUESTED_BUT_UNAVAILABLE,),
    )


def _cuda_device_name(torch) -> str | None:
    try:
        return str(torch.cuda.get_device_name(0))
    except Exception:  # noqa: BLE001 - device name is diagnostic only.
        return None


def _load_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for policy training device resolution") from exc
    return torch
