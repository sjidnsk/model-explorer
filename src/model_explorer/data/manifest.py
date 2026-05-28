from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class DataManifestError(ValueError):
    """Raised when a local data manifest does not match local raw files."""


@dataclass(frozen=True)
class DataManifestIssue:
    code: str
    path: str
    message: str
    expected: Any = None
    actual: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {
            key: value
            for key, value in {
                "code": self.code,
                "path": self.path,
                "message": self.message,
                "expected": self.expected,
                "actual": self.actual,
            }.items()
            if value is not None
        }


@dataclass(frozen=True)
class DataManifestValidationResult:
    status: str
    dataset_id: str
    data_class: str
    manifest_path: str
    raw_dir: str
    checked_file_count: int
    total_bytes: int
    issues: tuple[DataManifestIssue, ...]

    def require_valid(self) -> None:
        if self.status == "passed":
            return
        issue_text = "; ".join(f"{issue.code}: {issue.path}" for issue in self.issues)
        raise DataManifestError(f"{self.dataset_id} manifest validation failed: {issue_text}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "dataset_id": self.dataset_id,
            "data_class": self.data_class,
            "manifest_path": self.manifest_path,
            "raw_dir": self.raw_dir,
            "checked_file_count": self.checked_file_count,
            "total_bytes": self.total_bytes,
            "issues": [issue.to_dict() for issue in self.issues],
        }


def load_data_manifest(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise DataManifestError("data manifest root must be a JSON object")
    return payload


def validate_data_manifest(
    path: str | Path,
    *,
    project_root: str | Path | None = None,
    verify_hashes: bool = True,
) -> DataManifestValidationResult:
    manifest_path = Path(path).resolve()
    payload = load_data_manifest(manifest_path)
    dataset_id = str(payload.get("dataset_id", manifest_path.stem))
    data_class = str(payload.get("data_class", "unknown"))
    root = Path(project_root).resolve() if project_root is not None else _infer_project_root(manifest_path)
    raw_dir = _resolve_raw_dir(payload.get("local_raw_dir", ""), root=root)

    issues: list[DataManifestIssue] = []
    checked_file_count = 0
    total_bytes = 0
    for file_info in _manifest_files(payload):
        file_name = str(file_info.get("name", ""))
        file_path = raw_dir / file_name
        display_path = str(file_path)
        if not file_name:
            issues.append(
                DataManifestIssue(
                    code="file_name_missing",
                    path=str(raw_dir),
                    message="manifest product file entry is missing name",
                )
            )
            continue
        if not file_path.exists():
            issues.append(
                DataManifestIssue(
                    code="file_missing",
                    path=display_path,
                    message=f"raw file is missing: {file_name}",
                    expected="present",
                    actual="missing",
                )
            )
            continue
        if not file_path.is_file():
            issues.append(
                DataManifestIssue(
                    code="not_a_file",
                    path=display_path,
                    message=f"raw path is not a file: {file_name}",
                    expected="file",
                    actual="non-file",
                )
            )
            continue

        checked_file_count += 1
        actual_bytes = file_path.stat().st_size
        total_bytes += actual_bytes
        expected_bytes = file_info.get("bytes")
        if expected_bytes is not None and int(expected_bytes) != actual_bytes:
            issues.append(
                DataManifestIssue(
                    code="bytes_mismatch",
                    path=display_path,
                    message=f"raw file byte count mismatch: {file_name}",
                    expected=int(expected_bytes),
                    actual=actual_bytes,
                )
            )
        expected_sha256 = file_info.get("sha256")
        if verify_hashes and expected_sha256:
            actual_sha256 = _sha256(file_path)
            if str(expected_sha256).upper() != actual_sha256:
                issues.append(
                    DataManifestIssue(
                        code="sha256_mismatch",
                        path=display_path,
                        message=f"raw file sha256 mismatch: {file_name}",
                        expected=str(expected_sha256).upper(),
                        actual=actual_sha256,
                    )
                )

    return DataManifestValidationResult(
        status="failed" if issues else "passed",
        dataset_id=dataset_id,
        data_class=data_class,
        manifest_path=str(manifest_path),
        raw_dir=str(raw_dir),
        checked_file_count=checked_file_count,
        total_bytes=total_bytes,
        issues=tuple(issues),
    )


def _manifest_files(payload: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    products = payload.get("products", [])
    if not isinstance(products, list):
        return ()
    files: list[dict[str, Any]] = []
    for product in products:
        if not isinstance(product, dict):
            continue
        product_files = product.get("files", [])
        if not isinstance(product_files, list):
            continue
        files.extend(file_info for file_info in product_files if isinstance(file_info, dict))
    return tuple(files)


def _resolve_raw_dir(value: Any, *, root: Path) -> Path:
    raw_dir = Path(str(value))
    return raw_dir.resolve() if raw_dir.is_absolute() else (root / raw_dir).resolve()


def _infer_project_root(path: Path) -> Path:
    for parent in (path.parent, *path.parents):
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
            return parent
    return path.parent


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()
