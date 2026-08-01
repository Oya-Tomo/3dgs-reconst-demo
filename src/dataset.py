"""Validation helpers for a Nerfstudio-format dataset."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class DatasetValidationError(ValueError):
    """Raised when a dataset cannot be consumed safely by Nerfstudio."""


@dataclass(frozen=True, slots=True)
class DatasetReport:
    """Summary of a validated dataset."""

    dataset_path: Path
    frame_count: int
    point_cloud_path: Path | None
    warnings: tuple[str, ...]


def _load_transforms(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise DatasetValidationError(f"transforms.json does not exist: {path}") from error
    except json.JSONDecodeError as error:
        raise DatasetValidationError(f"transforms.json is invalid JSON: {error}") from error

    if not isinstance(value, dict):
        raise DatasetValidationError("The transforms.json root must be a JSON object.")
    return value


def _validate_transform_matrix(matrix: object, frame_index: int) -> None:
    if not isinstance(matrix, list) or len(matrix) != 4:
        raise DatasetValidationError(f"frames[{frame_index}].transform_matrix must be a 4x4 matrix.")
    for row in matrix:
        if not isinstance(row, list) or len(row) != 4 or not all(isinstance(value, int | float) for value in row):
            raise DatasetValidationError(f"frames[{frame_index}].transform_matrix must be a numeric 4x4 matrix.")


def validate_dataset(dataset_path: Path) -> DatasetReport:
    """Validate metadata and every referenced image before a costly training run."""

    dataset_path = dataset_path.expanduser().resolve()
    transforms = _load_transforms(dataset_path / "transforms.json")
    frames = transforms.get("frames")
    if not isinstance(frames, list) or not frames:
        raise DatasetValidationError("transforms.json must contain at least one frame.")

    missing_images: list[Path] = []
    for index, frame in enumerate(frames):
        if not isinstance(frame, dict):
            raise DatasetValidationError(f"frames[{index}] must be a JSON object.")
        file_path = frame.get("file_path")
        if not isinstance(file_path, str) or not file_path:
            raise DatasetValidationError(f"frames[{index}].file_path is missing.")
        image_path = (dataset_path / file_path).resolve()
        if not image_path.is_file():
            missing_images.append(image_path)
        _validate_transform_matrix(frame.get("transform_matrix"), index)

    if missing_images:
        sample = "\n".join(f"  - {path}" for path in missing_images[:5])
        suffix = f"\n  ... and {len(missing_images) - 5} more" if len(missing_images) > 5 else ""
        raise DatasetValidationError(f"Images referenced by transforms.json are missing:\n{sample}{suffix}")

    warnings: list[str] = []
    point_cloud_path: Path | None = None
    ply_file_path = transforms.get("ply_file_path")
    if isinstance(ply_file_path, str) and ply_file_path:
        candidate = (dataset_path / ply_file_path).resolve()
        if candidate.is_file():
            point_cloud_path = candidate
        else:
            warnings.append(f"The point cloud referenced by ply_file_path does not exist: {candidate}")
    else:
        warnings.append("ply_file_path is absent; Splatfacto will initialize from random points.")

    return DatasetReport(
        dataset_path=dataset_path,
        frame_count=len(frames),
        point_cloud_path=point_cloud_path,
        warnings=tuple(warnings),
    )
