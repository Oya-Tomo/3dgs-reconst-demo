import json
from pathlib import Path

import pytest

from dataset import DatasetValidationError, validate_dataset

IDENTITY = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
]


def write_transforms(dataset_path: Path, transforms: dict[str, object]) -> None:
    (dataset_path / "transforms.json").write_text(json.dumps(transforms), encoding="utf-8")


def test_validate_dataset_with_point_cloud(tmp_path: Path) -> None:
    image_path = tmp_path / "images" / "frame.jpg"
    image_path.parent.mkdir()
    image_path.touch()
    point_cloud = tmp_path / "sparse_pc.ply"
    point_cloud.touch()
    write_transforms(
        tmp_path,
        {
            "frames": [{"file_path": "images/frame.jpg", "transform_matrix": IDENTITY}],
            "ply_file_path": "sparse_pc.ply",
        },
    )

    report = validate_dataset(tmp_path)

    assert report.frame_count == 1
    assert report.point_cloud_path == point_cloud
    assert report.warnings == ()


def test_validate_dataset_warns_without_point_cloud(tmp_path: Path) -> None:
    image_path = tmp_path / "frame.jpg"
    image_path.touch()
    write_transforms(tmp_path, {"frames": [{"file_path": "frame.jpg", "transform_matrix": IDENTITY}]})

    report = validate_dataset(tmp_path)

    assert report.point_cloud_path is None
    assert "random points" in report.warnings[0]


def test_validate_dataset_rejects_missing_image(tmp_path: Path) -> None:
    write_transforms(tmp_path, {"frames": [{"file_path": "missing.jpg", "transform_matrix": IDENTITY}]})

    with pytest.raises(DatasetValidationError, match="referenced by transforms.json"):
        validate_dataset(tmp_path)


def test_validate_dataset_rejects_invalid_matrix(tmp_path: Path) -> None:
    (tmp_path / "frame.jpg").touch()
    write_transforms(tmp_path, {"frames": [{"file_path": "frame.jpg", "transform_matrix": [[1, 0], [0, 1]]}]})

    with pytest.raises(DatasetValidationError, match="4x4"):
        validate_dataset(tmp_path)
