import json
from pathlib import Path

from export import create_portable_dataset, find_latest_checkpoint, find_training_config
from settings import ExportSettings

IDENTITY = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
]


def test_find_latest_training_config(tmp_path: Path) -> None:
    run_root = tmp_path / "experiment" / "splatfacto"
    older = run_root / "2026-01-01_000000" / "config.yml"
    newer = run_root / "2026-01-02_000000" / "config.yml"
    older.parent.mkdir(parents=True)
    newer.parent.mkdir(parents=True)
    older.touch()
    newer.touch()
    settings = ExportSettings(
        training_output_dir=tmp_path,
        experiment_name="experiment",
        output_dir=tmp_path / "viewer_output",
    )

    assert find_training_config(settings) == newer


def test_find_latest_checkpoint_uses_highest_training_step(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yml"
    checkpoint_dir = tmp_path / "nerfstudio_models"
    checkpoint_dir.mkdir()
    config_path.touch()
    older = checkpoint_dir / "step-000002000.ckpt"
    newer = checkpoint_dir / "step-000030000.ckpt"
    ignored = checkpoint_dir / "latest.ckpt"
    older.touch()
    newer.touch()
    ignored.touch()

    assert find_latest_checkpoint(config_path) == newer


def test_create_portable_dataset_preserves_camera_metadata_without_photos(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    destination_dir = tmp_path / "portable"
    images_dir = source_dir / "images"
    images_dir.mkdir(parents=True)
    (images_dir / "b.jpg").write_bytes(b"source-b")
    (images_dir / "a.jpg").write_bytes(b"source-a")
    (source_dir / "mask.png").touch()
    (source_dir / "depth.png").touch()
    (source_dir / "points.ply").touch()
    transforms = {
        "fl_x": 500.0,
        "fl_y": 510.0,
        "cx": 320.0,
        "cy": 240.0,
        "w": 640,
        "h": 480,
        "ply_file_path": "points.ply",
        "train_filenames": ["images/b.jpg"],
        "val_filenames": ["images/a.jpg"],
        "frames": [
            {
                "file_path": "images/b.jpg",
                "mask_path": "mask.png",
                "depth_file_path": "depth.png",
                "transform_matrix": IDENTITY,
            },
            {"file_path": "images/a.jpg", "transform_matrix": IDENTITY},
        ],
    }
    (source_dir / "transforms.json").write_text(json.dumps(transforms), encoding="utf-8")

    frame_count = create_portable_dataset(source_dir, destination_dir)

    portable = json.loads((destination_dir / "transforms.json").read_text(encoding="utf-8"))
    frame_paths = [frame["file_path"] for frame in portable["frames"]]
    assert frame_count == 2
    assert frame_paths == ["images/000001_b.jpg", "images/000000_a.jpg"]
    assert portable["train_filenames"] == ["images/000001_b.jpg"]
    assert portable["val_filenames"] == ["images/000000_a.jpg"]
    assert portable["test_filenames"] == ["images/000000_a.jpg"]
    assert "ply_file_path" not in portable
    assert "mask_path" not in portable["frames"][0]
    assert "depth_file_path" not in portable["frames"][0]
    assert portable["frames"][0]["transform_matrix"] == IDENTITY
    assert portable["fl_x"] == 500.0

    assert (destination_dir / frame_paths[0]).read_bytes() == b"source-b"
    assert (destination_dir / frame_paths[1]).read_bytes() == b"source-a"
