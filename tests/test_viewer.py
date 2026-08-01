import json
import os
from pathlib import Path

import pytest
import yaml
from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanagerConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.models.splatfacto import SplatfactoModelConfig
from PIL import Image

from settings import VIEWER_BUNDLE_VERSION, ViewerSettings
from train import build_trainer_config
from viewer import (
    build_runtime_config,
    build_viewer_runner,
    load_viewer_bundle,
    trusted_checkpoint_loading,
    write_runtime_config,
)


def _transform(x: float) -> list[list[float]]:
    return [
        [1.0, 0.0, 0.0, x],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def write_viewer_bundle(directory: Path) -> ViewerSettings:
    dataset_dir = directory / "dataset"
    images_dir = dataset_dir / "images"
    checkpoint_dir = directory / "nerfstudio_models"
    images_dir.mkdir(parents=True)
    checkpoint_dir.mkdir()
    for index in range(3):
        Image.new("RGB", (1, 1), color=(index, index, index)).save(images_dir / f"{index:06d}.png")

    transforms = {
        "camera_model": "OPENCV",
        "fl_x": 1.0,
        "fl_y": 1.0,
        "cx": 0.5,
        "cy": 0.5,
        "w": 1,
        "h": 1,
        "frames": [
            {"file_path": f"images/{index:06d}.png", "transform_matrix": _transform(float(index))} for index in range(3)
        ],
    }
    (dataset_dir / "transforms.json").write_text(json.dumps(transforms), encoding="utf-8")

    config = build_trainer_config()
    (directory / "config.yml").write_text(yaml.dump(config), encoding="utf-8")
    checkpoint_path = checkpoint_dir / "step-000030000.ckpt"
    checkpoint_path.touch()
    manifest = {
        "format_version": VIEWER_BUNDLE_VERSION,
        "source_run": "2026-08-01_000000",
        "config_file": "config.yml",
        "checkpoint_dir": "nerfstudio_models",
        "checkpoint_file": checkpoint_path.name,
        "dataset_dir": "dataset",
        "camera_count": 3,
    }
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return ViewerSettings(input_dir=directory, host="127.0.0.1", port=7008, max_display_cameras=2)


def test_load_viewer_bundle_validates_relocatable_paths_and_camera_dataset(tmp_path: Path) -> None:
    settings = write_viewer_bundle(tmp_path)

    bundle = load_viewer_bundle(settings)

    assert bundle.directory == tmp_path
    assert bundle.config_path == tmp_path / "config.yml"
    assert bundle.checkpoint_path == tmp_path / "nerfstudio_models" / "step-000030000.ckpt"
    assert bundle.dataset_dir == tmp_path / "dataset"
    assert bundle.camera_count == 3


def test_load_viewer_bundle_rejects_paths_outside_bundle(tmp_path: Path) -> None:
    settings = write_viewer_bundle(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["config_file"] = "../config.yml"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="escapes the Viewer bundle"):
        load_viewer_bundle(settings)


def test_runtime_config_restores_checkpoint_and_camera_metadata_from_bundle(tmp_path: Path) -> None:
    settings = write_viewer_bundle(tmp_path)
    bundle = load_viewer_bundle(settings)

    config = build_runtime_config(bundle)
    datamanager = config.pipeline.datamanager
    model = config.pipeline.model
    assert isinstance(datamanager, FullImageDatamanagerConfig)
    assert isinstance(datamanager.dataparser, NerfstudioDataParserConfig)
    assert isinstance(model, SplatfactoModelConfig)
    dataparser_config = datamanager.dataparser

    assert config.get_checkpoint_dir() == bundle.checkpoint_dir
    assert datamanager.data == bundle.dataset_dir
    assert dataparser_config.data == bundle.dataset_dir
    assert dataparser_config.downscale_factor == 1
    assert dataparser_config.load_3D_points is False
    assert datamanager.cache_images == "cpu"
    assert model.random_init is True
    assert model.num_random == 4

    dataparser = dataparser_config.setup()
    outputs = dataparser.get_dataparser_outputs(split="train")
    dataset = InputDataset(outputs)
    assert dataset[0]["image"].shape == (1, 1, 3)
    assert dataset.cameras.camera_to_worlds.shape[-2:] == (3, 4)


def test_runtime_config_and_runner_use_official_nerfstudio_viewer(tmp_path: Path) -> None:
    settings = write_viewer_bundle(tmp_path)
    bundle = load_viewer_bundle(settings)

    runtime_path = write_runtime_config(bundle, settings)
    runner = build_viewer_runner(runtime_path, settings)
    reloaded = yaml.load(runtime_path.read_text(encoding="utf-8"), Loader=yaml.Loader)

    assert runner.load_config == runtime_path
    assert runner.vis == "viewer"
    assert runner.viewer.websocket_host == "127.0.0.1"
    assert runner.viewer.websocket_port == 7008
    assert runner.viewer.max_num_display_images == 2
    assert reloaded.get_checkpoint_dir() == bundle.checkpoint_dir
    assert reloaded.get_base_dir().is_dir()


def test_trusted_checkpoint_loading_is_scoped(monkeypatch: pytest.MonkeyPatch) -> None:
    variable = "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"
    monkeypatch.delenv(variable, raising=False)

    with trusted_checkpoint_loading():
        assert os.environ[variable] == "1"

    assert variable not in os.environ
