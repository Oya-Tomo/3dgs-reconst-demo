"""Launch Nerfstudio's official Viewer from a portable checkpoint bundle."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import yaml
from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanagerConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.models.splatfacto import SplatfactoModelConfig
from nerfstudio.scripts.viewer.run_viewer import RunViewer, ViewerConfigWithoutNumRays

from dataset import validate_dataset
from settings import VIEWER, VIEWER_BUNDLE_VERSION, ViewerSettings


@dataclass(frozen=True, slots=True)
class ViewerBundle:
    """Validated paths needed to restore a copied Nerfstudio run."""

    directory: Path
    config_path: Path
    checkpoint_dir: Path
    checkpoint_path: Path
    dataset_dir: Path
    camera_count: int


def _manifest_string(manifest: dict[str, object], field: str) -> str:
    value = manifest.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"manifest.json {field} must be a non-empty string.")
    return value


def _resolve_relative_path(root: Path, value: str, field: str) -> Path:
    relative_path = Path(value)
    if relative_path.is_absolute():
        raise ValueError(f"manifest.json {field} must be relative to the Viewer bundle.")
    resolved_path = (root / relative_path).resolve()
    if not resolved_path.is_relative_to(root):
        raise ValueError(f"manifest.json {field} escapes the Viewer bundle: {value}")
    return resolved_path


def load_viewer_bundle(settings: ViewerSettings = VIEWER) -> ViewerBundle:
    """Validate a copied bundle without loading its checkpoint into GPU memory."""

    directory = settings.input_dir.expanduser().resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"The Viewer input directory does not exist: {directory}")

    manifest_path = directory / settings.manifest_filename
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"The Viewer bundle manifest does not exist: {manifest_path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"The Viewer bundle manifest is invalid JSON: {error}") from error
    if not isinstance(manifest, dict):
        raise ValueError("The Viewer bundle manifest root must be an object.")
    if manifest.get("format_version") != VIEWER_BUNDLE_VERSION:
        raise ValueError(
            f"Unsupported Viewer bundle format: {manifest.get('format_version')}; expected {VIEWER_BUNDLE_VERSION}."
        )

    config_path = _resolve_relative_path(directory, _manifest_string(manifest, "config_file"), "config_file")
    checkpoint_dir = _resolve_relative_path(
        directory,
        _manifest_string(manifest, "checkpoint_dir"),
        "checkpoint_dir",
    )
    checkpoint_path = _resolve_relative_path(
        checkpoint_dir,
        _manifest_string(manifest, "checkpoint_file"),
        "checkpoint_file",
    )
    dataset_dir = _resolve_relative_path(directory, _manifest_string(manifest, "dataset_dir"), "dataset_dir")

    if not config_path.is_file():
        raise FileNotFoundError(f"The bundled Nerfstudio config does not exist: {config_path}")
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"The bundled checkpoint directory does not exist: {checkpoint_dir}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"The bundled checkpoint does not exist: {checkpoint_path}")
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"The bundled camera dataset does not exist: {dataset_dir}")

    camera_count = manifest.get("camera_count")
    if isinstance(camera_count, bool) or not isinstance(camera_count, int) or camera_count <= 0:
        raise ValueError("manifest.json camera_count must be a positive integer.")
    dataset_report = validate_dataset(dataset_dir)
    if dataset_report.frame_count != camera_count:
        raise ValueError(
            "The Viewer bundle camera count does not match its dataset: "
            f"manifest={camera_count}, dataset={dataset_report.frame_count}."
        )

    return ViewerBundle(
        directory=directory,
        config_path=config_path,
        checkpoint_dir=checkpoint_dir,
        checkpoint_path=checkpoint_path,
        dataset_dir=dataset_dir,
        camera_count=camera_count,
    )


def build_runtime_config(bundle: ViewerBundle) -> TrainerConfig:
    """Relocate a trusted training config to paths inside the copied bundle."""

    config = yaml.load(bundle.config_path.read_text(encoding="utf-8"), Loader=yaml.Loader)
    if not isinstance(config, TrainerConfig):
        raise ValueError(f"The bundled config is not a Nerfstudio TrainerConfig: {bundle.config_path}")
    if not isinstance(config.pipeline.datamanager, FullImageDatamanagerConfig):
        raise ValueError("The bundled config does not use Nerfstudio's full-image data manager.")
    if not isinstance(config.pipeline.datamanager.dataparser, NerfstudioDataParserConfig):
        raise ValueError("The bundled config does not use Nerfstudio's data parser.")
    if not isinstance(config.pipeline.model, SplatfactoModelConfig):
        raise ValueError("The bundled config is not a Splatfacto training config.")

    config.output_dir = bundle.directory / ".viewer"
    config.experiment_name = "local"
    config.timestamp = "runtime"
    config.relative_model_dir = bundle.checkpoint_dir
    config.data = None
    config.load_dir = None
    config.load_step = None
    config.load_config = None
    config.load_checkpoint = None

    datamanager = cast(FullImageDatamanagerConfig, config.pipeline.datamanager)
    dataparser = cast(NerfstudioDataParserConfig, datamanager.dataparser)
    model = cast(SplatfactoModelConfig, config.pipeline.model)
    datamanager.data = bundle.dataset_dir
    datamanager.camera_res_scale_factor = 1.0
    datamanager.cache_images = "cpu"
    datamanager.cache_images_type = "uint8"
    datamanager.cache_compressed_images = False
    dataparser.data = bundle.dataset_dir
    dataparser.downscale_factor = 1
    dataparser.load_3D_points = False

    # The checkpoint immediately resizes and restores every Gaussian tensor. A tiny
    # temporary initialization avoids rebuilding the original seed point cloud.
    model.random_init = True
    model.num_random = 4
    model.random_scale = 1.0
    return config


def write_runtime_config(bundle: ViewerBundle, settings: ViewerSettings = VIEWER) -> Path:
    """Write the relocated config consumed by Nerfstudio's Viewer entry point."""

    runtime_filename = Path(settings.runtime_config_filename)
    if runtime_filename.is_absolute() or runtime_filename.name != settings.runtime_config_filename:
        raise ValueError("VIEWER.runtime_config_filename must be a filename, not a path.")
    runtime_config_path = bundle.directory / runtime_filename
    runtime_config_path.write_text(yaml.dump(build_runtime_config(bundle)), encoding="utf-8")
    return runtime_config_path


def build_viewer_runner(
    runtime_config_path: Path,
    settings: ViewerSettings = VIEWER,
) -> RunViewer:
    """Configure the same RunViewer implementation used by the ns-viewer command."""

    viewer_config = ViewerConfigWithoutNumRays(
        websocket_host=settings.host,
        websocket_port=settings.port,
        max_num_display_images=settings.max_display_cameras,
        image_format=settings.image_format,
        jpeg_quality=settings.jpeg_quality,
        make_share_url=False,
        camera_frustum_scale=settings.camera_frustum_scale,
        default_composite_depth=settings.default_composite_depth,
        quit_on_train_completion=False,
    )
    return RunViewer(load_config=runtime_config_path, viewer=viewer_config, vis="viewer")


def main() -> None:
    bundle = load_viewer_bundle()
    runtime_config_path = write_runtime_config(bundle)
    print(f"Viewer bundle: {bundle.directory}")
    print(f"Checkpoint: {bundle.checkpoint_path.name}")
    print(f"Camera records: {bundle.camera_count:,}")
    print("Launching Nerfstudio's official Viewer with the checkpoint's full spherical harmonics.")
    try:
        build_viewer_runner(runtime_config_path).main()
    except KeyboardInterrupt:
        print("Viewer stopped.")


if __name__ == "__main__":
    main()
