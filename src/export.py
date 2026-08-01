"""Export the latest Splatfacto run as a portable viewer bundle."""

from __future__ import annotations

import json
from pathlib import Path

from settings import EXPORT, VIEWER_BUNDLE_VERSION, ExportSettings

CAMERA_EXPORTS = (
    ("train", "transforms_train.json"),
    ("eval", "transforms_eval.json"),
)


def find_training_config(settings: ExportSettings = EXPORT) -> Path:
    """Resolve an explicit config or the newest config from a training run."""

    if settings.checkpoint_config is not None:
        config_path = settings.checkpoint_config.expanduser().resolve()
        if not config_path.is_file():
            raise FileNotFoundError(f"The training config does not exist: {config_path}")
        return config_path

    run_root = settings.training_output_dir / settings.experiment_name / settings.method_name
    candidates = list(run_root.glob("*/config.yml"))
    if not candidates:
        raise FileNotFoundError(f"No trained config.yml exists under {run_root}. Run src/train.py first.")
    return max(candidates, key=lambda path: (path.parent.name, path.stat().st_mtime_ns, str(path)))


def _ensure_empty_output_directory(output_dir: Path) -> None:
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError(
            f"The viewer output directory is not empty: {output_dir}\n"
            "Configure a different EXPORT.output_dir in settings.py to protect existing artifacts."
        )


def _collect_exported_cameras(output_dir: Path) -> list[dict[str, object]]:
    """Combine Nerfstudio's temporary pose files without retaining image paths."""

    cameras: list[dict[str, object]] = []
    for split, filename in CAMERA_EXPORTS:
        path = output_dir / filename
        if not path.is_file():
            continue
        frames = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(frames, list):
            raise ValueError(f"Nerfstudio produced invalid camera poses: {path}")
        for index, frame in enumerate(frames):
            if not isinstance(frame, dict) or "transform" not in frame:
                raise ValueError(f"A camera pose is missing: {path} frames[{index}]")
            cameras.append({"split": split, "transform": frame["transform"]})
        path.unlink()
    return cameras


def _write_manifest(output_dir: Path, config_path: Path, cameras: list[dict[str, object]]) -> Path:
    manifest_path = output_dir / EXPORT.manifest_filename
    manifest = {
        "format_version": VIEWER_BUNDLE_VERSION,
        "source_run": config_path.parent.name,
        "splat_file": EXPORT.splat_filename,
        "cameras": cameras,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def main() -> None:
    config_path = find_training_config()
    output_dir = EXPORT.output_dir.expanduser().resolve()
    _ensure_empty_output_directory(output_dir)
    from nerfstudio.scripts.exporter import ExportCameraPoses, ExportGaussianSplat

    print(f"Exporting checkpoint: {config_path}")

    ExportGaussianSplat(
        load_config=config_path,
        output_dir=output_dir,
        output_filename=EXPORT.splat_filename,
        ply_color_mode=EXPORT.ply_color_mode,
    ).main()

    cameras: list[dict[str, object]] = []
    if EXPORT.include_camera_poses:
        ExportCameraPoses(load_config=config_path, output_dir=output_dir).main()
        cameras = _collect_exported_cameras(output_dir)

    splat_path = output_dir / EXPORT.splat_filename
    if not splat_path.is_file():
        raise RuntimeError(f"The Gaussian Splat export was not created: {splat_path}")
    manifest_path = _write_manifest(output_dir, config_path, cameras)

    print(f"Viewer bundle directory: {output_dir}")
    print(f"  - {manifest_path.name}")
    print(f"  - {splat_path.name} ({splat_path.stat().st_size / 1_000_000:.1f} MB)")


if __name__ == "__main__":
    main()
