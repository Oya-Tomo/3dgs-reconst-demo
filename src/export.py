"""Create a portable checkpoint bundle for Nerfstudio's official Viewer."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

from dataset import validate_dataset
from settings import EXPORT, VIEWER_BUNDLE_VERSION, ExportSettings

CHECKPOINT_STEP_PATTERN = re.compile(r"step-(\d{9})\.ckpt$")


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


def find_latest_checkpoint(config_path: Path) -> Path:
    """Find the highest-step checkpoint belonging to a training config."""

    checkpoint_dir = config_path.parent / "nerfstudio_models"
    candidates = [path for path in checkpoint_dir.glob("*.ckpt") if CHECKPOINT_STEP_PATTERN.fullmatch(path.name)]
    if not candidates:
        raise FileNotFoundError(f"No Nerfstudio step-XXXXXXXXX.ckpt checkpoint exists under {checkpoint_dir}.")

    def sort_key(path: Path) -> tuple[int, int, str]:
        match = CHECKPOINT_STEP_PATTERN.fullmatch(path.name)
        assert match is not None
        step = int(match.group(1))
        return (step, path.stat().st_mtime_ns, path.name)

    return max(candidates, key=sort_key)


def _ensure_empty_output_directory(output_dir: Path) -> None:
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError(
            f"The Viewer output directory is not empty: {output_dir}\n"
            "Configure a different EXPORT.output_dir in settings.py to protect existing artifacts."
        )


def _normalize_frame_path(value: str) -> str:
    return Path(value).as_posix().removeprefix("./")


def _portable_image_name(original_path: str, rank: int) -> str:
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", Path(original_path).stem).strip("_") or "frame"
    suffix = Path(original_path).suffix.lower() or ".image"
    return f"images/{rank:06d}_{stem[:48]}{suffix}"


def create_portable_dataset(source_dir: Path, destination_dir: Path) -> int:
    """Copy camera metadata and referenced photographs into a relocatable dataset."""

    source_dir = source_dir.expanduser().resolve()
    transforms_path = source_dir / "transforms.json"
    try:
        transforms = json.loads(transforms_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"The source dataset transforms.json does not exist: {transforms_path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"The source dataset transforms.json is invalid JSON: {error}") from error
    if not isinstance(transforms, dict):
        raise ValueError("The source transforms.json root must be an object.")
    frames = transforms.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("The source transforms.json must contain at least one frame.")

    original_paths: list[str] = []
    for index, frame in enumerate(frames):
        if not isinstance(frame, dict):
            raise ValueError(f"The source transforms.json frames[{index}] must be an object.")
        file_path = frame.get("file_path")
        if not isinstance(file_path, str) or not file_path:
            raise ValueError(f"The source transforms.json frames[{index}].file_path is missing.")
        original_paths.append(file_path)

    normalized_paths = [_normalize_frame_path(value) for value in original_paths]
    if len(set(normalized_paths)) != len(normalized_paths):
        raise ValueError("The source transforms.json contains duplicate frame file paths.")

    sorted_indices = sorted(range(len(frames)), key=lambda index: (_normalize_frame_path(original_paths[index]), index))
    rank_by_index = {frame_index: rank for rank, frame_index in enumerate(sorted_indices)}
    path_mapping: dict[str, str] = {}
    portable_frames: list[dict[str, object]] = []
    for index, frame in enumerate(frames):
        original_path = original_paths[index]
        portable_path = _portable_image_name(original_path, rank_by_index[index])
        path_mapping[normalized_paths[index]] = portable_path
        portable_frame = dict(frame)
        portable_frame["file_path"] = portable_path
        portable_frame.pop("mask_path", None)
        portable_frame.pop("depth_file_path", None)
        portable_frames.append(portable_frame)

    portable_transforms = dict(transforms)
    portable_transforms["frames"] = portable_frames
    portable_transforms.pop("ply_file_path", None)
    for split_key in ("train_filenames", "val_filenames", "test_filenames"):
        if split_key not in portable_transforms:
            continue
        split_paths = portable_transforms[split_key]
        if (
            not isinstance(split_paths, list)
            or not split_paths
            or not all(isinstance(value, str) for value in split_paths)
        ):
            raise ValueError(f"The source transforms.json {split_key} field must be a non-empty array of paths.")
        try:
            portable_transforms[split_key] = [path_mapping[_normalize_frame_path(value)] for value in split_paths]
        except KeyError as error:
            raise ValueError(
                f"The source transforms.json {split_key} field references an unknown frame: {error}"
            ) from error

    explicit_split_keys = {
        split_key
        for split_key in ("train_filenames", "val_filenames", "test_filenames")
        if split_key in portable_transforms
    }
    if explicit_split_keys:
        if "train_filenames" not in explicit_split_keys:
            raise ValueError("Explicit dataset splits must include train_filenames.")
        if "test_filenames" not in explicit_split_keys and "val_filenames" not in explicit_split_keys:
            raise ValueError("Explicit dataset splits must include val_filenames or test_filenames.")
        if "val_filenames" not in explicit_split_keys:
            portable_transforms["val_filenames"] = list(portable_transforms["test_filenames"])
        if "test_filenames" not in explicit_split_keys:
            portable_transforms["test_filenames"] = list(portable_transforms["val_filenames"])

    images_dir = destination_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    for index, frame in enumerate(portable_frames):
        source_image_path = (source_dir / original_paths[index]).resolve()
        if not source_image_path.is_file():
            raise FileNotFoundError(f"A source dataset image does not exist: {source_image_path}")
        image_path = destination_dir / str(frame["file_path"])
        shutil.copy2(source_image_path, image_path)
    (destination_dir / "transforms.json").write_text(
        json.dumps(portable_transforms, indent=2) + "\n",
        encoding="utf-8",
    )
    validate_dataset(destination_dir)
    return len(portable_frames)


def _write_manifest(
    output_dir: Path,
    config_path: Path,
    checkpoint_path: Path,
    frame_count: int,
    settings: ExportSettings,
) -> Path:
    manifest_path = output_dir / settings.manifest_filename
    manifest = {
        "format_version": VIEWER_BUNDLE_VERSION,
        "source_run": config_path.parent.name,
        "config_file": settings.config_filename,
        "checkpoint_dir": settings.checkpoint_dirname,
        "checkpoint_file": checkpoint_path.name,
        "dataset_dir": settings.dataset_dirname,
        "camera_count": frame_count,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def main() -> None:
    config_path = find_training_config()
    checkpoint_path = find_latest_checkpoint(config_path)
    source_dataset = EXPORT.dataset_path.expanduser().resolve()
    validate_dataset(source_dataset)

    output_dir = EXPORT.output_dir.expanduser().resolve()
    _ensure_empty_output_directory(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    copied_config = output_dir / EXPORT.config_filename
    shutil.copy2(config_path, copied_config)
    checkpoint_dir = output_dir / EXPORT.checkpoint_dirname
    checkpoint_dir.mkdir()
    copied_checkpoint = checkpoint_dir / checkpoint_path.name
    shutil.copy2(checkpoint_path, copied_checkpoint)
    frame_count = create_portable_dataset(source_dataset, output_dir / EXPORT.dataset_dirname)
    manifest_path = _write_manifest(output_dir, config_path, checkpoint_path, frame_count, EXPORT)

    print(f"Official Nerfstudio Viewer bundle: {output_dir}")
    print(f"  - {manifest_path.name}")
    print(f"  - {copied_config.name}")
    print(f"  - {copied_checkpoint.relative_to(output_dir)} ({copied_checkpoint.stat().st_size / 1_000_000:.1f} MB)")
    print(f"  - {EXPORT.dataset_dirname}/ ({frame_count:,} camera images)")


if __name__ == "__main__":
    main()
