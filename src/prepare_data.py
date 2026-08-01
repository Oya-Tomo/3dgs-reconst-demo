"""Convert a Spectacular AI recording into Nerfstudio format."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from dataset import validate_dataset
from settings import MAPPING, MappingSettings


def build_process_command(settings: MappingSettings) -> list[str]:
    """Build the supported Spectacular AI Mapping Tools invocation."""

    command = [
        "sai-cli",
        "process",
        str(settings.recording_path),
        f"--key_frame_distance={settings.key_frame_distance}",
    ]
    if settings.preview_3d:
        command.append("--preview3d")
    if settings.fast:
        command.append("--fast")
    command.append(str(settings.dataset_path))
    return command


def _ensure_empty_destination(path: Path) -> None:
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(
            f"The conversion destination is not empty: {path}\n"
            "The operation was aborted to protect existing data. Configure a different dataset_path in settings.py."
        )


def main() -> None:
    recording_path = MAPPING.recording_path.expanduser().resolve()
    dataset_path = MAPPING.dataset_path.expanduser().resolve()

    if not recording_path.exists():
        raise FileNotFoundError(f"The Spectacular AI recording does not exist: {recording_path}")
    if shutil.which("sai-cli") is None:
        raise RuntimeError("sai-cli was not found. Run this script in the environment created by `uv sync`.")
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg was not found. Install it with the operating system package manager.")

    _ensure_empty_destination(dataset_path)
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    command = build_process_command(
        MappingSettings(
            recording_path=recording_path,
            dataset_path=dataset_path,
            key_frame_distance=MAPPING.key_frame_distance,
            preview_3d=MAPPING.preview_3d,
            fast=MAPPING.fast,
        )
    )
    print("Converting the Spectacular AI recording:", " ".join(command))
    subprocess.run(command, check=True)

    report = validate_dataset(dataset_path)
    print(f"Conversion complete: {report.frame_count} frames -> {report.dataset_path}")
    for warning in report.warnings:
        print(f"Warning: {warning}")


if __name__ == "__main__":
    main()
