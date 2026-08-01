from pathlib import Path

from prepare_data import build_process_command
from settings import MappingSettings


def test_build_process_command_contains_code_managed_options(tmp_path: Path) -> None:
    settings = MappingSettings(
        recording_path=tmp_path / "recording",
        dataset_path=tmp_path / "dataset",
        key_frame_distance=0.15,
        preview_3d=True,
        fast=True,
    )

    assert build_process_command(settings) == [
        "sai-cli",
        "process",
        str(settings.recording_path),
        "--key_frame_distance=0.15",
        "--preview3d",
        "--fast",
        str(settings.dataset_path),
    ]
