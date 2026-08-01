from pathlib import Path

from prepare_data import build_process_command
from settings import SpectacularAISettings


def test_build_process_command_contains_code_managed_options(tmp_path: Path) -> None:
    settings = SpectacularAISettings(
        recording_path=tmp_path / "recording",
        key_frame_distance=0.15,
        preview_3d=True,
        fast=True,
    )
    dataset_path = tmp_path / "dataset"

    assert build_process_command(settings, dataset_path) == [
        "sai-cli",
        "process",
        str(settings.recording_path),
        "--key_frame_distance=0.15",
        "--preview3d",
        "--fast",
        str(dataset_path),
    ]
