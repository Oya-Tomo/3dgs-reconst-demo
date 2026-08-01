import io
import subprocess
import sys
from pathlib import Path

import pytest
from rich.console import Console

from prepare_data import _conversion_stage, build_process_command, run_process_with_progress
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


def test_conversion_stage_recognizes_finalization_output() -> None:
    assert _conversion_stage("generating a simplified point cloud (this may take a while...)") == (
        "Building the point cloud"
    )
    assert _conversion_stage("unrelated converter message") is None


def test_run_process_with_progress_forwards_converter_output() -> None:
    output = io.StringIO()
    console = Console(file=output, force_terminal=True, color_system=None, width=120)

    run_process_with_progress(
        [sys.executable, "-c", "print('converter output', flush=True)"],
        console=console,
        refresh_seconds=0.01,
    )

    assert "converter output" in output.getvalue()
    assert "Conversion finished" in output.getvalue()


def test_run_process_with_progress_propagates_failure() -> None:
    output = io.StringIO()
    console = Console(file=output, force_terminal=False, color_system=None)

    with pytest.raises(subprocess.CalledProcessError) as error:
        run_process_with_progress(
            [sys.executable, "-c", "raise SystemExit(7)"],
            console=console,
            refresh_seconds=0.01,
        )

    assert error.value.returncode == 7
