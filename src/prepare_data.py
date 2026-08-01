"""Convert a Spectacular AI recording into Nerfstudio format."""

from __future__ import annotations

import os
import queue
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import TextIO

from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from dataset import validate_dataset
from settings import DATASET, SPECTACULAR_AI, SpectacularAISettings

_STATUS_REFRESH_SECONDS = 0.5
_CONVERSION_STAGES = (
    ("skipping blurry frame", "Selecting sharp keyframes"),
    ("blur filter range", "Selecting sharp keyframes"),
    ("generating a simplified point cloud", "Building the point cloud"),
    ("filtering out points", "Filtering the point cloud"),
)


def build_process_command(settings: SpectacularAISettings, dataset_path: Path) -> list[str]:
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
    command.append(str(dataset_path))
    return command


def _ensure_empty_destination(path: Path) -> None:
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(
            f"The conversion destination is not empty: {path}\n"
            "The operation was aborted to protect existing data. Configure a different DATASET.dataset_path in "
            "settings.py."
        )


def _read_cpu_seconds(pid: int) -> float | None:
    """Read aggregate CPU time for a Linux process without adding a monitoring dependency."""

    try:
        fields = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").rpartition(") ")[2].split()
        clock_ticks = os.sysconf("SC_CLK_TCK")
        return (int(fields[11]) + int(fields[12])) / clock_ticks
    except (FileNotFoundError, IndexError, OSError, ValueError):
        return None


def _conversion_stage(line: str) -> str | None:
    normalized = line.casefold()
    for marker, stage in _CONVERSION_STAGES:
        if marker in normalized:
            return stage
    return None


def _collect_output(stream: TextIO, lines: queue.SimpleQueue[str]) -> None:
    with stream:
        for line in stream:
            lines.put(line)


def _print_pending_output(
    lines: queue.SimpleQueue[str],
    console: Console,
) -> str | None:
    latest_stage = None
    while True:
        try:
            line = lines.get_nowait()
        except queue.Empty:
            return latest_stage
        console.print(line.rstrip("\n"), markup=False, highlight=False)
        latest_stage = _conversion_stage(line) or latest_stage


def run_process_with_progress(
    command: list[str],
    *,
    console: Console | None = None,
    refresh_seconds: float = _STATUS_REFRESH_SECONDS,
) -> None:
    """Run the converter while showing truthful, indeterminate progress."""

    if refresh_seconds <= 0:
        raise ValueError("refresh_seconds must be greater than zero")

    console = console or Console()
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert process.stdout is not None

    output_lines: queue.SimpleQueue[str] = queue.SimpleQueue()
    output_thread = threading.Thread(
        target=_collect_output,
        args=(process.stdout, output_lines),
        name="sai-cli-output",
        daemon=True,
    )
    output_thread.start()

    previous_wall_time = time.monotonic()
    previous_cpu_time = _read_cpu_seconds(process.pid)
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=24),
        TimeElapsedColumn(),
        TextColumn("CPU {task.fields[cpu]}", justify="right"),
        console=console,
        transient=False,
        disable=not console.is_terminal,
    )

    return_code: int | None = None
    try:
        with progress:
            task_id = progress.add_task("Mapping and optimizing the recording", total=None, cpu="measuring")
            while (return_code := process.poll()) is None:
                if stage := _print_pending_output(output_lines, console):
                    progress.update(task_id, description=stage)

                time.sleep(refresh_seconds)
                current_wall_time = time.monotonic()
                current_cpu_time = _read_cpu_seconds(process.pid)
                if previous_cpu_time is not None and current_cpu_time is not None:
                    elapsed = current_wall_time - previous_wall_time
                    cpu_cores = max(0.0, (current_cpu_time - previous_cpu_time) / elapsed)
                    progress.update(task_id, cpu=f"{cpu_cores:.1f} cores")
                previous_wall_time = current_wall_time
                previous_cpu_time = current_cpu_time

            output_thread.join()
            if stage := _print_pending_output(output_lines, console):
                progress.update(task_id, description=stage)
            if return_code == 0:
                progress.update(task_id, description="Conversion finished", total=1, completed=1, cpu="done")
            else:
                progress.update(task_id, description="Conversion failed", total=1, completed=1, cpu="stopped")
    except KeyboardInterrupt:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        raise

    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def main() -> None:
    recording_path = SPECTACULAR_AI.recording_path.expanduser().resolve()
    dataset_path = DATASET.dataset_path.expanduser().resolve()

    if not recording_path.exists():
        raise FileNotFoundError(f"The Spectacular AI recording does not exist: {recording_path}")
    if shutil.which("sai-cli") is None:
        raise RuntimeError("sai-cli was not found. Run this script in the environment created by `uv sync`.")
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg was not found. Install it with the operating system package manager.")

    _ensure_empty_destination(dataset_path)
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    command = build_process_command(
        SpectacularAISettings(
            recording_path=recording_path,
            key_frame_distance=SPECTACULAR_AI.key_frame_distance,
            preview_3d=SPECTACULAR_AI.preview_3d,
            fast=SPECTACULAR_AI.fast,
        ),
        dataset_path,
    )
    print("Converting the Spectacular AI recording:", " ".join(command))
    run_process_with_progress(command)

    report = validate_dataset(dataset_path)
    print(f"Conversion complete: {report.frame_count} frames -> {report.dataset_path}")
    for warning in report.warnings:
        print(f"Warning: {warning}")


if __name__ == "__main__":
    main()
