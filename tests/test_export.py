from pathlib import Path

from export import find_training_config
from settings import ExportSettings


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
