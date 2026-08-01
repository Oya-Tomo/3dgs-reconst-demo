import json
import math
from pathlib import Path

import numpy as np

from settings import VIEWER_BUNDLE_VERSION, ViewerSettings
from viewer import load_gaussian_splats, load_viewer_bundle


def write_gaussian_ply(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "ply",
                "format ascii 1.0",
                "element vertex 1",
                "property float x",
                "property float y",
                "property float z",
                "property float scale_0",
                "property float scale_1",
                "property float scale_2",
                "property float rot_0",
                "property float rot_1",
                "property float rot_2",
                "property float rot_3",
                "property float opacity",
                "property float f_dc_0",
                "property float f_dc_1",
                "property float f_dc_2",
                "end_header",
                f"1 2 3 0 {math.log(2)} {math.log(3)} 1 0 0 0 0 0 0 0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_load_viewer_bundle_and_gaussians(tmp_path: Path) -> None:
    splat_path = tmp_path / "splat.ply"
    write_gaussian_ply(splat_path)
    transform = [
        [1.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 2.0],
        [0.0, 0.0, 1.0, 3.0],
    ]
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "format_version": VIEWER_BUNDLE_VERSION,
                "splat_file": "splat.ply",
                "cameras": [{"split": "train", "transform": transform}],
            }
        ),
        encoding="utf-8",
    )

    bundle = load_viewer_bundle(ViewerSettings(input_dir=tmp_path))
    splats = load_gaussian_splats(bundle.splat_path)

    assert bundle.splat_path == splat_path
    assert len(bundle.cameras) == 1
    np.testing.assert_allclose(bundle.cameras[0].position, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(bundle.cameras[0].view_direction, [0.0, 0.0, -1.0])
    assert splats.count == 1
    np.testing.assert_allclose(splats.centers, [[1.0, 2.0, 3.0]])
    np.testing.assert_allclose(splats.covariances[0], np.diag([1.0, 4.0, 9.0]), rtol=1e-5)
    np.testing.assert_allclose(splats.colors, [[0.5, 0.5, 0.5]])
    np.testing.assert_allclose(splats.opacities, [[0.5]])
