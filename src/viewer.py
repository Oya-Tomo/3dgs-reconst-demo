"""Load and render a copied viewer bundle in a local browser."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray
from plyfile import PlyData

from settings import VIEWER, VIEWER_BUNDLE_VERSION, ViewerSettings

FloatArray = NDArray[np.float32]
CameraSplit = Literal["train", "eval"]
SH_C0 = np.float32(0.28209479177387814)
TRAIN_CAMERA_COLOR = (80, 170, 255)
EVAL_CAMERA_COLOR = (255, 180, 80)

if TYPE_CHECKING:
    import viser


@dataclass(frozen=True, slots=True)
class CameraPose:
    """Camera-to-world pose in Nerfstudio's normalized coordinates."""

    rotation: FloatArray
    position: FloatArray
    split: CameraSplit

    @property
    def view_direction(self) -> FloatArray:
        return -self.rotation[:, 2]


@dataclass(frozen=True, slots=True)
class ViewerBundle:
    """Validated contents of a downloaded viewer-output directory."""

    directory: Path
    splat_path: Path
    cameras: tuple[CameraPose, ...]


@dataclass(frozen=True, slots=True)
class GaussianSplats:
    """Gaussian attributes in the form expected by Viser."""

    centers: FloatArray
    covariances: FloatArray
    colors: FloatArray
    opacities: FloatArray

    @property
    def count(self) -> int:
        return int(self.centers.shape[0])


def _parse_camera(frame: object, index: int) -> CameraPose:
    if not isinstance(frame, dict):
        raise ValueError(f"manifest.json cameras[{index}] must be an object.")
    split = frame.get("split")
    if split not in ("train", "eval"):
        raise ValueError(f"manifest.json cameras[{index}].split is invalid: {split}")
    transform = np.asarray(frame.get("transform"), dtype=np.float32)
    if transform.shape == (4, 4):
        transform = transform[:3]
    if transform.shape != (3, 4) or not np.isfinite(transform).all():
        raise ValueError(f"manifest.json cameras[{index}].transform must be a finite 3x4 matrix.")
    return CameraPose(
        rotation=transform[:, :3].copy(),
        position=transform[:, 3].copy(),
        split=split,
    )


def load_viewer_bundle(settings: ViewerSettings = VIEWER) -> ViewerBundle:
    """Validate the directory contract shared by export.py and viewer.py."""

    directory = settings.input_dir.expanduser().resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"The viewer input directory does not exist: {directory}")
    manifest_path = directory / settings.manifest_filename
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"The viewer manifest does not exist: {manifest_path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"The viewer manifest is invalid JSON: {error}") from error
    if not isinstance(manifest, dict):
        raise ValueError("The viewer manifest root must be an object.")
    if manifest.get("format_version") != VIEWER_BUNDLE_VERSION:
        raise ValueError(
            f"Viewer bundle version mismatch: {manifest.get('format_version')} (expected {VIEWER_BUNDLE_VERSION})"
        )

    splat_file = manifest.get("splat_file")
    if not isinstance(splat_file, str) or not splat_file:
        raise ValueError("The viewer manifest does not define splat_file.")
    splat_path = (directory / splat_file).resolve()
    if not splat_path.is_relative_to(directory) or not splat_path.is_file():
        raise FileNotFoundError(f"The viewer bundle Gaussian Splat does not exist: {splat_path}")

    camera_values = manifest.get("cameras", [])
    if not isinstance(camera_values, list):
        raise ValueError("The viewer manifest cameras field must be an array.")
    cameras = tuple(_parse_camera(frame, index) for index, frame in enumerate(camera_values))
    return ViewerBundle(directory=directory, splat_path=splat_path, cameras=cameras)


def _stack_ply_properties(data: np.ndarray, names: tuple[str, ...]) -> FloatArray:
    available = set(data.dtype.names or ())
    missing = [name for name in names if name not in available]
    if missing:
        raise ValueError(f"The Gaussian Splat PLY is missing required properties: {', '.join(missing)}")
    return np.column_stack([np.asarray(data[name], dtype=np.float32) for name in names]).astype(
        np.float32,
        copy=False,
    )


def _quaternions_to_rotation_matrices(quaternions: FloatArray) -> FloatArray:
    norms = np.linalg.norm(quaternions, axis=1, keepdims=True)
    if np.any(norms <= np.finfo(np.float32).eps):
        raise ValueError("The Gaussian Splat PLY contains a zero-length quaternion.")
    normalized = quaternions / norms
    w, x, y, z = normalized.T

    matrices = np.empty((normalized.shape[0], 3, 3), dtype=np.float32)
    matrices[:, 0, 0] = 1 - 2 * (y * y + z * z)
    matrices[:, 0, 1] = 2 * (x * y - w * z)
    matrices[:, 0, 2] = 2 * (x * z + w * y)
    matrices[:, 1, 0] = 2 * (x * y + w * z)
    matrices[:, 1, 1] = 1 - 2 * (x * x + z * z)
    matrices[:, 1, 2] = 2 * (y * z - w * x)
    matrices[:, 2, 0] = 2 * (x * z - w * y)
    matrices[:, 2, 1] = 2 * (y * z + w * x)
    matrices[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return matrices


def _load_ply_colors(data: np.ndarray) -> FloatArray:
    available = set(data.dtype.names or ())
    if {"red", "green", "blue"}.issubset(available):
        colors = _stack_ply_properties(data, ("red", "green", "blue"))
        if colors.size and float(colors.max()) > 1.0:
            colors /= 255.0
        return np.clip(colors, 0.0, 1.0)
    if {"f_dc_0", "f_dc_1", "f_dc_2"}.issubset(available):
        sh_dc = _stack_ply_properties(data, ("f_dc_0", "f_dc_1", "f_dc_2"))
        return np.clip(sh_dc * SH_C0 + 0.5, 0.0, 1.0)
    raise ValueError("The Gaussian Splat PLY has neither RGB nor f_dc color properties.")


def load_gaussian_splats(path: Path) -> GaussianSplats:
    """Read a Nerfstudio Gaussian Splat PLY and derive covariance matrices."""

    ply = PlyData.read(str(path))
    try:
        data = ply["vertex"].data
    except KeyError as error:
        raise ValueError("The Gaussian Splat PLY does not contain a vertex element.") from error
    if len(data) == 0:
        raise ValueError("The Gaussian Splat PLY does not contain any Gaussians.")

    centers = _stack_ply_properties(data, ("x", "y", "z"))
    log_scales = _stack_ply_properties(data, ("scale_0", "scale_1", "scale_2"))
    quaternions = _stack_ply_properties(data, ("rot_0", "rot_1", "rot_2", "rot_3"))
    opacity_logits = _stack_ply_properties(data, ("opacity",))
    colors = _load_ply_colors(data)

    scales = np.exp(log_scales).astype(np.float32, copy=False)
    rotations = _quaternions_to_rotation_matrices(quaternions)
    scaled_rotations = rotations * scales[:, None, :]
    covariances = scaled_rotations @ np.transpose(scaled_rotations, (0, 2, 1))
    opacities = 1.0 / (1.0 + np.exp(-np.clip(opacity_logits, -80.0, 80.0)))
    values = (centers, covariances, colors, opacities)
    if not all(np.isfinite(value).all() for value in values):
        raise ValueError("The Gaussian Splat PLY contains non-finite values.")

    return GaussianSplats(
        centers=centers,
        covariances=covariances.astype(np.float32, copy=False),
        colors=colors.astype(np.float32, copy=False),
        opacities=opacities.astype(np.float32, copy=False),
    )


def _sample_camera_poses(poses: tuple[CameraPose, ...], maximum: int) -> tuple[CameraPose, ...]:
    if maximum <= 0 or len(poses) <= maximum:
        return poses
    indices = np.linspace(0, len(poses) - 1, maximum, dtype=np.int64)
    return tuple(poses[int(index)] for index in indices)


def _add_camera_overlays(
    server: viser.ViserServer,
    poses: tuple[CameraPose, ...],
    settings: ViewerSettings,
) -> None:
    import viser.transforms as vtf

    poses = _sample_camera_poses(poses, settings.max_cameras)
    if not poses:
        print("No camera poses are present; displaying Gaussians only.")
        return

    cameras_root = server.scene.add_frame("/capture_cameras", show_axes=False, visible=settings.show_cameras)
    directions_root = server.scene.add_frame(
        "/view_directions",
        show_axes=False,
        visible=settings.show_view_directions,
    )
    line_points: list[np.ndarray] = []
    line_colors: list[tuple[int, int, int]] = []
    for index, pose in enumerate(poses):
        color = TRAIN_CAMERA_COLOR if pose.split == "train" else EVAL_CAMERA_COLOR
        rotation = vtf.SO3.from_matrix(pose.rotation) @ vtf.SO3.from_x_radians(np.pi)
        server.scene.add_camera_frustum(
            f"/capture_cameras/{pose.split}_{index:04d}",
            fov=np.deg2rad(settings.camera_fov_degrees),
            aspect=settings.camera_aspect,
            scale=settings.camera_frustum_scale,
            color=color,
            wxyz=rotation.wxyz,
            position=pose.position,
        )
        line_points.append(
            np.stack([pose.position, pose.position + pose.view_direction * settings.view_direction_length])
        )
        line_colors.append(color)

    colors = np.repeat(np.asarray(line_colors, dtype=np.uint8)[:, None, :], 2, axis=1)
    server.scene.add_line_segments(
        "/view_directions/rays",
        points=np.asarray(line_points, dtype=np.float32),
        colors=colors,
        line_width=1.5,
    )
    cameras_toggle = server.gui.add_checkbox("Capture cameras", initial_value=settings.show_cameras)
    directions_toggle = server.gui.add_checkbox("View directions", initial_value=settings.show_view_directions)

    @cameras_toggle.on_update
    def _(_) -> None:
        cameras_root.visible = cameras_toggle.value

    @directions_toggle.on_update
    def _(_) -> None:
        directions_root.visible = directions_toggle.value

    print(f"Camera poses: {len(poses):,} (toggle cameras and view directions in the Viewer)")


def main() -> None:
    bundle = load_viewer_bundle()
    print(f"Loading viewer bundle: {bundle.directory}")
    splats = load_gaussian_splats(bundle.splat_path)
    import viser

    server = viser.ViserServer(host=VIEWER.host, port=VIEWER.port, label="3DGS reconstruction")
    server.gui.configure_theme(dark_mode=True, show_share_button=False)
    server.scene.set_up_direction(VIEWER.up_direction)
    server.scene.add_gaussian_splats(
        "/scene/gaussians",
        centers=splats.centers,
        covariances=splats.covariances,
        rgbs=splats.colors,
        opacities=splats.opacities,
    )
    _add_camera_overlays(server, bundle.cameras, VIEWER)
    print(f"Loaded {splats.count:,} Gaussians")
    print(f"Open http://{VIEWER.host}:{VIEWER.port} in a browser.")

    try:
        while True:
            time.sleep(0.25)
    except KeyboardInterrupt:
        server.stop()
        print("Viewer stopped.")


if __name__ == "__main__":
    main()
