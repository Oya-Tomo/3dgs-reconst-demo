"""Project settings.

Edit this file instead of passing training parameters on the command line.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VIEWER_BUNDLE_VERSION = 1


@dataclass(frozen=True, slots=True)
class MappingSettings:
    """Spectacular AI recording conversion settings."""

    recording_path: Path = PROJECT_ROOT / "data" / "recording"
    dataset_path: Path = PROJECT_ROOT / "data" / "nerfstudio"
    key_frame_distance: float = 0.05
    preview_3d: bool = False
    fast: bool = False


@dataclass(frozen=True, slots=True)
class DatasetSettings:
    """Nerfstudio data parser settings."""

    dataset_path: Path = PROJECT_ROOT / "data" / "nerfstudio"
    downscale_factor: int | None = None
    scale_factor: float = 1.0
    scene_scale: float = 1.0
    orientation_method: Literal["pca", "up", "vertical", "none"] = "up"
    center_method: Literal["poses", "focus", "none"] = "poses"
    auto_scale_poses: bool = True
    eval_mode: Literal["fraction", "filename", "interval", "all"] = "fraction"
    train_split_fraction: float = 0.9
    eval_interval: int = 8
    depth_unit_scale_factor: float = 1e-3
    load_3d_points: bool = True


@dataclass(frozen=True, slots=True)
class TrainingSettings:
    """Splatfacto trainer, model, optimizer, and viewer settings."""

    output_dir: Path = PROJECT_ROOT / "outputs"
    experiment_name: str = "spectacular-ai-3dgs"
    project_name: str = "3dgs-reconst-demo"
    seed: int = 42
    num_devices: int = 1
    expected_cuda_version: str = "13.2"

    max_num_iterations: int = 30_000
    steps_per_save: int = 2_000
    steps_per_eval_image: int = 100
    steps_per_eval_all_images: int = 1_000
    steps_per_log: int = 10
    save_only_latest_checkpoint: bool = True

    cache_images: Literal["cpu", "gpu", "disk"] = "gpu"
    cache_images_type: Literal["uint8", "float32"] = "uint8"
    camera_res_scale_factor: float = 1.0
    train_cameras_sampling_strategy: Literal["random", "fps"] = "random"

    warmup_length: int = 500
    refine_every: int = 100
    resolution_schedule: int = 3_000
    num_downscales: int = 2
    background_color: Literal["random", "black", "white"] = "random"
    # Nerfstudio's high-quality splatfacto-big preset values.
    cull_alpha_threshold: float = 0.005
    cull_scale_threshold: float = 0.5
    reset_alpha_every: int = 30
    densify_gradient_threshold: float = 0.0005
    use_absolute_gradient: bool = True
    densify_size_threshold: float = 0.01
    split_samples: int = 2
    spherical_harmonics_degree: int = 3
    spherical_harmonics_degree_interval: int = 1_000
    cull_screen_size: float = 0.15
    split_screen_size: float = 0.05
    stop_screen_size_at: int = 4_000
    stop_split_at: int = 15_000
    random_init: bool = False
    num_random_points: int = 50_000
    random_init_scale: float = 10.0
    ssim_lambda: float = 0.2
    use_scale_regularization: bool = False
    max_gaussian_scale_ratio: float = 10.0
    output_depth_during_training: bool = False
    rasterize_mode: Literal["classic", "antialiased"] = "classic"
    camera_optimizer_mode: Literal["off", "SO3xR3", "SE3"] = "off"
    use_bilateral_grid: bool = False
    bilateral_grid_shape: tuple[int, int, int] = (16, 16, 8)
    color_corrected_metrics: bool = False
    strategy: Literal["default", "mcmc"] = "default"
    mcmc_max_gaussians: int = 1_000_000
    mcmc_noise_learning_rate: float = 5e5
    mcmc_opacity_regularization: float = 0.01
    mcmc_scale_regularization: float = 0.01

    means_learning_rate: float = 1.6e-4
    means_final_learning_rate: float = 1.6e-6
    features_dc_learning_rate: float = 2.5e-3
    features_rest_learning_rate: float = 2.5e-3 / 20
    opacities_learning_rate: float = 5e-2
    scales_learning_rate: float = 5e-3
    quaternions_learning_rate: float = 1e-3
    camera_learning_rate: float = 1e-4
    camera_final_learning_rate: float = 5e-7
    bilateral_grid_learning_rate: float = 2e-3
    bilateral_grid_final_learning_rate: float = 1e-4

    viewer_during_training: bool = False
    viewer_host: str = "127.0.0.1"
    viewer_port: int = 7_007
    viewer_max_display_images: int = 512
    viewer_image_format: Literal["jpeg", "png"] = "jpeg"
    viewer_jpeg_quality: int = 85
    viewer_camera_frustum_scale: float = 0.1
    viewer_quit_on_train_completion: bool = True


@dataclass(frozen=True, slots=True)
class ExportSettings:
    """Settings for exporting a directory that can be copied to the viewer machine."""

    checkpoint_config: Path | None = None
    training_output_dir: Path = PROJECT_ROOT / "outputs"
    experiment_name: str = "spectacular-ai-3dgs"
    method_name: str = "splatfacto"
    output_dir: Path = PROJECT_ROOT / "viewer_output"
    manifest_filename: str = "manifest.json"
    splat_filename: str = "splat.ply"
    ply_color_mode: Literal["sh_coeffs", "rgb"] = "sh_coeffs"
    include_camera_poses: bool = True


@dataclass(frozen=True, slots=True)
class ViewerSettings:
    """Settings for loading a copied viewer-output directory locally."""

    input_dir: Path = PROJECT_ROOT / "viewer_output"
    manifest_filename: str = "manifest.json"
    host: str = "127.0.0.1"
    port: int = 7_007
    up_direction: Literal["+x", "+y", "+z", "-x", "-y", "-z"] = "+z"
    show_cameras: bool = True
    show_view_directions: bool = False
    max_cameras: int = 256
    camera_frustum_scale: float = 0.08
    camera_fov_degrees: float = 50.0
    camera_aspect: float = 16 / 9
    view_direction_length: float = 0.3


MAPPING = MappingSettings()
DATASET = DatasetSettings(dataset_path=MAPPING.dataset_path)
TRAINING = TrainingSettings()
EXPORT = ExportSettings(
    training_output_dir=TRAINING.output_dir,
    experiment_name=TRAINING.experiment_name,
)
VIEWER = ViewerSettings(
    input_dir=EXPORT.output_dir,
    manifest_filename=EXPORT.manifest_filename,
)
