"""Train Nerfstudio Splatfacto entirely through its Python API."""

from __future__ import annotations

import shutil
import subprocess

import torch
from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.configs.base_config import LocalWriterConfig, LoggingConfig, MachineConfig, ViewerConfig
from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanagerConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.models.splatfacto import SplatfactoModelConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.scripts.train import main as run_nerfstudio_training

from dataset import validate_dataset
from settings import DATASET, TRAINING, TrainingSettings


def build_trainer_config(settings: TrainingSettings = TRAINING) -> TrainerConfig:
    """Construct the complete Splatfacto configuration without CLI parsing."""

    dataparser = NerfstudioDataParserConfig(
        data=DATASET.dataset_path,
        scale_factor=DATASET.scale_factor,
        downscale_factor=DATASET.downscale_factor,
        scene_scale=DATASET.scene_scale,
        orientation_method=DATASET.orientation_method,
        center_method=DATASET.center_method,
        auto_scale_poses=DATASET.auto_scale_poses,
        eval_mode=DATASET.eval_mode,
        train_split_fraction=DATASET.train_split_fraction,
        eval_interval=DATASET.eval_interval,
        depth_unit_scale_factor=DATASET.depth_unit_scale_factor,
        load_3D_points=DATASET.load_3d_points,
    )
    datamanager = FullImageDatamanagerConfig(
        data=DATASET.dataset_path,
        dataparser=dataparser,
        camera_res_scale_factor=settings.camera_res_scale_factor,
        cache_images=settings.cache_images,
        cache_images_type=settings.cache_images_type,
        train_cameras_sampling_strategy=settings.train_cameras_sampling_strategy,
    )
    model = SplatfactoModelConfig(
        warmup_length=settings.warmup_length,
        refine_every=settings.refine_every,
        resolution_schedule=settings.resolution_schedule,
        background_color=settings.background_color,
        num_downscales=settings.num_downscales,
        cull_alpha_thresh=settings.cull_alpha_threshold,
        cull_scale_thresh=settings.cull_scale_threshold,
        reset_alpha_every=settings.reset_alpha_every,
        densify_grad_thresh=settings.densify_gradient_threshold,
        use_absgrad=settings.use_absolute_gradient,
        densify_size_thresh=settings.densify_size_threshold,
        n_split_samples=settings.split_samples,
        sh_degree_interval=settings.spherical_harmonics_degree_interval,
        cull_screen_size=settings.cull_screen_size,
        split_screen_size=settings.split_screen_size,
        stop_screen_size_at=settings.stop_screen_size_at,
        random_init=settings.random_init,
        num_random=settings.num_random_points,
        random_scale=settings.random_init_scale,
        ssim_lambda=settings.ssim_lambda,
        stop_split_at=settings.stop_split_at,
        sh_degree=settings.spherical_harmonics_degree,
        use_scale_regularization=settings.use_scale_regularization,
        max_gauss_ratio=settings.max_gaussian_scale_ratio,
        output_depth_during_training=settings.output_depth_during_training,
        rasterize_mode=settings.rasterize_mode,
        camera_optimizer=CameraOptimizerConfig(mode=settings.camera_optimizer_mode),
        use_bilateral_grid=settings.use_bilateral_grid,
        grid_shape=settings.bilateral_grid_shape,
        color_corrected_metrics=settings.color_corrected_metrics,
        strategy=settings.strategy,
        max_gs_num=settings.mcmc_max_gaussians,
        noise_lr=settings.mcmc_noise_learning_rate,
        mcmc_opacity_reg=settings.mcmc_opacity_regularization,
        mcmc_scale_reg=settings.mcmc_scale_regularization,
    )

    return TrainerConfig(
        output_dir=settings.output_dir,
        method_name="splatfacto",
        experiment_name=settings.experiment_name,
        project_name=settings.project_name,
        machine=MachineConfig(
            seed=settings.seed,
            num_devices=settings.num_devices,
            num_machines=1,
            machine_rank=0,
            dist_url="auto",
            device_type="cuda",
        ),
        logging=LoggingConfig(
            steps_per_log=settings.steps_per_log,
            local_writer=LocalWriterConfig(enable=True, max_log_size=10),
            profiler="basic",
        ),
        viewer=ViewerConfig(
            websocket_port=settings.viewer_port,
            websocket_host=settings.viewer_host,
            num_rays_per_chunk=1 << 15,
            max_num_display_images=settings.viewer_max_display_images,
            quit_on_train_completion=settings.viewer_quit_on_train_completion,
            image_format=settings.viewer_image_format,
            jpeg_quality=settings.viewer_jpeg_quality,
            make_share_url=False,
            camera_frustum_scale=settings.viewer_camera_frustum_scale,
            default_composite_depth=True,
        ),
        pipeline=VanillaPipelineConfig(datamanager=datamanager, model=model),
        optimizers={
            "means": {
                "optimizer": AdamOptimizerConfig(lr=settings.means_learning_rate, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=settings.means_final_learning_rate,
                    max_steps=settings.max_num_iterations,
                ),
            },
            "features_dc": {
                "optimizer": AdamOptimizerConfig(lr=settings.features_dc_learning_rate, eps=1e-15),
                "scheduler": None,
            },
            "features_rest": {
                "optimizer": AdamOptimizerConfig(lr=settings.features_rest_learning_rate, eps=1e-15),
                "scheduler": None,
            },
            "opacities": {
                "optimizer": AdamOptimizerConfig(lr=settings.opacities_learning_rate, eps=1e-15),
                "scheduler": None,
            },
            "scales": {
                "optimizer": AdamOptimizerConfig(lr=settings.scales_learning_rate, eps=1e-15),
                "scheduler": None,
            },
            "quats": {
                "optimizer": AdamOptimizerConfig(lr=settings.quaternions_learning_rate, eps=1e-15),
                "scheduler": None,
            },
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=settings.camera_learning_rate, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=settings.camera_final_learning_rate,
                    max_steps=settings.max_num_iterations,
                    warmup_steps=1_000,
                    lr_pre_warmup=0,
                ),
            },
            "bilateral_grid": {
                "optimizer": AdamOptimizerConfig(lr=settings.bilateral_grid_learning_rate, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=settings.bilateral_grid_final_learning_rate,
                    max_steps=settings.max_num_iterations,
                    warmup_steps=1_000,
                    lr_pre_warmup=0,
                ),
            },
        },
        vis="viewer" if settings.viewer_during_training else "tensorboard",
        steps_per_eval_image=settings.steps_per_eval_image,
        steps_per_eval_batch=0,
        steps_per_save=settings.steps_per_save,
        steps_per_eval_all_images=settings.steps_per_eval_all_images,
        max_num_iterations=settings.max_num_iterations,
        mixed_precision=False,
        save_only_latest_checkpoint=settings.save_only_latest_checkpoint,
    )


def validate_cuda_environment(settings: TrainingSettings = TRAINING) -> None:
    """Fail early when PyTorch and the CUDA extension toolchain do not match."""

    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot access a CUDA GPU. Check the CUDA 13.2 PyTorch build and NVIDIA driver.")
    if torch.version.cuda != settings.expected_cuda_version:
        raise RuntimeError(
            f"PyTorch uses CUDA runtime {torch.version.cuda}; expected {settings.expected_cuda_version}."
        )
    nvcc_path = shutil.which("nvcc")
    if nvcc_path is None:
        raise RuntimeError("nvcc is not on PATH. The initial gsplat build requires the CUDA 13.2 compiler.")
    nvcc_version = subprocess.run(
        [nvcc_path, "--version"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if f"release {settings.expected_cuda_version}" not in nvcc_version:
        raise RuntimeError(
            f"nvcc does not report CUDA {settings.expected_cuda_version}. Output:\n{nvcc_version.strip()}"
        )


def main() -> None:
    report = validate_dataset(DATASET.dataset_path)
    print(f"Dataset: {report.frame_count} frames ({report.dataset_path})")
    if report.point_cloud_path is not None:
        print(f"Initial point cloud: {report.point_cloud_path}")
    for warning in report.warnings:
        print(f"Warning: {warning}")

    validate_cuda_environment()
    TRAINING.output_dir.mkdir(parents=True, exist_ok=True)
    run_nerfstudio_training(build_trainer_config())


if __name__ == "__main__":
    main()
