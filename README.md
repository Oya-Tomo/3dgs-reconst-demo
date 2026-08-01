# Spectacular AI + Nerfstudio 3DGS Demo

This project converts a recording captured with Spectacular AI Mapping Tools and trains Nerfstudio's
3D Gaussian Splatting implementation, `Splatfacto`.

It does not use Nerfstudio's `ns-train`, `ns-viewer`, or `ns-export` commands. Dataset paths, model parameters,
optimizers, training iterations, export paths, and Viewer settings are all managed in
[src/settings.py](src/settings.py). Project scripts are run directly in the form `uv run src/<script>.py`.

The intended workflow is:

1. Convert a Spectacular AI recording on the training machine.
2. Train Splatfacto on the training machine.
3. Export a portable Viewer bundle to a configured directory.
4. Copy only that directory to the local machine with `scp`.
5. Run `viewer.py` locally to display the Gaussians, capture cameras, and view directions.

The source images and Nerfstudio dataset do not need to be copied to the local machine.

## Project Layout

```text
.
├── data/
│   ├── recording/        # Original Spectacular AI recording (not tracked by Git)
│   └── nerfstudio/       # Converted dataset (not tracked by Git)
├── outputs/              # Nerfstudio checkpoints (not tracked by Git)
├── viewer_output/        # Portable directory copied with scp (not tracked by Git)
├── src/
│   ├── settings.py       # All paths and parameters
│   ├── prepare_data.py   # Convert a Spectacular AI recording
│   ├── dataset.py        # Validate transforms.json and referenced files
│   ├── train.py          # Train Splatfacto through the Python API
│   ├── export.py         # Create the portable Viewer bundle
│   └── viewer.py         # Load and display the copied bundle locally
├── tests/
├── pyproject.toml
└── uv.lock
```

## Requirements

### Training Machine

- Ubuntu x86-64
- An NVIDIA GPU that supports CUDA 13.2
- NVIDIA driver and CUDA Toolkit 13.2
- FFmpeg
- [uv](https://docs.astral.sh/uv/)

The project is not restricted to a particular GPU model. The default settings favor quality, so adjust image caching
and downscaling to match the available VRAM and dataset size.

### Local Viewer Machine

- Linux x86-64
- A browser with WebGL support
- uv

The local Viewer reads a PLY file and uses Viser's browser-side Gaussian renderer. The Viewer itself does not use CUDA.

## 1. Setup

### 1.1 Clone the Repository

```bash
git clone https://github.com/Oya-Tomo/3dgs-reconst-demo.git
cd 3dgs-reconst-demo
```

### 1.2 Verify CUDA 13.2 on the Training Machine

```bash
nvidia-smi
nvcc --version
```

`nvcc --version` must report `release 13.2`. If CUDA Toolkit was installed in a non-default location, add its `bin`
directory to `PATH`.

```bash
export PATH=/usr/local/cuda-13.2/bin:$PATH
```

Before training, `src/train.py` validates CUDA GPU availability, the PyTorch CUDA runtime, and `nvcc`. Training stops
if either the PyTorch runtime or toolkit does not match CUDA 13.2.

### 1.3 Install System Packages and uv

Install the training-machine system dependencies:

```bash
sudo apt update
sudo apt install --yes build-essential ffmpeg git libgl1 libglib2.0-0
```

If uv is not installed, follow the [official uv installation guide](https://docs.astral.sh/uv/getting-started/installation/).

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
exec "$SHELL" -l
```

### 1.4 Synchronize the Python Environment

```bash
uv sync --all-groups
```

uv creates Python 3.12 and `.venv` according to `.python-version`, then installs the versions of Nerfstudio,
Spectacular AI, CUDA 13.2 PyTorch, gsplat, and other packages recorded in `uv.lock`.

On the training machine, verify that the first command reports `'cuda': '13.2'` and `'available': True`.

```bash
uv run python -c "import torch; print({'torch': torch.__version__, 'cuda': torch.version.cuda, 'available': torch.cuda.is_available()})"
uv run sai-cli --help
```

## 2. Configure Parameters in Code

Edit [src/settings.py](src/settings.py). The scripts intentionally do not override these values with CLI arguments.

| Setting | Purpose | Default |
| --- | --- | --- |
| `MAPPING.recording_path` | Spectacular AI recording | `data/recording` |
| `MAPPING.dataset_path` | Converted Nerfstudio dataset | `data/nerfstudio` |
| `MAPPING.key_frame_distance` | Keyframe spacing | `0.05` m |
| `DATASET.downscale_factor` | Training-image downscale factor | Automatic |
| `TRAINING.output_dir` | Checkpoint directory | `outputs` |
| `TRAINING.max_num_iterations` | Training iterations | `30000` |
| `TRAINING.cache_images` | Image cache location | `gpu` |
| `EXPORT.checkpoint_config` | Run selected for export | Latest run automatically |
| `EXPORT.output_dir` | Directory copied with `scp` | `viewer_output` |
| `VIEWER.input_dir` | Directory loaded locally | `viewer_output` |
| `VIEWER.show_cameras` | Initial camera visibility | `True` |
| `VIEWER.show_view_directions` | Initial view-direction visibility | `False` |

By default, `VIEWER.input_dir` is connected to `EXPORT.output_dir`. Change `VIEWER.input_dir` only when the bundle is
downloaded to a different local path.

For a small tabletop scene, `key_frame_distance = 0.05` is a useful starting point. The
[Spectacular AI guide](https://spectacularai.github.io/docs/sdk/tools/nerf.html) suggests approximately `0.15` for a
room-sized capture.

## 3. Convert the Spectacular AI Recording

Place the complete Spectacular AI recording under `data/recording/`, then run:

```bash
uv run src/prepare_data.py
```

The script constructs an operation in the following form from the code settings:

```text
sai-cli process data/recording --key_frame_distance=0.05 data/nerfstudio
```

After conversion, it validates `transforms.json`, referenced images, 4x4 camera matrices, and the optional initial point
cloud. To protect existing data, it stops instead of overwriting a non-empty destination.

If a Nerfstudio-format dataset already exists, skip conversion and set `DATASET.dataset_path` to that directory.

## 4. Train Splatfacto

```bash
uv run src/train.py
```

`src/train.py` constructs `TrainerConfig`, the data parser, Splatfacto model, optimizers, and schedulers in code, then
passes the configuration to Nerfstudio's Python API. The training-time Viewer is disabled by default.

Each run is stored under:

```text
outputs/spectacular-ai-3dgs/splatfacto/<timestamp>/
├── config.yml
└── nerfstudio_models/
```

`config.yml` records the complete Nerfstudio configuration for that run.

## 5. Export the Viewer Bundle

Run this on the training machine:

```bash
uv run src/export.py
```

When `EXPORT.checkpoint_config` is `None`, the newest run is selected. Set it to a specific `config.yml` to export a
particular run. `EXPORT.output_dir` must not exist or must be empty; the exporter refuses to overwrite an existing bundle.

The default output is:

```text
viewer_output/
├── manifest.json          # Bundle version, PLY filename, and camera poses
└── splat.ply              # Trained Gaussian parameters
```

`manifest.json` does not contain image paths. Each camera entry stores its train/eval split and a 3x4 camera-to-world
transform. The Viewer reconstructs camera position and orientation from that matrix and derives the view direction from
the camera's forward axis. Set `EXPORT.include_camera_poses = False` to write an empty camera list.

## 6. Copy Only the Bundle with scp

Run this from the repository root on the local machine. Replace the remote path with its actual absolute path.

```bash
scp -r USER@TRAIN_HOST:/absolute/path/to/3dgs-reconst-demo/viewer_output ./
```

After copying, only `viewer_output/` is required locally. The `data/` and `outputs/` directories and source images are not
needed. If the bundle is copied elsewhere, update `VIEWER.input_dir`.

## 7. Display the Gaussians Locally

Install the local native runtime dependencies and synchronize the Python environment once:

```bash
sudo apt update
sudo apt install --yes libgl1 libglib2.0-0
```

```bash
uv sync --all-groups
uv run src/viewer.py
```

Open the following URL in a browser:

```text
http://127.0.0.1:7007
```

The Viewer provides checkboxes for:

- `Capture cameras`: display train and evaluation camera frustums.
- `View directions`: display each camera's forward direction.

Camera FOV, aspect ratio, frustum scale, and display count are managed by `ViewerSettings`. The manifest stores camera
poses, not intrinsics, so FOV and aspect ratio are visualization values. Press `Ctrl+C` to stop the Viewer.

The PLY retains spherical-harmonic coefficients. The current local Viewer passes each Gaussian's DC color to Viser's
client-side renderer, so it does not apply higher-order view-dependent color.

## Reducing VRAM Usage

If training exceeds the available VRAM, first move the image cache to CPU in
[src/settings.py](src/settings.py). If that is insufficient, downscale the images.

```python
class DatasetSettings:
    downscale_factor: int | None = 2


class TrainingSettings:
    cache_images: Literal["cpu", "gpu", "disk"] = "cpu"
```

A larger `downscale_factor` lowers VRAM usage and training time at the cost of fine detail.

## Updating Dependencies

`pyproject.toml` defines compatibility ranges. Nerfstudio tracks the official repository's `main` branch, while PyTorch
and torchvision use the official CUDA 13.2 index. `uv.lock` records the tested package versions and Git commit.

Update all packages within the allowed compatibility ranges with:

```bash
uv lock --upgrade
uv sync --all-groups
```

Nerfstudio, PyTorch, torchvision, and gsplat are tightly coupled. After an update, validate training and export with a real
dataset before committing the new `uv.lock`.

## Development Checks

```bash
uv lock --check
uv run ruff check src tests
uv run ruff format --check src tests
uv run pyright
uv run pytest
```

## Troubleshooting

- `PyTorch cannot access a CUDA GPU`: check `nvidia-smi` and
  `uv run python -c "import torch; print(torch.cuda.is_available())"`.
- `PyTorch uses CUDA runtime ...`: verify the CUDA 13.2 index in `pyproject.toml` and the resolved packages in `uv.lock`,
  then run `uv sync` again.
- `nvcc is not on PATH`: add the CUDA Toolkit 13.2 `bin` directory to `PATH`.
- `The conversion destination is not empty`: move the existing data or change `MAPPING.dataset_path`.
- `The viewer output directory is not empty`: move the existing bundle or change `EXPORT.output_dir`.
- `The viewer input directory does not exist`: make the `scp` destination match `VIEWER.input_dir`.
- Missing point-cloud warning: training can continue, but 3DGS generally initializes more reliably with a point cloud.
- Viewer connection failure: check for a port conflict on 7007 and confirm that the browser supports WebGL.

## Capture Guidance and Licensing

Capture every part of the scene from multiple overlapping viewpoints. Avoid sudden motion, motion blur, and large exposure
changes. Recordings, converted datasets, checkpoints, and Viewer bundles are excluded from Git because they can be large.

Spectacular AI SDK and Mapping Tools have usage-specific license terms. Review the
[official licensing information](https://www.spectacularai.com/mapping) before commercial use.
