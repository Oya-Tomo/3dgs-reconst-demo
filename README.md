# Nerfstudio 3DGS Reconstruction Demo

This project trains and views Nerfstudio's 3D Gaussian Splatting implementation, Splatfacto, with every project
parameter managed in Python code.

Training is started with a script instead of the `ns-train` CLI. [src/train.py](src/train.py) builds a complete
`TrainerConfig` and passes it to `nerfstudio.scripts.train.main`, the same training entry point used by `ns-train` after
CLI parsing. The default model uses degree-3 spherical harmonics, so it learns and renders view-dependent color.

The core training and viewing flow accepts any valid Nerfstudio-format dataset. The repository currently includes a
Spectacular AI Mapping Tools input adapter in [src/prepare_data.py](src/prepare_data.py); additional capture and
conversion tools can be added without changing the trainer, exporter, or Viewer.

The workflow is:

1. Prepare a Nerfstudio-format dataset on the training machine.
2. Train Splatfacto by running `src/train.py`.
3. Package the checkpoint and camera metadata into `viewer_output/`.
4. Copy only `viewer_output/` to the local machine with `scp`.
5. Run `src/viewer.py` locally to launch Nerfstudio's official Viewer.

## Project Layout

```text
.
├── data/
│   ├── recording/        # Optional input recording for an adapter
│   └── nerfstudio/       # Training dataset
├── outputs/              # Nerfstudio training runs and checkpoints
├── viewer_output/        # Portable bundle copied with scp
├── src/
│   ├── settings.py       # All project paths and parameters
│   ├── prepare_data.py   # Current Spectacular AI input adapter
│   ├── dataset.py        # Nerfstudio dataset validation
│   ├── train.py          # Programmatic Splatfacto training
│   ├── export.py         # Portable official-Viewer bundle creation
│   └── viewer.py         # Official Nerfstudio Viewer launcher
├── tests/
├── pyproject.toml
└── uv.lock
```

Large recordings, datasets, checkpoints, and Viewer bundles are excluded from Git.

## Requirements

### Training Machine

- Ubuntu x86-64
- An NVIDIA CUDA-capable GPU
- An NVIDIA driver and CUDA Toolkit 13.2
- FFmpeg when using the included Spectacular AI adapter
- uv

The project is not restricted to a particular GPU model. Image resolution, caching, and training settings determine
the required VRAM.

### Local Viewer Machine

- Linux x86-64
- An NVIDIA CUDA-capable GPU and compatible driver
- CUDA Toolkit 13.2
- A local web browser
- uv

Nerfstudio's official Viewer renders through the restored Splatfacto model on the local machine. It is not a standalone
browser-only PLY renderer.

## 1. Set Up the Project

Clone the repository and install the required system libraries:

```bash
git clone https://github.com/Oya-Tomo/3dgs-reconst-demo.git
cd 3dgs-reconst-demo
sudo apt update
sudo apt install --yes build-essential ffmpeg git libgl1 libglib2.0-0
```

Synchronize the Python environment from the lockfile:

```bash
uv sync --all-groups
```

The lockfile resolves recent compatible versions of Nerfstudio, CUDA 13.2 PyTorch, torchvision, gsplat, Spectacular AI,
and the development tools.

## 2. Configure the Project in Code

Edit [src/settings.py](src/settings.py). The project scripts intentionally have no parameter CLI; paths and model
settings stay reviewable and versioned in code.

| Setting | Purpose | Default |
| --- | --- | --- |
| `SPECTACULAR_AI.recording_path` | Input for the included adapter | `data/recording` |
| `SPECTACULAR_AI.key_frame_distance` | Adapter keyframe spacing | `0.05` m |
| `DATASET.dataset_path` | Nerfstudio-format training data | `data/nerfstudio` |
| `DATASET.downscale_factor` | Training-image downscale factor | Automatic |
| `TRAINING.output_dir` | Training run root | `outputs` |
| `TRAINING.max_num_iterations` | Training iterations | `30000` |
| `TRAINING.cache_images` | Training image cache location | `gpu` |
| `TRAINING.spherical_harmonics_degree` | View-dependent color degree | `3` |
| `EXPORT.checkpoint_config` | Run selected for packaging | Latest run |
| `EXPORT.dataset_path` | Camera metadata packaged for that run | `DATASET.dataset_path` |
| `EXPORT.output_dir` | Directory copied with `scp` | `viewer_output` |
| `VIEWER.input_dir` | Copied bundle loaded locally | `viewer_output` |
| `VIEWER.max_display_cameras` | Maximum displayed camera frustums | `256` |

`VIEWER.input_dir` defaults to `EXPORT.output_dir`. Change it when the downloaded bundle is placed elsewhere.

## 3. Prepare Training Data

The trainer consumes a standard Nerfstudio dataset containing `transforms.json`, referenced images, camera transforms,
intrinsics, and optionally an initial point cloud.

### Included Spectacular AI Adapter

Place a complete Spectacular AI Mapping Tools recording under `data/recording/`, then run:

```bash
uv run src/prepare_data.py
```

The script builds the `sai-cli process` operation from `SpectacularAISettings`, writes the result to
`DATASET.dataset_path`, and validates the generated dataset. It refuses to overwrite a non-empty destination.

### Other Data Sources

For any other supported tool, convert its output to Nerfstudio format, set `DATASET.dataset_path`, and continue with
training. `src/train.py` and later stages do not depend on Spectacular AI.

## 4. Train Splatfacto

Run this on the training machine:

```bash
uv run src/train.py
```

`src/train.py` constructs the data parser, full-image data manager, Splatfacto model, optimizers, schedulers, Viewer
configuration, and `TrainerConfig` in Python. It then calls Nerfstudio's training entry point directly. This provides the
same training pipeline as `ns-train splatfacto`; only CLI-based configuration is replaced by code-based configuration.

By default, degree-3 spherical harmonics are enabled and activated progressively every 1,000 steps. The training-time
Viewer is disabled by default.

Each run is written under:

```text
outputs/3dgs-reconstruction/splatfacto/<timestamp>/
├── config.yml
└── nerfstudio_models/
    └── step-XXXXXXXXX.ckpt
```

## 5. Build the Portable Viewer Bundle

Run this on the training machine after training:

```bash
uv run src/export.py
```

When `EXPORT.checkpoint_config` is `None`, the newest run is selected. Set it to a particular run's `config.yml` to
package that run instead. The exporter refuses to overwrite a non-empty output directory.

The default bundle is:

```text
viewer_output/
├── manifest.json
├── config.yml
├── nerfstudio_models/
│   └── step-XXXXXXXXX.ckpt
└── dataset/
    ├── transforms.json
    └── images/
        └── *                  # Copied source images
```

The checkpoint retains all Gaussian parameters, including higher-order spherical-harmonic coefficients. The portable
`transforms.json` retains camera transforms and intrinsics, and every referenced source image is copied into the bundle.
Masks, depth maps, and the seed point cloud are omitted because the local Viewer does not need them.

`manifest.json` alone is not enough to reconstruct camera poses. It identifies and validates the bundle components;
camera transforms and viewing orientations come from `dataset/transforms.json`.

## 6. Copy Only the Bundle

From the repository root on the local machine, replace the host and remote path with the actual values:

```bash
scp -r USER@TRAIN_HOST:/absolute/path/to/3dgs-reconst-demo/viewer_output ./
```

The local repository and `viewer_output/` are sufficient. The original `data/` and `outputs/` directories do not need
to be copied.

## 7. Render Locally with Nerfstudio Viewer

Synchronize the local environment, then launch the Viewer:

```bash
uv sync --all-groups
uv run src/viewer.py
```

`src/viewer.py` validates the copied directory, generates a relocatable runtime config inside it, and calls the same
official `RunViewer` implementation that powers `ns-viewer`. The URL is printed by Nerfstudio, normally using port 7007.

The official Viewer restores the full Splatfacto checkpoint, so degree-3 spherical harmonics produce view-dependent
color. Training camera frustums are available in the Viewer's scene tree. Each frustum's orientation represents that
camera's viewing direction, and clicking one moves the Viewer camera to the captured pose. Camera thumbnails use the
copied source images.

## Reducing VRAM Usage

If training exceeds available VRAM, first cache images in CPU memory. If necessary, also downscale the dataset images in
[src/settings.py](src/settings.py):

```python
class DatasetSettings:
    downscale_factor: int | None = 2


class TrainingSettings:
    cache_images: Literal["cpu", "gpu", "disk"] = "cpu"
```

A larger downscale factor reduces VRAM use and training time at the cost of fine detail.

## Updating Dependencies

The dependency ranges in `pyproject.toml` allow compatible updates, while `uv.lock` records the tested resolution.
Nerfstudio follows its official `main` branch and PyTorch uses the official CUDA 13.2 package index.
Pillow remains on the newest compatible 11.x release because the current Nerfstudio image loader is incompatible with
Pillow 12.x.

Update and re-resolve the environment with:

```bash
uv lock --upgrade
uv sync --all-groups
```

Nerfstudio, PyTorch, torchvision, and gsplat evolve together. After an upgrade, validate training and local checkpoint
viewing with representative data.

## Development Checks

```bash
uv lock --check
uv run ruff check src tests
uv run ruff format --check src tests
uv run pyright
uv run pytest
```

## Troubleshooting

- `The conversion destination is not empty`: move the existing data or change `DATASET.dataset_path`.
- `No trained config.yml exists`: finish a training run or set `EXPORT.checkpoint_config` explicitly.
- `The Viewer output directory is not empty`: move the previous bundle or change `EXPORT.output_dir`.
- `The Viewer input directory does not exist`: make the `scp` destination match `VIEWER.input_dir`.
- Missing point-cloud warning: training can continue with random initialization, but a suitable point cloud usually gives
  a better starting state.
- Viewer launch or rendering failure: confirm that the local NVIDIA driver supports the locked CUDA runtime and that
  port 7007 is available.

## Capture Guidance and Licensing

Capture every part of the scene from multiple overlapping viewpoints. Avoid sudden motion, motion blur, and large
exposure changes.

The included Spectacular AI adapter depends on Spectacular AI SDK and Mapping Tools. Review its usage-specific license
terms before commercial use.
