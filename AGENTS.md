# Working in the sfmtool Repository

This repository uses [Pixi](https://pixi.sh) for environments and dependencies. Config in `pyproject.toml` and `pixi.toml`.

## Environment Setup

Pixi manages Python, Rust, and all dependencies. The key environments are:

| Environment | Purpose |
|-------------|---------|
| `default` | Runtime (Python + Rust toolchain) |
| `test` | Testing + coverage + linting + maturin |
| `dev` | Development (includes ipython) |
| `docs` | Documentation (zensical) |
| `cuda` | CUDA-enabled pycolmap |

## Running Tests

```bash
# Python tests
pixi run test

# Rust tests
pixi run test-rust

# Combined coverage (Rust + Python)
pixi run coverage-all

# Specific test file
pixi run test -- tests/test_filenames.py -v
```

### Task Completion Requirements

When completing a task, run the relevant format/lint checks for the languages that changed:

```bash
# If Python code changed
pixi run fmt && pixi run check

# If Rust code changed
pixi run cargo fmt && pixi run cargo clippy --workspace
```

Rebuilding Python bindings after changes to any Rust crate:

```bash
pixi run maturin develop --release
```

## Running CLI Commands

The `sfm` CLI should be run through pixi:

```bash
# Will be the workspace
cd workspace-dir
# Initialize the SfM workspace
pixi run sfm init .
# SIFT extraction for everything in the images/ dir
pixi run sfm sift --extract images
# Incremental/global SfM for all images in images/
pixi run sfm solve -i images
pixi run sfm solve -g images
```

The full set of CLI commands:

| Command | Purpose |
|---------|---------|
| `sfm init` | Initialize a workspace |
| `sfm sift` | SIFT feature extraction, printing, drawing |
| `sfm match` | Feature matching (exhaustive, sequential, flow) |
| `sfm solve` | Run SfM solver (incremental `-i` or global `-g`) |
| `sfm xform` | Apply transforms (filter, scale, rotate, bundle adjust, etc.) |
| `sfm inspect` | Inspect reconstruction statistics |
| `sfm align` | Align multiple reconstructions |
| `sfm merge` | Merge reconstructions |
| `sfm compare` | Compare reconstruction invariants |
| `sfm densify` | Point cloud densification |
| `sfm epipolar` | Epipolar geometry visualization |
| `sfm flow` | Optical flow visualization |
| `sfm heatmap` | Generate heatmaps (reprojection error, tracks, angles) |
| `sfm undistort` | Undistort images |
| `sfm pano2rig` | Convert equirectangular panoramas to perspective rig images |
| `sfm insv2rig` | Extract dual-fisheye frames from Insta360 .insv video files |
| `sfm from-colmap-bin` | Import from COLMAP binary format |
| `sfm to-colmap-bin` | Export .sfmr to COLMAP binary format |
| `sfm to-colmap-db` | Export to COLMAP database |

## Project Structure

### Overview

The project is a multi-language SfM toolkit with two main components:

1. **Python package** (`src/sfmtool/`) — CLI, SfM pipeline orchestration, COLMAP/GLOMAP integration
2. **Rust workspace** (`crates/`) — Core algorithms, file format I/O, GUI viewer, Python bindings

From a user perspective, it's divided into:

1. **SfM CLI command** - Run commands to create and evaluate SfM reconstructions
2. **Workspaces** - File formats and commands are organized around a workspace concept
3. **SfM Explorer** - 3D viewer to inspect SfM reconstructions interactively

### Rust Workspace (`crates/`)

A Cargo workspace with 7 crates:

| Crate | Purpose |
|-------|---------|
| `sift-format` | `.sift` file format read/write/verify |
| `matches-format` | `.matches` file format read/write/verify |
| `sfmr-format` | `.sfmr` file format (ZIP + zstd) read/write/verify |
| `sfmr-colmap` | COLMAP format interop — binary reconstruction I/O and SQLite database read/write |
| `sfmtool-core` | Core data structures and algorithms (camera, alignment, distortion, epipolar, feature matching, frustum, optical flow, transforms, spatial indexing) |
| `sfm-explorer` | Native GUI viewer application (egui + wgpu), window title "SfM Explorer" |
| `sfmtool-py` | PyO3 Python bindings exposing Rust to Python as `sfmtool._sfmtool` |

Key Rust commands:

```bash
pixi run cargo build               # Debug build
pixi run cargo build --release     # Release build
pixi run cargo test --workspace    # Run all Rust tests
pixi run cargo check --workspace   # Type-check without building
pixi run cargo clippy --workspace  # Lint
pixi run cargo fmt --check         # Check formatting
pixi run gui                       # Build and run the GUI (release mode)
```

### Python Package (`src/sfmtool/`)

~88 modules organized into subpackages:

- `cli.py` / `_cli_group.py` — CLI entry point (Click with `CategoryGroup` for categorized help)
- `_commands/` — CLI command implementations (19 commands)
- `feature_match/` — Feature matching algorithms (descriptor matching, polar sweep, rectified sweep, geometric filtering, flow matching)
- `xform/` — Reconstruction transforms (align, filter, scale, rotate, translate, bundle adjust, etc.)
- `visualization/` — Colormap and heatmap rendering
- `_sift_file.py` — SIFT feature file handling
- `_cameras.py` — Camera model and intrinsic parameter handling
- `_colmap_db.py` / `_colmap_io.py` / `_to_colmap_db.py` — COLMAP database and format I/O
- `_isfm.py` / `_gsfm.py` — Incremental and global SfM runners
- `_densify.py` — Point cloud densification pipeline
- `_merge.py` / `_merge_correspondences.py` / `_merge_pose_refinement.py` — Reconstruction merging
- `_align_by_cameras.py` / `_align_by_points.py` / `_multi_align.py` — Reconstruction alignment
- `_rectification.py` — Stereo rectification
- `_pano2rig.py` / `_insv2rig.py` — Panorama and Insta360 video conversion
- `_workspace.py` — Workspace management
- `_rig_config.py` / `_rig_frames.py` — Multi-camera rig support

### Python Bindings (`sfmtool-py` crate)

The `sfmtool-py` crate compiles to `sfmtool._sfmtool`, a native Python extension built with PyO3 + maturin. It exposes:

- **File I/O**: SFMR, SIFT, and matches file read/write/verify; COLMAP binary and DB import
- **Geometric types**: `Se3Transform`, `CameraIntrinsics`, `RigidTransform`, `RotQuaternion`
- **Reconstruction**: `SfmrReconstruction` with point/image access, filtering, masking, transforms
- **Feature matching**: Descriptor matching, geometric filtering, polar/rectified sweep algorithms
- **Image pair graph**: Graph operations for image connectivity
- **KD-tree**: Spatial indexing
- **Optical flow**: DIS optical flow
- **GUI**: Launch the sfm-explorer GUI from Python

Build the bindings with:

```bash
pixi run maturin develop --release
```

### GUI Application (`sfm-explorer` crate)

A native 3D viewer for SfM reconstructions using winit + wgpu + egui.

```bash
pixi run gui                                    # Run with no file
pixi run gui -- path/to/reconstruction.sfmr     # Open a file directly
```

### Tests

- **Python tests** (`tests/`) — pytest, ~906 tests across ~27 test modules + `xform/` subdir
  - `conftest.py` — fixtures (`isolated_seoul_bull_image`, `isolated_seoul_bull_17_images`)
  - Rust binding integration tests (`test_distortion_rust_bindings.py`, `test_descriptor_rust_bindings.py`, `test_kdtree_rust_bindings.py`)
- **Rust tests** — In-crate `#[cfg(test)]` modules, ~405 tests across 7 crates

### Specifications (`specs/`)

Design specifications organized by area:

- `specs/cli/` — 20 CLI command specs (one per command)
- `specs/formats/` — File format specs (sfmr, sift, matches)
- `specs/gui/` — 13 GUI design docs (architecture, viewport, rendering, UX)
- `specs/workspace/` — Workspace configuration spec

### Documentation (`docs/`)

- Built with Zensical, configured in `zensical.toml`
- Served via GitHub Pages
- Build/serve: `pixi run docs-build` / `pixi run docs-serve`

## Test Datasets

Three datasets are checked into the repository in `test-data/images/`:

| Dataset | Images | Resolution | Location |
|---------|--------|------------|----------|
| Seoul Bull | 17 | 270x480 | `test-data/images/seoul_bull_sculpture/` |
| Dino Dog Toy | 85 | 2040x1536 | `test-data/images/dino_dog_toy/` |
| Seattle Backyard | 26 | 360x640 | `test-data/images/seattle_backyard/` |

Initialize a workspace from a dataset with the scripts in `scripts/`:

```bash
scripts/init_dataset_seoul_bull.sh
scripts/init_dataset_dino_dog_toy.sh
scripts/init_dataset_seattle_backyard.sh
```

## CI/CD

GitHub Actions workflows in `.github/workflows/`:

- `ci.yml` — Linux + Windows matrix, runs `pixi run -e test coverage-all`, uploads to codecov.io
- `docs.yml` — Builds and deploys docs to GitHub Pages
- `publish_to_pypi.yml` — PyPI publication
