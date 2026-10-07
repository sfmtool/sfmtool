# `sfm flow` Command

## Overview

Computes dense optical flow (DIS algorithm) between two images and visualizes SIFT keypoint
advection. Optionally compares flow correspondences against an existing reconstruction's
feature matches.

## Command Syntax

```bash
sfm flow <IMAGE1> <IMAGE2> [OPTIONS...]
```

## Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--draw / -d` | path | | Save visualization (omit for statistics only). Writes derived files next to the path: `<stem>_flow<ext>` for the flow field, plus `<stem>_A<ext>` and `<stem>_B<ext>` for the annotated images, or the path itself with `--side-by-side` |
| `--preset` | `fast` \| `default` \| `high_quality` | `default` | Flow quality preset |
| `--reconstruction / -r` | path | | `.sfmr` file to compare against |
| `--max-features` | int | all | Maximum features to visualize |
| `--tolerance` | float | 3.0 | Pixel tolerance for advection matching |
| `--descriptor-threshold` | float | 100.0 | L2 descriptor distance threshold |
| `--feature-size` | int | 4 | Feature marker size (pixels) |
| `--line-thickness` | int | 1 | Line thickness (pixels) |
| `--side-by-side / --separate` | bool | `false` | Single combined image or two separate files |

## Visualization Modes

### Flow only (no `--reconstruction`)

Shows flow-colored arrows and keypoint connections between the two images. Color encodes flow
direction using the Middlebury color wheel. Includes a flow legend.

### Comparison mode (`--reconstruction`)

Compares flow-based correspondences against reconstruction matches:

- **Green** — Agreement: both flow and reconstruction match the same features
- **Red** — Reconstruction only: feature match exists in `.sfmr` but flow disagrees
- **Yellow** — Flow only: flow suggests a correspondence not in the reconstruction

Each image is found in the reconstruction by its path relative to the
reconstruction's workspace, so on a rig `fisheye_left/frame_01.jpg` and
`fisheye_right/frame_01.jpg` are told apart. An image inside the workspace that
the reconstruction does not contain is an error. An image outside the workspace
is matched by file name, and the command fails when no reconstruction image or
more than one has that name.

## Usage Examples

```bash
# Compute flow statistics between two images
sfm flow image_001.jpg image_002.jpg

# Visualize flow
sfm flow image_001.jpg image_002.jpg --draw flow.png --preset high_quality

# Compare flow against reconstruction
sfm flow image_001.jpg image_002.jpg --draw compare.png -r solve.sfmr
```

> Adjacent-pairs batch mode is not supported by `sfm flow`. Use `sfm epipolar
> --pairs-dir` for batched per-image visualizations, or wrap `sfm flow` in a
> shell loop for batch flow.
