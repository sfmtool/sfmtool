# `sfm epipolar` Command

## Overview

Visualizes epipolar geometry between image pairs in a reconstruction. Draws epipolar lines
through shared feature observations, optionally with stereo rectification or undistortion.

## Command Syntax

```bash
# Single pair mode
sfm epipolar <RECONSTRUCTION.sfmr> <IMAGE1> <IMAGE2> --draw <OUTPUT> [OPTIONS...]

# Adjacent pairs batch mode
sfm epipolar <RECONSTRUCTION.sfmr> --pairs-dir <DIR> [OPTIONS...]
```

Image arguments accept filenames (e.g., `image_003.jpg`) or file numbers (e.g., `3`).

## Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--draw / -d` | path | | Save visualization to this path |
| `--max-features` | int | all | Maximum shared features to visualize |
| `--line-thickness` | int | 1 | Thickness of epipolar lines (pixels) |
| `--feature-size` | int | 3 | Size of feature markers (pixels) |
| `--rectify / --no-rectify` | bool | `false` | Apply stereo rectification |
| `--undistort` | flag | | Remove lens distortion (mutually exclusive with `--rectify`) |
| `--draw-lines / --no-lines` | bool | `true` | Draw epipolar lines vs. horizontal scanlines |
| `--side-by-side / --separate` | bool | `false` | Single combined image or two separate files (`_A`, `_B`) |
| `--sweep-with-max-features` | int | | Run sort-and-sweep matching with this many features |
| `--sweep-window-size` | int | 30 | Window size for sweep matching |
| `--pairs-dir` | path | | Process all adjacent pairs, saving to directory |

## Modes

### Single pair

Requires `IMAGE1`, `IMAGE2`, and `--draw`. Produces one visualization showing epipolar
geometry between the two images.

### Adjacent pairs

Uses `--pairs-dir` to batch-process all adjacent image pairs in the reconstruction. Saves
one visualization per pair to the output directory.

## Usage Examples

```bash
# Visualize epipolar geometry between two images
sfm epipolar solve.sfmr 3 7 --draw epipolar_3_7.png

# Rectified side-by-side view
sfm epipolar solve.sfmr image_003.jpg image_007.jpg --draw rect.png --rectify --side-by-side

# Batch all adjacent pairs
sfm epipolar solve.sfmr --pairs-dir epipolar_viz/

# With sweep matching overlay
sfm epipolar solve.sfmr 3 7 --draw sweep.png --sweep-with-max-features 2000
```
