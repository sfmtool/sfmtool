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
| `--draw / -d` | path | | Save visualization (omit for statistics only) |
| `--preset` | `fast` \| `default` \| `high_quality` | `default` | Flow quality preset |
| `--reconstruction / -r` | path | | `.sfmr` file to compare against |
| `--max-features` | int | all | Maximum features to visualize |
| `--tolerance` | float | 3.0 | Pixel tolerance for advection matching |
| `--descriptor-threshold` | float | 100.0 | L2 descriptor distance threshold |
| `--feature-size` | int | 4 | Feature marker size (pixels) |
| `--line-thickness` | int | 1 | Line thickness (pixels) |
| `--side-by-side / --separate` | bool | `false` | Single combined image or two separate files |
| `--pairs-dir` | path | | Process all adjacent pairs from reconstruction |

## Visualization Modes

### Flow only (no `--reconstruction`)

Shows flow-colored arrows and keypoint connections between the two images. Color encodes flow
direction using the Middlebury color wheel. Includes a flow legend.

### Comparison mode (`--reconstruction`)

Compares flow-based correspondences against reconstruction matches:

- **Green** — Agreement: both flow and reconstruction match the same features
- **Red** — Reconstruction only: feature match exists in `.sfmr` but flow disagrees
- **Yellow** — Flow only: flow suggests a correspondence not in the reconstruction

## Usage Examples

```bash
# Compute flow statistics between two images
sfm flow image_001.jpg image_002.jpg

# Visualize flow
sfm flow image_001.jpg image_002.jpg --draw flow.png --preset high_quality

# Compare flow against reconstruction
sfm flow image_001.jpg image_002.jpg --draw compare.png -r solve.sfmr

# Batch all adjacent pairs
sfm flow image_001.jpg image_002.jpg --pairs-dir flow_viz/ -r solve.sfmr
```
